"""Render original video frames with synchronized player, pose, and shuttle overlays."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import cv2
import numpy as np

from src.court_projection import CourtHomography, draw_court_overlay
from src.pose_estimator import POSE_CONNECTIONS, POSE_NUM_LANDMARKS

from .data import SourceBundle


ROLE_COLORS = {
    "P1": (255, 190, 0),
    "P2": (0, 190, 255),
}


@lru_cache(maxsize=64)
def _load_court_homography(path: str) -> CourtHomography:
    return CourtHomography.load(path)


def decode_video_frame(bundle: SourceBundle, frame: int) -> np.ndarray:
    if not 0 <= frame < bundle.frame_count:
        raise ValueError(f"frame outside source bounds: {frame}")
    capture = cv2.VideoCapture(str(bundle.video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok or image is None:
        raise ValueError(f"could not decode {bundle.source_id} frame {frame}")
    if image.shape[:2] != bundle.image_size[::-1]:
        raise ValueError(f"decoded frame geometry changed for {bundle.source_id} frame {frame}")
    return image


def _draw_pose(frame: np.ndarray, pose: Mapping[str, Any]) -> None:
    landmarks = pose.get("pose_landmarks")
    if not isinstance(landmarks, list) or len(landmarks) != POSE_NUM_LANDMARKS:
        return
    bbox = pose.get("pose_bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    x1, y1, x2, y2 = (float(value) for value in bbox)
    points: list[tuple[int, int, float]] = []
    for landmark in landmarks:
        if not isinstance(landmark, Mapping):
            continue
        nx = float(landmark.get("x", 0.0))
        ny = float(landmark.get("y", 0.0))
        visibility = float(landmark.get("visibility", 0.0))
        points.append((int(round(x1 + nx * (x2 - x1))), int(round(y1 + ny * (y2 - y1))), visibility))
    if len(points) != POSE_NUM_LANDMARKS:
        return
    for start, end in POSE_CONNECTIONS:
        if start >= len(points) or end >= len(points):
            continue
        ax, ay, av = points[start]
        bx, by, bv = points[end]
        if av >= 0.5 and bv >= 0.5:
            cv2.line(frame, (ax, ay), (bx, by), (255, 180, 0), 2, cv2.LINE_AA)
    for x, y, visibility in points:
        if visibility >= 0.5:
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)


def draw_players_and_pose(
    frame: np.ndarray,
    bundle: SourceBundle,
    frame_index: int,
    *,
    show_pose: bool = True,
    visibility_threshold: float = 0.5,
) -> np.ndarray:
    assignment = bundle.assignments.get(frame_index, {})
    for role, color in ROLE_COLORS.items():
        slot = assignment.get("slots", {}).get(role, {})
        item = slot.get("assignment")
        if not isinstance(item, Mapping) or not item.get("bbox"):
            continue
        bbox = item["bbox"]
        try:
            x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
        except (TypeError, ValueError):
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        activity = float(item.get("activity", 0.0))
        confidence = float(slot.get("confidence", 0.0))
        label = f"{role} act={activity:.2f} conf={confidence:.2f}"
        cv2.putText(
            frame, label, (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
        if show_pose and item.get("track_id") is not None:
            pose = bundle.poses.get((frame_index, int(item["track_id"])))
            if isinstance(pose, Mapping) and pose.get("status") == "detected":
                _draw_pose(frame, pose)
    return frame


def draw_shuttle_overlay(
    frame: np.ndarray,
    shuttle: Mapping[str, Any],
) -> np.ndarray:
    height, width = frame.shape[:2]
    position = shuttle.get("top_position_normalized")
    bbox = shuttle.get("top_bbox_normalized")
    if isinstance(position, (list, tuple)) and len(position) == 2:
        try:
            cx = int(round(float(position[0]) * width))
            cy = int(round(float(position[1]) * height))
        except (TypeError, ValueError):
            cx = cy = None
        if cx is not None and cy is not None:
            cv2.circle(frame, (cx, cy), 12, (0, 255, 255), 2, cv2.LINE_AA)
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x1 = int(round(float(bbox[0]) * width))
            y1 = int(round(float(bbox[1]) * height))
            x2 = int(round(float(bbox[2]) * width))
            y2 = int(round(float(bbox[3]) * height))
        except (TypeError, ValueError):
            x1 = y1 = x2 = y2 = None
        if None not in (x1, y1, x2, y2):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
    count = shuttle.get("candidate_count", 0)
    peak = shuttle.get("top_peak", 0.0)
    cv2.putText(
        frame,
        f"shuttle candidates={count} peak={float(peak):.3f}",
        (12, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def render_frame_jpeg(
    bundle: SourceBundle,
    frame: int,
    *,
    show_pose: bool = True,
    show_court: bool = False,
    show_shuttle: bool = True,
    shuttle: Mapping[str, Any] | None = None,
) -> bytes:
    image = decode_video_frame(bundle, frame)
    if show_court:
        homography = _load_court_homography(str(bundle.calibration_path))
        image = draw_court_overlay(image, homography)
    image = draw_players_and_pose(image, bundle, frame, show_pose=show_pose)
    if show_shuttle and shuttle is not None:
        image = draw_shuttle_overlay(image, shuttle)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError(f"could not encode {bundle.source_id} frame {frame}")
    return encoded.tobytes()
