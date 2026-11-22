"""Shared player/court observation parsing and feature representation.

These helpers do not depend on candidate proposals, annotation queues, or study
orchestration. Both rally and selector datasets consume the same representation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from src.court_projection import HALF_LENGTH, HALF_WIDTH
from .batch import MASKED_TARGET
from .rally_intervals import RallyInterval

FrameView = Literal["candidates_only", "players_court", "full_context"]
PoseCoordinateMode = Literal["image", "player_relative"]

POSE_REPRESENTATION_VERSION = {
    "image": "image-normalized-v1",
    "player_relative": "pelvis-root-shoulder-torso-bbox-scale-v1",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl_artifact(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("type") != "metadata":
        raise ValueError(f"artifact has no metadata record: {path}")
    return records[0], records[1:]


def rally_boundary_targets(
    intervals: Sequence[RallyInterval],
    partial_start_rally_ids: frozenset[str],
    frame_indices: Sequence[int],
    owned_frames: frozenset[int],
    inplay_targets: Sequence[int],
    *,
    fps: float,
    taper_seconds: float = 0.25,
    boundary_range: range | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive owned/reviewed soft boundary targets from strict intervals."""
    if fps <= 0 or taper_seconds <= 0:
        raise ValueError("boundary target FPS and taper must be positive")
    starts: list[float] = []
    ends: list[float] = []
    for frame, inplay in zip(frame_indices, inplay_targets):
        if frame not in owned_frames or int(inplay) not in (0, 1):
            starts.append(float(MASKED_TARGET))
            ends.append(float(MASKED_TARGET))
            continue
        start = end = 0.0
        for interval in intervals:
            if (
                interval.rally_id not in partial_start_rally_ids
                and (boundary_range is None or interval.start_frame in boundary_range)
            ):
                start = max(
                    start,
                    max(0.0, 1.0 - abs(frame - interval.start_frame) / (fps * taper_seconds)),
                )
            if boundary_range is None or interval.end_frame in boundary_range:
                end = max(
                    end,
                    max(0.0, 1.0 - abs(frame - interval.end_frame) / (fps * taper_seconds)),
                )
        starts.append(start)
        ends.append(end)
    return torch.tensor(starts), torch.tensor(ends)


def validate_calibration(path: Path, image_size: tuple[int, int]) -> None:
    """Validate source calibration without exposing camera geometry to the model."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if tuple(value.get("image_size", ())) != image_size:
        raise ValueError("calibration image_size does not match source artifacts")
    matrix = np.asarray(value.get("image_to_court"), dtype=float)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or abs(np.linalg.det(matrix)) < 1e-12
    ):
        raise ValueError("calibration homography is invalid")


def frame_context(
    assignment: Mapping[str, Any],
    poses: Mapping[tuple[int, int], Mapping[str, Any]],
    frame: int,
    image_size: tuple[int, int],
    mode: FrameView,
    visibility_threshold: float,
    pose_coordinate_mode: PoseCoordinateMode,
) -> tuple[list[float], list[bool]]:
    if mode == "candidates_only":
        return [], []
    width, height = image_size
    values, validity = [], []
    player_poses: list[tuple[list[float], list[bool]]] = []
    for role in ("P1", "P2"):
        slot = assignment.get("slots", {}).get(role, {})
        item = slot.get("assignment")
        if not isinstance(item, Mapping):
            values.extend([0.0] * 11)
            validity.extend([False] * 11)
            player_poses.append(([0.0] * 66, [False] * 66))
            continue
        bbox, foot = item.get("bbox"), item.get("foot")
        player = [
            float(bbox[0]) / width,
            float(bbox[1]) / height,
            float(bbox[2]) / width,
            float(bbox[3]) / height,
            float(foot[0]) / width,
            float(foot[1]) / height,
            float(item["court_x"]) / HALF_WIDTH,
            float(item["court_y"]) / HALF_LENGTH,
            float(item["detection_confidence"]),
            float(slot["confidence"]),
            float(item["activity"]),
        ]
        if not all(math.isfinite(value) for value in player):
            raise ValueError(f"non-finite player assignment at frame {frame}")
        values.extend(player)
        validity.extend([True] * 11)
        pose = poses.get((frame, int(item["track_id"])))
        pose_values, pose_valid = [0.0] * 66, [False] * 66
        if (
            pose is not None
            and pose.get("status") == "detected"
            and len(pose.get("pose_landmarks", ())) == 33
        ):
            pb = pose.get("pose_bbox")
            if not isinstance(pb, list) or len(pb) != 4:
                raise ValueError("accepted pose record has no pose_bbox")
            points: list[tuple[float, float]] = []
            point_valid: list[bool] = []
            for landmark in pose["pose_landmarks"]:
                x = (
                    float(pb[0]) + float(landmark["x"]) * (float(pb[2]) - float(pb[0]))
                )
                y = (
                    float(pb[1]) + float(landmark["y"]) * (float(pb[3]) - float(pb[1]))
                )
                visible = (
                    math.isfinite(x)
                    and math.isfinite(y)
                    and float(landmark.get("visibility", 0)) >= visibility_threshold
                )
                points.append((x, y))
                point_valid.append(visible)
            if pose_coordinate_mode == "image":
                normalized = [(x / width, y / height) for x, y in points]
            else:
                hips = [index for index in (23, 24) if point_valid[index]]
                root = (
                    tuple(
                        sum(points[index][axis] for index in hips) / len(hips)
                        for axis in (0, 1)
                    )
                    if hips
                    else (
                        (float(pb[0]) + float(pb[2])) / 2,
                        (float(pb[1]) + float(pb[3])) / 2,
                    )
                )
                scales: list[float] = []
                if point_valid[11] and point_valid[12]:
                    scales.append(math.dist(points[11], points[12]))
                shoulders = [index for index in (11, 12) if point_valid[index]]
                if shoulders and hips:
                    shoulder_mid = tuple(
                        sum(points[index][axis] for index in shoulders) / len(shoulders)
                        for axis in (0, 1)
                    )
                    scales.append(math.dist(shoulder_mid, root))
                scale = next(
                    (value for value in scales if value > 1.0),
                    max(1.0, float(pb[3]) - float(pb[1])),
                )
                normalized = [
                    ((x - root[0]) / scale, (y - root[1]) / scale)
                    for x, y in points
                ]
            pose_values, pose_valid = [], []
            for (x, y), visible in zip(normalized, point_valid):
                pose_values.extend([x if visible else 0.0, y if visible else 0.0])
                pose_valid.extend([visible, visible])
        player_poses.append((pose_values, pose_valid))
    if mode == "full_context":
        for pose_values, pose_valid in player_poses:
            values.extend(pose_values)
            validity.extend(pose_valid)
    return values, validity
