"""Two-player tracking and pose activity extraction.

Ultralytics ByteTrack supplies persistent IDs. MediaPipe runs on each selected
crop; this module writes a separate player CSV consumed optionally by segment.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections import deque
from pathlib import Path

import numpy as np


def crop_point_to_image(
    point: tuple[float, float], crop: tuple[float, float, float, float]
) -> tuple[float, float]:
    x1, y1, x2, y2 = crop
    return x1 + point[0] * (x2 - x1), y1 + point[1] * (y2 - y1)


def select_on_court_tracks(
    observations: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    frame_size: tuple[int, int],
    court_bounds: tuple[float, float, float, float] | None = None,
) -> list[int]:
    """Select persistent central/lower-image tracks, suppressing spectators."""
    width, height = frame_size
    scores: list[tuple[float, int]] = []
    for track_id, entries in observations.items():
        if not entries:
            continue
        centers = [((box[0] + box[2]) / 2, box[3]) for _, box in entries]
        if court_bounds:
            left, top, right, bottom = court_bounds
            region = sum(left <= x <= right and top <= y <= bottom for x, y in centers) / len(centers)
        else:
            region = sum(
                0.08 * width <= x <= 0.92 * width and 0.20 * height <= y <= 1.02 * height
                for x, y in centers
            ) / len(centers)
        scores.append((len(entries) * region, track_id))
    return [track_id for score, track_id in sorted(scores, reverse=True)[:2] if score > 0]


def _court_bounds(path: str | Path | None) -> tuple[float, float, float, float] | None:
    if path is None:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "image_landmarks" in data:
            points = np.asarray(list(data["image_landmarks"].values()), dtype=float)
        else:
            return None
        low, high = points.min(axis=0), points.max(axis=0)
        margin = (high - low) * 0.15
        return tuple([*(low - margin), *(high + margin)])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def run(
    video: str | Path,
    output: str | Path,
    model: str = "yolov8n.pt",
    court_calibration: str | Path | None = None,
) -> None:
    import cv2
    import mediapipe as mp
    from ultralytics import YOLO

    detector = YOLO(model)
    capture = cv2.VideoCapture(str(video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    results = detector.track(
        source=str(video), classes=[0], tracker="bytetrack.yaml", stream=True, persist=True
    )
    raw_frames: list[tuple[int, np.ndarray, list[tuple[int, tuple[float, float, float, float]]]]] = []
    observations: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    for frame_index, result in enumerate(results):
        detections = []
        if result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            for track_id, box_values in zip(ids, boxes):
                box = tuple(float(value) for value in box_values)
                detections.append((track_id, box))
                observations[track_id].append((frame_index, box))
        raw_frames.append((frame_index, result.orig_img, detections))
    selected = select_on_court_tracks(
        observations, (width, height), _court_bounds(court_calibration)
    )
    pose = mp.solutions.pose.Pose(static_image_mode=True)
    rows = []
    previous_feet: dict[int, tuple[float, float]] = {}
    previous_poses: dict[int, np.ndarray] = {}
    activity_window: deque[float] = deque(maxlen=30)
    for frame_index, image, detections in raw_frames:
        activities = []
        row: dict[str, object] = {"Frame": frame_index}
        by_id = dict(detections)
        for slot, track_id in enumerate(selected, 1):
            if track_id not in by_id:
                row.update(
                    {
                        f"player{slot}_track_id": track_id,
                        f"player{slot}_valid": 0,
                    }
                )
                continue
            box = by_id[track_id]
            x1, y1, x2, y2 = [int(value) for value in box]
            crop = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            foot = ((x1 + x2) / 2, y2)
            valid = crop.size > 0
            pose_activity = 0.0
            if valid:
                pose_result = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if pose_result.pose_landmarks:
                    landmarks = pose_result.pose_landmarks.landmark
                    source_pose = np.asarray(
                        [
                            crop_point_to_image((landmark.x, landmark.y), box)
                            for landmark in landmarks
                        ]
                    ) / np.asarray([width, height])
                    if track_id in previous_poses:
                        pose_activity = float(
                            np.linalg.norm(source_pose - previous_poses[track_id], axis=1).mean()
                        )
                    previous_poses[track_id] = source_pose
                    feet = [
                        crop_point_to_image((landmarks[index].x, landmarks[index].y), box)
                        for index in (29, 30, 31, 32)
                        if landmarks[index].visibility >= 0.5
                    ]
                    if feet:
                        foot = tuple(np.mean(feet, axis=0))
                if track_id in previous_feet and pose_activity == 0.0:
                    pose_activity = float(
                        np.linalg.norm(np.asarray(foot) - previous_feet[track_id])
                        / np.hypot(width, height)
                    )
                previous_feet[track_id] = foot
            row.update(
                {
                    f"player{slot}_track_id": track_id,
                    f"player{slot}_valid": int(valid),
                    f"player{slot}_foot_x": foot[0],
                    f"player{slot}_foot_y": foot[1],
                    f"player{slot}_activity": pose_activity,
                }
            )
            activities.append(pose_activity)
        row["player_activity"] = min(1.0, sum(activities) * 20) if activities else 0.0
        row["players_inactive"] = int(not activities or max(activities) < 0.001)
        activity_window.append(float(row["player_activity"]))
        row["activity_window"] = sum(activity_window) / len(activity_window)
        row["inactivity_window"] = int(
            len(activity_window) == activity_window.maxlen
            and row["activity_window"] < 0.05
        )
        rows.append(row)
    pose.close()
    fieldnames = [
        "Frame", "player_activity", "players_inactive", "activity_window", "inactivity_window",
        "player1_track_id", "player1_valid", "player1_foot_x", "player1_foot_y", "player1_activity",
        "player2_track_id", "player2_valid", "player2_foot_x", "player2_foot_y", "player2_activity",
    ]
    with Path(output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--court-calibration")
    args = parser.parse_args(argv)
    run(args.video, args.output, args.model, args.court_calibration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
