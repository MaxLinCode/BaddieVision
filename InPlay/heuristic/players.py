"""Two-player tracking and pose activity extraction.

Ultralytics ByteTrack supplies persistent IDs. MediaPipe runs on each selected
crop; this module writes a separate player CSV consumed optionally by segment.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import defaultdict
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from src.court_projection import HALF_LENGTH, HALF_WIDTH, CourtHomography

PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"
COURT_SELECTION_MARGIN_METERS = 0.3


def crop_point_to_image(
    point: tuple[float, float], crop: tuple[float, float, float, float]
) -> tuple[float, float]:
    x1, y1, x2, y2 = crop
    return x1 + point[0] * (x2 - x1), y1 + point[1] * (y2 - y1)


def select_on_court_tracks(
    observations: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    frame_size: tuple[int, int],
    court_homography: CourtHomography | None = None,
) -> list[int]:
    """Select persistent central/lower-image tracks, suppressing spectators."""
    width, height = frame_size
    scores: list[tuple[float, int]] = []
    for track_id, entries in observations.items():
        if not entries:
            continue
        centers = [((box[0] + box[2]) / 2, box[3]) for _, box in entries]
        if court_homography is not None:
            inside = 0
            for x, y in centers:
                try:
                    point = court_homography.project_to_court([(x, y)])[0]
                except ValueError:
                    continue
                if (
                    -HALF_WIDTH - COURT_SELECTION_MARGIN_METERS
                    <= point[0]
                    <= HALF_WIDTH + COURT_SELECTION_MARGIN_METERS
                    and -HALF_LENGTH - COURT_SELECTION_MARGIN_METERS
                    <= point[1]
                    <= HALF_LENGTH + COURT_SELECTION_MARGIN_METERS
                ):
                    inside += 1
            region = inside / len(centers)
        else:
            region = sum(
                0.08 * width <= x <= 0.92 * width and 0.20 * height <= y <= 1.02 * height
                for x, y in centers
            ) / len(centers)
        scores.append((len(entries) * region, track_id))
    return [track_id for score, track_id in sorted(scores, reverse=True)[:2] if score > 0]


def _court_homography(path: str | Path | None) -> CourtHomography | None:
    if path is None:
        return None
    try:
        return CourtHomography.load(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
        return None


def detector_class_name(detector: Any, class_id: int = PERSON_CLASS_ID) -> str | None:
    names = getattr(detector, "names", None)
    if isinstance(names, dict):
        value = names.get(class_id)
    elif isinstance(names, (list, tuple)) and class_id < len(names):
        value = names[class_id]
    else:
        value = None
    return str(value).strip().lower() if value is not None else None


def ensure_person_detector(detector: Any, model: str | Path) -> None:
    class_name = detector_class_name(detector)
    if class_name != PERSON_CLASS_NAME:
        raise ValueError(
            f"{model} is not a COCO person detector: class {PERSON_CLASS_ID} is "
            f"{class_name!r}, expected {PERSON_CLASS_NAME!r}"
        )


def _serialize_landmarks(landmarks: Any) -> list[dict[str, float]]:
    return [
        {
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(landmark.z),
            "visibility": float(landmark.visibility),
        }
        for landmark in landmarks
    ]


def _draw_pose_landmarks(
    frame: Any, crop: tuple[float, float, float, float], pose_landmarks: list[dict[str, float]] | None
) -> None:
    if not pose_landmarks:
        return
    import cv2
    import mediapipe as mp

    points = [
        (
            int(round(crop_point_to_image((item["x"], item["y"]), crop)[0])),
            int(round(crop_point_to_image((item["x"], item["y"]), crop)[1])),
            float(item.get("visibility", 0.0)),
        )
        for item in pose_landmarks
    ]
    for start, end in mp.solutions.pose.POSE_CONNECTIONS:
        if start >= len(points) or end >= len(points):
            continue
        ax, ay, av = points[start]
        bx, by, bv = points[end]
        if av < 0.5 or bv < 0.5:
            continue
        cv2.line(frame, (ax, ay), (bx, by), (255, 180, 0), 2, cv2.LINE_AA)
    for x, y, visibility in points:
        if visibility >= 0.5:
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)


def build_player_rows(raw_data: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    activity_window: deque[float] = deque(maxlen=30)
    selected_track_ids = [int(track_id) for track_id in raw_data.get("selected_track_ids", [])]
    for frame in raw_data.get("frames", []):
        rows.append(_player_row_from_frame(frame, selected_track_ids, activity_window))
    return rows


def write_player_csv(rows: list[dict[str, object]], output: str | Path) -> None:
    fieldnames = [
        "Frame", "player_activity", "players_inactive", "activity_window", "inactivity_window",
        "player1_track_id", "player1_valid", "player1_foot_x", "player1_foot_y", "player1_activity",
        "player2_track_id", "player2_valid", "player2_foot_x", "player2_foot_y", "player2_activity",
    ]
    with Path(output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _player_fieldnames() -> list[str]:
    return [
        "Frame", "player_activity", "players_inactive", "activity_window", "inactivity_window",
        "player1_track_id", "player1_valid", "player1_foot_x", "player1_foot_y", "player1_activity",
        "player2_track_id", "player2_valid", "player2_foot_x", "player2_foot_y", "player2_activity",
    ]


def _player_row_from_frame(
    frame: dict[str, Any], selected_track_ids: list[int], activity_window: deque[float]
) -> dict[str, object]:
    activities = []
    row: dict[str, object] = {"Frame": int(frame["frame"])}
    detections = {
        int(item["track_id"]): item
        for item in frame.get("detections", [])
        if item.get("selected")
    }
    for slot, track_id in enumerate(selected_track_ids, 1):
        detection = detections.get(track_id)
        if detection is None:
            row.update(
                {
                    f"player{slot}_track_id": track_id,
                    f"player{slot}_valid": 0,
                }
            )
            continue
        foot = detection.get("foot", [0.0, 0.0])
        activity = float(detection.get("activity", 0.0))
        valid = int(detection.get("crop_valid", 0))
        row.update(
            {
                f"player{slot}_track_id": track_id,
                f"player{slot}_valid": valid,
                f"player{slot}_foot_x": float(foot[0]),
                f"player{slot}_foot_y": float(foot[1]),
                f"player{slot}_activity": activity,
            }
        )
        activities.append(activity)
    row["player_activity"] = min(1.0, sum(activities) * 20) if activities else 0.0
    row["players_inactive"] = int(not activities or max(activities) < 0.001)
    activity_window.append(float(row["player_activity"]))
    row["activity_window"] = sum(activity_window) / len(activity_window)
    row["inactivity_window"] = int(
        len(activity_window) == activity_window.maxlen
        and row["activity_window"] < 0.05
    )
    return row


def run(
    video: str | Path,
    output: str | Path,
    model: str = "yolov8n.pt",
    court_calibration: str | Path | None = None,
    raw_output: str | Path | None = None,
    vis_output: str | Path | None = None,
) -> None:
    import cv2
    import mediapipe as mp
    from ultralytics import YOLO

    detector = YOLO(model)
    ensure_person_detector(detector, model)
    capture = cv2.VideoCapture(str(video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    results = detector.track(
        source=str(video),
        classes=[PERSON_CLASS_ID],
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
    )
    frame_dump = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl")
    frame_dump_path = Path(frame_dump.name)
    observations: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    try:
        for frame_index, result in enumerate(results):
            detections: list[dict[str, object]] = []
            if result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
                boxes = result.boxes.xyxy.cpu().tolist()
                for track_id, box_values in zip(ids, boxes):
                    box = tuple(float(value) for value in box_values)
                    detections.append({"track_id": int(track_id), "bbox": [float(value) for value in box]})
                    observations[track_id].append((frame_index, box))
            frame_dump.write(json.dumps({"frame": frame_index, "detections": detections}) + "\n")
        frame_dump.close()
        if court_calibration is None:
            print(
                "Warning: no court calibration provided; player selection falls back to a "
                "screen-space heuristic that may be unreliable for skewed cameras."
            )
        selected = select_on_court_tracks(
            observations, (width, height), _court_homography(court_calibration)
        )
        pose = mp.solutions.pose.Pose(static_image_mode=True)
        previous_feet: dict[int, tuple[float, float]] = {}
        previous_poses: dict[int, np.ndarray] = {}
        capture = cv2.VideoCapture(str(video))
        writer = None
        raw_handle = None
        player_handle = Path(output).open("w", newline="", encoding="utf-8")
        player_writer = csv.DictWriter(
            player_handle, fieldnames=_player_fieldnames(), extrasaction="ignore"
        )
        player_writer.writeheader()
        activity_window: deque[float] = deque(maxlen=30)
        if raw_output is not None:
            raw_handle = Path(raw_output).open("w", encoding="utf-8")
            raw_handle.write(
                json.dumps(
                    {
                        "type": "metadata",
                        "schema_version": 1,
                        "video": str(video),
                        "frame_size": [width, height],
                        "detector": {
                            "model": str(model),
                            "class_id": PERSON_CLASS_ID,
                            "class_name": PERSON_CLASS_NAME,
                        },
                        "selected_track_ids": selected,
                    }
                )
                + "\n"
            )
        if vis_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            writer = cv2.VideoWriter(str(vis_output), fourcc, fps, (width, height))
        with frame_dump_path.open(encoding="utf-8") as frame_handle:
            for raw_line in frame_handle:
                frame_info = json.loads(raw_line)
                frame_index = int(frame_info["frame"])
                ret, image = capture.read()
                if not ret:
                    break
                detections = list(frame_info["detections"])
                frame_entry: dict[str, Any] = {"frame": frame_index, "detections": []}
                by_id = {int(item["track_id"]): item["bbox"] for item in detections}
                for slot, track_id in enumerate(selected, 1):
                    if track_id not in by_id:
                        continue
                    box = by_id[track_id]
                    x1, y1, x2, y2 = [int(value) for value in box]
                    crop = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                    foot = ((x1 + x2) / 2, y2)
                    valid = crop.size > 0
                    pose_activity = 0.0
                    serialized_landmarks = None
                    if valid:
                        pose_result = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        if pose_result.pose_landmarks:
                            landmarks = pose_result.pose_landmarks.landmark
                            serialized_landmarks = _serialize_landmarks(landmarks)
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
                    frame_entry["detections"].append(
                        {
                            "track_id": track_id,
                            "slot": slot,
                            "bbox": [float(value) for value in box],
                            "selected": True,
                            "crop_valid": bool(valid),
                            "foot": [float(foot[0]), float(foot[1])],
                            "activity": pose_activity,
                            "pose_landmarks": serialized_landmarks,
                        }
                    )
                    if writer is not None:
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)
                        cv2.putText(
                            image,
                            f"P{slot} ID {track_id} act {pose_activity:.3f}",
                            (x1, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 200, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        _draw_pose_landmarks(image, box, serialized_landmarks)
                selected_ids = set(selected)
                existing_ids = {int(item["track_id"]) for item in frame_entry["detections"]}
                for detection in detections:
                    track_id = int(detection["track_id"])
                    if track_id in existing_ids:
                        continue
                    frame_entry["detections"].append(
                        {
                            "track_id": track_id,
                            "bbox": [float(value) for value in detection["bbox"]],
                            "selected": track_id in selected_ids,
                            "crop_valid": False,
                            "pose_landmarks": None,
                        }
                    )
                    if writer is not None:
                        x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
                        cv2.rectangle(image, (x1, y1), (x2, y2), (120, 120, 120), 1)
                        cv2.putText(
                            image,
                            f"ID {track_id}",
                            (x1, max(16, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (180, 180, 180),
                            1,
                            cv2.LINE_AA,
                        )
                player_writer.writerow(_player_row_from_frame(frame_entry, selected, activity_window))
                if raw_handle is not None:
                    raw_handle.write(json.dumps({"type": "frame", **frame_entry}) + "\n")
                if writer is not None:
                    writer.write(image)
        capture.release()
        if writer is not None:
            writer.release()
        if raw_handle is not None:
            raw_handle.close()
        player_handle.close()
        pose.close()
    finally:
        frame_dump.close()
        if frame_dump_path.exists():
            frame_dump_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="COCO person detector checkpoint for player tracking; class 0 is treated as person.",
    )
    parser.add_argument("--court-calibration")
    parser.add_argument("--raw-output")
    parser.add_argument("--vis-output")
    args = parser.parse_args(argv)
    run(
        args.video,
        args.output,
        args.model,
        args.court_calibration,
        raw_output=args.raw_output,
        vis_output=args.vis_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
