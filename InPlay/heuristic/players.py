"""Compatibility helpers for layered player tracking and historical artifacts."""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Any

PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"


def crop_point_to_image(
    point: tuple[float, float], crop: tuple[float, float, float, float]
) -> tuple[float, float]:
    x1, y1, x2, y2 = crop
    return x1 + point[0] * (x2 - x1), y1 + point[1] * (y2 - y1)


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


def build_player_rows(raw_data: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    activity_window: deque[float] = deque(maxlen=30)
    selected_track_ids = [
        int(track_id)
        for track_id in raw_data.get("player_slot_track_ids", raw_data.get("selected_track_ids", []))
    ]
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
