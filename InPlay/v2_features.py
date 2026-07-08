"""InPlay v2 learned-model feature schema.

This module is intentionally separate from the shot-classifier feature schema.
Shot classification owns the 66 pose + 7 shuttle + 3 court-anchor order; InPlay
v2 uses richer rally-segmentation signals from the heuristic pipeline.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.court_projection import HALF_LENGTH, HALF_WIDTH, CourtHomography

from .heuristic.tracks import FrameFeature

POSE_KEYPOINTS = 33

POSE_FEATURE_NAMES = [
    f"pose_{keypoint}_{axis}"
    for keypoint in range(POSE_KEYPOINTS)
    for axis in ("x", "y")
]

SHUTTLE_FEATURE_NAMES = [
    "shuttle_x",
    "shuttle_y",
    "shuttle_visible",
    "shuttle_peak_value",
    "shuttle_confidence_available",
    "shuttle_cleaned",
    "shuttle_interpolated",
    "shuttle_removed",
    "shuttle_missing_count",
    "shuttle_visible_streak",
    "shuttle_smooth_x",
    "shuttle_smooth_y",
    "shuttle_speed",
    "shuttle_acceleration",
    "shuttle_cumulative_distance",
    "shuttle_inside_courtish",
]

PLAYER_FEATURE_NAMES = [
    "player_activity",
    "players_inactive",
    "activity_window",
    "player1_court_x",
    "player1_court_y",
    "player1_observed",
    "player2_court_x",
    "player2_court_y",
    "player2_observed",
]

FEATURE_NAMES = POSE_FEATURE_NAMES + SHUTTLE_FEATURE_NAMES + PLAYER_FEATURE_NAMES
FEATURE_DIM = len(FEATURE_NAMES)


def load_pose_frames(path: str | Path) -> list[Mapping]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("pose JSON must contain a frame list")
    return data


def pose_features_for_frames(
    pose_frames: Sequence[Mapping],
    frame_indices: Sequence[int],
    num_keypoints: int = POSE_KEYPOINTS,
) -> np.ndarray:
    features = np.zeros((len(frame_indices), num_keypoints * 2), dtype=np.float32)
    valid = np.zeros(len(frame_indices), dtype=bool)
    for row, frame_index in enumerate(frame_indices):
        if frame_index < 0 or frame_index >= len(pose_frames):
            continue
        keypoints = pose_frames[frame_index].get("keypoints", {})
        if not keypoints:
            continue
        valid[row] = True
        for keypoint in range(num_keypoints):
            landmark = keypoints.get(str(keypoint), {})
            features[row, keypoint * 2] = float(landmark.get("x", 0.0) or 0.0)
            features[row, keypoint * 2 + 1] = float(landmark.get("y", 0.0) or 0.0)
    _fill_missing_rows(features, valid)
    return features


def shuttle_features_for_frames(
    track_frames: Sequence[FrameFeature],
    frame_indices: Sequence[int],
) -> np.ndarray:
    by_frame = {item.frame: item for item in track_frames}
    rows = np.zeros((len(frame_indices), len(SHUTTLE_FEATURE_NAMES)), dtype=np.float32)
    for row, frame in enumerate(frame_indices):
        item = by_frame.get(frame)
        if item is None:
            continue
        rows[row] = np.asarray(
            [
                _finite_or_zero(item.x),
                _finite_or_zero(item.y),
                float(item.visibility),
                float(item.peak_value) if item.peak_value is not None else 0.0,
                float(item.confidence_available),
                float(item.cleaned),
                float(item.interpolated),
                float(item.removed),
                float(item.missing_count),
                float(item.visible_streak),
                _finite_or_zero(item.smooth_x),
                _finite_or_zero(item.smooth_y),
                float(item.speed),
                float(item.acceleration),
                float(item.cumulative_distance),
                _inside_court_value(item.inside_courtish),
            ],
            dtype=np.float32,
        )
    return rows


def read_player_rows(path: str | Path | None) -> dict[int, Mapping[str, str]]:
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "Frame" not in set(reader.fieldnames or ()):
            raise ValueError("player CSV requires a Frame column")
        return {int(row["Frame"]): row for row in reader}


def player_features_for_frames(
    player_rows: Mapping[int, Mapping[str, object]],
    frame_indices: Sequence[int],
    court_homography: CourtHomography | None = None,
) -> np.ndarray:
    rows = np.zeros((len(frame_indices), len(PLAYER_FEATURE_NAMES)), dtype=np.float32)
    for index, frame in enumerate(frame_indices):
        source = player_rows.get(frame)
        if source is None:
            continue
        rows[index, 0] = _float_field(source, "player_activity")
        rows[index, 1] = _float_field(source, "players_inactive")
        rows[index, 2] = _float_field(source, "activity_window")
        for slot in (1, 2):
            offset = 3 + (slot - 1) * 3
            observed = _float_field(source, f"player{slot}_valid") > 0
            foot_x = _float_field(source, f"player{slot}_foot_x")
            foot_y = _float_field(source, f"player{slot}_foot_y")
            if observed and court_homography is not None:
                try:
                    court = court_homography.project_to_court([(foot_x, foot_y)])[0]
                except ValueError:
                    observed = False
                else:
                    rows[index, offset] = float(court[0] / HALF_WIDTH)
                    rows[index, offset + 1] = float(court[1] / HALF_LENGTH)
            rows[index, offset + 2] = float(observed)
    return rows


def build_inplay_v2_features(
    pose_frames: Sequence[Mapping],
    track_frames: Sequence[FrameFeature],
    frame_indices: Sequence[int],
    player_rows: Mapping[int, Mapping[str, object]] | None = None,
    court_homography: CourtHomography | None = None,
) -> np.ndarray:
    """Return InPlay v2 features in the order described by ``FEATURE_NAMES``."""

    pose = pose_features_for_frames(pose_frames, frame_indices)
    shuttle = shuttle_features_for_frames(track_frames, frame_indices)
    players = player_features_for_frames(player_rows or {}, frame_indices, court_homography)
    features = np.concatenate([pose, shuttle, players], axis=1).astype(np.float32)
    if features.shape[1] != FEATURE_DIM:
        raise AssertionError(f"expected {FEATURE_DIM} features, got {features.shape[1]}")
    return features


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_length: int,
    source_id: str,
    frame_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same frame count")
    if len(features) < sequence_length:
        raise ValueError("not enough frames for one sequence")
    frame_indices = list(frame_indices or range(len(features)))
    sequences = []
    targets = []
    metadata: list[dict[str, object]] = []
    for start in range(0, len(features) - sequence_length + 1):
        end = start + sequence_length
        sequences.append(features[start:end])
        targets.append(labels[start:end])
        metadata.append(
            {
                "source_id": source_id,
                "start_frame": frame_indices[start],
                "end_frame": frame_indices[end - 1],
            }
        )
    return (
        np.stack(sequences).astype(np.float32),
        np.stack(targets).astype(np.float32),
        metadata,
    )


def _fill_missing_rows(values: np.ndarray, valid: np.ndarray) -> None:
    for index, is_valid in enumerate(valid):
        if is_valid:
            continue
        previous = next((j for j in range(index - 1, -1, -1) if valid[j]), None)
        following = next((j for j in range(index + 1, len(valid)) if valid[j]), None)
        if previous is not None and following is not None:
            values[index] = (values[previous] + values[following]) / 2
        elif previous is not None:
            values[index] = values[previous]
        elif following is not None:
            values[index] = values[following]


def _finite_or_zero(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def _inside_court_value(value: bool | None) -> float:
    if value is True:
        return 1.0
    if value is False:
        return -1.0
    return 0.0


def _float_field(row: Mapping[str, object], field: str) -> float:
    value = row.get(field, 0.0)
    if value in ("", None):
        return 0.0
    return float(value)
