from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from InPlay.decoding import decode_probabilities, scale_config_for_fps
from InPlay.heuristic.config import HeuristicConfig
from InPlay.heuristic.evaluate import Interval, evaluate, frame_classification_metrics
from InPlay.heuristic.tracks import preprocess_tracks, read_track_csv
from InPlay.splits import source_group_split
from InPlay.v2_features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    build_inplay_v2_features,
    build_sequences,
)
from src.court_projection import CourtHomography


def _write_tracks(path: Path, peak: bool) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Frame", "X", "Y", "Visibility"] + (["PeakValue"] if peak else []))
        for frame in range(8):
            row = [frame, 100 + frame * 10, 100 + frame * 5, 1]
            if peak:
                row.append(0.9)
            writer.writerow(row)


def _pose_frames(count: int) -> list[dict[str, object]]:
    return [
        {
            "keypoints": {
                str(keypoint): {
                    "x": frame / 100 + keypoint / 1000,
                    "y": frame / 200 + keypoint / 2000,
                    "visibility": 1.0,
                }
                for keypoint in range(33)
            }
        }
        for frame in range(count)
    ]


def test_v2_feature_schema_shape_and_legacy_peak_compatibility(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.csv"
    with_peak = tmp_path / "peak.csv"
    _write_tracks(legacy, peak=False)
    _write_tracks(with_peak, peak=True)
    config = HeuristicConfig(smoothing_window=3)
    legacy_frames = preprocess_tracks(read_track_csv(legacy, (640, 360), config), config)
    peak_frames = preprocess_tracks(read_track_csv(with_peak, (640, 360), config), config)

    player_rows = {
        frame: {
            "Frame": frame,
            "player_activity": 0.25,
            "players_inactive": 0,
            "activity_window": 0.5,
            "player1_valid": 1,
            "player1_foot_x": 1.0,
            "player1_foot_y": 2.0,
            "player2_valid": 0,
        }
        for frame in range(8)
    }
    features = build_inplay_v2_features(
        _pose_frames(8),
        peak_frames,
        list(range(8)),
        player_rows,
        CourtHomography(np.eye(3)),
    )
    legacy_features = build_inplay_v2_features(_pose_frames(8), legacy_frames, list(range(8)))

    assert features.shape == (8, FEATURE_DIM)
    assert legacy_features.shape == (8, FEATURE_DIM)
    assert FEATURE_NAMES[:4] == ["pose_0_x", "pose_0_y", "pose_1_x", "pose_1_y"]
    assert FEATURE_NAMES[66:69] == ["shuttle_x", "shuttle_y", "shuttle_visible"]
    assert features[:, FEATURE_NAMES.index("shuttle_peak_value")].mean() == pytest.approx(0.9)
    assert legacy_features[:, FEATURE_NAMES.index("shuttle_confidence_available")].sum() == 0
    assert features[0, FEATURE_NAMES.index("player1_observed")] == 1
    assert features[0, FEATURE_NAMES.index("player2_observed")] == 0


def test_build_sequences_records_source_metadata() -> None:
    features = np.arange(30, dtype=np.float32).reshape(10, 3)
    labels = np.arange(10, dtype=np.float32)
    sequences, targets, metadata = build_sequences(features, labels, 4, "cam1", range(100, 110))
    assert sequences.shape == (7, 4, 3)
    assert targets.shape == (7, 4)
    assert metadata[0] == {"source_id": "cam1", "start_frame": 100, "end_frame": 103}


def test_source_group_split_keeps_sources_disjoint() -> None:
    metadata = (
        [{"source_id": "a"} for _ in range(3)]
        + [{"source_id": "b"} for _ in range(4)]
        + [{"source_id": "c"} for _ in range(5)]
    )
    train, validation = source_group_split(metadata, validation_fraction=0.34, seed=1)
    train_sources = {metadata[index]["source_id"] for index in train}
    validation_sources = {metadata[index]["source_id"] for index in validation}
    assert train_sources
    assert validation_sources
    assert train_sources.isdisjoint(validation_sources)


def test_decode_probabilities_handles_adjacent_rallies_and_outside_end() -> None:
    config = HeuristicConfig(
        visible_streak=2,
        start_confirmation=3,
        start_buffer=0,
        short_gap=2,
        long_missing_gap=4,
        stopped_window=45,
        end_confirmation=2,
        end_buffer=0,
        minimum_rally=3,
        minimum_visible_frames=2,
        recent_motion_window=2,
        recent_motion_minimum=0.005,
        minimum_motion=0.01,
        smoothing_window=3,
    )
    probabilities = [0.9] * 10 + [0.1] * 8 + [0.9] * 10
    motion = [0.02 if value > 0.5 else 0.0 for value in probabilities]
    rallies = decode_probabilities(
        probabilities, list(range(len(probabilities))), "cam", 30, config, 0.5, motion
    )
    assert len([rally for rally in rallies if rally.status != "rejected"]) == 2

    outside = [None] * 8 + [False] * 5
    outside_rallies = decode_probabilities(
        [0.9] * 13,
        list(range(13)),
        "outside",
        30,
        HeuristicConfig(
            visible_streak=2,
            start_confirmation=3,
            start_buffer=0,
            outside_court_frames=3,
            end_confirmation=2,
            end_buffer=0,
            minimum_rally=3,
            minimum_visible_frames=2,
            minimum_motion=0.01,
            smoothing_window=3,
        ),
        0.5,
        [0.02] * 13,
        outside,
    )
    assert outside_rallies and outside_rallies[0].end_frame < 12


def test_decode_probabilities_scales_frame_windows_for_fps() -> None:
    scaled = scale_config_for_fps(
        HeuristicConfig(visible_streak=10, start_confirmation=20, smoothing_window=7),
        15,
    )
    assert scaled.visible_streak == 5
    assert scaled.start_confirmation == 10
    assert scaled.smoothing_window % 2 == 1


def test_evaluation_reports_frame_f1_split_and_merge_counts() -> None:
    labels = [Interval("s", "l1", 10, 20), Interval("s", "l2", 30, 40)]
    predictions = [
        Interval("s", "p1", 10, 14),
        Interval("s", "p2", 16, 20),
        Interval("s", "p3", 28, 42),
    ]
    metrics, _ = evaluate(predictions, labels, threshold=0.3, frame_ranges={"s": (0, 50)})
    assert metrics["false_split_count"] == 1
    assert "frame_f1" in metrics

    frame_metrics = frame_classification_metrics(
        [Interval("s", "p", 0, 4)], [Interval("s", "l", 2, 6)], {"s": (0, 9)}
    )
    assert frame_metrics["frame_true_positive"] == 3
    assert frame_metrics["frame_false_positive"] == 2
    assert frame_metrics["frame_false_negative"] == 2
