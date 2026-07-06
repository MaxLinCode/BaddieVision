from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from InPlay.heuristic.config import HeuristicConfig
from InPlay.heuristic.corrections import finalize, validate_rows
from InPlay.heuristic.court import add_court_signal
from InPlay.heuristic.evaluate import Interval, evaluate, interval_iou
from InPlay.heuristic.players import (
    build_player_rows,
    crop_point_to_image,
    detector_class_name,
    ensure_person_detector,
    select_on_court_tracks,
    write_player_csv,
)
from InPlay.heuristic.segment import CANONICAL_FIELDS, segment_tracks
from InPlay.heuristic.tracks import preprocess_tracks, read_track_csv
from src.court_projection import CourtHomography

FIXTURES = Path(__file__).parent / "fixtures"


def write_tracks(path: Path, rows: list[tuple], peak: bool = False) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Frame", "X", "Y", "Visibility"] + (["PeakValue"] if peak else []))
        writer.writerows(rows)


def moving_rows(count: int, visible_start: int = 10, visible_end: int = 130) -> list[tuple]:
    return [
        (
            frame,
            100 + (frame * 11) % 400 if visible_start <= frame < visible_end else 0,
            80 + (frame * 7) % 240 if visible_start <= frame < visible_end else 0,
            int(visible_start <= frame < visible_end),
        )
        for frame in range(count)
    ]


def test_legacy_schema_and_interpolation_boundaries(tmp_path: Path) -> None:
    rows = moving_rows(50, 2, 45)
    rows[10] = (10, 0, 0, 0)
    rows[11] = (11, 0, 0, 0)
    path = tmp_path / "legacy.csv"
    write_tracks(path, rows)
    frames = preprocess_tracks(read_track_csv(path, (640, 360), HeuristicConfig()), HeuristicConfig())
    assert not frames[0].confidence_available
    assert frames[10].interpolated and frames[11].interpolated
    assert not frames[0].interpolated
    assert frames[9].speed > 0
    assert frames[11].cumulative_distance >= frames[10].cumulative_distance


def test_low_peak_invalid_jump_and_isolated_cleanup(tmp_path: Path) -> None:
    rows = [(i, 100 + i, 100 + i, 1, 0.9) for i in range(12)]
    rows[3] = (3, 103, 103, 1, 0.2)
    rows[7] = (7, 600, 20, 1, 0.9)
    path = tmp_path / "new.csv"
    write_tracks(path, rows, peak=True)
    frames = preprocess_tracks(read_track_csv(path, (640, 360), HeuristicConfig()), HeuristicConfig())
    assert frames[3].removal_reason == "low_peak"
    assert frames[7].removal_reason == "single_frame_jump"
    assert all(item.confidence_available for item in frames)


@pytest.mark.parametrize(
    "rows,error",
    [
        ([(0, 1, 1, 1), (0, 2, 2, 1)], "duplicate"),
        ([(1, 1, 1, 1), (0, 2, 2, 1)], "unordered"),
        ([(0, 1, 1, 1), (2, 2, 2, 1)], "missing frame"),
    ],
)
def test_frame_validation(tmp_path: Path, rows: list[tuple], error: str) -> None:
    path = tmp_path / "bad.csv"
    write_tracks(path, rows)
    with pytest.raises(ValueError, match=error):
        read_track_csv(path, (10, 10), HeuristicConfig())


def test_state_machine_buffer_eof_and_rejection(tmp_path: Path) -> None:
    path = tmp_path / "tracks.csv"
    write_tracks(path, moving_rows(180, 20, 130))
    config = HeuristicConfig()
    frames = preprocess_tracks(read_track_csv(path, (640, 360), config), config)
    rallies = segment_tracks(frames, "camera", 30, config)
    assert len(rallies) == 1
    assert rallies[0].start_frame == 5  # visible start minus 15-frame buffer
    assert rallies[0].end_frame >= 130
    assert rallies[0].status in {"accepted", "review"}
    assert {"CANDIDATE_START", "IN_RALLY", "CANDIDATE_END"} <= {item.state for item in frames}

    short_path = tmp_path / "short.csv"
    write_tracks(short_path, moving_rows(35, 5, 25))
    short = preprocess_tracks(read_track_csv(short_path, (640, 360), config), config)
    rejected = segment_tracks(short, "short", 30, config)
    assert rejected and rejected[0].status == "rejected"
    assert "short_rally" in rejected[0].flags


def test_stationary_false_detection_and_outside_court_end(tmp_path: Path) -> None:
    stationary_path = tmp_path / "stationary.csv"
    write_tracks(stationary_path, [(i, 200, 150, 1) for i in range(150)])
    config = HeuristicConfig()
    stationary = preprocess_tracks(
        read_track_csv(stationary_path, (640, 360), config), config
    )
    result = segment_tracks(stationary, "stationary", 30, config)
    assert result[0].status == "rejected"
    assert "insufficient_motion" in result[0].failure_reason

    moving_path = tmp_path / "outside.csv"
    write_tracks(moving_path, moving_rows(120, 0, 120))
    outside_config = HeuristicConfig(
        outside_court_frames=10, end_confirmation=5, minimum_rally=20
    )
    moving = preprocess_tracks(
        read_track_csv(moving_path, (640, 360), outside_config), outside_config
    )
    for item in moving:
        item.inside_courtish = False
    outside = segment_tracks(moving, "outside", 30, outside_config)
    assert outside and outside[0].end_frame < moving[-1].frame


def test_non_30_fps_and_optional_player_cannot_start(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    write_tracks(path, moving_rows(100, 200, 200))
    config = HeuristicConfig()
    frames = preprocess_tracks(read_track_csv(path, (640, 360), config), config)
    assert segment_tracks(frames, "none", 24, config, {i: 1.0 for i in range(100)}) == []

    moving = tmp_path / "moving.csv"
    write_tracks(moving, moving_rows(150, 5, 100))
    features = preprocess_tracks(read_track_csv(moving, (640, 360), config), config)
    result = segment_tracks(features, "fps", 24, config)
    assert "non_30_fps" in result[0].flags


def test_adjacent_rallies_and_court_registry(tmp_path: Path) -> None:
    rows = moving_rows(300, 10, 80)
    for frame in range(180, 250):
        rows[frame] = (frame, 100 + (frame * 11) % 400, 80 + (frame * 7) % 240, 1)
    path = tmp_path / "two.csv"
    write_tracks(path, rows)
    config = HeuristicConfig(end_confirmation=10, long_missing_gap=40)
    frames = preprocess_tracks(read_track_csv(path, (640, 360), config), config)
    rallies = segment_tracks(frames, "cam", 30, config)
    assert len(rallies) == 2

    calibration = tmp_path / "cam.json"
    calibration.write_text(
        json.dumps(
            {
                "version": 1,
                "image_size": [640, 360],
                "image_landmarks": {
                    "a": [80, 40], "b": [560, 40], "c": [560, 350], "d": [80, 350]
                },
                "image_to_court": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        )
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {"version": 1, "sources": {"cam": {"calibration": "cam.json"}}}
        )
    )
    assert add_court_signal(frames, registry, 0.2, "cam") is None
    assert any(item.inside_courtish is not None for item in frames)
    assert add_court_signal(frames, registry, 0.2, "missing") == "court_projection_unstable"


def test_evaluation_uses_inclusive_iou_and_one_to_one() -> None:
    prediction = Interval("s", "p", 10, 20)
    label = Interval("s", "l", 10, 20)
    assert interval_iou(prediction, label) == 1
    metrics, details = evaluate(
        [prediction, Interval("s", "p2", 10, 20)], [label], 0.5
    )
    assert metrics["matched_count"] == 1
    assert metrics["precision"] == 0.5
    assert any(item["match_status"] == "unmatched_prediction" for item in details)


def test_correction_validation_and_finalize() -> None:
    base = {
        "source_id": "s", "rally_id": "r1", "start_frame": "10", "end_frame": "50",
        "start_time": "0.3", "end_time": "1.7", "status": "review", "confidence": "0.6",
        "confidence_band": "medium", "flags": "manual_review_needed", "failure_reason": "",
        "manual_start_frame": "12", "manual_end_frame": "48", "manual_decision": "accept",
    }
    assert not validate_rows([base], 0, 100)
    result = finalize([base], 30)
    assert result[0]["start_frame"] == "12"
    assert result[0]["status"] == "accepted"
    overlap = dict(base, rally_id="r2", manual_start_frame="40", manual_end_frame="60")
    assert any("overlapping" in error for error in validate_rows([base, overlap]))


def test_player_geometry_and_spectator_rejection() -> None:
    assert crop_point_to_image((0.5, 0.25), (100, 20, 300, 220)) == (200, 70)
    observations = {
        1: [(i, (100, 100, 200, 500)) for i in range(20)],
        2: [(i, (500, 100, 600, 450)) for i in range(18)],
        9: [(i, (0, 0, 20, 30)) for i in range(100)],
        8: [(0, (300, 100, 350, 300))],
    }
    assert set(select_on_court_tracks(observations, (800, 600))) == {1, 2}


def test_player_selection_uses_homography_when_available() -> None:
    calibration = CourtHomography(np.eye(3))
    observations = {
        1: [(i, (-0.5, -2.0, 0.5, -1.0)) for i in range(20)],
        2: [(i, (1.5, 1.0, 2.5, 2.0)) for i in range(18)],
        9: [(i, (7.5, 0.0, 8.5, 1.0)) for i in range(100)],
    }
    assert set(select_on_court_tracks(observations, (800, 600), calibration)) == {1, 2}


def test_player_detector_must_map_class_zero_to_person() -> None:
    class Detector:
        def __init__(self, names: object) -> None:
            self.names = names

    assert detector_class_name(Detector({0: "person"})) == "person"
    assert detector_class_name(Detector(["person", "bicycle"])) == "person"
    ensure_person_detector(Detector({0: "person"}), "yolov8n.pt")
    with pytest.raises(ValueError, match="not a COCO person detector"):
        ensure_person_detector(Detector({0: "shuttle"}), "best.pt")


def test_raw_player_artifact_builds_heuristic_csv(tmp_path: Path) -> None:
    raw_data = {
        "schema_version": 1,
        "frame_size": [640, 360],
        "selected_track_ids": [11, 22],
        "frames": [
            {
                "frame": 0,
                "detections": [
                    {
                        "track_id": 11,
                        "selected": True,
                        "crop_valid": True,
                        "foot": [100.0, 200.0],
                        "activity": 0.02,
                        "pose_landmarks": [{"x": 0.1, "y": 0.2, "z": 0.0, "visibility": 1.0}],
                    },
                    {"track_id": 99, "selected": False, "crop_valid": False, "pose_landmarks": None},
                ],
            },
            {
                "frame": 1,
                "detections": [
                    {
                        "track_id": 11,
                        "selected": True,
                        "crop_valid": True,
                        "foot": [104.0, 201.0],
                        "activity": 0.01,
                        "pose_landmarks": [{"x": 0.11, "y": 0.2, "z": 0.0, "visibility": 1.0}],
                    },
                    {
                        "track_id": 22,
                        "selected": True,
                        "crop_valid": True,
                        "foot": [504.0, 210.0],
                        "activity": 0.03,
                        "pose_landmarks": [{"x": 0.6, "y": 0.3, "z": 0.0, "visibility": 1.0}],
                    },
                ],
            },
        ],
    }
    rows = build_player_rows(raw_data)
    assert rows[0]["Frame"] == 0
    assert rows[0]["player1_track_id"] == 11
    assert rows[0]["player2_track_id"] == 22
    assert rows[0]["player2_valid"] == 0
    assert rows[0]["player_activity"] == pytest.approx(0.4)
    assert rows[1]["player_activity"] == pytest.approx(0.8)
    assert rows[1]["activity_window"] == pytest.approx(0.6)

    output = tmp_path / "players.csv"
    write_player_csv(rows, output)
    saved = list(csv.DictReader(output.open()))
    assert saved[0]["Frame"] == "0"
    assert saved[1]["player2_track_id"] == "22"


def test_cli_segment_evaluate_and_corrections(tmp_path: Path) -> None:
    output = tmp_path / "rallies.csv"
    debug = tmp_path / "debug.csv"
    command = [
        sys.executable, "-m", "InPlay.heuristic.segment",
        "--tracks", str(FIXTURES / "heuristic_tracks.csv"),
        "--fps", "30", "--image-size", "640", "360", "--source-id", "fixture",
        "--output", str(output), "--debug-frames", str(debug),
        "--minimum-rally", "20", "--end-confirmation", "5", "--long-missing-gap", "5",
    ]
    subprocess.run(command, check=True)
    rows = list(csv.DictReader(output.open()))
    assert rows and list(rows[0]) == CANONICAL_FIELDS
    assert len(list(csv.DictReader(debug.open()))) == 54

    metrics = tmp_path / "metrics.json"
    matches = tmp_path / "matches.csv"
    subprocess.run(
        [
            sys.executable, "-m", "InPlay.heuristic.evaluate",
            "--predictions", str(output), "--labels", str(FIXTURES / "heuristic_labels.csv"),
            "--metrics", str(metrics), "--matches", str(matches),
        ],
        check=True,
    )
    assert "f1" in json.loads(metrics.read_text())
    subprocess.run(
        [sys.executable, "-m", "InPlay.heuristic.validate_corrections", "--input", str(output)],
        check=True,
    )
    final = tmp_path / "final.csv"
    subprocess.run(
        [
            sys.executable, "-m", "InPlay.heuristic.finalize",
            "--input", str(output), "--output", str(final), "--fps", "30",
        ],
        check=True,
    )
    assert final.exists()


def test_tracknet_peak_export_retains_one_row_per_frame(tmp_path: Path) -> None:
    script = r"""
import torch
from predict import predict
from utils.general import write_pred_csv
indices = torch.tensor([[[0, 0], [0, 1]]])
heat = torch.zeros((1, 2, 288, 512))
heat[0, 0, 10, 20] = 0.9
heat[0, 1, 20, 30] = 0.8
result = predict(indices, y_pred=heat)
assert len(result["Frame"]) == len(result["PeakValue"]) == 2
write_pred_csv(result, r"%s")
""" % str(tmp_path / "tracknet.csv")
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1] / "src" / "TrackNetV3",
        check=True,
    )
    rows = list(csv.DictReader((tmp_path / "tracknet.csv").open()))
    assert len(rows) == 2
    assert [round(float(row["PeakValue"]), 1) for row in rows] == [0.9, 0.8]
