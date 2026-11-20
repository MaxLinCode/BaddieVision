"""Tests for the rally error explorer data loader, renderer, and app."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.rally_error_explorer.app import create_app
from src.rally_error_explorer.data import load_explorer_data
from src.rally_error_explorer.render import render_frame_jpeg
from src.temporal_selector.rally_intervals import file_sha256
from src.workflow.identity import SourceIdentity, sha256_file


def _write_source(
    root: Path,
    source_id: str = "synthetic",
    frame_count: int = 3,
) -> Path:
    base = root / source_id
    (base / "players").mkdir(parents=True)
    (base / "court").mkdir(parents=True)
    width, height, fps = 64, 48, 10
    video_path = base / f"{source_id}_input.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened()
    for frame in range(frame_count):
        writer.write(np.full((height, width, 3), frame * 60 + 10, dtype=np.uint8))
    writer.release()
    identity = SourceIdentity(
        source_id=source_id,
        video_sha256=sha256_file(video_path),
        frame_count=frame_count,
        width=width,
        height=height,
        fps_numerator=fps,
        fps_denominator=1,
    )
    identity.write(base / "source.json")
    (base / "court" / "calibration.json").write_text(
        json.dumps(
            {
                "version": 1,
                "image_size": [width, height],
                "image_to_court": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            }
        ),
        encoding="utf-8",
    )
    raw_fingerprint = "sha256:synthetic"
    assignment_lines = [
        json.dumps(
            {
                "type": "metadata",
                "schema": "player_assignments",
                "schema_version": 3,
                "source_id": source_id,
                "raw_artifact_fingerprint": raw_fingerprint,
                "frame_size": [width, height],
                "fps": fps,
                "frame_count": frame_count,
            }
        )
    ]
    for frame in range(frame_count):
        assignment_lines.append(
            json.dumps(
                {
                    "type": "frame",
                    "frame": frame,
                    "slots": {
                        "P1": {
                            "assignment": {
                                "track_id": 1,
                                "bbox": [8, 8, 28, 44],
                                "detection_confidence": 0.9,
                                "court_x": 0.0,
                                "court_y": -1.0,
                                "activity": 0.5,
                                "foot": [18, 44],
                                "pose_status": "detected",
                            },
                            "confidence": 0.9,
                        }
                    },
                }
            )
        )
    (base / "players" / "assignments.jsonl").write_text(
        "\n".join(assignment_lines) + "\n", encoding="utf-8"
    )
    landmarks = [
        {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}
        for _ in range(33)
    ]
    pose_lines = [
        json.dumps(
            {
                "type": "metadata",
                "schema": "pose_cache",
                "schema_version": 2,
                "raw_artifact_fingerprint": raw_fingerprint,
                "pose_model_fingerprint": "sha256:pose",
                "preprocessing_fingerprint": "sha256:pre",
            }
        )
    ]
    for frame in range(frame_count):
        pose_lines.append(
            json.dumps(
                {
                    "type": "pose",
                    "frame": frame,
                    "track_id": 1,
                    "bbox": [8, 8, 28, 44],
                    "pose_bbox": [0, 0, width, height],
                    "status": "detected",
                    "temporal_status": "original",
                    "pose_quality": 1.0,
                    "pose_landmarks": landmarks,
                }
            )
        )
    (base / "players" / "poses.jsonl").write_text(
        "\n".join(pose_lines) + "\n", encoding="utf-8"
    )
    return base


def _write_predictions(path: Path, source_id: str = "synthetic") -> None:
    rows = []
    for frame, probability, target in (
        (0, 0.1, 0),
        (1, 0.9, 1),
        (2, 0.9, 1),
    ):
        rows.append(
            {
                "source_id": source_id,
                "frame": frame,
                "inplay_probability": probability,
                "inplay_logit": 0.0,
                "rally_start_logit": -2.0,
                "rally_end_logit": -2.0,
                "inplay_target": target,
                "decoder_config": {
                    "threshold": 0.5,
                    "max_gap_seconds": 0.0,
                    "minimum_duration_seconds": 0.0,
                    "preserve_edge_runs": False,
                },
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_rallies(base: Path, source_id: str = "synthetic") -> tuple[Path, Path]:
    intervals = base / "rallies.csv"
    intervals.write_text(
        "source_id,rally_id,start_frame,end_frame\n"
        f"{source_id},{source_id}-0001,1,2\n",
        encoding="utf-8",
    )
    manifest = base / "rallies.manifest.json"
    video_path = base / f"{source_id}_input.mp4"
    manifest.write_text(
        json.dumps(
            {
                "schema": "rally_interval_manifest",
                "schema_version": 3,
                "boundary_definition": "serve_contact_through_terminal_event_inclusive",
                "annotation_revision": "test",
                "intervals_sha256": file_sha256(intervals),
                "sources": [
                    {
                        "source_id": source_id,
                        "video_sha256": sha256_file(video_path),
                        "fps": 10.0,
                        "frame_count": 3,
                    }
                ],
                "reviewed_coverage": {source_id: "full_source"},
                "partial_start_rally_ids": [],
            }
        ),
        encoding="utf-8",
    )
    return intervals, manifest


def test_load_explorer_data_and_consistency(tmp_path: Path) -> None:
    base = _write_source(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions)
    rallies, manifest = _write_rallies(base)
    data = load_explorer_data(
        predictions,
        artifact_root=tmp_path,
        rallies_path=rallies,
        manifest_path=manifest,
    )
    series = data.runs["primary"].series["synthetic"]
    assert series.frames == [0, 1, 2]
    assert series.decoded == [False, True, True]
    assert series.start_available and series.end_available
    assert data.consistency["synthetic"]["mismatched_frames"] == 0
    assert data.warnings == []


def test_compare_alignment_failure(tmp_path: Path) -> None:
    _write_source(tmp_path)
    primary = tmp_path / "primary.jsonl"
    _write_predictions(primary)
    compare = tmp_path / "compare.jsonl"
    compare.write_text(
        "".join(
            json.dumps(
                {
                    "source_id": "synthetic",
                    "frame": frame,
                    "inplay_probability": 0.5,
                    "inplay_target": 1,
                }
            )
            + "\n"
            for frame in (1, 2)
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_explorer_data(primary, compare_path=compare, artifact_root=tmp_path)


def test_render_frame_and_app_endpoint(tmp_path: Path) -> None:
    _write_source(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    _write_predictions(predictions)
    data = load_explorer_data(predictions, artifact_root=tmp_path)
    bundle = data.bundles["synthetic"]
    payload = render_frame_jpeg(bundle, 1, show_pose=True, show_court=True)
    assert payload.startswith(b"\xff\xd8")
    app = create_app(data)
    response = app.server.test_client().get(
        "/rally-explorer/frame/synthetic/1.jpg?pose=1&court=0&shuttle=0&run=primary"
    )
    assert response.status_code == 200
    assert response.content_type == "image/jpeg"
    assert response.data.startswith(b"\xff\xd8")
