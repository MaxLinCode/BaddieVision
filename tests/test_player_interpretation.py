import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from InPlay.heuristic.person_tracks import import_legacy_player_poses, raw_artifact_fingerprint
from InPlay.heuristic.player_interpretation import (
    assign_singles_slots,
    enrich_pose_cache,
    load_calibration,
    pose_model_fingerprint,
)
from src.court_projection import CourtHomography
from src.pose_estimator import PoseEstimate, PoseLandmark


def _observation(frame: int, track_id: int, x: float, y: float, confidence: float = 0.9):
    return {
        "type": "observation", "frame": frame, "track_id": track_id,
        "bbox": [x - 0.5, y - 1.0, x + 0.5, y], "confidence": confidence,
    }


def _assignment_ids(frames, slot):
    return {
        frame["slots"][slot]["assignment"]["track_id"]
        for frame in frames
        if frame["slots"][slot]["assignment"] is not None
    }


def test_fragmented_ids_stay_on_their_court_roles_and_absences_are_not_filled():
    observations = []
    for frame in range(120):
        if frame < 52:
            observations.append(_observation(frame, 1816, 0, -3))
        if 60 <= frame < 120:
            observations.append(_observation(frame, 2794, 0, -3))
        far_id = 1500 if frame < 40 else 1923 if frame < 80 else 2233
        observations.append(_observation(frame, far_id, 0, 3))
        observations.append(_observation(frame, 99, 10, 3))  # spectator outside court

    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), 120)

    assert _assignment_ids(frames, "P1") <= {1816, 2794}
    assert _assignment_ids(frames, "P2") <= {1500, 1923, 2233}
    assert 99 not in _assignment_ids(frames, "P1") | _assignment_ids(frames, "P2")
    assert all(
        frame["slots"]["P1"]["assignment"] is None
        for frame in frames[52:60]
    )
    for frame in frames:
        assignments = [frame["slots"][slot]["assignment"] for slot in ("P1", "P2")]
        ids = [item["track_id"] for item in assignments if item]
        assert len(ids) == len(set(ids))
        if assignments[0]:
            assert assignments[0]["court_y"] <= 0
        if assignments[1]:
            assert assignments[1]["court_y"] >= 0


def test_hysteresis_retains_recent_incumbent_at_eighty_percent_occupancy():
    observations = []
    for frame in range(90):
        observations.append(_observation(frame, 1, 0, -2, 0.8))
        if 30 <= frame < 90 and frame % 5 != 0:
            observations.append(_observation(frame, 2, 0.2, -2, 0.99))
        observations.append(_observation(frame, 3, 0, 2))

    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), 90)

    assert _assignment_ids(frames, "P1") == {1}


def test_equal_new_same_half_candidates_are_ambiguous():
    observations = []
    for frame in range(60):
        observations.extend([
            _observation(frame, 10, -0.5, -2),
            _observation(frame, 11, 0.5, -2),
            _observation(frame, 20, 0, 2),
        ])

    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), 60)

    assert all(frame["slots"]["P1"]["assignment"] is None for frame in frames)
    assert {frame["slots"]["P1"]["ambiguity_reason"] for frame in frames} == {"ambiguous_candidates"}


def test_calibration_must_exist_contain_image_size_and_match(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="required"):
        load_calibration(missing, (10, 10))
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"version": 1, "image_to_court": np.eye(3).tolist()}))
    with pytest.raises(ValueError, match="missing image_size"):
        load_calibration(invalid, (10, 10))
    invalid.write_text(json.dumps({"version": 1, "image_size": [20, 10], "image_to_court": np.eye(3).tolist()}))
    with pytest.raises(ValueError, match="does not match"):
        load_calibration(invalid, (10, 10))


def _write_video(path: Path, frames: int = 2):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (16, 16))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()


class _FakeEstimator:
    calls = 0

    def estimate_pose(self, _image):
        type(self).calls += 1
        return PoseEstimate([PoseLandmark(0.5, 0.5, 0, 1)] * 33)

    def close(self):
        pass


def test_pose_cache_is_idempotent_and_selection_changes_compute_only_missing(tmp_path: Path):
    video = tmp_path / "video.mp4"
    raw = tmp_path / "person_tracks.jsonl"
    cache = tmp_path / "pose_cache.jsonl"
    _write_video(video)
    raw.write_text(
        json.dumps({"type": "metadata", "schema": "person_tracks", "schema_version": 1,
                    "frame_size": [16, 16], "frame_count": 2, "fps": 25}) + "\n"
        + json.dumps(_observation(0, 1, 4, 10)) + "\n"
        + json.dumps(_observation(1, 2, 8, 10)) + "\n"
    )
    factory = lambda **_kwargs: _FakeEstimator()
    assignment = lambda frame, track_id, bbox: {
        "type": "frame", "frame": frame,
        "slots": {"P1": {"assignment": {"track_id": track_id, "bbox": bbox}},
                  "P2": {"assignment": None}},
    }
    first = [assignment(0, 1, [2, 4, 6, 10])]
    _FakeEstimator.calls = 0
    _, computed = enrich_pose_cache(video, first, raw, cache, estimator_factory=factory)
    assert computed == _FakeEstimator.calls == 1
    _, computed = enrich_pose_cache(video, first, raw, cache, estimator_factory=factory)
    assert computed == 0 and _FakeEstimator.calls == 1
    changed = first + [assignment(1, 2, [6, 4, 10, 10])]
    _, computed = enrich_pose_cache(video, changed, raw, cache, estimator_factory=factory)
    assert computed == 1 and _FakeEstimator.calls == 2


def test_pose_fingerprint_normalizes_relative_and_absolute_model_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    model = tmp_path / "pose.task"
    model.write_bytes(b"same-model")
    monkeypatch.chdir(tmp_path)

    relative = pose_model_fingerprint({"model_asset_path": "pose.task", "package_version": "1"})
    absolute = pose_model_fingerprint({"model_asset_path": str(model), "package_version": "1"})

    assert relative == absolute


def test_legacy_import_preserves_video_shape_and_seeds_attempted_poses(tmp_path: Path):
    video = tmp_path / "video.mp4"
    legacy = tmp_path / "player_poses.jsonl"
    raw = tmp_path / "person_tracks.jsonl"
    cache = tmp_path / "pose_cache.jsonl"
    _write_video(video, frames=2)
    legacy.write_text("\n".join([
        json.dumps({"type": "metadata", "frame_size": [16, 16], "video": str(video), "pose_backend": {}}),
        json.dumps({"type": "frame", "frame": 0, "detections": [
            {"track_id": 1, "bbox": [1, 1, 5, 8], "selected": True, "pose_landmarks": None},
            {"track_id": 9, "bbox": [10, 1, 14, 8], "selected": False, "slot": 2},
        ]}),
        json.dumps({"type": "frame", "frame": 1, "detections": [
            {"track_id": 2, "bbox": [2, 1, 6, 8], "selected": True,
             "pose_landmarks": [{"x": 0.5, "y": 0.5, "z": 0, "visibility": 1}]},
        ]}),
    ]) + "\n")

    observations, seeded = import_legacy_player_poses(legacy, raw, cache, video_path=video)

    assert observations == 3
    assert seeded == 2
    raw_lines = [json.loads(line) for line in raw.read_text().splitlines()]
    assert raw_lines[0]["frame_count"] == 2
    assert raw_lines[0]["fps"] == pytest.approx(25)
    assert {item["track_id"] for item in raw_lines[1:]} == {1, 2, 9}
    assert raw_lines[0].get("selected") is None
    cache_lines = [json.loads(line) for line in cache.read_text().splitlines()]
    assert [item["status"] for item in cache_lines[1:]] == ["no_pose", "detected"]
    assert cache_lines[0]["raw_artifact_fingerprint"] == raw_artifact_fingerprint(raw)
