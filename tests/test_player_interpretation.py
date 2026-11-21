import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from InPlay.heuristic.person_tracks import import_legacy_player_poses, raw_artifact_fingerprint
from InPlay.heuristic import player_interpretation
from InPlay.heuristic.player_interpretation import (
    apply_temporal_pose_validation,
    assign_singles_slots,
    enrich_pose_cache,
    expand_pose_bbox,
    load_calibration,
    load_pose_cache,
    pose_model_fingerprint,
)
from src.court_projection import HALF_LENGTH, HALF_WIDTH, CourtHomography
from src.pose_estimator import PoseEstimate, PoseLandmark
from src import pose_estimator


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


def test_pose_bbox_expands_each_edge_and_clips_to_frame():
    assert expand_pose_bbox([20, 10, 60, 50], (100, 80), 0.20) == [12, 2, 68, 58]
    assert expand_pose_bbox([2, 3, 42, 43], (100, 80), 0.20) == [0, 0, 50, 51]


class _QualityEstimator:
    def __init__(self, visibilities):
        self.visibilities = iter(visibilities)
        self.calls = 0

    def estimate_pose(self, _image):
        self.calls += 1
        visibility = next(self.visibilities)
        return PoseEstimate([PoseLandmark(0.5, 0.5, 0, visibility)] * 33) if visibility is not None else PoseEstimate(None)

    def close(self):
        pass


def test_low_quality_pose_retries_and_retains_better_padded_crop(tmp_path: Path):
    video, raw, cache = tmp_path / "video.mp4", tmp_path / "raw.jsonl", tmp_path / "cache.jsonl"
    _write_video(video, frames=1)
    bbox = [4, 4, 12, 12]
    raw.write_text(json.dumps({"type": "metadata", "schema": "person_tracks", "schema_version": 1,
                               "frame_size": [16, 16], "frame_count": 1, "fps": 25}) + "\n")
    assignments = [{"type": "frame", "frame": 0, "slots": {
        "P1": {"assignment": {"track_id": 1, "bbox": bbox}}, "P2": {"assignment": None}}}]
    estimator = _QualityEstimator([0.1, 0.9])
    active, computed = enrich_pose_cache(video, assignments, raw, cache, estimator_factory=lambda **_: estimator)
    record = active[(0, 1, tuple(bbox))]
    assert computed == 1 and estimator.calls == 2
    assert record["pose_attempts"] == 2 and record["pose_quality"] == pytest.approx(0.9)
    assert record["pose_bbox"] == [2, 2, 14, 14]
    assert record["bbox"] == bbox


def test_temporal_validation_interpolates_one_frame_but_not_long_gap():
    bbox = [0, 0, 100, 100]
    def assignment(frame):
        return {"type": "frame", "frame": frame, "slots": {
            "P1": {"assignment": {"track_id": 1, "bbox": bbox}}, "P2": {"assignment": None}}}
    frames = [assignment(i) for i in range(4)]
    def record(frame, x, landmarks=True):
        points = [{"x": x, "y": 0.5, "z": 0, "visibility": 1}] * 33 if landmarks else None
        return {"frame": frame, "track_id": 1, "bbox": bbox, "pose_bbox": bbox,
                "pose_landmarks": points, "pose_quality": 1 if landmarks else 0,
                "status": "detected" if landmarks else "no_pose", "temporal_status": "original"}
    cache = {(0, 1, tuple(bbox)): record(0, .40), (1, 1, tuple(bbox)): record(1, .95),
             (2, 1, tuple(bbox)): record(2, .42), (3, 1, tuple(bbox)): record(3, .43, False)}
    apply_temporal_pose_validation(frames, cache, (100, 100))
    assert cache[(1, 1, tuple(bbox))]["temporal_status"] == "rejected_jump_interpolated"
    assert cache[(1, 1, tuple(bbox))]["pose_landmarks"][0]["x"] == pytest.approx(.41)
    assert cache[(3, 1, tuple(bbox))]["pose_landmarks"] is None


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
    def factory(**_kwargs):
        return _FakeEstimator()
    def assignment(frame, track_id, bbox):
        return {
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


def _pose_recovery_fixture(tmp_path: Path):
    video, raw = tmp_path / "video.mp4", tmp_path / "raw.jsonl"
    _write_video(video, frames=2)
    raw.write_text(json.dumps({
        "type": "metadata", "schema": "person_tracks", "schema_version": 1,
        "frame_size": [16, 16], "frame_count": 2, "fps": 25,
    }) + "\n")
    assignments = [{"type": "frame", "frame": frame, "slots": {
        "P1": {"assignment": {"track_id": 1, "bbox": [2, 4, 6, 10]}},
        "P2": {"assignment": None},
    }} for frame in range(2)]
    return video, raw, assignments


def test_interrupted_pose_enrichment_reuses_committed_crops_in_a_new_stage(tmp_path: Path):
    video, raw, assignments = _pose_recovery_fixture(tmp_path)
    work = tmp_path / "durable" / "pose-work"

    class InterruptedEstimator(_FakeEstimator):
        def estimate_pose(self, image):
            if self.calls:
                raise KeyboardInterrupt("runtime interrupted")
            return super().estimate_pose(image)

    InterruptedEstimator.calls = 0
    with pytest.raises(KeyboardInterrupt, match="runtime interrupted"):
        enrich_pose_cache(
            video, assignments, raw, tmp_path / "abandoned-stage" / "cache.jsonl",
            estimator_factory=lambda **_: InterruptedEstimator(), pose_work_dir=work,
        )

    _FakeEstimator.calls = 0
    published = tmp_path / "new-stage" / "cache.jsonl"
    cache, computed = enrich_pose_cache(
        video, assignments, raw, published,
        estimator_factory=lambda **_: _FakeEstimator(), pose_work_dir=work,
    )
    assert computed == _FakeEstimator.calls == 1
    assert {key[0] for key in cache} == {0, 1}
    assert len(load_pose_cache(published)[1]) == 2


def test_failed_pose_publication_keeps_previous_cache_and_committed_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    video, raw, assignments = _pose_recovery_fixture(tmp_path)
    published = tmp_path / "cache.jsonl"
    def factory(**_):
        return _FakeEstimator()
    enrich_pose_cache(video, assignments[:1], raw, published, estimator_factory=factory)
    previous = published.read_bytes()
    replace = Path.replace

    def interrupted_replace(path, target):
        if Path(target) == published:
            raise OSError("publication interrupted")
        return replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupted_replace)
        with pytest.raises(OSError, match="publication interrupted"):
            enrich_pose_cache(video, assignments, raw, published, estimator_factory=factory)
    assert published.read_bytes() == previous

    _FakeEstimator.calls = 0
    _, computed = enrich_pose_cache(video, assignments, raw, published, estimator_factory=factory)
    assert computed == _FakeEstimator.calls == 0
    assert len(load_pose_cache(published)[1]) == 2


def test_reused_pose_work_restores_original_estimate_before_temporal_validation(tmp_path: Path):
    video, raw, assignments = _pose_recovery_fixture(tmp_path)
    published = tmp_path / "cache.jsonl"
    def factory(**_):
        return _FakeEstimator()
    enrich_pose_cache(video, assignments, raw, published, estimator_factory=factory)
    metadata, records = load_pose_cache(published)
    records[0]["pose_landmarks"][0]["x"] = 0.99
    records[0]["temporal_status"] = "rejected_jump_interpolated"
    published.write_text("\n".join(json.dumps(value) for value in [metadata, *records]) + "\n")

    _FakeEstimator.calls = 0
    active, computed = enrich_pose_cache(video, assignments, raw, published, estimator_factory=factory)

    assert computed == _FakeEstimator.calls == 0
    restored = active[(0, 1, (2, 4, 6, 10))]
    assert restored["temporal_status"] == "original"
    assert restored["pose_landmarks"][0]["x"] == 0.5


@pytest.mark.parametrize("change", ["backend", "preprocessing", "raw", "bbox"])
def test_pose_recovery_respects_all_observation_and_backend_inputs(tmp_path: Path, change: str):
    video, raw, assignments = _pose_recovery_fixture(tmp_path)
    work = tmp_path / "pose-work"
    options = {"pose_backend_identity": {"name": "test", "version": 1}}
    def factory(**_):
        return _FakeEstimator()
    enrich_pose_cache(
        video, assignments[:1], raw, tmp_path / "first.jsonl",
        estimator_factory=factory, pose_work_dir=work, **options,
    )
    if change == "backend":
        options["pose_backend_identity"] = {"name": "test", "version": 2}
    elif change == "preprocessing":
        options["pose_crop_padding"] = 0.25
    elif change == "raw":
        raw.write_text(raw.read_text() + "\n")
    else:
        assignments[0]["slots"]["P1"]["assignment"]["bbox"][0] += 1
    _FakeEstimator.calls = 0
    _, computed = enrich_pose_cache(
        video, assignments[:1], raw, tmp_path / "second.jsonl",
        estimator_factory=factory, pose_work_dir=work, **options,
    )
    assert computed == _FakeEstimator.calls == 1


def test_pose_cache_is_rebuilt_when_the_raw_person_track_artifact_changes(tmp_path: Path):
    video, raw, cache = tmp_path / "video.mp4", tmp_path / "person_tracks.jsonl", tmp_path / "pose_cache.jsonl"
    _write_video(video, frames=1)
    raw.write_text(json.dumps({"type": "metadata", "schema": "person_tracks", "schema_version": 1,
                               "frame_size": [16, 16], "frame_count": 1, "fps": 25}) + "\n")
    assignments = [{"type": "frame", "frame": 0, "slots": {
        "P1": {"assignment": {"track_id": 1, "bbox": [2, 4, 6, 10]}}, "P2": {"assignment": None},
    }}]
    _FakeEstimator.calls = 0
    def factory(**_kwargs):
        return _FakeEstimator()
    enrich_pose_cache(video, assignments, raw, cache, estimator_factory=factory)
    raw.write_text(raw.read_text() + "\n")  # A rerun replaced the immutable source layer.

    _, computed = enrich_pose_cache(video, assignments, raw, cache, estimator_factory=factory)

    assert computed == 1 and _FakeEstimator.calls == 2
    assert json.loads(cache.read_text().splitlines()[0])["raw_artifact_fingerprint"] == raw_artifact_fingerprint(raw)


def test_legacy_pose_cache_with_matching_model_is_upgraded_without_recomputing(tmp_path: Path):
    video, raw, cache = tmp_path / "video.mp4", tmp_path / "raw.jsonl", tmp_path / "cache.jsonl"
    _write_video(video, frames=1)
    raw.write_text(json.dumps({"type": "metadata", "schema": "person_tracks", "schema_version": 1,
                               "frame_size": [16, 16], "frame_count": 1, "fps": 25}) + "\n")
    assignment = [{"type": "frame", "frame": 0, "slots": {
        "P1": {"assignment": {"track_id": 1, "bbox": [2, 4, 6, 10]}}, "P2": {"assignment": None},
    }}]
    _FakeEstimator.calls = 0
    def factory(**_kwargs):
        return _FakeEstimator()
    enrich_pose_cache(video, assignment, raw, cache, estimator_factory=factory)
    lines = [json.loads(line) for line in cache.read_text().splitlines()]
    lines[0]["schema_version"] = 1
    for record in lines[1:]:
        record.pop("preprocessing_fingerprint")
    cache.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    _, computed = enrich_pose_cache(
        video, assignment, raw, cache, estimator_factory=factory,
        pose_work_dir=tmp_path / "empty-work",
    )

    assert computed == 0 and _FakeEstimator.calls == 1
    upgraded = [json.loads(line) for line in cache.read_text().splitlines()]
    assert upgraded[0]["schema_version"] == 2
    assert upgraded[1]["preprocessing_fingerprint"]


def test_legacy_model_match_without_preprocessing_proof_recomputes(tmp_path: Path):
    video, raw, assignments = _pose_recovery_fixture(tmp_path)
    cache = tmp_path / "cache.jsonl"
    enrich_pose_cache(video, assignments, raw, cache, estimator_factory=lambda **_: _FakeEstimator())
    metadata, records = load_pose_cache(cache)
    metadata["schema_version"] = 1
    metadata.pop("preprocessing_fingerprint")
    for record in records:
        record.pop("preprocessing_fingerprint")
    cache.write_text("\n".join(json.dumps(value) for value in [metadata, *records]) + "\n")

    _FakeEstimator.calls = 0
    _, computed = enrich_pose_cache(
        video, assignments, raw, cache, estimator_factory=lambda **_: _FakeEstimator(),
        pose_work_dir=tmp_path / "empty-work",
    )

    assert computed == _FakeEstimator.calls == 2


@pytest.mark.parametrize("verified_implementation", [True, False])
def test_legacy_path_fingerprint_migrates_only_with_verified_backend_and_preprocessing(
    tmp_path: Path, verified_implementation: bool,
):
    video, raw, assignments = _pose_recovery_fixture(tmp_path)
    model, cache = tmp_path / "pose.task", tmp_path / "cache.jsonl"
    model.write_bytes(b"same-model")
    enrich_pose_cache(
        video, assignments, raw, cache, pose_model_asset=model,
        estimator_factory=lambda **_: _FakeEstimator(),
    )
    metadata, records = load_pose_cache(cache)
    legacy_backend = dict(metadata["pose_backend"])
    if not verified_implementation:
        legacy_backend.pop("implementation_fingerprint")
    legacy_payload = {**legacy_backend, "model_asset_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    legacy_fingerprint = "sha256:" + hashlib.sha256(
        json.dumps(legacy_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata["pose_backend"] = legacy_backend
    metadata["pose_model_fingerprint"] = legacy_fingerprint
    for record in records:
        record["pose_model_fingerprint"] = legacy_fingerprint
    cache.write_text("\n".join(json.dumps(value) for value in [metadata, *records]) + "\n")

    _FakeEstimator.calls = 0
    _, computed = enrich_pose_cache(
        video, assignments, raw, cache, pose_model_asset=model,
        estimator_factory=lambda **_: _FakeEstimator(), pose_work_dir=tmp_path / "empty-work",
    )

    assert computed == _FakeEstimator.calls == (0 if verified_implementation else 2)


def test_interpolated_import_without_raw_journal_recomputes_original_crops(tmp_path: Path):
    video, raw, assignments = _pose_recovery_fixture(tmp_path)
    cache = tmp_path / "cache.jsonl"
    enrich_pose_cache(video, assignments, raw, cache, estimator_factory=lambda **_: _FakeEstimator())
    metadata, records = load_pose_cache(cache)
    records[0]["pose_landmarks"][0]["x"] = 0.99
    records[0]["temporal_status"] = "rejected_jump_interpolated"
    cache.write_text("\n".join(json.dumps(value) for value in [metadata, *records]) + "\n")

    _FakeEstimator.calls = 0
    active, computed = enrich_pose_cache(
        video, assignments, raw, cache, estimator_factory=lambda **_: _FakeEstimator(),
        pose_work_dir=tmp_path / "empty-work",
    )

    assert computed == _FakeEstimator.calls == 1
    assert active[(0, 1, (2, 4, 6, 10))]["pose_landmarks"][0]["x"] == 0.5
    assert len(load_pose_cache(cache)[1]) == 2


@pytest.mark.parametrize("component", ["backend", "crop", "temporal", "assignment", "drawing"])
def test_pose_recovery_tracks_inference_code_without_invalidating_for_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str,
):
    video, raw, assignments = _pose_recovery_fixture(tmp_path)
    work = tmp_path / "work"
    enrich_pose_cache(
        video, assignments, raw, tmp_path / "first.jsonl", pose_work_dir=work,
        estimator_factory=lambda **_: _FakeEstimator(),
    )
    if component == "backend":
        original = pose_estimator.normalize_pose_landmarks

        def normalize_pose_landmarks(landmarks):
            return original(landmarks)

        monkeypatch.setattr(pose_estimator, "normalize_pose_landmarks", normalize_pose_landmarks)
    elif component == "crop":
        original = player_interpretation.expand_pose_bbox

        def expand_pose_bbox(*args, **kwargs):
            box = original(*args, **kwargs)
            box[0] = max(0, box[0] - 1)
            return box

        monkeypatch.setattr(player_interpretation, "expand_pose_bbox", expand_pose_bbox)
    elif component == "temporal":
        monkeypatch.setattr(player_interpretation, "TEMPORAL_JUMP_THRESHOLD", 0.05)
    elif component == "assignment":
        monkeypatch.setattr(player_interpretation, "PLAYER_COURT_PADDING_METERS", 0.5)
    else:
        monkeypatch.setattr(pose_estimator, "POSE_CONNECTIONS", ((0, 1),))

    _FakeEstimator.calls = 0
    _, computed = enrich_pose_cache(
        video, assignments, raw, tmp_path / "second.jsonl", pose_work_dir=work,
        estimator_factory=lambda **_: _FakeEstimator(),
    )

    assert computed == _FakeEstimator.calls == (2 if component in {"backend", "crop"} else 0)


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


@pytest.mark.parametrize("fps", [30, 60])
def test_continuous_slots_hold_then_recover_with_strongest_interior_track(fps):
    observations = [
        _observation(frame, 1, 0.0, -2.0, 0.4)
        for frame in range(2 * fps)
        if not fps // 3 <= frame < fps // 2 and frame < fps
    ]
    observations += [
        _observation(frame, 2, HALF_WIDTH + 0.8, -2.0)
        for frame in range(fps // 4, 2 * fps)
    ]
    observations += [
        _observation(frame, 3, 0.0, -2.0, 0.6)
        for frame in range(fps, 2 * fps)
    ]
    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), 2 * fps, fps)
    near = [frame["slots"]["P1"] for frame in frames]
    assert near[fps // 4]["assignment"]["track_id"] == 1
    assert all(slot["assignment"] is None for slot in near[fps // 3:fps // 2])
    assert near[fps // 2]["decision"] == "returned"
    assert all(slot["assignment"] is None for slot in near[fps:fps + fps // 2])
    assert near[fps + fps // 2]["assignment"]["track_id"] == 3
    assert near[fps + fps // 2]["decision"] == "replacement"
    assert near[fps + fps // 2]["ambiguity_reason"] is None


def test_continuous_slots_allow_lunges_inside_one_meter_and_reject_beyond_it():
    observations = [
        item for frame in range(30)
        for item in (
            _observation(frame, 1, HALF_WIDTH + 0.8, -HALF_LENGTH - 0.9),
            _observation(frame, 2, HALF_WIDTH + 1.01, 2.0),
            _observation(frame, 3, 0.0, HALF_LENGTH + 1.01),
            _observation(frame, 4, 0.0, 2.0),
        )
    ]
    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), 30, 30)
    assert _assignment_ids(frames, "P1") == {1}
    assert _assignment_ids(frames, "P2") == {4}


def test_foot_disagreement_uses_original_pose_crop_and_does_not_clear_assignment():
    observations = [_observation(frame, 1, 0.0, -2.0) for frame in range(12)]
    court = CourtHomography(np.eye(3))
    frames = assign_singles_slots(observations, court, 12, 30)
    landmarks = [{"x": 0.5, "y": 0.75, "visibility": 0.95} for _ in range(33)]
    pose = {
        "pose_landmarks": landmarks,
        "pose_bbox": [-1.0, -3.0, 1.0, 1.0],
    }
    outside = {**pose, "pose_bbox": [5.0, -6.0, 7.0, -2.0]}
    bbox = tuple(observations[0]["bbox"])
    player_interpretation._record_foot_diagnostics(
        frames, {(0, 1, bbox): pose, (1, 1, bbox): outside}, court,
    )
    near = [frame["slots"]["P1"] for frame in frames]
    assert near[0]["foot_diagnostic"]["status"] == "disagrees"
    assert near[1]["foot_diagnostic"]["status"] == "disagrees"
    assert near[1]["foot_diagnostic"]["pose_court"] == pytest.approx([6.0, -3.0])
    assert near[2]["foot_diagnostic"]["status"] == "unavailable"
    assert _assignment_ids(frames, "P1") == {1}


def test_open_role_requires_enough_observations_and_rejects_exact_ties():
    observations = [
        item for frame in range(30)
        for item in (
            _observation(frame, 1, HALF_WIDTH + 0.5 if frame == 0 else 0.0, -2.0, 0.9),
            _observation(frame, 2, HALF_WIDTH + 0.5 if frame == 14 else 1.0, -2.0, 0.9),
        )
        if item["track_id"] == 1 or frame <= 14
    ]
    observations += [_observation(frame, 3, 0.0, 2.0) for frame in range(7)]
    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), 30, 30)
    assert frames[0]["slots"]["P1"]["ambiguity_reason"] == "ambiguous_candidates"
    assert frames[1]["slots"]["P1"]["assignment"]["track_id"] == 1
    assert not _assignment_ids(frames, "P2")


def test_duplicate_raw_id_cannot_be_assigned_to_both_halves():
    observations = [_observation(0, 1, 0.0, y) for y in (-2.0, 2.0)]
    with pytest.raises(ValueError, match="duplicate observation"):
        assign_singles_slots(observations, CourtHomography(np.eye(3)), 30, 30)


def test_assignment_rejects_nonpositive_fps():
    with pytest.raises(ValueError, match="fps must be positive"):
        assign_singles_slots([], CourtHomography(np.eye(3)), 30, 0)
