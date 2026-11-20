import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from InPlay.heuristic.person_tracks import import_legacy_player_poses, raw_artifact_fingerprint
from InPlay.heuristic import player_interpretation
from InPlay.heuristic.player_interpretation import (
    AssignmentPolicy,
    Candidate,
    apply_temporal_pose_validation,
    assign_singles_slots,
    enrich_pose_cache,
    expand_pose_bbox,
    load_calibration,
    load_pose_cache,
    pose_model_fingerprint,
)
from src.court_projection import CourtHomography
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

    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), 120, 30)

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


def test_continuous_raw_id_fragment_handoffs_within_reassignment_interval():
    observations = []
    for frame in range(180):
        if frame < 100:
            observations.append(_observation(frame, 1, 0, -3.35))
        elif frame < 160:
            observations.append(_observation(frame, 2, 0.05, -3.35))
        observations.append(_observation(frame, 10, 0, 3.35))
    raw_snapshot = json.loads(json.dumps(observations))

    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), 180, 30)

    handoffs = [frame["slots"]["P1"] for frame in frames if frame["slots"]["P1"].get("handoff")]
    assert handoffs
    first = handoffs[0]
    assert first["assignment"]["track_id"] == 2
    assert first["ambiguity_reason"] == "track_handoff"
    assert first["handoff"]["prior_track_id"] == 1
    assert first["handoff"]["replacement_track_id"] == 2
    assert all(frame["slots"]["P1"]["assignment"] is not None for frame in frames[100:105])
    assert observations == raw_snapshot


def test_handoff_rejects_tied_or_implausible_current_frame_replacements():
    observations = []
    for frame in range(180):
        if frame < 100:
            observations.append(_observation(frame, 1, 0, -3.35))
        elif frame < 160:
            observations.extend([
                _observation(frame, 2, -0.05, -3.35),
                _observation(frame, 3, 0.05, -3.35),  # tied plausible replacement
                _observation(frame, 4, 10, -3.35),  # outside court spectator
                _observation(frame, 5, 0, 3.35),  # wrong-half player
            ])
        observations.append(_observation(frame, 10, 0, 3.35))

    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), 180, 30)

    assert all(frame["slots"]["P1"].get("handoff") is None for frame in frames)
    assert all(frame["slots"]["P1"]["assignment"] is None for frame in frames[100:105])
    assert _assignment_ids(frames, "P1").isdisjoint({2, 3, 4, 5})


def test_center_weighted_incumbent_is_retained_within_score_tolerance():
    observations = []
    for frame in range(90):
        observations.append(_observation(frame, 1, 0, -2, 0.8))
        if 30 <= frame < 90 and frame % 5 != 0:
            observations.append(_observation(frame, 2, 0.2, -2, 0.99))
        observations.append(_observation(frame, 3, 0, 2))

    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), 90, 30)

    assert _assignment_ids(frames, "P1") == {1}


@pytest.mark.parametrize("challenger_confidence", [0.25, 0.85])
def test_replacement_guard_distinguishes_floor_box_from_reliable_challenger(challenger_confidence):
    observations = [
        _observation(frame, 31, 0 if frame < 30 else 2.5, -3.35, 0.85)
        for frame in range(80)
    ] + [
        _observation(frame, 99, 0, -3.35, challenger_confidence)
        for frame in range(25, 80)
    ]
    court = CourtHomography(np.eye(3))
    baseline = assign_singles_slots(
        observations, court, 80, 10, policy=AssignmentPolicy(replacement_guard=False),
    )
    guarded = assign_singles_slots(observations, court, 80, 10)

    assert baseline[0]["slots"]["P1"]["assignment"]["track_id"] == 31
    assert baseline[45]["slots"]["P1"]["assignment"]["track_id"] == 99
    decision = guarded[45]["slots"]["P1"]
    if challenger_confidence < 0.35:
        assert decision["assignment"]["track_id"] == 31
        assert decision["ambiguity_reason"] == "low_confidence_replacement_guard"
        assert [item["track_id"] for item in decision["candidates"]] == [99, 31]
    else:
        assert decision == baseline[45]["slots"]["P1"]


@pytest.mark.parametrize(
    ("selected_track", "return_frame", "position", "confidence", "other_slot_claim", "reacquired"),
    [
        (None, 5, (0, -3), 0.60, False, True),
        (99, 9, (0, -3), 0.90, False, True),
        (None, 10, (0, -3), 0.90, False, False),
        (None, 5, (0, 3), 0.90, False, False),
        (None, 5, (10, -3), 0.90, False, False),
        (None, 5, (0, -3), 0.59, False, False),
        (None, 5, (0, 0), 0.90, True, False),
        (None, None, (0, -3), 0.90, False, False),
    ],
)
def test_same_id_reacquisition_requires_recent_eligible_current_observation(
    monkeypatch, selected_track, return_frame, position, confidence, other_slot_claim, reacquired,
):
    def rank(_items, _start, end, _occupancy_start, _occupancy_end, slot, _excluded, _fps):
        if slot == 1:
            choice = 31 if end == 25 else selected_track
        else:
            choice = 31 if end > 25 and other_slot_claim else None
        return [] if choice is None else [Candidate(choice, 5, 1.0, 1.0, 0.9, 0.99)]

    monkeypatch.setattr(player_interpretation, "_rank_candidates", rank)
    observations = [_observation(4, 31, 0, -3)]
    if return_frame is not None:
        observations.append(_observation(return_frame, 31, *position, confidence))
    court = CourtHomography(np.eye(3))
    baseline = assign_singles_slots(
        observations, court, 40, 10, policy=AssignmentPolicy(replacement_guard=False),
    )
    frames = assign_singles_slots(
        observations, court, 40, 10,
        policy=AssignmentPolicy(replacement_guard=False, same_id_reacquisition=True),
    )

    assert baseline[4]["slots"]["P1"]["assignment"]["track_id"] == 31
    index = return_frame if return_frame is not None else 5
    assert baseline[index]["slots"]["P1"]["assignment"] is None
    decision = frames[index]["slots"]["P1"]
    if reacquired:
        assert decision["assignment"]["track_id"] == 31
        assert decision["ambiguity_reason"] == "same_id_reacquired"
        assert decision["confidence"] == 0.99
        assert frames[index + 1]["slots"]["P1"]["assignment"] is None
    else:
        assert decision["assignment"] is None
    if other_slot_claim:
        assert frames[index]["slots"]["P2"]["assignment"]["track_id"] == 31


def test_reacquisition_cannot_bypass_confidence_or_replace_the_period_track(monkeypatch):
    def rank(_items, _start, end, _occupancy_start, _occupancy_end, slot, _excluded, _fps):
        if slot == 2:
            return []
        incumbent = Candidate(31, 5, 1.0, 0.5, 0.85, 0.86)
        selected = Candidate(99, 5, 1.0, 1.0, 0.9, 0.99)
        return [incumbent] if end == 25 else [selected, incumbent]

    monkeypatch.setattr(player_interpretation, "_rank_candidates", rank)
    observations = [
        _observation(4, 31, 0, -3),
        _observation(5, 31, 0, -3),
        _observation(6, 31, 0, -3, 0.50),
        _observation(7, 31, 0, -3),
        _observation(7, 99, 0, -3),
    ]
    frames = assign_singles_slots(
        observations, CourtHomography(np.eye(3)), 40, 10,
        policy=AssignmentPolicy(replacement_guard=False, same_id_reacquisition=True),
    )

    assert frames[5]["slots"]["P1"]["ambiguity_reason"] == "same_id_reacquired"
    assert frames[6]["slots"]["P1"]["assignment"] is None
    assert frames[7]["slots"]["P1"]["assignment"]["track_id"] == 99


def test_equal_new_same_half_candidates_are_ambiguous():
    observations = []
    for frame in range(60):
        observations.extend([
            _observation(frame, 10, -0.5, -2),
            _observation(frame, 11, 0.5, -2),
            _observation(frame, 20, 0, 2),
        ])

    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), 60, 30)

    assert all(frame["slots"]["P1"]["assignment"] is None for frame in frames)
    assert {frame["slots"]["P1"]["ambiguity_reason"] for frame in frames} == {"ambiguous_candidates"}


def test_center_access_beats_persistent_edge_spectator():
    observations = []
    for frame in range(180):
        observations.append(_observation(frame, 91, 3.0, -3.35, 0.99))
        if frame % 3 != 0:
            observations.append(_observation(frame, 7, 0.0, -3.35, 0.8))
    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), 180, 30)
    assert _assignment_ids(frames, "P1") == {7}


def test_sideline_incumbent_expires_without_switching_to_edge_spectator():
    observations = []
    for frame in range(300):
        observations.append(_observation(frame, 91, 3.0, -3.35))
        if frame < 75:
            observations.append(_observation(frame, 7, 0.0, -3.35))
        elif frame < 210:
            observations.append(_observation(frame, 7, 3.0, -3.35))
    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), 300, 30)
    assert any(frame["slots"]["P1"]["assignment"] is not None for frame in frames[75:120])
    assert all(frame["slots"]["P1"]["assignment"] is None for frame in frames[180:])
    assert 91 not in _assignment_ids(frames, "P1")


@pytest.mark.parametrize("fps", [20, 50])
def test_assignment_timing_is_fps_aware(fps: int):
    frame_count = 8 * fps
    observations = [
        _observation(frame, 1, 0.0 if frame < 2 * fps else 3.0, -3.35)
        for frame in range(frame_count)
    ]
    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), frame_count, fps)
    assigned_seconds = sum(
        frame["slots"]["P1"]["assignment"] is not None for frame in frames
    ) / fps
    assert assigned_seconds == pytest.approx(4.5, abs=0.51)


def test_wrong_half_and_missing_observations_are_not_assigned_and_scores_are_exposed():
    observations = []
    for frame in range(60):
        observations.append(_observation(frame, 1, 0.0, -3.35))
        observations.append(_observation(frame, 2, 0.0, 3.35))
    frames = assign_singles_slots(observations, CourtHomography(np.eye(3)), 75, 30)

    assert _assignment_ids(frames, "P1") == {1}
    assert _assignment_ids(frames, "P2") == {2}
    assert all(frame["slots"]["P1"]["assignment"] is None for frame in frames[60:])
    candidate = frames[0]["slots"]["P1"]["candidates"][0]
    assert set(candidate) == {
        "track_id", "observations", "center_access", "occupancy_ratio",
        "mean_detection_confidence", "score",
    }


def test_assignment_rejects_nonpositive_fps():
    with pytest.raises(ValueError, match="fps must be positive"):
        assign_singles_slots([], CourtHomography(np.eye(3)), 30, 0)


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
        monkeypatch.setattr(player_interpretation, "COURT_MARGIN_METERS", 0.5)
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
