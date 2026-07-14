from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.annotation_platform import (
    AnnotationEvent,
    AnnotationRegistry,
    EventStore,
    LabelOption,
    ShuttleSelectionPlugin,
    build_adaptive_queue,
    build_playback_view,
    build_uniform_audit_queue,
    draw_candidates,
    native_fps_burst,
    render_center_frame,
    replay_events,
)
from src.annotation_platform.__main__ import _parser, _print_session_status, _queues
from src.annotation_platform.dash_app import (
    candidate_click_matches_target,
    create_dash_app,
    step_preview_frame,
)
from src.annotation_platform.sessions import SessionConflictError, SessionManager
from src.annotation_platform.views import (
    MARKER_COLORS,
    _intersection_area,
    _layout_candidate_labels,
    candidate_display_items,
)


def _write_candidates(
    path: Path,
    *,
    frame_count: int = 12,
    fps: float = 4.0,
    schema_version: int = 2,
    candidate_count: int = 2,
) -> Path:
    metadata = {
        "type": "metadata",
        "schema": "shuttle_candidates",
        "schema_version": schema_version,
        "fps": fps,
        "image_size": [64, 48],
    }
    if schema_version == 2:
        metadata["source_frame_range"] = [0, frame_count - 1]
        metadata.update({
            "model_stage": "tracknet_pre_inpaint",
            "thresholds": [0.2, 0.3, 0.4, 0.5],
            "threshold_comparator": ">",
            "extraction_version": "tracknet-components-v2.0",
            "legacy_compatibility_threshold": 0.5,
            "checkpoint_sha256": "a" * 64,
            "inference_model_sha256": "b" * 64,
            "provenance_verified": True,
        })
    with path.open("w", encoding="utf-8") as output:
        output.write(json.dumps(metadata) + "\n")
        for frame in range(frame_count):
            candidates = []
            for index in range(candidate_count):
                suffix = chr(ord("a") + index)
                candidates.append(
                    {
                        "candidate_id": f"f{frame}-{suffix}",
                        "center": [5.0 + (index % 6) * 9, 8.0 + (index // 6) * 15],
                        "center_normalized": [
                            (5.0 + (index % 6) * 9) / 64,
                            (8.0 + (index // 6) * 15) / 48,
                        ],
                        "peak_position_normalized": [
                            (4.0 + (index % 6) * 9) / 64,
                            (7.0 + (index // 6) * 15) / 48,
                        ],
                        "weighted_centroid_normalized": [
                            (4.5 + (index % 6) * 9) / 64,
                            (7.5 + (index // 6) * 15) / 48,
                        ],
                        "peak_activation" if schema_version == 2 else "peak_value": (
                            0.9 if index == 0 else 0.3
                        ),
                        "threshold": 0.5,
                        "legacy_largest_component": index == 0,
                    }
                )
            output.write(json.dumps({"type": "frame", "frame": frame, "candidates": candidates}) + "\n")
    return path


def _registry(tmp_path: Path, *, frame_count: int = 12, fps: float = 4.0, schema_version: int = 2):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"registered-video-bytes")
    candidates = _write_candidates(
        tmp_path / "candidates.jsonl",
        frame_count=frame_count,
        fps=fps,
        schema_version=schema_version,
    )
    registry = AnnotationRegistry()
    registry.register_source(
        "match",
        video,
        fps=fps,
        frame_count=frame_count,
        image_size=(64, 48),
        artifacts={"candidates": candidates},
    )
    plugin = ShuttleSelectionPlugin()
    registry.register_task(plugin)
    return registry, plugin, video, candidates


class DummyPlugin:
    task_name = "dummy"
    display_name = "Dummy review task"

    def __init__(self) -> None:
        self.score_calls = 0

    @property
    def label_options(self):
        return (LabelOption("yes", "Yes", "y"),)

    def prepare_source(self, source):
        pass

    def artifact_sha256(self, source):
        return "d" * 64

    def verify_artifact_fingerprint(self, source):
        return self.artifact_sha256(source)

    def eligible_frames(self, source):
        return tuple(range(source.frame_count))

    def image_size(self, source):
        return source.image_size

    def validate_label(self, source, **payload):
        if payload["label_kind"] not in {"yes", "undo"}:
            raise ValueError("bad dummy label")

    def overlays(self, source, frame):
        return ()

    def queue_score(self, source, frame):
        self.score_calls += 1
        return float(frame)


def test_general_registry_accepts_a_second_task_plugin(tmp_path: Path) -> None:
    video = tmp_path / "dummy.mp4"
    video.write_bytes(b"dummy")
    registry = AnnotationRegistry()
    registry.register_source("s", video, fps=2.5, frame_count=8, image_size=(16, 12))
    plugin = DummyPlugin()
    registration = registry.register_task(plugin)

    resolved, source = registry.resolve("dummy", "s")
    assert registration.source_ids == ("s",)
    assert resolved is plugin
    assert source.image_size == (16, 12)
    assert resolved.overlays(source, 3) == ()


@pytest.mark.parametrize("schema_version", [1, 2])
def test_shuttle_plugin_reads_v1_and_v2_candidate_artifacts(tmp_path: Path, schema_version: int) -> None:
    registry, plugin, _, candidates = _registry(tmp_path, schema_version=schema_version)
    source = registry.sources["match"]

    assert plugin.artifact_sha256(source) == hashlib.sha256(candidates.read_bytes()).hexdigest()
    assert [item["candidate_id"] for item in plugin.overlays(source, 2)] == ["f2-a", "f2-b"]
    assert plugin.queue_score(source, 2) == pytest.approx(2.1)


def test_v2_candidates_require_source_image_size_and_contiguous_declared_range(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    candidates = _write_candidates(tmp_path / "candidates.jsonl", frame_count=4)
    registry = AnnotationRegistry()
    registry.register_source(
        "mismatch",
        video,
        fps=4.0,
        frame_count=4,
        image_size=(80, 48),
        artifacts={"candidates": candidates},
    )
    with pytest.raises(ValueError, match="image_size mismatch"):
        registry.register_task(ShuttleSelectionPlugin())

    lines = candidates.read_text(encoding="utf-8").splitlines()
    candidates.write_text("\n".join([*lines[:3], *lines[4:]]) + "\n", encoding="utf-8")
    registry = AnnotationRegistry()
    registry.register_source(
        "gap",
        video,
        fps=4.0,
        frame_count=4,
        image_size=(64, 48),
        artifacts={"candidates": candidates},
    )
    with pytest.raises(ValueError, match="ordered, contiguous"):
        registry.register_task(ShuttleSelectionPlugin())


def test_v1_missing_records_are_ineligible_not_implicit_empty_frames(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    candidates = _write_candidates(
        tmp_path / "candidates.jsonl", frame_count=6, fps=2.0, schema_version=1
    )
    lines = candidates.read_text(encoding="utf-8").splitlines()
    candidates.write_text("\n".join([*lines[:3], *lines[4:]]) + "\n", encoding="utf-8")
    registry = AnnotationRegistry()
    registry.register_source(
        "legacy",
        video,
        fps=2.0,
        frame_count=6,
        image_size=(64, 48),
        artifacts={"candidates": candidates},
    )
    plugin = ShuttleSelectionPlugin()
    registry.register_task(plugin)

    with pytest.raises(ValueError, match="no frame record"):
        plugin.overlays(registry.sources["legacy"], 2)
    audit = build_uniform_audit_queue(
        registry, "shuttle_selection", seed=7, anchor_count=2
    )
    assert ("legacy", 2) not in audit.frame_keys()


def test_select_correction_undo_replay_and_restart(tmp_path: Path) -> None:
    registry, plugin, _, _ = _registry(tmp_path)
    source = registry.sources["match"]
    fingerprint = plugin.artifact_sha256(source)
    store = EventStore(tmp_path / "events.jsonl", registry)
    first = store.record(
        task="shuttle_selection",
        source_id="match",
        frame=2,
        label_kind="selected",
        candidate_id="f2-a",
        candidate_artifact_sha256=fingerprint,
        annotator="alice",
        session_id="session",
    )
    correction = store.record(
        task="shuttle_selection",
        source_id="match",
        frame=2,
        label_kind="selected",
        candidate_id="f2-b",
        candidate_artifact_sha256=fingerprint,
        annotator="alice",
        session_id="session",
    )
    undo = store.undo_last(annotator="alice", session_id="session")

    assert correction.superseded_revision == first.revision_id
    assert undo.superseded_revision == correction.revision_id
    restarted = EventStore(store.path, registry).replay()
    assert restarted.heads[first.key].revision_id == undo.revision_id
    assert restarted.active[first.key].revision_id == first.revision_id
    assert restarted.active[first.key].candidate_id == "f2-a"
    assert first.candidate_position == {
        "coordinate_space": "normalized_image_xy",
        "canonical_field": "peak_position_normalized",
        "peak_position_normalized": [4.0 / 64, 7.0 / 48],
        "weighted_centroid_normalized": [4.5 / 64, 7.5 / 48],
        "center_normalized": [5.0 / 64, 8.0 / 48],
    }
    assert undo.candidate_position is None
    assert "candidate_position" not in undo.to_mapping()


def test_selected_position_validation_rejects_malformed_and_nonselected_values() -> None:
    base = {
        "revision_id": "revision", "task": "shuttle_selection", "source_id": "match",
        "frame": 0, "label_kind": "selected", "candidate_id": "candidate",
        "candidate_artifact_sha256": "a" * 64, "source_video_sha256": "b" * 64,
        "annotator": "alice", "session_id": "session",
        "timestamp": "2026-01-01T00:00:00Z", "superseded_revision": None,
        "candidate_position": {
            "coordinate_space": "normalized_image_xy",
            "canonical_field": "peak_position_normalized",
            "peak_position_normalized": [0.25, 0.5],
            "weighted_centroid_normalized": [0.3, 0.55],
            "center_normalized": [0.35, 0.6],
        },
    }
    malformed = json.loads(json.dumps(base))
    malformed["candidate_position"]["peak_position_normalized"] = [float("nan"), 0.5]
    with pytest.raises(ValueError, match="finite coordinates"):
        AnnotationEvent.from_mapping(malformed)

    nonselected = json.loads(json.dumps(base))
    nonselected.update({"label_kind": "no_in_frame_target", "candidate_id": None})
    with pytest.raises(ValueError, match="only selected"):
        AnnotationEvent.from_mapping(nonselected)


def test_selected_write_rejects_malformed_artifact_coordinates_and_client_snapshot(
    tmp_path: Path,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"registered-video-bytes")
    candidates = _write_candidates(tmp_path / "candidates.jsonl")
    records = [json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines()]
    records[1]["candidates"][0]["peak_position_normalized"] = [1.01, 0.5]
    candidates.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    registry = AnnotationRegistry()
    registry.register_source(
        "match", video, fps=4, frame_count=12, image_size=(64, 48),
        artifacts={"candidates": candidates},
    )
    plugin = ShuttleSelectionPlugin()
    registry.register_task(plugin)
    store = EventStore(tmp_path / "events.jsonl", registry)
    common = dict(
        task="shuttle_selection", source_id="match", frame=0, label_kind="selected",
        candidate_id="f0-a", candidate_artifact_sha256=plugin.artifact_sha256(
            registry.sources["match"]
        ), annotator="alice", session_id="session",
    )
    with pytest.raises(ValueError, match="finite coordinates"):
        store.record(**common)
    with pytest.raises(TypeError, match="candidate_position"):
        store.record(**common, candidate_position={"peak_position_normalized": [0.1, 0.2]})
    assert not store.path.exists()


def test_export_enriches_historical_selection_or_marks_coordinates_unavailable(
    tmp_path: Path,
) -> None:
    registry, plugin, _, _ = _registry(tmp_path)
    source = registry.sources["match"]
    fingerprint = plugin.artifact_sha256(source)
    old = AnnotationEvent(
        revision_id="old", task="shuttle_selection", source_id="match", frame=2,
        label_kind="selected", candidate_id="f2-b",
        candidate_artifact_sha256=fingerprint, source_video_sha256=source.video_sha256,
        annotator="alice", session_id="legacy", timestamp="2025-01-01T00:00:00Z",
        superseded_revision=None,
    )
    exact_store = EventStore(tmp_path / "exact.jsonl", registry)
    exact_store.path.write_text(json.dumps(old.to_mapping()) + "\n", encoding="utf-8")
    exact = exact_store.export_current(tmp_path / "exact-export.jsonl")
    exact_label = json.loads(exact.read_text(encoding="utf-8").splitlines()[1])
    assert exact_label["candidate_position"]["peak_position_normalized"] == [13 / 64, 7 / 48]
    assert exact_store.replay().active[old.key].candidate_position is None

    unavailable = AnnotationEvent(
        **{**asdict(old), "revision_id": "unavailable", "candidate_artifact_sha256": "0" * 64}
    )
    unavailable_store = EventStore(tmp_path / "unavailable.jsonl", registry)
    unavailable_store.path.write_text(
        json.dumps(unavailable.to_mapping()) + "\n", encoding="utf-8"
    )
    exported = unavailable_store.export_current(tmp_path / "unavailable-export.jsonl")
    unavailable_label = json.loads(exported.read_text(encoding="utf-8").splitlines()[1])
    assert unavailable_label["candidate_position"]["coordinates_available"] is False


def test_verified_suggestion_and_semantic_review_actions(tmp_path: Path) -> None:
    registry, plugin, _, _ = _registry(tmp_path)
    source = registry.sources["match"]
    suggestion = plugin.suggestion(source, 2)
    assert suggestion is not None
    assert suggestion.provider == "legacy_tracknet"
    assert suggestion.candidate_id == "f2-a"
    assert suggestion.metadata["opencv_version"] == cv2.__version__
    store = EventStore(tmp_path / "events.jsonl", registry)
    common = dict(
        task="shuttle_selection", source_id="match", frame=2,
        candidate_artifact_sha256=plugin.artifact_sha256(source), annotator="alice",
        session_id="session", annotation_suggestion=suggestion,
    )
    confirmed = store.record(**common, label_kind="selected", candidate_id="f2-a")
    corrected = store.record(**common, label_kind="unsure")
    assert confirmed.review_action == "human_confirmed"
    assert corrected.review_action == "human_corrected"
    assert corrected.annotation_suggestion["candidate_id"] == "f2-a"


def test_suggestions_require_exact_verified_v2_provenance(tmp_path: Path) -> None:
    registry, plugin, _, candidates = _registry(tmp_path)
    assert plugin.suggestion(registry.sources["match"], 0) is not None
    records = candidates.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(records[0])
    metadata["extraction_version"] = "older"
    candidates.write_text(json.dumps(metadata) + "\n" + "\n".join(records[1:]) + "\n")
    second = tmp_path / "second"
    second.mkdir()
    registry, plugin, _, _ = _registry(second, schema_version=1)
    assert plugin.suggestion(registry.sources["match"], 0) is None


def test_no_in_frame_target_confirmation_and_v2_unsure_export_filter(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    candidates = _write_candidates(tmp_path / "candidates.jsonl", candidate_count=0)
    registry = AnnotationRegistry()
    registry.register_source("match", video, fps=4, frame_count=12, image_size=(64, 48),
                             artifacts={"candidates": candidates})
    plugin = ShuttleSelectionPlugin()
    registry.register_task(plugin)
    suggestion = plugin.suggestion(registry.sources["match"], 0)
    assert suggestion is not None and suggestion.semantic_label == "no_in_frame_target"
    store = EventStore(tmp_path / "events.jsonl", registry)
    common = dict(task="shuttle_selection", source_id="match",
                  candidate_artifact_sha256=plugin.artifact_sha256(registry.sources["match"]),
                  annotator="alice", session_id="session")
    confirmed = store.record(**common, frame=0, label_kind="no_in_frame_target",
                             annotation_suggestion=suggestion)
    store.record(**common, frame=1, label_kind="unsure")
    assert confirmed.review_action == "human_confirmed"
    exported = store.export_current(tmp_path / "export.jsonl")
    records = [json.loads(line) for line in exported.read_text().splitlines()]
    assert records[0]["schema_version"] == 2
    assert [record["label_kind"] for record in records[1:]] == ["no_in_frame_target"]
    store.export_current(tmp_path / "all.jsonl", include_unsure=True)
    assert "unsure" in (tmp_path / "all.jsonl").read_text()


def test_legacy_no_shuttle_is_readable_but_cannot_be_written(tmp_path: Path) -> None:
    registry, plugin, _, _ = _registry(tmp_path)
    fingerprint = plugin.artifact_sha256(registry.sources["match"])
    legacy = AnnotationEvent.from_mapping({
        "revision_id": "legacy", "task": "shuttle_selection", "source_id": "match",
        "frame": 0, "label_kind": "no_shuttle", "candidate_id": None,
        "candidate_artifact_sha256": fingerprint, "source_video_sha256": "v" * 64,
        "annotator": "alice", "session_id": "old", "timestamp": "2025-01-01T00:00:00Z",
        "superseded_revision": None,
    })
    assert replay_events([legacy]).active[legacy.key].label_kind == "no_shuttle"
    with pytest.raises(ValueError, match="readable but cannot be written"):
        EventStore(tmp_path / "new.jsonl", registry).record(
            task="shuttle_selection", source_id="match", frame=0,
            label_kind="no_shuttle", candidate_artifact_sha256=fingerprint,
            annotator="alice", session_id="new",
        )


def test_repeated_undo_walks_back_session_edits_instead_of_toggling(tmp_path: Path) -> None:
    registry, plugin, _, _ = _registry(tmp_path)
    fingerprint = plugin.artifact_sha256(registry.sources["match"])
    store = EventStore(tmp_path / "events.jsonl", registry)
    first = store.record(
        task="shuttle_selection",
        source_id="match",
        frame=1,
        label_kind="no_in_frame_target",
        candidate_artifact_sha256=fingerprint,
        annotator="alice",
        session_id="session",
    )
    second = store.record(
        task="shuttle_selection",
        source_id="match",
        frame=2,
        label_kind="missing_proposal",
        candidate_artifact_sha256=fingerprint,
        annotator="alice",
        session_id="session",
    )

    undo_second = store.undo_last(annotator="alice", session_id="session")
    undo_first = store.undo_last(annotator="alice", session_id="session")
    state = store.replay()

    assert undo_second.frame == second.frame
    assert undo_first.frame == first.frame
    assert state.active == {}


def _event(revision: str, frame: int, *, supersedes: str | None = None) -> AnnotationEvent:
    return AnnotationEvent(
        revision_id=revision,
        task="dummy",
        source_id="s",
        frame=frame,
        label_kind="yes",
        candidate_id=None,
        candidate_artifact_sha256="d" * 64,
        source_video_sha256="v" * 64,
        annotator="a",
        session_id="session",
        timestamp="2026-01-01T00:00:00+00:00",
        superseded_revision=supersedes,
    )


def test_replay_rejects_cross_frame_and_already_superseded_revisions() -> None:
    first = _event("first", 1)
    cross_frame = _event("cross", 2, supersedes="first")
    with pytest.raises(ValueError, match="expected None"):
        replay_events((first, cross_frame))

    correction = _event("correction", 1, supersedes="first")
    stale_branch = _event("stale", 1, supersedes="first")
    with pytest.raises(ValueError, match="expected 'correction'"):
        replay_events((first, correction, stale_branch))


def test_valid_but_unterminated_tail_can_be_replayed_recovered_and_resumed(tmp_path: Path) -> None:
    registry, plugin, _, _ = _registry(tmp_path)
    fingerprint = plugin.artifact_sha256(registry.sources["match"])
    store = EventStore(tmp_path / "events.jsonl", registry)
    store.record(
        task="shuttle_selection",
        source_id="match",
        frame=1,
        label_kind="no_in_frame_target",
        candidate_artifact_sha256=fingerprint,
        annotator="alice",
        session_id="session",
    )
    original_prefix = store.path.read_bytes()
    pending = AnnotationEvent(
        revision_id="unterminated-valid-revision",
        task="shuttle_selection",
        source_id="match",
        frame=2,
        label_kind="no_in_frame_target",
        candidate_id=None,
        candidate_artifact_sha256=fingerprint,
        source_video_sha256=registry.sources["match"].video_sha256,
        annotator="alice",
        session_id="session",
        timestamp="2026-01-01T00:00:00+00:00",
        superseded_revision=None,
    )
    with store.path.open("ab") as output:
        output.write(json.dumps(pending.to_mapping()).encode("utf-8"))

    interrupted_bytes = store.path.read_bytes()
    assert store.replay().ignored_interrupted_tail is True
    with pytest.raises(ValueError, match="start a new event segment"):
        store.record(
            task="shuttle_selection",
            source_id="match",
            frame=2,
            label_kind="no_in_frame_target",
            candidate_artifact_sha256=fingerprint,
            annotator="alice",
            session_id="session",
        )

    segment = store.recover_interrupted_tail()
    assert store.path.read_bytes() == interrupted_bytes
    assert original_prefix in interrupted_bytes
    resumed = store.record(
        task="shuttle_selection",
        source_id="match",
        frame=2,
        label_kind="no_in_frame_target",
        candidate_artifact_sha256=fingerprint,
        annotator="alice",
        session_id="session",
    )
    replayed = store.replay()
    assert segment.exists()
    assert len(replayed.events) == 2
    assert replayed.active[resumed.key] == resumed
    assert replayed.recovered_interrupted_tails == 1


def test_label_write_rejects_stale_hash_wrong_frame_id_and_changed_artifact(tmp_path: Path) -> None:
    registry, plugin, _, candidates = _registry(tmp_path)
    source = registry.sources["match"]
    fingerprint = plugin.artifact_sha256(source)
    store = EventStore(tmp_path / "events.jsonl", registry)
    common = {
        "task": "shuttle_selection",
        "source_id": "match",
        "frame": 1,
        "label_kind": "selected",
        "annotator": "alice",
        "session_id": "session",
    }
    with pytest.raises(ValueError, match="SHA-256"):
        store.record(**common, candidate_id="f1-a", candidate_artifact_sha256="0" * 64)
    with pytest.raises(ValueError, match="belongs to frame 2"):
        store.record(**common, candidate_id="f2-a", candidate_artifact_sha256=fingerprint)

    candidates.write_bytes(candidates.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed after registration"):
        store.record(**common, candidate_id="f1-a", candidate_artifact_sha256=fingerprint)
    assert not store.path.exists()


def test_fractional_native_fps_bursts_use_half_up_rounding_and_shift_at_edges() -> None:
    assert len(native_fps_burst(50, fps=29.97, frame_count=100)) == 30
    assert native_fps_burst(5, fps=2.5, frame_count=20) == (4, 5, 6)
    assert native_fps_burst(0, fps=4.0, frame_count=10) == (0, 1, 2, 3)
    assert native_fps_burst(9, fps=4.0, frame_count=10) == (6, 7, 8, 9)


def test_uniform_audit_queue_is_immutable_and_separate_from_adaptive_scores(tmp_path: Path) -> None:
    video = tmp_path / "dummy.mp4"
    video.write_bytes(b"dummy")
    registry = AnnotationRegistry()
    registry.register_source("s", video, fps=2.5, frame_count=30, image_size=(16, 12))
    plugin = DummyPlugin()
    registry.register_task(plugin)
    audit_path = tmp_path / "queues" / "audit.json"

    audit = build_uniform_audit_queue(
        registry, "dummy", seed=19, anchor_count=2, manifest_path=audit_path
    )
    assert len(audit.bursts) == 2
    assert len(audit.frame_keys()) == sum(len(burst.frames) for burst in audit.bursts)
    audit_rows = EventStore(tmp_path / "audit-events.jsonl", registry).audit_view(audit)
    assert len(audit_rows) == len(audit.frame_keys())
    repeated = build_uniform_audit_queue(registry, "dummy", seed=19, anchor_count=2)
    assert repeated.to_mapping() == audit.to_mapping()
    assert plugin.score_calls == 0
    before = audit_path.read_bytes()
    adaptive = build_adaptive_queue(registry, "dummy", anchor_count=3, audit_queue=audit)
    assert plugin.score_calls == 30
    assert adaptive.frame_keys().isdisjoint(audit.frame_keys())
    assert audit_path.read_bytes() == before
    with pytest.raises(FileExistsError, match="immutable queue"):
        build_uniform_audit_queue(
            registry, "dummy", seed=20, anchor_count=2, manifest_path=audit_path
        )
    with pytest.raises(ValueError, match="non-overlapping burst capacity"):
        build_uniform_audit_queue(registry, "dummy", seed=19, anchor_count=11)


def test_headless_view_has_full_context_overlays_and_exact_center_render(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer is unavailable")
    for frame in range(8):
        image = np.full((48, 64, 3), frame * 20, dtype=np.uint8)
        writer.write(image)
    writer.release()
    candidates = _write_candidates(
        tmp_path / "candidates.jsonl",
        frame_count=8,
        fps=4.0,
        candidate_count=12,
    )
    registry = AnnotationRegistry()
    registry.register_source(
        "video",
        video,
        fps=4.0,
        frame_count=8,
        artifacts={"candidates": candidates},
    )
    registry.register_task(ShuttleSelectionPlugin())
    assert registry.sources["video"].image_size == (64, 48)

    view = build_playback_view(
        registry, task="shuttle_selection", source_id="video", center_frame=4
    )
    encoded = render_center_frame(
        registry, task="shuttle_selection", source_id="video", frame=4
    )
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert view.clip_start_seconds == 0.0
    assert view.clip_end_seconds == 2.0
    assert set(view.overlays_by_frame) == set(range(8))
    assert all(len(items) == 12 for items in view.overlays_by_frame.values())
    assert [item["candidate_id"] for item in view.center_candidates] == [
        f"f4-{chr(ord('a') + index)}" for index in range(12)
    ]
    assert decoded.shape[:2] == (48, 64)

    queue = build_uniform_audit_queue(
        registry, "shuttle_selection", seed=2, anchor_count=1
    )
    sessions = SessionManager(tmp_path / "sessions")
    session = sessions.create("alice", queue)
    app = create_dash_app(
        registry,
        queue,
        EventStore(tmp_path / "events.jsonl", registry),
        sessions,
        session,
    )
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    exact = client.get("/annotation/frame/video/4.jpg")
    assert exact.status_code == 200
    assert exact.mimetype == "image/jpeg"
    compact = client.get("/annotation/frame/video/4.jpg?verbose=0")
    verbose = client.get("/annotation/frame/video/4.jpg?verbose=1")
    assert compact.status_code == verbose.status_code == 200
    assert compact.data != verbose.data
    preview = client.get("/annotation/preview/video/3.jpg")
    assert preview.status_code == 200
    assert preview.mimetype == "image/jpeg"
    layout = client.get("/_dash-layout")
    assert b"context-video" not in layout.data
    assert b"candidate-legend" in layout.data
    assert b"Verbose candidate labels" in layout.data
    assert b"Earlier preview [,]" in layout.data
    assert b"preview-center" in layout.data
    assert b"Later preview [.]" in layout.data
    assert "'/':'preview-center'" in app.index_string


def test_candidate_display_items_keep_image_and_control_numbering_aligned() -> None:
    items = candidate_display_items([
        {"candidate_id": "a", "center": [1, 2]},
        {"candidate_id": "missing-center"},
        {"candidate_id": "b", "center": [3, 4]},
    ])
    assert [(item["number"], item["candidate_id"]) for item in items] == [(1, "a"), (2, "b")]
    assert items[0]["color"] != items[1]["color"]


def test_motion_preview_steps_around_fixed_center_and_clamps_to_video() -> None:
    assert step_preview_frame(10, 10, -1, 20) == 9
    assert step_preview_frame(9, 10, -1, 20) == 8
    assert step_preview_frame(8, 10, 0, 20) == 10
    assert step_preview_frame(0, 10, -1, 20) == 0
    assert step_preview_frame(19, 10, 1, 20) == 19


def test_candidate_click_must_belong_to_current_target() -> None:
    current = {
        "type": "candidate",
        "source_id": "video",
        "frame": 328,
        "candidate_id": "candidate",
    }
    stale = {**current, "frame": 327}

    assert candidate_click_matches_target(current, "video", 328)
    assert not candidate_click_matches_target(stale, "video", 328)


def test_candidate_palette_is_deterministic_tab20_and_wraps() -> None:
    items = candidate_display_items([
        {"candidate_id": str(index), "center": [index, index]} for index in range(21)
    ])
    assert len(set(item["color"] for item in items[:20])) == 20
    assert tuple(item["color"] for item in items[:20]) == MARKER_COLORS
    assert items[20]["color"] == items[0]["color"]


def test_candidate_layout_uses_ordered_offsets_and_avoids_markers_and_labels() -> None:
    candidates = [
        {"candidate_id": "a", "center": [50, 50], "peak_activation": 0.9},
        {"candidate_id": "invalid", "center": [float("nan"), 8]},
        {"candidate_id": "b", "center": [100, 50], "peak_value": 0.4},
    ]
    items = candidate_display_items(candidates)
    layout = _layout_candidate_labels(items, (160, 100), False)
    assert [item["label"] for item in layout] == ["#1", "#2"]
    assert layout[0]["offset_index"] == 0
    assert layout[0]["radius"] == 6 and layout[0]["outline"] == 2
    assert layout[1]["radius"] == 4 and layout[1]["outline"] == 1
    assert _intersection_area(layout[0]["label_box"], layout[1]["label_box"]) == 0
    for item in layout:
        left, top, right, bottom = item["label_box"]
        assert 2 <= left < right <= 158
        assert 2 <= top < bottom <= 98


def test_candidate_verbose_label_is_conditional_and_render_does_not_mutate_source() -> None:
    candidates = [
        {"candidate_id": "a", "center": [30, 30], "peak_activation": 0.943, "area": 17.6},
        {"candidate_id": "b", "center": [80, 50], "peak_activation": "bad", "area": None},
    ]
    items = candidate_display_items(candidates)
    compact = _layout_candidate_labels(items, (180, 100), False)
    verbose = _layout_candidate_labels(items, (180, 100), True)
    assert [item["label"] for item in compact] == ["#1", "#2"]
    assert [item["label"] for item in verbose] == ["#1 P=.94 A=18", "#2"]
    source = np.zeros((100, 180, 3), dtype=np.uint8)
    before = source.copy()
    annotated = draw_candidates(source, candidates)
    assert np.array_equal(source, before)
    assert annotated is not source
    assert np.any(annotated != source)
    highlighted = draw_candidates(source, candidates, highlighted_candidate_id="a")
    assert np.any(highlighted != annotated)


def test_dash_app_constructs_from_generic_no_candidate_plugin(tmp_path: Path) -> None:
    video = tmp_path / "dummy.mp4"
    video.write_bytes(b"dummy")
    registry = AnnotationRegistry()
    registry.register_source(
        "dummy",
        video,
        fps=2.0,
        frame_count=6,
        image_size=(16, 12),
    )
    registry.register_task(DummyPlugin())
    queue = build_uniform_audit_queue(registry, "dummy", seed=3, anchor_count=1)
    sessions = SessionManager(tmp_path / "sessions")
    session = sessions.create("alice", queue)

    app = create_dash_app(
        registry,
        queue,
        EventStore(tmp_path / "events.jsonl", registry),
        sessions,
        session,
    )
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    layout = client.get("/_dash-layout")
    assert layout.status_code == 200
    assert b"Dummy review task" in layout.data
    assert b"Yes [Y]" in layout.data


def test_adaptive_suggestion_is_numbered_without_covering_image(tmp_path: Path) -> None:
    registry, _, _, _ = _registry(tmp_path)
    queue = build_adaptive_queue(registry, "shuttle_selection", anchor_count=1)
    sessions = SessionManager(tmp_path / "sessions")
    session = sessions.create("alice", queue)
    app = create_dash_app(
        registry,
        queue,
        EventStore(tmp_path / "events.jsonl", registry),
        sessions,
        session,
    )
    show_target = next(
        callback["callback"].__wrapped__
        for callback in app.callback_map.values()
        if callback["callback"].__wrapped__.__name__ == "show_target"
    )
    result = show_target(asdict(session))
    progress_text = result[2].to_plotly_json()["props"]["children"]
    button_styles = [button.to_plotly_json()["props"]["style"] for button in result[0]]

    assert "suggestion: candidate #1" in progress_text
    assert "Space accepts" in progress_text
    assert "suggestion: f0-a" not in progress_text
    assert button_styles[0]["border"] == "3px solid #ffd54f"
    assert "border" not in button_styles[1]
    layout = app.server.test_client().get("/_dash-layout")
    assert b"suggestion-overlay" not in layout.data
    assert b"Hold O for a clean image" in layout.data
    assert "imageOnlyTimer" in app.index_string
    assert "e.key.toLowerCase() === 'o'" in app.index_string
    assert "e.key.toLowerCase() === 'm'" not in app.index_string


def test_persisted_adaptive_queue_keeps_session_fingerprint_on_restart(tmp_path: Path) -> None:
    registry, _, _, _ = _registry(tmp_path, frame_count=30, fps=2.5)
    runtime = tmp_path / "runtime"
    first, _ = _queues(
        registry,
        runtime,
        audit_seed=11,
        audit_count=2,
        adaptive_count=3,
    )
    sessions = SessionManager(runtime / "sessions")
    created = sessions.create("alice", first)

    restarted, _ = _queues(
        registry,
        runtime,
        audit_seed=11,
        audit_count=2,
        adaptive_count=3,
    )
    loaded = sessions.load(created.session_id)

    assert restarted.queue_id == first.queue_id
    assert sessions.current(loaded, restarted) is not None


def test_legacy_session_loads_with_zero_cursor_revision(tmp_path: Path) -> None:
    registry, _, _, _ = _registry(tmp_path)
    queue = build_uniform_audit_queue(registry, "shuttle_selection", seed=9, anchor_count=1)
    sessions = SessionManager(tmp_path / "sessions")
    created = sessions.create("alice", queue, session_id="legacy")
    path = tmp_path / "sessions" / "legacy.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("cursor_revision")
    raw["audit_note"] = "not a SessionState field"
    raw.pop("audit_note")  # Ensure all original schema fields remain untouched.
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = sessions.load("legacy")

    assert loaded.cursor_revision == 0
    assert loaded.session_id == created.session_id
    assert loaded.started_at == created.started_at


def test_reconcile_recovers_event_saved_before_cursor_advance(tmp_path: Path) -> None:
    registry, _, _, _ = _registry(tmp_path)
    queue = build_uniform_audit_queue(registry, "shuttle_selection", seed=10, anchor_count=1)
    sessions = SessionManager(tmp_path / "sessions")
    state = sessions.create("alice", queue)
    burst, first_frame = sessions.current(state, queue)

    recovered = sessions.reconcile(
        state,
        queue,
        {(burst.task, burst.source_id, first_frame)},
    )

    assert sessions.position(recovered, queue) == 1
    assert recovered.cursor_revision == 1
    assert sessions.load(state.session_id) == recovered


def test_reconcile_counts_active_labels_from_any_annotator(tmp_path: Path) -> None:
    registry, _, _, _ = _registry(tmp_path)
    queue = build_uniform_audit_queue(registry, "shuttle_selection", seed=12, anchor_count=1)
    sessions = SessionManager(tmp_path / "sessions")
    state = sessions.create("alice", queue)
    flattened = [
        (burst.task, burst.source_id, frame)
        for burst in queue.bursts
        for frame in burst.frames
    ]

    state = sessions.reconcile(state, queue, set(flattened[:3]))

    assert sessions.position(state, queue) == 3


def test_auto_selection_uses_reconciled_progress_then_recent_update(tmp_path: Path) -> None:
    registry, _, _, _ = _registry(tmp_path, frame_count=30)
    queue = build_uniform_audit_queue(registry, "shuttle_selection", seed=13, anchor_count=2)
    sessions = SessionManager(tmp_path / "sessions")
    older_advanced = sessions.create("alice", queue, session_id="advanced")
    for _ in range(8):
        older_advanced = sessions.advance(older_advanced, queue)
    recent_start = sessions.create("alice", queue, session_id="recent")
    flattened = [
        (burst.task, burst.source_id, frame)
        for burst in queue.bursts
        for frame in burst.frames
    ]

    selected = sessions.select("alice", queue, set(flattened[:20]))

    assert selected is not None
    assert selected.session_id == recent_start.session_id


def test_stale_cursor_revision_cannot_advance_twice(tmp_path: Path) -> None:
    registry, _, _, _ = _registry(tmp_path)
    queue = build_uniform_audit_queue(registry, "shuttle_selection", seed=14, anchor_count=1)
    sessions = SessionManager(tmp_path / "sessions")
    stale = sessions.create("alice", queue)
    advanced = sessions.advance(stale, queue)

    with pytest.raises(SessionConflictError, match="stale session cursor revision"):
        sessions.advance(stale, queue)

    assert sessions.load(stale.session_id) == advanced


def test_explicit_session_rejects_annotator_and_queue_mismatches(tmp_path: Path) -> None:
    registry, _, _, _ = _registry(tmp_path, frame_count=30)
    first = build_uniform_audit_queue(registry, "shuttle_selection", seed=15, anchor_count=1)
    second = build_uniform_audit_queue(registry, "shuttle_selection", seed=16, anchor_count=1)
    sessions = SessionManager(tmp_path / "sessions")
    state = sessions.create("alice", first)

    with pytest.raises(ValueError, match="annotator mismatch"):
        sessions.load_compatible(state.session_id, "bob", first)
    with pytest.raises(ValueError, match="queue fingerprint"):
        sessions.load_compatible(state.session_id, "alice", second)


def test_page_refresh_loads_latest_durable_cursor(tmp_path: Path) -> None:
    video = tmp_path / "dummy.mp4"
    video.write_bytes(b"dummy")
    registry = AnnotationRegistry()
    registry.register_source("dummy", video, fps=2.0, frame_count=20, image_size=(16, 12))
    registry.register_task(DummyPlugin())
    queue = build_uniform_audit_queue(registry, "dummy", seed=17, anchor_count=1)
    sessions = SessionManager(tmp_path / "sessions")
    initial = sessions.create("alice", queue)
    app = create_dash_app(
        registry,
        queue,
        EventStore(tmp_path / "events.jsonl", registry),
        sessions,
        initial,
    )
    durable = sessions.advance(initial, queue)

    layout = app.server.test_client().get("/_dash-layout").get_json()
    stored = layout["props"]["children"][0]["props"]["data"]

    assert stored == asdict(durable)


def test_serve_session_options_are_mutually_exclusive() -> None:
    parser = _parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--config", "sources.json", "serve", "--annotator", "alice",
            "--session-id", "existing", "--new-session",
        ])


def test_session_status_prints_identity_position_target_and_resume_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry, _, _, _ = _registry(tmp_path)
    queue = build_uniform_audit_queue(registry, "shuttle_selection", seed=18, anchor_count=1)
    sessions = SessionManager(tmp_path / "sessions")
    state = sessions.create("alice", queue, session_id="durable-id")
    target = sessions.current(state, queue)
    assert target is not None

    _print_session_status("auto-resumed session", state, sessions, queue, "python resume-me")
    output = capsys.readouterr().out

    assert "Session: durable-id (auto-resumed session)" in output
    assert "Queue position: 1/" in output
    assert f"Current frame: {target[0].source_id}:{target[1]}" in output
    assert "Resume command: python resume-me" in output


def test_duplicate_dash_submission_records_and_advances_exactly_once(tmp_path: Path) -> None:
    from dash._callback_context import context_value
    from dash._utils import AttributeDict

    registry, _, _, _ = _registry(tmp_path)
    queue = build_adaptive_queue(registry, "shuttle_selection", anchor_count=1)
    sessions = SessionManager(tmp_path / "sessions")
    initial = sessions.create("alice", queue)
    events = EventStore(tmp_path / "events.jsonl", registry)
    app = create_dash_app(registry, queue, events, sessions, initial)
    edit = next(
        callback["callback"].__wrapped__
        for callback in app.callback_map.values()
        if callback["callback"].__wrapped__.__name__ == "edit_target"
    )
    token = context_value.set(AttributeDict(
        triggered_inputs=[{"prop_id": "accept-suggestion.n_clicks", "value": 1}]
    ))
    try:
        first_state, _ = edit([], None, 1, None, None, [], asdict(initial))
        duplicate_state, duplicate_status = edit([], None, 1, None, None, [], asdict(initial))
    finally:
        context_value.reset(token)

    assert len(events.replay().events) == 1
    assert duplicate_state == first_state
    assert "Ignored stale action" in duplicate_status
    assert sessions.position(sessions.load(initial.session_id), queue) == 1


def test_previous_navigation_allows_one_superseding_correction(tmp_path: Path) -> None:
    from dash._callback_context import context_value
    from dash._utils import AttributeDict

    registry, _, _, _ = _registry(tmp_path)
    queue = build_adaptive_queue(registry, "shuttle_selection", anchor_count=1)
    sessions = SessionManager(tmp_path / "sessions")
    initial = sessions.create("alice", queue)
    events = EventStore(tmp_path / "events.jsonl", registry)
    app = create_dash_app(registry, queue, events, sessions, initial)
    edit = next(
        callback["callback"].__wrapped__
        for callback in app.callback_map.values()
        if callback["callback"].__wrapped__.__name__ == "edit_target"
    )

    def invoke(trigger: str, raw: dict[str, object]) -> dict[str, object]:
        token = context_value.set(AttributeDict(
            triggered_inputs=[{"prop_id": f"{trigger}.n_clicks", "value": 1}]
        ))
        try:
            return edit([], None, 1 if trigger == "accept-suggestion" else None,
                        1 if trigger == "previous" else None, None, [], raw)[0]
        finally:
            context_value.reset(token)

    labeled = invoke("accept-suggestion", asdict(initial))
    correction_target = invoke("previous", labeled)
    corrected = invoke("accept-suggestion", correction_target)
    replayed = events.replay()

    assert len(replayed.events) == 2
    assert replayed.events[1].superseded_revision == replayed.events[0].revision_id
    assert corrected["cursor_revision"] > labeled["cursor_revision"]


def test_callback_retry_reconciles_event_saved_before_cursor_write(tmp_path: Path) -> None:
    from dash._callback_context import context_value
    from dash._utils import AttributeDict

    registry, _, _, _ = _registry(tmp_path)
    queue = build_adaptive_queue(registry, "shuttle_selection", anchor_count=1)
    sessions = SessionManager(tmp_path / "sessions")
    initial = sessions.create("alice", queue)
    events = EventStore(tmp_path / "events.jsonl", registry)
    app = create_dash_app(registry, queue, events, sessions, initial)
    burst, frame = sessions.current(initial, queue)
    events.record(
        task=burst.task,
        source_id=burst.source_id,
        frame=frame,
        label_kind="no_in_frame_target",
        candidate_artifact_sha256=burst.candidate_artifact_sha256,
        annotator="alice",
        session_id=initial.session_id,
    )
    edit = next(
        callback["callback"].__wrapped__
        for callback in app.callback_map.values()
        if callback["callback"].__wrapped__.__name__ == "edit_target"
    )
    token = context_value.set(AttributeDict(
        triggered_inputs=[{"prop_id": "accept-suggestion.n_clicks", "value": 1}]
    ))
    try:
        recovered, status = edit([], None, 1, None, None, [], asdict(initial))
    finally:
        context_value.reset(token)

    assert len(events.replay().events) == 1
    assert sessions.position(sessions.load(initial.session_id), queue) == 1
    assert recovered["cursor_revision"] == 1
    assert "Ignored stale action" in status
