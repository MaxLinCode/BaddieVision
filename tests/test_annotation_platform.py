from __future__ import annotations

import hashlib
import json
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
    native_fps_burst,
    render_center_frame,
    replay_events,
)
from src.annotation_platform.__main__ import _queues
from src.annotation_platform.dash_app import create_dash_app
from src.annotation_platform.sessions import SessionManager


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


def test_no_shuttle_confirmation_and_v2_unsure_export_filter(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    candidates = _write_candidates(tmp_path / "candidates.jsonl", candidate_count=0)
    registry = AnnotationRegistry()
    registry.register_source("match", video, fps=4, frame_count=12, image_size=(64, 48),
                             artifacts={"candidates": candidates})
    plugin = ShuttleSelectionPlugin()
    registry.register_task(plugin)
    suggestion = plugin.suggestion(registry.sources["match"], 0)
    assert suggestion is not None and suggestion.semantic_label == "no_shuttle"
    store = EventStore(tmp_path / "events.jsonl", registry)
    common = dict(task="shuttle_selection", source_id="match",
                  candidate_artifact_sha256=plugin.artifact_sha256(registry.sources["match"]),
                  annotator="alice", session_id="session")
    confirmed = store.record(**common, frame=0, label_kind="no_shuttle",
                             annotation_suggestion=suggestion)
    store.record(**common, frame=1, label_kind="unsure")
    assert confirmed.review_action == "human_confirmed"
    exported = store.export_current(tmp_path / "export.jsonl")
    records = [json.loads(line) for line in exported.read_text().splitlines()]
    assert records[0]["schema_version"] == 2
    assert [record["label_kind"] for record in records[1:]] == ["no_shuttle"]
    store.export_current(tmp_path / "all.jsonl", include_unsure=True)
    assert "unsure" in (tmp_path / "all.jsonl").read_text()


def test_repeated_undo_walks_back_session_edits_instead_of_toggling(tmp_path: Path) -> None:
    registry, plugin, _, _ = _registry(tmp_path)
    fingerprint = plugin.artifact_sha256(registry.sources["match"])
    store = EventStore(tmp_path / "events.jsonl", registry)
    first = store.record(
        task="shuttle_selection",
        source_id="match",
        frame=1,
        label_kind="no_shuttle",
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
        label_kind="no_shuttle",
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
        label_kind="no_shuttle",
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
            label_kind="no_shuttle",
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
        label_kind="no_shuttle",
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
