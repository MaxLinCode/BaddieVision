import json
from pathlib import Path

import pytest

from src.annotation_platform.events import AnnotationEvent
from src.workflow.artifacts import calibration_fingerprint, import_artifact, import_calibration
from src.workflow.coverage import ReviewedSpan, derive_inplay_targets, normalize_reviewed_spans
from src.workflow.identity import ParentVideoProvenance, SourceIdentity
from src.workflow.labels import canonicalize_shuttle_event, match_candidate_by_position
from src.workflow.registry import SourceEntry


def _identity(source_id: str = "source") -> SourceIdentity:
    return SourceIdentity(
        source_id=source_id,
        video_sha256="a" * 64,
        frame_count=100,
        width=1280,
        height=720,
        fps_numerator=30000,
        fps_denominator=1001,
        parent_video=ParentVideoProvenance(
            video_sha256="b" * 64,
            start_time_seconds=30.0,
            end_time_seconds=90.0,
            handling="fps-transcode",
        ),
    )


def test_source_identity_always_uses_source_local_frames() -> None:
    identity = _identity()
    assert identity.frame_range_start == 0
    assert identity.frame_range_end_inclusive == 99
    assert identity.parent_video.start_time_seconds == 30.0
    with pytest.raises(ValueError, match="start at zero"):
        SourceIdentity(**{**identity.__dict__, "frame_range_start": 30})


def test_reviewed_coverage_controls_negative_targets() -> None:
    spans = normalize_reviewed_spans("source", 10, [ReviewedSpan("source", 2, 7)])
    assert derive_inplay_targets(10, spans, [(4, 5)]) == [
        -100, -100, 0, 0, 1, 1, 0, 0, -100, -100,
    ]
    assert normalize_reviewed_spans("source", 3, full_source=True) == (
        ReviewedSpan("source", 0, 2),
    )
    with pytest.raises(ValueError, match="not contained"):
        derive_inplay_targets(10, spans, [(1, 3)])


def test_calibration_import_creates_the_only_canonical_artifact(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    source = SourceEntry("source", video, tmp_path / "artifacts")
    identity = _identity()
    identity.write(source.identity_path)
    external = tmp_path / "legacy.json"
    external.write_text(json.dumps({"image_size": [1280, 720], "homography": [1, 2, 3]}))

    imported = import_calibration(source, external)

    assert imported == source.artifact_root / "court" / "calibration.json"
    value = json.loads(imported.read_text())
    assert value["artifact_metadata"]["source_identity_sha256"] == identity.fingerprint
    assert calibration_fingerprint(source)
    with pytest.raises(FileExistsError):
        import_calibration(source, external)


def test_artifact_import_copies_without_touching_legacy_file(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    source = SourceEntry("source", video, tmp_path / "artifacts")
    identity = _identity()
    identity.write(source.identity_path)
    legacy = tmp_path / "person_tracks.jsonl"
    original = json.dumps({
        "type": "metadata", "schema": "person_tracks", "schema_version": 1,
        "frame_count": 100,
    }) + "\n"
    legacy.write_text(original)

    imported = import_artifact(source, "person-tracks", legacy)

    assert imported == source.artifact_root / "players" / "person_tracks.jsonl"
    assert imported.read_text() == original
    assert legacy.read_text() == original


def _event(label_kind: str, *, position=None) -> AnnotationEvent:
    return AnnotationEvent(
        revision_id="revision", task="shuttle_selection", source_id="source", frame=7,
        label_kind=label_kind, candidate_id="old-id" if label_kind == "selected" else None,
        candidate_artifact_sha256="c" * 64, source_video_sha256="a" * 64,
        annotator="a", session_id="s", timestamp="now", superseded_revision=None,
        candidate_position=position,
    )


def test_visible_label_matches_by_coordinate_not_candidate_id() -> None:
    position = {
        "coordinate_space": "normalized_image_xy",
        "canonical_field": "peak_position_normalized",
        "peak_position_normalized": [0.25, 0.5],
        "weighted_centroid_normalized": [0.25, 0.5],
        "center_normalized": [0.25, 0.5],
    }
    label = canonicalize_shuttle_event(_event("selected", position=position))
    candidates = [
        {"candidate_id": "new-policy-id", "peak_position_normalized": [0.251, 0.499]},
        {"candidate_id": "old-id", "peak_position_normalized": [0.8, 0.8]},
    ]
    assert match_candidate_by_position(label, candidates) == 0


def test_legacy_missing_proposal_is_visible_but_masked() -> None:
    label = canonicalize_shuttle_event(_event("missing_proposal"))
    assert label.visibility == "visible"
    assert label.coordinate_status == "legacy_unavailable"
    assert not label.trains_selector
