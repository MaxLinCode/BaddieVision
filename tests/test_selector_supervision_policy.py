from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from src.annotation_platform.events import AnnotationEvent
from src.temporal_selector import dataset as selector_dataset
from src.temporal_selector.batch import MASKED_TARGET, NULL_TARGET
from src.temporal_selector.rally_intervals import (
    RallyInterval,
    RallySourceManifest,
    file_sha256,
    write_rally_intervals,
)


@pytest.fixture
def supervision_config(tmp_path, monkeypatch):
    source = selector_dataset.SelectorSourceConfig(
        source_id="match",
        video_path=tmp_path / "video.mp4",
        candidates_path=tmp_path / "candidates.jsonl",
        assignments_path=tmp_path / "assignments.jsonl",
        pose_cache_path=tmp_path / "poses.jsonl",
        calibration_path=tmp_path / "calibration.json",
    )
    source.video_path.write_bytes(b"video")
    source.candidates_path.write_bytes(b"candidates")
    video_sha256 = file_sha256(source.video_path)
    candidate_sha256 = file_sha256(source.candidates_path)
    intervals = tmp_path / "rallies.csv"
    rally_manifest = tmp_path / "rallies.json"
    write_rally_intervals(
        intervals,
        rally_manifest,
        [RallyInterval("match", "rally", 3, 6)],
        [RallySourceManifest("match", video_sha256, 4.0, 14)],
        annotation_revision="test",
        reviewed_coverage={"match": [[0, 10]]},
    )
    labels = {
        0: "selected",
        1: "no_in_frame_target",
        2: "no_in_frame_target",
        3: "no_in_frame_target",
        4: "selected",
        5: "no_in_frame_target",
        6: "selected",
        7: "missing_proposal",
        8: "occluded_inferable",
        9: "unsure",
        10: "no_shuttle",
        11: "no_in_frame_target",
    }
    events = []
    for frame, kind in labels.items():
        position = [0.7, 0.7] if frame == 6 else [0.3, 0.3]
        events.append(
            AnnotationEvent(
                revision_id=str(frame),
                task="shuttle_selection",
                source_id="match",
                frame=frame,
                label_kind=kind,
                candidate_id=f"raw-{frame}" if kind == "selected" else None,
                candidate_artifact_sha256=candidate_sha256,
                source_video_sha256=video_sha256,
                annotator="test",
                session_id="test",
                timestamp="2026-01-01T00:00:00+00:00",
                superseded_revision=None,
                candidate_position={
                    "coordinate_space": "normalized_image_xy",
                    "canonical_field": "peak_position_normalized",
                    "peak_position_normalized": position,
                    "weighted_centroid_normalized": position,
                    "center_normalized": position,
                } if kind == "selected" else None,
            )
        )
    annotations = tmp_path / "events.jsonl"
    annotations.write_text(
        "".join(json.dumps(event.to_mapping()) + "\n" for event in events)
    )
    frames = {
        frame: (
            {
                "candidate_id": f"retained-{frame}",
                "raw_member_ids": [f"raw-{frame}"],
                "peak_position_normalized": [0.3, 0.3],
                "area_normalized": 0.01,
                "peak_activation_normalized": 0.9,
                "mean_activation_normalized": 0.8,
                "total_activation_normalized": 0.7,
            },
        ) if frame in {0, 1, 3, 4, 6} else ()
        for frame in range(14)
    }
    candidate_meta = {
        "fps": 4.0,
        "image_size": [64, 48],
        "grouping_version": selector_dataset.GROUPING_VERSION,
        "frozen_minimum_cutoff": 0.05,
        "frozen_retention_k": 8,
    }
    monkeypatch.setattr(
        selector_dataset, "read_shuttle_candidates", lambda _: (candidate_meta, list(frames))
    )
    monkeypatch.setattr(
        selector_dataset, "_frozen_candidate_frames", lambda *args, **kwargs: frames
    )
    video_props = {
        selector_dataset.cv2.CAP_PROP_FPS: 4.0,
        selector_dataset.cv2.CAP_PROP_FRAME_COUNT: 14,
        selector_dataset.cv2.CAP_PROP_FRAME_WIDTH: 64,
        selector_dataset.cv2.CAP_PROP_FRAME_HEIGHT: 48,
    }
    monkeypatch.setattr(
        selector_dataset.cv2,
        "VideoCapture",
        lambda _: SimpleNamespace(get=video_props.__getitem__, release=lambda: None),
    )
    assignment_meta = {"fps": 4.0, "frame_size": [64, 48], "frame_count": 14}
    pose_meta = {
        "pose_model_fingerprint": "sha256:pose",
        "preprocessing_fingerprint": "sha256:preprocessing",
    }
    monkeypatch.setattr(
        selector_dataset,
        "_jsonl",
        lambda path: (assignment_meta, [{"frame": frame} for frame in frames])
        if path == source.assignments_path else (pose_meta, []),
    )
    monkeypatch.setattr(selector_dataset, "_validate_calibration", lambda *args: None)
    return selector_dataset.SelectorDataConfig(
        sources=(source,),
        queue_paths=(),
        annotations_path=annotations,
        context_mode="candidates_only",
        rally_intervals_path=intervals,
        rally_manifest_path=rally_manifest,
    )


def test_explicit_targets_cross_rally_boundary_without_inferred_nulls(supervision_config):
    legacy = selector_dataset.SelectorWindowDataset(supervision_config)
    explicit = selector_dataset.SelectorWindowDataset(
        replace(supervision_config, selection_policy="explicit_target_shuttle")
    )

    def owned_targets(dataset):
        return {
            frame: (status, int(window.targets[index]), int(window.inplay_targets[index]))
            for window in dataset.windows
            for index, (frame, status) in enumerate(zip(window.frame_indices, window.target_status))
            if frame in window.owned_frames
        }

    assert owned_targets(legacy)[0] == ("out_of_play", MASKED_TARGET, 0)
    assert owned_targets(legacy)[1] == ("out_of_play", MASKED_TARGET, 0)
    assert owned_targets(explicit) == {
        0: ("selected_retained", 0, 0),
        1: ("null", NULL_TARGET, 0),
        2: ("null", NULL_TARGET, 0),
        3: ("null", NULL_TARGET, 1),
        4: ("selected_retained", 0, 1),
        5: ("null", NULL_TARGET, 1),
        6: ("dropped_by_k", MASKED_TARGET, 1),
        7: ("missing_proposal", MASKED_TARGET, 0),
        8: ("occluded_inferable", MASKED_TARGET, 0),
        9: ("unsure", MASKED_TARGET, 0),
        10: ("no_shuttle", MASKED_TARGET, 0),
        11: ("null", NULL_TARGET, MASKED_TARGET),
        12: ("context", MASKED_TARGET, MASKED_TARGET),
        13: ("context", MASKED_TARGET, MASKED_TARGET),
    }
    assert explicit.manifest["dataset_fingerprint"] != legacy.manifest["dataset_fingerprint"]
    assert explicit.manifest["selection_supervision_counts"]["match"] == {
        "in_play": {
            "with_candidates": {"selected_retained": 1, "null": 1},
            "without_candidates": {"selected_retained": 0, "null": 1},
        },
        "out_of_play": {
            "with_candidates": {"selected_retained": 1, "null": 1},
            "without_candidates": {"selected_retained": 0, "null": 1},
        },
        "unknown": {
            "with_candidates": {"selected_retained": 0, "null": 0},
            "without_candidates": {"selected_retained": 0, "null": 1},
        },
    }
    assert explicit.manifest["target_reason_counts"]["dropped_by_k"] == 1


def test_explicit_policy_checks_annotation_fingerprints_outside_rally(supervision_config):
    path = supervision_config.annotations_path
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[0]["candidate_artifact_sha256"] = "0" * 64
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(ValueError, match="annotation fingerprint mismatch"):
        selector_dataset.SelectorWindowDataset(
            replace(supervision_config, selection_policy="explicit_target_shuttle")
        )
