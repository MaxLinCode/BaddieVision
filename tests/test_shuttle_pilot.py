from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.annotation_platform.pilot import (
    evaluate_threshold_pilot,
    filter_pilot_artifact,
    freeze_threshold_policy,
    migrate_v1_runtime,
    validate_artifact_lineage,
    wilson_interval,
)
from src.annotation_platform import AnnotationRegistry, EventStore
from src.annotation_platform.queues import AnnotationQueue, QueueBurst
from src.annotation_platform.shuttle import (
    GROUPING_VERSION,
    ShuttleSelectionPlugin,
    group_shuttle_candidates,
    selector_training_target,
)


def _candidate(
    candidate_id: str,
    threshold: float,
    bbox: list[float],
    peak: list[float],
    centroid: list[float] | None = None,
    activation: float | None = None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "threshold": threshold,
        "bbox": bbox,
        "center": peak,
        "peak_position": peak,
        "weighted_centroid": centroid or peak,
        "peak_activation": activation if activation is not None else threshold,
        "mean_activation": threshold,
        "area_normalized": 0.01,
    }


def _artifact(path: Path, frames: dict[int, list[dict]], thresholds=(0.1, 0.5)) -> Path:
    metadata = {
        "type": "metadata",
        "schema": "shuttle_candidates",
        "schema_version": 2,
        "thresholds": list(thresholds),
        "fps": 30.0,
        "image_size": [100, 100],
        "heatmap_size": [10, 10],
        "source_frame_range": [min(frames), max(frames)],
        "source_frame_index_space": "zero_based_working_video",
        "checkpoint_sha256": "a" * 64,
        "inference_model_sha256": "b" * 64,
        "model_stage": "tracknet_pre_inpaint",
        "extraction_version": "tracknet-components-v2.0",
        "threshold_comparator": ">",
        "tracknet_config": {"sequence_length": 8},
        "overlap_ensemble_mode": "weight",
    }
    with path.open("w", encoding="utf-8") as output:
        output.write(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n")
        for frame in range(min(frames), max(frames) + 1):
            output.write(json.dumps(
                {"type": "frame", "frame": frame, "candidates": frames.get(frame, [])},
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n")
    return path


def test_high_to_low_grouping_prevents_bridges_and_duplicate_threshold_members() -> None:
    raw = [
        _candidate("high-a", 0.5, [0, 0, 3, 3], [1, 1]),
        _candidate("high-b", 0.5, [8, 0, 11, 3], [9, 1]),
        # Both peaks lie in this lower component. It must choose only high-a
        # by centroid distance and may not bridge the two groups.
        _candidate("low-bridge", 0.1, [0, 0, 11, 3], [5, 1], [2, 1]),
        # Same threshold cannot join high-a after low-bridge already did.
        _candidate("low-a2", 0.1, [0, 0, 3, 3], [1, 1]),
    ]
    groups = group_shuttle_candidates(raw)

    flattened = [candidate_id for group in groups for candidate_id in group["raw_member_ids"]]
    assert sorted(flattened) == sorted(item["candidate_id"] for item in raw)
    assert len(flattened) == len(set(flattened))
    assert next(group for group in groups if group["candidate_id"] == "high-a")["raw_member_ids"] == [
        "high-a", "low-a2"
    ]
    assert next(group for group in groups if group["candidate_id"] == "high-b")["raw_member_ids"] == [
        "high-b", "low-bridge"
    ]
    assert all(
        len({next(item["threshold"] for item in raw if item["candidate_id"] == member)
             for member in group["raw_member_ids"]}) == len(group["raw_member_ids"])
        for group in groups
    )
    assert all(group["grouping_version"] == GROUPING_VERSION for group in groups)


def test_observation_contract_selector_mapping_and_legacy_write_rejection(tmp_path: Path) -> None:
    assert selector_training_target("selected") == "SELECTED_PROPOSAL"
    assert selector_training_target("no_in_frame_target") == "NO_SHUTTLE"
    assert selector_training_target("missing_proposal") is None
    assert selector_training_target("occluded_inferable") is None
    assert selector_training_target("no_shuttle") is None
    assert [(item.kind, item.hotkey) for item in ShuttleSelectionPlugin().label_options] == [
        ("occluded_inferable", "i"),
        ("no_in_frame_target", "n"),
        ("missing_proposal", "m"),
        ("unsure", "u"),
    ]


def test_pilot_metrics_freeze_and_filter_keep_stable_raw_ids(tmp_path: Path) -> None:
    frames = {
        0: [
            _candidate("f0-high", 0.5, [0, 0, 4, 4], [1, 1], activation=0.9),
            _candidate("f0-low", 0.1, [0, 0, 5, 5], [1, 1], activation=0.9),
        ],
        1: [_candidate("f1-low", 0.1, [5, 5, 8, 8], [6, 6], activation=0.2)],
        2: [],
        3: [],
    }
    pilot = _artifact(tmp_path / "pilot.jsonl", frames)
    labels = [
        {"frame": 0, "label_kind": "selected", "candidate_id": "f0-high"},
        {"frame": 1, "label_kind": "selected", "candidate_id": "f1-low"},
        {"frame": 2, "label_kind": "missing_proposal"},
        {"frame": 3, "label_kind": "occluded_inferable"},
    ]
    report = evaluate_threshold_pilot(pilot, labels, cutoffs=(0.1, 0.5), ks=(1, 2, None))
    low, high = report["threshold_results"]
    assert low["observed_target_frames"] == 3
    assert low["observed_hits"] == 2
    assert low["observed_proposal_recall"] == pytest.approx(2 / 3)
    assert high["observed_hits"] == 1
    assert high["wilson_95"] == wilson_interval(1, 3)
    assert low["occluded_inferable_frames"] == 1
    assert low["raw_candidates_per_frame"]["maximum"] == 2
    assert low["grouped_candidates_per_frame"]["maximum"] == 1

    freeze = freeze_threshold_policy(report, target_recall=0.99)
    assert freeze["minimum_cutoff"] == 0.1
    assert freeze["selection_reason"] == "pareto_best_recall_ceiling"
    assert freeze["retention_k"] == 1
    production = filter_pilot_artifact(pilot, tmp_path / "production.jsonl", freeze)
    metadata, *records = [json.loads(line) for line in production.read_text().splitlines()]
    assert metadata["parent_artifact_sha256"] == hashlib.sha256(pilot.read_bytes()).hexdigest()
    assert metadata["grouping_version"] == GROUPING_VERSION
    assert [item["candidate_id"] for item in records[0]["candidates"]] == ["f0-high", "f0-low"]
    assert filter_pilot_artifact(pilot, production, freeze) == production


def test_lineage_validation_rejects_shared_threshold_record_changes(tmp_path: Path) -> None:
    old = _artifact(
        tmp_path / "old.jsonl",
        {0: [_candidate("shared", 0.5, [0, 0, 2, 2], [1, 1])]},
        thresholds=(0.5,),
    )
    pilot = _artifact(
        tmp_path / "pilot.jsonl",
        {0: [
            _candidate("low", 0.1, [0, 0, 2, 2], [1, 1]),
            _candidate("shared", 0.5, [0, 0, 2, 2], [1, 1]),
        ]},
    )
    validate_artifact_lineage(old, pilot)
    records = pilot.read_text().splitlines()
    changed = json.loads(records[1])
    changed["candidates"][1]["center"] = [9, 9]
    records[1] = json.dumps(changed, sort_keys=True, separators=(",", ":"))
    pilot.write_text("\n".join(records) + "\n")
    with pytest.raises(ValueError, match="shared-threshold candidate records differ"):
        validate_artifact_lineage(old, pilot)


def test_freeze_uses_highest_qualifying_cutoff_and_smallest_equivalent_k() -> None:
    report = {
        "threshold_results": [
            {
                "minimum_cutoff": 0.05,
                "observed_proposal_recall": 1.0,
                "observed_target_frames": 100,
                "wilson_95": [0.96, 1.0],
                "hits_at_k": {"1": 98, "2": 100, "all": 100},
                "grouped_candidates_per_frame": {"p50": 8},
            },
            {
                "minimum_cutoff": 0.10,
                "observed_proposal_recall": 0.99,
                "observed_target_frames": 100,
                "wilson_95": [0.94, 1.0],
                "hits_at_k": {"1": 98, "2": 99, "all": 99},
                "grouped_candidates_per_frame": {"p50": 4},
            },
            {
                "minimum_cutoff": 0.15,
                "observed_proposal_recall": 0.98,
                "observed_target_frames": 100,
                "wilson_95": [0.92, 1.0],
                "hits_at_k": {"1": 98, "2": 98, "all": 98},
                "grouped_candidates_per_frame": {"p50": 2},
            },
        ]
    }
    frozen = freeze_threshold_policy(report)
    assert frozen["minimum_cutoff"] == 0.10
    assert frozen["retention_k"] == 2
    assert frozen["selection_reason"] == "highest_cutoff_reaching_target_point_recall"


def test_v1_migration_is_transactional_preserves_queue_order_and_classifies_labels(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"immutable-video")
    old_candidates = _artifact(tmp_path / "old.jsonl", {
        frame: [_candidate(f"f{frame}-shared", 0.5, [0, 0, 2, 2], [1, 1])]
        for frame in range(3)
    }, thresholds=(0.5,))
    pilot_candidates = _artifact(tmp_path / "pilot.jsonl", {
        frame: [
            _candidate(f"f{frame}-low", 0.1, [0, 0, 2, 2], [1, 1]),
            _candidate(f"f{frame}-shared", 0.5, [0, 0, 2, 2], [1, 1]),
        ]
        for frame in range(3)
    })

    def registry(path: Path) -> AnnotationRegistry:
        value = AnnotationRegistry()
        value.register_source(
            "source", video, fps=30, frame_count=3, image_size=(100, 100),
            artifacts={"candidates": path},
        )
        value.register_task(ShuttleSelectionPlugin())
        return value

    old_registry = registry(old_candidates)
    pilot_registry = registry(pilot_candidates)
    v1 = tmp_path / "v1"
    store = EventStore(v1 / "events" / "shuttle.jsonl", old_registry)
    fingerprint = hashlib.sha256(old_candidates.read_bytes()).hexdigest()
    common = {
        "task": "shuttle_selection", "source_id": "source",
        "candidate_artifact_sha256": fingerprint, "annotator": "alice",
        "session_id": "v1-session",
    }
    store.record(**common, frame=0, label_kind="selected", candidate_id="f0-shared")
    store.record(**common, frame=1, label_kind="selected", candidate_id="f1-shared")
    store.record(**common, frame=2, label_kind="missing_proposal")
    source = old_registry.sources["source"]
    burst = QueueBurst(
        burst_id="adaptive-source-0", task="shuttle_selection", source_id="source",
        anchor_frame=0, frames=(2, 0, 1), candidate_artifact_sha256=fingerprint,
        source_video_sha256=source.video_sha256, score=1.0,
    )
    queue = AnnotationQueue(
        queue_id="v1-queue", kind="adaptive", task="shuttle_selection", seed=None,
        bursts=(burst,), construction={
            "requested_anchor_count": 1,
            "audit_queue_id": "v1-audit-queue",
        },
    )
    queue.write(v1 / "queues" / "shuttle-adaptive.json", immutable=False)
    audit_queue = AnnotationQueue(
        queue_id="v1-audit-queue", kind="audit", task="shuttle_selection", seed=1729,
        bursts=(burst,), construction={"anchor_count": 1},
    )
    audit_queue.write(v1 / "queues" / "shuttle-audit.json", immutable=False)
    before = {path: path.read_bytes() for path in v1.rglob("*") if path.is_file()}

    output = tmp_path / "pilot-runtime"
    lineage = migrate_v1_runtime(
        v1, output, pilot_registry, {"source": old_candidates},
        review_keys={("source", 1)},
    )

    assert lineage["migrated_selected_count"] == 1
    assert lineage["rereview_count"] == 2
    assert {path: path.read_bytes() for path in v1.rglob("*") if path.is_file()} == before
    rebound = AnnotationQueue.read(output / "queues" / "shuttle-adaptive.json")
    assert rebound.bursts[0].frames == (2, 0, 1)
    assert rebound.bursts[0].candidate_artifact_sha256 == hashlib.sha256(
        pilot_candidates.read_bytes()
    ).hexdigest()
    rebound_audit = AnnotationQueue.read(output / "queues" / "shuttle-audit.json")
    assert rebound.construction["audit_queue_id"] == rebound_audit.queue_id
    migrated = [json.loads(line) for line in (output / "events" / "shuttle.jsonl").read_text().splitlines()]
    assert len(migrated) == 1
    assert migrated[0]["review_action"] == "migrated"
    assert migrated[0]["annotation_metadata"]["migration"]["v1_candidate_id"] == "f0-shared"

    with pytest.raises(FileExistsError, match="already exists"):
        migrate_v1_runtime(v1, output, pilot_registry, {"source": old_candidates})
