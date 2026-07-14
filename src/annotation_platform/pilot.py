"""Threshold-pilot evaluation, deterministic freezing, and v1 migration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.single_video.shuttle import (
    CANDIDATE_RETENTION_KS,
    candidate_retention_key,
    read_shuttle_candidates,
)

from .core import AnnotationRegistry
from .events import AnnotationEvent, EventStore, replay_events
from .queues import AnnotationQueue
from .shuttle import GROUPING_VERSION, SHUTTLE_TASK, group_shuttle_candidates


PILOT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
DEFAULT_REVIEW_KEYS = frozenset({
    ("malaysia_max_30s_to_120s", 2341),
    ("max_vs_nik_30s_to_120s", 27),
    ("max_vs_nik_30s_to_120s", 725),
    ("max_vs_nik_30s_to_120s", 726),
})
PROVENANCE_FIELDS = (
    "checkpoint_sha256",
    "inference_model_sha256",
    "model_stage",
    "extraction_version",
    "threshold_comparator",
    "fps",
    "image_size",
    "heatmap_size",
    "coordinate_scaling",
    "connectivity",
    "source_frame_range",
    "source_frame_count",
    "source_frame_range_inclusive",
    "source_frame_index_space",
    "pixel_position_convention",
    "normalization",
    "retention_policy",
    "tracknet_config",
    "overlap_ensemble_mode",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _count_summary(values: Sequence[int]) -> dict[str, float | int]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "maximum": max(values, default=0),
    }


def _count_histogram(values: Sequence[int]) -> dict[str, int]:
    return {
        str(value): sum(1 for item in values if item == value)
        for value in sorted(set(values))
    }


def wilson_interval(hits: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Return the two-sided Wilson score interval for a binomial proportion."""
    hits, total = int(hits), int(total)
    if total < 0 or hits < 0 or hits > total:
        raise ValueError("Wilson counts must satisfy 0 <= hits <= total")
    if total == 0:
        return None
    p = hits / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _label_kind(label: Mapping[str, Any]) -> str:
    return str(label.get("label_kind", label.get("outcome", ""))).lower()


def evaluate_threshold_pilot(
    candidate_path: str | Path,
    labels: Iterable[Mapping[str, Any]],
    *,
    cutoffs: Sequence[float] = PILOT_THRESHOLDS,
    ks: Sequence[int | None] = CANDIDATE_RETENTION_KS,
) -> dict[str, Any]:
    """Evaluate grouped threshold cutoffs against observation-aware labels."""
    candidate_path = Path(candidate_path)
    metadata, records = read_shuttle_candidates(candidate_path)
    available = {float(value) for value in metadata.get("thresholds", ())}
    normalized_cutoffs = sorted({float(value) for value in cutoffs})
    if any(value not in available for value in normalized_cutoffs):
        raise ValueError("every evaluated cutoff must be present in the artifact threshold ladder")
    normalized_ks = []
    for value in ks:
        value = None if value is None else int(value)
        if value is not None and value < 1:
            raise ValueError("recall K values must be positive or None")
        if value not in normalized_ks:
            normalized_ks.append(value)
    frames = {int(record["frame"]): list(record.get("candidates", ())) for record in records}
    resolved: dict[int, Mapping[str, Any]] = {}
    for label in labels:
        frame = int(label["frame"])
        if frame in resolved:
            raise ValueError(f"multiple resolved pilot labels for frame {frame}")
        resolved[frame] = label

    results = []
    for cutoff in normalized_cutoffs:
        raw_by_frame = {
            frame: [item for item in candidates if float(item.get("threshold", 0.0)) >= cutoff]
            for frame, candidates in frames.items()
        }
        grouped_by_frame = {
            frame: list(group_shuttle_candidates(candidates))
            for frame, candidates in raw_by_frame.items()
        }
        observed = selected = retained_selected = missing = annotated_missing = cutoff_misses = 0
        occluded = no_target = unsure = legacy = 0
        hits = {k: 0 for k in normalized_ks}
        for frame, label in resolved.items():
            kind = _label_kind(label)
            if kind in {"missing_proposal", "missing"}:
                observed += 1
                missing += 1
                annotated_missing += 1
                continue
            if kind == "occluded_inferable":
                occluded += 1
                continue
            if kind == "no_in_frame_target":
                no_target += 1
                continue
            if kind == "no_shuttle":
                legacy += 1
                continue
            if kind == "unsure":
                unsure += 1
                continue
            if kind not in {"selected", "candidate"}:
                raise ValueError(f"unsupported pilot label kind at frame {frame}: {kind!r}")
            selected_id = str(label.get("candidate_id", ""))
            full_candidates = frames.get(frame)
            if full_candidates is None:
                raise ValueError(f"label references absent candidate frame: {frame}")
            full_groups = group_shuttle_candidates(full_candidates)
            selected_group = next(
                (group for group in full_groups if selected_id in group["raw_member_ids"]), None
            )
            if selected_group is None:
                raise ValueError(f"selected candidate {selected_id!r} is absent at frame {frame}")
            selected_members = set(selected_group["raw_member_ids"])
            groups = grouped_by_frame[frame]
            ranked = sorted(groups, key=candidate_retention_key)
            matched_rank = next(
                (
                    index for index, group in enumerate(ranked, start=1)
                    if selected_members.intersection(group["raw_member_ids"])
                ),
                None,
            )
            observed += 1
            selected += 1
            if matched_rank is None:
                missing += 1
                cutoff_misses += 1
                continue
            retained_selected += 1
            for k in normalized_ks:
                if k is None or matched_rank <= k:
                    hits[k] += 1
        recall = retained_selected
        raw_counts = [len(value) for value in raw_by_frame.values()]
        grouped_counts = [len(value) for value in grouped_by_frame.values()]
        results.append({
            "minimum_cutoff": cutoff,
            "observed_target_frames": observed,
            "selected_label_frames": selected,
            "missing_proposal_frames": missing,
            "annotated_missing_proposal_frames": annotated_missing,
            "selected_lost_at_cutoff_frames": cutoff_misses,
            "observed_hits": recall,
            "observed_proposal_recall": recall / observed if observed else None,
            "wilson_95": wilson_interval(recall, observed),
            "occluded_inferable_frames": occluded,
            "no_in_frame_target_frames": no_target,
            "unsure_frames": unsure,
            "legacy_no_shuttle_frames": legacy,
            "raw_candidates_per_frame": _count_summary(raw_counts),
            "grouped_candidates_per_frame": _count_summary(grouped_counts),
            "raw_candidate_count_histogram": _count_histogram(raw_counts),
            "grouped_candidate_count_histogram": _count_histogram(grouped_counts),
            "recall_at_k": {
                "all" if k is None else str(k): hits[k] / observed if observed else None
                for k in normalized_ks
            },
            "hits_at_k": {"all" if k is None else str(k): hits[k] for k in normalized_ks},
        })
    return {
        "schema": "shuttle_threshold_pilot_report",
        "schema_version": 1,
        "candidate_artifact": candidate_path.name,
        "candidate_sha256": _sha256(candidate_path),
        "grouping_version": GROUPING_VERSION,
        "threshold_results": results,
    }


def freeze_threshold_policy(report: Mapping[str, Any], target_recall: float = 0.99) -> dict[str, Any]:
    """Freeze highest qualifying cutoff, or the Pareto-best recall ceiling."""
    rows = list(report.get("threshold_results", ()))
    if not rows:
        raise ValueError("pilot report has no threshold results")
    qualified = [row for row in rows if row.get("observed_proposal_recall") is not None
                 and float(row["observed_proposal_recall"]) >= target_recall]
    if qualified:
        chosen = max(qualified, key=lambda row: float(row["minimum_cutoff"]))
        reason = "highest_cutoff_reaching_target_point_recall"
    else:
        chosen = min(
            rows,
            key=lambda row: (
                -float(row.get("observed_proposal_recall") or -1.0),
                float(row["grouped_candidates_per_frame"].get(
                    "mean", row["grouped_candidates_per_frame"]["p50"]
                )),
                -float(row["minimum_cutoff"]),
            ),
        )
        reason = "pareto_best_recall_ceiling"
    all_hits = int(chosen["hits_at_k"]["all"])
    numeric = sorted(int(key) for key in chosen["hits_at_k"] if key != "all")
    frozen_k: int | None = next(
        (key for key in numeric if int(chosen["hits_at_k"][str(key)]) == all_hits), None
    )
    return {
        "schema": "shuttle_threshold_freeze",
        "schema_version": 1,
        "target_point_recall": float(target_recall),
        "minimum_cutoff": float(chosen["minimum_cutoff"]),
        "retention_k": frozen_k,
        "observed_hits": all_hits,
        "observed_target_frames": int(chosen["observed_target_frames"]),
        "observed_proposal_recall": chosen["observed_proposal_recall"],
        "wilson_95": chosen["wilson_95"],
        "selection_reason": reason,
        "parent_candidate_sha256": report.get("candidate_sha256"),
        "grouping_version": report.get("grouping_version"),
    }


def filter_pilot_artifact(
    pilot_path: str | Path,
    output_path: str | Path,
    freeze: Mapping[str, Any],
) -> Path:
    """Materialize a fingerprinted production artifact without changing raw IDs."""
    pilot_path, output_path = Path(pilot_path), Path(output_path)
    metadata, records = read_shuttle_candidates(pilot_path)
    cutoff = float(freeze["minimum_cutoff"])
    retention_k = freeze.get("retention_k")
    retention_k = None if retention_k is None else int(retention_k)
    thresholds = sorted(float(value) for value in metadata.get("thresholds", ()) if float(value) >= cutoff)
    if not thresholds or cutoff not in thresholds:
        raise ValueError("frozen cutoff is not present in the pilot artifact")
    output_metadata = dict(metadata)
    output_metadata.update({
        "artifact_stage": "production_frozen_threshold_pilot",
        "thresholds": thresholds,
        "frozen_minimum_cutoff": cutoff,
        "frozen_retention_k": retention_k,
        "grouping_version": GROUPING_VERSION,
        "parent_artifact": pilot_path.name,
        "parent_artifact_sha256": _sha256(pilot_path),
        "freeze_policy": dict(freeze),
    })
    lines = [json.dumps(output_metadata, sort_keys=True, separators=(",", ":"))]
    for record in records:
        candidates = [
            dict(item) for item in record.get("candidates", ())
            if float(item.get("threshold", 0.0)) >= cutoff
        ]
        if retention_k is not None:
            groups = sorted(group_shuttle_candidates(candidates), key=candidate_retention_key)
            retained_ids = {
                candidate_id
                for group in groups[:retention_k]
                for candidate_id in group["raw_member_ids"]
            }
            candidates = [item for item in candidates if item["candidate_id"] in retained_ids]
        lines.append(json.dumps(
            {"type": "frame", "frame": int(record["frame"]), "candidates": candidates},
            sort_keys=True,
            separators=(",", ":"),
        ))
    encoded = ("\n".join(lines) + "\n").encode()
    if output_path.exists():
        if output_path.read_bytes() != encoded:
            raise FileExistsError(f"production candidate artifact already differs: {output_path}")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def validate_artifact_lineage(old_path: str | Path, new_path: str | Path) -> None:
    """Require exact provenance and exact records at every shared threshold."""
    old_path, new_path = Path(old_path), Path(new_path)
    old_meta, old_records = read_shuttle_candidates(old_path)
    new_meta, new_records = read_shuttle_candidates(new_path)
    for field in PROVENANCE_FIELDS:
        if old_meta.get(field) != new_meta.get(field):
            raise ValueError(f"candidate provenance mismatch for {field}")
    if [int(item["frame"]) for item in old_records] != [int(item["frame"]) for item in new_records]:
        raise ValueError("candidate frame ranges/order do not match")
    shared = set(float(value) for value in old_meta.get("thresholds", ())).intersection(
        float(value) for value in new_meta.get("thresholds", ())
    )
    if not shared:
        raise ValueError("candidate artifacts have no shared thresholds")
    for old_record, new_record in zip(old_records, new_records):
        old_shared = [item for item in old_record.get("candidates", ())
                      if float(item.get("threshold", math.nan)) in shared]
        new_shared = [item for item in new_record.get("candidates", ())
                      if float(item.get("threshold", math.nan)) in shared]
        if old_shared != new_shared:
            raise ValueError(
                f"shared-threshold candidate records differ at frame {old_record['frame']}"
            )


def rebind_queue(
    queue: AnnotationQueue,
    artifact_hashes: Mapping[str, str],
    *,
    lineage: Mapping[str, Any],
) -> AnnotationQueue:
    bursts = tuple(replace(
        burst,
        candidate_artifact_sha256=artifact_hashes[burst.source_id],
    ) for burst in queue.bursts)
    identity = json.dumps({
        "parent_queue_id": queue.queue_id,
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "lineage": dict(lineage),
    }, sort_keys=True, separators=(",", ":"))
    return replace(
        queue,
        queue_id=str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
        bursts=bursts,
        construction={**queue.construction, "migration_lineage": dict(lineage)},
    )


def _read_events(path: Path) -> tuple[AnnotationEvent, ...]:
    records = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    return replay_events(AnnotationEvent.from_mapping(item) for item in records).events


def migrate_v1_runtime(
    v1_runtime: str | Path,
    pilot_runtime: str | Path,
    pilot_registry: AnnotationRegistry,
    old_artifacts: Mapping[str, str | Path],
    *,
    review_keys: Iterable[tuple[str, int]] = DEFAULT_REVIEW_KEYS,
) -> dict[str, Any]:
    """Create a new pilot runtime transactionally; never edit the v1 runtime."""
    v1_runtime, pilot_runtime = Path(v1_runtime).resolve(), Path(pilot_runtime).resolve()
    if pilot_runtime.exists():
        raise FileExistsError(f"pilot runtime already exists: {pilot_runtime}")
    event_path = v1_runtime / "events" / "shuttle.jsonl"
    source_paths = [event_path, *sorted((v1_runtime / "queues").glob("*.json"))]
    before = {path: _sha256(path) for path in source_paths}
    old_events = _read_events(event_path)
    active = replay_events(old_events).active
    plugin = pilot_registry.tasks[SHUTTLE_TASK].plugin
    new_hashes: dict[str, str] = {}
    for source_id, source in pilot_registry.sources.items():
        if source_id not in old_artifacts:
            raise ValueError(f"missing old candidate artifact for {source_id}")
        validate_artifact_lineage(old_artifacts[source_id], source.artifacts["candidates"])
        new_hashes[source_id] = plugin.verify_artifact_fingerprint(source)
        old_source_events = [event for event in old_events if event.source_id == source_id]
        if any(event.source_video_sha256 != source.video_sha256 for event in old_source_events):
            raise ValueError(f"source-video fingerprint mismatch for {source_id}")
    queues = [AnnotationQueue.read(path) for path in sorted((v1_runtime / "queues").glob("*.json"))]
    for queue in queues:
        for burst in queue.bursts:
            source = pilot_registry.sources[burst.source_id]
            if burst.source_video_sha256 != source.video_sha256:
                raise ValueError(f"queue source-video fingerprint mismatch for {burst.source_id}")
            if burst.candidate_artifact_sha256 != _sha256(Path(old_artifacts[burst.source_id])):
                raise ValueError(f"queue candidate fingerprint mismatch for {burst.source_id}")
    review = set(review_keys)
    safe = [event for event in active.values()
            if event.label_kind == "selected" and (event.source_id, event.frame) not in review]
    ambiguous = [event for event in active.values() if event not in safe]
    event_sha = _sha256(event_path)
    lineage = {
        "kind": "v1_to_threshold_pilot",
        "v1_runtime": str(v1_runtime),
        "v1_event_log_sha256": event_sha,
        "v1_revision_count": len(old_events),
        "migrated_selected_count": len(safe),
        "rereview_count": len(ambiguous),
        "grouping_version": GROUPING_VERSION,
    }
    temporary = pilot_runtime.with_name(f".{pilot_runtime.name}.{uuid.uuid4().hex}.tmp")
    try:
        (temporary / "events").mkdir(parents=True)
        rebound = [rebind_queue(queue, new_hashes, lineage=lineage) for queue in queues]
        rebound_audit_ids = {
            original.queue_id: migrated.queue_id
            for original, migrated in zip(queues, rebound)
            if original.kind == "audit"
        }
        rebound = [
            replace(
                migrated,
                construction={
                    **migrated.construction,
                    "audit_queue_id": rebound_audit_ids[original.construction["audit_queue_id"]],
                },
            )
            if original.construction.get("audit_queue_id") is not None
            else migrated
            for original, migrated in zip(queues, rebound)
        ]
        for queue in rebound:
            name = "shuttle-audit.json" if queue.kind == "audit" else "shuttle-adaptive.json"
            queue.write(temporary / "queues" / name, immutable=False)
        store = EventStore(temporary / "events" / "shuttle.jsonl", pilot_registry)
        for event in sorted(safe, key=lambda item: (item.source_id, item.frame)):
            source = pilot_registry.sources[event.source_id]
            representative = plugin.representative_candidate_id(source, event.frame, event.candidate_id or "")
            group = next(
                item for item in plugin.annotator_overlays(source, event.frame)
                if representative == item["candidate_id"]
            )
            store.record(
                task=event.task,
                source_id=event.source_id,
                frame=event.frame,
                label_kind="selected",
                candidate_id=representative,
                candidate_artifact_sha256=new_hashes[event.source_id],
                annotator=event.annotator,
                session_id="migration-v1-" + event.session_id,
                review_action_override="migrated",
                annotation_metadata={
                    "migration": {
                        **lineage,
                        "v1_revision_id": event.revision_id,
                        "v1_candidate_artifact_sha256": event.candidate_artifact_sha256,
                        "v1_candidate_id": event.candidate_id,
                    },
                    "grouping_version": GROUPING_VERSION,
                    "raw_member_ids": list(group["raw_member_ids"]),
                    "representative_candidate_id": representative,
                },
            )
        manifest = {**lineage, "artifact_hashes": new_hashes,
                    "parent_file_sha256": {str(path.relative_to(v1_runtime)): digest for path, digest in before.items()}}
        (temporary / "migration.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        for path, digest in before.items():
            if _sha256(path) != digest:
                raise RuntimeError(f"v1 file changed during migration: {path}")
        os.replace(temporary, pilot_runtime)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return lineage


def materialize_final_runtime(
    pilot_runtime: str | Path,
    final_runtime: str | Path,
    final_registry: AnnotationRegistry,
    pilot_artifacts: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Bind reviewed pilot labels/queues to frozen production artifacts."""
    pilot_runtime, final_runtime = Path(pilot_runtime).resolve(), Path(final_runtime).resolve()
    if final_runtime.exists():
        raise FileExistsError(f"final runtime already exists: {final_runtime}")
    pilot_event_path = pilot_runtime / "events" / "shuttle.jsonl"
    pilot_events = _read_events(pilot_event_path)
    active = replay_events(pilot_events).active
    plugin = final_registry.tasks[SHUTTLE_TASK].plugin
    final_hashes: dict[str, str] = {}
    for source_id, source in final_registry.sources.items():
        if source_id not in pilot_artifacts:
            raise ValueError(f"missing pilot artifact for {source_id}")
        validate_artifact_lineage(pilot_artifacts[source_id], source.artifacts["candidates"])
        final_hashes[source_id] = plugin.verify_artifact_fingerprint(source)
        if any(event.source_video_sha256 != source.video_sha256 for event in pilot_events
               if event.source_id == source_id):
            raise ValueError(f"source-video fingerprint mismatch for {source_id}")
    lineage = {
        "kind": "threshold_pilot_to_final",
        "pilot_runtime": str(pilot_runtime),
        "pilot_event_log_sha256": _sha256(pilot_event_path),
        "pilot_revision_count": len(pilot_events),
        "grouping_version": GROUPING_VERSION,
    }
    temporary = final_runtime.with_name(f".{final_runtime.name}.{uuid.uuid4().hex}.tmp")
    derived_missing = 0
    try:
        (temporary / "events").mkdir(parents=True)
        for path in sorted((pilot_runtime / "queues").glob("*.json")):
            queue = rebind_queue(AnnotationQueue.read(path), final_hashes, lineage=lineage)
            queue.write(temporary / "queues" / path.name, immutable=False)
        store = EventStore(temporary / "events" / "shuttle.jsonl", final_registry)
        for event in sorted(active.values(), key=lambda item: (item.source_id, item.frame)):
            source = final_registry.sources[event.source_id]
            label_kind = event.label_kind
            candidate_id = None
            mapping: dict[str, Any] = {}
            if label_kind == "selected":
                pilot_members = set(
                    (event.annotation_metadata or {}).get(
                        "raw_member_ids", (event.candidate_id,) if event.candidate_id else ()
                    )
                )
                final_groups = plugin.annotator_overlays(source, event.frame)
                retained = next(
                    (group for group in final_groups if pilot_members.intersection(group["raw_member_ids"])),
                    None,
                )
                if retained is None:
                    label_kind = "missing_proposal"
                    derived_missing += 1
                    mapping["derived_reason"] = "selected_pilot_group_has_no_retained_member"
                else:
                    candidate_id = str(retained["candidate_id"])
                    mapping.update({
                        "grouping_version": GROUPING_VERSION,
                        "raw_member_ids": list(retained["raw_member_ids"]),
                        "representative_candidate_id": candidate_id,
                    })
            if label_kind == "no_shuttle":
                raise ValueError("legacy no_shuttle must be re-reviewed before final materialization")
            store.record(
                task=event.task,
                source_id=event.source_id,
                frame=event.frame,
                label_kind=label_kind,
                candidate_id=candidate_id,
                candidate_artifact_sha256=final_hashes[event.source_id],
                annotator=event.annotator,
                session_id="final-" + event.session_id,
                review_action_override="derived",
                annotation_metadata={
                    **mapping,
                    "pilot_lineage": {
                        **lineage,
                        "pilot_revision_id": event.revision_id,
                        "pilot_label_kind": event.label_kind,
                        "pilot_candidate_id": event.candidate_id,
                    },
                },
            )
        lineage["derived_missing_proposal_count"] = derived_missing
        lineage["final_artifact_hashes"] = final_hashes
        (temporary / "migration.json").write_text(
            json.dumps(lineage, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, final_runtime)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return lineage
