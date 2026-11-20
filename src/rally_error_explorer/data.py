"""Load, validate, and align rally predictions with their source artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.pose_estimator import POSE_NUM_LANDMARKS
from src.temporal_selector.dataset import _jsonl as read_jsonl_artifact
from src.temporal_selector.joint_inference import (
    InPlayDecoderConfig,
    decode_inplay_probabilities,
)
from src.temporal_selector.rally_intervals import RallyIntervalIndex, read_rally_intervals
from src.workflow.identity import SourceIdentity, sha256_file


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-value)))


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class SourceBundle:
    """A validated source and its immutable player/pose artifacts."""

    source_id: str
    video_path: Path
    assignments_path: Path
    pose_cache_path: Path
    calibration_path: Path
    identity_path: Path
    identity: SourceIdentity
    assignment_meta: Mapping[str, Any]
    assignments: Mapping[int, Mapping[str, Any]]
    pose_meta: Mapping[str, Any]
    poses: Mapping[tuple[int, int], Mapping[str, Any]]
    fps: float
    frame_count: int
    image_size: tuple[int, int]


@dataclass
class FrameSeries:
    """One source's normalized prediction signal aligned to source-local frames."""

    source_id: str
    frames: list[int]
    inplay_probability: list[float]
    start_probability: list[float]
    end_probability: list[float]
    decoded: list[bool]
    targets: list[int]
    decoder_threshold: float
    start_available: bool = False
    end_available: bool = False
    coverage_warning: str | None = None
    shuttle_rows: dict[int, Mapping[str, Any]] = field(default_factory=dict)


@dataclass
class PredictionRun:
    """One normalized prediction file."""

    name: str
    path: Path
    series: dict[str, FrameSeries]
    has_targets: bool
    shuttle_in_rows: bool


@dataclass
class PlayerSeries:
    """Per-source player detection, pose validity, and activity signals."""

    source_id: str
    frames: list[int]
    p1_detected: list[int]
    p2_detected: list[int]
    p1_pose_valid: list[float]
    p2_pose_valid: list[float]
    p1_activity: list[float]
    p2_activity: list[float]


@dataclass
class ExplorerData:
    """Everything the Dash app needs for one launch."""

    runs: Mapping[str, PredictionRun]
    bundles: Mapping[str, SourceBundle]
    source_ids: list[str]
    player_series: Mapping[str, PlayerSeries]
    rally_index: RallyIntervalIndex | None = None
    consistency: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    shuttle_sidecar: Mapping[str, Mapping[int, Mapping[str, Any]]] = field(
        default_factory=dict
    )
    shuttle_sidecar_provenance: str | None = None


def _load_jsonl_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            records.append(value)
    if not records:
        raise ValueError(f"prediction file is empty: {path}")
    metadata = records[0] if records[0].get("type") == "metadata" else {}
    return metadata, records


def _metadata_source_ids(metadata: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    single = metadata.get("source_id")
    many = metadata.get("source_ids")
    return (
        str(single) if single is not None else None,
        [str(value) for value in many] if isinstance(many, list) else [],
    )


def discover_source_ids(path: Path) -> list[str]:
    """Return unique source IDs without requiring source artifacts yet."""
    metadata, records = _load_jsonl_records(path)
    single, many = _metadata_source_ids(metadata)
    found: set[str] = set()
    for row in records:
        if row.get("type") == "metadata":
            continue
        source = row.get("source_id") or row.get("prediction_source_id")
        if source is not None:
            found.add(str(source))
        elif single is not None:
            found.add(single)
        elif len(many) == 1:
            found.add(many[0])
    if not found:
        raise ValueError(f"prediction file has no resolvable source IDs: {path}")
    return sorted(found)


def _resolve_config_sources(
    config_path: Path, source_ids: set[str]
) -> dict[str, dict[str, Path]]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = raw.get("sources")
    if candidates is None and isinstance(raw.get("dataset"), Mapping):
        candidates = raw["dataset"].get("sources")
    if not isinstance(candidates, list):
        raise ValueError("--sources JSON must contain a 'sources' or 'dataset.sources' list")

    def resolve(value: Any) -> Path:
        path = Path(str(value)).expanduser()
        return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()

    resolved: dict[str, dict[str, Path]] = {}
    for source in candidates:
        if not isinstance(source, Mapping):
            continue
        source_id = str(source.get("source_id", ""))
        if source_id not in source_ids:
            continue
        missing = [
            name
            for name in ("video_path", "assignments_path", "pose_cache_path", "calibration_path")
            if not source.get(name)
        ]
        if missing:
            raise ValueError(f"source {source_id!r} is missing paths: {missing}")
        resolved[source_id] = {
            "video_path": resolve(source["video_path"]),
            "assignments_path": resolve(source["assignments_path"]),
            "pose_cache_path": resolve(source["pose_cache_path"]),
            "calibration_path": resolve(source["calibration_path"]),
        }
    return resolved


def _resolve_artifact_root_source(
    root: Path, source_id: str
) -> dict[str, Path] | None:
    bases = [root / source_id, root / source_id / "artifacts"]
    for base in bases:
        identity = base / "source.json"
        if not identity.is_file():
            continue
        video_candidates = [base / f"{source_id}_input.mp4", base / "video.mp4"]
        video_candidates.extend(
            path
            for path in sorted(base.glob("*.mp4"))
            if path.name not in {candidate.name for candidate in video_candidates}
            and "preview" not in path.name.lower()
        )
        sibling_video_dir = base.parent / "video"
        if sibling_video_dir.is_dir():
            video_candidates.extend(
                path
                for path in sorted(sibling_video_dir.iterdir())
                if path.is_file() and path.suffix.lower() in {".mp4", ".mov"}
            )
        video = next((path for path in video_candidates if path.is_file()), None)
        assignments = next(
            (base / name for name in ("players/assignments.jsonl", "assignments.jsonl") if (base / name).is_file()),
            None,
        )
        poses = next(
            (base / name for name in ("players/poses.jsonl", "pose_cache.jsonl") if (base / name).is_file()),
            None,
        )
        calibration = base / "court" / "calibration.json"
        if (
            video is not None
            and assignments is not None
            and poses is not None
            and calibration.is_file()
        ):
            return {
                "identity_path": identity,
                "video_path": video,
                "assignments_path": assignments,
                "pose_cache_path": poses,
                "calibration_path": calibration,
            }
    return None


def load_sources(
    source_ids: set[str],
    *,
    sources_path: Path | None = None,
    artifact_root: Path | None = None,
) -> dict[str, SourceBundle]:
    if not source_ids:
        raise ValueError("no source IDs found in predictions")
    configured: dict[str, dict[str, Path]] = {}
    if sources_path is not None:
        configured = _resolve_config_sources(sources_path, source_ids)
    elif artifact_root is None:
        raise ValueError("provide --sources or --artifact-root to resolve prediction sources")

    bundles: dict[str, SourceBundle] = {}
    missing: list[str] = []
    for source_id in sorted(source_ids):
        paths = configured.get(source_id)
        if paths is None and artifact_root is not None:
            paths = _resolve_artifact_root_source(artifact_root, source_id)
        if paths is None:
            missing.append(source_id)
            continue
        bundles[source_id] = load_source_bundle(
            source_id,
            video_path=paths["video_path"],
            assignments_path=paths["assignments_path"],
            pose_cache_path=paths["pose_cache_path"],
            calibration_path=paths["calibration_path"],
            identity_path=paths.get("identity_path", paths["calibration_path"].parent.parent / "source.json"),
        )
    if missing:
        raise FileNotFoundError(
            "could not resolve sources: " + ", ".join(sorted(missing))
            + "; provide --sources or --artifact-root"
        )
    return bundles


def load_source_bundle(
    source_id: str,
    *,
    video_path: Path,
    assignments_path: Path,
    pose_cache_path: Path,
    calibration_path: Path,
    identity_path: Path | None = None,
) -> SourceBundle:
    video_path = video_path.expanduser().resolve()
    assignments_path = assignments_path.expanduser().resolve()
    pose_cache_path = pose_cache_path.expanduser().resolve()
    calibration_path = calibration_path.expanduser().resolve()
    identity_path = (
        identity_path.expanduser().resolve()
        if identity_path is not None
        else calibration_path.parent.parent / "source.json"
    )
    for name, path in (
        ("video", video_path),
        ("assignments", assignments_path),
        ("pose cache", pose_cache_path),
        ("calibration", calibration_path),
        ("source identity", identity_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} artifact is missing for {source_id}: {path}")
    identity = SourceIdentity.read(identity_path)
    if identity.source_id != source_id:
        raise ValueError(f"source identity ID {identity.source_id!r} != {source_id!r}")
    assignment_meta, assignment_records = read_jsonl_artifact(assignments_path)
    pose_meta, pose_records = read_jsonl_artifact(pose_cache_path)
    if assignment_meta.get("source_id") != source_id:
        raise ValueError(f"assignment metadata source ID mismatch for {source_id}")
    frame_count = int(assignment_meta["frame_count"])
    fps = float(assignment_meta["fps"])
    image_size = tuple(int(value) for value in assignment_meta["frame_size"])
    if (
        frame_count != identity.frame_count
        or not math.isclose(fps, identity.fps_numerator / identity.fps_denominator, rel_tol=0, abs_tol=1e-6)
        or image_size != (identity.width, identity.height)
    ):
        raise ValueError(f"source identity geometry does not match player artifacts: {source_id}")
    assignments = {int(record["frame"]): record for record in assignment_records}
    if sorted(assignments) != list(range(frame_count)):
        raise ValueError(f"player assignments must be complete for {source_id}")
    if pose_meta.get("raw_artifact_fingerprint") != assignment_meta.get("raw_artifact_fingerprint"):
        raise ValueError(f"pose-cache lineage does not match player assignments: {source_id}")
    poses = {
        (int(record["frame"]), int(record["track_id"])): record
        for record in pose_records
    }
    return SourceBundle(
        source_id=source_id,
        video_path=video_path,
        assignments_path=assignments_path,
        pose_cache_path=pose_cache_path,
        calibration_path=calibration_path,
        identity_path=identity_path,
        identity=identity,
        assignment_meta=assignment_meta,
        assignments=assignments,
        pose_meta=pose_meta,
        poses=poses,
        fps=fps,
        frame_count=frame_count,
        image_size=image_size,
    )


def _row_probability(row: Mapping[str, Any], key: str) -> float | None:
    value = _finite_float(row.get(key))
    if value is not None:
        return value if 0.0 <= value <= 1.0 else None
    logit = _finite_float(row.get(key.replace("probability", "logit")))
    if logit is not None:
        probability = _sigmoid(logit)
        return probability if 0.0 <= probability <= 1.0 else None
    return None


def _row_inplay_probability(row: Mapping[str, Any]) -> float | None:
    for key in ("inplay_probability", "in_play_probability"):
        value = _finite_float(row.get(key))
        if value is not None and 0.0 <= value <= 1.0:
            return value
    logit = _finite_float(row.get("inplay_logit"))
    if logit is not None:
        probability = _sigmoid(logit)
        return probability if 0.0 <= probability <= 1.0 else None
    return None


def _row_decoder(row: Mapping[str, Any]) -> InPlayDecoderConfig | None:
    raw = row.get("decoder_config") or row.get("decoder_configuration")
    return _decoder_from_mapping(raw)


def _decoder_from_mapping(raw: Any) -> InPlayDecoderConfig | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        return InPlayDecoderConfig(
            threshold=float(raw.get("threshold", 0.5)),
            max_gap_seconds=float(raw.get("max_gap_seconds", 0.2)),
            minimum_duration_seconds=float(raw.get("minimum_duration_seconds", 0.5)),
            preserve_edge_runs=bool(raw.get("preserve_edge_runs", False)),
        )
    except (TypeError, ValueError):
        return None


def _runs(values: Sequence[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate((*values, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def derive_gt_intervals(frames: Sequence[int], targets: Sequence[int]) -> list[tuple[int, int]]:
    if len(frames) != len(targets):
        raise ValueError("frames and targets must align")
    return [
        (frames[start], frames[end])
        for start, end in _runs([int(target) == 1 for target in targets])
    ]


def _normalize_run(
    path: Path,
    name: str,
    bundles: Mapping[str, SourceBundle],
) -> PredictionRun:
    metadata, records = _load_jsonl_records(path)
    metadata_decoder = _decoder_from_mapping(metadata.get("decoder"))
    single, many = _metadata_source_ids(metadata)
    grouped: dict[str, list[dict[str, Any]]] = {}
    shuttle_in_rows = False
    for row in records:
        if row.get("type") == "metadata":
            continue
        source = row.get("source_id") or row.get("prediction_source_id")
        if source is None:
            source = single if single is not None else (many[0] if len(many) == 1 else None)
        if source is None:
            raise ValueError(f"{name} row has no resolvable source ID: {path}")
        source = str(source)
        frame = int(row["frame"])
        inplay = _row_inplay_probability(row)
        if inplay is None:
            raise ValueError(f"{name} row {source}:{frame} has no InPlay probability")
        start = _row_probability(row, "rally_start_probability")
        end = _row_probability(row, "rally_end_probability")
        target = int(row["inplay_target"]) if row.get("inplay_target") in (0, 1) else -1
        decoded = row.get("decoded_inplay")
        if not isinstance(decoded, bool):
            decoded = None
        if any(key in row for key in ("candidate_count", "candidate_logits", "selected_position")):
            shuttle_in_rows = True
        grouped.setdefault(source, []).append(
            {
                "frame": frame,
                "inplay": inplay,
                "start": start if start is not None else 0.0,
                "end": end if end is not None else 0.0,
                "has_start": start is not None,
                "has_end": end is not None,
                "decoded": decoded,
                "target": target,
                "decoder": _row_decoder(row),
                "shuttle": {
                    "candidate_count": int(row.get("candidate_count", 0)),
                    "selected_outcome": str(row.get("predicted_selection_outcome", "")),
                    "selected_position": row.get("selected_position"),
                    "candidate_positions": row.get("candidate_positions"),
                },
            }
        )
    if not grouped:
        raise ValueError(f"{name} has no prediction rows: {path}")
    series: dict[str, FrameSeries] = {}
    has_targets = False
    for source, rows in grouped.items():
        if source not in bundles:
            raise ValueError(f"{name} references unknown source {source!r}")
        rows.sort(key=lambda row: row["frame"])
        frames = [int(row["frame"]) for row in rows]
        if len(frames) != len(set(frames)):
            raise ValueError(f"{name} has duplicate frames for {source}")
        if frames != list(range(frames[0], frames[-1] + 1)):
            raise ValueError(f"{name} has non-contiguous frames for {source}")
        bundle = bundles[source]
        warning = None
        if frames[0] != 0 or len(frames) != bundle.frame_count:
            warning = (
                f"{name} {source} covers frames {frames[0]}..{frames[-1]}, "
                f"not the full 0..{bundle.frame_count - 1}"
            )
        row_decoders = [row["decoder"] for row in rows if row["decoder"] is not None]
        decoder_configs = row_decoders or ([metadata_decoder] if metadata_decoder is not None else [])
        decoder = decoder_configs[0] if decoder_configs else InPlayDecoderConfig()
        inplay_values = [float(row["inplay"]) for row in rows]
        if any(row["decoded"] is None for row in rows):
            states = decode_inplay_probabilities(inplay_values, fps=bundle.fps, config=decoder)
            for row, state in zip(rows, states):
                row["decoded"] = bool(state)
        targets = [int(row["target"]) for row in rows]
        if any(target != -1 for target in targets):
            has_targets = True
        shuttle_rows = {
            int(row["frame"]): row["shuttle"] for row in rows if int(row["shuttle"]["candidate_count"]) > 0
        }
        series[source] = FrameSeries(
            source_id=source,
            frames=frames,
            inplay_probability=inplay_values,
            start_probability=[float(row["start"]) for row in rows],
            end_probability=[float(row["end"]) for row in rows],
            decoded=[bool(row["decoded"]) for row in rows],
            targets=targets,
            decoder_threshold=float(decoder.threshold),
            start_available=any(row["has_start"] for row in rows),
            end_available=any(row["has_end"] for row in rows),
            coverage_warning=warning,
            shuttle_rows=shuttle_rows,
        )
    return PredictionRun(
        name=name,
        path=path,
        series=series,
        has_targets=has_targets,
        shuttle_in_rows=shuttle_in_rows,
    )


def compute_player_series(bundle: SourceBundle, visibility_threshold: float = 0.5) -> PlayerSeries:
    frames: list[int] = []
    p1_detected: list[int] = []
    p2_detected: list[int] = []
    p1_pose: list[float] = []
    p2_pose: list[float] = []
    p1_activity: list[float] = []
    p2_activity: list[float] = []
    for frame in range(bundle.frame_count):
        frames.append(frame)
        assignment = bundle.assignments.get(frame, {})
        role_values = []
        for role in ("P1", "P2"):
            slot = assignment.get("slots", {}).get(role, {})
            item = slot.get("assignment")
            detected = 0
            activity = 0.0
            track_id = None
            if isinstance(item, Mapping) and item.get("bbox"):
                detected = 1
                activity = float(item.get("activity", 0.0))
                track_id = item.get("track_id")
            role_values.append((detected, activity, track_id))
        (p1_det, p1_act, p1_track), (p2_det, p2_act, p2_track) = role_values
        p1_detected.append(p1_det)
        p2_detected.append(p2_det)
        p1_activity.append(p1_act)
        p2_activity.append(p2_act)
        p1_pose.append(_pose_validity(bundle, frame, p1_track, visibility_threshold))
        p2_pose.append(_pose_validity(bundle, frame, p2_track, visibility_threshold))
    return PlayerSeries(
        source_id=bundle.source_id,
        frames=frames,
        p1_detected=p1_detected,
        p2_detected=p2_detected,
        p1_pose_valid=p1_pose,
        p2_pose_valid=p2_pose,
        p1_activity=p1_activity,
        p2_activity=p2_activity,
    )


def _pose_validity(
    bundle: SourceBundle,
    frame: int,
    track_id: Any,
    visibility_threshold: float,
) -> float:
    if track_id is None:
        return 0.0
    pose = bundle.poses.get((frame, int(track_id)))
    if not isinstance(pose, Mapping) or pose.get("status") != "detected":
        return 0.0
    landmarks = pose.get("pose_landmarks")
    if not isinstance(landmarks, list) or len(landmarks) != POSE_NUM_LANDMARKS:
        return 0.0
    valid = sum(
        float(landmark.get("visibility", 0.0)) >= visibility_threshold
        for landmark in landmarks
        if isinstance(landmark, Mapping)
    )
    return valid / POSE_NUM_LANDMARKS


def _load_shuttle_jsonl(path: Path) -> dict[int, Mapping[str, Any]]:
    frames: dict[int, Mapping[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") != "frame":
            continue
        candidates = row.get("frozen_candidates") or row.get("candidates") or []
        if not isinstance(candidates, list) or not candidates:
            frames[int(row["frame"])] = {"candidate_count": 0, "top_peak": 0.0}
            continue
        top = max(
            candidates,
            key=lambda item: float(
                item.get("peak_activation", item.get("peak_value", 0.0))
            ),
        )
        peak_position = top.get("peak_position_normalized") or top.get("center_normalized")
        bbox = top.get("bbox_normalized") or top.get("bbox")
        frames[int(row["frame"])] = {
            "candidate_count": len(candidates),
            "top_peak": float(top.get("peak_activation", top.get("peak_value", 0.0))),
            "top_position_normalized": peak_position,
            "top_bbox_normalized": bbox,
        }
    return frames


def _load_shuttle_sidecars(
    source_ids: set[str],
    *,
    shuttle_artifacts_path: Path | None,
    bundles: Mapping[str, SourceBundle],
) -> tuple[dict[str, dict[int, Mapping[str, Any]]], str | None]:
    if shuttle_artifacts_path is not None:
        raw = json.loads(shuttle_artifacts_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("--shuttle-artifacts JSON must be an object")

        def resolve(value: Any) -> Path:
            path = Path(str(value)).expanduser()
            return path.resolve() if path.is_absolute() else (shuttle_artifacts_path.parent / path).resolve()

        return {
            source_id: _load_shuttle_jsonl(resolve(raw[source_id]))
            for source_id in source_ids
            if source_id in raw
        }, "external_diagnostic"
    # Auto-discover from artifact root conventions when the bundle lives under one.
    sidecars: dict[str, dict[int, Mapping[str, Any]]] = {}
    for source_id in sorted(source_ids):
        bundle = bundles[source_id]
        base = bundle.identity_path.parent
        candidates = [
            base / "shuttle" / "candidates.frozen.jsonl",
            base / "shuttle_candidates_frozen.jsonl",
            base / "shuttle_candidates.jsonl",
        ]
        for candidate in candidates:
            if candidate.is_file():
                sidecars[source_id] = _load_shuttle_jsonl(candidate)
                break
    return sidecars, ("external_diagnostic" if sidecars else None)


def run_consistency_check(
    bundles: Mapping[str, SourceBundle],
    primary: PredictionRun,
    rally_index: RallyIntervalIndex,
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    consistency: dict[str, Mapping[str, Any]] = {}
    warnings: list[str] = []
    for source_id, bundle in bundles.items():
        series = primary.series.get(source_id)
        if series is None:
            continue
        manifest = rally_index.sources.get(source_id)
        provenance_ok = True
        if manifest is not None:
            if manifest.video_sha256 != sha256_file(bundle.video_path):
                provenance_ok = False
            if not math.isclose(manifest.fps, bundle.fps, rel_tol=0, abs_tol=1e-6):
                provenance_ok = False
            if manifest.frame_count != bundle.frame_count:
                provenance_ok = False
        derived = derive_gt_intervals(series.frames, series.targets)
        mismatched = 0
        compared = 0
        if all(target == -1 for target in series.targets):
            continue
        for frame, target in zip(series.frames, series.targets):
            if target == -1:
                continue
            compared += 1
            if bool(target) != rally_index.is_inplay(source_id, frame):
                mismatched += 1
        strict = rally_index.for_source(source_id)
        message = (
            f"{source_id}: {len(derived)} derived GT intervals vs {len(strict)} strict intervals; "
            f"{mismatched}/{compared} frame mismatches; provenance {'ok' if provenance_ok else 'FAILED'}"
        )
        consistency[source_id] = {
            "derived_interval_count": len(derived),
            "strict_interval_count": len(strict),
            "mismatched_frames": mismatched,
            "compared_frames": compared,
            "provenance_ok": provenance_ok,
            "message": message,
        }
        if mismatched or not provenance_ok:
            warnings.append(message)
    return consistency, warnings


def load_explorer_data(
    predictions_path: Path,
    *,
    compare_path: Path | None = None,
    sources_path: Path | None = None,
    artifact_root: Path | None = None,
    rallies_path: Path | None = None,
    manifest_path: Path | None = None,
    shuttle_artifacts_path: Path | None = None,
) -> ExplorerData:
    predictions_path = predictions_path.expanduser().resolve()
    if not predictions_path.is_file():
        raise FileNotFoundError(f"predictions file is missing: {predictions_path}")
    if sources_path is not None:
        sources_path = sources_path.expanduser().resolve()
    if artifact_root is not None:
        artifact_root = artifact_root.expanduser().resolve()
    if rallies_path is not None:
        rallies_path = rallies_path.expanduser().resolve()
    if manifest_path is not None:
        manifest_path = manifest_path.expanduser().resolve()
    if shuttle_artifacts_path is not None:
        shuttle_artifacts_path = shuttle_artifacts_path.expanduser().resolve()
    source_ids = set(discover_source_ids(predictions_path))
    if compare_path is not None:
        compare_path = compare_path.expanduser().resolve()
        if not compare_path.is_file():
            raise FileNotFoundError(f"compare predictions file is missing: {compare_path}")
        source_ids.update(discover_source_ids(compare_path))
    bundles = load_sources(
        source_ids,
        sources_path=sources_path,
        artifact_root=artifact_root,
    )
    primary = _normalize_run(predictions_path, "primary", bundles)
    runs: dict[str, PredictionRun] = {"primary": primary}
    warnings: list[str] = []
    for series in primary.series.values():
        if series.coverage_warning:
            warnings.append(series.coverage_warning)
    if compare_path is not None:
        compare = _normalize_run(compare_path, "compare", bundles)
        if set(compare.series) != set(primary.series):
            raise ValueError("primary and compare runs must cover the same source IDs")
        for source_id in compare.series:
            if compare.series[source_id].frames != primary.series[source_id].frames:
                raise ValueError(f"primary and compare runs disagree on frames for {source_id}")
        runs["compare"] = compare
    player_series = {
        source_id: compute_player_series(bundle) for source_id, bundle in bundles.items()
    }
    shuttle_sidecar, shuttle_provenance = _load_shuttle_sidecars(
        set(bundles), shuttle_artifacts_path=shuttle_artifacts_path, bundles=bundles
    )
    rally_index = None
    consistency: dict[str, Mapping[str, Any]] = {}
    if rallies_path is not None or manifest_path is not None:
        if rallies_path is None or manifest_path is None:
            raise ValueError("--rallies and --manifest must be provided together")
        rally_index = read_rally_intervals(rallies_path, manifest_path)
        consistency, consistency_warnings = run_consistency_check(bundles, primary, rally_index)
        warnings.extend(consistency_warnings)
    return ExplorerData(
        runs=runs,
        bundles=bundles,
        source_ids=sorted(bundles),
        player_series=player_series,
        rally_index=rally_index,
        consistency=consistency,
        warnings=warnings,
        shuttle_sidecar=shuttle_sidecar,
        shuttle_sidecar_provenance=shuttle_provenance,
    )
