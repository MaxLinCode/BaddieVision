"""Leakage-safe oracle-track audit and frozen-checkpoint corruption utilities.

This is an experiment-only adapter.  It deliberately preserves the selector's
12-wide candidate tensor contract while defining a much smaller semantic
schema: only values 0 and 1 contain normalized image x/y.  Every real frame has
exactly one token, including frames where the shuttle was not observed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.annotation_platform.events import AnnotationEvent

from .batch import MASKED_TARGET
from .config import SelectorConfig
from .dataset import FRAME_DIMS, SelectorWindow, SelectorWindowDataset
from .experiment import load_dataset_config, prepare_output_directory, select_device
from .partial_oracle_experiment import (
    MATCHED_ANNOTATION_SHA256,
    MATCHED_FRAME_RANGE,
    MATCHED_SPLIT_FRAME,
    _active_events,
    _contiguous_runs,
    _eligible_windows,
    _negative_split_frame,
    _run_variant,
)
from .within_camera_experiment import DEFAULT_SOURCE_ORDER


SCHEMA = "oracle_track_xy_v1"
VARIANTS = (
    "neutral_track",
    "observation_only",
    "corrected_coordinates",
    "shuffled_coordinates",
    "legacy_selected_only_mask",
)
OBSERVED_KINDS = frozenset({"selected", "missing_proposal"})
UNOBSERVED_KINDS = frozenset(
    {"occluded_inferable", "no_in_frame_target", "no_shuttle", "unsure"}
)


def _canonical_xy(event: AnnotationEvent | None) -> tuple[float, float] | None:
    if event is None or event.label_kind not in OBSERVED_KINDS:
        return None
    position = event.candidate_position
    if not isinstance(position, Mapping):
        return None
    point = position.get("peak_position_normalized")
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    x, y = float(point[0]), float(point[1])
    return (x, y) if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 else None


def _partition_coordinates(
    windows: Sequence[SelectorWindow],
    events: Mapping[tuple[str, int], AnnotationEvent],
) -> dict[tuple[str, int], tuple[float, float]]:
    keys = {
        (window.source_id, int(frame))
        for window in windows
        for frame in window.frame_indices
    }
    return {
        key: xy
        for key in sorted(keys)
        if (xy := _canonical_xy(events.get(key))) is not None
    }


def shuffled_coordinate_map(
    coordinates: Mapping[tuple[str, int], tuple[float, float]], seed: int
) -> dict[tuple[str, int], tuple[float, float]]:
    """Permute positions without consulting rally labels or changing missingness."""
    keys = sorted(coordinates)
    values = [coordinates[key] for key in keys]
    random.Random(seed).shuffle(values)
    return dict(zip(keys, values))


def oracle_track_windows(
    windows: Sequence[SelectorWindow],
    events: Mapping[tuple[str, int], AnnotationEvent],
    *,
    variant: str,
    shuffle_seed: int | None = None,
) -> list[SelectorWindow]:
    """Return windows with one forced-routed synthetic token for every frame."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown oracle-track variant: {variant}")
    coordinates = _partition_coordinates(windows, events)
    if variant == "shuffled_coordinates":
        if shuffle_seed is None:
            raise ValueError("shuffled coordinates require a seed")
        coordinates = shuffled_coordinate_map(coordinates, shuffle_seed)

    output: list[SelectorWindow] = []
    for window in windows:
        values = torch.zeros((len(window.frame_indices), 12), dtype=torch.float32)
        validity = torch.zeros_like(values, dtype=torch.bool)
        statuses: list[str] = []
        for local, frame in enumerate(window.frame_indices):
            key = (window.source_id, int(frame))
            event = events.get(key)
            xy = coordinates.get(key)
            observed = xy is not None
            if variant == "legacy_selected_only_mask":
                observed = observed and event is not None and event.label_kind == "selected"
            if variant in {"neutral_track", "observation_only"}:
                xy = None
            if variant == "neutral_track":
                observed = False
            if xy is not None and variant not in {"neutral_track", "observation_only"}:
                values[local, :2] = torch.tensor(xy)
            if observed:
                # Validity is the explicit observation channel.  All unused
                # compatibility fields remain invalid and therefore inert.
                validity[local, :2] = True
            statuses.append(event.label_kind if event is not None else "unavailable")
        output.append(
            replace(
                window,
                candidate_values=values,
                candidate_validity=validity,
                candidate_frame_indices=torch.arange(len(window.frame_indices)),
                candidate_ids=tuple(
                    f"{SCHEMA}:{window.source_id}:{frame}"
                    for frame in window.frame_indices
                ),
                targets=torch.full_like(window.targets, MASKED_TARGET),
                target_status=tuple(f"oracle_track:{status}" for status in statuses),
                metadata={
                    **window.metadata,
                    "candidate_input_mode": SCHEMA,
                    "oracle_track_variant": variant,
                },
            )
        )
    return output


def _frame_records(
    windows: Sequence[SelectorWindow],
    events: Mapping[tuple[str, int], AnnotationEvent],
) -> dict[tuple[str, int], tuple[str, int]]:
    records: dict[tuple[str, int], tuple[str, int]] = {}
    for window in windows:
        for local, frame in enumerate(window.frame_indices):
            key = (window.source_id, int(frame))
            target = int(window.inplay_targets[local]) if window.inplay_targets is not None else MASKED_TARGET
            status = events[key].label_kind if key in events else "unavailable"
            previous = records.setdefault(key, (status, target))
            if previous != (status, target):
                raise ValueError(f"inconsistent overlapping window data for {key}")
    return records


def label_status_contingency(
    windows: Sequence[SelectorWindow],
    events: Mapping[tuple[str, int], AnnotationEvent],
) -> dict[str, Any]:
    records = _frame_records(windows, events)
    counts = Counter((status, target) for status, target in records.values())
    by_state: dict[str, dict[str, int]] = defaultdict(dict)
    for (status, target), count in sorted(counts.items()):
        by_state[str(target)][status] = count
    rates = {}
    for target in (0, 1):
        rows = [(status, value) for status, value in records.values() if value == target]
        observed = sum(status in OBSERVED_KINDS and _canonical_xy(events.get(key)) is not None
                       for key, (status, value) in records.items() if value == target)
        rates[str(target)] = observed / len(rows) if rows else None
    return {
        "frame_count": len(records),
        "counts_by_inplay_target": dict(by_state),
        "observed_rate_by_inplay_target": rates,
    }


def windows_fingerprint(windows: Sequence[SelectorWindow]) -> str:
    digest = hashlib.sha256()
    for window in windows:
        digest.update(window.source_id.encode())
        digest.update(json.dumps(window.frame_indices).encode())
        digest.update(window.candidate_values.numpy().tobytes())
        digest.update(window.candidate_validity.numpy().tobytes())
        digest.update(window.candidate_frame_indices.numpy().tobytes())
    return digest.hexdigest()


def _unique_frame_data(
    windows: Sequence[SelectorWindow],
) -> dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]]:
    result: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for window in windows:
        if len(window.candidate_ids) != len(window.frame_indices):
            raise ValueError("oracle-track windows must contain exactly one token per frame")
        for local, frame in enumerate(window.frame_indices):
            key = (window.source_id, int(frame))
            value = (window.candidate_values[local].clone(), window.candidate_validity[local].clone())
            if key in result and (
                not torch.equal(result[key][0], value[0])
                or not torch.equal(result[key][1], value[1])
            ):
                raise ValueError(f"overlapping oracle-track windows disagree at {key}")
            result[key] = value
    return result


def _span_mask(
    eligible: Sequence[tuple[str, int]],
    *,
    fraction: float,
    lengths: tuple[int, int],
    seed: int,
) -> set[tuple[str, int]]:
    """Choose state-blind contiguous spans until the requested frame count."""
    if not 0.0 <= fraction <= 1.0 or lengths[0] <= 0 or lengths[1] < lengths[0]:
        raise ValueError("invalid corruption fraction or span lengths")
    target = round(len(eligible) * fraction)
    if target == 0:
        return set()
    allowed = set(eligible)
    starts = list(eligible)
    rng = random.Random(seed)
    rng.shuffle(starts)
    selected: set[tuple[str, int]] = set()
    for source, start in starts:
        remaining = target - len(selected)
        if remaining <= 0:
            break
        length = min(rng.randint(*lengths), remaining)
        span = {(source, start + offset) for offset in range(length)}
        if span <= allowed and not span & selected:
            selected.update(span)
    if len(selected) != target:
        # Deterministic singletons finish fragmented edge cases; the main span
        # lengths are retained in provenance so these can be surfaced.
        selected.update(key for key in eligible if key not in selected for _ in range(1) if len(selected) < target)
    return selected


def _distractor_coordinates(
    raw_windows: Sequence[SelectorWindow],
) -> dict[tuple[str, int], tuple[float, float, str]]:
    choices: dict[tuple[str, int], tuple[float, float, str]] = {}
    for window in raw_windows:
        for slot, local in enumerate(window.candidate_frame_indices.tolist()):
            if not bool(window.candidate_validity[slot, 2:4].all()):
                continue
            key = (window.source_id, int(window.frame_indices[int(local)]))
            if key not in choices:
                choices[key] = (
                    float(window.candidate_values[slot, 2]),
                    float(window.candidate_values[slot, 3]),
                    window.candidate_ids[slot],
                )
    return choices


def corrupt_oracle_track_windows(
    windows: Sequence[SelectorWindow],
    *,
    family: str,
    seed: int,
    amount: float | int,
    image_size: tuple[int, int] | None = None,
    gap_lengths: tuple[int, int] | None = None,
    raw_windows: Sequence[SelectorWindow] | None = None,
) -> list[SelectorWindow]:
    """Apply a deterministic corruption without mutating source windows.

    Supported families are ``gaussian``, ``gap``, ``distractor``, ``lag``, and
    the explicitly label-targeted ``boundary_stress``.  ``amount`` is pixels,
    missing/replacement fraction, lag frames, or early-termination frames,
    respectively.
    """
    if family not in {"gaussian", "gap", "distractor", "lag", "boundary_stress"}:
        raise ValueError(f"unsupported corruption family: {family}")
    source = _unique_frame_data(windows)
    transformed = {key: (value.clone(), valid.clone()) for key, (value, valid) in source.items()}
    observed = sorted(key for key, (_, valid) in source.items() if bool(valid[:2].all()))
    provenance: dict[str, Any] = {"family": family, "amount": amount, "seed": seed, "state_blind": family != "boundary_stress"}
    if float(amount) == 0.0:
        affected: set[tuple[str, int]] = set()
    elif family == "gaussian":
        if image_size is None or min(image_size) <= 0:
            raise ValueError("Gaussian pixel-equivalent noise requires image_size")
        rng = random.Random(seed)
        width, height = image_size
        affected = set(observed)
        for key in observed:
            value, valid = transformed[key]
            value[0] = min(1.0, max(0.0, float(value[0]) + rng.gauss(0.0, float(amount) / width)))
            value[1] = min(1.0, max(0.0, float(value[1]) + rng.gauss(0.0, float(amount) / height)))
    elif family == "gap":
        lengths = gap_lengths or (1, 3)
        affected = _span_mask(observed, fraction=float(amount), lengths=lengths, seed=seed)
        provenance["span_lengths"] = list(lengths)
        for key in affected:
            transformed[key] = (torch.zeros(12), torch.zeros(12, dtype=torch.bool))
    elif family == "distractor":
        if raw_windows is None:
            raise ValueError("distractor corruption requires frozen raw candidate windows")
        distractors = _distractor_coordinates(raw_windows)
        eligible = sorted(set(observed) & set(distractors))
        affected = _span_mask(eligible, fraction=float(amount), lengths=(3, 10), seed=seed)
        provenance["candidate_ids"] = {}
        for key in affected:
            x, y, candidate_id = distractors[key]
            transformed[key][0][:2] = torch.tensor([x, y])
            provenance["candidate_ids"][f"{key[0]}:{key[1]}"] = candidate_id
    elif family == "lag":
        lag = int(amount)
        if lag < 0:
            raise ValueError("lag cannot be negative")
        affected = set()
        for source_id, frame in sorted(source):
            prior = (source_id, frame - lag)
            if prior in source:
                transformed[(source_id, frame)] = (source[prior][0].clone(), source[prior][1].clone())
                affected.add((source_id, frame))
            else:
                transformed[(source_id, frame)] = (torch.zeros(12), torch.zeros(12, dtype=torch.bool))
                affected.add((source_id, frame))
    else:
        early = int(amount)
        affected = set()
        records = _frame_records(windows, {})
        by_source: dict[str, list[int]] = defaultdict(list)
        for (source_id, frame), (_, target) in records.items():
            if target == 1:
                by_source[source_id].append(frame)
        for source_id, frames in by_source.items():
            runs = _contiguous_runs(frames)
            for _, end in runs:
                for frame in range(max(0, end - early + 1), end + 1):
                    key = (source_id, frame)
                    if key in transformed:
                        transformed[key] = (torch.zeros(12), torch.zeros(12, dtype=torch.bool))
                        affected.add(key)
    provenance["affected_frame_count"] = len(affected)
    output: list[SelectorWindow] = []
    for window in windows:
        values = torch.stack([transformed[(window.source_id, int(frame))][0] for frame in window.frame_indices])
        validity = torch.stack([transformed[(window.source_id, int(frame))][1] for frame in window.frame_indices])
        output.append(replace(window, candidate_values=values, candidate_validity=validity, metadata={**window.metadata, "oracle_track_corruption": copy.deepcopy(provenance)}))
    return output


def _metric_value(metrics: Mapping[str, Any], *path: str) -> float | None:
    value: Any = metrics
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return None if value is None else float(value)


def _paired_summary(runs: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    paths = {
        "bce": ("raw_0_5", "mean_inplay_loss"),
        "frame_f1": ("decoder_metrics", "inplay", "f1"),
        "interval_f1": ("decoder_metrics", "intervals", "f1"),
        "boundary_mae": ("decoder_metrics", "intervals", "mean_absolute_boundary_error"),
        "start_mae": ("decoder_metrics", "intervals", "mean_start_boundary_error"),
        "end_mae": ("decoder_metrics", "intervals", "mean_end_boundary_error"),
        "false_splits": ("decoder_metrics", "intervals", "false_split_count"),
        "false_merges": ("decoder_metrics", "intervals", "false_merge_count"),
    }
    summary: dict[str, Any] = {}
    for variant, values in runs.items():
        summary[variant] = {}
        for name, path in paths.items():
            samples = [value for run in values if (value := _metric_value(run, *path)) is not None]
            summary[variant][name] = {
                "values": samples,
                "mean": sum(samples) / len(samples) if samples else None,
                "range": [min(samples), max(samples)] if samples else None,
            }
    corrected = summary.get("corrected_coordinates", {})
    controls = ("neutral_track", "observation_only", "shuffled_coordinates")
    # Strict seed-range separation is intentionally conservative and directly
    # encodes "beyond seed variation" without a post-hoc significance test.
    bce_pass = all(
        corrected["bce"]["range"] and summary[c]["bce"]["range"]
        and corrected["bce"]["range"][1] < summary[c]["bce"]["range"][0]
        for c in controls
    )
    f1_pass = all(
        corrected["frame_f1"]["range"] and summary[c]["frame_f1"]["range"]
        and corrected["frame_f1"]["range"][0] > summary[c]["frame_f1"]["range"][1]
        for c in controls
    )
    return {
        "variants": summary,
        "trajectory_gate_passed": bool(bce_pass or f1_pass),
        "gate_rule": "corrected coordinates must strictly clear every control seed range on BCE or frame F1",
        "classification_if_failed": (
            "encoder/player-context effect with label-status leakage; no evidence for shuttle dynamics"
        ),
    }


def run_oracle_track_audit(
    *,
    config_path: Path,
    baseline_validation_path: Path,
    output_dir: Path,
    seeds: Sequence[int] = (1729, 1730, 1731, 1732, 1733),
    epochs: int = 25,
    batch_size: int = 16,
    device: torch.device | None = None,
    require_matched_snapshot: bool = True,
) -> dict[str, Any]:
    if require_matched_snapshot and (epochs, batch_size, len(seeds)) != (25, 16, 5):
        raise ValueError("matched audit requires 25 epochs, batch size 16, and five seeds")
    device = device or select_device()
    if require_matched_snapshot and device.type != "mps":
        raise ValueError("matched audit requires host MPS; use --allow-unmatched-snapshot for smoke tests")
    output_dir = prepare_output_directory(output_dir)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    annotations = Path(raw_config["dataset"]["annotations_path"])
    if not annotations.is_absolute():
        annotations = (config_path.parent / annotations).resolve()
    snapshot = output_dir / "annotations.snapshot.jsonl"
    snapshot.write_bytes(annotations.read_bytes())
    annotation_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if require_matched_snapshot and annotation_sha != MATCHED_ANNOTATION_SHA256:
        raise ValueError(f"matched annotation hash required: {MATCHED_ANNOTATION_SHA256}; got {annotation_sha}")
    events = _active_events(snapshot)
    config, _ = load_dataset_config(config_path, context_mode="full_context", pose_coordinate_mode="player_relative")
    config = replace(
        config,
        sources=tuple(s for s in config.sources if s.source_id in set(DEFAULT_SOURCE_ORDER)),
        annotations_path=snapshot,
        expected_annotation_sha256=annotation_sha,
    )
    dataset = SelectorWindowDataset(config)
    baseline_rows = [json.loads(line) for line in baseline_validation_path.read_text().splitlines() if line.strip()]
    source_id = Counter(str(row["prediction_source_id"]) for row in baseline_rows).most_common(1)[0][0]
    baseline_frames = {int(row["frame"]) for row in baseline_rows if row["prediction_source_id"] == source_id}
    compatible = {frame for event_source, frame in events if event_source == source_id and frame in baseline_frames}
    runs = _contiguous_runs(compatible)
    if not runs:
        raise ValueError("baseline validation has no compatible continuous annotation run")
    run_start, run_end = max(runs, key=lambda pair: pair[1] - pair[0])
    eligible = _eligible_windows(dataset.windows, source_id=source_id, annotated_frames=set(range(run_start, run_end + 1)))
    split_frame = _negative_split_frame(dataset, source_id, run_start, run_end)
    if require_matched_snapshot and ((run_start, run_end) != MATCHED_FRAME_RANGE or split_frame != MATCHED_SPLIT_FRAME):
        raise ValueError("matched frame range or chronological split drifted")
    partitions = {
        "train": [w for w in eligible if max(w.frame_indices) < split_frame],
        "validation": [w for w in eligible if min(w.frame_indices) > split_frame],
    }
    if any(not values for values in partitions.values()):
        raise ValueError("oracle-track split produced an empty partition")
    model_config = SelectorConfig(context_mode="full_context", frame_feature_dim=FRAME_DIMS["full_context"])
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    fingerprints: dict[str, dict[str, str]] = {}
    for seed in seeds:
        seed_dir = output_dir / f"seed-{seed}"
        seed_dir.mkdir()
        for variant in VARIANTS:
            train = oracle_track_windows(partitions["train"], events, variant=variant, shuffle_seed=seed)
            validation = oracle_track_windows(partitions["validation"], events, variant=variant, shuffle_seed=seed)
            fingerprints.setdefault(variant, {})[str(seed)] = hashlib.sha256(
                (windows_fingerprint(train) + windows_fingerprint(validation)).encode()
            ).hexdigest()
            results[variant].append(_run_variant(
                name=variant, train=train, validation=validation,
                model_config=model_config, output_dir=seed_dir, device=device,
                epochs=epochs, batch_size=batch_size, seed=seed, oracle=True,
                conditioning_mode="hard_isolated", force_candidate_routing=True,
                candidate_input_mode=SCHEMA,
            ))
    paired = _paired_summary(results)
    summary = {
        "schema": "oracle_track_audit",
        "schema_version": 1,
        "track_schema": {
            "name": SCHEMA,
            "coordinate_space": "normalized_image_xy",
            "value_layout": ["x", "y", "reserved_0", "reserved_1", "reserved_2", "reserved_3", "reserved_4", "reserved_5", "reserved_6", "reserved_7", "reserved_8", "reserved_9"],
            "observed_rule": "selected or missing_proposal with canonical clicked position",
            "unobserved_rule": "zero coordinates and false x/y validity",
            "tokens_per_frame": 1,
        },
        "annotation_snapshot_sha256": annotation_sha,
        "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
        "source_id": source_id,
        "split": {"strategy": "chronological annotated run", "run": [run_start, run_end], "split_frame": split_frame, "train_windows": len(partitions["train"]), "validation_windows": len(partitions["validation"])},
        "seeds": list(seeds), "epochs": epochs, "batch_size": batch_size,
        "device": str(device), "conditioning_mode": "hard_isolated",
        "force_candidate_routing": True, "sampling_mode": "boundary_balanced",
        "boundary_aux_weight": 0.25, "offline_bidirectional": True,
        "contingencies": {name: label_status_contingency(values, events) for name, values in partitions.items()},
        "variant_fingerprints": fingerprints,
        "paired_comparison": paired,
        "next_stage": "multi_camera_annotation" if paired["trajectory_gate_passed"] else "stop_no_trajectory_evidence",
        "runs": results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 1730, 1731, 1732, 1733])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--allow-unmatched-snapshot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = None if args.device == "auto" else torch.device(args.device)
    result = run_oracle_track_audit(
        config_path=args.config, baseline_validation_path=args.baseline_validation,
        output_dir=args.output_dir, seeds=args.seeds, epochs=args.epochs,
        batch_size=args.batch_size, device=device,
        require_matched_snapshot=not args.allow_unmatched_snapshot,
    )
    print(json.dumps(result["paired_comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
