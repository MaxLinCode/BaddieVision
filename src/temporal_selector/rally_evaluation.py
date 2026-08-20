"""Leakage-safe splits, metrics, decoder tuning, and paired rally comparison."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from InPlay.heuristic.evaluate import Interval, evaluate as evaluate_intervals

from .joint_inference import InPlayDecoderConfig, decode_inplay_probabilities
from .rally_dataset import RallyWindow


DEFAULT_CAMERA_SOURCES: Mapping[str, tuple[str, ...]] = {
    "Malaysia": (
        "malaysia_max_30s_to_120s",
        "malaysia_max_390s_to_ends",
    ),
    "Max-vs-Nik": (
        "max_vs_nik_30s_to_120s",
        "max_vs_nik_120s_to_ends",
    ),
    "Bothell": ("ff_bothell-seg3",),
}
DECODER_THRESHOLDS = tuple(value / 100 for value in range(5, 96, 5))
DECODER_GAPS_SECONDS = (0.0, 0.1, 0.2, 0.3, 0.5)
DECODER_MINIMUM_SECONDS = (0.0, 0.1, 0.2, 0.3)


@dataclass(frozen=True)
class CameraHeldOutFold:
    fold_id: str
    held_out_camera: str
    training_source_ids: tuple[str, ...]
    validation_source_ids: tuple[str, ...]
    test_source_ids: tuple[str, ...]


def camera_held_out_folds(
    camera_sources: Mapping[str, Sequence[str]] = DEFAULT_CAMERA_SOURCES,
) -> tuple[CameraHeldOutFold, ...]:
    """Return deterministic folds holding out each complete camera once."""
    groups = {
        str(camera): tuple(map(str, sources))
        for camera, sources in camera_sources.items()
    }
    if len(groups) < 2 or any(not sources for sources in groups.values()):
        raise ValueError("camera folds need at least two non-empty cameras")
    flattened = [
        source for sources in groups.values() for source in sources
    ]
    if len(flattened) != len(set(flattened)):
        raise ValueError("each source must belong to exactly one camera")
    return tuple(
        CameraHeldOutFold(
            fold_id=chr(ord("A") + index),
            held_out_camera=camera,
            training_source_ids=tuple(
                source
                for other, sources in groups.items()
                if other != camera
                for source in sources
            ),
            # Validation is drawn chronologically from these same sources.
            validation_source_ids=tuple(
                source
                for other, sources in groups.items()
                if other != camera
                for source in sources
            ),
            test_source_ids=sources,
        )
        for index, (camera, sources) in enumerate(groups.items())
    )


def chronological_camera_train_validation_split(
    windows: Sequence[RallyWindow],
    training_camera_sources: Mapping[str, Sequence[str]],
    *,
    train_fraction: float = 0.8,
    guard_seconds: float = 1.0,
) -> tuple[list[RallyWindow], list[RallyWindow], dict[str, Any]]:
    """Split each training camera in chronology with context containment."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train fraction must be between zero and one")
    if guard_seconds < 0:
        raise ValueError("guard seconds cannot be negative")
    by_source: dict[str, list[RallyWindow]] = {}
    for window in windows:
        by_source.setdefault(window.source_id, []).append(window)

    train: list[RallyWindow] = []
    validation: list[RallyWindow] = []
    camera_metadata: dict[str, Any] = {}
    seen_sources: set[str] = set()
    for camera, raw_sources in training_camera_sources.items():
        sources = tuple(map(str, raw_sources))
        if not sources or any(not by_source.get(source) for source in sources):
            raise ValueError(
                f"training camera {camera!r} has unresolved dense sources"
            )
        if seen_sources.intersection(sources):
            raise ValueError("training camera source groups overlap")
        seen_sources.update(sources)
        frame_counts = {
            source: int(by_source[source][0].metadata["frame_count"])
            for source in sources
        }
        offsets: dict[str, int] = {}
        total_frames = 0
        for source in sources:
            offsets[source] = total_frames
            total_frames += frame_counts[source]
        cut = int(total_frames * train_fraction)
        fps_values = {
            float(window.metadata["fps"])
            for source in sources
            for window in by_source[source]
        }
        if len(fps_values) != 1:
            raise ValueError(
                "sources from one camera must share FPS for guard bands"
            )
        guard_frames = round(next(iter(fps_values)) * guard_seconds)
        train_range = (0, cut - guard_frames - 1)
        validation_range = (cut + guard_frames, total_frames - 1)
        camera_train: list[RallyWindow] = []
        camera_validation: list[RallyWindow] = []
        for source in sources:
            offset = offsets[source]
            for window in by_source[source]:
                first = offset + min(window.frame_indices)
                last = offset + max(window.frame_indices)
                if train_range[0] <= first and last <= train_range[1]:
                    camera_train.append(window)
                elif (
                    validation_range[0] <= first
                    and last <= validation_range[1]
                ):
                    camera_validation.append(window)
        if not camera_train or not camera_validation:
            raise ValueError(
                f"camera {camera!r} produced an empty chronological partition"
            )
        train.extend(camera_train)
        validation.extend(camera_validation)
        camera_metadata[str(camera)] = {
            "source_order": list(sources),
            "source_frame_counts": frame_counts,
            "source_offsets": offsets,
            "total_frames": total_frames,
            "cut_frame": cut,
            "guard_seconds": guard_seconds,
            "guard_frames": guard_frames,
            "train_global_range": list(train_range),
            "validation_global_range": list(validation_range),
            "train_window_ids": [window.window_id for window in camera_train],
            "validation_window_ids": [
                window.window_id for window in camera_validation
            ],
        }
    return train, validation, {
        "strategy": "per-training-camera-chronological-80-20",
        "train_fraction": train_fraction,
        "guard_seconds": guard_seconds,
        "cameras": camera_metadata,
        "train_window_count": len(train),
        "validation_window_count": len(validation),
        "test_labels_used_for_selection": False,
    }


def decode_prediction_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    fps_by_source: Mapping[str, float],
    config: InPlayDecoderConfig,
) -> list[dict[str, Any]]:
    """Decode each source-local contiguous sequence independently."""
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (str(row["source_id"]), int(row["frame"])),
    )
    groups: list[list[dict[str, Any]]] = []
    for row in ordered:
        if (
            not groups
            or row["source_id"] != groups[-1][-1]["source_id"]
            or int(row["frame"]) != int(groups[-1][-1]["frame"]) + 1
        ):
            groups.append([])
        groups[-1].append(row)
    result: list[dict[str, Any]] = []
    for group in groups:
        source = str(group[0]["source_id"])
        if source not in fps_by_source:
            raise ValueError(f"missing FPS for prediction source {source!r}")
        decoded = decode_inplay_probabilities(
            [float(row["inplay_probability"]) for row in group],
            fps=float(fps_by_source[source]),
            config=config,
        )
        result.extend(
            {
                **row,
                "decoded_inplay": bool(state),
                "decoder_configuration": asdict(config),
            }
            for row, state in zip(group, decoded)
        )
    return result


def _binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if not positives or not negatives:
        return float("nan")
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    rank = 1
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (rank + rank + end - index - 1) / 2
        positive_rank_sum += average_rank * sum(
            label == 1 for _, label in ordered[index:end]
        )
        rank += end - index
        index = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def _binary_pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(label == 1 for label in labels)
    if not positives:
        return float("nan")
    ordered = sorted(
        zip(scores, labels), key=lambda item: item[0], reverse=True
    )
    true_positive = false_positive = 0
    area = 0.0
    previous_recall = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        true_positive += sum(
            label == 1 for _, label in ordered[index:end]
        )
        false_positive += sum(
            label == 0 for _, label in ordered[index:end]
        )
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return area


def _calibration(
    labels: Sequence[int], scores: Sequence[float], *, bin_count: int = 10
) -> dict[str, Any]:
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bin_count):
        lower, upper = index / bin_count, (index + 1) / bin_count
        indices = [
            row
            for row, score in enumerate(scores)
            if lower <= score < upper
            or (index == bin_count - 1 and score == upper)
        ]
        if not indices:
            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_probability": None,
                    "positive_rate": None,
                }
            )
            continue
        mean_score = statistics.fmean(scores[row] for row in indices)
        positive_rate = statistics.fmean(labels[row] for row in indices)
        ece += (
            len(indices)
            / len(labels)
            * abs(mean_score - positive_rate)
        )
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(indices),
                "mean_probability": mean_score,
                "positive_rate": positive_rate,
            }
        )
    return {
        "brier_score": statistics.fmean(
            (score - label) ** 2
            for label, score in zip(labels, scores)
        ),
        "expected_calibration_error": ece,
        "bins": bins,
    }


def _intervals_from_rows(
    rows: Sequence[Mapping[str, Any]], field: str
) -> list[Interval]:
    intervals: list[Interval] = []
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_id"]), []).append(row)
    for source, source_rows in by_source.items():
        source_rows.sort(key=lambda row: int(row["frame"]))
        start: int | None = None
        last: int | None = None
        interval_number = 0
        for row in (*source_rows, None):
            if row is None:
                frame = (last + 1) if last is not None else 0
                state = False
            else:
                frame = int(row["frame"])
                state = bool(row[field])
                if last is not None and frame != last + 1 and start is not None:
                    interval_number += 1
                    intervals.append(
                        Interval(
                            source,
                            f"{field}-{interval_number:04d}",
                            start,
                            last,
                        )
                    )
                    start = None
            if state and start is None:
                start = frame
            elif not state and start is not None:
                interval_number += 1
                intervals.append(
                    Interval(
                        source,
                        f"{field}-{interval_number:04d}",
                        start,
                        last if last is not None else frame - 1,
                    )
                )
                start = None
            last = frame
    return intervals


def rally_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute frame, calibration, interval, and boundary metrics."""
    values = [
        dict(row)
        for row in rows
        if int(row.get("inplay_target", -1)) in (0, 1)
    ]
    if not values:
        raise ValueError("rally metrics require binary labeled rows")
    labels = [int(row["inplay_target"]) for row in values]
    scores = [float(row["inplay_probability"]) for row in values]
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("rally probabilities must be finite and in [0, 1]")
    decoder_thresholds = {
        float(row["decoder_configuration"]["threshold"])
        for row in values
        if isinstance(row.get("decoder_configuration"), Mapping)
        and "threshold" in row["decoder_configuration"]
    }
    if len(decoder_thresholds) > 1:
        raise ValueError("metric rows mix decoder thresholds")
    threshold = next(iter(decoder_thresholds), 0.5)
    raw_predicted = [score >= threshold for score in scores]
    decoded_predicted = [
        bool(row.get("decoded_inplay", score >= threshold))
        for row, score in zip(values, scores)
    ]

    def classification(states: Sequence[bool]) -> dict[str, Any]:
        tp = sum(
            label == 1 and state for label, state in zip(labels, states)
        )
        fp = sum(
            label == 0 and state for label, state in zip(labels, states)
        )
        fn = sum(
            label == 1 and not state for label, state in zip(labels, states)
        )
        tn = sum(
            label == 0 and not state for label, state in zip(labels, states)
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        true_negative_rate = tn / (tn + fp) if tn + fp else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": (recall + true_negative_rate) / 2,
            "predicted_positive_rate": sum(states) / len(states),
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        }

    raw_classification = classification(raw_predicted)
    decoded_classification = classification(decoded_predicted)
    epsilon = 1e-7
    bce = statistics.fmean(
        -label * math.log(min(1 - epsilon, max(epsilon, score)))
        - (1 - label)
        * math.log(min(1 - epsilon, max(epsilon, 1 - score)))
        for label, score in zip(labels, scores)
    )
    interval_rows = [
        {
            **row,
            "truth_state": bool(int(row["inplay_target"])),
            "prediction_state": state,
        }
        for row, state in zip(values, decoded_predicted)
    ]
    truth_intervals = _intervals_from_rows(interval_rows, "truth_state")
    predicted_intervals = _intervals_from_rows(
        interval_rows, "prediction_state"
    )
    interval_metrics, matches = evaluate_intervals(
        predicted_intervals, truth_intervals, threshold=0.5
    )
    overlap_counts = {
        label.rally_id: sum(
            prediction.source_id == label.source_id
            and min(prediction.end, label.end)
            >= max(prediction.start, label.start)
            for prediction in predicted_intervals
        )
        for label in truth_intervals
    }
    fragment_counts = [max(0, count - 1) for count in overlap_counts.values()]
    return {
        "frame_count": len(values),
        "positive_rate": sum(labels) / len(labels),
        "raw_bce": bce,
        "roc_auc": _binary_roc_auc(labels, scores),
        "pr_auc": _binary_pr_auc(labels, scores),
        "threshold": threshold,
        **raw_classification,
        "raw_frame": raw_classification,
        "decoded_frame": decoded_classification,
        "calibration": _calibration(labels, scores),
        "intervals": {
            **interval_metrics,
            "fragmentation_count": sum(fragment_counts),
            "mean_fragments_per_true_interval": (
                statistics.fmean(overlap_counts.values())
                if overlap_counts
                else 0.0
            ),
        },
        "interval_matches": matches,
    }


def metrics_by_source(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    sources = sorted({str(row["source_id"]) for row in rows})
    return {
        source: rally_metrics(
            [row for row in rows if str(row["source_id"]) == source]
        )
        for source in sources
    }


def tune_rally_decoder(
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    fps_by_source: Mapping[str, float],
) -> tuple[InPlayDecoderConfig, list[dict[str, Any]]]:
    """Tune threshold/gap/minimum duration using validation labels only."""
    if not validation_rows:
        raise ValueError("decoder tuning requires validation predictions")
    candidates: list[
        tuple[tuple[float, ...], InPlayDecoderConfig, dict[str, Any]]
    ] = []
    for threshold in DECODER_THRESHOLDS:
        for gap in DECODER_GAPS_SECONDS:
            for minimum in DECODER_MINIMUM_SECONDS:
                config = InPlayDecoderConfig(
                    threshold=threshold,
                    max_gap_seconds=gap,
                    minimum_duration_seconds=minimum,
                    preserve_edge_runs=True,
                )
                decoded = decode_prediction_rows(
                    validation_rows,
                    fps_by_source=fps_by_source,
                    config=config,
                )
                metrics = rally_metrics(decoded)
                intervals = metrics["intervals"]
                boundary = intervals["mean_absolute_boundary_error"]
                score = (
                    float(intervals["f1"]),
                    -float(boundary if boundary is not None else math.inf),
                    float(metrics["f1"]),
                    -gap,
                    -minimum,
                    -abs(threshold - 0.5),
                )
                candidates.append(
                    (
                        score,
                        config,
                        {
                            "configuration": asdict(config),
                            "validation_interval_f1": intervals["f1"],
                            "validation_boundary_mae": boundary,
                            "validation_frame_f1": metrics["f1"],
                        },
                    )
                )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], [candidate[2] for candidate in candidates]


def _paired_cycle_values(
    clean_rows: Sequence[Mapping[str, Any]],
    legacy_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[tuple[int, float]]]:
    clean = {
        (str(row["source_id"]), int(row["frame"])): row for row in clean_rows
    }
    legacy = {
        (str(row["source_id"]), int(row["frame"])): row for row in legacy_rows
    }
    if set(clean) != set(legacy):
        raise ValueError("paired bootstrap needs identical source/frame rows")
    epsilon = 1e-7
    output: dict[str, list[tuple[int, float]]] = {}
    by_source: dict[str, list[tuple[int, Mapping[str, Any], Mapping[str, Any]]]] = {}
    for key in sorted(clean):
        by_source.setdefault(key[0], []).append(
            (key[1], clean[key], legacy[key])
        )
    for source, values in by_source.items():
        positive_runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, (_, clean_row, _) in enumerate((*values, (0, {}, {}))):
            state = (
                int(clean_row["inplay_target"]) == 1
                if index < len(values)
                else False
            )
            if state and start is None:
                start = index
            elif not state and start is not None:
                positive_runs.append((start, index - 1))
                start = None
        if not positive_runs:
            positive_runs = [(0, len(values) - 1)]
        boundaries = [0]
        for left, right in zip(positive_runs, positive_runs[1:]):
            boundaries.append((left[1] + right[0] + 1) // 2)
        boundaries.append(len(values))
        cycles: list[tuple[int, float]] = []
        for cycle_index in range(len(boundaries) - 1):
            cycle = values[boundaries[cycle_index] : boundaries[cycle_index + 1]]
            differences = []
            for _, clean_row, legacy_row in cycle:
                label = int(clean_row["inplay_target"])
                if label not in (0, 1) or label != int(
                    legacy_row["inplay_target"]
                ):
                    raise ValueError("paired bootstrap labels are not aligned")
                clean_probability = min(
                    1 - epsilon,
                    max(epsilon, float(clean_row["inplay_probability"])),
                )
                legacy_probability = min(
                    1 - epsilon,
                    max(epsilon, float(legacy_row["inplay_probability"])),
                )
                clean_loss = -label * math.log(clean_probability) - (
                    1 - label
                ) * math.log(1 - clean_probability)
                legacy_loss = -label * math.log(legacy_probability) - (
                    1 - label
                ) * math.log(1 - legacy_probability)
                differences.append(legacy_loss - clean_loss)
            if differences:
                cycles.append((len(differences), statistics.fmean(differences)))
        output[source] = cycles
    return output


def paired_bce_bootstrap(
    clean_rows: Sequence[Mapping[str, Any]],
    legacy_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 1729,
) -> dict[str, Any]:
    """Bootstrap legacy-minus-clean BCE over source-local rally cycles."""
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    strata = _paired_cycle_values(clean_rows, legacy_rows)
    if any(not cycles for cycles in strata.values()):
        raise ValueError("every source bootstrap stratum needs a rally cycle")
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        numerator = denominator = 0.0
        for cycles in strata.values():
            for _ in range(len(cycles)):
                frame_count, improvement = cycles[
                    generator.randrange(len(cycles))
                ]
                numerator += frame_count * improvement
                denominator += frame_count
        estimates.append(numerator / denominator)
    estimates.sort()

    def percentile(probability: float) -> float:
        position = probability * (len(estimates) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return estimates[lower]
        fraction = position - lower
        return estimates[lower] * (1 - fraction) + estimates[upper] * fraction

    return {
        "estimand": "legacy_bce_minus_clean_bce",
        "seed": seed,
        "samples": samples,
        "stratification": "source_local_rally_cycles",
        "source_cycle_counts": {
            source: len(cycles) for source, cycles in strata.items()
        },
        "mean_improvement": statistics.fmean(estimates),
        "confidence_interval_95": [percentile(0.025), percentile(0.975)],
    }


def independent_model_acceptance(
    clean_by_camera: Mapping[str, Mapping[str, Any]],
    legacy_by_camera: Mapping[str, Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply all predeclared clean-model acceptance guardrails."""
    if set(clean_by_camera) != set(legacy_by_camera):
        raise ValueError("camera metrics are not paired")
    cameras = sorted(clean_by_camera)
    interval_changes = {
        camera: float(clean_by_camera[camera]["intervals"]["f1"])
        - float(legacy_by_camera[camera]["intervals"]["f1"])
        for camera in cameras
    }
    boundary_changes = {
        camera: (
            float(
                clean_by_camera[camera]["intervals"][
                    "mean_absolute_boundary_error"
                ]
            )
            - float(
                legacy_by_camera[camera]["intervals"][
                    "mean_absolute_boundary_error"
                ]
            )
            if clean_by_camera[camera]["intervals"][
                "mean_absolute_boundary_error"
            ]
            is not None
            and legacy_by_camera[camera]["intervals"][
                "mean_absolute_boundary_error"
            ]
            is not None
            else math.inf
        )
        for camera in cameras
    }
    checks = {
        "bootstrap_interval_excludes_zero": float(
            bootstrap["confidence_interval_95"][0]
        )
        > 0,
        "every_camera_bce_improves": all(
            float(clean_by_camera[camera]["raw_bce"])
            < float(legacy_by_camera[camera]["raw_bce"])
            for camera in cameras
        ),
        "macro_interval_f1_does_not_regress": statistics.fmean(
            float(clean_by_camera[camera]["intervals"]["f1"])
            for camera in cameras
        )
        >= statistics.fmean(
            float(legacy_by_camera[camera]["intervals"]["f1"])
            for camera in cameras
        ),
        "no_camera_interval_f1_loses_more_than_0_05": all(
            change >= -0.05 for change in interval_changes.values()
        ),
        "no_camera_boundary_mae_worsens_more_than_5_frames": all(
            change <= 5.0 for change in boundary_changes.values()
        ),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "per_camera_interval_f1_change": interval_changes,
        "per_camera_boundary_mae_change_frames": boundary_changes,
    }


def stable_track_fusion_acceptance(
    track_macro: Mapping[str, Any],
    clean_macro: Mapping[str, Any],
    availability_macro: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    *,
    interval_and_boundary_guardrails_satisfied: bool,
) -> dict[str, Any]:
    """Gate the optional shuttle-track follow-up without changing TrackNet."""
    improvements = {
        "over_clean": float(clean_macro["raw_bce"])
        - float(track_macro["raw_bce"]),
        "over_availability": float(availability_macro["raw_bce"])
        - float(track_macro["raw_bce"]),
    }
    checks = {
        "beats_both_controls_by_0_01_macro_bce": all(
            value >= 0.01 for value in improvements.values()
        ),
        "paired_bce_interval_excludes_zero": float(
            bootstrap["confidence_interval_95"][0]
        )
        > 0,
        "interval_and_boundary_guardrails_satisfied": bool(
            interval_and_boundary_guardrails_satisfied
        ),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "macro_bce_improvements": improvements,
    }
