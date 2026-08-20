"""Tune an offline rally decoder on within-camera validation predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .config import SelectorConfig
from .crossfit import CrossFitFold
from .dataset import SelectorWindow, SelectorWindowDataset
from .experiment import (
    _predict_windows,
    aggregate_joint_predictions,
    compile_metrics,
    load_dataset_config,
    prepare_output_directory,
    select_device,
)
from .joint_inference import (
    InPlayDecoderConfig,
    decode_inplay_probabilities,
    decoded_intervals,
)
from .model import JointRallyShuttleModel
from .rally_intervals import RallyInterval
from .within_camera_experiment import chronological_window_split


THRESHOLDS = tuple(value / 100 for value in range(5, 96, 5))
MAX_GAP_SECONDS = (0.0, 0.1, 0.2, 0.3, 0.5)
MINIMUM_DURATION_SECONDS = (0.0, 0.1, 0.2, 0.3)


def contiguous_row_groups(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    """Group sorted predictions without crossing sources or missing frames."""
    values = sorted(
        rows,
        key=lambda row: (str(row["prediction_source_id"]), int(row["frame"])),
    )
    groups: list[list[Mapping[str, Any]]] = []
    for row in values:
        if (
            not groups
            or str(row["prediction_source_id"])
            != str(groups[-1][-1]["prediction_source_id"])
            or int(row["frame"]) != int(groups[-1][-1]["frame"]) + 1
        ):
            groups.append([])
        groups[-1].append(row)
    return tuple(tuple(group) for group in groups)


def decode_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    fps: float,
    config: InPlayDecoderConfig,
) -> list[dict[str, Any]]:
    """Apply a decoder independently to every contiguous observed range."""
    decoded_rows: list[dict[str, Any]] = []
    for group in contiguous_row_groups(rows):
        thresholded = [
            float(row["in_play_probability"]) >= config.threshold for row in group
        ]
        decoded = decode_inplay_probabilities(
            [float(row["in_play_probability"]) for row in group],
            fps=fps,
            config=config,
        )
        for row, frame_state, decoded_state in zip(group, thresholded, decoded):
            decoded_rows.append(
                {
                    **row,
                    "frame_threshold_inplay": frame_state,
                    "decoded_inplay": decoded_state,
                }
            )
    return decoded_rows


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    intervals = metrics["intervals"] or {}
    return {
        "frame_f1": metrics["inplay"]["f1"],
        "frame_precision": metrics["inplay"]["precision"],
        "frame_recall": metrics["inplay"]["recall"],
        "interval_f1": intervals.get("f1"),
        "matched_count": intervals.get("matched_count", 0),
        "prediction_count": intervals.get("prediction_count", 0),
        "mean_absolute_boundary_error": intervals.get(
            "mean_absolute_boundary_error"
        ),
    }


def calibrate_validation_decoder(
    rows: Sequence[Mapping[str, Any]],
    *,
    fps: float,
) -> tuple[InPlayDecoderConfig, list[dict[str, Any]]]:
    """Select a compact decoder by validation interval quality."""
    candidates: list[tuple[tuple[float, ...], InPlayDecoderConfig, dict[str, Any]]] = []
    for threshold in THRESHOLDS:
        for max_gap in MAX_GAP_SECONDS:
            for minimum in MINIMUM_DURATION_SECONDS:
                config = InPlayDecoderConfig(
                    threshold=threshold,
                    max_gap_seconds=max_gap,
                    minimum_duration_seconds=minimum,
                    preserve_edge_runs=True,
                )
                metrics = compile_metrics(decode_rows(rows, fps=fps, config=config))
                summary = _metric_summary(metrics)
                boundary_error = summary["mean_absolute_boundary_error"]
                score = (
                    float(summary["interval_f1"] or 0.0),
                    float(summary["matched_count"]),
                    -float(boundary_error if boundary_error is not None else math.inf),
                    float(summary["frame_f1"] or 0.0),
                    -max_gap,
                    -minimum,
                    -abs(threshold - 0.5),
                )
                candidates.append(
                    (score, config, {"config": asdict(config), "metrics": summary})
                )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], [item[2] for item in candidates]


def _threshold_rows(
    rows: Iterable[Mapping[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "frame_threshold_inplay": float(row["in_play_probability"]) >= threshold,
            "decoded_inplay": float(row["in_play_probability"]) >= threshold,
        }
        for row in rows
    ]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_intervals(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source_id",
                "rally_id",
                "start_frame",
                "end_frame",
                "open_start",
                "open_end",
            ),
        )
        writer.writeheader()
        counts: dict[str, int] = {}
        for group in contiguous_row_groups(rows):
            source_id = str(group[0]["prediction_source_id"])
            frames = [int(row["frame"]) for row in group]
            states = [bool(row["decoded_inplay"]) for row in group]
            for start, end in decoded_intervals(frames, states):
                counts[source_id] = counts.get(source_id, 0) + 1
                writer.writerow(
                    {
                        "source_id": source_id,
                        "rally_id": f"{source_id}-{counts[source_id]:04d}",
                        "start_frame": start,
                        "end_frame": end,
                        "open_start": start == frames[0],
                        "open_end": end == frames[-1],
                    }
                )


def _interval_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]) + 1)
    union = left[1] - left[0] + 1 + right[1] - right[0] + 1 - intersection
    return intersection / union if union else 0.0


def _longest_low_run(
    frames: Sequence[int], probabilities: Sequence[float], threshold: float
) -> tuple[int | None, int | None, int]:
    best: tuple[int | None, int | None, int] = (None, None, 0)
    start: int | None = None
    for index, probability in enumerate((*probabilities, threshold)):
        if index < len(probabilities) and probability < threshold and start is None:
            start = index
        elif (index == len(probabilities) or probability >= threshold) and start is not None:
            length = index - start
            if length > best[2]:
                best = frames[start], frames[index - 1], length
            start = None
    return best


def build_rally_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    rallies: Sequence[RallyInterval],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Describe probability and interval failures for every observed rally."""
    row_groups = contiguous_row_groups(rows)
    decoded_by_source: dict[str, list[tuple[int, int]]] = {}
    for group in row_groups:
        source_id = str(group[0]["prediction_source_id"])
        decoded_by_source.setdefault(source_id, []).extend(
            decoded_intervals(
                [int(row["frame"]) for row in group],
                [bool(row["decoded_inplay"]) for row in group],
            )
        )
    diagnostics: list[dict[str, Any]] = []
    for rally in rallies:
        source_rows = [
            row
            for row in rows
            if str(row["prediction_source_id"]) == rally.source_id
            and rally.start_frame <= int(row["frame"]) <= rally.end_frame
        ]
        if not source_rows:
            continue
        frames = [int(row["frame"]) for row in source_rows]
        probabilities = [float(row["in_play_probability"]) for row in source_rows]
        truth = rally.start_frame, rally.end_frame
        overlaps = [
            interval
            for interval in decoded_by_source.get(rally.source_id, [])
            if _interval_iou(truth, interval) > 0
        ]
        best = max(overlaps, key=lambda interval: _interval_iou(truth, interval), default=None)
        iou = _interval_iou(truth, best) if best is not None else 0.0
        low_start, low_end, low_count = _longest_low_run(
            frames, probabilities, threshold
        )
        statuses = []
        if iou >= 0.5:
            statuses.append("matched")
        else:
            statuses.append("missed")
        if len(overlaps) > 1:
            statuses.append("split")
        if best is not None and (
            best[0] > rally.start_frame or best[1] < rally.end_frame
        ):
            statuses.append("truncated")
        if best is not None and (
            best[0] < rally.start_frame or best[1] > rally.end_frame
        ):
            statuses.append("overextended")
        diagnostics.append(
            {
                "source_id": rally.source_id,
                "rally_id": rally.rally_id,
                "start_frame": rally.start_frame,
                "end_frame": rally.end_frame,
                "duration_frames": rally.end_frame - rally.start_frame + 1,
                "mean_in_rally_probability": sum(probabilities) / len(probabilities),
                "minimum_in_rally_probability": min(probabilities),
                "longest_internal_low_gap_frames": low_count,
                "low_gap_start_frame": low_start,
                "low_gap_end_frame": low_end,
                "overlapping_prediction_count": len(overlaps),
                "best_prediction_start": best[0] if best else None,
                "best_prediction_end": best[1] if best else None,
                "best_iou": iou,
                "start_error": best[0] - rally.start_frame if best else None,
                "end_error": best[1] - rally.end_frame if best else None,
                "matched": iou >= 0.5,
                "status": "+".join(statuses),
            }
        )
    return sorted(
        diagnostics,
        key=lambda item: (bool(item["matched"]), int(item["start_frame"])),
    )


def _write_diagnostic_table(path: Path, diagnostics: Sequence[Mapping[str, Any]]) -> None:
    if not diagnostics:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)


def _plot_rally_diagnostics(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    context_frames: int = 30,
) -> None:
    if not diagnostics:
        return
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(diagnostics),
        1,
        figsize=(16, max(3.0, 2.6 * len(diagnostics))),
        squeeze=False,
    )
    for axis, diagnostic in zip(axes[:, 0], diagnostics):
        source_id = str(diagnostic["source_id"])
        view_start = int(diagnostic["start_frame"]) - context_frames
        view_end = int(diagnostic["end_frame"]) + context_frames
        panel = [
            row
            for row in rows
            if str(row["prediction_source_id"]) == source_id
            and view_start <= int(row["frame"]) <= view_end
        ]
        frames = [int(row["frame"]) for row in panel]
        probabilities = [float(row["in_play_probability"]) for row in panel]
        axis.plot(frames, probabilities, color="#1f77b4", linewidth=1.4)
        axis.axhline(
            threshold,
            color="#d62728",
            linestyle="--",
            linewidth=1,
            label=f"enter = exit = {threshold:.2f}",
        )
        axis.axvspan(
            int(diagnostic["start_frame"]),
            int(diagnostic["end_frame"]),
            color="#2ca02c",
            alpha=0.14,
            label="ground truth",
        )
        raw_states = [bool(row["frame_threshold_inplay"]) for row in panel]
        decoded_states = [bool(row["decoded_inplay"]) for row in panel]
        for start, end in decoded_intervals(frames, raw_states):
            axis.plot([start, end], [-0.08, -0.08], color="#ff7f0e", linewidth=5)
        for start, end in decoded_intervals(frames, decoded_states):
            axis.plot([start, end], [-0.17, -0.17], color="#9467bd", linewidth=5)
        low_start = diagnostic["low_gap_start_frame"]
        low_end = diagnostic["low_gap_end_frame"]
        if low_start is not None and low_end is not None:
            axis.axvspan(int(low_start), int(low_end), color="#d62728", alpha=0.18)
            axis.annotate(
                f'{diagnostic["longest_internal_low_gap_frames"]}f low gap',
                xy=((int(low_start) + int(low_end)) / 2, threshold),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        predicted = (
            "none"
            if diagnostic["best_prediction_start"] is None
            else f'{diagnostic["best_prediction_start"]}-{diagnostic["best_prediction_end"]}'
        )
        axis.set_title(
            f'{diagnostic["rally_id"]}  GT {diagnostic["start_frame"]}-'
            f'{diagnostic["end_frame"]}  best {predicted}  '
            f'IoU={diagnostic["best_iou"]:.3f}  {diagnostic["status"]}',
            fontsize=10,
            loc="left",
        )
        axis.text(view_start, -0.08, "raw", va="center", ha="right", fontsize=8)
        axis.text(view_start, -0.17, "decoded", va="center", ha="right", fontsize=8)
        axis.set_ylim(-0.24, 1.04)
        axis.set_xlim(frames[0], frames[-1])
        axis.set_ylabel("P(in-play)")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("source-local frame")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _print_diagnostic_summary(diagnostics: Sequence[Mapping[str, Any]]) -> None:
    print(
        "rally duration mean_p min_p low_gap best_prediction IoU "
        "start_err end_err status",
        flush=True,
    )
    for item in diagnostics:
        predicted = (
            "none"
            if item["best_prediction_start"] is None
            else f'{item["best_prediction_start"]}-{item["best_prediction_end"]}'
        )
        print(
            f'{item["rally_id"]} {item["duration_frames"]} '
            f'{item["mean_in_rally_probability"]:.3f} '
            f'{item["minimum_in_rally_probability"]:.3f} '
            f'{item["longest_internal_low_gap_frames"]} {predicted} '
            f'{item["best_iou"]:.3f} {item["start_error"]} '
            f'{item["end_error"]} {item["status"]}',
            flush=True,
        )


def _predict_partition(
    model: JointRallyShuttleModel,
    windows: Sequence[SelectorWindow],
    fold: CrossFitFold,
    *,
    device: torch.device,
    batch_size: int,
    mask_candidates: bool,
) -> list[dict[str, Any]]:
    return aggregate_joint_predictions(
        _predict_windows(
            model,
            windows,
            fold,
            device=device,
            batch_size=batch_size,
            owned_only=True,
            mask_candidates=mask_candidates,
        )
    )


def run_decoder_experiment(
    *,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selector_config = SelectorConfig(**checkpoint["selector_config"])
    pose_mode = str(checkpoint["pose_coordinate_mode"])
    mask_candidates = bool(checkpoint["mask_candidates"])
    dataset_config, _ = load_dataset_config(
        config_path,
        context_mode=selector_config.context_mode,
        pose_coordinate_mode=pose_mode,
    )
    source_order = tuple(checkpoint["split"]["source_order"])
    allowed = set(source_order)
    dataset_config = replace(
        dataset_config,
        sources=tuple(
            source for source in dataset_config.sources if source.source_id in allowed
        ),
    )
    dataset = SelectorWindowDataset(dataset_config)
    if dataset.manifest["dataset_fingerprint"] != checkpoint["dataset_fingerprint"]:
        raise ValueError("checkpoint and decoder dataset fingerprints differ")
    _, validation, test, split = chronological_window_split(
        dataset.windows,
        source_order,
        guard_seconds=float(checkpoint["split"]["guard_seconds"]),
    )
    if split != checkpoint["split"]:
        raise ValueError("checkpoint and reconstructed chronological splits differ")
    fps_values = {float(window.metadata["fps"]) for window in validation + test}
    if len(fps_values) != 1:
        raise ValueError("decoder partitions must share one FPS")
    fps = next(iter(fps_values))
    device = device or select_device()
    model = JointRallyShuttleModel.from_checkpoint(checkpoint).to(device)
    model.eval()
    fold = CrossFitFold("within-camera", source_order, source_order)
    validation_rows = _predict_partition(
        model,
        validation,
        fold,
        device=device,
        batch_size=batch_size,
        mask_candidates=mask_candidates,
    )
    test_rows = _predict_partition(
        model,
        test,
        fold,
        device=device,
        batch_size=batch_size,
        mask_candidates=mask_candidates,
    )
    decoder, candidates = calibrate_validation_decoder(validation_rows, fps=fps)
    validation_decoded = decode_rows(validation_rows, fps=fps, config=decoder)
    test_decoded = decode_rows(test_rows, fps=fps, config=decoder)
    output_dir = prepare_output_directory(output_dir)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    for row in validation_decoded + test_decoded:
        row["checkpoint_sha256"] = checkpoint_sha256
        row["decoder_config"] = asdict(decoder)
    decoder_artifact = {
        "schema": "within_camera_inplay_decoder",
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha256,
        "selection_objective": [
            "interval_f1",
            "matched_count",
            "negative_boundary_mae",
            "frame_f1",
            "smaller_temporal_transforms",
            "threshold_nearest_0.5",
        ],
        "selected_config": asdict(decoder),
        "validation_selected_metrics": _metric_summary(
            compile_metrics(validation_decoded)
        ),
        "candidates": candidates,
    }
    metrics = {
        "schema": "within_camera_inplay_decoded_metrics",
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
        "pose_coordinate_mode": pose_mode,
        "context_mode": selector_config.context_mode,
        "mask_candidates": mask_candidates,
        "fps": fps,
        "split": split,
        "decoder_config": asdict(decoder),
        "validation": {
            "raw_0_5": compile_metrics(_threshold_rows(validation_rows, 0.5)),
            "selected_threshold_only": compile_metrics(
                _threshold_rows(validation_rows, decoder.threshold)
            ),
            "decoded": compile_metrics(validation_decoded),
        },
        "test": {
            "raw_0_5": compile_metrics(_threshold_rows(test_rows, 0.5)),
            "selected_threshold_only": compile_metrics(
                _threshold_rows(test_rows, decoder.threshold)
            ),
            "decoded": compile_metrics(test_decoded),
        },
    }
    (output_dir / "decoder.json").write_text(
        json.dumps(decoder_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_jsonl(output_dir / "validation-predictions.jsonl", validation_decoded)
    _write_jsonl(output_dir / "predictions.jsonl", test_decoded)
    _write_intervals(output_dir / "decoded-rallies.csv", test_decoded)
    if dataset.rally_index is None:
        raise ValueError("rally diagnostics require strict rally intervals")
    diagnostics = build_rally_diagnostics(
        test_decoded,
        dataset.rally_index.intervals,
        threshold=decoder.threshold,
    )
    _write_diagnostic_table(output_dir / "rally-diagnostics.csv", diagnostics)
    _plot_rally_diagnostics(
        output_dir / "rally-segmentation-timeline.png",
        test_decoded,
        diagnostics,
        threshold=decoder.threshold,
    )
    _print_diagnostic_summary(diagnostics)
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = None if args.device == "auto" else torch.device(args.device)
    metrics = run_decoder_experiment(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=device,
    )
    print(json.dumps(metrics["test"]["decoded"]["intervals"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
