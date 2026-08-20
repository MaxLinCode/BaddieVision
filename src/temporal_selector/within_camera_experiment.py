"""Leakage-safe chronological learning diagnostic for one camera sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .config import ContextMode, SelectorConfig
from .crossfit import CrossFitFold
from .dataset import (
    FRAME_DIMS,
    SelectorWindow,
    SelectorWindowDataset,
)
from .experiment import (
    _predict_windows,
    _seed_everything,
    _train_fold,
    _heldout_boundary_epoch_metrics,
    _write_epoch_diagnostic_plot,
    aggregate_joint_predictions,
    compile_metrics,
    load_dataset_config,
    prepare_output_directory,
    select_device,
)
from .model import JointRallyShuttleModel


DEFAULT_SOURCE_ORDER = (
    "max_vs_nik_30s_to_120s",
    "max_vs_nik_120s_to_ends",
)


def chronological_window_split(
    windows: Sequence[SelectorWindow],
    source_order: Sequence[str],
    *,
    guard_seconds: float = 1.0,
) -> tuple[list[SelectorWindow], list[SelectorWindow], list[SelectorWindow], Mapping[str, Any]]:
    """Split 60/20/20 by parent chronology without merging source-local frames."""
    if guard_seconds < 0:
        raise ValueError("guard seconds must be non-negative")
    grouped = {source_id: [] for source_id in source_order}
    for window in windows:
        if window.source_id in grouped:
            grouped[window.source_id].append(window)
    if any(not grouped[source_id] for source_id in source_order):
        raise ValueError("every within-camera source must have dense windows")
    frame_counts = {
        source_id: max(frame for window in grouped[source_id] for frame in window.frame_indices) + 1
        for source_id in source_order
    }
    offsets: dict[str, int] = {}
    total = 0
    for source_id in source_order:
        offsets[source_id] = total
        total += frame_counts[source_id]
    cut_train, cut_validation = int(total * 0.60), int(total * 0.80)
    fps_values = {
        float(window.metadata["fps"])
        for source_id in source_order
        for window in grouped[source_id]
    }
    if len(fps_values) != 1:
        raise ValueError("within-camera sources must share one FPS")
    guard = round(next(iter(fps_values)) * guard_seconds)
    ranges = {
        "train": (0, cut_train - guard - 1),
        "validation": (cut_train + guard, cut_validation - guard - 1),
        "test": (cut_validation + guard, total - 1),
    }
    result = {name: [] for name in ranges}
    for source_id in source_order:
        offset = offsets[source_id]
        for window in grouped[source_id]:
            first = offset + min(window.frame_indices)
            last = offset + max(window.frame_indices)
            for name, (start, end) in ranges.items():
                if start <= first and last <= end:
                    result[name].append(window)
                    break
    if any(not result[name] for name in result):
        raise ValueError("chronological split produced an empty partition")
    metadata = {
        "strategy": "parent-chronology-contiguous-60-20-20",
        "source_order": list(source_order),
        "source_frame_counts": frame_counts,
        "source_offsets": offsets,
        "total_frames": total,
        "guard_seconds": guard_seconds,
        "guard_frames": guard,
        "global_ranges": {name: list(bounds) for name, bounds in ranges.items()},
        "window_counts": {name: len(values) for name, values in result.items()},
    }
    return result["train"], result["validation"], result["test"], metadata


def _balanced_accuracy(rows: Sequence[Mapping[str, Any]], threshold: float) -> float:
    labels = [int(row["inplay_target"]) for row in rows if row.get("inplay_target") in (0, 1)]
    scores = [float(row["in_play_probability"]) for row in rows if row.get("inplay_target") in (0, 1)]
    tp = sum(label == 1 and score >= threshold for label, score in zip(labels, scores))
    fn = sum(label == 1 and score < threshold for label, score in zip(labels, scores))
    tn = sum(label == 0 and score < threshold for label, score in zip(labels, scores))
    fp = sum(label == 0 and score >= threshold for label, score in zip(labels, scores))
    return ((tp / (tp + fn)) + (tn / (tn + fp))) / 2


def _select_threshold(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    candidates = [value / 100 for value in range(5, 96)]
    return max(
        ((threshold, _balanced_accuracy(rows, threshold)) for threshold in candidates),
        key=lambda item: (item[1], -abs(item[0] - 0.5)),
    )


def run_within_camera_experiment(
    dataset: SelectorWindowDataset,
    output_dir: Path,
    *,
    source_order: Sequence[str] = DEFAULT_SOURCE_ORDER,
    context_mode: ContextMode = "full_context",
    mask_candidates: bool = True,
    epochs: int = 25,
    batch_size: int = 8,
    seed: int = 1729,
    device: torch.device | None = None,
    boundary_weight: float = 1.0,
    boundary_window_seconds: float = 1.0,
    guard_seconds: float = 1.0,
    checkpoint_selection: str = "final",
    sampling_mode: str = "uniform",
    boundary_heads: bool = False,
    boundary_aux_weight: float = 0.25,
    defer_test: bool = False,
) -> dict[str, Any]:
    output_dir = prepare_output_directory(output_dir)
    device = device or select_device()
    _seed_everything(seed)
    train, validation, test, split = chronological_window_split(
        dataset.windows, source_order, guard_seconds=guard_seconds
    )
    fold = CrossFitFold("within-camera", tuple(source_order), tuple(source_order))
    model_config = SelectorConfig(
        context_mode=context_mode,
        frame_feature_dim=FRAME_DIMS[context_mode],
    )
    model = JointRallyShuttleModel(
        model_config, boundary_heads=boundary_heads
    ).to(device)
    print(
        f"within-camera train={len(train)} validation={len(validation)} "
        f"test={len(test)} device={device}",
        flush=True,
    )
    history = _train_fold(
        model,
        train,
        validation,
        fold,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        selection_weight=0.0 if mask_candidates else 1.0,
        boundary_weight=boundary_weight,
        boundary_window_seconds=boundary_window_seconds,
        mask_candidates=mask_candidates,
        evaluation_owned_only=True,
        sampling_mode=sampling_mode,
        checkpoint_selection=checkpoint_selection,
        validation_partition_designated=True,
        boundary_aux_weight=boundary_aux_weight,
    )
    model.eval()
    validation_rows = aggregate_joint_predictions(
        _predict_windows(
            model, validation, fold, device=device, batch_size=batch_size,
            owned_only=True, mask_candidates=mask_candidates,
        )
    )
    test_rows = (
        []
        if defer_test
        else aggregate_joint_predictions(
            _predict_windows(
                model, test, fold, device=device, batch_size=batch_size,
                owned_only=True, mask_candidates=mask_candidates,
            )
        )
    )
    threshold, validation_balanced_accuracy = _select_threshold(validation_rows)
    for row in test_rows:
        row["decoded_inplay"] = float(row["in_play_probability"]) >= threshold
        row["within_camera_threshold"] = threshold
    checkpoint = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "JointRallyShuttleModel",
            "selector_config": asdict(model_config),
            "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
            "pose_coordinate_mode": dataset.manifest["pose_coordinate_mode"],
            "split": split,
            "mask_candidates": mask_candidates,
            "epoch_mean_losses": history,
            "checkpoint_selection": model.checkpoint_selection,
            "sampling_mode": sampling_mode,
            "boundary_heads": boundary_heads,
            "conditioning_mode": model.conditioning_mode,
            "boundary_aux_weight": boundary_aux_weight,
            "validation_threshold": threshold,
        },
        checkpoint,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    artifact_rows = validation_rows if defer_test else test_rows
    for row in artifact_rows:
        row["checkpoint_sha256"] = checkpoint_sha256
        row["partition"] = "validation" if defer_test else "test"
    prediction_name = (
        "validation-predictions.jsonl" if defer_test else "predictions.jsonl"
    )
    (output_dir / prediction_name).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in artifact_rows),
        encoding="utf-8",
    )
    metrics = {
        "schema": "within_camera_inplay_experiment",
        "schema_version": 1,
        "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
        "pose_coordinate_mode": dataset.manifest["pose_coordinate_mode"],
        "pose_representation_version": dataset.manifest["pose_representation_version"],
        "context_mode": context_mode,
        "mask_candidates": mask_candidates,
        "boundary_weight": boundary_weight,
        "boundary_aux_weight": boundary_aux_weight,
        "boundary_heads": boundary_heads,
        "sampling_mode": sampling_mode,
        "checkpoint_selection": model.checkpoint_selection,
        "split": split,
        "validation_selected_threshold": threshold,
        "validation_balanced_accuracy": validation_balanced_accuracy,
        "validation_at_raw_0_5": compile_metrics(validation_rows),
        "test_evaluation_deferred": defer_test,
        "test": None if defer_test else compile_metrics(test_rows),
        "validation_boundary_heads": (
            _heldout_boundary_epoch_metrics(validation_rows)
            if boundary_heads else None
        ),
        "test_boundary_heads": (
            _heldout_boundary_epoch_metrics(test_rows)
            if boundary_heads and not defer_test else None
        ),
        "epoch_mean_losses": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_epoch_diagnostic_plot(
        output_dir / "epoch-inplay-diagnostics.png",
        {"within-camera": {"epoch_mean_losses": history}},
    )
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-mode", choices=tuple(FRAME_DIMS), default="full_context")
    parser.add_argument("--pose-coordinate-mode", choices=("image", "player_relative"), default="image")
    parser.add_argument("--with-candidates", action="store_true")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--boundary-window-seconds", type=float, default=1.0)
    parser.add_argument("--guard-seconds", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("validation_loss", "final"),
        default="final",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("uniform", "boundary_balanced"),
        default="uniform",
    )
    parser.add_argument("--boundary-heads", action="store_true")
    parser.add_argument("--boundary-aux-weight", type=float, default=0.25)
    parser.add_argument(
        "--defer-test",
        action="store_true",
        help="do not run inference or metrics on the fixed test partition",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config, _ = load_dataset_config(
        args.config,
        context_mode=args.context_mode,
        pose_coordinate_mode=args.pose_coordinate_mode,
    )
    allowed = set(DEFAULT_SOURCE_ORDER)
    config = replace(
        config, sources=tuple(source for source in config.sources if source.source_id in allowed)
    )
    dataset = SelectorWindowDataset(config)
    device = None if args.device == "auto" else torch.device(args.device)
    metrics = run_within_camera_experiment(
        dataset,
        args.output_dir,
        context_mode=args.context_mode,
        mask_candidates=not args.with_candidates,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        boundary_weight=args.boundary_weight,
        boundary_window_seconds=args.boundary_window_seconds,
        guard_seconds=args.guard_seconds,
        checkpoint_selection=args.checkpoint_selection,
        sampling_mode=args.sampling_mode,
        boundary_heads=args.boundary_heads,
        boundary_aux_weight=args.boundary_aux_weight,
        defer_test=args.defer_test,
    )
    summary = (
        metrics["validation_at_raw_0_5"]["inplay"]
        if args.defer_test else metrics["test"]["inplay"]
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
