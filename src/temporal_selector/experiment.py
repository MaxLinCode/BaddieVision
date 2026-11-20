"""Lean source-disjoint training and evaluation for the temporal selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .config import ContextMode, SelectorConfig
from .crossfit import CrossFitManifest, validate_out_of_source_predictions
from .dataset import FRAME_DIMS, SelectorWindowDataset
from .joint_inference import InPlayDecoderConfig, calibrate_inplay_decoder, decode_inplay_probabilities, write_joint_artifacts
from .model import JointRallyShuttleModel, TemporalShuttleSelector

# Compatibility re-exports keep historical study/test imports working.
from .training import (
    select_device as select_device,
    prepare_output_directory as prepare_output_directory,
    windows_for_sources as windows_for_sources,
    windows_for_queue as windows_for_queue,
    resolve_retained_candidate as resolve_retained_candidate,
    _ratio as _ratio,
    _interval_metrics as _interval_metrics,
    compile_metrics as compile_metrics,
    _optional_mean as _optional_mean,
    _macro_metrics as _macro_metrics,
    seed_everything as _seed_everything,
    _inplay_pos_weight as _inplay_pos_weight,
    _apply_candidate_dropout as _apply_candidate_dropout,
    _mask_candidate_inputs as _mask_candidate_inputs,
    _TokenBucketBatchSampler as _TokenBucketBatchSampler,
    train_selector_fold as _train_fold,
    _binary_roc_auc as _binary_roc_auc,
    _heldout_inplay_epoch_metrics as _heldout_inplay_epoch_metrics,
    _heldout_boundary_epoch_metrics as _heldout_boundary_epoch_metrics,
    _write_epoch_diagnostic_plot as _write_epoch_diagnostic_plot,
    _prediction_records_for_window as _prediction_records_for_window,
    predict_windows as _predict_windows,
    _predict_window as _predict_window,
    aggregate_joint_predictions as aggregate_joint_predictions,
    _resolve as _resolve,
    load_crossfit_manifest as load_crossfit_manifest,
    load_dataset_config as load_dataset_config,
)

NULL_SELECTION = "NULL_SELECTION"
NOT_REQUIRED = "not_required"
METRIC_STATUSES = {"selected_retained", "null"}


def run_experiment(
    dataset: SelectorWindowDataset,
    manifest: CrossFitManifest,
    output_dir: Path,
    *,
    context_mode: ContextMode = "candidates_only",
    epochs: int = 25,
    batch_size: int = 1,
    seed: int = 1729,
    device: torch.device | None = None,
    inplay_only: bool = False,
    boundary_weight: float = 1.0,
    boundary_window_seconds: float = 1.0,
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if boundary_weight < 1.0 or boundary_window_seconds <= 0:
        raise ValueError("boundary weight must be >= 1 and its window must be positive")
    if seed != manifest.seed:
        raise ValueError("experiment seed must match the checked-in cross-fit manifest")
    output_dir = prepare_output_directory(output_dir)
    device = device or select_device()
    _seed_everything(seed)
    joint = bool(dataset.windows and dataset.windows[0].inplay_targets is not None)
    if any((window.inplay_targets is not None) != joint for window in dataset.windows):
        raise ValueError("experiment dataset mixes legacy and joint windows")
    if joint and context_mode != "full_context":
        raise ValueError("joint rally experiments require full_context")
    if inplay_only and not joint:
        raise ValueError("InPlay-only ablation requires rally targets")
    model_config = SelectorConfig(
        context_mode=context_mode,
        frame_feature_dim=FRAME_DIMS[context_mode],
    )
    all_predictions: list[dict[str, Any]] = []
    training_summaries: dict[str, Any] = {}
    source_ids = {window.source_id for window in dataset.windows}
    for fold_index, fold in enumerate(manifest.folds):
        if not set(fold.training_source_ids + fold.evaluation_source_ids) <= source_ids:
            raise ValueError(f"fold {fold.fold_id} references a source absent from the dataset")
        train_windows = windows_for_sources(dataset.windows, fold.training_source_ids)
        evaluation_windows = windows_for_sources(
            dataset.windows, fold.evaluation_source_ids
        )
        if not train_windows or not evaluation_windows:
            raise ValueError(f"fold {fold.fold_id} has an empty train or evaluation split")
        _seed_everything(seed + fold_index)
        model = (
            JointRallyShuttleModel(model_config)
            if joint
            else TemporalShuttleSelector(model_config)
        ).to(device)
        print(
            f"fold {fold.fold_id}: train={fold.training_source_ids} "
            f"evaluate={fold.evaluation_source_ids} device={device}",
            flush=True,
        )
        history = _train_fold(
            model,
            train_windows,
            evaluation_windows,
            fold,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + fold_index,
            selection_weight=0.0 if inplay_only else 1.0,
            boundary_weight=boundary_weight,
            boundary_window_seconds=boundary_window_seconds,
            mask_candidates=inplay_only,
        )
        model.eval()
        decoder_config: InPlayDecoderConfig | None = None
        if joint:
            calibration_rows = aggregate_joint_predictions(
                _predict_windows(
                    model,
                    train_windows,
                    fold,
                    device=device,
                    batch_size=batch_size,
                    owned_only=False,
                    mask_candidates=inplay_only,
                )
            )
            calibration_rows.sort(
                key=lambda row: (row["prediction_source_id"], row["frame"])
            )
            calibration_rows = [
                row for row in calibration_rows if row.get("inplay_target") in (0, 1)
            ]
            decoder_config = calibrate_inplay_decoder(
                [float(row["in_play_probability"]) for row in calibration_rows],
                [int(row["inplay_target"]) for row in calibration_rows],
                fps=float(train_windows[0].metadata["fps"]),
            )
        fold_observations = _predict_windows(
            model,
            evaluation_windows,
            fold,
            device=device,
            batch_size=batch_size,
            owned_only=not joint,
            mask_candidates=inplay_only,
        )
        fold_predictions = (
            aggregate_joint_predictions(fold_observations)
            if joint
            else fold_observations
        )
        if joint and decoder_config is not None:
            for source_id in sorted(
                {str(row["prediction_source_id"]) for row in fold_predictions}
            ):
                source_rows = sorted(
                    (
                        row
                        for row in fold_predictions
                        if row["prediction_source_id"] == source_id
                    ),
                    key=lambda row: int(row["frame"]),
                )
                source_window = next(
                    window for window in evaluation_windows if window.source_id == source_id
                )
                decoded = decode_inplay_probabilities(
                    [float(row["in_play_probability"]) for row in source_rows],
                    fps=float(source_window.metadata["fps"]),
                    config=decoder_config,
                )
                for row, state in zip(source_rows, decoded):
                    row["decoded_inplay"] = state
                    row["decoder_config"] = asdict(decoder_config)
                    if not state:
                        row["predicted_outcome"] = NOT_REQUIRED
                        row["selected_position"] = None
                    elif row["predicted_selection_outcome"] == NULL_SELECTION:
                        row["predicted_outcome"] = "no_shuttle"
                        row["selected_position"] = None
                    else:
                        row["predicted_outcome"] = "selected"
        validate_out_of_source_predictions(manifest, fold_predictions)
        if any(
            record["selection_loss"] is not None
            and not math.isfinite(record["selection_loss"])
            for record in fold_predictions
        ):
            raise RuntimeError("evaluation produced a non-finite selection loss")
        all_predictions.extend(fold_predictions)
        training_summaries[fold.fold_id] = {
            "epoch_mean_losses": history,
            "final_mean_losses": history[-1],
            "training_window_count": len(train_windows),
            "evaluation_window_count": len(evaluation_windows),
            "decoder_config": asdict(decoder_config) if decoder_config else None,
        }
        checkpoint_path = output_dir / f"fold-{fold.fold_id}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": (
                    "JointRallyShuttleModel" if joint else "TemporalShuttleSelector"
                ),
                "selector_config": asdict(model_config),
                "fold": asdict(fold),
                "crossfit_fingerprint": manifest.fingerprint,
                "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "optimizer": {"name": "AdamW", "learning_rate": 3e-4},
                "gradient_clip_norm": 1.0,
                "device": str(device),
                "epoch_mean_losses": history,
                "candidate_dropout_probability": 0.0 if inplay_only else 0.2 if joint else 0.0,
                "candidate_inputs_masked": inplay_only,
                "selection_loss_weight": 0.0 if inplay_only else 1.0,
                "inplay_only_ablation": inplay_only,
                "conditioning_mode": (
                    model.conditioning_mode
                    if isinstance(model, JointRallyShuttleModel)
                    else None
                ),
                "boundary_weight": boundary_weight,
                "boundary_window_seconds": boundary_window_seconds,
                "decoder_config": asdict(decoder_config) if decoder_config else None,
            },
            checkpoint_path,
        )
        with checkpoint_path.open("rb") as handle:
            checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        for row in fold_predictions:
            row["checkpoint_sha256"] = checkpoint_sha256

    validate_out_of_source_predictions(manifest, all_predictions)
    prediction_path = output_dir / "predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_predictions),
        encoding="utf-8",
    )
    if joint:
        write_joint_artifacts(all_predictions, output_dir)
    fold_metrics = {
        fold.fold_id: compile_metrics(
            row for row in all_predictions if row["fold_id"] == fold.fold_id
        )
        for fold in manifest.folds
    }
    sources = sorted({row["prediction_source_id"] for row in all_predictions})
    metrics = {
        "schema": "temporal_selector_experiment_metrics",
        "schema_version": 2 if joint else 1,
        "task_definition": (
            "strict_inplay_without_shuttle_candidate_inputs"
            if inplay_only
            else "strict_inplay_with_conditional_shuttle_selection"
            if joint
            else "continuous_shuttle_selection"
        ),
        "context_mode": context_mode,
        "device": str(device),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "inplay_only_ablation": inplay_only,
        "candidate_inputs_masked": inplay_only,
        "boundary_weight": boundary_weight,
        "boundary_window_seconds": boundary_window_seconds,
        "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
        "crossfit_fingerprint": manifest.fingerprint,
        "training": training_summaries,
        "folds": fold_metrics,
        "sources": {
            source: compile_metrics(
                row
                for row in all_predictions
                if row["prediction_source_id"] == source
            )
            for source in sources
        },
        "queues": {
            queue: compile_metrics(
                row for row in all_predictions if row["label_queue"] == queue
            )
            for queue in sorted({str(row["label_queue"]) for row in all_predictions})
        },
        "crossfit_macro": _macro_metrics(list(fold_metrics.values())),
    }
    if len(manifest.folds) == 2:
        metrics["two_fold_macro"] = metrics["crossfit_macro"]
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "epoch_metrics.json").write_text(
        json.dumps(training_summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if joint:
        _write_epoch_diagnostic_plot(
            output_dir / "epoch-inplay-diagnostics.png", training_summaries
        )
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--context-mode",
        choices=("candidates_only", "players_court", "full_context"),
        default="full_context",
    )
    parser.add_argument(
        "--pose-coordinate-mode",
        choices=("image", "player_relative"),
        default="image",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=1.0,
        help="maximum tapered InPlay BCE weight at true state transitions",
    )
    parser.add_argument("--boundary-window-seconds", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--inplay-only",
        action="store_true",
        help="set shuttle-selection loss weight to zero for an InPlay-only ablation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset_config, manifest = load_dataset_config(
        args.config,
        context_mode=args.context_mode,
        pose_coordinate_mode=args.pose_coordinate_mode,
    )
    dataset = SelectorWindowDataset(dataset_config)
    device = None if args.device == "auto" else torch.device(args.device)
    metrics = run_experiment(
        dataset,
        manifest,
        args.output_dir,
        context_mode=args.context_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        inplay_only=args.inplay_only,
        boundary_weight=args.boundary_weight,
        boundary_window_seconds=args.boundary_window_seconds,
    )
    print(json.dumps(metrics["crossfit_macro"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
