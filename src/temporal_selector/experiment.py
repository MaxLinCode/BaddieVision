"""Lean source-disjoint training and evaluation for the temporal selector."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .batch import MASKED_TARGET, NULL_TARGET
from .config import ContextMode, SelectorConfig
from .crossfit import (
    CrossFitFold,
    CrossFitManifest,
    build_two_source_crossfit,
    validate_out_of_source_predictions,
)
from .dataset import (
    FRAME_DIMS,
    SelectorDataConfig,
    SelectorSourceConfig,
    SelectorWindow,
    SelectorWindowDataset,
    collate_selector_windows,
)
from .model import TemporalShuttleSelector

NULL_SELECTION = "NULL_SELECTION"
METRIC_STATUSES = {"selected_retained", "null"}


def select_device() -> torch.device:
    """Prefer accelerators without making the experiment depend on one."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_output_directory(path: Path) -> Path:
    """Create an output directory, refusing to overwrite any prior run."""
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"experiment output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def windows_for_sources(
    windows: Iterable[SelectorWindow], source_ids: Sequence[str]
) -> list[SelectorWindow]:
    allowed = set(map(str, source_ids))
    return [window for window in windows if window.source_id in allowed]


def windows_for_queue(
    windows: Iterable[SelectorWindow], queue_kind: str
) -> list[SelectorWindow]:
    return [window for window in windows if window.queue_kind == queue_kind]


def resolve_retained_candidate(
    window: SelectorWindow, local_frame: int, target: int
) -> str:
    """Resolve a frame-local retained target to its bookkeeping ID."""
    slots = torch.nonzero(
        window.candidate_frame_indices == local_frame, as_tuple=False
    ).flatten().tolist()
    if target < 0 or target >= len(slots):
        raise ValueError("retained target does not resolve to a frame candidate")
    return window.candidate_ids[slots[target]]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compile_metrics(predictions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compile selector metrics from provenance-validated frame predictions."""
    rows = list(predictions)
    statuses = [str(row["true_status"]) for row in rows]
    selected = statuses.count("selected_retained")
    missing = statuses.count("missing_proposal")
    dropped = statuses.count("dropped_by_k")
    coverage_denominator = selected + missing + dropped

    retained_rows = [row for row in rows if row["true_status"] == "selected_retained"]
    null_rows = [row for row in rows if row["true_status"] == "null"]
    supervised_rows = retained_rows + null_rows
    retained_correct = sum(
        row.get("predicted_candidate_id") == row.get("target_candidate_id")
        for row in retained_rows
    )
    overall_correct = retained_correct + sum(
        row.get("predicted_outcome") == NULL_SELECTION for row in null_rows
    )
    true_positive = sum(
        row.get("predicted_outcome") == NULL_SELECTION for row in null_rows
    )
    false_positive = sum(
        row.get("predicted_outcome") == NULL_SELECTION for row in retained_rows
    )
    false_negative = len(null_rows) - true_positive
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    losses = [float(row["selection_loss"]) for row in rows if row.get("selection_loss") is not None]
    return {
        "owned_frame_count": len(rows),
        "supervised_frame_count": len(supervised_rows),
        "mean_selection_loss": sum(losses) / len(losses) if losses else None,
        "proposal_coverage": {
            "value": _ratio(selected, coverage_denominator),
            "selected_retained": selected,
            "missing_proposal": missing,
            "dropped_by_k": dropped,
            "denominator": coverage_denominator,
        },
        "retained_target_accuracy": {
            "value": _ratio(retained_correct, len(retained_rows)),
            "correct": retained_correct,
            "total": len(retained_rows),
        },
        "overall_accuracy": {
            "value": _ratio(overall_correct, len(supervised_rows)),
            "correct": overall_correct,
            "total": len(supervised_rows),
        },
        "null_selection": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
    }


def _optional_mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _macro_metrics(fold_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    coverage = [metric["proposal_coverage"] for metric in fold_metrics]
    retained = [metric["retained_target_accuracy"] for metric in fold_metrics]
    overall = [metric["overall_accuracy"] for metric in fold_metrics]
    null = [metric["null_selection"] for metric in fold_metrics]
    return {
        "fold_count": len(fold_metrics),
        "mean_selection_loss": _optional_mean(
            metric["mean_selection_loss"] for metric in fold_metrics
        ),
        "proposal_coverage": {
            "value": _optional_mean(item["value"] for item in coverage),
            "selected_retained": sum(item["selected_retained"] for item in coverage),
            "missing_proposal": sum(item["missing_proposal"] for item in coverage),
            "dropped_by_k": sum(item["dropped_by_k"] for item in coverage),
            "denominator": sum(item["denominator"] for item in coverage),
        },
        "retained_target_accuracy": {
            "value": _optional_mean(item["value"] for item in retained),
            "correct": sum(item["correct"] for item in retained),
            "total": sum(item["total"] for item in retained),
        },
        "overall_accuracy": {
            "value": _optional_mean(item["value"] for item in overall),
            "correct": sum(item["correct"] for item in overall),
            "total": sum(item["total"] for item in overall),
        },
        "null_selection": {
            "precision": _optional_mean(item["precision"] for item in null),
            "recall": _optional_mean(item["recall"] for item in null),
            "f1": _optional_mean(item["f1"] for item in null),
            "true_positive": sum(item["true_positive"] for item in null),
            "false_positive": sum(item["false_positive"] for item in null),
            "false_negative": sum(item["false_negative"] for item in null),
        },
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_fold(
    model: TemporalShuttleSelector,
    windows: Sequence[SelectorWindow],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        list(windows),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_selector_windows,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history: list[float] = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("training produced a non-finite selection loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            raise ValueError("training fold has no windows")
        history.append(sum(losses) / len(losses))
        print(
            f"epoch {epoch + 1}/{epochs}: mean selection loss {history[-1]:.6f}",
            flush=True,
        )
    return history


def _predict_window(
    model: TemporalShuttleSelector,
    window: SelectorWindow,
    fold: CrossFitFold,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    batch = collate_selector_windows([window]).to(device)
    with torch.no_grad():
        output = model(batch)
    records = []
    for local_frame, frame in enumerate(window.frame_indices):
        if frame not in window.owned_frames:
            continue
        slots = torch.nonzero(
            window.candidate_frame_indices == local_frame, as_tuple=False
        ).flatten().tolist()
        candidate_ids = [window.candidate_ids[slot] for slot in slots]
        frame_logits = torch.cat(
            (
                output.candidate_logits[0, slots],
                output.null_logits[0, local_frame].view(1),
            )
        )
        predicted_index = int(torch.argmax(frame_logits).cpu())
        predicted_candidate = (
            None if predicted_index == len(slots) else candidate_ids[predicted_index]
        )
        target = int(window.targets[local_frame])
        status = window.target_status[local_frame]
        target_candidate = (
            resolve_retained_candidate(window, local_frame, target)
            if status == "selected_retained"
            else None
        )
        selection_loss = None
        if target != MASKED_TARGET:
            resolved_target = len(slots) if target == NULL_TARGET else target
            selection_loss = float(
                F.cross_entropy(
                    frame_logits.view(1, -1),
                    torch.tensor([resolved_target], device=device),
                ).cpu()
            )
        true_outcome = (
            target_candidate
            if status == "selected_retained"
            else NULL_SELECTION
            if status == "null"
            else status
        )
        records.append(
            {
                "fold_id": fold.fold_id,
                "training_source_ids": list(fold.training_source_ids),
                "evaluation_source_ids": list(fold.evaluation_source_ids),
                "prediction_source_id": window.source_id,
                "label_queue": window.queue_kind,
                "burst_id": window.burst_id,
                "frame": frame,
                "true_status": status,
                "true_outcome": true_outcome,
                "target_candidate_id": target_candidate,
                "predicted_outcome": predicted_candidate or NULL_SELECTION,
                "predicted_candidate_id": predicted_candidate,
                "candidate_ids": candidate_ids,
                "selection_loss": selection_loss,
            }
        )
    return records


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
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if seed != manifest.seed:
        raise ValueError("experiment seed must match the checked-in cross-fit manifest")
    output_dir = prepare_output_directory(output_dir)
    device = device or select_device()
    _seed_everything(seed)
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
        model = TemporalShuttleSelector(model_config).to(device)
        print(
            f"fold {fold.fold_id}: train={fold.training_source_ids} "
            f"evaluate={fold.evaluation_source_ids} device={device}",
            flush=True,
        )
        history = _train_fold(
            model,
            train_windows,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + fold_index,
        )
        model.eval()
        fold_predictions = [
            record
            for window in evaluation_windows
            for record in _predict_window(model, window, fold, device=device)
        ]
        validate_out_of_source_predictions(manifest, fold_predictions)
        if any(
            record["selection_loss"] is not None
            and not math.isfinite(record["selection_loss"])
            for record in fold_predictions
        ):
            raise RuntimeError("evaluation produced a non-finite selection loss")
        all_predictions.extend(fold_predictions)
        training_summaries[fold.fold_id] = {
            "epoch_mean_selection_losses": history,
            "final_mean_selection_loss": history[-1],
            "training_window_count": len(train_windows),
            "evaluation_window_count": len(evaluation_windows),
        }
        torch.save(
            {
                "model_state_dict": model.state_dict(),
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
                "epoch_mean_selection_losses": history,
            },
            output_dir / f"fold-{fold.fold_id}.pt",
        )

    validate_out_of_source_predictions(manifest, all_predictions)
    prediction_path = output_dir / "predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_predictions),
        encoding="utf-8",
    )
    fold_metrics = {
        fold.fold_id: compile_metrics(
            row for row in all_predictions if row["fold_id"] == fold.fold_id
        )
        for fold in manifest.folds
    }
    sources = sorted({row["prediction_source_id"] for row in all_predictions})
    metrics = {
        "schema": "temporal_selector_experiment_metrics",
        "schema_version": 1,
        "context_mode": context_mode,
        "device": str(device),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
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
            for queue in ("adaptive", "audit")
        },
        "two_fold_macro": _macro_metrics(list(fold_metrics.values())),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def load_crossfit_manifest(path: Path) -> CrossFitManifest:
    value = json.loads(path.read_text(encoding="utf-8"))
    folds = tuple(
        CrossFitFold(
            str(fold["fold_id"]),
            tuple(map(str, fold["training_source_ids"])),
            tuple(map(str, fold["evaluation_source_ids"])),
        )
        for fold in value["folds"]
    )
    manifest = CrossFitManifest(int(value["seed"]), folds, str(value["fingerprint"]))
    if len(folds) != 2 or any(
        set(fold.training_source_ids) & set(fold.evaluation_source_ids)
        for fold in folds
    ):
        raise ValueError("cross-fit manifest must contain two source-disjoint folds")
    if any(
        len(fold.training_source_ids) != 1 or len(fold.evaluation_source_ids) != 1
        for fold in folds
    ):
        raise ValueError("the two-fold experiment requires one source on each side")
    expected = build_two_source_crossfit(
        (folds[0].training_source_ids[0], folds[0].evaluation_source_ids[0]),
        seed=manifest.seed,
    )
    if manifest != expected:
        raise ValueError("cross-fit manifest topology or fingerprint is invalid")
    return manifest


def load_dataset_config(
    path: Path, *, context_mode: ContextMode
) -> tuple[SelectorDataConfig, CrossFitManifest]:
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    data = value["dataset"]
    sources = tuple(
        SelectorSourceConfig(
            source_id=str(source["source_id"]),
            video_path=_resolve(path, source["video_path"]),
            candidates_path=_resolve(path, source["candidates_path"]),
            assignments_path=_resolve(path, source["assignments_path"]),
            pose_cache_path=_resolve(path, source["pose_cache_path"]),
            calibration_path=_resolve(path, source["calibration_path"]),
        )
        for source in data["sources"]
    )
    dataset_config = SelectorDataConfig(
        sources=sources,
        queue_paths=tuple(_resolve(path, item) for item in data["queue_paths"]),
        annotations_path=_resolve(path, data["annotations_path"]),
        context_mode=context_mode,
        minimum_cutoff=float(data.get("minimum_cutoff", 0.05)),
        retention_k=int(data.get("retention_k", 8)),
        pose_visibility_threshold=float(data.get("pose_visibility_threshold", 0.5)),
        expected_annotation_sha256=data.get("expected_annotation_sha256"),
    )
    manifest = load_crossfit_manifest(_resolve(path, value["crossfit_manifest_path"]))
    if manifest.seed != int(value.get("seed", manifest.seed)):
        raise ValueError("dataset config and cross-fit manifest seeds differ")
    return dataset_config, manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--context-mode",
        choices=("candidates_only", "players_court", "full_context"),
        default="candidates_only",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset_config, manifest = load_dataset_config(
        args.config, context_mode=args.context_mode
    )
    dataset = SelectorWindowDataset(dataset_config)
    metrics = run_experiment(
        dataset,
        manifest,
        args.output_dir,
        context_mode=args.context_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(metrics["two_fold_macro"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
