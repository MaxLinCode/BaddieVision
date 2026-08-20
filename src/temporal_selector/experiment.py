"""Lean source-disjoint training and evaluation for the temporal selector."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import warnings
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from InPlay.heuristic.evaluate import Interval, evaluate as evaluate_intervals

from .batch import MASKED_TARGET, NULL_TARGET, SelectorBatch
from .config import ContextMode, SelectorConfig
from .crossfit import (
    CrossFitFold,
    CrossFitManifest,
    validate_out_of_source_predictions,
)
from .dataset import (
    FRAME_DIMS,
    SelectorDataConfig,
    SelectorSourceConfig,
    SelectorWindow,
    SelectorWindowDataset,
    PoseCoordinateMode,
    collate_selector_windows,
)
from .joint_inference import (
    InPlayDecoderConfig,
    calibrate_inplay_decoder,
    decode_inplay_probabilities,
    decoded_intervals,
    write_joint_artifacts,
)
from .model import JointRallyShuttleModel, SelectorOutput, TemporalShuttleSelector

NULL_SELECTION = "NULL_SELECTION"
NOT_REQUIRED = "not_required"
METRIC_STATUSES = {"selected_retained", "null"}


def select_device() -> torch.device:
    """Prefer accelerators without making the experiment depend on one."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.backends.mps.is_built():
        warnings.warn(
            "PyTorch was built with MPS support, but MPS is unavailable; "
            "training will use CPU. Verify that an MPS tensor can be allocated "
            "outside any sandbox that may hide Metal before running a long "
            "experiment.",
            RuntimeWarning,
            stacklevel=2,
        )
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


def _interval_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    state_rows = [row for row in rows if row.get("inplay_target") in (0, 1)]
    if not state_rows:
        return None
    predictions: list[Interval] = []
    labels: list[Interval] = []
    frame_ranges: dict[str, tuple[int, int]] = {}
    for source_id in sorted({str(row["prediction_source_id"]) for row in state_rows}):
        source_rows = sorted(
            (row for row in state_rows if row["prediction_source_id"] == source_id),
            key=lambda row: int(row["frame"]),
        )
        frames = [int(row["frame"]) for row in source_rows]
        predicted = [bool(row.get("decoded_inplay")) for row in source_rows]
        target = [bool(int(row["inplay_target"])) for row in source_rows]
        frame_ranges[source_id] = frames[0], frames[-1]
        runs: list[tuple[list[int], list[bool], list[bool]]] = []
        for frame, predicted_state, target_state in zip(frames, predicted, target):
            if not runs or frame != runs[-1][0][-1] + 1:
                runs.append(([], [], []))
            runs[-1][0].append(frame)
            runs[-1][1].append(predicted_state)
            runs[-1][2].append(target_state)
        prediction_number = len(predictions)
        label_number = len(labels)
        for run_frames, run_predictions, run_targets in runs:
            for start, end in decoded_intervals(run_frames, run_predictions):
                prediction_number += 1
                predictions.append(Interval(
                    source_id, f"prediction-{prediction_number:04d}", start, end
                ))
            for start, end in decoded_intervals(run_frames, run_targets):
                label_number += 1
                labels.append(Interval(
                    source_id, f"label-{label_number:04d}", start, end
                ))
    metrics, _ = evaluate_intervals(
        predictions, labels, threshold=0.5, frame_ranges=frame_ranges
    )
    return metrics


def compile_metrics(predictions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compile conditional selector and binary rally metrics."""
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
    def selection_outcome(row: Mapping[str, Any]) -> object:
        return row.get("predicted_selection_outcome", row.get("predicted_outcome"))

    overall_correct = retained_correct + sum(
        selection_outcome(row) == NULL_SELECTION for row in null_rows
    )
    true_positive = sum(
        selection_outcome(row) == NULL_SELECTION for row in null_rows
    )
    false_positive = sum(
        selection_outcome(row) == NULL_SELECTION for row in retained_rows
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
    state_rows = [row for row in rows if row.get("inplay_target") in (0, 1)]
    state_tp = sum(
        int(row["inplay_target"]) == 1 and bool(row.get("decoded_inplay"))
        for row in state_rows
    )
    state_fp = sum(
        int(row["inplay_target"]) == 0 and bool(row.get("decoded_inplay"))
        for row in state_rows
    )
    state_fn = sum(
        int(row["inplay_target"]) == 1 and not bool(row.get("decoded_inplay"))
        for row in state_rows
    )
    state_tn = len(state_rows) - state_tp - state_fp - state_fn
    state_precision = _ratio(state_tp, state_tp + state_fp)
    state_recall = _ratio(state_tp, state_tp + state_fn)
    state_f1 = (
        2 * state_precision * state_recall / (state_precision + state_recall)
        if state_precision is not None
        and state_recall is not None
        and state_precision + state_recall
        else None
    )
    zero_candidate_positive = [
        row
        for row in state_rows
        if int(row["inplay_target"]) == 1 and int(row.get("candidate_count", 0)) == 0
    ]
    candidate_negative = [
        row
        for row in state_rows
        if int(row["inplay_target"]) == 0 and int(row.get("candidate_count", 0)) > 0
    ]
    inplay_losses = [
        float(row["inplay_loss"])
        for row in state_rows
        if row.get("inplay_loss") is not None
    ]
    return {
        "owned_frame_count": len(rows),
        "supervised_frame_count": len(supervised_rows),
        "mean_selection_loss": sum(losses) / len(losses) if losses else None,
        "mean_inplay_loss": (
            sum(inplay_losses) / len(inplay_losses) if inplay_losses else None
        ),
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
        "inplay": {
            "precision": state_precision,
            "recall": state_recall,
            "f1": state_f1,
            "true_positive": state_tp,
            "false_positive": state_fp,
            "true_negative": state_tn,
            "false_negative": state_fn,
            "total": len(state_rows),
        },
        "shortcut_slices": {
            "positive_zero_candidates": {
                "recall": _ratio(
                    sum(bool(row.get("decoded_inplay")) for row in zero_candidate_positive),
                    len(zero_candidate_positive),
                ),
                "total": len(zero_candidate_positive),
            },
            "negative_with_candidates": {
                "false_positive_rate": _ratio(
                    sum(bool(row.get("decoded_inplay")) for row in candidate_negative),
                    len(candidate_negative),
                ),
                "total": len(candidate_negative),
            },
        },
        "intervals": _interval_metrics(rows),
    }


def _optional_mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _macro_metrics(fold_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    coverage = [metric["proposal_coverage"] for metric in fold_metrics]
    retained = [metric["retained_target_accuracy"] for metric in fold_metrics]
    overall = [metric["overall_accuracy"] for metric in fold_metrics]
    null = [metric["null_selection"] for metric in fold_metrics]
    inplay = [metric["inplay"] for metric in fold_metrics]
    intervals = [metric["intervals"] for metric in fold_metrics if metric["intervals"]]
    return {
        "fold_count": len(fold_metrics),
        "mean_selection_loss": _optional_mean(
            metric["mean_selection_loss"] for metric in fold_metrics
        ),
        "mean_inplay_loss": _optional_mean(
            metric["mean_inplay_loss"] for metric in fold_metrics
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
        "inplay": {
            "precision": _optional_mean(item["precision"] for item in inplay),
            "recall": _optional_mean(item["recall"] for item in inplay),
            "f1": _optional_mean(item["f1"] for item in inplay),
            "true_positive": sum(item["true_positive"] for item in inplay),
            "false_positive": sum(item["false_positive"] for item in inplay),
            "true_negative": sum(item["true_negative"] for item in inplay),
            "false_negative": sum(item["false_negative"] for item in inplay),
            "total": sum(item["total"] for item in inplay),
        },
        "intervals": {
            "f1": _optional_mean(item["f1"] for item in intervals),
            "mean_absolute_boundary_error": _optional_mean(
                item["mean_absolute_boundary_error"] for item in intervals
            ),
            "false_split_count": sum(item["false_split_count"] for item in intervals),
            "false_merge_count": sum(item["false_merge_count"] for item in intervals),
        }
        if intervals
        else None,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _inplay_pos_weight(windows: Sequence[SelectorWindow], device: torch.device) -> torch.Tensor:
    targets = [
        int(window.inplay_targets[index])
        for window in windows
        if window.inplay_targets is not None
        for index, frame in enumerate(window.frame_indices)
        if frame in window.owned_frames
    ]
    positives, negatives = targets.count(1), targets.count(0)
    if not positives or not negatives:
        raise ValueError("joint training requires positive and negative InPlay frames")
    return torch.tensor(negatives / positives, dtype=torch.float32, device=device)


def _apply_candidate_dropout(
    batch: SelectorBatch,
    *,
    generator: torch.Generator,
    probability: float,
) -> SelectorBatch:
    """Drop complete candidate streams while retaining rally supervision."""
    if probability <= 0:
        return batch
    dropped = torch.rand(batch.candidate_mask.shape[0], generator=generator) < probability
    if not bool(dropped.any()):
        return batch
    dropped = dropped.to(batch.candidate_mask.device)
    candidate_mask = batch.candidate_mask.clone()
    candidate_mask[dropped] = False
    candidate_frames = batch.candidate_frame_indices.clone()
    candidate_frames[dropped] = -1
    selection_targets = batch.targets.clone()
    selection_targets[dropped] = MASKED_TARGET
    # The incoming batch has already been validated by collation. These mask
    # updates preserve its shape and target invariants, so validating again
    # here would only repeat data-dependent device synchronizations before the
    # model performs its own boundary validation.
    return replace(
        batch,
        candidate_mask=candidate_mask,
        candidate_frame_indices=candidate_frames,
        targets=selection_targets,
    )


def _mask_candidate_inputs(batch: SelectorBatch) -> SelectorBatch:
    """Remove every shuttle-candidate token and its selection supervision."""
    return replace(
        batch,
        candidate_mask=torch.zeros_like(batch.candidate_mask),
        candidate_frame_indices=torch.full_like(batch.candidate_frame_indices, -1),
        targets=torch.full_like(batch.targets, MASKED_TARGET),
    )


class _TokenBucketBatchSampler(Sampler[list[int]]):
    """Deterministically shuffle while limiting attention padding per batch."""

    def __init__(
        self,
        windows: Sequence[SelectorWindow],
        *,
        batch_size: int,
        generator: torch.Generator,
        bucket_multiplier: int = 8,
        sampling_mode: str = "uniform",
    ) -> None:
        self.windows = windows
        self.batch_size = batch_size
        self.generator = generator
        self.bucket_size = max(batch_size, batch_size * bucket_multiplier)
        if sampling_mode not in {"uniform", "boundary_balanced"}:
            raise ValueError("sampling mode must be uniform or boundary_balanced")
        self.sampling_mode = sampling_mode
        self.last_sampling_stats: dict[str, Any] | None = None

    @staticmethod
    def _category(window: SelectorWindow) -> str:
        if window.metadata.get("owns_short_rally"):
            return "short_rally"
        if window.metadata.get("owns_rally_boundary"):
            return "boundary"
        return "ordinary"

    def __len__(self) -> int:
        return math.ceil(len(self.windows) / self.batch_size)

    def __iter__(self):
        categories = [self._category(window) for window in self.windows]
        if self.sampling_mode == "boundary_balanced":
            weights = torch.tensor(
                [{"ordinary": 1, "boundary": 3, "short_rally": 5}[item] for item in categories],
                dtype=torch.float64,
            )
            shuffled = torch.multinomial(
                weights, len(self.windows), replacement=True, generator=self.generator
            ).tolist()
        else:
            shuffled = torch.randperm(len(self.windows), generator=self.generator).tolist()
        self.last_sampling_stats = {
            "requested_window_count": len(self.windows),
            "requested_category_counts": dict(sorted(Counter(categories).items())),
            "realized_category_counts": dict(
                sorted(Counter(categories[index] for index in shuffled).items())
            ),
            "unique_sampled_window_count": len(set(shuffled)),
        }
        batches: list[list[int]] = []
        for start in range(0, len(shuffled), self.bucket_size):
            bucket = shuffled[start : start + self.bucket_size]
            bucket.sort(
                key=lambda index: (
                    len(self.windows[index].frame_indices)
                    + len(self.windows[index].candidate_ids)
                )
            )
            batches.extend(
                bucket[offset : offset + self.batch_size]
                for offset in range(0, len(bucket), self.batch_size)
            )
        if batches:
            batch_order = torch.randperm(
                len(batches), generator=self.generator
            ).tolist()
            for index in batch_order:
                yield batches[index]


def _train_fold(
    model: TemporalShuttleSelector,
    windows: Sequence[SelectorWindow],
    evaluation_windows: Sequence[SelectorWindow],
    fold: CrossFitFold,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    candidate_dropout: float = 0.2,
    selection_weight: float = 1.0,
    boundary_weight: float = 1.0,
    boundary_window_seconds: float = 1.0,
    mask_candidates: bool = False,
    evaluation_owned_only: bool = False,
    sampling_mode: str = "uniform",
    checkpoint_selection: str = "final",
    validation_partition_designated: bool = False,
    boundary_aux_weight: float = 0.25,
) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    dropout_generator = torch.Generator().manual_seed(seed + 10_000)
    sampler = _TokenBucketBatchSampler(
        windows,
        batch_size=batch_size,
        generator=generator,
        sampling_mode=sampling_mode,
    )
    loader = DataLoader(
        list(windows),
        batch_sampler=sampler,
        collate_fn=collate_selector_windows,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    joint = bool(windows[0].inplay_targets is not None)
    if any((window.inplay_targets is not None) != joint for window in windows):
        raise ValueError("training fold mixes legacy and joint windows")
    pos_weight = _inplay_pos_weight(windows, device) if joint else None
    if checkpoint_selection not in {"final", "validation_loss"}:
        raise ValueError("checkpoint selection must be final or validation_loss")
    if checkpoint_selection == "validation_loss" and not validation_partition_designated:
        raise ValueError(
            "validation-loss checkpoint selection requires a designated validation partition"
        )

    def _soft_pos_weight(field: str) -> torch.Tensor | None:
        values: list[float] = []
        for window in windows:
            targets = getattr(window, field)
            if targets is None:
                continue
            values.extend(
                float(value)
                for frame, value in zip(window.frame_indices, targets)
                if frame in window.owned_frames and float(value) != MASKED_TARGET
            )
        positive_mass = sum(values)
        if not values or positive_mass <= 0:
            return None
        return torch.tensor(
            min(10.0, (len(values) - positive_mass) / positive_mass),
            dtype=torch.float32,
            device=device,
        )

    start_pos_weight = _soft_pos_weight("rally_start_targets")
    end_pos_weight = _soft_pos_weight("rally_end_targets")
    history: list[dict[str, Any]] = []
    best: tuple[float, float, int, dict[str, torch.Tensor]] | None = None
    for epoch in range(epochs):
        model.train()
        metric_sums = torch.zeros(5, device=device)
        batch_count = 0
        for batch in loader:
            batch = batch.to(device)
            if mask_candidates:
                batch = _mask_candidate_inputs(batch)
            elif joint:
                batch = _apply_candidate_dropout(
                    batch,
                    generator=dropout_generator,
                    probability=candidate_dropout,
                )
            optimizer.zero_grad(set_to_none=True)
            if joint:
                components = model.joint_losses(
                    batch,
                    inplay_pos_weight=pos_weight,
                    selection_weight=selection_weight,
                    boundary_weight=boundary_weight,
                    boundary_window_seconds=boundary_window_seconds,
                    return_counts=False,
                    boundary_start_pos_weight=start_pos_weight,
                    boundary_end_pos_weight=end_pos_weight,
                    boundary_aux_weight=boundary_aux_weight,
                )
                loss = components.total
                metric_sums[1] += components.selection.detach()
                metric_sums[2] += components.inplay.detach()
                if components.rally_start is not None:
                    metric_sums[3] += components.rally_start.detach()
                    metric_sums[4] += components.rally_end.detach()
            else:
                loss = model.loss(batch)
                metric_sums[1] += loss.detach()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    "training produced a non-finite joint loss "
                    f"(total={float(loss.detach())}, "
                    f"selection={float(components.selection.detach()) if joint else float(loss.detach())}, "
                    f"inplay={float(components.inplay.detach()) if joint else 0.0})"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            metric_sums[0] += loss.detach()
            batch_count += 1
        if not batch_count:
            raise ValueError("training fold has no windows")
        total_sum, selection_sum, inplay_sum, start_sum, end_sum = metric_sums.cpu().tolist()
        summary = {
            "total": total_sum / batch_count,
            "selection": selection_sum / batch_count,
            "inplay": inplay_sum / batch_count if joint else 0.0,
            "rally_start": start_sum / batch_count,
            "rally_end": end_sum / batch_count,
            "sampling": sampler.last_sampling_stats,
        }
        if joint:
            model.eval()
            heldout_rows = aggregate_joint_predictions(
                _predict_windows(
                    model,
                    evaluation_windows,
                    fold,
                    device=device,
                    batch_size=batch_size,
                    owned_only=evaluation_owned_only,
                    mask_candidates=mask_candidates,
                )
            )
            summary.update(_heldout_inplay_epoch_metrics(heldout_rows, evaluation_windows))
            if getattr(model, "boundary_heads", False):
                summary.update(_heldout_boundary_epoch_metrics(heldout_rows))
            candidate = (
                float(summary["heldout_inplay_loss"]),
                -float(summary["heldout_roc_auc"]),
                epoch,
                copy.deepcopy(model.state_dict()),
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate
            model.train()
        history.append(summary)
        print(
            f"epoch {epoch + 1}/{epochs}: total={summary['total']:.6f} "
            f"selection={summary['selection']:.6f} inplay={summary['inplay']:.6f}"
            + (
                f" heldout_loss={summary['heldout_inplay_loss']:.6f}"
                f" P={summary['heldout_precision']:.3f}"
                f" R={summary['heldout_recall']:.3f}"
                f" F1={summary['heldout_f1']:.3f}"
                if joint else ""
            ),
            flush=True,
        )
    selected_epoch = epochs
    selected_loss = history[-1].get("heldout_inplay_loss")
    selected_auc = history[-1].get("heldout_roc_auc")
    if checkpoint_selection == "validation_loss":
        if best is None:
            raise ValueError("validation selection requires joint held-out metrics")
        model.load_state_dict(best[3])
        selected_loss, selected_auc, selected_epoch = best[0], -best[1], best[2] + 1
    model.checkpoint_selection = {
        "policy": checkpoint_selection,
        "selected_epoch": selected_epoch,
        "validation_inplay_loss": selected_loss,
        "validation_roc_auc": selected_auc,
    }
    return history


def _binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Return tie-aware binary ROC AUC without an optional sklearn dependency."""
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    rank = 1
    index = 0
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
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _heldout_inplay_epoch_metrics(
    rows: Sequence[Mapping[str, Any]],
    windows: Sequence[SelectorWindow],
) -> dict[str, float]:
    """Compile thresholded frame and one-second boundary diagnostics."""
    supervised = [row for row in rows if row.get("inplay_target") in (0, 1)]
    labels = [int(row["inplay_target"]) for row in supervised]
    probabilities = [float(row["in_play_probability"]) for row in supervised]
    predicted = [value >= 0.5 for value in probabilities]
    tp = sum(label == 1 and value for label, value in zip(labels, predicted))
    fp = sum(label == 0 and value for label, value in zip(labels, predicted))
    fn = sum(label == 1 and not value for label, value in zip(labels, predicted))
    negatives = sum(label == 0 for label in labels)
    precision = _ratio(tp, tp + fp) or 0.0
    recall = _ratio(tp, tp + fn) or 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in supervised:
        by_source.setdefault(str(row["prediction_source_id"]), []).append(row)
    fps_by_source = {
        window.source_id: float(window.metadata["fps"]) for window in windows
    }
    boundary_keys: set[tuple[str, int]] = set()
    for source_id, source_rows in by_source.items():
        source_rows.sort(key=lambda row: int(row["frame"]))
        transitions = [
            int(right["frame"])
            for left, right in zip(source_rows, source_rows[1:])
            if int(left["inplay_target"]) != int(right["inplay_target"])
        ]
        radius = max(1, round(fps_by_source[source_id]))
        for transition in transitions:
            boundary_keys.update(
                (source_id, frame)
                for frame in range(transition - radius, transition + radius + 1)
            )
    boundary_losses = [
        float(row["inplay_loss"])
        for row in supervised
        if (str(row["prediction_source_id"]), int(row["frame"])) in boundary_keys
        and row.get("inplay_loss") is not None
    ]
    losses = [float(row["inplay_loss"]) for row in supervised if row.get("inplay_loss") is not None]
    return {
        "heldout_inplay_loss": sum(losses) / len(losses) if losses else 0.0,
        "heldout_precision": precision,
        "heldout_recall": recall,
        "heldout_f1": f1,
        "heldout_negative_false_positive_rate": fp / negatives if negatives else 0.0,
        "heldout_roc_auc": _binary_roc_auc(labels, probabilities) or 0.0,
        "heldout_boundary_window_loss": (
            sum(boundary_losses) / len(boundary_losses) if boundary_losses else 0.0
        ),
    }


def _heldout_boundary_epoch_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in ("rally_start", "rally_end"):
        supervised = [
            row for row in rows if row.get(f"{name}_target") != MASKED_TARGET
        ]
        losses = [float(row[f"{name}_loss"]) for row in supervised]
        labels = [int(float(row[f"{name}_target"]) == 1.0) for row in supervised]
        scores = [float(row[f"{name}_probability"]) for row in supervised]
        output[f"heldout_{name}_loss"] = sum(losses) / len(losses) if losses else 0.0
        output[f"heldout_{name}_roc_auc"] = _binary_roc_auc(labels, scores) or 0.0
    return output


def _write_epoch_diagnostic_plot(
    path: Path, training_summaries: Mapping[str, Mapping[str, Any]]
) -> None:
    """Plot held-out InPlay curves for every fold on a shared epoch axis."""
    import matplotlib.pyplot as plt

    fields = (
        ("heldout_inplay_loss", "Binary loss"),
        ("heldout_precision", "Precision"),
        ("heldout_recall", "Recall"),
        ("heldout_f1", "F1"),
        ("heldout_negative_false_positive_rate", "Negative-frame FPR"),
        ("heldout_roc_auc", "ROC AUC"),
        ("heldout_boundary_window_loss", "Boundary ±1s loss"),
    )
    figure, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat, fields):
        for fold_id, summary in sorted(training_summaries.items()):
            history = summary["epoch_mean_losses"]
            axis.plot(
                range(1, len(history) + 1),
                [float(epoch[field]) for epoch in history],
                marker="o",
                markersize=2,
                label=f"fold {fold_id}",
            )
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(fields):]:
        axis.set_visible(False)
    axes.flat[0].legend()
    figure.suptitle("Held-out InPlay diagnostics")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _prediction_records_for_window(
    output: SelectorOutput,
    output_index: int,
    window: SelectorWindow,
    fold: CrossFitFold,
    *,
    owned_only: bool = True,
    candidates_masked: bool = False,
) -> list[dict[str, Any]]:
    records = []
    max_time = max(
        (abs(float(value)) for value in window.relative_time_seconds),
        default=0.0,
    )
    for local_frame, frame in enumerate(window.frame_indices):
        owned = frame in window.owned_frames
        if owned_only and not owned:
            continue
        slots = (
            []
            if candidates_masked
            else torch.nonzero(
                window.candidate_frame_indices == local_frame, as_tuple=False
            ).flatten().tolist()
        )
        candidate_ids = [window.candidate_ids[slot] for slot in slots]
        candidate_logits_by_id = {
            candidate_id: float(output.candidate_logits[output_index, slot])
            for candidate_id, slot in zip(candidate_ids, slots)
        }
        frame_logits = torch.cat(
            (
                output.candidate_logits[output_index, slots],
                output.null_logits[output_index, local_frame].view(1),
            )
        )
        predicted_index = int(torch.argmax(frame_logits))
        predicted_candidate = (
            None if predicted_index == len(slots) else candidate_ids[predicted_index]
        )
        predicted_slot = None if predicted_candidate is None else slots[predicted_index]
        target = int(window.targets[local_frame])
        original_status = window.target_status[local_frame]
        status = "candidate_inputs_masked" if candidates_masked else original_status
        target_candidate = (
            resolve_retained_candidate(window, local_frame, target)
            if not candidates_masked and status == "selected_retained"
            else None
        )
        selection_loss = None
        if target != MASKED_TARGET and not candidates_masked:
            resolved_target = len(slots) if target == NULL_TARGET else target
            selection_loss = float(
                F.cross_entropy(
                    frame_logits.view(1, -1),
                    torch.tensor([resolved_target]),
                )
            )
        inplay_target = (
            int(window.inplay_targets[local_frame])
            if window.inplay_targets is not None
            else None
        )
        inplay_logit = (
            float(output.inplay_logits[output_index, local_frame])
            if output.inplay_logits is not None
            else None
        )
        inplay_probability = (
            float(torch.sigmoid(output.inplay_logits[output_index, local_frame]))
            if output.inplay_logits is not None
            else None
        )
        inplay_loss = None
        if inplay_target in (0, 1) and output.inplay_logits is not None:
            inplay_loss = float(
                F.binary_cross_entropy_with_logits(
                    output.inplay_logits[output_index, local_frame].view(1),
                    torch.tensor([float(inplay_target)]),
                )
            )
        boundary_values: dict[str, Any] = {}
        for name in ("rally_start", "rally_end"):
            logits = getattr(output, f"{name}_logits")
            targets = getattr(window, f"{name}_targets")
            target = float(targets[local_frame]) if targets is not None else None
            logit = float(logits[output_index, local_frame]) if logits is not None else None
            probability = (
                float(torch.sigmoid(logits[output_index, local_frame]))
                if logits is not None else None
            )
            loss_value = None
            if target is not None and target != MASKED_TARGET and logits is not None:
                loss_value = float(F.binary_cross_entropy_with_logits(
                    logits[output_index, local_frame].view(1),
                    torch.tensor([target]),
                ))
            boundary_values.update({
                f"{name}_target": target,
                f"{name}_logit": logit,
                f"{name}_probability": probability,
                f"{name}_loss": loss_value,
            })
        selected_position = None
        candidate_positions = {}
        for candidate_id, slot in zip(candidate_ids, slots):
            values = window.candidate_values[slot]
            candidate_positions[candidate_id] = {
                "coordinate_space": "normalized_image_xy",
                "canonical_field": "peak_position_normalized",
                "weighted_centroid_normalized": [
                    float(values[0]),
                    float(values[1]),
                ],
                "peak_position_normalized": [float(values[2]), float(values[3])],
                "bbox_normalized": [float(value) for value in values[4:8]],
            }
        if predicted_slot is not None:
            selected_position = candidate_positions[predicted_candidate]
        relative_time = float(window.relative_time_seconds[local_frame])
        aggregation_weight = (
            1.0 - 0.5 * abs(relative_time) / max_time if max_time else 1.0
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
                "label_queue": window.metadata.get("label_queue_by_frame", {}).get(
                    str(frame), window.queue_kind
                ),
                "burst_id": window.burst_id,
                "frame": frame,
                "owned": owned,
                "relative_time_seconds": relative_time,
                "aggregation_weight": aggregation_weight,
                "true_status": status,
                "original_true_status": original_status,
                "true_outcome": true_outcome,
                "target_candidate_id": target_candidate,
                "predicted_outcome": predicted_candidate or NULL_SELECTION,
                "predicted_selection_outcome": predicted_candidate or NULL_SELECTION,
                "predicted_candidate_id": predicted_candidate,
                "candidate_ids": candidate_ids,
                "candidate_logits": candidate_logits_by_id,
                "candidate_positions": candidate_positions,
                "null_logit": float(output.null_logits[output_index, local_frame]),
                "candidate_count": len(candidate_ids),
                "candidate_inputs_masked": candidates_masked,
                "selection_loss": selection_loss,
                "inplay_target": inplay_target,
                "inplay_logit": inplay_logit,
                "in_play_probability": inplay_probability,
                "inplay_loss": inplay_loss,
                **boundary_values,
                "decoded_inplay": (
                    inplay_probability >= 0.5
                    if inplay_probability is not None
                    else None
                ),
                "selected_position": selected_position,
                "candidate_artifact_sha256": window.metadata.get("candidate_sha256"),
                "annotation_artifact_sha256": window.metadata.get("annotation_sha256"),
                "rally_intervals_sha256": window.metadata.get("rally_intervals_sha256"),
            }
        )
    return records


def _predict_windows(
    model: TemporalShuttleSelector,
    windows: Sequence[SelectorWindow],
    fold: CrossFitFold,
    *,
    device: torch.device,
    batch_size: int,
    owned_only: bool = True,
    mask_candidates: bool = False,
) -> list[dict[str, Any]]:
    """Predict windows in batches and transfer logits to CPU once per batch."""
    records: list[dict[str, Any]] = []
    for start in range(0, len(windows), batch_size):
        window_batch = windows[start : start + batch_size]
        batch = collate_selector_windows(window_batch).to(device)
        if mask_candidates:
            batch = _mask_candidate_inputs(batch)
        with torch.no_grad():
            device_output = model(batch)
        output = replace(
            device_output,
            candidate_logits=device_output.candidate_logits.cpu(),
            null_logits=device_output.null_logits.cpu(),
            inplay_logits=(
                device_output.inplay_logits.cpu()
                if device_output.inplay_logits is not None
                else None
            ),
            rally_start_logits=(
                device_output.rally_start_logits.cpu()
                if device_output.rally_start_logits is not None else None
            ),
            rally_end_logits=(
                device_output.rally_end_logits.cpu()
                if device_output.rally_end_logits is not None else None
            ),
        )
        for output_index, window in enumerate(window_batch):
            records.extend(
                _prediction_records_for_window(
                    output,
                    output_index,
                    window,
                    fold,
                    owned_only=owned_only,
                    candidates_masked=mask_candidates,
                )
            )
    return records


def _predict_window(
    model: TemporalShuttleSelector,
    window: SelectorWindow,
    fold: CrossFitFold,
    *,
    device: torch.device,
    owned_only: bool = True,
    mask_candidates: bool = False,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for callers predicting a single window."""
    return _predict_windows(
        model,
        [window],
        fold,
        device=device,
        batch_size=1,
        owned_only=owned_only,
        mask_candidates=mask_candidates,
    )


def aggregate_joint_predictions(
    observations: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Center-weight repeated frame/candidate logits by stable bookkeeping ID."""
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in observations:
        grouped.setdefault(
            (str(row["prediction_source_id"]), int(row["frame"])), []
        ).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        owners = [row for row in rows if bool(row.get("owned"))]
        if len(owners) != 1:
            raise ValueError(f"joint frame must have exactly one loss owner: {key}")
        owner = owners[0]
        weights = [float(row["aggregation_weight"]) for row in rows]
        weight_total = sum(weights)
        if weight_total <= 0:
            raise ValueError("joint aggregation weights must be positive")
        inplay_logit = sum(
            float(row["inplay_logit"]) * weight for row, weight in zip(rows, weights)
        ) / weight_total
        boundary_logits = {}
        for name in ("rally_start", "rally_end"):
            values = [row.get(f"{name}_logit") for row in rows]
            boundary_logits[name] = (
                sum(float(value) * weight for value, weight in zip(values, weights))
                / weight_total
                if all(value is not None for value in values) else None
            )
        null_logit = sum(
            float(row["null_logit"]) * weight for row, weight in zip(rows, weights)
        ) / weight_total
        candidate_ids = list(owner["candidate_ids"])
        candidate_logits: dict[str, float] = {}
        candidate_positions: dict[str, Any] = {}
        for candidate_id in candidate_ids:
            values = [
                (float(row["candidate_logits"][candidate_id]), weight)
                for row, weight in zip(rows, weights)
                if candidate_id in row["candidate_logits"]
            ]
            candidate_logits[candidate_id] = sum(
                value * weight for value, weight in values
            ) / sum(weight for _, weight in values)
            candidate_positions[candidate_id] = next(
                row["candidate_positions"][candidate_id]
                for row in rows
                if candidate_id in row["candidate_positions"]
            )
        frame_logits = [candidate_logits[item] for item in candidate_ids] + [null_logit]
        predicted_index = max(range(len(frame_logits)), key=frame_logits.__getitem__)
        predicted_candidate = (
            None
            if predicted_index == len(candidate_ids)
            else candidate_ids[predicted_index]
        )
        record = dict(owner)
        record.update(
            {
                "aggregation_observation_count": len(rows),
                "aggregation_weight_sum": weight_total,
                "candidate_logits": candidate_logits,
                "null_logit": null_logit,
                "inplay_logit": inplay_logit,
                "in_play_probability": float(torch.sigmoid(torch.tensor(inplay_logit))),
                **{
                    f"{name}_logit": logit
                    for name, logit in boundary_logits.items()
                },
                **{
                    f"{name}_probability": (
                        float(torch.sigmoid(torch.tensor(logit)))
                        if logit is not None else None
                    )
                    for name, logit in boundary_logits.items()
                },
                "predicted_candidate_id": predicted_candidate,
                "predicted_selection_outcome": predicted_candidate or NULL_SELECTION,
                "predicted_outcome": predicted_candidate or NULL_SELECTION,
                "selected_position": (
                    candidate_positions[predicted_candidate]
                    if predicted_candidate is not None
                    else None
                ),
            }
        )
        target = int(owner["inplay_target"])
        record["inplay_loss"] = (
            float(
                F.binary_cross_entropy_with_logits(
                    torch.tensor([inplay_logit]), torch.tensor([float(target)])
                )
            )
            if target in (0, 1)
            else None
        )
        if bool(owner.get("candidate_inputs_masked")):
            target_index = None
        elif owner["true_status"] == "selected_retained":
            target_index = candidate_ids.index(str(owner["target_candidate_id"]))
        elif owner["true_status"] == "null":
            target_index = len(candidate_ids)
        else:
            target_index = None
        record["selection_loss"] = (
            float(
                F.cross_entropy(
                    torch.tensor(frame_logits).view(1, -1),
                    torch.tensor([target_index]),
                )
            )
            if target_index is not None
            else None
        )
        output.append(record)
    return output


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
    if len(folds) < 2 or any(
        set(fold.training_source_ids) & set(fold.evaluation_source_ids)
        for fold in folds
    ):
        raise ValueError("cross-fit manifest must contain source-disjoint folds")
    if any(not fold.training_source_ids or not fold.evaluation_source_ids for fold in folds):
        raise ValueError("cross-fit folds require training and held-out sources")
    universe = set(folds[0].training_source_ids + folds[0].evaluation_source_ids)
    evaluated = [source for fold in folds for source in fold.evaluation_source_ids]
    if (
        len(evaluated) != len(set(evaluated))
        or set(evaluated) != universe
        or any(
            set(fold.training_source_ids) | set(fold.evaluation_source_ids) != universe
            for fold in folds
        )
    ):
        raise ValueError("cross-fit folds must hold out every source exactly once")
    payload = {"seed": manifest.seed, "folds": [vars(fold) for fold in folds]}
    expected_fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.fingerprint != expected_fingerprint:
        raise ValueError("cross-fit manifest topology or fingerprint is invalid")
    return manifest


def load_dataset_config(
    path: Path, *, context_mode: ContextMode, pose_coordinate_mode: PoseCoordinateMode = "image"
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
        pose_coordinate_mode=pose_coordinate_mode,
        expected_annotation_sha256=data.get("expected_annotation_sha256"),
        rally_intervals_path=(
            _resolve(path, data["rally_intervals_path"])
            if data.get("rally_intervals_path")
            else None
        ),
        rally_manifest_path=(
            _resolve(path, data["rally_manifest_path"])
            if data.get("rally_manifest_path")
            else None
        ),
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
