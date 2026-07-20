"""Lean source-disjoint training and evaluation for the temporal selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from InPlay.heuristic.evaluate import Interval, evaluate as evaluate_intervals

from .batch import MASKED_TARGET, NULL_TARGET, SelectorBatch
from .config import ContextMode, SelectorConfig
from .crossfit import (
    CrossFitFold,
    CrossFitManifest,
    build_leave_one_source_out,
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
from .joint_inference import (
    InPlayDecoderConfig,
    calibrate_inplay_decoder,
    decode_inplay_probabilities,
    decoded_intervals,
    write_joint_artifacts,
)
from .model import JointRallyShuttleModel, TemporalShuttleSelector

NULL_SELECTION = "NULL_SELECTION"
NOT_REQUIRED = "not_required"
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
    return replace(
        batch,
        candidate_mask=candidate_mask,
        candidate_frame_indices=candidate_frames,
        targets=selection_targets,
    ).validate(frame_feature_dim=batch.frame_values.shape[-1])


def _train_fold(
    model: TemporalShuttleSelector,
    windows: Sequence[SelectorWindow],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    candidate_dropout: float = 0.2,
) -> list[dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    dropout_generator = torch.Generator().manual_seed(seed + 10_000)
    loader = DataLoader(
        list(windows),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_selector_windows,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    joint = bool(windows[0].inplay_targets is not None)
    if any((window.inplay_targets is not None) != joint for window in windows):
        raise ValueError("training fold mixes legacy and joint windows")
    pos_weight = _inplay_pos_weight(windows, device) if joint else None
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        totals, selections, inplays = [], [], []
        for batch in loader:
            batch = batch.to(device)
            if joint:
                batch = _apply_candidate_dropout(
                    batch,
                    generator=dropout_generator,
                    probability=candidate_dropout,
                )
            optimizer.zero_grad(set_to_none=True)
            if joint:
                components = model.joint_losses(
                    batch, inplay_pos_weight=pos_weight
                )
                loss = components.total
                selections.append(float(components.selection.detach().cpu()))
                inplays.append(float(components.inplay.detach().cpu()))
            else:
                loss = model.loss(batch)
                selections.append(float(loss.detach().cpu()))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("training produced a non-finite joint loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(loss.detach().cpu()))
        if not totals:
            raise ValueError("training fold has no windows")
        summary = {
            "total": sum(totals) / len(totals),
            "selection": sum(selections) / len(selections),
            "inplay": sum(inplays) / len(inplays) if inplays else 0.0,
        }
        history.append(summary)
        print(
            f"epoch {epoch + 1}/{epochs}: total={summary['total']:.6f} "
            f"selection={summary['selection']:.6f} inplay={summary['inplay']:.6f}",
            flush=True,
        )
    return history


def _predict_window(
    model: TemporalShuttleSelector,
    window: SelectorWindow,
    fold: CrossFitFold,
    *,
    device: torch.device,
    owned_only: bool = True,
) -> list[dict[str, Any]]:
    batch = collate_selector_windows([window]).to(device)
    with torch.no_grad():
        output = model(batch)
    records = []
    max_time = max(
        (abs(float(value)) for value in window.relative_time_seconds),
        default=0.0,
    )
    for local_frame, frame in enumerate(window.frame_indices):
        owned = frame in window.owned_frames
        if owned_only and not owned:
            continue
        slots = torch.nonzero(
            window.candidate_frame_indices == local_frame, as_tuple=False
        ).flatten().tolist()
        candidate_ids = [window.candidate_ids[slot] for slot in slots]
        candidate_logits_by_id = {
            candidate_id: float(output.candidate_logits[0, slot].cpu())
            for candidate_id, slot in zip(candidate_ids, slots)
        }
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
        predicted_slot = None if predicted_candidate is None else slots[predicted_index]
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
        inplay_target = (
            int(window.inplay_targets[local_frame])
            if window.inplay_targets is not None
            else None
        )
        inplay_logit = (
            float(output.inplay_logits[0, local_frame].cpu())
            if output.inplay_logits is not None
            else None
        )
        inplay_probability = (
            float(torch.sigmoid(output.inplay_logits[0, local_frame]).cpu())
            if output.inplay_logits is not None
            else None
        )
        inplay_loss = None
        if inplay_target in (0, 1) and output.inplay_logits is not None:
            inplay_loss = float(
                F.binary_cross_entropy_with_logits(
                    output.inplay_logits[0, local_frame].view(1),
                    torch.tensor([float(inplay_target)], device=device),
                ).cpu()
            )
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
                "true_outcome": true_outcome,
                "target_candidate_id": target_candidate,
                "predicted_outcome": predicted_candidate or NULL_SELECTION,
                "predicted_selection_outcome": predicted_candidate or NULL_SELECTION,
                "predicted_candidate_id": predicted_candidate,
                "candidate_ids": candidate_ids,
                "candidate_logits": candidate_logits_by_id,
                "candidate_positions": candidate_positions,
                "null_logit": float(output.null_logits[0, local_frame].cpu()),
                "candidate_count": len(candidate_ids),
                "selection_loss": selection_loss,
                "inplay_target": inplay_target,
                "inplay_logit": inplay_logit,
                "in_play_probability": inplay_probability,
                "inplay_loss": inplay_loss,
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
        record["inplay_loss"] = float(
            F.binary_cross_entropy_with_logits(
                torch.tensor([inplay_logit]), torch.tensor([float(target)])
            )
        )
        if owner["true_status"] == "selected_retained":
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
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
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
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + fold_index,
        )
        model.eval()
        decoder_config: InPlayDecoderConfig | None = None
        if joint:
            calibration_rows = aggregate_joint_predictions(
                record
                for window in train_windows
                for record in _predict_window(
                    model, window, fold, device=device, owned_only=False
                )
            )
            calibration_rows.sort(
                key=lambda row: (row["prediction_source_id"], row["frame"])
            )
            decoder_config = calibrate_inplay_decoder(
                [float(row["in_play_probability"]) for row in calibration_rows],
                [int(row["inplay_target"]) for row in calibration_rows],
                fps=float(train_windows[0].metadata["fps"]),
            )
        fold_observations = [
            record
            for window in evaluation_windows
            for record in _predict_window(
                model,
                window,
                fold,
                device=device,
                owned_only=not joint,
            )
        ]
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
                "candidate_dropout_probability": 0.2 if joint else 0.0,
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
            "strict_inplay_with_conditional_shuttle_selection"
            if joint
            else "continuous_shuttle_selection"
        ),
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
            for queue in sorted({str(row["label_queue"]) for row in all_predictions})
        },
        "crossfit_macro": _macro_metrics(list(fold_metrics.values())),
    }
    if len(manifest.folds) == 2:
        metrics["two_fold_macro"] = metrics["crossfit_macro"]
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
    if len(folds) < 2 or any(
        set(fold.training_source_ids) & set(fold.evaluation_source_ids)
        for fold in folds
    ):
        raise ValueError("cross-fit manifest must contain source-disjoint folds")
    if any(
        not fold.training_source_ids or len(fold.evaluation_source_ids) != 1
        for fold in folds
    ):
        raise ValueError("cross-fit folds require training sources and one held-out source")
    if len(folds) == 2 and all(len(fold.training_source_ids) == 1 for fold in folds):
        expected = build_two_source_crossfit(
            (folds[0].training_source_ids[0], folds[0].evaluation_source_ids[0]),
            seed=manifest.seed,
        )
    else:
        expected = build_leave_one_source_out(
            tuple(fold.evaluation_source_ids[0] for fold in folds),
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
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset_config, manifest = load_dataset_config(
        args.config, context_mode=args.context_mode
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
    )
    print(json.dumps(metrics["crossfit_macro"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
