"""Evaluate predicted rally intervals against labeled intervals."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

LABEL_FIELDS = {"source_id", "rally_id", "start_frame", "end_frame"}


@dataclass(frozen=True)
class Interval:
    source_id: str
    rally_id: str
    start: int
    end: int


def read_intervals(path: str | Path, include_rejected: bool = False) -> list[Interval]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = LABEL_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"interval CSV missing columns: {sorted(missing)}")
        intervals = []
        for row_number, row in enumerate(reader, 2):
            if not include_rejected and row.get("status") == "rejected":
                continue
            start, end = int(row["start_frame"]), int(row["end_frame"])
            if start < 0 or end < start:
                raise ValueError(f"row {row_number}: invalid inclusive interval")
            intervals.append(Interval(row["source_id"], row["rally_id"], start, end))
        return intervals


def interval_iou(left: Interval, right: Interval) -> float:
    if left.source_id != right.source_id:
        return 0.0
    intersection = max(0, min(left.end, right.end) - max(left.start, right.start) + 1)
    union = left.end - left.start + 1 + right.end - right.start + 1 - intersection
    return intersection / union


def evaluate(
    predictions: list[Interval],
    labels: list[Interval],
    threshold: float = 0.5,
    frame_ranges: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    # Maximum-weight one-to-one assignment. SciPy is already a runtime dependency.
    from scipy.optimize import linear_sum_assignment

    candidates = [
        [interval_iou(prediction, label) for label in labels]
        for prediction in predictions
    ]
    pairs: list[tuple[int, int]] = []
    if predictions and labels:
        rows, columns = linear_sum_assignment(candidates, maximize=True)
        pairs = [
            (int(row), int(column))
            for row, column in zip(rows, columns)
            if candidates[row][column] >= threshold
        ]
    matched_predictions = {row for row, _ in pairs}
    matched_labels = {column for _, column in pairs}
    details: list[dict[str, object]] = []
    boundary_errors: list[float] = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    for row, column in pairs:
        prediction, label = predictions[row], labels[column]
        start_error = prediction.start - label.start
        end_error = prediction.end - label.end
        start_errors.append(start_error)
        end_errors.append(end_error)
        boundary_errors.extend([abs(start_error), abs(end_error)])
        details.append(
            {
                "source_id": prediction.source_id,
                "prediction_rally_id": prediction.rally_id,
                "label_rally_id": label.rally_id,
                "iou": round(candidates[row][column], 6),
                "start_error": start_error,
                "end_error": end_error,
                "match_status": "matched",
            }
        )
    for index, item in enumerate(predictions):
        if index not in matched_predictions:
            details.append(
                {
                    "source_id": item.source_id,
                    "prediction_rally_id": item.rally_id,
                    "label_rally_id": "",
                    "iou": 0.0,
                    "start_error": "",
                    "end_error": "",
                    "match_status": "unmatched_prediction",
                }
            )
    for index, item in enumerate(labels):
        if index not in matched_labels:
            details.append(
                {
                    "source_id": item.source_id,
                    "prediction_rally_id": "",
                    "label_rally_id": item.rally_id,
                    "iou": 0.0,
                    "start_error": "",
                    "end_error": "",
                    "match_status": "unmatched_label",
                }
            )
    true_positive = len(pairs)
    precision = true_positive / len(predictions) if predictions else 0.0
    recall = true_positive / len(labels) if labels else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    split_count = sum(
        1
        for label in labels
        if sum(
            prediction.source_id == label.source_id and _overlap_frames(prediction, label) > 0
            for prediction in predictions
        )
        > 1
    )
    merge_count = sum(
        1
        for prediction in predictions
        if sum(
            prediction.source_id == label.source_id and _overlap_frames(prediction, label) > 0
            for label in labels
        )
        > 1
    )
    metrics: dict[str, object] = {
        "iou_threshold": threshold,
        "prediction_count": len(predictions),
        "label_count": len(labels),
        "matched_count": true_positive,
        "unmatched_prediction_count": len(predictions) - true_positive,
        "unmatched_label_count": len(labels) - true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_start_boundary_error": statistics.fmean(start_errors) if start_errors else None,
        "mean_end_boundary_error": statistics.fmean(end_errors) if end_errors else None,
        "mean_absolute_boundary_error": statistics.fmean(boundary_errors) if boundary_errors else None,
        "median_absolute_boundary_error": statistics.median(boundary_errors) if boundary_errors else None,
        "false_split_count": split_count,
        "false_merge_count": merge_count,
    }
    if frame_ranges is not None:
        metrics.update(frame_classification_metrics(predictions, labels, frame_ranges))
    return metrics, details


def frame_classification_metrics(
    predictions: list[Interval],
    labels: list[Interval],
    frame_ranges: Mapping[str, tuple[int, int]],
) -> dict[str, object]:
    """Compute frame-level precision/recall/F1 over inclusive source ranges."""

    tp = fp = fn = tn = 0
    for source_id, (start, end) in frame_ranges.items():
        if start < 0 or end < start:
            raise ValueError(f"invalid frame range for {source_id!r}")
        pred_mask = _mask_for_source(predictions, source_id, start, end)
        label_mask = _mask_for_source(labels, source_id, start, end)
        tp += int((pred_mask & label_mask).sum())
        fp += int((pred_mask & ~label_mask).sum())
        fn += int((~pred_mask & label_mask).sum())
        tn += int((~pred_mask & ~label_mask).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "frame_true_positive": tp,
        "frame_false_positive": fp,
        "frame_false_negative": fn,
        "frame_true_negative": tn,
        "frame_precision": precision,
        "frame_recall": recall,
        "frame_f1": f1,
    }


def _mask_for_source(
    intervals: list[Interval], source_id: str, start: int, end: int
) -> np.ndarray:
    mask = np.zeros(end - start + 1, dtype=bool)
    for interval in intervals:
        if interval.source_id != source_id:
            continue
        left, right = max(start, interval.start), min(end, interval.end)
        if left <= right:
            mask[left - start : right - start + 1] = True
    return mask


def _overlap_frames(left: Interval, right: Interval) -> int:
    if left.source_id != right.source_id:
        return 0
    return max(0, min(left.end, right.end) - max(left.start, right.start) + 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--matches", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--frame-ranges",
        help="optional JSON object mapping source_id to inclusive [start, end] frames",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.iou_threshold <= 1:
        parser.error("--iou-threshold must be between 0 and 1")
    frame_ranges = None
    if args.frame_ranges:
        raw_ranges = json.loads(Path(args.frame_ranges).read_text(encoding="utf-8"))
        frame_ranges = {
            str(source): (int(values[0]), int(values[1]))
            for source, values in raw_ranges.items()
        }
    metrics, details = evaluate(
        read_intervals(args.predictions),
        read_intervals(args.labels),
        args.iou_threshold,
        frame_ranges,
    )
    Path(args.metrics).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "source_id", "prediction_rally_id", "label_rally_id", "iou",
        "start_error", "end_error", "match_status",
    ]
    with Path(args.matches).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
