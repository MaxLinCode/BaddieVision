"""Evaluate predicted rally intervals against labeled intervals."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

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
    predictions: list[Interval], labels: list[Interval], threshold: float = 0.5
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
    }
    return metrics, details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--matches", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not 0 <= args.iou_threshold <= 1:
        parser.error("--iou-threshold must be between 0 and 1")
    metrics, details = evaluate(
        read_intervals(args.predictions), read_intervals(args.labels), args.iou_threshold
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
