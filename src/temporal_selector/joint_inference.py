"""Deterministic decoding and output contracts for joint rally predictions."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class InPlayDecoderConfig:
    threshold: float = 0.5
    max_gap_seconds: float = 0.2
    minimum_duration_seconds: float = 0.5
    preserve_edge_runs: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError("InPlay threshold must be between zero and one")
        if self.max_gap_seconds < 0 or self.minimum_duration_seconds < 0:
            raise ValueError("decoder durations cannot be negative")


def _runs(values: Sequence[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate((*values, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def decode_inplay_probabilities(
    probabilities: Sequence[float],
    *,
    fps: float,
    config: InPlayDecoderConfig | None = None,
) -> tuple[bool, ...]:
    """Threshold, bridge short gaps, and reject implausibly short rallies."""
    config = config or InPlayDecoderConfig()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in probabilities):
        raise ValueError("InPlay probabilities must be finite values in [0, 1]")
    decoded = [float(value) >= config.threshold for value in probabilities]
    max_gap = round(config.max_gap_seconds * fps)
    if max_gap:
        for left, right in zip(_runs(decoded), _runs(decoded)[1:]):
            gap_start, gap_end = left[1] + 1, right[0] - 1
            if gap_end - gap_start + 1 <= max_gap:
                decoded[gap_start : gap_end + 1] = [True] * (gap_end - gap_start + 1)
    minimum = max(1, round(config.minimum_duration_seconds * fps))
    for start, end in _runs(decoded):
        if config.preserve_edge_runs and (start == 0 or end == len(decoded) - 1):
            continue
        if end - start + 1 < minimum:
            decoded[start : end + 1] = [False] * (end - start + 1)
    return tuple(decoded)


def _binary_f1(targets: Sequence[int], predictions: Sequence[bool]) -> float:
    tp = sum(target == 1 and prediction for target, prediction in zip(targets, predictions))
    fp = sum(target == 0 and prediction for target, prediction in zip(targets, predictions))
    fn = sum(target == 1 and not prediction for target, prediction in zip(targets, predictions))
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def calibrate_inplay_decoder(
    probabilities: Sequence[float],
    targets: Sequence[int],
    *,
    fps: float,
) -> InPlayDecoderConfig:
    """Select a decoder only from labeled training-source frames."""
    if len(probabilities) != len(targets) or not probabilities:
        raise ValueError("decoder calibration needs aligned nonempty probabilities and targets")
    if any(target not in (0, 1) for target in targets):
        raise ValueError("decoder calibration targets must be binary")
    candidates = (
        InPlayDecoderConfig(threshold / 100, gap, minimum)
        for threshold in range(30, 71, 5)
        for gap in (0.0, 0.1, 0.2, 0.3, 0.5)
        for minimum in (0.25, 0.5, 1.0)
    )
    return max(
        candidates,
        key=lambda item: (
            _binary_f1(
                targets,
                decode_inplay_probabilities(probabilities, fps=fps, config=item),
            ),
            -abs(item.threshold - 0.5),
            -item.max_gap_seconds,
            -item.minimum_duration_seconds,
        ),
    )


def decoded_intervals(
    frames: Sequence[int], decoded: Sequence[bool]
) -> tuple[tuple[int, int], ...]:
    if len(frames) != len(decoded):
        raise ValueError("frames and decoded states must align")
    if any(right != left + 1 for left, right in zip(frames, frames[1:])):
        raise ValueError("decoded frame sequence must be contiguous")
    return tuple((frames[start], frames[end]) for start, end in _runs(decoded))


def write_joint_artifacts(
    rows: Iterable[Mapping[str, object]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write one aligned frame record plus canonical decoded rally intervals."""
    output_dir = Path(output_dir)
    values = sorted(
        (dict(row) for row in rows),
        key=lambda row: (str(row["prediction_source_id"]), int(row["frame"])),
    )
    frame_path = output_dir / "rally_shuttle_predictions.jsonl"
    metadata = {
        "type": "metadata",
        "schema": "rally_shuttle_predictions",
        "schema_version": 1,
        "task_definition": "strict_inplay_with_conditional_shuttle_selection",
        "source_ids": sorted(
            {str(row["prediction_source_id"]) for row in values}
        ),
        "candidate_artifact_sha256": {
            str(row["prediction_source_id"]): row.get("candidate_artifact_sha256")
            for row in values
        },
        "rally_intervals_sha256": sorted(
            {
                str(row["rally_intervals_sha256"])
                for row in values
                if row.get("rally_intervals_sha256")
            }
        ),
        "checkpoint_sha256": {
            str(row["fold_id"]): row.get("checkpoint_sha256")
            for row in values
            if row.get("fold_id") is not None
        },
    }
    frame_path.write_text(
        json.dumps(metadata, sort_keys=True)
        + "\n"
        + "".join(json.dumps(row, sort_keys=True) + "\n" for row in values),
        encoding="utf-8",
    )
    rally_path = output_dir / "rallies.csv"
    with rally_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_id", "rally_id", "start_frame", "end_frame"),
        )
        writer.writeheader()
        sources = sorted({str(row["prediction_source_id"]) for row in values})
        for source_id in sources:
            source_rows = [row for row in values if row["prediction_source_id"] == source_id]
            frames = [int(row["frame"]) for row in source_rows]
            decoded = [bool(row["decoded_inplay"]) for row in source_rows]
            for number, (start, end) in enumerate(decoded_intervals(frames, decoded), 1):
                writer.writerow(
                    {
                        "source_id": source_id,
                        "rally_id": f"{source_id}-{number:04d}",
                        "start_frame": start,
                        "end_frame": end,
                    }
                )
    return frame_path, rally_path
