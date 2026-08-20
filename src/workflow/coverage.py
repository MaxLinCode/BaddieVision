"""Reviewed rally coverage and source-local InPlay target derivation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, order=True)
class ReviewedSpan:
    source_id: str
    start_frame: int
    end_frame: int

    def validate(self, frame_count: int) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("reviewed spans require 0 <= start_frame <= end_frame")
        if self.end_frame >= frame_count:
            raise ValueError("reviewed span lies outside the source-local frame domain")


def normalize_reviewed_spans(
    source_id: str,
    frame_count: int,
    spans: Iterable[ReviewedSpan] = (),
    *,
    full_source: bool = False,
) -> tuple[ReviewedSpan, ...]:
    if full_source:
        if tuple(spans):
            raise ValueError("full_source cannot be combined with explicit reviewed spans")
        return (ReviewedSpan(source_id, 0, frame_count - 1),)
    ordered = sorted(spans)
    previous_end = -1
    for span in ordered:
        if span.source_id != source_id:
            raise ValueError("reviewed span source differs from requested source")
        span.validate(frame_count)
        if span.start_frame <= previous_end:
            raise ValueError("reviewed spans overlap")
        previous_end = span.end_frame
    return tuple(ordered)


def read_reviewed_spans(path: str | Path) -> tuple[ReviewedSpan, ...]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"source_id", "start_frame", "end_frame"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("reviewed coverage CSV is missing required columns")
        return tuple(
            ReviewedSpan(str(row["source_id"]), int(row["start_frame"]), int(row["end_frame"]))
            for row in reader
        )


def derive_inplay_targets(
    frame_count: int,
    reviewed_spans: Iterable[ReviewedSpan],
    rally_intervals: Iterable[tuple[int, int]],
) -> list[int]:
    """Return 1 in rallies, 0 in reviewed non-rally frames, and -100 elsewhere."""
    targets = [-100] * frame_count
    for span in reviewed_spans:
        span.validate(frame_count)
        for frame in range(span.start_frame, span.end_frame + 1):
            targets[frame] = 0
    previous_end = -1
    for start, end in sorted(rally_intervals):
        if start < 0 or end < start or end >= frame_count:
            raise ValueError("rally interval lies outside the source-local frame domain")
        if start <= previous_end:
            raise ValueError("rally intervals overlap")
        previous_end = end
        if any(targets[frame] == -100 for frame in range(start, end + 1)):
            raise ValueError("rally interval is not contained in reviewed coverage")
        for frame in range(start, end + 1):
            targets[frame] = 1
    return targets
