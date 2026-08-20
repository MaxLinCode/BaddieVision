"""Fingerprint-validated strict binary rally interval annotations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


INTERVAL_FIELDS = ("source_id", "rally_id", "start_frame", "end_frame")
BOUNDARY_DEFINITION = "serve_contact_through_terminal_event_inclusive"


@dataclass(frozen=True, order=True)
class RallyCoverageSpan:
    source_id: str
    start_frame: int
    end_frame: int

    def contains(self, frame: int) -> bool:
        return self.start_frame <= int(frame) <= self.end_frame


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True, order=True)
class RallyInterval:
    source_id: str
    rally_id: str
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if not self.source_id or not self.rally_id:
            raise ValueError("rally source_id and rally_id must not be empty")
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("rally interval must be an inclusive non-negative range")

    def contains(self, frame: int) -> bool:
        return self.start_frame <= int(frame) <= self.end_frame


@dataclass(frozen=True)
class RallySourceManifest:
    source_id: str
    video_sha256: str
    fps: float
    frame_count: int

    def __post_init__(self) -> None:
        if (
            not self.source_id
            or len(self.video_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.video_sha256.lower())
        ):
            raise ValueError("rally source manifest has an invalid ID or SHA-256")
        if not math.isfinite(self.fps) or self.fps <= 0 or self.frame_count <= 0:
            raise ValueError("rally source manifest has invalid FPS/frame count")


@dataclass(frozen=True)
class RallyIntervalIndex:
    intervals: tuple[RallyInterval, ...]
    sources: Mapping[str, RallySourceManifest]
    intervals_sha256: str
    annotation_revision: str
    reviewed_coverage: tuple[RallyCoverageSpan, ...] = ()
    partial_start_rally_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        grouped: dict[str, list[RallyInterval]] = {}
        ids: set[tuple[str, str]] = set()
        for interval in self.intervals:
            if interval.source_id not in self.sources:
                raise ValueError(
                    f"rally interval references unknown source: {interval.source_id}"
                )
            key = interval.source_id, interval.rally_id
            if key in ids:
                raise ValueError(f"duplicate rally ID for source: {key}")
            ids.add(key)
            source = self.sources[interval.source_id]
            if interval.end_frame >= source.frame_count:
                raise ValueError(
                    f"rally interval exceeds source frame count: {interval.rally_id}"
                )
            grouped.setdefault(interval.source_id, []).append(interval)
        for source_id, values in grouped.items():
            values.sort(key=lambda item: (item.start_frame, item.end_frame))
            for previous, current in zip(values, values[1:]):
                if current.start_frame <= previous.end_frame:
                    raise ValueError(f"overlapping rally intervals for source {source_id}")
        coverage_by_source: dict[str, list[RallyCoverageSpan]] = {}
        for span in self.reviewed_coverage:
            if span.source_id not in self.sources:
                raise ValueError(f"reviewed coverage references unknown source: {span.source_id}")
            if span.start_frame < 0 or span.end_frame < span.start_frame:
                raise ValueError("reviewed coverage must be an inclusive non-negative range")
            if span.end_frame >= self.sources[span.source_id].frame_count:
                raise ValueError("reviewed coverage exceeds source frame count")
            coverage_by_source.setdefault(span.source_id, []).append(span)
        for source_id, spans in coverage_by_source.items():
            spans.sort(key=lambda item: (item.start_frame, item.end_frame))
            for previous, current in zip(spans, spans[1:]):
                if current.start_frame <= previous.end_frame:
                    raise ValueError(f"overlapping reviewed coverage for source {source_id}")
        for interval in self.intervals:
            if not any(
                span.source_id == interval.source_id
                and span.start_frame <= interval.start_frame
                and span.end_frame >= interval.end_frame
                for span in self.reviewed_coverage
            ):
                raise ValueError(f"rally interval is outside reviewed coverage: {interval.rally_id}")
        known_ids = {item.rally_id for item in self.intervals}
        unknown_partial = self.partial_start_rally_ids - known_ids
        if unknown_partial:
            raise ValueError(f"partial-start metadata references unknown rallies: {sorted(unknown_partial)}")
        for interval in self.intervals:
            if interval.rally_id in self.partial_start_rally_ids and interval.start_frame != 0:
                raise ValueError("a rally starting before the source must begin at frame 0")

    def is_inplay(self, source_id: str, frame: int) -> bool:
        return any(
            interval.contains(frame)
            for interval in self.intervals
            if interval.source_id == source_id
        )

    def is_reviewed(self, source_id: str, frame: int) -> bool:
        return any(
            span.source_id == source_id and span.contains(frame)
            for span in self.reviewed_coverage
        )

    def target(self, source_id: str, frame: int) -> int:
        if source_id not in self.sources:
            raise ValueError(f"unknown rally source: {source_id}")
        if not self.is_reviewed(source_id, frame):
            return -100
        return int(self.is_inplay(source_id, frame))

    def targets(self, source_id: str, frames: Sequence[int]) -> tuple[int, ...]:
        if source_id not in self.sources:
            raise ValueError(f"unknown rally source: {source_id}")
        return tuple(self.target(source_id, frame) for frame in frames)

    def for_source(self, source_id: str) -> tuple[RallyInterval, ...]:
        return tuple(item for item in self.intervals if item.source_id == source_id)

    def starts_before_source(self, rally_id: str) -> bool:
        return rally_id in self.partial_start_rally_ids


def read_rally_intervals(
    intervals_path: str | Path,
    manifest_path: str | Path,
) -> RallyIntervalIndex:
    intervals_path, manifest_path = Path(intervals_path), Path(manifest_path)
    with intervals_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != set(INTERVAL_FIELDS):
            raise ValueError(f"rally interval CSV must contain {INTERVAL_FIELDS}")
        intervals = tuple(
            RallyInterval(
                source_id=str(row["source_id"]).strip(),
                rally_id=str(row["rally_id"]).strip(),
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
            )
            for row in reader
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "rally_interval_manifest"
        or manifest.get("schema_version") not in {1, 2, 3}
        or manifest.get("boundary_definition") != BOUNDARY_DEFINITION
    ):
        raise ValueError("expected strict rally_interval_manifest schema v1/v2/v3")
    digest = file_sha256(intervals_path)
    if manifest.get("intervals_sha256") != digest:
        raise ValueError("rally interval CSV fingerprint mismatch")
    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("rally interval manifest must contain source metadata")
    sources = {
        str(item["source_id"]): RallySourceManifest(
            source_id=str(item["source_id"]),
            video_sha256=str(item["video_sha256"]),
            fps=float(item["fps"]),
            frame_count=int(item["frame_count"]),
        )
        for item in raw_sources
    }
    if len(sources) != len(raw_sources):
        raise ValueError("duplicate sources in rally interval manifest")
    raw_coverage = manifest.get("reviewed_coverage")
    if raw_coverage is None:
        # Schema v1 did not establish that non-rally frames were reviewed. Its
        # only safe reviewed domain is the strict rally intervals themselves.
        coverage = tuple(
            RallyCoverageSpan(item.source_id, item.start_frame, item.end_frame)
            for item in intervals
        )
    else:
        coverage_values: list[RallyCoverageSpan] = []
        for source_id, declaration in raw_coverage.items():
            if declaration == "full_source":
                coverage_values.append(RallyCoverageSpan(source_id, 0, sources[source_id].frame_count - 1))
            else:
                coverage_values.extend(
                    RallyCoverageSpan(source_id, int(span[0]), int(span[1]))
                    for span in declaration
                )
        coverage = tuple(sorted(coverage_values))
    return RallyIntervalIndex(
        tuple(sorted(intervals)),
        sources,
        digest,
        str(manifest.get("annotation_revision", "")),
        coverage,
        frozenset(str(value) for value in manifest.get("partial_start_rally_ids", ())),
    )


def write_rally_intervals(
    intervals_path: str | Path,
    manifest_path: str | Path,
    intervals: Iterable[RallyInterval],
    sources: Iterable[RallySourceManifest],
    *,
    annotation_revision: str,
    reviewed_coverage: Mapping[str, str | Sequence[Sequence[int]]] | None = None,
    partial_start_rally_ids: Iterable[str] = (),
) -> RallyIntervalIndex:
    """Write canonical intervals and their provenance manifest."""
    intervals_path, manifest_path = Path(intervals_path), Path(manifest_path)
    values = tuple(sorted(intervals))
    source_values = tuple(sorted(sources, key=lambda item: item.source_id))
    # Validate before writing, using a temporary digest placeholder.
    coverage_mapping = reviewed_coverage or {
        source.source_id: [[0, source.frame_count - 1]] for source in source_values
    }
    coverage_values = []
    for source in source_values:
        declaration = coverage_mapping.get(source.source_id, [])
        if declaration == "full_source":
            coverage_values.append(RallyCoverageSpan(source.source_id, 0, source.frame_count - 1))
        else:
            coverage_values.extend(
                RallyCoverageSpan(source.source_id, int(span[0]), int(span[1]))
                for span in declaration
            )
    partial_ids = frozenset(str(value) for value in partial_start_rally_ids)
    RallyIntervalIndex(
        values, {item.source_id: item for item in source_values}, "", annotation_revision,
        tuple(sorted(coverage_values)), partial_ids,
    )
    intervals_path.parent.mkdir(parents=True, exist_ok=True)
    with intervals_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INTERVAL_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(item) for item in values)
    digest = file_sha256(intervals_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "rally_interval_manifest",
                "schema_version": 3,
                "boundary_definition": BOUNDARY_DEFINITION,
                "annotation_revision": str(annotation_revision),
                "intervals_sha256": digest,
                "sources": [asdict(item) for item in source_values],
                "reviewed_coverage": coverage_mapping,
                "partial_start_rally_ids": sorted(partial_ids),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return read_rally_intervals(intervals_path, manifest_path)


def refill_frames(
    interval: RallyInterval,
    *,
    fps: float,
    already_labeled: Iterable[int] = (),
) -> tuple[int, ...]:
    """Return unlabeled first/last/midpoint one-second coverage for a rally."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    width = max(1, math.floor(fps + 0.5))
    right_span = width - 1 - width // 2
    centers = (
        interval.start_frame + width // 2,
        (interval.start_frame + interval.end_frame) // 2,
        interval.end_frame - right_span,
    )
    covered: set[int] = set()
    for center in centers:
        start = max(interval.start_frame, center - width // 2)
        end = min(interval.end_frame, start + width - 1)
        start = max(interval.start_frame, end - width + 1)
        covered.update(range(start, end + 1))
    return tuple(sorted(covered - {int(frame) for frame in already_labeled}))
