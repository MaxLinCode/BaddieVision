"""Portable, source-local video identity.

Every source is its own zero-based frame sequence.  Parent-video information is
provenance only and must never be used as an artifact or annotation frame key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from src.single_video.video import probe_video, read_video_info


SOURCE_IDENTITY_SCHEMA = "baddievision_source_identity"
SOURCE_IDENTITY_VERSION = 1


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _rate(value: str | float) -> tuple[int, int]:
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        rate = Fraction(int(numerator), int(denominator))
    else:
        rate = Fraction(float(value)).limit_denominator(1_000_000)
    if rate <= 0:
        raise ValueError(f"source reports invalid average frame rate: {value}")
    return rate.numerator, rate.denominator


@dataclass(frozen=True)
class ParentVideoProvenance:
    video_sha256: str | None = None
    start_time_seconds: float | None = None
    end_time_seconds: float | None = None
    handling: str | None = None


@dataclass(frozen=True)
class SourceIdentity:
    source_id: str
    video_sha256: str
    frame_count: int
    width: int
    height: int
    fps_numerator: int
    fps_denominator: int
    parent_video: ParentVideoProvenance | None = None
    schema: str = SOURCE_IDENTITY_SCHEMA
    schema_version: int = SOURCE_IDENTITY_VERSION
    frame_indexing: str = "source_local_zero_based"
    frame_range_start: int = 0
    frame_range_end_inclusive: int | None = None
    fps_interpretation: str = "average_frame_rate"

    def __post_init__(self) -> None:
        expected_end = self.frame_count - 1
        if self.frame_count <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("source identity requires positive dimensions and frame count")
        if self.fps_numerator <= 0 or self.fps_denominator <= 0:
            raise ValueError("source identity requires a positive rational frame rate")
        if self.frame_range_start != 0:
            raise ValueError("source frame ranges must start at zero")
        current_end = self.frame_range_end_inclusive
        if current_end is None:
            object.__setattr__(self, "frame_range_end_inclusive", expected_end)
        elif current_end != expected_end:
            raise ValueError("source frame range must be [0, frame_count - 1]")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        if self.parent_video is None:
            value.pop("parent_video")
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceIdentity":
        if value.get("schema") != SOURCE_IDENTITY_SCHEMA or value.get("schema_version") != 1:
            raise ValueError("unsupported source identity schema")
        parent = value.get("parent_video")
        return cls(
            source_id=str(value["source_id"]),
            video_sha256=str(value["video_sha256"]),
            frame_count=int(value["frame_count"]),
            width=int(value["width"]),
            height=int(value["height"]),
            fps_numerator=int(value["fps_numerator"]),
            fps_denominator=int(value["fps_denominator"]),
            parent_video=ParentVideoProvenance(**parent) if isinstance(parent, dict) else None,
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
            frame_indexing=str(value.get("frame_indexing", "")),
            frame_range_start=int(value.get("frame_range_start", -1)),
            frame_range_end_inclusive=int(value["frame_range_end_inclusive"]),
            fps_interpretation=str(value.get("fps_interpretation", "")),
        )

    @classmethod
    def read(cls, path: str | Path) -> "SourceIdentity":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_mapping(), indent=2) + "\n", encoding="utf-8")
        return path


def inspect_source(source_id: str, video_path: str | Path) -> SourceIdentity:
    """Inspect an immutable working video and assign it a source-local domain."""
    video_path = Path(video_path).expanduser().resolve()
    info = read_video_info(video_path)
    try:
        stream = probe_video(video_path)
        numerator, denominator = _rate(str(stream.get("avg_frame_rate") or info["fps"]))
    except (RuntimeError, ValueError, OSError):
        numerator, denominator = _rate(float(info["fps"]))
    return SourceIdentity(
        source_id=source_id,
        video_sha256=sha256_file(video_path),
        frame_count=int(info["frame_count"]),
        width=int(info["width"]),
        height=int(info["height"]),
        fps_numerator=numerator,
        fps_denominator=denominator,
    )
