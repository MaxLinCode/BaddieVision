"""Exporter-only padding for canonical inclusive rally boundaries."""

from __future__ import annotations

from .segment import Rally


def padded_bounds(rally: Rally, *, fps: float, frame_count: int,
                  before_seconds: float = 0.0, after_seconds: float = 0.0) -> tuple[int, int]:
    if fps <= 0 or frame_count <= 0 or before_seconds < 0 or after_seconds < 0:
        raise ValueError("fps/frame_count must be positive and padding cannot be negative")
    return (max(0, rally.start_frame - round(before_seconds * fps)),
            min(frame_count - 1, rally.end_frame + round(after_seconds * fps)))
