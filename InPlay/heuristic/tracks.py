"""TrackNet CSV ingestion, cleanup, interpolation, and motion features."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from .config import HeuristicConfig

REQUIRED_COLUMNS = {"Frame", "X", "Y", "Visibility"}


@dataclass
class FrameFeature:
    frame: int
    raw_x: float
    raw_y: float
    visibility: int
    peak_value: float | None
    confidence_available: bool
    x: float = math.nan
    y: float = math.nan
    cleaned: bool = False
    removed: bool = False
    removal_reason: str = ""
    interpolated: bool = False
    smooth_x: float = math.nan
    smooth_y: float = math.nan
    speed: float = 0.0
    acceleration: float = 0.0
    cumulative_distance: float = 0.0
    missing_count: int = 0
    visible_streak: int = 0
    reliable: bool = False
    state: str = "IDLE"
    inside_courtish: bool | None = None
    players_on_courtish: bool | None = None
    player_court_count: int = 0
    players_opposite_service_regions: bool | None = None


def _number(value: str, name: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: invalid {name}") from exc
    if not math.isfinite(result):
        raise ValueError(f"row {row_number}: non-finite {name}")
    return result


def read_track_csv(
    path: str | Path, image_size: tuple[int, int], config: HeuristicConfig
) -> list[FrameFeature]:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"track CSV missing required columns: {sorted(missing)}")
        has_peak = "PeakValue" in columns
        result: list[FrameFeature] = []
        previous: int | None = None
        first: int | None = None
        for row_number, row in enumerate(reader, 2):
            frame_value = _number(row["Frame"], "Frame", row_number)
            if not frame_value.is_integer():
                raise ValueError(f"row {row_number}: Frame must be an integer")
            frame = int(frame_value)
            if frame < 0:
                raise ValueError(f"row {row_number}: Frame cannot be negative")
            if previous is not None and frame <= previous:
                reason = "duplicate" if frame == previous else "unordered"
                raise ValueError(f"row {row_number}: {reason} frame {frame}")
            if previous is not None and frame != previous + 1:
                raise ValueError(
                    f"row {row_number}: missing frame records between {previous} and {frame}"
                )
            first = frame if first is None else first
            previous = frame
            x = _number(row["X"], "X", row_number)
            y = _number(row["Y"], "Y", row_number)
            visibility = int(_number(row["Visibility"], "Visibility", row_number))
            if visibility not in (0, 1):
                raise ValueError(f"row {row_number}: Visibility must be 0 or 1")
            peak: float | None = None
            if has_peak and row.get("PeakValue", "").strip():
                peak = _number(row["PeakValue"], "PeakValue", row_number)
            feature = FrameFeature(
                frame=frame,
                raw_x=x,
                raw_y=y,
                visibility=visibility,
                peak_value=peak,
                confidence_available=has_peak,
            )
            coordinate_valid = visibility == 1 and 0 <= x < width and 0 <= y < height
            confidence_valid = peak is None or peak >= config.peak_cutoff
            if coordinate_valid and confidence_valid:
                feature.x, feature.y, feature.cleaned = x / width, y / height, True
            elif visibility:
                feature.removed = True
                feature.removal_reason = (
                    "low_peak" if coordinate_valid and not confidence_valid else "invalid_coordinate"
                )
            result.append(feature)
    if not result:
        raise ValueError("track CSV contains no frame records")
    return result


def _remove_isolated_and_jumps(frames: list[FrameFeature], config: HeuristicConfig) -> None:
    initially_valid = [item.cleaned for item in frames]
    for index, item in enumerate(frames):
        if not initially_valid[index]:
            continue
        lo = max(0, index - config.isolated_radius)
        hi = min(len(frames), index + config.isolated_radius + 1)
        if not any(initially_valid[j] for j in range(lo, hi) if j != index):
            item.cleaned = False
            item.removed = True
            item.removal_reason = "isolated"
            item.x = item.y = math.nan

    valid_indices = [i for i, item in enumerate(frames) if item.cleaned]
    for left, middle, right in zip(valid_indices, valid_indices[1:], valid_indices[2:]):
        if middle != left + 1 or right != middle + 1:
            continue
        a, b, c = frames[left], frames[middle], frames[right]
        jump_in = math.hypot(b.x - a.x, b.y - a.y)
        jump_out = math.hypot(c.x - b.x, c.y - b.y)
        bypass = math.hypot(c.x - a.x, c.y - a.y)
        if (
            jump_in > config.max_single_frame_jump
            and jump_out > config.max_single_frame_jump
            and bypass <= config.max_single_frame_jump
        ):
            b.cleaned = False
            b.removed = True
            b.removal_reason = "single_frame_jump"
            b.x = b.y = math.nan


def _interpolate(frames: list[FrameFeature], max_gap: int) -> None:
    valid = [i for i, item in enumerate(frames) if item.cleaned]
    for left, right in zip(valid, valid[1:]):
        gap = right - left - 1
        if not 0 < gap <= max_gap:
            continue
        for index in range(left + 1, right):
            fraction = (index - left) / (right - left)
            frames[index].x = frames[left].x + fraction * (frames[right].x - frames[left].x)
            frames[index].y = frames[left].y + fraction * (frames[right].y - frames[left].y)
            frames[index].interpolated = True


def _trajectory_spans(frames: list[FrameFeature]) -> list[tuple[int, int]]:
    valid = [item.cleaned or item.interpolated for item in frames]
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, present in enumerate(valid + [False]):
        if present and start is None:
            start = index
        elif not present and start is not None:
            spans.append((start, index))
            start = None
    return spans


def preprocess_tracks(frames: list[FrameFeature], config: HeuristicConfig) -> list[FrameFeature]:
    _remove_isolated_and_jumps(frames, config)
    _interpolate(frames, config.interpolation_gap)
    for start, end in _trajectory_spans(frames):
        length = end - start
        window = min(config.smoothing_window, length if length % 2 else length - 1)
        xs = np.asarray([frames[i].x for i in range(start, end)])
        ys = np.asarray([frames[i].y for i in range(start, end)])
        if window >= 3 and window > config.smoothing_polyorder:
            xs = savgol_filter(xs, window, min(config.smoothing_polyorder, window - 1))
            ys = savgol_filter(ys, window, min(config.smoothing_polyorder, window - 1))
        for offset, index in enumerate(range(start, end)):
            frames[index].smooth_x = float(xs[offset])
            frames[index].smooth_y = float(ys[offset])

    cumulative = 0.0
    missing = streak = 0
    previous_xy: tuple[float, float] | None = None
    previous_speed = 0.0
    for item in frames:
        item.reliable = item.cleaned
        if item.reliable:
            streak += 1
            missing = 0
        else:
            streak = 0
            missing += 1
        item.visible_streak, item.missing_count = streak, missing
        if math.isfinite(item.smooth_x) and previous_xy is not None:
            item.speed = math.hypot(
                item.smooth_x - previous_xy[0], item.smooth_y - previous_xy[1]
            )
            cumulative += item.speed
            item.acceleration = item.speed - previous_speed
            previous_speed = item.speed
        elif not math.isfinite(item.smooth_x):
            previous_xy = None
            previous_speed = 0.0
        if math.isfinite(item.smooth_x):
            previous_xy = (item.smooth_x, item.smooth_y)
        item.cumulative_distance = cumulative
    return frames


def debug_fieldnames() -> list[str]:
    return [item.name for item in fields(FrameFeature)]
