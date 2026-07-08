"""Constrained interval decoding for learned InPlay probabilities."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Sequence

import numpy as np

from .heuristic.config import HeuristicConfig
from .heuristic.segment import Rally, segment_tracks
from .heuristic.tracks import FrameFeature


def scale_config_for_fps(config: HeuristicConfig, fps: float) -> HeuristicConfig:
    """Scale frame-window thresholds from the 30 FPS heuristic defaults."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    ratio = fps / config.target_fps
    values = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if field.name == "target_fps":
            values[field.name] = config.target_fps
        elif isinstance(value, int) and field.name.endswith(
            ("gap", "window", "confirmation", "buffer", "frames", "streak", "radius")
        ):
            values[field.name] = max(1, int(round(value * ratio)))
        else:
            values[field.name] = value
    if values["smoothing_window"] % 2 == 0:
        values["smoothing_window"] += 1
    values["smoothing_window"] = max(3, values["smoothing_window"])
    values["smoothing_polyorder"] = min(
        values["smoothing_polyorder"], values["smoothing_window"] - 1
    )
    return HeuristicConfig(**values)


def decode_probabilities(
    probabilities: Sequence[float],
    frame_indices: Sequence[int],
    source_id: str,
    fps: float,
    config: HeuristicConfig | None = None,
    threshold: float = 0.5,
    shuttle_motion: Sequence[float] | None = None,
    inside_courtish: Sequence[bool | None] | None = None,
    player_activity: dict[int, float] | None = None,
) -> list[Rally]:
    """Decode per-frame learned probabilities through the heuristic state model."""

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional sequence")
    if len(probs) != len(frame_indices):
        raise ValueError("probabilities and frame_indices must have the same length")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    base_config = config or HeuristicConfig()
    scaled = scale_config_for_fps(base_config, fps)
    motion = np.asarray(shuttle_motion if shuttle_motion is not None else np.zeros(len(probs)))
    if motion.shape != probs.shape:
        raise ValueError("shuttle_motion must match probabilities length")
    court = list(inside_courtish if inside_courtish is not None else [None] * len(probs))
    if len(court) != len(probs):
        raise ValueError("inside_courtish must match probabilities length")

    frames: list[FrameFeature] = []
    visible_streak = 0
    missing_count = 0
    cumulative = 0.0
    previous_speed = 0.0
    for frame, probability, speed, court_value in zip(frame_indices, probs, motion, court):
        reliable = bool(probability >= threshold)
        if reliable:
            visible_streak += 1
            missing_count = 0
        else:
            visible_streak = 0
            missing_count += 1
        speed = float(max(0.0, speed))
        cumulative += speed
        item = FrameFeature(
            frame=int(frame),
            raw_x=0.0,
            raw_y=0.0,
            visibility=int(reliable),
            peak_value=float(probability),
            confidence_available=True,
            x=0.0 if reliable else math.nan,
            y=0.0 if reliable else math.nan,
            cleaned=reliable,
            smooth_x=0.0 if reliable else math.nan,
            smooth_y=0.0 if reliable else math.nan,
            speed=speed,
            acceleration=speed - previous_speed,
            cumulative_distance=cumulative,
            missing_count=missing_count,
            visible_streak=visible_streak,
            reliable=reliable,
            inside_courtish=court_value,
        )
        previous_speed = speed
        frames.append(item)
    base_flags = ["decoded_from_model_probabilities"]
    return segment_tracks(frames, source_id, fps, scaled, player_activity, base_flags)
