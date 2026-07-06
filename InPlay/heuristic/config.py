"""Central configuration for heuristic rally segmentation."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class HeuristicConfig:
    target_fps: float = 30.0
    visible_streak: int = 5
    start_confirmation: int = 15
    start_buffer: int = 15
    short_gap: int = 30
    long_missing_gap: int = 60
    stopped_window: int = 45
    end_confirmation: int = 45
    end_buffer: int = 15
    minimum_rally: int = 60
    minimum_visible_frames: int = 20
    interpolation_gap: int = 12
    smoothing_window: int = 7
    smoothing_polyorder: int = 2
    peak_cutoff: float = 0.5
    max_single_frame_jump: float = 0.35
    isolated_radius: int = 2
    stopped_speed: float = 0.0015
    recent_motion_window: int = 30
    recent_motion_minimum: float = 0.015
    minimum_motion: float = 0.08
    outside_court_frames: int = 30
    court_tolerance: float = 0.20
    reliable_weight: float = 0.40
    motion_weight: float = 0.25
    player_weight: float = 0.15
    long_gap_weight: float = -0.10
    removed_weight: float = -0.10

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, (int, float)) and item.name.endswith(
                ("gap", "window", "confirmation", "buffer", "frames", "streak", "radius")
            ) and value < 0:
                raise ValueError(f"{item.name} cannot be negative")
        if self.smoothing_window < 3 or self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be an odd integer >= 3")
        if self.smoothing_polyorder >= self.smoothing_window:
            raise ValueError("smoothing_polyorder must be less than smoothing_window")

    @classmethod
    def cli_fields(cls) -> tuple[str, ...]:
        return tuple(item.name for item in fields(cls) if item.name != "target_fps")
