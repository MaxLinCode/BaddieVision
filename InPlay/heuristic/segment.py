"""CLI and state machine for heuristic rally segmentation."""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

from .config import HeuristicConfig
from .court import add_court_signal, add_player_court_signal
from .tracks import FrameFeature, debug_fieldnames, preprocess_tracks, read_track_csv

CANONICAL_FIELDS = [
    "source_id", "rally_id", "start_frame", "end_frame", "start_time", "end_time",
    "status", "confidence", "confidence_band", "flags", "failure_reason",
    "manual_start_frame", "manual_end_frame", "manual_decision",
]


@dataclass
class Rally:
    source_id: str
    rally_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    status: str
    confidence: float
    confidence_band: str
    flags: str
    failure_reason: str = ""
    manual_start_frame: str = ""
    manual_end_frame: str = ""
    manual_decision: str = ""


def _recent_motion(frames: list[FrameFeature], index: int, config: HeuristicConfig) -> float:
    start = max(0, index - config.recent_motion_window + 1)
    return sum(item.speed for item in frames[start : index + 1])


def _candidate_score(
    span: list[FrameFeature], config: HeuristicConfig, player_activity: dict[int, float] | None
) -> tuple[float, list[str]]:
    flags: list[str] = []
    reliable = sum(item.reliable for item in span) / len(span)
    trajectory_steps = [
        item.speed
        for item in span
        if math.isfinite(item.smooth_x) and (item.cleaned or item.interpolated)
    ]
    motion = (
        sum(config.stopped_speed < speed <= config.max_single_frame_jump for speed in trajectory_steps)
        / len(trajectory_steps)
        if trajectory_steps
        else 0.0
    )
    removed = sum(item.removed for item in span) / len(span)
    longest_gap = max((item.missing_count for item in span), default=0)
    gap_penalty = min(1.0, longest_gap / max(config.long_missing_gap, 1))
    positive = [(reliable, config.reliable_weight), (motion, config.motion_weight)]
    if player_activity:
        values = [player_activity[item.frame] for item in span if item.frame in player_activity]
        if values:
            positive.append((sum(values) / len(values), config.player_weight))
    denominator = sum(weight for _, weight in positive)
    score = sum(value * weight for value, weight in positive) / denominator
    score += config.long_gap_weight * gap_penalty + config.removed_weight * removed
    score = max(0.0, min(1.0, score))
    if reliable < 0.5:
        flags.append("low_shuttle_confidence")
    if longest_gap >= config.long_missing_gap:
        flags.append("large_missing_gap")
    return score, flags


def _read_player_rows(path: str | Path | None) -> dict[int, dict[str, str]] | None:
    if path is None:
        return None
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "Frame" not in set(reader.fieldnames or ()):
            raise ValueError("player CSV requires a Frame column")
        return {int(row["Frame"]): row for row in reader}


def _player_activity(rows: dict[int, dict[str, str]] | None) -> dict[int, float] | None:
    if rows is None:
        return None
    if any("player_activity" not in row for row in rows.values()):
        raise ValueError("player CSV requires player_activity column")
    return {frame: float(row["player_activity"]) for frame, row in rows.items()}


def _service_start_ok(frames: list[FrameFeature], candidate_start: int, index: int) -> bool:
    span = frames[candidate_start : index + 1]
    observed = [
        item.players_opposite_service_regions
        for item in span
        if item.players_opposite_service_regions is not None
    ]
    return not observed or any(observed)


def segment_tracks(
    frames: list[FrameFeature],
    source_id: str,
    fps: float,
    config: HeuristicConfig,
    player_activity: dict[int, float] | None = None,
    base_flags: Iterable[str] = (),
) -> list[Rally]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    flags_base = list(base_flags)
    if not math.isclose(fps, config.target_fps):
        flags_base.append("non_30_fps")
    state = "IDLE"
    candidate_start = 0
    candidate_end = 0
    candidate_end_reason = ""
    outside_streak = 0
    spans: list[tuple[int, int, list[str]]] = []

    for index, item in enumerate(frames):
        if item.inside_courtish is False and item.players_on_courtish is not True:
            outside_streak += 1
        elif item.inside_courtish is True or item.players_on_courtish is True:
            outside_streak = 0
        if state == "IDLE":
            if item.visible_streak >= config.visible_streak:
                candidate_start = index - config.visible_streak + 1
                state = "CANDIDATE_START"
        elif state == "CANDIDATE_START":
            if item.missing_count > config.interpolation_gap:
                spans.append((candidate_start, index - 1, []))
                state = "IDLE"
            elif index - candidate_start + 1 >= config.start_confirmation:
                if _service_start_ok(frames, candidate_start, index):
                    candidate_start = max(0, candidate_start - config.start_buffer)
                    state = "IN_RALLY"
                else:
                    state = "IDLE"
        elif state == "IN_RALLY":
            recent_motion = _recent_motion(frames, index, config)
            stopped = (
                index >= config.stopped_window - 1
                and sum(f.speed for f in frames[index - config.stopped_window + 1 : index + 1])
                <= config.stopped_speed * config.stopped_window
            )
            medium_inactive = (
                item.missing_count > config.short_gap
                and recent_motion < config.recent_motion_minimum
            )
            if medium_inactive and player_activity:
                recent_players = [
                    player_activity[frame.frame]
                    for frame in frames[max(0, index - config.recent_motion_window + 1) : index + 1]
                    if frame.frame in player_activity
                ]
                # Active players can keep a medium shuttle gap alive. They can
                # never create a start transition.
                if recent_players and sum(recent_players) / len(recent_players) >= 0.25:
                    medium_inactive = False
            if stopped and player_activity:
                recent_players = [
                    player_activity[frame.frame]
                    for frame in frames[max(0, index - config.stopped_window + 1) : index + 1]
                    if frame.frame in player_activity
                ]
                # With optional player evidence, shuttle inactivity is only a
                # strong end signal when players are inactive too.
                if recent_players and sum(recent_players) / len(recent_players) >= 0.15:
                    stopped = False
            long_gap = item.missing_count >= config.long_missing_gap
            outside = outside_streak >= config.outside_court_frames
            if stopped or medium_inactive or long_gap or outside:
                candidate_end = index
                candidate_end_reason = (
                    "outside"
                    if outside
                    else "long_gap"
                    if long_gap
                    else "medium_gap"
                    if medium_inactive
                    else "stopped"
                )
                state = "CANDIDATE_END"
        elif state == "CANDIDATE_END":
            motion_returned = item.reliable and _recent_motion(frames, index, config) >= config.recent_motion_minimum
            recovered = (
                item.inside_courtish is True or item.players_on_courtish is True
                if candidate_end_reason == "outside"
                else motion_returned and item.visible_streak >= config.visible_streak
            )
            if recovered:
                state = "IN_RALLY"
                outside_streak = 0
            elif index - candidate_end + 1 >= config.end_confirmation:
                end = max(candidate_start, candidate_end - 1 + config.end_buffer)
                spans.append((candidate_start, min(end, len(frames) - 1), []))
                state = "IDLE"
        item.state = state

    if state in {"IN_RALLY", "CANDIDATE_END"}:
        spans.append((candidate_start, len(frames) - 1, ["manual_review_needed"]))
    elif state == "CANDIDATE_START" and _service_start_ok(frames, candidate_start, len(frames) - 1):
        spans.append((candidate_start, len(frames) - 1, ["manual_review_needed"]))

    rallies: list[Rally] = []
    for number, (start, end, extra_flags) in enumerate(spans, 1):
        span = frames[start : end + 1]
        frame_count = end - start + 1
        reliable_count = sum(item.reliable for item in span)
        motion = sum(item.speed for item in span)
        score, score_flags = _candidate_score(span, config, player_activity)
        candidate_flags = list(dict.fromkeys(flags_base + extra_flags + score_flags))
        reasons = []
        if frame_count < config.minimum_rally:
            reasons.append("short_rally")
            candidate_flags.append("short_rally")
        if reliable_count < config.minimum_visible_frames:
            reasons.append("insufficient_reliable_detections")
        if motion < config.minimum_motion:
            reasons.append("insufficient_motion")
        status = "rejected" if reasons else ("review" if candidate_flags else "accepted")
        if status == "review" and "manual_review_needed" not in candidate_flags:
            candidate_flags.append("manual_review_needed")
        band = "high" if score >= 0.75 else "medium" if score >= 0.50 else "low"
        start_frame, end_frame = frames[start].frame, frames[end].frame
        rallies.append(
            Rally(
                source_id=source_id,
                rally_id=f"{source_id}-{number:04d}",
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_frame / fps,
                end_time=end_frame / fps,
                status=status,
                confidence=round(score, 6),
                confidence_band=band,
                flags=";".join(dict.fromkeys(candidate_flags)),
                failure_reason=";".join(reasons),
            )
        )
    return rallies


def write_rallies(path: str | Path, rallies: list[Rally]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(item) for item in rallies)


def write_debug(path: str | Path, frames: list[FrameFeature]) -> None:
    names = debug_fieldnames()
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(asdict(item) for item in frames)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--image-size", required=True, nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug-frames")
    parser.add_argument("--players")
    parser.add_argument("--court-calibration")
    defaults = HeuristicConfig()
    for field in fields(defaults):
        if field.name == "target_fps":
            continue
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=type(getattr(defaults, field.name)),
            default=None,
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    defaults = HeuristicConfig()
    overrides = {
        field.name: getattr(args, field.name)
        for field in fields(defaults)
        if field.name != "target_fps" and getattr(args, field.name) is not None
    }
    config = HeuristicConfig(**overrides)
    if not math.isclose(args.fps, config.target_fps):
        warnings.warn("thresholds are calibrated for 30 FPS and were not rescaled", stacklevel=1)
    source_id = args.source_id or Path(args.tracks).stem
    frames = preprocess_tracks(read_track_csv(args.tracks, tuple(args.image_size), config), config)
    base_flags: list[str] = []
    player_rows = _read_player_rows(args.players)
    if args.court_calibration:
        court_flag = add_court_signal(
            frames, args.court_calibration, config.court_tolerance, source_id
        )
        if court_flag:
            base_flags.extend([court_flag, "manual_review_needed"])
        if player_rows:
            player_court_flag = add_player_court_signal(
                frames, player_rows, args.court_calibration, config.court_tolerance, source_id
            )
            if player_court_flag:
                base_flags.extend([player_court_flag, "manual_review_needed"])
    rallies = segment_tracks(
        frames,
        source_id,
        args.fps,
        config,
        _player_activity(player_rows),
        base_flags,
    )
    write_rallies(args.output, rallies)
    if args.debug_frames:
        write_debug(args.debug_frames, frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
