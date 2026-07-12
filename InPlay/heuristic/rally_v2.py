"""Offline, precision-first rally evidence fusion for static-camera singles."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .segment import Rally

SCHEMA_VERSION = 1
DECODER_VERSION = "heuristic-rally-v2.0"
STATES = ("RESET", "SERVE_ARMED", "START_PENDING", "LIVE", "END_PENDING", "COOLDOWN")


@dataclass(frozen=True)
class RallyV2Profile:
    readiness_seconds: float = 0.5
    start_lookback_seconds: float = 1.5
    start_confirmation_seconds: float = 1.0
    recoverable_gap_seconds: float = 0.5
    end_confirmation_seconds: float = 1.0
    cooldown_seconds: float = 1.0
    hypothesis_score_floor: float = 0.60
    hypothesis_ambiguity: float = 0.10
    opposite_half_ratio: float = 0.80
    settled_mad_multiplier: float = 2.5
    activity_burst_mad_multiplier: float = 4.0
    minimum_travel_diagonal: float = 0.08

    def frames(self, seconds: float, fps: float) -> int:
        return max(1, int(round(seconds * fps)))


@dataclass(frozen=True)
class V2Inputs:
    candidates: Path
    tracklets: Path
    hypotheses: Path
    player_assignments: Path
    pose_cache: Path
    metadata: Path
    calibration: Path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, schema: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty required artifact: {path}")
    metadata = json.loads(lines[0])
    if metadata.get("type") != "metadata" or metadata.get("schema") != schema:
        raise ValueError(f"expected {schema} metadata: {path}")
    return metadata, [json.loads(line) for line in lines[1:] if line.strip()]


def _source_metadata(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if "image_size" not in value and "width" in value and "height" in value:
        value["image_size"] = [value["width"], value["height"]]
    required = {"source_id", "fps", "image_size", "frame_count"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"source metadata missing fields: {sorted(missing)}")
    return value


def _validate(inputs: V2Inputs, degraded: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = asdict(inputs)
    missing = [name for name, path in paths.items() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required rally inputs: {', '.join(missing)}")
    source = _source_metadata(inputs.metadata)
    fps, frame_count = float(source["fps"]), int(source["frame_count"])
    image_size = tuple(map(int, source["image_size"]))
    if fps <= 0 or frame_count <= 0 or len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("invalid source fps, frame_count, or image_size")
    calibration = json.loads(inputs.calibration.read_text(encoding="utf-8"))
    if "image_size" not in calibration:
        raise ValueError("court calibration is missing image_size")
    if tuple(map(int, calibration["image_size"])) != image_size:
        raise ValueError("court calibration image_size does not match source metadata")
    cm, _ = _jsonl(inputs.candidates, "shuttle_candidates")
    tm, _ = _jsonl(inputs.tracklets, "shuttle_tracklets")
    hm, _ = _jsonl(inputs.hypotheses, "shuttle_hypotheses")
    pm, _ = _jsonl(inputs.player_assignments, "player_assignments")
    posem, _ = _jsonl(inputs.pose_cache, "pose_cache")
    if not math.isclose(float(cm.get("fps", -1)), fps, rel_tol=0, abs_tol=1e-6):
        raise ValueError("shuttle candidate fps does not match source metadata")
    if tuple(map(int, cm.get("image_size", ()))) != image_size:
        raise ValueError("shuttle candidate image_size does not match source metadata")
    for name, meta in (("player assignments", pm),):
        if int(meta.get("frame_count", -1)) != frame_count:
            raise ValueError(f"{name} frame_count does not match source metadata")
        if tuple(map(int, meta.get("frame_size", ()))) != image_size:
            raise ValueError(f"{name} image_size does not match source metadata")
        if not math.isclose(float(meta.get("fps", -1)), fps, abs_tol=1e-6):
            raise ValueError(f"{name} fps does not match source metadata")
    candidate_hex = _sha256(inputs.candidates).split(":", 1)[1]
    tracklet_hex = _sha256(inputs.tracklets).split(":", 1)[1]
    if tm.get("candidate_sha256") != candidate_hex or hm.get("candidate_sha256") != candidate_hex:
        raise ValueError("layered shuttle candidate fingerprint mismatch")
    if hm.get("tracklet_sha256") != tracklet_hex:
        raise ValueError("layered shuttle tracklet fingerprint mismatch")
    raw_fp = pm.get("raw_artifact_fingerprint")
    if raw_fp and posem.get("raw_artifact_fingerprint") != raw_fp:
        raise ValueError("pose cache and player assignments raw fingerprint mismatch")
    fingerprints = {name: _sha256(Path(path)) for name, path in paths.items()}
    return source, fingerprints


def _median_mad(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    median = float(np.median(values))
    return median, float(np.median(np.abs(np.asarray(values) - median)))


def _player_evidence(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_frame = {int(item["frame"]): item for item in records if item.get("type") == "frame"}
    result: list[dict[str, Any]] = []
    previous: dict[str, tuple[float, float] | None] = {"P1": None, "P2": None}
    history: list[float] = []
    for frame in range(count):
        item = by_frame.get(frame, {})
        slots = item.get("slots", {})
        positions: dict[str, tuple[float, float] | None] = {}
        unambiguous = True
        activity = 0.0
        for role in ("P1", "P2"):
            slot = slots.get(role, {})
            assignment = slot.get("assignment")
            if assignment is None or slot.get("ambiguity_reason"):
                positions[role] = None
                unambiguous = False
                continue
            pos = (float(assignment["court_x"]), float(assignment["court_y"]))
            positions[role] = pos
            if previous[role] is not None:
                activity += math.dist(pos, previous[role])
            previous[role] = pos
        opposite = bool(positions["P1"] and positions["P2"] and positions["P1"][1] < 0 < positions["P2"][1])
        baseline, mad = _median_mad(history[-150:])
        settled = activity <= baseline + 2.5 * max(mad, 1e-4)
        burst = activity > baseline + 4.0 * max(mad, 1e-4)
        result.append({"unambiguous": unambiguous, "opposite_halves": opposite,
                       "positions": positions, "activity": activity, "activity_baseline": baseline,
                       "activity_mad": mad, "settled": settled, "burst": burst})
        history.append(activity)
    return result


def _shuttle_evidence(candidate_records: list[dict[str, Any]], hypothesis_records: list[dict[str, Any]],
                      count: int, image_size: tuple[int, int], profile: RallyV2Profile) -> list[dict[str, Any]]:
    candidates = {candidate["candidate_id"]: candidate for row in candidate_records
                  if row.get("type") == "frame" for candidate in row.get("candidates", [])}
    by_frame: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    for hypothesis in hypothesis_records:
        if hypothesis.get("type") != "hypothesis" or float(hypothesis.get("total_score", 0)) < profile.hypothesis_score_floor:
            continue
        for candidate_id in hypothesis.get("candidate_ids", []):
            candidate = candidates.get(candidate_id)
            if candidate is None:
                continue
            try:
                frame = int(candidate_id.split("-", 1)[0][1:])
            except (ValueError, IndexError):
                continue
            if 0 <= frame < count:
                by_frame[frame].append({"id": f"{hypothesis['region_id']}:{hypothesis['rank']}",
                                        "score": float(hypothesis["total_score"]), "center": candidate["center"]})
    diagonal = math.hypot(*image_size)
    result = []
    previous = None
    for options in by_frame:
        options.sort(key=lambda item: (-item["score"], item["id"]))
        ambiguous = len(options) > 1 and options[0]["score"] - options[1]["score"] <= profile.hypothesis_ambiguity
        selected = None if ambiguous or not options else options[0]
        travel = math.dist(selected["center"], previous["center"]) / diagonal if selected and previous else 0.0
        result.append({"selected": selected, "ambiguous": ambiguous, "travel": travel,
                       "outside": bool(selected and not (0 <= selected["center"][0] < image_size[0] and 0 <= selected["center"][1] < image_size[1]))})
        if selected:
            previous = selected
    return result


def decode_rallies(inputs: V2Inputs, *, degraded: bool = False,
                   profile: RallyV2Profile | None = None) -> tuple[list[Rally], list[dict[str, Any]], dict[str, Any]]:
    """Validate all artifacts, decode offline, and return canonical rallies plus frame events."""
    profile = profile or RallyV2Profile()
    source, fingerprints = _validate(inputs, degraded)
    _, candidate_records = _jsonl(inputs.candidates, "shuttle_candidates")
    _, hypothesis_records = _jsonl(inputs.hypotheses, "shuttle_hypotheses")
    _, player_records = _jsonl(inputs.player_assignments, "player_assignments")
    count, fps = int(source["frame_count"]), float(source["fps"])
    players = _player_evidence(player_records, count)
    shuttle = _shuttle_evidence(candidate_records, hypothesis_records, count,
                                tuple(map(int, source["image_size"])), profile)
    readiness = profile.frames(profile.readiness_seconds, fps)
    start_confirm = profile.frames(profile.start_confirmation_seconds, fps)
    gap_limit = profile.frames(profile.recoverable_gap_seconds, fps)
    end_confirm = profile.frames(profile.end_confirmation_seconds, fps)
    cooldown = profile.frames(profile.cooldown_seconds, fps)
    lookback = profile.frames(profile.start_lookback_seconds, fps)
    state, state_since, gap = "RESET", 0, 0
    serve_events: list[tuple[int, float]] = []
    start = last_live = None
    opponent_response = False
    response_activity_seen = False
    review_reasons: set[str] = set()
    spans: list[tuple[int, int, bool, set[str]]] = []
    events: list[dict[str, Any]] = []
    travel_sum = 0.0
    for frame in range(count):
        p, s = players[frame], shuttle[frame]
        transition = None
        readiness_window = players[max(0, frame - readiness + 1):frame + 1]
        ready = len(readiness_window) == readiness and sum(x["unambiguous"] and x["opposite_halves"] and x["settled"] for x in readiness_window) / readiness >= profile.opposite_half_ratio
        if s["ambiguous"] and state in {"START_PENDING", "LIVE", "END_PENDING"}:
            review_reasons.add("competing_hypotheses")
        if state == "RESET" and ready:
            transition, state = "readiness_confirmed", "SERVE_ARMED"
        elif state == "SERVE_ARMED":
            if not p["unambiguous"]:
                transition, state = "player_roles_lost", "RESET"
            elif s["selected"] and p["burst"]:
                proximity = 0.0
                center = s["selected"]["center"]
                # Candidate proximity is image-normalized here; pose is optional support.
                proximity = 1.0 if center else 0.0
                serve_events.append((frame, s["selected"]["score"] + 0.05 * proximity))
                transition, state, state_since = "serve_evidence", "START_PENDING", frame
        elif state == "START_PENDING":
            if s["selected"]:
                travel_sum += s["travel"]
                last_live = frame
            if p["burst"] and frame > state_since:
                response_activity_seen = True
            if travel_sum >= profile.minimum_travel_diagonal and response_activity_seen:
                opponent_response = True
            if opponent_response and frame - state_since + 1 <= start_confirm:
                eligible = [(f, score) for f, score in serve_events if f >= frame - lookback]
                start = max(eligible, key=lambda item: (item[1], -item[0]))[0]
                transition, state = "opponent_response_confirmed", "LIVE"
            elif frame - state_since + 1 > start_confirm:
                if serve_events:
                    review_reasons.add("serve_only")
                    spans.append((serve_events[-1][0], last_live or frame, False, set(review_reasons)))
                transition, state, serve_events, travel_sum = "start_unconfirmed", "COOLDOWN", [], 0.0
                state_since = frame
        elif state == "LIVE":
            if s["selected"] and not s["outside"]:
                last_live, gap = frame, 0
            else:
                gap += 1
            if s["outside"] or gap > gap_limit:
                transition, state, state_since = "live_support_lost", "END_PENDING", frame
        elif state == "END_PENDING":
            if s["selected"] and not s["outside"] and frame - state_since <= gap_limit:
                transition, state, gap = "support_recovered", "LIVE", 0
            elif frame - state_since + 1 >= end_confirm:
                assert start is not None
                spans.append((start, last_live if last_live is not None else start,
                              opponent_response, set(review_reasons)))
                transition, state, state_since = "end_confirmed", "COOLDOWN", frame
                start = last_live = None
                serve_events, review_reasons, opponent_response, response_activity_seen, travel_sum, gap = [], set(), False, False, 0.0, 0
        elif state == "COOLDOWN" and frame - state_since + 1 >= cooldown:
            transition, state = "cooldown_complete", "RESET"
        events.append({"type": "frame", "frame": frame, "state": state,
                       "selected_hypothesis": s["selected"]["id"] if s["selected"] else None,
                       "player_readiness": ready, "player_activity": round(p["activity"], 8),
                       "side_evidence": p["opposite_halves"], "contact_evidence": p["burst"],
                       "live_score": round(min(1.0, travel_sum / max(profile.minimum_travel_diagonal, 1e-9)), 6),
                       "end_score": round(min(1.0, gap / max(gap_limit, 1)), 6),
                       "transition": transition, "reason_codes": sorted(review_reasons)})
    if state in {"START_PENDING", "LIVE", "END_PENDING"}:
        boundary_start = start if start is not None else (serve_events[-1][0] if serve_events else state_since)
        spans.append((boundary_start, last_live if last_live is not None else count - 1,
                      opponent_response, set(review_reasons) | {"eof_unresolved"}))
    rallies = []
    for number, (begin, end, response, reasons) in enumerate(spans, 1):
        status = "accepted" if response and not reasons and not degraded else "review"
        if degraded:
            reasons.add("degraded_inputs")
        confidence = 0.85 if status == "accepted" else 0.55
        rallies.append(Rally(source_id=str(source["source_id"]), rally_id=f"{source['source_id']}-{number:04d}",
                             start_frame=begin, end_frame=max(begin, end), start_time=begin / fps,
                             end_time=max(begin, end) / fps, status=status, confidence=confidence,
                             confidence_band="high" if status == "accepted" else "medium",
                             flags=";".join(sorted(reasons | ({"manual_review_needed"} if status == "review" else set())))))
    metadata = {"type": "metadata", "schema": "rally_state_events", "schema_version": SCHEMA_VERSION,
                "decoder_version": DECODER_VERSION, "config": asdict(profile), "source": source,
                "input_fingerprints": fingerprints, "degraded_mode": degraded}
    return rallies, events, metadata


def write_state_events(path: str | Path, metadata: dict[str, Any], events: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in (metadata, *events):
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
