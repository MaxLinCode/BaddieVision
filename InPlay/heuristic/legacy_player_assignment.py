"""Periodic court-role assignment used by the six-source rally release."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from src.court_projection import HALF_LENGTH, HALF_WIDTH, CourtHomography

COURT_MARGIN_METERS = 0.3
REASSIGN_INTERVAL_SECONDS = 0.5
OCCUPANCY_WINDOW_SECONDS = 2.0
EVIDENCE_WINDOW_SECONDS = 5.0
MIN_OBSERVATION_SECONDS = 0.5
CENTER_RADIUS_METERS = 2.0
CENTER_Y_METERS = 3.35
CENTER_ACCESS_WEIGHT = 0.65
OCCUPANCY_WEIGHT = 0.25
DETECTION_CONFIDENCE_WEIGHT = 0.10
MIN_CENTER_ACCESS = 0.10
MIN_TOTAL_SCORE = 0.35
SCORE_TOLERANCE = 0.08
HANDOFF_MIN_DETECTION_CONFIDENCE = 0.35
HANDOFF_MIN_BOX_IOU = 0.10
HANDOFF_MAX_IMAGE_FOOT_DISPLACEMENT_RATIO = 0.08
HANDOFF_MAX_COURT_FOOT_DISPLACEMENT_METERS = 1.25
HANDOFF_SCORE_LEAD = SCORE_TOLERANCE


@dataclass(frozen=True)
class Candidate:
    track_id: int
    observations: int
    center_access: float
    occupancy_ratio: float
    mean_confidence: float
    score: float


def _project_observation(observation: dict[str, Any], homography: CourtHomography) -> dict[str, Any]:
    x1, _y1, x2, y2 = observation["bbox"]
    court_x, court_y = homography.project_to_court([((x1 + x2) / 2.0, y2)])[0]
    result = dict(observation)
    result["court_x"] = float(court_x)
    result["court_y"] = float(court_y)
    result["in_court"] = bool(
        -HALF_WIDTH - COURT_MARGIN_METERS <= court_x <= HALF_WIDTH + COURT_MARGIN_METERS
        and -HALF_LENGTH - COURT_MARGIN_METERS <= court_y <= HALF_LENGTH + COURT_MARGIN_METERS
    )
    return result


def _rank_candidates(
    projected: Iterable[dict[str, Any]], evidence_start: int, evidence_end: int,
    occupancy_start: int, occupancy_end: int, slot: int, excluded: set[int], fps: float,
) -> list[Candidate]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in projected:
        frame = int(item["frame"])
        track_id = int(item["track_id"])
        y = float(item["court_y"])
        correct_half = y <= 0.0 if slot == 1 else y >= 0.0
        if (evidence_start <= frame < evidence_end and item["in_court"]
                and correct_half and track_id not in excluded):
            grouped[track_id].append(item)
    candidates = []
    minimum_observations = max(1, int(np.ceil(MIN_OBSERVATION_SECONDS * fps)))
    occupancy_frames = max(1, occupancy_end - occupancy_start)
    center_y = -CENTER_Y_METERS if slot == 1 else CENTER_Y_METERS
    for track_id, items in grouped.items():
        if len(items) < minimum_observations:
            continue
        center_access = float(np.mean([
            np.hypot(float(item["court_x"]), float(item["court_y"]) - center_y)
            <= CENTER_RADIUS_METERS for item in items
        ]))
        occupancy = len({int(item["frame"]) for item in items
                         if occupancy_start <= int(item["frame"]) < occupancy_end})
        occupancy_ratio = min(1.0, occupancy / occupancy_frames)
        mean_confidence = float(np.mean([float(item.get("confidence", 0.0)) for item in items]))
        score = (CENTER_ACCESS_WEIGHT * center_access
                 + OCCUPANCY_WEIGHT * occupancy_ratio
                 + DETECTION_CONFIDENCE_WEIGHT * mean_confidence)
        if center_access >= MIN_CENTER_ACCESS and score >= MIN_TOTAL_SCORE:
            candidates.append(Candidate(
                track_id, len(items), center_access, occupancy_ratio, mean_confidence, score,
            ))
    return sorted(candidates, key=lambda item: (-item.score, item.track_id))


def _choose_candidate(
    candidates: list[Candidate], incumbent: int | None,
) -> tuple[int | None, str | None, float, list[dict[str, Any]]]:
    details = [
        {"track_id": item.track_id, "observations": item.observations,
         "center_access": item.center_access, "occupancy_ratio": item.occupancy_ratio,
         "mean_detection_confidence": item.mean_confidence, "score": item.score}
        for item in candidates
    ]
    if not candidates:
        return None, "insufficient_observations", 0.0, details
    leader = candidates[0]
    incumbent_candidate = next((item for item in candidates if item.track_id == incumbent), None)
    if (
        incumbent_candidate is not None
        and leader.score - incumbent_candidate.score <= SCORE_TOLERANCE
    ):
        chosen = incumbent_candidate
        reason = "incumbent_retained" if chosen.track_id != leader.track_id else None
    elif (
        len(candidates) > 1
        and leader.score - candidates[1].score <= SCORE_TOLERANCE
        and leader.track_id != incumbent
        and candidates[1].track_id != incumbent
    ):
        return None, "ambiguous_candidates", 0.0, details
    else:
        chosen, reason = leader, None
    return chosen.track_id, reason, chosen.score, details


def _bbox_iou(left: Iterable[float], right: Iterable[float]) -> float:
    left_x1, left_y1, left_x2, left_y2 = map(float, left)
    right_x1, right_y1, right_x2, right_y2 = map(float, right)
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    union = ((left_x2 - left_x1) * (left_y2 - left_y1)
             + (right_x2 - right_x1) * (right_y2 - right_y1) - intersection)
    return intersection / union if union > 0 else 0.0


def _foot_from_bbox(bbox: Iterable[float]) -> np.ndarray:
    x1, _y1, x2, y2 = map(float, bbox)
    return np.asarray([(x1 + x2) / 2.0, y2], dtype=float)


def _choose_track_handoff(
    current_observations: Iterable[dict[str, Any]], previous_assignment: dict[str, Any] | None,
    slot: int, used: set[int], candidate_details: list[dict[str, Any]], frame_diagonal: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Choose a clearly better, continuous raw-ID fragment for one missing slot."""
    if previous_assignment is None:
        return None, None
    scores = {int(item["track_id"]): float(item["score"]) for item in candidate_details}
    previous_bbox = previous_assignment["bbox"]
    previous_image_foot = _foot_from_bbox(previous_bbox)
    previous_court_foot = np.asarray(
        [float(previous_assignment["court_x"]), float(previous_assignment["court_y"])], dtype=float
    )
    eligible: list[tuple[float, int, dict[str, Any], dict[str, float]]]=[]
    for observation in current_observations:
        track_id = int(observation["track_id"])
        correct_half = float(observation["court_y"]) <= 0.0 if slot == 1 else float(observation["court_y"]) >= 0.0
        if (
            track_id in used
            or track_id not in scores
            or not observation["in_court"]
            or not correct_half
            or float(observation.get("confidence", 0.0)) < HANDOFF_MIN_DETECTION_CONFIDENCE
        ):
            continue
        overlap = _bbox_iou(previous_bbox, observation["bbox"])
        image_displacement = float(np.linalg.norm(_foot_from_bbox(observation["bbox"]) - previous_image_foot))
        court_displacement = float(np.linalg.norm(
            np.asarray([float(observation["court_x"]), float(observation["court_y"])]) - previous_court_foot
        ))
        continuous = (
            overlap >= HANDOFF_MIN_BOX_IOU
            or image_displacement <= HANDOFF_MAX_IMAGE_FOOT_DISPLACEMENT_RATIO * frame_diagonal
            or court_displacement <= HANDOFF_MAX_COURT_FOOT_DISPLACEMENT_METERS
        )
        if continuous:
            eligible.append((scores[track_id], track_id, observation, {
                "box_iou": overlap,
                "image_foot_displacement": image_displacement,
                "court_foot_displacement": court_displacement,
            }))
    eligible.sort(key=lambda item: (-item[0], item[1]))
    if not eligible:
        return None, None
    leader = eligible[0]
    if len(eligible) > 1 and leader[0] - eligible[1][0] <= HANDOFF_SCORE_LEAD:
        return None, None
    return leader[2], {
        "reason": "track_handoff", "prior_track_id": int(previous_assignment["track_id"]),
        "replacement_track_id": leader[1], "role_score": leader[0], **leader[3],
    }


def assign_singles_slots(
    observations: list[dict[str, Any]], homography: CourtHomography, frame_count: int, fps: float,
    *, frame_size: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Assign near P1/far P2 periodically while treating raw IDs as fragments."""
    projected = [_project_observation(item, homography) for item in observations]
    by_frame: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for item in projected:
        by_frame[int(item["frame"])][int(item["track_id"])] = item
    periods: list[tuple[int, int, dict[int, dict[str, Any]]]] = []
    incumbents: dict[int, int | None] = {1: None, 2: None}
    if fps <= 0:
        raise ValueError("fps must be positive")
    reassign_interval = max(1, int(round(REASSIGN_INTERVAL_SECONDS * fps)))
    evidence_window = max(1, int(round(EVIDENCE_WINDOW_SECONDS * fps)))
    occupancy_window = max(1, int(round(OCCUPANCY_WINDOW_SECONDS * fps)))
    for center in range(0, frame_count, reassign_interval):
        evidence_start = max(0, center - evidence_window // 2)
        evidence_end = min(frame_count, center + (evidence_window + 1) // 2)
        occupancy_start = max(0, center - occupancy_window // 2)
        occupancy_end = min(frame_count, center + (occupancy_window + 1) // 2)
        chosen: dict[int, dict[str, Any]] = {}
        excluded: set[int] = set()
        for slot in (1, 2):
            candidates = _rank_candidates(
                projected, evidence_start, evidence_end, occupancy_start, occupancy_end,
                slot, excluded, fps,
            )
            track_id, reason, confidence, details = _choose_candidate(candidates, incumbents[slot])
            chosen[slot] = {
                "track_id": track_id, "confidence": confidence, "reason": reason,
                "candidates": details, "evidence_window": [evidence_start, evidence_end],
                "occupancy_window": [occupancy_start, occupancy_end],
            }
            if track_id is not None:
                excluded.add(track_id)
                incumbents[slot] = track_id
        periods.append((center, min(frame_count, center + reassign_interval), chosen))

    frames: list[dict[str, Any]] = []
    if frame_size is None:
        image_extent = max((max(map(float, item["bbox"])) for item in projected), default=1.0)
        frame_diagonal = float(np.hypot(image_extent, image_extent))
    else:
        frame_diagonal = float(np.hypot(*frame_size))
    for start, end, chosen in periods:
        active_track_ids = {slot: chosen[slot]["track_id"] for slot in (1, 2)}
        previous_assignments: dict[int, dict[str, Any] | None] = {1: None, 2: None}
        for frame_index in range(start, end):
            slots: dict[str, Any] = {}
            used: set[int] = set()
            for slot in (1, 2):
                decision = chosen[slot]
                track_id = active_track_ids[slot]
                observation = by_frame.get(frame_index, {}).get(track_id) if track_id is not None else None
                assignment = None
                reason = decision["reason"]
                handoff = None
                if observation is None and track_id is not None:
                    reason = "selected_track_absent"
                    replacement, handoff = _choose_track_handoff(
                        by_frame.get(frame_index, {}).values(), previous_assignments[slot], slot,
                        used, decision["candidates"], frame_diagonal,
                    )
                    if replacement is not None:
                        observation = replacement
                        track_id = int(observation["track_id"])
                        active_track_ids[slot] = track_id
                        reason = "track_handoff"
                if observation is not None:
                    correct_half = observation["court_y"] <= 0 if slot == 1 else observation["court_y"] >= 0
                    if not observation["in_court"]:
                        reason = "outside_court"
                    elif not correct_half:
                        reason = "wrong_half"
                    elif track_id in used:
                        reason = "track_already_assigned"
                    else:
                        used.add(track_id)
                        assignment = {
                            "track_id": track_id,
                            "bbox": observation["bbox"],
                            "detection_confidence": float(observation.get("confidence", 0.0)),
                            "court_x": observation["court_x"], "court_y": observation["court_y"],
                        }
                        previous_assignments[slot] = assignment
                slots[f"P{slot}"] = {
                    "assignment": assignment, "confidence": decision["confidence"],
                    "ambiguity_reason": reason, "candidates": decision["candidates"],
                }
                if handoff is not None:
                    slots[f"P{slot}"]["handoff"] = handoff
            frames.append({"type": "frame", "frame": frame_index, "slots": slots})
    return frames
