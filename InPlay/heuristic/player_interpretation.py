"""Court-aware singles slot interpretation, incremental pose caching, and CSV output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from tqdm.auto import tqdm

from InPlay.heuristic.person_tracks import load_person_tracks, raw_artifact_fingerprint
from src.court_projection import HALF_LENGTH, HALF_WIDTH, CourtHomography
from src.pose_estimator import create_pose_estimator, pose_backend_info
from src.single_video.video import read_video_info

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
ACTIVITY_WINDOW = 30
POSE_CACHE_SCHEMA_VERSION = 2
POSE_CROP_PADDING = 0.20
POSE_RETRY_PADDING = 0.25
POSE_QUALITY_THRESHOLD = 0.55
POSE_VISIBILITY_THRESHOLD = 0.5
TEMPORAL_MAX_BRACKET_MOTION = 0.12
TEMPORAL_JUMP_THRESHOLD = 0.10
TORSO_LANDMARKS = (11, 12, 23, 24)
QUALITY_LANDMARKS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


@dataclass(frozen=True)
class Candidate:
    track_id: int
    observations: int
    center_access: float
    occupancy_ratio: float
    mean_confidence: float
    score: float


def load_calibration(path: str | Path, frame_size: tuple[int, int]) -> CourtHomography:
    """Load a required calibration and reject old or mismatched files explicitly."""
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"court calibration is required: {calibration_path}")
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    if "image_size" not in data:
        raise ValueError(f"court calibration is missing image_size: {calibration_path}")
    image_size = tuple(int(value) for value in data["image_size"])
    if image_size != tuple(frame_size):
        raise ValueError(
            f"court calibration image_size {image_size} does not match video {tuple(frame_size)}"
        )
    return CourtHomography.load(calibration_path)


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


def pose_model_fingerprint(backend: dict[str, Any]) -> str:
    normalized = dict(backend)
    model_path = Path(str(normalized.get("model_asset_path", "")))
    if model_path.is_file():
        # Relative and absolute references to the same model must share a cache key.
        normalized["model_asset_path"] = str(model_path.resolve())
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        normalized["model_asset_sha256"] = digest
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _bbox_key(bbox: Iterable[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 6) for value in bbox)


def expand_pose_bbox(
    bbox: Iterable[float], frame_size: tuple[int, int], padding: float
) -> list[float]:
    """Expand every detector-box edge by a fraction of its original dimension."""
    x1, y1, x2, y2 = map(float, bbox)
    width, height = frame_size
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return [max(0.0, x1 - dx), max(0.0, y1 - dy), min(float(width), x2 + dx), min(float(height), y2 + dy)]


def pose_quality(landmarks: list[dict[str, Any]] | None) -> float:
    """Score visible torso and limb coverage without affecting classifier features."""
    if not landmarks or len(landmarks) <= max(QUALITY_LANDMARKS):
        return 0.0
    visibility = [min(1.0, max(0.0, float(landmarks[i].get("visibility", 0.0)))) for i in QUALITY_LANDMARKS]
    torso = visibility[:2] + visibility[6:8]
    limbs = visibility[2:6] + visibility[8:]
    return float(0.6 * np.mean(torso) + 0.4 * np.mean(limbs))


def _pose_preprocessing(padding: float) -> dict[str, Any]:
    return {
        "pose_crop_padding": padding, "retry_padding": POSE_RETRY_PADDING,
        "quality_threshold": POSE_QUALITY_THRESHOLD,
        "visibility_threshold": POSE_VISIBILITY_THRESHOLD,
        "temporal": {"max_bracket_motion": TEMPORAL_MAX_BRACKET_MOTION,
                     "jump_threshold": TEMPORAL_JUMP_THRESHOLD, "max_gap_frames": 1},
    }


def pose_preprocessing_fingerprint(padding: float) -> str:
    payload = json.dumps(_pose_preprocessing(padding), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _landmarks_to_image(landmarks: list[dict[str, Any]], bbox: Iterable[float]) -> np.ndarray:
    x1, y1, x2, y2 = map(float, bbox)
    return np.asarray([[x1 + p["x"] * (x2 - x1), y1 + p["y"] * (y2 - y1)] for p in landmarks])


def _image_to_landmarks(points: np.ndarray, template: list[dict[str, Any]], bbox: Iterable[float]) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = map(float, bbox)
    width, height = max(x2 - x1, 1e-9), max(y2 - y1, 1e-9)
    return [{**item, "x": float((point[0] - x1) / width), "y": float((point[1] - y1) / height)}
            for point, item in zip(points, template)]


def load_pose_cache(path: str | Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    cache_path = Path(path)
    if not cache_path.exists():
        return None, []
    with cache_path.open(encoding="utf-8") as handle:
        metadata = json.loads(next(handle))
        records = [json.loads(line) for line in handle if line.strip()]
    if metadata.get("schema") != "pose_cache":
        raise ValueError(f"invalid pose-cache metadata: {path}")
    return metadata, records


def compact_pose_cache(path: str | Path, fingerprint_aliases: dict[str, str] | None = None) -> int:
    """Normalize known fingerprint aliases and retain one record per complete cache key."""
    metadata, records = load_pose_cache(path)
    if metadata is None:
        return 0
    aliases = fingerprint_aliases or {}
    unique: dict[tuple[int, int, tuple[float, ...], str], dict[str, Any]] = {}
    for item in records:
        fingerprint = aliases.get(item["pose_model_fingerprint"], item["pose_model_fingerprint"])
        item["pose_model_fingerprint"] = fingerprint
        key = (int(item["frame"]), int(item["track_id"]), _bbox_key(item["bbox"]), fingerprint)
        unique[key] = item
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, separators=(",", ":")) + "\n")
        for item in unique.values():
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return len(records) - len(unique)


def enrich_pose_cache(
    video_path: str | Path,
    assignments: list[dict[str, Any]],
    person_tracks_path: str | Path,
    pose_cache_path: str | Path,
    *,
    pose_model_asset: str | Path | None = None,
    pose_crop_padding: float = POSE_CROP_PADDING,
    estimator_factory: Callable[..., Any] = create_pose_estimator,
) -> tuple[dict[tuple[int, int, tuple[float, ...]], dict[str, Any]], int]:
    """Compute only selected observation keys absent for this raw/model fingerprint."""
    raw_fingerprint = raw_artifact_fingerprint(person_tracks_path)
    backend = pose_backend_info(pose_model_asset)
    model_fingerprint = pose_model_fingerprint(backend)
    preprocessing_fingerprint = pose_preprocessing_fingerprint(pose_crop_padding)
    metadata, records = load_pose_cache(pose_cache_path)
    if metadata is not None and metadata.get("raw_artifact_fingerprint") != raw_fingerprint:
        # Person tracks are an immutable source layer, while poses are derived
        # from it.  A notebook rerun may replace the raw artifact in the same
        # result directory, so retaining this cache would attach poses to the
        # wrong detections.  Discard only the derived cache and rebuild it.
        metadata, records = None, []
    legacy_cache_matches_active_preprocessing = bool(
        metadata is not None
        and int(metadata.get("schema_version", 1)) < POSE_CACHE_SCHEMA_VERSION
        and metadata.get("pose_model_fingerprint") == model_fingerprint
    )
    cache_needs_upgrade = False
    existing = {
        (int(item["frame"]), int(item["track_id"]), _bbox_key(item["bbox"]),
         item.get("pose_model_fingerprint"),
         item.get("preprocessing_fingerprint") or (
             preprocessing_fingerprint if legacy_cache_matches_active_preprocessing else None
         )): item
        for item in records
    }
    if legacy_cache_matches_active_preprocessing:
        for item in records:
            if "preprocessing_fingerprint" not in item:
                item["preprocessing_fingerprint"] = preprocessing_fingerprint
                cache_needs_upgrade = True
    required: dict[tuple[int, int, tuple[float, ...]], dict[str, Any]] = {}
    for frame in assignments:
        for slot in ("P1", "P2"):
            assignment = frame["slots"][slot]["assignment"]
            if assignment is not None:
                key = (int(frame["frame"]), int(assignment["track_id"]), _bbox_key(assignment["bbox"]))
                required[key] = assignment
    missing = [key for key in required if (*key, model_fingerprint, preprocessing_fingerprint) not in existing]
    new_records: list[dict[str, Any]] = []
    if missing:
        missing_by_frame: dict[int, list[tuple[int, int, tuple[float, ...]]]] = defaultdict(list)
        for key in missing:
            missing_by_frame[key[0]].append(key)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"failed to open pose source: {video_path}")
        estimator = estimator_factory(model_asset_path=pose_model_asset, running_mode="image")
        try:
            last_frame = max(missing_by_frame)
            progress = tqdm(
                range(last_frame + 1),
                desc=f"MediaPipe pose enrichment ({len(missing)} crops)",
                unit="frame",
            )
            computed = 0
            for frame_index in progress:
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"video ended before required pose frame {frame_index}")
                for key in missing_by_frame.get(frame_index, []):
                    assignment = required[key]
                    attempts = []
                    paddings = [pose_crop_padding]
                    if pose_crop_padding != POSE_RETRY_PADDING:
                        paddings.append(POSE_RETRY_PADDING)
                    for padding in paddings:
                        pose_bbox = expand_pose_bbox(assignment["bbox"], (image.shape[1], image.shape[0]), padding)
                        x1, y1, x2, y2 = [int(round(value)) for value in pose_bbox]
                        pose_bbox = [float(x1), float(y1), float(x2), float(y2)]
                        crop = image[y1:y2, x1:x2]
                        landmarks = None
                        status = "invalid_crop"
                        if crop.size:
                            result = estimator.estimate_pose(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            landmarks = result.to_raw_landmarks() if result.detected else None
                            status = "detected" if landmarks else "no_pose"
                        attempts.append((pose_quality(landmarks), landmarks, pose_bbox, status))
                        if attempts[-1][0] >= POSE_QUALITY_THRESHOLD:
                            break
                    quality, landmarks, pose_bbox, status = max(attempts, key=lambda item: item[0])
                    if landmarks and quality < POSE_QUALITY_THRESHOLD:
                        status = "low_quality"
                    record = {
                        "type": "pose", "frame": key[0], "track_id": key[1],
                        "bbox": list(key[2]), "raw_artifact_fingerprint": raw_fingerprint,
                        "pose_bbox": pose_bbox, "raw_artifact_fingerprint": raw_fingerprint,
                        "pose_model_fingerprint": model_fingerprint,
                        "preprocessing_fingerprint": preprocessing_fingerprint,
                        "status": status, "temporal_status": "original",
                        "pose_quality": quality, "pose_attempts": len(attempts), "pose_landmarks": landmarks,
                    }
                    new_records.append(record)
                    existing[(*key, model_fingerprint, preprocessing_fingerprint)] = record
                    computed += 1
                    progress.set_postfix(computed=f"{computed}/{len(missing)}", refresh=False)
        finally:
            estimator.close()
            cap.release()
    cache_metadata = {
        "type": "metadata", "schema": "pose_cache", "schema_version": POSE_CACHE_SCHEMA_VERSION,
        "raw_artifact_fingerprint": raw_fingerprint,
        "pose_model_fingerprint": model_fingerprint, "pose_backend": backend,
        "preprocessing": _pose_preprocessing(pose_crop_padding),
        "preprocessing_fingerprint": preprocessing_fingerprint,
    }
    if new_records or metadata is None or cache_needs_upgrade:
        with Path(pose_cache_path).open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(cache_metadata, separators=(",", ":")) + "\n")
            for item in [*records, *new_records]:
                handle.write(json.dumps(item, separators=(",", ":")) + "\n")
    active = {
        (frame, track_id, bbox): item
        for (frame, track_id, bbox, fingerprint, preprocessing), item in existing.items()
        if fingerprint == model_fingerprint and preprocessing == preprocessing_fingerprint
    }
    return active, len(new_records)


def apply_temporal_pose_validation(
    assignments: list[dict[str, Any]],
    pose_cache: dict[tuple[int, int, tuple[float, ...]], dict[str, Any]],
    frame_size: tuple[int, int],
) -> None:
    """Reject isolated torso outliers and fill only bracketed one-frame pose gaps."""
    diagonal = float(np.hypot(*frame_size))
    for slot_name in ("P1", "P2"):
        series: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
        for frame in assignments:
            assignment = frame["slots"][slot_name]["assignment"]
            record = None if assignment is None else pose_cache.get(
                (int(frame["frame"]), int(assignment["track_id"]), _bbox_key(assignment["bbox"]))
            )
            series.append((assignment, record))
        for index in range(1, len(series) - 1):
            prev_assignment, prev = series[index - 1]
            assignment, current = series[index]
            next_assignment, nxt = series[index + 1]
            if not all((prev_assignment, assignment, next_assignment, prev, current, nxt)):
                continue
            if not (prev.get("pose_landmarks") and nxt.get("pose_landmarks")):
                continue
            prev_points = _landmarks_to_image(prev["pose_landmarks"], prev.get("pose_bbox", prev["bbox"]))
            next_points = _landmarks_to_image(nxt["pose_landmarks"], nxt.get("pose_bbox", nxt["bbox"]))
            if len(prev_points) <= max(TORSO_LANDMARKS) or len(next_points) != len(prev_points):
                continue
            prev_torso = prev_points[list(TORSO_LANDMARKS)].mean(axis=0)
            next_torso = next_points[list(TORSO_LANDMARKS)].mean(axis=0)
            if np.linalg.norm(next_torso - prev_torso) / diagonal > TEMPORAL_MAX_BRACKET_MOTION:
                continue
            interpolated = (prev_points + next_points) / 2.0
            usable = current.get("pose_landmarks") and current.get("pose_quality", 0.0) >= POSE_QUALITY_THRESHOLD
            if usable:
                current_points = _landmarks_to_image(current["pose_landmarks"], current.get("pose_bbox", current["bbox"]))
                torso = current_points[list(TORSO_LANDMARKS)].mean(axis=0)
                if np.linalg.norm(torso - (prev_torso + next_torso) / 2.0) / diagonal <= TEMPORAL_JUMP_THRESHOLD:
                    current["temporal_status"] = "validated"
                    continue
                current["temporal_status"] = "rejected_jump_interpolated"
            else:
                current["temporal_status"] = "interpolated_single_gap"
            template = prev["pose_landmarks"]
            current["pose_landmarks"] = _image_to_landmarks(
                interpolated, template, current.get("pose_bbox", current["bbox"])
            )
            current["pose_quality"] = min(float(prev.get("pose_quality", 0)), float(nxt.get("pose_quality", 0)))
            current["status"] = "detected"


def derive_activity_and_rows(
    assignments: list[dict[str, Any]],
    pose_cache: dict[tuple[int, int, tuple[float, ...]], dict[str, Any]],
    frame_size: tuple[int, int],
) -> list[dict[str, Any]]:
    width, height = frame_size
    diagonal = float(np.hypot(width, height))
    previous_pose: dict[int, np.ndarray] = {}
    previous_foot: dict[int, np.ndarray] = {}
    activity_window: deque[float] = deque(maxlen=ACTIVITY_WINDOW)
    rows = []
    for frame in assignments:
        activities = []
        row: dict[str, Any] = {"Frame": int(frame["frame"])}
        for slot_number, slot_name in enumerate(("P1", "P2"), 1):
            assignment = frame["slots"][slot_name]["assignment"]
            if assignment is None:
                row[f"player{slot_number}_valid"] = 0
                continue
            track_id = int(assignment["track_id"])
            bbox = assignment["bbox"]
            foot = np.asarray([(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=float)
            cache = pose_cache.get((int(frame["frame"]), track_id, _bbox_key(bbox)))
            pose_array = None
            if cache and cache.get("pose_landmarks"):
                landmarks = cache["pose_landmarks"]
                pose_bbox = cache.get("pose_bbox", bbox)
                pose_array = np.asarray([
                    [pose_bbox[0] + point["x"] * (pose_bbox[2] - pose_bbox[0]), pose_bbox[1] + point["y"] * (pose_bbox[3] - pose_bbox[1])]
                    for point in landmarks
                ]) / np.asarray([width, height])
                visible_feet = [
                    [pose_bbox[0] + landmarks[index]["x"] * (pose_bbox[2] - pose_bbox[0]), pose_bbox[1] + landmarks[index]["y"] * (pose_bbox[3] - pose_bbox[1])]
                    for index in (29, 30, 31, 32)
                    if index < len(landmarks) and float(landmarks[index].get("visibility", 0.0)) >= 0.5
                ]
                if visible_feet:
                    foot = np.mean(visible_feet, axis=0)
            activity = 0.0
            if pose_array is not None and slot_number in previous_pose:
                activity = float(np.linalg.norm(pose_array - previous_pose[slot_number], axis=1).mean())
            elif slot_number in previous_foot:
                activity = float(np.linalg.norm(foot - previous_foot[slot_number]) / diagonal)
            if pose_array is not None:
                previous_pose[slot_number] = pose_array
            previous_foot[slot_number] = foot
            assignment["activity"] = activity
            assignment["foot"] = [float(foot[0]), float(foot[1])]
            assignment["pose_status"] = cache.get("status") if cache else "missing"
            row.update({
                f"player{slot_number}_track_id": track_id, f"player{slot_number}_valid": 1,
                f"player{slot_number}_foot_x": float(foot[0]), f"player{slot_number}_foot_y": float(foot[1]),
                f"player{slot_number}_activity": activity,
            })
            activities.append(activity)
        row["player_activity"] = min(1.0, sum(activities) * 20) if activities else 0.0
        row["players_inactive"] = int(not activities or max(activities) < 0.001)
        activity_window.append(float(row["player_activity"]))
        row["activity_window"] = sum(activity_window) / len(activity_window)
        row["inactivity_window"] = int(len(activity_window) == ACTIVITY_WINDOW and row["activity_window"] < 0.05)
        rows.append(row)
    return rows


PLAYER_FIELDS = [
    "Frame", "player_activity", "players_inactive", "activity_window", "inactivity_window",
    "player1_track_id", "player1_valid", "player1_foot_x", "player1_foot_y", "player1_activity",
    "player2_track_id", "player2_valid", "player2_foot_x", "player2_foot_y", "player2_activity",
]


def write_player_assignments(path: str | Path, metadata: dict[str, Any], frames: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, separators=(",", ":")) + "\n")
        for frame in frames:
            handle.write(json.dumps(frame, separators=(",", ":")) + "\n")


def write_players_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAYER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def interpret_players(
    video_path: str | Path,
    person_tracks_path: str | Path,
    calibration_path: str | Path,
    pose_cache_path: str | Path,
    assignments_path: str | Path,
    players_csv_path: str | Path,
    *,
    pose_model_asset: str | Path | None = None,
    pose_crop_padding: float = POSE_CROP_PADDING,
    estimator_factory: Callable[..., Any] = create_pose_estimator,
) -> tuple[list[dict[str, Any]], int]:
    raw_metadata, observations = load_person_tracks(person_tracks_path)
    frame_size = tuple(map(int, raw_metadata["frame_size"]))
    info = read_video_info(Path(video_path))
    if frame_size != (int(info["width"]), int(info["height"])):
        raise ValueError(f"person tracks frame_size {frame_size} does not match video")
    frame_count = int(raw_metadata.get("frame_count") or info["frame_count"])
    if frame_count != int(info["frame_count"]):
        raise ValueError("person tracks frame_count does not match video")
    homography = load_calibration(calibration_path, frame_size)
    assignments = assign_singles_slots(
        observations, homography, frame_count, float(info["fps"]), frame_size=frame_size,
    )
    cache, computed = enrich_pose_cache(
        video_path, assignments, person_tracks_path, pose_cache_path,
        pose_model_asset=pose_model_asset, pose_crop_padding=pose_crop_padding,
        estimator_factory=estimator_factory,
    )
    apply_temporal_pose_validation(assignments, cache, frame_size)
    cache_metadata, cache_records = load_pose_cache(pose_cache_path)
    if cache_metadata is not None:
        with Path(pose_cache_path).open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(cache_metadata, separators=(",", ":")) + "\n")
            for item in cache_records:
                active_item = cache.get((int(item["frame"]), int(item["track_id"]), _bbox_key(item["bbox"])))
                if (active_item is not None
                        and item.get("preprocessing_fingerprint") == active_item.get("preprocessing_fingerprint")
                        and item.get("pose_model_fingerprint") == active_item.get("pose_model_fingerprint")):
                    item = active_item
                handle.write(json.dumps(item, separators=(",", ":")) + "\n")
    rows = derive_activity_and_rows(assignments, cache, frame_size)
    assignment_metadata = {
        "type": "metadata", "schema": "player_assignments", "schema_version": 3,
        "raw_artifact_fingerprint": raw_artifact_fingerprint(person_tracks_path),
        "calibration": str(calibration_path), "frame_size": list(frame_size),
        "fps": float(info["fps"]), "frame_count": frame_count,
        "roles": {"P1": "near", "P2": "far"},
        "parameters": {
            "reassign_interval_seconds": REASSIGN_INTERVAL_SECONDS,
            "occupancy_window_seconds": OCCUPANCY_WINDOW_SECONDS,
            "evidence_window_seconds": EVIDENCE_WINDOW_SECONDS,
            "minimum_observation_seconds": MIN_OBSERVATION_SECONDS,
            "center": {"P1": [0.0, -CENTER_Y_METERS], "P2": [0.0, CENTER_Y_METERS]},
            "center_radius_meters": CENTER_RADIUS_METERS,
            "weights": {"center_access": CENTER_ACCESS_WEIGHT,
                        "occupancy_ratio": OCCUPANCY_WEIGHT,
                        "mean_detection_confidence": DETECTION_CONFIDENCE_WEIGHT},
            "minimum_center_access": MIN_CENTER_ACCESS,
            "minimum_score": MIN_TOTAL_SCORE,
            "score_tolerance": SCORE_TOLERANCE,
            "track_handoff": {
                "enabled_for": "selected_track_absent",
                "minimum_detection_confidence": HANDOFF_MIN_DETECTION_CONFIDENCE,
                "minimum_box_iou": HANDOFF_MIN_BOX_IOU,
                "maximum_image_foot_displacement_ratio": HANDOFF_MAX_IMAGE_FOOT_DISPLACEMENT_RATIO,
                "maximum_court_foot_displacement_meters": HANDOFF_MAX_COURT_FOOT_DISPLACEMENT_METERS,
                "minimum_score_lead": HANDOFF_SCORE_LEAD,
            },
        },
    }
    write_player_assignments(assignments_path, assignment_metadata, assignments)
    write_players_csv(players_csv_path, rows)
    return assignments, computed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--person-tracks", required=True)
    parser.add_argument("--court-calibration", required=True)
    parser.add_argument("--pose-cache", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--players-csv", required=True)
    parser.add_argument("--pose-model-asset")
    parser.add_argument("--pose-crop-padding", type=float, choices=(0.15, 0.20, 0.25), default=POSE_CROP_PADDING)
    parser.add_argument("--tracks-csv", help="shuttle tracks used when rendering --preview")
    parser.add_argument("--preview", help="rendered layered MP4 output")
    args = parser.parse_args(argv)
    if bool(args.tracks_csv) != bool(args.preview):
        parser.error("--tracks-csv and --preview must be provided together")
    interpret_players(
        args.video, args.person_tracks, args.court_calibration, args.pose_cache,
        args.assignments, args.players_csv, pose_model_asset=args.pose_model_asset,
        pose_crop_padding=args.pose_crop_padding,
    )
    if args.preview:
        from src.pose_estimator import pose_connections
        from src.single_video import render_player_preview

        render_player_preview(
            Path(args.video), Path(args.tracks_csv), Path(args.person_tracks),
            Path(args.assignments), Path(args.pose_cache), Path(args.preview), pose_connections(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
