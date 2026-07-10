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

from InPlay.heuristic.person_tracks import load_person_tracks, raw_artifact_fingerprint
from src.court_projection import HALF_LENGTH, HALF_WIDTH, CourtHomography
from src.pose_estimator import create_pose_estimator, pose_backend_info
from src.single_video.video import read_video_info

COURT_MARGIN_METERS = 0.3
REASSIGN_INTERVAL = 15
ASSIGNMENT_WINDOW = 60
MIN_IN_COURT_OBSERVATIONS = 15
INCUMBENT_RECENCY = 30
INCUMBENT_OCCUPANCY_RATIO = 0.8
AMBIGUITY_RATIO = 0.9
ACTIVITY_WINDOW = 30


@dataclass(frozen=True)
class Candidate:
    track_id: int
    occupancy: int
    mean_confidence: float
    last_frame: int


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
    projected: Iterable[dict[str, Any]], start: int, end: int, slot: int, excluded: set[int]
) -> list[Candidate]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in projected:
        frame = int(item["frame"])
        track_id = int(item["track_id"])
        y = float(item["court_y"])
        correct_half = y <= 0.0 if slot == 1 else y >= 0.0
        if start <= frame < end and item["in_court"] and correct_half and track_id not in excluded:
            grouped[track_id].append(item)
    candidates = [
        Candidate(
            track_id=track_id,
            occupancy=len(items),
            mean_confidence=float(np.mean([float(item.get("confidence", 0.0)) for item in items])),
            last_frame=max(int(item["frame"]) for item in items),
        )
        for track_id, items in grouped.items()
        if len(items) >= MIN_IN_COURT_OBSERVATIONS
    ]
    return sorted(candidates, key=lambda item: (-item.occupancy, -item.mean_confidence, item.track_id))


def _choose_candidate(
    candidates: list[Candidate], incumbent: int | None, center: int
) -> tuple[int | None, str | None, float, list[dict[str, Any]]]:
    details = [
        {"track_id": item.track_id, "occupancy": item.occupancy, "mean_confidence": item.mean_confidence}
        for item in candidates
    ]
    if not candidates:
        return None, "insufficient_observations", 0.0, details
    leader = candidates[0]
    incumbent_candidate = next((item for item in candidates if item.track_id == incumbent), None)
    if (
        incumbent_candidate is not None
        and center - incumbent_candidate.last_frame <= INCUMBENT_RECENCY
        and incumbent_candidate.occupancy >= INCUMBENT_OCCUPANCY_RATIO * leader.occupancy
    ):
        chosen = incumbent_candidate
        reason = "incumbent_retained" if chosen.track_id != leader.track_id else None
    elif (
        len(candidates) > 1
        and candidates[1].occupancy >= AMBIGUITY_RATIO * leader.occupancy
        and leader.track_id != incumbent
        and candidates[1].track_id != incumbent
    ):
        return None, "ambiguous_candidates", 0.0, details
    else:
        chosen, reason = leader, None
    confidence = chosen.occupancy / ASSIGNMENT_WINDOW
    return chosen.track_id, reason, min(1.0, confidence), details


def assign_singles_slots(
    observations: list[dict[str, Any]], homography: CourtHomography, frame_count: int
) -> list[dict[str, Any]]:
    """Assign near P1/far P2 periodically while treating raw IDs as fragments."""
    projected = [_project_observation(item, homography) for item in observations]
    by_frame: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for item in projected:
        by_frame[int(item["frame"])][int(item["track_id"])] = item
    periods: list[tuple[int, int, dict[int, dict[str, Any]]]] = []
    incumbents: dict[int, int | None] = {1: None, 2: None}
    for center in range(0, frame_count, REASSIGN_INTERVAL):
        start = max(0, center - ASSIGNMENT_WINDOW // 2)
        end = min(frame_count, center + ASSIGNMENT_WINDOW // 2)
        chosen: dict[int, dict[str, Any]] = {}
        excluded: set[int] = set()
        for slot in (1, 2):
            candidates = _rank_candidates(projected, start, end, slot, excluded)
            track_id, reason, confidence, details = _choose_candidate(candidates, incumbents[slot], center)
            chosen[slot] = {
                "track_id": track_id, "confidence": confidence, "reason": reason,
                "candidates": details, "window": [start, end],
            }
            if track_id is not None:
                excluded.add(track_id)
                incumbents[slot] = track_id
        periods.append((center, min(frame_count, center + REASSIGN_INTERVAL), chosen))

    frames: list[dict[str, Any]] = []
    for start, end, chosen in periods:
        for frame_index in range(start, end):
            slots: dict[str, Any] = {}
            used: set[int] = set()
            for slot in (1, 2):
                decision = chosen[slot]
                track_id = decision["track_id"]
                observation = by_frame.get(frame_index, {}).get(track_id) if track_id is not None else None
                assignment = None
                reason = decision["reason"]
                if observation is None and track_id is not None:
                    reason = "selected_track_absent"
                elif observation is not None:
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
                slots[f"P{slot}"] = {
                    "assignment": assignment, "confidence": decision["confidence"],
                    "ambiguity_reason": reason, "candidates": decision["candidates"],
                }
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
    estimator_factory: Callable[..., Any] = create_pose_estimator,
) -> tuple[dict[tuple[int, int, tuple[float, ...]], dict[str, Any]], int]:
    """Compute only selected observation keys absent for this raw/model fingerprint."""
    raw_fingerprint = raw_artifact_fingerprint(person_tracks_path)
    backend = pose_backend_info(pose_model_asset)
    model_fingerprint = pose_model_fingerprint(backend)
    metadata, records = load_pose_cache(pose_cache_path)
    if metadata is not None and metadata.get("raw_artifact_fingerprint") != raw_fingerprint:
        raise ValueError("pose cache belongs to a different raw person-track artifact")
    existing = {
        (int(item["frame"]), int(item["track_id"]), _bbox_key(item["bbox"]), item.get("pose_model_fingerprint")): item
        for item in records
    }
    required: dict[tuple[int, int, tuple[float, ...]], dict[str, Any]] = {}
    for frame in assignments:
        for slot in ("P1", "P2"):
            assignment = frame["slots"][slot]["assignment"]
            if assignment is not None:
                key = (int(frame["frame"]), int(assignment["track_id"]), _bbox_key(assignment["bbox"]))
                required[key] = assignment
    missing = [key for key in required if (*key, model_fingerprint) not in existing]
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
            for frame_index in range(last_frame + 1):
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"video ended before required pose frame {frame_index}")
                for key in missing_by_frame.get(frame_index, []):
                    assignment = required[key]
                    x1, y1, x2, y2 = [int(round(value)) for value in assignment["bbox"]]
                    crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
                    landmarks = None
                    status = "invalid_crop"
                    if crop.size:
                        result = estimator.estimate_pose(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        landmarks = result.to_raw_landmarks() if result.detected else None
                        status = "detected" if landmarks else "no_pose"
                    record = {
                        "type": "pose", "frame": key[0], "track_id": key[1],
                        "bbox": list(key[2]), "raw_artifact_fingerprint": raw_fingerprint,
                        "pose_model_fingerprint": model_fingerprint, "status": status,
                        "pose_landmarks": landmarks,
                    }
                    new_records.append(record)
                    existing[(*key, model_fingerprint)] = record
        finally:
            estimator.close()
            cap.release()
    cache_metadata = metadata or {
        "type": "metadata", "schema": "pose_cache", "schema_version": 1,
        "raw_artifact_fingerprint": raw_fingerprint,
        "pose_model_fingerprint": model_fingerprint, "pose_backend": backend,
    }
    if new_records or metadata is None:
        with Path(pose_cache_path).open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(cache_metadata, separators=(",", ":")) + "\n")
            for item in [*records, *new_records]:
                handle.write(json.dumps(item, separators=(",", ":")) + "\n")
    active = {
        (frame, track_id, bbox): item
        for (frame, track_id, bbox, fingerprint), item in existing.items()
        if fingerprint == model_fingerprint
    }
    return active, len(new_records)


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
                pose_array = np.asarray([
                    [bbox[0] + point["x"] * (bbox[2] - bbox[0]), bbox[1] + point["y"] * (bbox[3] - bbox[1])]
                    for point in landmarks
                ]) / np.asarray([width, height])
                visible_feet = [
                    [bbox[0] + landmarks[index]["x"] * (bbox[2] - bbox[0]), bbox[1] + landmarks[index]["y"] * (bbox[3] - bbox[1])]
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
    assignments = assign_singles_slots(observations, homography, frame_count)
    cache, computed = enrich_pose_cache(
        video_path, assignments, person_tracks_path, pose_cache_path,
        pose_model_asset=pose_model_asset, estimator_factory=estimator_factory,
    )
    rows = derive_activity_and_rows(assignments, cache, frame_size)
    assignment_metadata = {
        "type": "metadata", "schema": "player_assignments", "schema_version": 1,
        "raw_artifact_fingerprint": raw_artifact_fingerprint(person_tracks_path),
        "calibration": str(calibration_path), "frame_size": list(frame_size),
        "fps": float(info["fps"]), "frame_count": frame_count,
        "roles": {"P1": "near", "P2": "far"},
        "parameters": {
            "reassign_interval": REASSIGN_INTERVAL, "window": ASSIGNMENT_WINDOW,
            "minimum_observations": MIN_IN_COURT_OBSERVATIONS,
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
    parser.add_argument("--tracks-csv", help="shuttle tracks used when rendering --preview")
    parser.add_argument("--preview", help="rendered layered MP4 output")
    args = parser.parse_args(argv)
    if bool(args.tracks_csv) != bool(args.preview):
        parser.error("--tracks-csv and --preview must be provided together")
    interpret_players(
        args.video, args.person_tracks, args.court_calibration, args.pose_cache,
        args.assignments, args.players_csv, pose_model_asset=args.pose_model_asset,
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
