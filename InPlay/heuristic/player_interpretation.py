"""Court-aware singles slot interpretation, incremental pose caching, and CSV output."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import os
import tempfile
import textwrap
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from tqdm.auto import tqdm

from InPlay.heuristic.person_tracks import load_person_tracks, raw_artifact_fingerprint
from src.court_projection import HALF_LENGTH, HALF_WIDTH, CourtHomography
from src.pose_estimator import create_pose_estimator, ensure_pose_model_asset, pose_backend_info
from src.single_video.video import read_video_info
from src.workflow.identity import SourceIdentity

PLAYER_COURT_PADDING_METERS = 1.0
ROLE_RETURN_SECONDS = 0.5
ROLE_LOOKAHEAD_SECONDS = 0.5
ROLE_CONFIRMATION_SECONDS = 0.25
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


def _inside_court(court_x: float, court_y: float, padding: float) -> bool:
    return bool(
        abs(court_x) <= HALF_WIDTH + padding
        and abs(court_y) <= HALF_LENGTH + padding
    )


def _project_observation(
    observation: dict[str, Any], homography: CourtHomography,
    padding: float = PLAYER_COURT_PADDING_METERS,
) -> dict[str, Any]:
    x1, _y1, x2, y2 = observation["bbox"]
    court_x, court_y = homography.project_to_court([((x1 + x2) / 2.0, y2)])[0]
    result = dict(observation)
    result["court_x"] = float(court_x)
    result["court_y"] = float(court_y)
    result["in_court"] = _inside_court(court_x, court_y, padding)
    return result


def _court_role(court_y: float) -> str:
    return "P1" if court_y < 0.0 else "P2"


def assign_singles_slots(
    observations: list[dict[str, Any]], homography: CourtHomography, frame_count: int, fps: float,
) -> list[dict[str, Any]]:
    """Keep each observed incumbent and use bounded lookahead to fill open roles."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    hold_frames = max(1, int(ROLE_RETURN_SECONDS * fps))
    lookahead_frames = max(1, int(np.ceil(ROLE_LOOKAHEAD_SECONDS * fps)))
    minimum_frames = max(1, int(np.ceil(ROLE_CONFIRMATION_SECONDS * fps)))
    roles = ("P1", "P2")
    by_role: dict[str, dict[int, dict[int, dict[str, Any]]]] = {
        role: defaultdict(dict) for role in roles
    }
    for item in observations:
        projected = _project_observation(item, homography)
        if projected["in_court"]:
            frame, track_id = int(item["frame"]), int(item["track_id"])
            if any(track_id in by_role[role].get(frame, {}) for role in roles):
                raise ValueError(f"duplicate observation for track {track_id} at frame {frame}")
            role = _court_role(projected["court_y"])
            by_role[role][frame][track_id] = projected

    incumbent: dict[str, tuple[int, int]] = {}
    frames = []
    for frame in range(frame_count):
        slots = {}
        for role in roles:
            eligible = by_role[role]
            current = eligible.get(frame, {})
            previous = incumbent.get(role)
            observation = None
            candidates = []
            reason = "no_eligible_observation"
            decision = None

            if previous is not None and frame - previous[1] <= hold_frames:
                observation = current.get(previous[0])
                if observation is not None:
                    decision = "retained" if frame == previous[1] + 1 else "returned"
                else:
                    reason = "waiting_for_incumbent"
            elif current:
                horizon = min(frame_count, frame + lookahead_frames)
                evidence = [eligible.get(index, {}) for index in range(frame, horizon)]
                for track_id in current:
                    samples = [items[track_id] for items in evidence if track_id in items]
                    if len(samples) < minimum_frames:
                        continue
                    candidates.append({
                        "track_id": track_id,
                        "observations": len(samples),
                        "inside_court_frames": sum(
                            _inside_court(item["court_x"], item["court_y"], 0.0)
                            for item in samples
                        ),
                        "mean_detection_confidence": float(np.mean([item["confidence"] for item in samples])),
                    })
                candidates.sort(key=lambda item: (
                    -item["inside_court_frames"], -item["mean_detection_confidence"], item["track_id"],
                ))
                reason = "insufficient_observations"
                if candidates:
                    leader = candidates[0]
                    if len(candidates) > 1 and all(
                        leader[key] == candidates[1][key]
                        for key in ("inside_court_frames", "mean_detection_confidence")
                    ):
                        reason = "ambiguous_candidates"
                    else:
                        observation = current[leader["track_id"]]
                        if previous is None:
                            decision = "acquired"
                        else:
                            decision = "returned" if leader["track_id"] == previous[0] else "replacement"

            assignment = None
            confidence = 0.0
            if observation is not None:
                track_id = int(observation["track_id"])
                incumbent[role] = track_id, frame
                confidence = float(observation["confidence"])
                assignment = {
                    "track_id": track_id,
                    "bbox": observation["bbox"],
                    "detection_confidence": confidence,
                    "court_x": observation["court_x"],
                    "court_y": observation["court_y"],
                }
            slots[role] = {
                "assignment": assignment,
                "confidence": confidence,
                "ambiguity_reason": reason if assignment is None else None,
                "candidates": candidates,
            }
            if decision is not None:
                slots[role]["decision"] = decision
        frames.append({"type": "frame", "frame": frame, "slots": slots})
    return frames


def pose_model_fingerprint(backend: dict[str, Any]) -> str:
    normalized = dict(backend)
    model_path = Path(str(normalized.get("model_asset_path", "")))
    if model_path.is_file():
        # Model bytes, backend, and package version define inference identity.
        # Absolute host paths are provenance only and must not invalidate an
        # otherwise identical imported cache.
        normalized.pop("model_asset_path", None)
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        normalized["model_asset_sha256"] = digest
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _legacy_path_pose_model_fingerprint(
    backend: dict[str, Any], model_asset_sha256: str
) -> str:
    """Reconstruct the pre-portability cache key for a verified model digest."""
    normalized = dict(backend)
    normalized["model_asset_sha256"] = model_asset_sha256
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


def _estimate_pose_record(
    image: np.ndarray, bbox: Iterable[float], estimator: Any, padding: float,
) -> dict[str, Any]:
    """Infer one independent image crop, including quality-based padding retry."""
    attempts = []
    paddings = [padding] if padding == POSE_RETRY_PADDING else [padding, POSE_RETRY_PADDING]
    for crop_padding in paddings:
        expanded = expand_pose_bbox(bbox, (image.shape[1], image.shape[0]), crop_padding)
        x1, y1, x2, y2 = [int(round(value)) for value in expanded]
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
    return {
        "pose_bbox": pose_bbox, "status": status, "temporal_status": "original",
        "pose_quality": quality, "pose_attempts": len(attempts), "pose_landmarks": landmarks,
    }


def _pose_preprocessing(padding: float) -> dict[str, Any]:
    implementation = "\n".join(
        ast.dump(ast.parse(textwrap.dedent(inspect.getsource(component))), include_attributes=False)
        for component in (_bbox_key, expand_pose_bbox, pose_quality, _estimate_pose_record)
    )
    return {
        "pose_crop_padding": padding, "retry_padding": POSE_RETRY_PADDING,
        "quality_threshold": POSE_QUALITY_THRESHOLD,
        "quality_landmarks": QUALITY_LANDMARKS,
        "implementation_fingerprint": "sha256:" + hashlib.sha256(implementation.encode()).hexdigest(),
        "runtime": {"numpy": np.__version__, "opencv": cv2.__version__},
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


def _atomic_json_lines(path: str | Path, values: Iterable[dict[str, Any]]) -> None:
    """Publish a complete, synced file without truncating the previous version."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{destination.name}.",
            suffix=".tmp", dir=destination.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for value in values:
                handle.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _pose_unit_name(record: dict[str, Any]) -> str:
    key = [
        record["raw_artifact_fingerprint"], int(record["frame"]), int(record["track_id"]),
        list(_bbox_key(record["bbox"])), record["pose_model_fingerprint"],
        record["preprocessing_fingerprint"],
    ]
    payload = json.dumps(key, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest() + ".json"


def _validate_pose_unit(record: dict[str, Any]) -> None:
    """Reject corrupt committed units before they can suppress inference."""
    if record.get("type") != "pose" or int(record["frame"]) < 0:
        raise ValueError("invalid committed pose work unit")
    if record.get("temporal_status") != "original":
        raise ValueError("committed pose work units must contain original inference results")
    int(record["track_id"])
    for name in ("bbox", "pose_bbox"):
        box = np.asarray(record[name], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError(f"invalid committed pose {name}")
    landmarks = record["pose_landmarks"]
    if landmarks is not None:
        if not isinstance(landmarks, list) or len(landmarks) != 33:
            raise ValueError("invalid committed pose landmark count")
        for landmark in landmarks:
            if not all(np.isfinite(float(landmark[name])) for name in ("x", "y", "z", "visibility")):
                raise ValueError("invalid committed pose landmarks")
    if not np.isfinite(float(record["pose_quality"])):
        raise ValueError("invalid committed pose quality")
    if record.get("status") not in {"detected", "low_quality", "no_pose", "invalid_crop"}:
        raise ValueError("invalid committed pose status")
    if bool(landmarks) != (record["status"] in {"detected", "low_quality"}):
        raise ValueError("committed pose status does not match landmarks")


def compact_pose_cache(path: str | Path, fingerprint_aliases: dict[str, str] | None = None) -> int:
    """Normalize known fingerprint aliases and retain one record per complete cache key."""
    metadata, records = load_pose_cache(path)
    if metadata is None:
        return 0
    aliases = fingerprint_aliases or {}
    unique: dict[tuple[int, int, tuple[float, ...], str, str | None], dict[str, Any]] = {}
    for item in records:
        fingerprint = aliases.get(item["pose_model_fingerprint"], item["pose_model_fingerprint"])
        item["pose_model_fingerprint"] = fingerprint
        key = (int(item["frame"]), int(item["track_id"]), _bbox_key(item["bbox"]),
               fingerprint, item.get("preprocessing_fingerprint"))
        unique[key] = item
    _atomic_json_lines(path, [metadata, *unique.values()])
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
    pose_work_dir: str | Path | None = None,
    pose_backend_identity: dict[str, Any] | None = None,
) -> tuple[dict[tuple[int, int, tuple[float, ...]], dict[str, Any]], int]:
    """Compute absent observations, committing each crop before continuing.

    ``pose_work_dir`` must live on durable storage when the final cache is staged
    in a temporary directory. Completed work is partitioned by raw artifact,
    backend/model, and preprocessing identity; an interrupted attempt can safely
    reuse it without publishing incomplete assignments or player features.
    """
    raw_fingerprint = raw_artifact_fingerprint(person_tracks_path)
    # Resolve/download the production asset before hashing it. Previously the
    # first run fingerprinted a missing path and the second run fingerprinted
    # the downloaded bytes, forcing an unnecessary full recomputation.
    resolved_model_asset = (
        ensure_pose_model_asset(pose_model_asset)
        if estimator_factory is create_pose_estimator
        else pose_model_asset
    )
    backend = pose_backend_info(resolved_model_asset)
    if pose_backend_identity is not None:
        backend["adapter_identity"] = pose_backend_identity
    model_fingerprint = pose_model_fingerprint(backend)
    preprocessing_fingerprint = pose_preprocessing_fingerprint(pose_crop_padding)
    metadata, records = load_pose_cache(pose_cache_path)
    if metadata is not None and metadata.get("raw_artifact_fingerprint") != raw_fingerprint:
        # Person tracks are an immutable source layer, while poses are derived
        # from it.  A notebook rerun may replace the raw artifact in the same
        # result directory, so retaining this cache would attach poses to the
        # wrong detections.  Discard only the derived cache and rebuild it.
        metadata, records = None, []
    cache_needs_upgrade = False
    if metadata is not None and metadata.get("pose_model_fingerprint") != model_fingerprint:
        asset_path = Path(str(backend.get("model_asset_path", "")))
        imported_backend = metadata.get("pose_backend")
        imported_fingerprint = str(metadata.get("pose_model_fingerprint", ""))
        if asset_path.is_file() and isinstance(imported_backend, dict):
            asset_digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
            verified_legacy = _legacy_path_pose_model_fingerprint(
                imported_backend, asset_digest
            )
            comparable_backend = {
                key: value
                for key, value in imported_backend.items()
                if key != "model_asset_path"
            }
            active_backend = {
                key: value for key, value in backend.items() if key != "model_asset_path"
            }
            if verified_legacy == imported_fingerprint and comparable_backend == active_backend:
                for item in records:
                    if item.get("pose_model_fingerprint") == imported_fingerprint:
                        item["pose_model_fingerprint"] = model_fingerprint
                metadata["pose_model_fingerprint"] = model_fingerprint
                metadata["pose_backend"] = backend
                cache_needs_upgrade = True
    legacy_cache_matches_active_preprocessing = bool(
        metadata is not None
        and int(metadata.get("schema_version", 1)) < POSE_CACHE_SCHEMA_VERSION
        and metadata.get("pose_model_fingerprint") == model_fingerprint
        and metadata.get("preprocessing_fingerprint") == preprocessing_fingerprint
    )
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
    work_root = (
        Path(pose_work_dir) if pose_work_dir is not None
        else Path(pose_cache_path).parent / f".{Path(pose_cache_path).name}.work"
    )
    unit_directory = work_root.joinpath(*(
        hashlib.sha256(value.encode()).hexdigest()
        for value in (raw_fingerprint, model_fingerprint, preprocessing_fingerprint)
    ))
    recovered = 0
    for unit_path in sorted(unit_directory.glob("*.json")):
        record = json.loads(unit_path.read_text(encoding="utf-8"))
        _validate_pose_unit(record)
        if (
            record["raw_artifact_fingerprint"] != raw_fingerprint
            or record["pose_model_fingerprint"] != model_fingerprint
            or record["preprocessing_fingerprint"] != preprocessing_fingerprint
            or unit_path.name != _pose_unit_name(record)
        ):
            raise ValueError(f"committed pose work unit identity mismatch: {unit_path}")
        key = (int(record["frame"]), int(record["track_id"]), _bbox_key(record["bbox"]),
               model_fingerprint, preprocessing_fingerprint)
        # Temporal validation depends on the selected assignment's neighbors.
        # Restore the original inference result before validating a new selection.
        if existing.get(key) != record:
            recovered += 1
        existing[key] = record
    # Imported/final caches can contain assignment-dependent interpolation. A
    # matching original journal unit was restored above; without one, infer the
    # crop again instead of treating a synthetic estimate as raw model output.
    existing = {
        key: item for key, item in existing.items()
        if item.get("temporal_status") in (None, "original")
    }
    records = list(existing.values())
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
        estimator = None
        try:
            estimator = estimator_factory(
                model_asset_path=resolved_model_asset, running_mode="image"
            )
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
                    record = {
                        "type": "pose", "frame": key[0], "track_id": key[1],
                        "bbox": list(key[2]), "raw_artifact_fingerprint": raw_fingerprint,
                        "pose_model_fingerprint": model_fingerprint,
                        "preprocessing_fingerprint": preprocessing_fingerprint,
                        **_estimate_pose_record(image, assignment["bbox"], estimator, pose_crop_padding),
                    }
                    _validate_pose_unit(record)
                    _atomic_json_lines(unit_directory / _pose_unit_name(record), [record])
                    new_records.append(record)
                    existing[(*key, model_fingerprint, preprocessing_fingerprint)] = record
                    computed += 1
                    progress.set_postfix(computed=f"{computed}/{len(missing)}", refresh=False)
        finally:
            if estimator is not None:
                estimator.close()
            cap.release()
    cache_metadata = {
        "type": "metadata", "schema": "pose_cache", "schema_version": POSE_CACHE_SCHEMA_VERSION,
        "raw_artifact_fingerprint": raw_fingerprint,
        "pose_model_fingerprint": model_fingerprint, "pose_backend": backend,
        "preprocessing": _pose_preprocessing(pose_crop_padding),
        "preprocessing_fingerprint": preprocessing_fingerprint,
    }
    if new_records or recovered or metadata is None or cache_needs_upgrade:
        _atomic_json_lines(pose_cache_path, [cache_metadata, *records, *new_records])
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


def _record_foot_diagnostics(
    frames: list[dict[str, Any]],
    pose_cache: dict[tuple[int, int, tuple[float, ...]], dict[str, Any]],
    homography: CourtHomography,
) -> None:
    """Record original pose evidence before interpolation can change the feet."""
    for frame in frames:
        for role, slot in frame["slots"].items():
            assignment = slot["assignment"]
            if assignment is None:
                continue
            bbox = assignment["bbox"]
            key = (int(frame["frame"]), int(assignment["track_id"]), _bbox_key(bbox))
            pose = pose_cache.get(key, {})
            landmarks = pose.get("pose_landmarks") or []
            visible_feet = [
                landmark for landmark in landmarks[29:33]
                if landmark.get("visibility", 0.0) >= POSE_VISIBILITY_THRESHOLD
                and np.isfinite([landmark["x"], landmark["y"]]).all()
            ]
            diagnostic = {
                "status": "unavailable",
                "box_court": [assignment["court_x"], assignment["court_y"]],
            }
            slot["foot_diagnostic"] = diagnostic
            if not visible_feet:
                continue
            foot = _landmarks_to_image(visible_feet, pose.get("pose_bbox", bbox)).mean(axis=0)
            try:
                court_x, court_y = homography.project_to_court([foot])[0]
            except ValueError:
                continue
            agrees = (
                _inside_court(court_x, court_y, PLAYER_COURT_PADDING_METERS)
                and _court_role(court_y) == role
            )
            diagnostic.update({
                "status": "in_bounds" if agrees else "disagrees",
                "image_foot": foot.tolist(),
                "pose_court": [float(court_x), float(court_y)],
            })


def derive_activity_and_rows(
    assignments: list[dict[str, Any]],
    pose_cache: dict[tuple[int, int, tuple[float, ...]], dict[str, Any]],
    frame_size: tuple[int, int],
    homography: CourtHomography,
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
            court_x, court_y = homography.project_to_court([foot])[0]
            assignment["court_x"] = float(court_x)
            assignment["court_y"] = float(court_y)
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
    pose_work_dir: str | Path | None = None,
    pose_backend_identity: dict[str, Any] | None = None,
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
    calibration_path = Path(calibration_path).expanduser().resolve()
    identity_path = calibration_path.parent.parent / "source.json"
    if calibration_path.name != "calibration.json" or calibration_path.parent.name != "court":
        raise ValueError("player assignment requires canonical court/calibration.json")
    if not identity_path.is_file():
        raise FileNotFoundError(f"source identity is required: {identity_path}")
    identity = SourceIdentity.read(identity_path)
    calibration_value = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_metadata = calibration_value.get("artifact_metadata", {})
    if (
        calibration_metadata.get("source_id") != identity.source_id
        or calibration_metadata.get("source_identity_sha256") != identity.fingerprint
    ):
        raise ValueError("canonical calibration does not match source identity")
    if (
        identity.video_sha256 != hashlib.sha256(Path(video_path).read_bytes()).hexdigest()
        or identity.frame_count != frame_count
        or (identity.width, identity.height) != frame_size
    ):
        raise ValueError("source identity does not match player-processing video")
    fps = float(info["fps"])
    assignments = assign_singles_slots(observations, homography, frame_count, fps)
    cache, computed = enrich_pose_cache(
        video_path, assignments, person_tracks_path, pose_cache_path,
        pose_model_asset=pose_model_asset, pose_crop_padding=pose_crop_padding,
        estimator_factory=estimator_factory,
        pose_work_dir=pose_work_dir, pose_backend_identity=pose_backend_identity,
    )
    _record_foot_diagnostics(assignments, cache, homography)
    apply_temporal_pose_validation(assignments, cache, frame_size)
    cache_metadata, cache_records = load_pose_cache(pose_cache_path)
    if cache_metadata is not None:
        validated_records = []
        for item in cache_records:
            active_item = cache.get((int(item["frame"]), int(item["track_id"]), _bbox_key(item["bbox"])))
            if (active_item is not None
                    and item.get("preprocessing_fingerprint") == active_item.get("preprocessing_fingerprint")
                    and item.get("pose_model_fingerprint") == active_item.get("pose_model_fingerprint")):
                item = active_item
            validated_records.append(item)
        _atomic_json_lines(pose_cache_path, [cache_metadata, *validated_records])
    rows = derive_activity_and_rows(assignments, cache, frame_size, homography)
    assignment_metadata = {
        "type": "metadata", "schema": "player_assignments", "schema_version": 3,
        "source_id": identity.source_id,
        "source_identity_sha256": identity.fingerprint,
        "frame_indexing": "source_local_zero_based",
        "raw_artifact_fingerprint": raw_artifact_fingerprint(person_tracks_path),
        "calibration": str(calibration_path),
        "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        "frame_size": list(frame_size),
        "fps": float(info["fps"]), "frame_count": frame_count,
        "roles": {"P1": "near", "P2": "far"},
        "parameters": {
            "policy": "continuous",
            "court_padding_meters": PLAYER_COURT_PADDING_METERS,
            "incumbent_return_seconds": ROLE_RETURN_SECONDS,
            "lookahead_seconds": ROLE_LOOKAHEAD_SECONDS,
            "minimum_observation_seconds": ROLE_CONFIRMATION_SECONDS,
            "foot_visibility_threshold": POSE_VISIBILITY_THRESHOLD,
        },
    }
    write_player_assignments(assignments_path, assignment_metadata, assignments)
    write_players_csv(players_csv_path, rows)
    return assignments, computed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use separate assignment, pose-cache, and player CSV paths for diagnostic runs.",
    )
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
