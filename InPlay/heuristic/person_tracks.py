"""Immutable YOLO/ByteTrack person observations and legacy import tools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from src.single_video.video import read_video_info

PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"


def _write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def raw_artifact_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_person_tracks(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with Path(path).open(encoding="utf-8") as handle:
        try:
            metadata = json.loads(next(handle))
        except StopIteration as exc:
            raise ValueError(f"empty person-track artifact: {path}") from exc
        if metadata.get("type") != "metadata" or metadata.get("schema") != "person_tracks":
            raise ValueError(f"invalid person-track metadata: {path}")
        observations = [json.loads(line) for line in handle if line.strip()]
    if any(item.get("type") != "observation" for item in observations):
        raise ValueError(f"person-track artifact contains a non-observation record: {path}")
    return metadata, observations


def extract_person_tracks(
    video: str | Path,
    output: str | Path,
    model: str = "yolov8n.pt",
) -> None:
    """Run YOLO/ByteTrack only; no court or pose interpretation occurs here."""
    from tqdm import tqdm
    from ultralytics import YOLO

    from InPlay.heuristic.players import ensure_person_detector

    info = read_video_info(Path(video))
    detector = YOLO(model)
    ensure_person_detector(detector, model)
    metadata = {
        "type": "metadata",
        "schema": "person_tracks",
        "schema_version": 1,
        "video": str(video),
        "frame_size": [int(info["width"]), int(info["height"])],
        "fps": float(info["fps"]),
        "frame_count": int(info["frame_count"]),
        "detector": {"model": str(model), "class_id": PERSON_CLASS_ID, "class_name": PERSON_CLASS_NAME},
        "tracker": "bytetrack.yaml",
    }
    with Path(output).open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, separators=(",", ":")) + "\n")
        results = detector.track(
            source=str(video), classes=[PERSON_CLASS_ID], tracker="bytetrack.yaml",
            stream=True, persist=True, verbose=False,
        )
        for frame_index, result in enumerate(
            tqdm(results, total=int(info["frame_count"]) or None, desc="YOLO person tracking")
        ):
            if result.boxes.id is None:
                continue
            ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            for track_id, bbox, confidence in zip(ids, boxes, confidences):
                handle.write(json.dumps({
                    "type": "observation", "frame": frame_index, "track_id": int(track_id),
                    "bbox": [float(value) for value in bbox], "confidence": float(confidence),
                }, separators=(",", ":")) + "\n")


def import_legacy_player_poses(
    legacy_path: str | Path,
    person_tracks_path: str | Path,
    pose_cache_path: str | Path,
    *,
    video_path: str | Path | None = None,
    pose_model_fingerprint: str | None = None,
) -> tuple[int, int]:
    """Import raw boxes and attempted pose results without trusting legacy slots."""
    with Path(legacy_path).open(encoding="utf-8") as handle:
        try:
            legacy_metadata = json.loads(next(handle))
        except StopIteration as exc:
            raise ValueError(f"empty legacy player artifact: {legacy_path}") from exc
        frames = [json.loads(line) for line in handle if line.strip()]
    if legacy_metadata.get("type") != "metadata":
        raise ValueError("legacy player artifact must start with metadata")
    width, height = map(int, legacy_metadata["frame_size"])
    resolved_video = Path(video_path or legacy_metadata.get("video", ""))
    video_info = read_video_info(resolved_video) if resolved_video.is_file() else {}
    frame_count = int(video_info.get("frame_count", len(frames)))
    fps = float(video_info.get("fps", legacy_metadata.get("fps", 0.0)))
    raw_metadata = {
        "type": "metadata", "schema": "person_tracks", "schema_version": 1,
        "video": str(resolved_video), "frame_size": [width, height], "fps": fps,
        "frame_count": frame_count, "detector": legacy_metadata.get("detector", {}),
        "tracker": "bytetrack.yaml", "imported_from": str(legacy_path),
    }
    raw_records: list[dict[str, Any]] = [raw_metadata]
    legacy_detections: list[tuple[int, dict[str, Any]]] = []
    for frame in frames:
        frame_index = int(frame["frame"])
        for detection in frame.get("detections", []):
            bbox = [float(value) for value in detection["bbox"]]
            raw_records.append({
                "type": "observation", "frame": frame_index,
                "track_id": int(detection["track_id"]), "bbox": bbox,
                "confidence": float(detection.get("confidence", 1.0)),
            })
            legacy_detections.append((frame_index, detection))
    _write_jsonl(person_tracks_path, raw_records)
    raw_fingerprint = raw_artifact_fingerprint(person_tracks_path)
    model_fingerprint = pose_model_fingerprint or _legacy_pose_fingerprint(legacy_metadata)
    cache_records: list[dict[str, Any]] = [{
        "type": "metadata", "schema": "pose_cache", "schema_version": 1,
        "raw_artifact_fingerprint": raw_fingerprint,
        "pose_model_fingerprint": model_fingerprint,
        "pose_backend": legacy_metadata.get("pose_backend", {}),
        "imported_from": str(legacy_path),
    }]
    seeded = 0
    for frame_index, detection in legacy_detections:
        # selected=True means MediaPipe was actually attempted in the legacy pipeline.
        if not detection.get("selected"):
            continue
        landmarks = detection.get("pose_landmarks")
        cache_records.append({
            "type": "pose", "frame": frame_index, "track_id": int(detection["track_id"]),
            "bbox": [float(value) for value in detection["bbox"]],
            "raw_artifact_fingerprint": raw_fingerprint,
            "pose_model_fingerprint": model_fingerprint,
            "status": "detected" if landmarks else "no_pose",
            "pose_landmarks": landmarks,
        })
        seeded += 1
    _write_jsonl(pose_cache_path, cache_records)
    return len(raw_records) - 1, seeded


def _legacy_pose_fingerprint(metadata: dict[str, Any]) -> str:
    backend = dict(metadata.get("pose_backend", {}))
    model_path = Path(str(backend.get("model_asset_path", "")))
    if model_path.is_file():
        backend["model_asset_path"] = str(model_path.resolve())
        backend["model_asset_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    payload = json.dumps(backend, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="run immutable YOLO/ByteTrack extraction")
    extract.add_argument("--video", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--model", default="yolov8n.pt")
    legacy = subparsers.add_parser("import-legacy", help="import player_poses.jsonl without YOLO")
    legacy.add_argument("--legacy", required=True)
    legacy.add_argument("--person-tracks", required=True)
    legacy.add_argument("--pose-cache", required=True)
    legacy.add_argument("--video")
    args = parser.parse_args(argv)
    if args.command == "extract":
        extract_person_tracks(args.video, args.output, args.model)
    else:
        import_legacy_player_poses(args.legacy, args.person_tracks, args.pose_cache, video_path=args.video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
