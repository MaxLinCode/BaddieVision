"""Preview and output artifact helpers for single-video extraction."""

from __future__ import annotations

import csv
import json
import sys
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import cv2


class _FFmpegPreviewWriter:
    """Pipe full-range OpenCV BGR frames into a tagged SDR H.264 encoder."""
    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg is required to encode color-managed previews")
        width, height = size
        self.process = subprocess.Popen([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pixel_format", "bgr24", "-video_size", f"{width}x{height}", "-framerate", f"{fps:.9f}",
            "-i", "pipe:0", "-an", "-vf", "scale=in_range=full:out_range=limited,format=yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-color_range", "tv", "-colorspace", "bt709", "-color_trc", "bt709",
            "-color_primaries", "bt709", "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(frame.tobytes())

    def release(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", "replace") if self.process.stderr else ""
        if self.process.wait():
            raise RuntimeError(f"FFmpeg preview encoding failed: {stderr.strip()}")


def load_track_rows(path: Path) -> dict[int, dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return {int(row["Frame"]): row for row in csv.DictReader(handle)}


def _draw_pose_landmarks(frame, bbox, pose_landmarks, pose_connection_pairs: Sequence[tuple[int, int]]) -> None:
    if not pose_landmarks:
        return
    from InPlay.heuristic.players import crop_point_to_image

    points = []
    for item in pose_landmarks:
        x, y = crop_point_to_image((item["x"], item["y"]), bbox)
        points.append((int(round(x)), int(round(y)), float(item.get("visibility", 0.0))))
    for start, end in pose_connection_pairs:
        if start >= len(points) or end >= len(points):
            continue
        ax, ay, av = points[start]
        bx, by, bv = points[end]
        if av >= 0.5 and bv >= 0.5:
            cv2.line(frame, (ax, ay), (bx, by), (255, 180, 0), 2, cv2.LINE_AA)
    for x, y, visibility in points:
        if visibility >= 0.5:
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)


def render_preview(
    video_path: Path,
    tracks_csv: Path,
    player_raw_jsonl: Path,
    output_path: Path,
    pose_connection_pairs: Sequence[tuple[int, int]],
) -> None:
    track_rows = load_track_rows(tracks_csv)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open preview source: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    writer = _FFmpegPreviewWriter(output_path, fps, (width, height))
    try:
        with Path(player_raw_jsonl).open(encoding="utf-8") as handle:
            metadata = json.loads(next(handle))
            if metadata.get("type") != "metadata":
                raise ValueError("player raw artifact must start with a metadata record")
            for raw_line in handle:
                ok, frame = cap.read()
                if not ok:
                    break
                player_frame = json.loads(raw_line)
                if player_frame.get("type") != "frame":
                    continue
                frame_index = int(player_frame["frame"])
                track = track_rows.get(frame_index)
                if track and int(track.get("Visibility", 0)) == 1:
                    x, y = int(float(track["X"])), int(float(track["Y"]))
                    cv2.circle(frame, (x, y), 7, (0, 255, 0), 2)
                    cv2.putText(frame, "Shuttle", (x + 8, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                for detection in player_frame.get("detections", []):
                    bbox = detection.get("bbox")
                    if not bbox:
                        continue
                    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
                    color = (0, 200, 255) if detection.get("selected") else (120, 120, 120)
                    thickness = 2 if detection.get("selected") else 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    label = f"ID {int(detection['track_id'])}"
                    if detection.get("slot"):
                        label = f"P{int(detection['slot'])} {label}"
                    if detection.get("activity") is not None:
                        label += f" act {float(detection['activity']):.3f}"
                    cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                    _draw_pose_landmarks(frame, bbox, detection.get("pose_landmarks"), pose_connection_pairs)
                writer.write(frame)
    finally:
        cap.release()
        writer.release()
    print(f"Preview written to {output_path}")


def _read_layered_jsonl(path: Path, expected_schema: str) -> tuple[dict, list[dict]]:
    with Path(path).open(encoding="utf-8") as handle:
        metadata = json.loads(next(handle))
        if metadata.get("schema") != expected_schema:
            raise ValueError(f"expected {expected_schema} artifact: {path}")
        return metadata, [json.loads(line) for line in handle if line.strip()]


def render_player_preview(
    video_path: Path,
    tracks_csv: Path,
    person_tracks_jsonl: Path,
    assignments_jsonl: Path,
    pose_cache_jsonl: Path,
    output_path: Path,
    pose_connection_pairs: Sequence[tuple[int, int]],
) -> None:
    """Render gray raw detections plus court-interpreted slots and cached poses."""
    track_rows = load_track_rows(tracks_csv)
    _, observations = _read_layered_jsonl(person_tracks_jsonl, "person_tracks")
    _, assignment_frames = _read_layered_jsonl(assignments_jsonl, "player_assignments")
    _, poses = _read_layered_jsonl(pose_cache_jsonl, "pose_cache")
    raw_by_frame: dict[int, list[dict]] = {}
    for observation in observations:
        raw_by_frame.setdefault(int(observation["frame"]), []).append(observation)
    assignments_by_frame = {int(frame["frame"]): frame for frame in assignment_frames}
    pose_by_key = {
        (int(item["frame"]), int(item["track_id"]), tuple(round(float(v), 6) for v in item["bbox"])): item
        for item in poses
    }
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open preview source: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    writer = _FFmpegPreviewWriter(output_path, fps, (width, height))
    colors = {"P1": (0, 170, 255), "P2": (255, 120, 20)}
    frame_index = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            shuttle = track_rows.get(frame_index)
            if shuttle and int(shuttle.get("Visibility", 0)) == 1:
                x, y = int(float(shuttle["X"])), int(float(shuttle["Y"]))
                cv2.circle(image, (x, y), 7, (0, 255, 0), 2)
                cv2.putText(image, "Shuttle", (x + 8, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            for raw in raw_by_frame.get(frame_index, []):
                x1, y1, x2, y2 = [int(round(value)) for value in raw["bbox"]]
                cv2.rectangle(image, (x1, y1), (x2, y2), (105, 105, 105), 1)
                cv2.putText(image, f"raw {raw['track_id']}", (x1, max(14, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
            interpreted = assignments_by_frame.get(frame_index, {"slots": {}})
            status_lines = []
            for slot_name in ("P1", "P2"):
                slot = interpreted.get("slots", {}).get(slot_name, {})
                assignment = slot.get("assignment")
                reason = slot.get("ambiguity_reason")
                confidence = float(slot.get("confidence", 0.0))
                if assignment is None:
                    status_lines.append(f"{slot_name}: unassigned ({reason or 'none'})")
                    continue
                bbox = assignment["bbox"]
                x1, y1, x2, y2 = [int(round(value)) for value in bbox]
                color = colors[slot_name]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                label = f"{slot_name} raw {assignment['track_id']} conf {confidence:.2f} act {float(assignment.get('activity', 0.0)):.3f}"
                cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                key = (frame_index, int(assignment["track_id"]), tuple(round(float(v), 6) for v in bbox))
                pose = pose_by_key.get(key, {})
                _draw_pose_landmarks(image, bbox, pose.get("pose_landmarks"), pose_connection_pairs)
                status_lines.append(f"{slot_name}: {assignment['track_id']} {assignment.get('pose_status', pose.get('status', 'missing'))}")
            for index, line in enumerate(status_lines):
                cv2.putText(image, line, (12, 22 + index * 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colors.get(line[:2], (230, 230, 230)), 1, cv2.LINE_AA)
            writer.write(image)
            frame_index += 1
    finally:
        cap.release()
        writer.release()
    print(f"Preview written to {output_path}")


def write_metadata(
    path: Path,
    source_id: str,
    original_video_path: Path,
    working_video_path: Path,
    original_video_info: dict[str, object],
    working_video_info: dict[str, object],
    models: dict[str, object],
    start_time_sec: int | None,
    end_time_sec: int | None,
    target_fps: int | None,
    *,
    player_detector: str,
    pose_backend: dict[str, object],
) -> None:
    import torch

    metadata = {
        "source_id": source_id,
        "original_filename": Path(original_video_path).name,
        "working_filename": Path(working_video_path).name,
        "fps": working_video_info["fps"],
        "width": working_video_info["width"],
        "height": working_video_info["height"],
        "frame_count": working_video_info["frame_count"],
        "original_video": {key: original_video_info[key] for key in ("fps", "width", "height", "frame_count")},
        "working_video": {
            **{key: working_video_info[key] for key in ("fps", "width", "height", "frame_count")},
            "target_fps": target_fps,
            "is_downsampled": target_fps is not None and working_video_info["fps"] + 1e-6 < original_video_info["fps"],
        },
        "segment": {
            "start_time_sec": start_time_sec,
            "end_time_sec": end_time_sec,
            "is_clipped": start_time_sec is not None or end_time_sec is not None,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_info": {
            "tracknet_checkpoint": models["tracknet_checkpoint"],
            "inpaintnet_checkpoint": models["inpaintnet_checkpoint"],
            "player_detector": player_detector,
            "player_detector_class_id": 0,
            "player_detector_class_name": "person",
            "pose_backend": pose_backend,
        },
        "runtime_info": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    }
    for section, info in (("original_video", original_video_info), ("working_video", working_video_info)):
        metadata[section]["color"] = {
            key: info.get(key) for key in (
                "codec_name", "pix_fmt", "color_range", "color_primaries", "color_transfer",
                "color_space", "dolby_vision",
            )
        }
    metadata["color_management"] = {
        "source_mode": original_video_info.get("source_mode"),
        "conversion_applied": working_video_info.get("conversion_applied"),
        "tone_map": working_video_info.get("tone_map"),
        "encoder": working_video_info.get("encoder", "libx264"),
    }
    Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata written to {path}")


def write_result_manifest(path: Path, result_dir: Path, source_id: str, artifact_paths: Iterable[Path]) -> dict[str, object]:
    result_dir = Path(result_dir)
    files = []
    for artifact_path in map(Path, artifact_paths):
        files.append({
            "name": artifact_path.name,
            "path": str(artifact_path),
            "relative_path": str(artifact_path.relative_to(result_dir)),
            "size_bytes": artifact_path.stat().st_size if artifact_path.is_file() else None,
        })
    manifest = {
        "source_id": source_id,
        "result_dir": str(result_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def zip_result_dir(
    result_dir: Path, zip_path: Path, artifact_paths: Iterable[Path] | None = None
) -> None:
    result_dir, zip_path = Path(result_dir), Path(zip_path)
    files = (
        sorted(map(Path, artifact_paths))
        if artifact_paths is not None
        else sorted(result_dir.rglob("*"))
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            if file_path.is_file() and file_path != zip_path:
                zf.write(file_path, arcname=file_path.relative_to(result_dir.parent))
    print(f"ZIP written to {zip_path}")
