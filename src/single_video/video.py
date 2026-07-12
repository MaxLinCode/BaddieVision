"""Video inspection and MP4 passthrough helpers for single-video notebooks."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
from tqdm.auto import tqdm


_FPS_TOLERANCE = 0.05


def probe_video(path: Path, ffprobe_path: str | None = None) -> dict[str, object]:
    """Return the first video stream's structural and colour metadata."""
    executable = ffprobe_path or shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("ffprobe is required for detailed video inspection")
    result = subprocess.run(
        [executable, "-v", "error", "-select_streams", "v:0", "-show_streams",
         "-show_entries", "stream=codec_name,pix_fmt,color_range,color_space,color_transfer,color_primaries,avg_frame_rate,r_frame_rate,nb_frames,width,height:stream_tags:stream_side_data",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream in {path}")
    stream = streams[0]
    stream["dolby_vision"] = any(
        "dovi" in json.dumps(item).lower() or "dolby vision" in json.dumps(item).lower()
        for item in stream.get("side_data_list", [])
    )
    return stream


def read_video_info(path: Path) -> dict[str, object]:
    """Read the video properties used by extraction and artifact metadata."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        return {
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()


def segment_suffix(start_time_sec: int | float | None, end_time_sec: int | float | None) -> str:
    if start_time_sec is None and end_time_sec is None:
        return "full"
    start_label = 0 if start_time_sec is None else int(start_time_sec)
    end_label = "end" if end_time_sec is None else int(end_time_sec)
    return f"{start_label}s_to_{end_label}s"


def _validate_segment(
    info: dict[str, object],
    start_time_sec: int | float | None,
    end_time_sec: int | float | None,
    target_fps: int | float | None,
) -> None:
    fps = float(info["fps"])
    frame_count = int(info["frame_count"])
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError(f"Video reports invalid FPS: {fps}")
    if frame_count <= 0:
        raise ValueError(f"Video reports invalid frame count: {frame_count}")
    if target_fps is not None and (target_fps <= 0 or not math.isfinite(float(target_fps))):
        raise ValueError(f"target_fps must be finite and positive, got {target_fps}")
    if start_time_sec is not None and (start_time_sec < 0 or not math.isfinite(float(start_time_sec))):
        raise ValueError(f"start_time_sec must be finite and non-negative, got {start_time_sec}")
    if end_time_sec is not None and (end_time_sec < 0 or not math.isfinite(float(end_time_sec))):
        raise ValueError(f"end_time_sec must be finite and non-negative, got {end_time_sec}")
    duration = frame_count / fps
    effective_start = 0.0 if start_time_sec is None else float(start_time_sec)
    effective_end = duration if end_time_sec is None else min(float(end_time_sec), duration)
    if effective_start >= effective_end:
        raise ValueError(
            "Invalid or empty segment range: "
            f"start={start_time_sec}, end={end_time_sec}, duration={duration:.6f}"
        )


def validate_prepared_video(
    path: Path,
    *,
    width: int,
    height: int,
    fps: float,
    frame_count: int | None = None,
    frame_count_tolerance: int = 0,
) -> dict[str, object]:
    """Reject missing, empty, unreadable, or structurally changed videos."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Prepared video is missing or empty: {path}")
    info = read_video_info(path)
    errors = []
    if int(info["width"]) != width or int(info["height"]) != height:
        errors.append(f"dimensions {info['width']}x{info['height']} (expected {width}x{height})")
    if abs(float(info["fps"]) - fps) > _FPS_TOLERANCE:
        errors.append(f"FPS {info['fps']} (expected {fps})")
    if frame_count is not None and abs(int(info["frame_count"]) - frame_count) > frame_count_tolerance:
        tolerance_label = f" +/- {frame_count_tolerance}" if frame_count_tolerance else ""
        errors.append(f"frame count {info['frame_count']} (expected {frame_count}{tolerance_label})")
    if errors:
        raise RuntimeError(f"Prepared video validation failed for {path}: " + "; ".join(errors))
    return info


def _temporary_sibling(output_path: Path) -> Path:
    fd, name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=f".tmp{output_path.suffix or '.mp4'}", dir=output_path.parent
    )
    os.close(fd)
    return Path(name)


def _run_ffmpeg_with_progress(
    command: list[str], *, total_frames: int, description: str
) -> subprocess.CompletedProcess[str]:
    """Run FFmpeg and update tqdm from its machine-readable frame counter."""
    progress_command = command[:-1] + ["-progress", "pipe:1", "-nostats", command[-1]]
    process = subprocess.Popen(
        progress_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    assert process.stdout is not None
    current_frame = 0
    with tqdm(total=total_frames, desc=description, unit="frame") as progress:
        for line in process.stdout:
            key, separator, value = line.strip().partition("=")
            if separator and key == "frame":
                try:
                    reported_frame = min(total_frames, int(value))
                except ValueError:
                    continue
                progress.update(max(0, reported_frame - current_frame))
                current_frame = max(current_frame, reported_frame)
        return_code = process.wait()
        if return_code == 0:
            progress.update(max(0, total_frames - current_frame))
    stderr = process.stderr.read() if process.stderr is not None else ""
    return subprocess.CompletedProcess(progress_command, return_code, "", stderr)


def prepare_working_video(
    video_path: Path,
    output_path: Path,
    start_time_sec: int | float | None,
    end_time_sec: int | float | None,
    target_fps: int | float | None = None,
) -> Path:
    """Copy/trim an MP4, transcoding only when its FPS must be reduced."""
    video_path, output_path = Path(video_path), Path(output_path)
    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"Source video must be an MP4 file: {video_path}")
    info = read_video_info(video_path)
    _validate_segment(info, start_time_sec, end_time_sec, target_fps)
    source_fps = float(info["fps"])
    convert_fps = target_fps is not None and source_fps > float(target_fps) + _FPS_TOLERANCE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_sibling(output_path)
    try:
        if start_time_sec is None and end_time_sec is None and not convert_fps:
            shutil.copyfile(video_path, temporary_path)
        else:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("FFmpeg is required when trimming or changing MP4 frame rate")
            command_prefix = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
            if start_time_sec is not None:
                command_prefix += ["-ss", f"{float(start_time_sec):.9f}"]
            command_prefix += ["-i", str(video_path)]
            if end_time_sec is not None:
                duration = float(end_time_sec) - (float(start_time_sec) if start_time_sec is not None else 0.0)
                command_prefix += ["-t", f"{duration:.9f}"]
            if convert_fps:
                keyframe_interval = max(1, int(round(float(target_fps) * 2)))
                common = [
                    "-map", "0:v:0", "-map", "0:a?",
                    "-vf", f"fps={float(target_fps):.9f}",
                    "-g", str(keyframe_interval),
                    "-c:a", "copy",
                ]
                encoders = [
                    ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll", "-cq", "23", "-b:v", "0"]),
                    ("libx264", ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]),
                ]
                errors = []
                source_duration = int(info["frame_count"]) / source_fps
                segment_start = float(start_time_sec) if start_time_sec is not None else 0.0
                segment_end = min(float(end_time_sec), source_duration) if end_time_sec is not None else source_duration
                expected_frames = max(1, int(round((segment_end - segment_start) * float(target_fps))))
                for encoder, encoder_args in encoders:
                    temporary_path.unlink(missing_ok=True)
                    command = command_prefix + common + encoder_args + ["-movflags", "+faststart", str(temporary_path)]
                    result = _run_ffmpeg_with_progress(
                        command,
                        total_frames=expected_frames,
                        description=f"Converting to {float(target_fps):g} FPS ({encoder})",
                    )
                    if result.returncode == 0:
                        break
                    errors.append(f"{encoder}: {result.stderr.strip()}")
                else:
                    raise RuntimeError("FFmpeg FPS conversion failed:\n- " + "\n- ".join(errors))
            else:
                command = command_prefix + ["-map", "0", "-c", "copy", str(temporary_path)]
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                if result.returncode:
                    raise RuntimeError(f"FFmpeg stream-copy trim failed: {result.stderr.strip()}")
        validate_prepared_video(
            temporary_path,
            width=int(info["width"]),
            height=int(info["height"]),
            fps=float(target_fps) if convert_fps else source_fps,
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    mode = ("fps-transcode" if convert_fps else
            "byte-copy" if start_time_sec is None and end_time_sec is None else "stream-copy-trim")
    print(f"Prepared working video with {mode}: {output_path}")
    return output_path
