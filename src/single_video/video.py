"""Video preparation helpers shared by the single-video notebooks."""

from __future__ import annotations

import math
import json
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Sequence

import cv2


_FPS_TOLERANCE = 0.05
_PROGRESS_WIDTH = 28
_MASTER_FPS = 30.0
_MASTER_COLOR = {"color_range": "tv", "color_space": "bt709", "color_transfer": "bt709", "color_primaries": "bt709"}


def probe_video(path: Path, ffprobe_path: str | None = None) -> dict[str, object]:
    """Return the first video stream's structural and colour metadata."""
    executable = ffprobe_path or shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("ffprobe is required for color-managed video ingest")
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


def classify_source(probe: dict[str, object], color_mode: str = "auto") -> str:
    """Classify a source, rejecting metadata that cannot be interpreted safely."""
    if color_mode not in {"auto", "sdr", "hdr-hlg", "hdr-pq"}:
        raise ValueError("color_mode must be one of: auto, sdr, hdr-hlg, hdr-pq")
    if color_mode != "auto":
        return {"sdr": "sdr", "hdr-hlg": "hdr-hlg", "hdr-pq": "hdr-pq"}[color_mode]
    transfer = str(probe.get("color_transfer") or "unknown")
    primaries = str(probe.get("color_primaries") or "unknown")
    pix_fmt = str(probe.get("pix_fmt") or "")
    if probe.get("dolby_vision"):
        return "dolby-vision"
    if transfer == "arib-std-b67" and primaries == "bt2020":
        return "hdr-hlg"
    if transfer == "smpte2084" and primaries == "bt2020":
        return "hdr-pq"
    if transfer in {"bt709", "iec61966-2-1"} and primaries == "bt709":
        return "sdr" if "10" not in pix_fmt and "12" not in pix_fmt else "sdr-10bit"
    if primaries not in {"bt709", "unknown"} and transfer not in {"arib-std-b67", "smpte2084"}:
        return "wide-gamut-sdr"
    raise ValueError(
        "Video color metadata is missing or ambiguous. Use COLOR_MODE='sdr', "
        "'hdr-hlg', or 'hdr-pq' only after identifying the original source."
    )


def _is_analysis_master(probe: dict[str, object], info: dict[str, object]) -> bool:
    return (
        probe.get("codec_name") == "h264" and probe.get("pix_fmt") == "yuv420p"
        and all(probe.get(key) == value for key, value in _MASTER_COLOR.items())
        and abs(float(info["fps"]) - _MASTER_FPS) <= _FPS_TOLERANCE
    )


def _color_filter(source_mode: str, probe: dict[str, object]) -> str:
    if source_mode in {"hdr-hlg", "hdr-pq", "dolby-vision"}:
        transfer = "arib-std-b67" if source_mode in {"hdr-hlg", "dolby-vision"} else "smpte2084"
        return (
            f"zscale=rangein=limited:pin=bt2020:tin={transfer}:min=bt2020nc:range=limited,"
            "zscale=transfer=linear:npl=100,format=gbrpf32le,"
            "zscale=primaries=bt709,tonemap=tonemap=mobius:param=0.3:desat=0,"
            "zscale=transfer=bt709:matrix=bt709:range=limited:dither=error_diffusion,format=yuv420p"
        )
    pin = str(probe.get("color_primaries") or "bt709")
    tin = str(probe.get("color_transfer") or "bt709")
    min_ = str(probe.get("color_space") or "bt709")
    rin = "full" if probe.get("color_range") == "pc" else "limited"
    return f"zscale=rangein={rin}:pin={pin}:tin={tin}:min={min_}:p=bt709:t=bt709:m=bt709:range=limited:dither=error_diffusion,format=yuv420p"


def _print_progress(label: str, completed: int, total: int, *, final: bool = False) -> None:
    """Render a dependency-free terminal progress bar for video preparation."""
    total = max(1, total)
    completed = min(max(0, completed), total)
    filled = round(_PROGRESS_WIDTH * completed / total)
    bar = "#" * filled + "-" * (_PROGRESS_WIDTH - filled)
    print(f"\r{label}: [{bar}] {completed}/{total} frames", end="\n" if final else "", flush=True)


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


def _segment_parameters(
    info: dict[str, object],
    start_time_sec: int | float | None,
    end_time_sec: int | float | None,
    target_fps: int | float | None,
) -> tuple[float, int, int, int]:
    source_fps = float(info["fps"])
    frame_count = int(info["frame_count"])
    if source_fps <= 0 or not math.isfinite(source_fps):
        raise ValueError(f"Video reports invalid FPS: {source_fps}")
    if frame_count <= 0:
        raise ValueError(f"Video reports invalid frame count: {frame_count}")
    if start_time_sec is not None and start_time_sec < 0:
        raise ValueError(f"start_time_sec must be non-negative, got {start_time_sec}")
    if end_time_sec is not None and end_time_sec < 0:
        raise ValueError(f"end_time_sec must be non-negative, got {end_time_sec}")
    if target_fps is not None and (target_fps <= 0 or not math.isfinite(float(target_fps))):
        raise ValueError(f"target_fps must be positive, got {target_fps}")

    effective_fps = min(float(target_fps), source_fps) if target_fps is not None else source_fps
    start_frame = 0 if start_time_sec is None else max(0, int(round(start_time_sec * source_fps)))
    end_frame = frame_count if end_time_sec is None else min(frame_count, int(round(end_time_sec * source_fps)))
    if start_frame >= end_frame:
        raise ValueError(
            "Invalid segment range: "
            f"start={start_time_sec}, end={end_time_sec}, fps={source_fps}, frame_count={frame_count}"
        )
    expected_frames = max(1, int(math.floor((end_frame - start_frame) * effective_fps / source_fps + 0.5)))
    return effective_fps, start_frame, end_frame, expected_frames


def build_ffmpeg_command(
    ffmpeg_path: str,
    video_path: Path,
    output_path: Path,
    *,
    source_fps: float,
    effective_fps: float,
    start_frame: int,
    end_frame: int,
    expected_frames: int,
    gpu: bool,
    filter_graph: str | None = None,
) -> list[str]:
    """Build a deterministic GPU or CPU FFmpeg transcode command."""
    start_seconds = start_frame / source_fps
    duration_seconds = (end_frame - start_frame) / source_fps
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-progress",
        "pipe:1",
        "-nostats",
        "-y",
    ]
    if start_frame:
        command += ["-ss", f"{start_seconds:.9f}"]
    if gpu:
        command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    command += ["-i", str(video_path), "-t", f"{duration_seconds:.9f}", "-an"]
    if gpu:
        # H.264 NVENC Main does not accept 10-bit HEVC decode surfaces. Convert
        # on the GPU so yuv420p10le sources do not fail with an unsupported
        # NVENC-features error or incur a GPU-to-CPU download.
        command += [
            "-vf", "scale_cuda=format=nv12",
            "-c:v", "h264_nvenc", "-preset", "p1", "-rc", "vbr", "-cq", "20", "-b:v", "0",
        ]
    else:
        if filter_graph:
            command += ["-vf", filter_graph]
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    command += [
        "-r",
        f"{effective_fps:.9f}",
        "-fps_mode",
        "cfr",
        "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
        "-color_trc", "bt709", "-color_primaries", "bt709", "-movflags", "+faststart",
        "-frames:v",
        str(expected_frames),
        str(output_path),
    ]
    return command


def validate_prepared_video(
    path: Path,
    *,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    frame_count_tolerance: int = 0,
) -> dict[str, object]:
    """Reject missing, empty, or structurally incorrect prepared videos."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Prepared video is missing or empty: {path}")
    info = read_video_info(path)
    errors = []
    if int(info["width"]) != width or int(info["height"]) != height:
        errors.append(f"dimensions {info['width']}x{info['height']} (expected {width}x{height})")
    if abs(float(info["fps"]) - fps) > _FPS_TOLERANCE:
        errors.append(f"FPS {info['fps']} (expected {fps})")
    actual_frame_count = int(info["frame_count"])
    if abs(actual_frame_count - frame_count) > frame_count_tolerance:
        tolerance_label = f" +/- {frame_count_tolerance}" if frame_count_tolerance else ""
        errors.append(
            f"frame count {actual_frame_count} (expected {frame_count}{tolerance_label})"
        )
    if errors:
        raise RuntimeError(f"Prepared video validation failed for {path}: " + "; ".join(errors))
    return info


def _run_ffmpeg(command: Sequence[str]) -> None:
    """Run FFmpeg while displaying its machine-readable frame progress."""
    frame_count = int(command[command.index("-frames:v") + 1])
    backend = "FFmpeg NVENC" if "h264_nvenc" in command else "FFmpeg CPU"
    _print_progress(backend, 0, frame_count)
    process = subprocess.Popen(
        command,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    output: list[str] = []
    for line in process.stdout:
        output.append(line)
        key, separator, value = line.partition("=")
        if separator and key == "frame":
            try:
                _print_progress(backend, int(value), frame_count)
            except ValueError:
                pass
    return_code = process.wait()
    _print_progress(backend, frame_count if return_code == 0 else 0, frame_count, final=True)
    if return_code:
        detail = "".join(output).strip() or f"exit status {return_code}"
        raise RuntimeError(detail)


def _prepare_with_opencv(
    video_path: Path,
    output_path: Path,
    *,
    source_fps: float,
    effective_fps: float,
    start_frame: int,
    end_frame: int,
    width: int,
    height: int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for OpenCV preprocessing: {video_path}")
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), effective_fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        source_frame_interval = 1.0 / source_fps
        output_frame_interval = 1.0 / effective_fps
        next_output_time = 0.0
        processed = 0
        total_frames = end_frame - start_frame
        chunk_size = max(1, min(250, total_frames // 100 or 1))
        _print_progress("OpenCV preparation", 0, total_frames)
        for frame_idx in range(start_frame, end_frame):
            if not cap.grab():
                raise RuntimeError(f"OpenCV stopped at source frame {frame_idx} before frame {end_frame}")
            current_time = (frame_idx - start_frame) * source_frame_interval
            if current_time + source_frame_interval * 0.5 >= next_output_time:
                ok, frame = cap.retrieve()
                if not ok:
                    raise RuntimeError(f"OpenCV failed to decode selected source frame {frame_idx}")
                writer.write(frame)
                next_output_time += output_frame_interval
            processed += 1
            if processed % chunk_size == 0:
                _print_progress("OpenCV preparation", processed, total_frames)
        _print_progress("OpenCV preparation", processed, total_frames, final=True)
    finally:
        cap.release()
        writer.release()


def _temporary_sibling(output_path: Path) -> Path:
    fd, name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=f".tmp{output_path.suffix or '.mp4'}", dir=output_path.parent
    )
    os.close(fd)
    path = Path(name)
    path.unlink()
    return path


def prepare_working_video(
    video_path: Path,
    output_path: Path,
    start_time_sec: int | None,
    end_time_sec: int | None,
    target_fps: int | None,
    color_mode: str = "auto",
) -> Path:
    """Create the shared 8-bit H.264/30 fps/limited-range Rec.709 master."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    info = read_video_info(video_path)
    requested_fps = _MASTER_FPS if target_fps is None else min(float(target_fps), _MASTER_FPS)
    effective_fps, start_frame, end_frame, expected_frames = _segment_parameters(
        info, start_time_sec, end_time_sec, requested_fps
    )
    probe = probe_video(video_path)
    source_mode = classify_source(probe, color_mode)
    source_fps = float(info["fps"])
    frame_count = int(info["frame_count"])
    if start_frame == 0 and end_frame == frame_count and _is_analysis_master(probe, info):
        return video_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(info["width"]), int(info["height"])
    # Container frame counts are estimates for some variable-frame-rate and
    # damaged-tail sources. A sub-percent discrepancy is acceptable after a
    # complete transcode; larger differences still indicate truncation.
    frame_count_tolerance = max(2, math.ceil(expected_frames * 0.005))
    ffmpeg_path = shutil.which("ffmpeg")
    errors: list[str] = []
    color_managed = source_mode != "sdr" or not _is_analysis_master(probe, info)
    backends = ["ffmpeg-cpu"] if ffmpeg_path else []
    if not ffmpeg_path:
        reason = "FFmpeg executable was not found"
        errors.append(f"ffmpeg: {reason}")
        if color_managed:
            raise RuntimeError(f"{reason}; color-managed {source_mode} input cannot use OpenCV fallback")
        warnings.warn(f"{reason}; falling back to OpenCV", RuntimeWarning, stacklevel=2)
    if not color_managed:
        backends.append("opencv")

    for backend in backends:
        temporary_path = _temporary_sibling(output_path)
        try:
            if backend.startswith("ffmpeg"):
                command = build_ffmpeg_command(
                    ffmpeg_path or "ffmpeg",
                    video_path,
                    temporary_path,
                    source_fps=source_fps,
                    effective_fps=effective_fps,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    expected_frames=expected_frames,
                    gpu=backend == "ffmpeg-nvenc",
                    filter_graph=_color_filter(source_mode, probe),
                )
                _run_ffmpeg(command)
            else:
                _prepare_with_opencv(
                    video_path,
                    temporary_path,
                    source_fps=source_fps,
                    effective_fps=effective_fps,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    width=width,
                    height=height,
                )
            validate_prepared_video(
                temporary_path,
                width=width,
                height=height,
                fps=effective_fps,
                frame_count=expected_frames,
                frame_count_tolerance=frame_count_tolerance,
            )
            temporary_path.replace(output_path)
            print(f"Prepared working video with {backend}: {output_path}")
            return output_path
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
            if backend != "opencv":
                warnings.warn(f"{backend} failed ({exc}); trying the next backend", RuntimeWarning, stacklevel=2)
        finally:
            temporary_path.unlink(missing_ok=True)

    raise RuntimeError("Every video preparation backend failed:\n- " + "\n- ".join(errors))
