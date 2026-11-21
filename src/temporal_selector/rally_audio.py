"""Source-local audio observations for the offline rally segmenter."""

from __future__ import annotations

import json
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np

# These names and extraction settings form the full_audio feature contract.
RALLY_AUDIO_FEATURE_NAMES = (
    "audio_log_rms",
    "audio_log_peak",
    "audio_log_power_100_500hz",
    "audio_log_power_500_1500hz",
    "audio_log_power_1500_3500hz",
    "audio_log_power_3500_7500hz",
    "audio_positive_spectral_flux",
)
_SAMPLE_RATE = 16_000
_WINDOW_SAMPLES = 768  # 48 ms
_FLUX_LAG_SAMPLES = 256  # 16 ms
_BAND_EDGES = (100, 500, 1500, 3500, 7500)


def _run_media_tool(tool: str, *args: str) -> bytes:
    executable = shutil.which(tool)
    if executable is None:
        raise RuntimeError(f"{tool} is required for rally audio features")
    result = subprocess.run([executable, *args], capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"{tool} failed during rally audio extraction: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def rally_audio_runtime() -> dict[str, str]:
    return {
        tool: _run_media_tool(tool, "-version").decode().splitlines()[0]
        for tool in ("ffmpeg", "ffprobe")
    }


def read_rally_audio(
    video: Path, *, frame_count: int, fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite audio values and a per-feature coverage mask for every frame."""
    values = np.zeros((frame_count, len(RALLY_AUDIO_FEATURE_NAMES)), dtype=np.float32)
    valid = np.zeros_like(values, dtype=bool)
    audio_streams = json.loads(
        _run_media_tool(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=start_time",
            "-of",
            "json",
            str(video),
        )
    )["streams"]
    if not audio_streams:
        return values, valid
    audio_start = float(audio_streams[0].get("start_time", "nan"))
    if not np.isfinite(audio_start):
        raise ValueError("rally audio stream has no usable start timestamp")

    # Packet order may differ from presentation order (e.g. H.264 B frames).
    # Using actual presentation timestamps also keeps VFR recordings aligned.
    packets = json.loads(
        _run_media_tool(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "json",
            str(video),
        )
    )["packets"]
    times = np.sort([float(packet.get("pts_time", "nan")) for packet in packets])
    if (
        len(times) != frame_count
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        # Some codecs/container layouts do not store one frame per packet.
        frames = json.loads(
            _run_media_tool(
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(video),
            )
        )["frames"]
        times = np.asarray(
            [float(frame.get("best_effort_timestamp_time", "nan")) for frame in frames]
        )
        if (
            len(times) != frame_count
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(
                "rally audio requires a presentation timestamp for each source frame"
            )
    ends = np.append(times[1:], times[-1] + 1 / fps)
    first_sample = round(audio_start * _SAMPLE_RATE)
    centers = (
        np.rint((times + ends) * (0.5 * _SAMPLE_RATE)).astype(np.int64) - first_sample
    )

    pcm = _run_media_tool(
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-copyts",
        "-i",
        str(video),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-af",
        # Rebase in seconds before resampling because the filter input
        # time base can differ from the output sample rate.
        f"asetpts=PTS-({audio_start})/TB,aresample={_SAMPLE_RATE}:async=1:first_pts=0",
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    )
    samples = np.frombuffer(pcm, dtype="<f4")
    if not np.isfinite(samples).all():
        raise ValueError("rally audio waveform contains non-finite samples")
    starts = centers - _WINDOW_SAMPLES // 2
    covered = np.flatnonzero((starts >= 0) & (starts + _WINDOW_SAMPLES <= len(samples)))
    offsets = np.arange(_WINDOW_SAMPLES)
    taper = np.hanning(_WINDOW_SAMPLES)
    freqs = np.fft.rfftfreq(_WINDOW_SAMPLES, 1 / _SAMPLE_RATE)
    bands = [(freqs >= low) & (freqs < high) for low, high in pairwise(_BAND_EDGES)]

    # Bound the temporary spectral matrices even on long camera segments.
    for begin in range(0, len(covered), 1024):
        indices = covered[begin : begin + 1024]
        sample_indices = starts[indices, None] + offsets
        windows = samples[sample_indices]
        spectrum = np.abs(np.fft.rfft(windows * taper)) / taper.sum()
        values[indices, 0] = np.log1p(100 * np.sqrt(np.mean(windows**2, axis=1)))
        values[indices, 1] = np.log1p(100 * np.max(np.abs(windows), axis=1))
        for column, band in enumerate(bands, start=2):
            values[indices, column] = np.log1p(
                1000 * np.sum(spectrum[:, band] ** 2, axis=1)
            )
        valid[indices, :-1] = True

        has_history = starts[indices] >= _FLUX_LAG_SAMPLES
        previous_windows = samples[sample_indices[has_history] - _FLUX_LAG_SAMPLES]
        previous = np.abs(np.fft.rfft(previous_windows * taper)) / taper.sum()
        rise = np.maximum(spectrum[has_history] - previous, 0)
        values[indices[has_history], -1] = np.log1p(10 * rise.sum(axis=1))
        valid[indices[has_history], -1] = True
    return values, valid
