"""Source-local audio observations for the offline rally segmenter."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
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
# Allow small numerical error in the resampled coverage channel.
_MIN_SAMPLE_COVERAGE = 0.999


@dataclass(frozen=True)
class RallyAudio:
    values: np.ndarray
    validity: np.ndarray
    timestamps: np.ndarray

    def zscore(self, *, start_frame: int = 0) -> np.ndarray:
        """Normalize observed channels over up to ten seconds of available history."""
        values = self.values[start_frame:]
        valid = self.validity[start_frame:]
        times = self.timestamps[start_frame:]
        start = np.searchsorted(times, times - 10.0, side="right")
        observed = np.where(valid, values, 0).astype(np.float64)

        def rolling_sum(series: np.ndarray) -> np.ndarray:
            prefix = np.vstack((np.zeros((1, series.shape[1])), np.cumsum(series, axis=0)))
            return prefix[1:] - prefix[start]

        count = rolling_sum(valid)
        divisor = np.maximum(count, 1)
        mean = rolling_sum(observed) / divisor
        std = np.sqrt(np.maximum(rolling_sum(observed**2) / divisor - mean**2, 0))
        result = np.zeros_like(values)
        np.divide(observed - mean, std, out=result, where=valid & (count > 1) & (std > 1e-6))
        return result


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


def video_frame_timestamps(video: Path, frame_count: int) -> np.ndarray:
    """Read one presentation timestamp per decoded frame, in presentation order."""
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
                "rally processing requires a presentation timestamp for each source frame"
            )
    return times


def read_rally_audio(
    video: Path, *, frame_count: int, fps: float
) -> RallyAudio:
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
            "stream=start_time,time_base",
            "-of",
            "json",
            str(video),
        )
    )["streams"]
    if not audio_streams:
        return RallyAudio(values, valid, np.arange(frame_count) / fps)
    audio_start = float(audio_streams[0].get("start_time", "nan"))
    if not np.isfinite(audio_start):
        raise ValueError("rally audio stream has no usable start timestamp")
    # Allow timestamp rounding within one container tick and one output sample.
    timestamp_tolerance = float(Fraction(audio_streams[0]["time_base"])) + 1 / _SAMPLE_RATE

    times = video_frame_timestamps(video, frame_count)
    ends = np.append(times[1:], times[-1] + 1 / fps)
    timestamps = (times + ends) / 2
    first_sample = round(audio_start * _SAMPLE_RATE)
    centers = np.rint(timestamps * _SAMPLE_RATE).astype(np.int64) - first_sample

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
        "-af",
        # Put a constant coverage channel alongside the decoded mono audio.
        # Timestamp compensation inserts zeros in both channels at a gap.
        f"asetpts=PTS-({audio_start})/TB,"
        "aformat=channel_layouts=mono,aeval=val(0)|1:c=stereo,"
        f"aresample={_SAMPLE_RATE}:async=1:first_pts=0:"
        f"min_comp={timestamp_tolerance}:min_hard_comp={timestamp_tolerance}",
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    )
    samples, coverage = np.frombuffer(pcm, dtype="<f4").reshape(-1, 2).T
    if not np.isfinite(samples).all():
        raise ValueError("rally audio waveform contains non-finite samples")
    starts = centers - _WINDOW_SAMPLES // 2
    candidates = np.flatnonzero((starts >= 0) & (starts + _WINDOW_SAMPLES <= len(samples)))
    offsets = np.arange(_WINDOW_SAMPLES)
    taper = np.hanning(_WINDOW_SAMPLES)
    freqs = np.fft.rfftfreq(_WINDOW_SAMPLES, 1 / _SAMPLE_RATE)
    bands = [(freqs >= low) & (freqs < high) for low, high in pairwise(_BAND_EDGES)]

    # Bound the temporary spectral matrices even on long camera segments.
    for begin in range(0, len(candidates), 1024):
        indices = candidates[begin : begin + 1024]
        sample_indices = starts[indices, None] + offsets
        observed = np.all(coverage[sample_indices] >= _MIN_SAMPLE_COVERAGE, axis=1)
        indices = indices[observed]
        sample_indices = sample_indices[observed]
        windows = samples[sample_indices]
        spectrum = np.abs(np.fft.rfft(windows * taper)) / taper.sum()
        values[indices, 0] = np.log1p(100 * np.sqrt(np.mean(windows**2, axis=1)))
        values[indices, 1] = np.log1p(100 * np.max(np.abs(windows), axis=1))
        for column, band in enumerate(bands, start=2):
            values[indices, column] = np.log1p(
                1000 * np.sum(spectrum[:, band] ** 2, axis=1)
            )
        valid[indices, :-1] = True

        history_rows = np.flatnonzero(starts[indices] >= _FLUX_LAG_SAMPLES)
        previous_indices = sample_indices[history_rows] - _FLUX_LAG_SAMPLES
        observed_history = np.all(
            coverage[previous_indices] >= _MIN_SAMPLE_COVERAGE, axis=1
        )
        history_rows = history_rows[observed_history]
        previous_windows = samples[previous_indices[observed_history]]
        previous = np.abs(np.fft.rfft(previous_windows * taper)) / taper.sum()
        rise = np.maximum(spectrum[history_rows] - previous, 0)
        values[indices[history_rows], -1] = np.log1p(10 * rise.sum(axis=1))
        valid[indices[history_rows], -1] = True
    return RallyAudio(values, valid, timestamps)
