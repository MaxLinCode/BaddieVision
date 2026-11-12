"""Prepare only the approved short raw-video sample, without private metadata."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.court_projection import CourtHomography
from src.single_video.video import read_video_info
from src.workflow.artifacts import atomic_write_json
from src.workflow.identity import inspect_source, sha256_file


def prepare_sample(
    video: Path,
    calibration: Path,
    output_dir: Path,
    *,
    center_frame: int = 2605,
    duration_seconds: float = 10.0,
    private_provenance: Path | None = None,
) -> Path:
    """Export a frame-exact silent crop and rebased public identity.

    The caller must supply the approved source video, not a diagnostic montage.
    Private origin/frame details are stored outside the public sample directory.
    """
    video, calibration, output = (
        Path(video).resolve(),
        Path(calibration).resolve(),
        Path(output_dir).resolve(),
    )
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to export the sample")
    info = read_video_info(video)
    fps, count = float(info["fps"]), int(info["frame_count"])
    if not math.isfinite(fps) or fps <= 0 or count <= 0:
        raise ValueError("sample source requires a positive frame rate and frame count")
    if type(center_frame) is not int or not 0 <= center_frame < count:
        raise ValueError("sample center frame is outside the source-local video")
    if not math.isfinite(duration_seconds) or not 0 < duration_seconds <= 10:
        raise ValueError(
            "approved public sample duration must be positive and at most ten seconds"
        )
    raw_calibration = json.loads(calibration.read_text())
    dimensions = [int(info["width"]), int(info["height"])]
    if raw_calibration.get("image_size") != dimensions:
        raise ValueError("sample calibration image_size differs from the source video")
    homography = CourtHomography.load(calibration)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("sample output directory must be empty")
    private = (
        Path(private_provenance).resolve()
        if private_provenance
        else (output.parent / f".{output.name}.private-provenance.json")
    )
    if private.is_relative_to(output):
        raise ValueError(
            "private provenance must be outside the public sample directory"
        )
    if private.exists():
        raise FileExistsError(
            "private provenance already exists; choose a new output location"
        )
    size = min(count, max(1, round(duration_seconds * fps)))
    start = min(max(0, center_frame - size // 2), count - size)
    end = start + size
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".sample-", dir=output.parent))
    try:
        sample = stage / "sample.mp4"
        pixel_format = (
            "yuv420p" if all(number % 2 == 0 for number in dimensions) else "yuv444p"
        )
        result = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-vf",
                f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS",
                "-fps_mode",
                "passthrough",
                "-an",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                pixel_format,
                "-movflags",
                "+faststart",
                str(sample),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"sample export failed: {result.stderr.strip()}")
        identity = inspect_source("rally-demo", sample)
        if (
            identity.frame_count != size
            or [identity.width, identity.height] != dimensions
            or not math.isclose(
                identity.fps_numerator / identity.fps_denominator, fps, rel_tol=1e-5
            )
        ):
            raise ValueError("sample export changed frame count, geometry, or FPS")
        identity.write(stage / "sample-source.json")
        homography.save(stage / "sample-calibration.json", image_size=tuple(dimensions))
        atomic_write_json(
            private,
            {
                "schema": "baddievision.private_sample_provenance",
                "schema_version": 1,
                "private_source_video": str(video),
                "source_video_sha256": sha256_file(video),
                "source_calibration": str(calibration),
                "source_calibration_sha256": sha256_file(calibration),
                "source_start_frame": start,
                "source_end_frame_inclusive": end - 1,
                "source_center_frame": center_frame,
                "sample_sha256": identity.video_sha256,
                "sample_frame_count": size,
            },
        )
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--center-frame", type=int, default=2605)
    parser.add_argument("--duration-seconds", type=float, default=10)
    parser.add_argument("--private-provenance", type=Path)
    args = parser.parse_args()
    print(
        prepare_sample(
            args.video,
            args.calibration,
            args.output_dir,
            center_frame=args.center_frame,
            duration_seconds=args.duration_seconds,
            private_provenance=args.private_provenance,
        )
    )


if __name__ == "__main__":
    main()
