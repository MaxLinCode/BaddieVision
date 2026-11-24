"""Isolated footage augmentation variants for rally robustness experiments.

Each variant is a normal source: its own source ID, zero-based frame sequence,
artifact root, identity, calibration, and provenance.  The parent footage is
only read.  Downstream tracking, player assignment, pose extraction, and rally
feature generation run on the variant exactly as they do on an original source.

The first transform is a horizontal mirror.  Mirroring the image and swapping
the camera's left/right court labels is equivalent to negating court x while
preserving court y.  Lateral velocity is therefore also negated, which is the
documented invariant for this augmentation.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.court_projection import COURT_CALIBRATION_LINES, COURT_LANDMARKS

from src.single_video.video import read_video_info

from .artifacts import atomic_write_json
from .identity import ParentVideoProvenance, SourceIdentity, sha256_file
from .registry import SourceCatalog, SourceEntry

AUGMENTATION_SCHEMA = "baddievision_footage_augmentation"
AUGMENTATION_VERSION = 1


@dataclass(frozen=True)
class FootageTransform:
    """A reproducible video transform and the calibration rule it needs."""

    name: str
    slug: str
    category: str
    description: str
    calibration: str
    video_filter: str | None = None
    codec: str = "libx264"
    preset: str = "medium"
    crf: int = 16
    pixel_format: str = "yuv420p"

    def encoder_parameters(self) -> dict[str, Any]:
        return {
            "codec": self.codec,
            "preset": self.preset,
            "crf": self.crf,
            "pixel_format": self.pixel_format,
            "video_filter": self.video_filter,
        }


AUGMENTATION_TRANSFORMS: dict[str, FootageTransform] = {
    "horizontal-flip": FootageTransform(
        name="horizontal-flip",
        slug="hflip",
        category="geometry",
        description="Mirror every frame left to right and negate court x.",
        calibration="horizontal_flip",
        video_filter="hflip",
    ),
    "lighting-color": FootageTransform(
        name="lighting-color",
        slug="lighting",
        category="photometric",
        description="Mild fixed exposure, contrast, saturation, and white-balance change.",
        calibration="identity",
        video_filter=(
            "eq=brightness=0.04:contrast=1.08:saturation=1.06:gamma=0.98,"
            "colorbalance=rs=0.03:gs=-0.01:bs=0.02:rm=0.02:gm=-0.01:bm=0.01"
        ),
    ),
    "h264-low": FootageTransform(
        name="h264-low",
        slug="h264-low",
        category="compression",
        description="Lower-quality H.264 re-encode at CRF 32 with libx264.",
        calibration="identity",
        crf=32,
    ),
    "h264-control": FootageTransform(
        name="h264-control",
        slug="h264-control",
        category="compression",
        description="High-quality H.264 re-encode at CRF 16 with the same encoder.",
        calibration="identity",
        crf=16,
    ),
}


def footage_transform(name: str) -> FootageTransform:
    try:
        return AUGMENTATION_TRANSFORMS[name]
    except KeyError as error:
        raise ValueError(f"unknown footage transform: {name}") from error


def swap_left_right(name: str) -> str:
    """Swap whole ``left``/``right`` tokens in a calibration key."""
    parts = []
    for part in name.split("__"):
        tokens = [
            "right" if token == "left" else "left" if token == "right" else token
            for token in part.split("_")
        ]
        parts.append("_".join(tokens))
    return "__".join(parts)


def _mirror_point(point: Any, image_width: int) -> list[float]:
    coordinates = np.asarray(point, dtype=np.float64)
    if coordinates.shape != (2,) or not np.isfinite(coordinates).all():
        raise ValueError("calibration point must be a finite [x, y] pair")
    return [float(image_width - 1 - coordinates[0]), float(coordinates[1])]


def _validate_named_point(name: str) -> None:
    """Accept court landmarks and line-intersection landmark names."""
    parts = name.split("__")
    if len(parts) == 1:
        if name not in COURT_LANDMARKS and name not in COURT_CALIBRATION_LINES:
            raise ValueError(f"landmark {name!r} is not a known court name")
        return
    for part in parts:
        if part not in COURT_CALIBRATION_LINES:
            raise ValueError(f"landmark {name!r} is not a known court name")


def _mirror_named_points(
    points: Mapping[str, Any],
    image_width: int,
    *,
    label: str,
) -> dict[str, list[float]]:
    mirrored: dict[str, list[float]] = {}
    for name, point in points.items():
        _validate_named_point(name)
        swapped = swap_left_right(name)
        if swapped in mirrored:
            raise ValueError(f"{label} mirror produces duplicate name {swapped!r}")
        mirrored[swapped] = _mirror_point(point, image_width)
    return mirrored


def _mirror_lines(
    lines: Mapping[str, Any], image_width: int
) -> dict[str, list[list[float]]]:
    mirrored: dict[str, list[list[float]]] = {}
    for name, endpoints in lines.items():
        if name not in COURT_CALIBRATION_LINES:
            raise ValueError(f"image line {name!r} is not a known court line")
        swapped = swap_left_right(name)
        if swapped in mirrored:
            raise ValueError(f"image line mirror produces duplicate name {swapped!r}")
        values = list(endpoints)
        if len(values) != 2:
            raise ValueError(f"image line {name!r} must have two endpoints")
        mirrored[swapped] = [
            _mirror_point(point, image_width) for point in values
        ]
    return mirrored


def mirror_calibration(value: Mapping[str, Any], image_width: int) -> dict[str, Any]:
    """Return a calibration for horizontally mirrored footage.

    The homography is composed with the image reflection and a court x sign
    flip: ``H' = diag(-1, 1, 1) @ H @ [[-1, 0, W - 1], [0, 1, 0], [0, 0, 1]]``.
    Named image landmarks and lines are mirrored, then their left/right labels
    are swapped so the mirrored pixel still projects to a known court name.
    """
    if not isinstance(image_width, int) or image_width <= 1:
        raise ValueError("image width must be an integer greater than one")
    if value.get("version") != 1:
        raise ValueError(f"unsupported calibration version: {value.get('version')}")
    matrix = np.asarray(value.get("image_to_court"), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("calibration image_to_court must be a finite 3x3 matrix")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("calibration image_to_court is singular")
    image_size = value.get("image_size")
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        if int(image_size[0]) != image_width:
            raise ValueError("mirror width does not match calibration image_size")

    to_mirrored = np.array(
        [[-1.0, 0.0, image_width - 1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    flip_court_x = np.diag([-1.0, 1.0, 1.0])
    mirrored = flip_court_x @ matrix @ to_mirrored
    scale = mirrored[2, 2]
    if abs(scale) < 1e-12:
        scale = np.linalg.norm(mirrored)
    mirrored = mirrored / scale

    result = copy.deepcopy(dict(value))
    result["image_to_court"] = mirrored.tolist()
    if "image_landmarks" in result:
        result["image_landmarks"] = _mirror_named_points(
            result["image_landmarks"],
            image_width,
            label="landmark",
        )
    if "image_lines" in result:
        result["image_lines"] = _mirror_lines(result["image_lines"], image_width)
    result.pop("artifact_metadata", None)
    return result


def identity_calibration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a calibration copy without stale source metadata."""
    result = copy.deepcopy(dict(value))
    result.pop("artifact_metadata", None)
    return result


def _same_geometry(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        int(actual["frame_count"]) == int(expected["frame_count"])
        and int(actual["width"]) == int(expected["width"])
        and int(actual["height"]) == int(expected["height"])
        and math.isclose(
            float(actual["fps"]), float(expected["fps"]), rel_tol=0.0, abs_tol=1e-6
        )
    )


def render_augmented_video(
    parent_video: Path, output: Path, transform: FootageTransform
) -> Path:
    """Render one transform with FFmpeg while preserving frames, size and audio."""
    parent_video = Path(parent_video).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not parent_video.is_file():
        raise FileNotFoundError(f"parent video is missing: {parent_video}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to render augmented footage")
    parent_info = read_video_info(parent_video)
    existing = output.is_file()
    if existing:
        if not _same_geometry(read_video_info(output), parent_info):
            raise ValueError(
                f"existing augmented video has different geometry: {output}"
            )
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix or '.mp4'}")
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(parent_video),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
    ]
    if transform.video_filter:
        command += ["-vf", transform.video_filter]
    command += [
        "-c:v",
        transform.codec,
        "-preset",
        transform.preset,
        "-crf",
        str(transform.crf),
        "-pix_fmt",
        transform.pixel_format,
        "-fps_mode",
        "passthrough",
        "-c:a",
        "copy",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(
                f"FFmpeg augmentation failed for {transform.name}: "
                f"{result.stderr.strip()}"
            )
        if not _same_geometry(read_video_info(temporary), parent_info):
            raise ValueError(
                "augmented video changed frame count, FPS, or dimensions"
            )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _calibration_for(
    parent_calibration: Mapping[str, Any],
    transform: FootageTransform,
    identity: SourceIdentity,
    *,
    parent_source_id: str,
    parent_video_sha256: str,
    parent_calibration_sha256: str,
) -> dict[str, Any]:
    if transform.calibration == "horizontal_flip":
        value = mirror_calibration(parent_calibration, identity.width)
    elif transform.calibration == "identity":
        value = identity_calibration(parent_calibration)
    else:
        raise ValueError(f"unknown calibration rule: {transform.calibration}")
    value["image_size"] = [identity.width, identity.height]
    value["artifact_metadata"] = {
        "schema": "baddievision_court_calibration_artifact",
        "schema_version": 1,
        "source_id": identity.source_id,
        "source_identity_sha256": identity.fingerprint,
        "source_frame": 0,
        "source_frame_status": "derived_from_parent",
        "frame_indexing": "source_local_zero_based",
        "producer": "src.workflow.augmentation",
        "augmentation": {
            "transform": transform.name,
            "calibration_rule": transform.calibration,
            "parent_source_id": parent_source_id,
            "parent_video_sha256": parent_video_sha256,
            "parent_calibration_sha256": parent_calibration_sha256,
        },
    }
    return value


def create_augmented_source(
    catalog: SourceCatalog,
    *,
    parent_source_id: str,
    parent_video: Path,
    parent_calibration: Path,
    transform: str,
    output_root: Path,
    source_id: str | None = None,
) -> SourceEntry:
    """Render and register one isolated augmented source.

    The parent video and calibration are only read.  Re-running with the same
    transform is idempotent; a different video or transform under the same
    variant ID is rejected rather than overwritten.
    """
    if not parent_source_id or "/" in parent_source_id:
        raise ValueError("parent_source_id must be a non-empty source ID")
    spec = footage_transform(transform)
    parent_video = Path(parent_video).expanduser().resolve()
    parent_calibration = Path(parent_calibration).expanduser().resolve()
    if not parent_video.is_file():
        raise FileNotFoundError(f"parent video is missing: {parent_video}")
    if not parent_calibration.is_file():
        raise FileNotFoundError(f"parent calibration is missing: {parent_calibration}")
    parent_video_sha256 = sha256_file(parent_video)
    parent_calibration_sha256 = sha256_file(parent_calibration)

    variant_id = source_id or f"{parent_source_id}__{spec.slug}"
    output_root = Path(output_root).expanduser().resolve()
    root = output_root / variant_id
    provenance_path = root / "augmentation.json"
    if provenance_path.is_file():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (
            previous.get("transform") != spec.name
            or previous.get("parent_source_id") != parent_source_id
            or previous.get("parent_video_sha256") != parent_video_sha256
        ):
            raise ValueError(
                f"artifact root already belongs to a different augmentation: {root}"
            )
    video_path = root / "video.mp4"
    render_augmented_video(parent_video, video_path, spec)

    entry = catalog.add(
        video_path,
        source_id=variant_id,
        output_dir=root,
        parent_video=ParentVideoProvenance(
            video_sha256=parent_video_sha256,
            handling=f"footage-augmentation:{spec.name}",
        ),
    )
    identity = SourceIdentity.read(entry.identity_path)
    parent_value = json.loads(parent_calibration.read_text(encoding="utf-8"))
    calibration = _calibration_for(
        parent_value,
        spec,
        identity,
        parent_source_id=parent_source_id,
        parent_video_sha256=parent_video_sha256,
        parent_calibration_sha256=parent_calibration_sha256,
    )
    atomic_write_json(entry.calibration_path, calibration)

    provenance = {
        "schema": AUGMENTATION_SCHEMA,
        "schema_version": AUGMENTATION_VERSION,
        "source_id": identity.source_id,
        "parent_source_id": parent_source_id,
        "parent_video_sha256": parent_video_sha256,
        "parent_calibration_sha256": parent_calibration_sha256,
        "transform": spec.name,
        "transform_category": spec.category,
        "calibration_rule": spec.calibration,
        "encoder": spec.encoder_parameters(),
        "frame_count": identity.frame_count,
        "fps": identity.fps_numerator / identity.fps_denominator,
        "image_size": [identity.width, identity.height],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(root / "augmentation.json", provenance)
    return entry


def augmentation_provenance(entry: SourceEntry) -> dict[str, Any]:
    path = entry.artifact_root / "augmentation.json"
    if not path.is_file():
        raise FileNotFoundError(f"augmentation provenance is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
