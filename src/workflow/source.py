"""Source registration, verification, calibration, and device selection."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

from .artifacts import (
    calibration_fingerprint,
    import_calibration,
    register_native_calibration,
)
from .identity import ParentVideoProvenance, SourceIdentity, inspect_source, sha256_file
from .publication import recovered_publication
from .registry import SourceCatalog, SourceEntry


def verify_source(source: SourceEntry) -> SourceIdentity:
    if not source.enabled:
        raise ValueError(f"source is disabled: {source.source_id}")
    identity = SourceIdentity.read(source.identity_path)
    if inspect_source(source.source_id, source.video_path) != replace(
        identity, parent_video=None
    ):
        raise ValueError("source video identity changed; register it as a new source")
    return identity


def resolve_source(
    catalog: SourceCatalog,
    *,
    source_id: str | None = None,
    video: Path | None = None,
    output_dir: Path | None = None,
    parent_video: ParentVideoProvenance | None = None,
) -> SourceEntry:
    if video is None:
        if source_id is None:
            raise ValueError("provide --video FILE or --source ID")
        source = catalog.select([source_id])[0]
        if output_dir is not None and output_dir.resolve() != source.artifact_root:
            raise ValueError(
                "--output-dir differs from the registered source artifact root"
            )
    else:
        video = Path(video).expanduser().resolve()
        if output_dir is None:
            raise ValueError("--output-dir is required with --video")
        output_dir = Path(output_dir).expanduser().resolve()
        previous = output_dir / "source.json"
        if previous.exists() and replace(
            SourceIdentity.read(previous), parent_video=None
        ) != inspect_source(source_id or video.stem, video):
            raise ValueError(
                "output directory belongs to a different source; choose a new --output-dir"
            )
        source = catalog.add(
            video, source_id=source_id, output_dir=output_dir, parent_video=parent_video
        )
        if source.artifact_root != output_dir:
            raise ValueError(
                "video is already registered with a different output directory"
            )
    verify_source(source)
    return source


def resolve_rally_30fps_source(
    catalog: SourceCatalog,
    *,
    source_id: str | None,
    video: Path | None,
    output_dir: Path | None,
) -> SourceEntry:
    """Register converted footage as a distinct source before deriving artifacts."""
    from src.single_video.video import has_constant_frame_rate, prepare_working_video, read_video_info
    from src.temporal_selector.rally_dataset import RALLY_INPUT_FPS, validate_rally_source_fps

    if video is not None:
        video = video.expanduser().resolve()
        fps = float(read_video_info(video)["fps"])
        if abs(fps - RALLY_INPUT_FPS) > 1e-6 or not has_constant_frame_rate(video, RALLY_INPUT_FPS):
            source_id = source_id or f"{video.stem}-30fps"
            if output_dir is None:
                raise ValueError("--output-dir is required to prepare a 30 fps source")
            root = output_dir.expanduser().resolve()
            digest = sha256_file(video)
            converted = root / "input" / f"{digest}-30fps.mp4"
            previous = root / "source.json"
            with recovered_publication(previous):
                if previous.is_file():
                    identity = SourceIdentity.read(previous)
                    if not converted.is_file() or identity.video_sha256 != sha256_file(converted):
                        raise ValueError(
                            "output directory belongs to a different source; choose a new --output-dir"
                        )
                else:
                    prepare_working_video(video, converted, None, None, RALLY_INPUT_FPS, exact_fps=True)
                return resolve_source(
                    catalog, source_id=source_id, video=converted, output_dir=root,
                    parent_video=ParentVideoProvenance(video_sha256=digest, handling="fps-transcode"),
                )
    source = resolve_source(catalog, source_id=source_id, video=video, output_dir=output_dir)
    identity = SourceIdentity.read(source.identity_path)
    validate_rally_source_fps(source.source_id, identity.fps_numerator / identity.fps_denominator)
    if video is None and not has_constant_frame_rate(source.video_path, RALLY_INPUT_FPS):
        raise ValueError(f"{source.source_id}: register a new constant 30 fps source")
    return source


def select_device(requested: str) -> str:
    import torch

    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if requested == "auto":
        return next(device for device in ("cuda", "mps", "cpu") if available[device])
    if requested not in available or not available[requested]:
        raise ValueError(
            f"requested device is unavailable: {requested}; use --device cpu or auto"
        )
    return requested


def calibrate_source(
    source: SourceEntry, external: Path | None, *, force: bool
) -> None:
    from src.court_projection import CourtHomography

    if external is not None:
        external = external.expanduser().resolve()
        if not external.is_file():
            raise FileNotFoundError(f"calibration file is missing: {external}")
        CourtHomography.load(external)
        if external != source.calibration_path.resolve():
            if source.calibration_path.is_file():
                existing = json.loads(
                    source.calibration_path.read_text(encoding="utf-8")
                )
                imported = existing.get("artifact_metadata", {}).get(
                    "imported_file_sha256"
                )
                if imported != sha256_file(external):
                    if not force:
                        raise ValueError("calibration changed; rerun with --force")
                    import_calibration(source, external, force=True)
            else:
                import_calibration(source, external)
    if not source.calibration_path.is_file():
        print(
            "Court calibration is missing. Opening the browser calibration UI. "
            "Save a calibration to continue, or cancel and pass --calibration FILE.",
            file=sys.stderr,
            flush=True,
        )
        try:
            from src.calibrate_court import main as calibrate_main

            frame = calibrate_main(
                [str(source.video_path), str(source.calibration_path), "--open-browser"]
            )
            if not source.calibration_path.is_file():
                raise ValueError("no calibration was saved")
            register_native_calibration(source, source_frame=frame)
        except (KeyboardInterrupt, OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                "calibration was not completed; rerun with --calibration FILE "
                "or complete the browser UI"
            ) from exc
    CourtHomography.load(source.calibration_path)
    calibration_fingerprint(source)
