"""One extraction workflow for local, Colab, and Kaggle callers."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from .artifacts import atomic_write_json
from .identity import SourceIdentity, sha256_file
from .registry import SourceCatalog, SourceEntry
from .source import resolve_source as resolve_source, calibrate_source, select_device
from .stages import (
    backend_identity,
    implementation_identity,
    run_person_stage,
    run_player_stage,
    run_stage,
    stage_status,
    verify_source,
)


def _tracknet_adapter(
    source: SourceEntry,
    tracks: Path,
    candidates: Path,
    *,
    model_root: Path,
    batch_size: int,
    num_workers: int,
    device: str,
) -> None:
    """Isolate TrackNet's historical absolute imports in a subprocess."""
    tracknet_root = Path(__file__).resolve().parents[1] / "TrackNetV3"
    config = {
        "video_file": str(source.video_path),
        "save_dir": str(tracks.parent),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "candidate_output_path": str(candidates),
        "large_video": True,
        "output_video": False,
        "optimized_inference": True,
    }
    code = """import json, sys, torch
from TrackNetV3 import predictArgs
from src.single_video.models import load_track_models
root, device, config = sys.argv[1:]
predictArgs.DEVICE = torch.device(device)
models = load_track_models(__import__('pathlib').Path(root))
predictArgs.run_prediction(reuse_models=models, **json.loads(config))
"""
    environment = dict(os.environ)
    # Put TrackNet first only in the isolated process, never in the caller kernel.
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(tracknet_root),
            str(tracknet_root.parent),
            str(tracknet_root.parent.parent),
        ]
    )
    subprocess.run(
        [sys.executable, "-c", code, str(model_root), device, json.dumps(config)],
        cwd=tracknet_root,
        env=environment,
        check=True,
    )
    generated = tracks.parent / f"{source.video_path.name[:-4].lower()}_ball.csv"
    generated.replace(tracks)
    # The vendored candidate contract predates source registration. Enrich its
    # header at this adapter boundary, preserving every candidate unchanged.
    enriched = candidates.with_suffix(candidates.suffix + ".lineage")
    identity = SourceIdentity.read(source.identity_path)
    with (
        candidates.open(encoding="utf-8") as incoming,
        enriched.open("w", encoding="utf-8") as outgoing,
    ):
        header = json.loads(next(incoming))
        header.update(
            source_video_sha256=identity.video_sha256,
            source_id=source.source_id,
            source_identity_sha256=identity.fingerprint,
        )
        outgoing.write(json.dumps(header) + "\n")
        shutil.copyfileobj(incoming, outgoing)
    enriched.replace(candidates)


def run_shuttle_stage(
    source: SourceEntry,
    *,
    model_root: Path,
    device: str,
    batch_size: int = 16,
    num_workers: int = 0,
    force: bool = False,
    extractor: Callable[..., Any] | None = None,
    extractor_identity: str | None = None,
) -> bool:
    identity = verify_source(source)
    model_root = Path(model_root).expanduser().resolve()
    model_paths = [
        model_root / name
        for name in (
            "models/TrackNet_torchscript.pt",
            "src/TrackNetV3/ckpts/TrackNet_best.pt",
        )
    ]
    inputs = {
        "identity": identity.fingerprint,
        "models": {
            str(path.relative_to(model_root)): sha256_file(path) for path in model_paths
        },
        "implementation": implementation_identity("shuttle"),
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "backend": backend_identity(extractor, extractor_identity),
    }
    root = source.artifact_root / "shuttle"
    outputs = [root / "tracks.csv", source.candidates_path]

    def execute(paths):
        if extractor is None:
            _tracknet_adapter(
                source,
                paths[0],
                paths[1],
                model_root=model_root,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
            )
        else:
            extractor(source.video_path, paths[0], paths[1])

    def validate(paths):
        from src.single_video.shuttle import read_shuttle_candidates

        metadata, rows = read_shuttle_candidates(paths[1])
        if (
            metadata.get("source_frame_range") != [0, identity.frame_count - 1]
            or metadata.get("image_size") != [identity.width, identity.height]
            or metadata.get("source_video_sha256") != identity.video_sha256
            or not math.isclose(
                float(metadata.get("fps", 0)),
                identity.fps_numerator / identity.fps_denominator,
                abs_tol=1e-6,
            )
            or metadata.get("checkpoint_sha256") != sha256_file(model_paths[1])
            or metadata.get("inference_model_sha256") != sha256_file(model_paths[0])
        ):
            raise ValueError(
                "shuttle observations do not match source coverage or lineage"
            )
        if [int(row["frame"]) for row in rows] != list(range(identity.frame_count)):
            raise ValueError("shuttle observation frame is outside source")
        with paths[0].open(newline="") as handle:
            frames = [int(row["Frame"]) for row in csv.DictReader(handle)]
        if frames != list(range(identity.frame_count)):
            raise ValueError(
                "shuttle tracks must cover every source frame exactly once"
            )

    return run_stage(
        source,
        "shuttle",
        inputs,
        outputs,
        root / "shuttle.manifest.json",
        execute,
        validate,
        force=force,
    )


def run_shuttle_links(
    source: SourceEntry, *, force: bool = False
) -> tuple[bool, dict[str, Path]]:
    from src.single_video.shuttle import (
        link_shuttle_candidates,
        link_shuttle_hypotheses,
    )

    root = source.artifact_root / "shuttle"
    paths = {
        "tracklets": root / "tracklets.jsonl",
        "hypotheses": root / "hypotheses.jsonl",
    }
    inputs = {
        "candidates_sha256": sha256_file(source.candidates_path),
        "implementation": implementation_identity("shuttle-links"),
    }

    def execute(staged):
        link_shuttle_candidates(source.candidates_path, staged[0])
        link_shuttle_hypotheses(source.candidates_path, staged[0], staged[1])

    def validate(staged):
        for path in staged:
            header = json.loads(path.open().readline())
            if header.get("candidate_sha256") != sha256_file(source.candidates_path):
                raise ValueError("shuttle links differ from candidate lineage")

    changed = run_stage(
        source,
        "shuttle-links",
        inputs,
        list(paths.values()),
        root / "links.manifest.json",
        execute,
        validate,
        force=force,
    )
    return changed, paths


def run_extraction(
    catalog: SourceCatalog,
    *,
    source_id: str | None = None,
    video: Path | None = None,
    output_dir: Path | None = None,
    calibration: Path | None = None,
    profile: str = "observations",
    person_model: str | Path | None = None,
    pose_model: Path | None = None,
    model_root: Path | None = None,
    device: str = "auto",
    batch_size: int = 16,
    num_workers: int = 0,
    rebuild: Sequence[str] = (),
    force: bool = False,
    preview: bool = False,
    export_zip: Path | None = None,
    person_extractor: Callable[..., Any] | None = None,
    player_interpreter: Callable[..., Any] | None = None,
    shuttle_extractor: Callable[..., Any] | None = None,
    person_backend_identity: str | None = None,
    pose_backend_identity: str | None = None,
    shuttle_backend_identity: str | None = None,
) -> dict[str, Path]:
    """Complete observations before optional previews/exports. Reuse verified stages."""
    if profile not in ("observations", "rally"):
        raise ValueError("profile must be observations or rally")
    if set(rebuild) - {
        "person",
        "pose",
        "shuttle",
        "shuttle-links",
        "preview",
        "export",
    }:
        raise ValueError("unknown extraction rebuild stage")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
        or isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise ValueError("batch size must be positive and worker count nonnegative")
    if preview and profile != "observations":
        raise ValueError("player preview requires the observations profile")
    if export_zip is not None and Path(export_zip).suffix.lower() != ".zip":
        raise ValueError("export destination must have a .zip extension")
    if profile == "observations" and model_root is None:
        raise ValueError("--model-root is required for shuttle observations")

    resolved_device = select_device(device)
    source = resolve_source(
        catalog, source_id=source_id, video=video, output_dir=output_dir
    )
    calibrate_source(source, calibration, force=force or "pose" in rebuild)
    planned = ["person", "pose"] + (
        ["shuttle", "shuttle-links"] if profile == "observations" else []
    )
    status_path = source.artifact_root / "run-status.json"
    previous = (
        json.loads(status_path.read_text()) if status_path.exists() else {"stages": {}}
    )
    for stage in planned:
        if stage not in previous["stages"]:
            stage_status(source, stage, "pending")
    person_changed = run_person_stage(
        source,
        model=person_model,
        device=resolved_device,
        force=force or "person" in rebuild,
        extractor=person_extractor,
        extractor_identity=person_backend_identity,
    )
    players_changed = run_player_stage(
        source,
        pose_model=pose_model,
        force=force or person_changed or "pose" in rebuild,
        interpreter=player_interpreter,
        interpreter_identity=pose_backend_identity,
    )
    outputs = {
        "person_tracks": source.person_tracks_path,
        "assignments": source.assignments_path,
        "poses": source.pose_cache_path,
        "features": source.artifact_root / "players" / "features.csv",
    }
    observation_changed = person_changed or players_changed
    if profile == "observations":
        shuttle_changed = run_shuttle_stage(
            source,
            model_root=model_root,
            device=resolved_device,
            batch_size=batch_size,
            num_workers=num_workers,
            force=force or "shuttle" in rebuild,
            extractor=shuttle_extractor,
            extractor_identity=shuttle_backend_identity,
        )
        links_changed, linked = run_shuttle_links(
            source, force=force or shuttle_changed or "shuttle-links" in rebuild
        )
        outputs.update(linked)
        observation_changed = observation_changed or shuttle_changed or links_changed
        outputs.update(
            tracks=source.artifact_root / "shuttle" / "tracks.csv",
            candidates=source.candidates_path,
        )
    manifest = source.artifact_root / "observations.manifest.json"
    manifest_value = {
        "schema": "baddievision_observations",
        "schema_version": 1,
        "source_id": source.source_id,
        "profile": profile,
        "identity": SourceIdentity.read(source.identity_path).fingerprint,
        "outputs": {
            str(path.relative_to(source.artifact_root)): sha256_file(path)
            for path in outputs.values()
        },
    }
    if not manifest.exists() or json.loads(manifest.read_text()) != manifest_value:
        atomic_write_json(manifest, manifest_value)
    outputs["manifest"] = manifest
    preview_changed = False
    if preview:
        from src.single_video.artifacts import render_player_preview
        from src.pose_estimator import pose_connections

        path = source.artifact_root / "preview.mp4"

        def render(paths):
            render_player_preview(
                source.video_path,
                outputs["tracks"],
                source.person_tracks_path,
                source.assignments_path,
                source.pose_cache_path,
                paths[0],
                pose_connections(),
                candidate_path=source.candidates_path,
                tracklet_path=outputs["tracklets"],
                hypotheses_path=outputs["hypotheses"],
            )

        preview_changed = run_stage(
            source,
            "preview",
            {
                "observations": sha256_file(manifest),
                "renderer": sha256_file(
                    Path(__file__).resolve().parents[1] / "single_video/artifacts.py"
                ),
            },
            [path],
            source.artifact_root / "preview.manifest.json",
            render,
            lambda paths: None,
            force=force or observation_changed or "preview" in rebuild,
        )
        outputs["preview"] = path
    if export_zip is not None:
        from src.single_video.artifacts import zip_result_dir

        export_zip = Path(export_zip).expanduser().resolve()
        paths = [source.identity_path, source.calibration_path, *outputs.values()]

        def export(staged):
            zip_result_dir(source.artifact_root, staged[0], paths)

        run_stage(
            source,
            "export",
            {
                "implementation": implementation_identity("export"),
                "files": {
                    str(path.relative_to(source.artifact_root)): sha256_file(path)
                    for path in paths
                },
            },
            [export_zip],
            export_zip.with_suffix(export_zip.suffix + ".manifest.json"),
            export,
            lambda paths: None,
            force=force
            or observation_changed
            or preview_changed
            or "export" in rebuild,
        )
        outputs["zip"] = export_zip
    return outputs
