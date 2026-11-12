"""Reproducible single-video inference using versioned, candidate-free assets."""

from __future__ import annotations

import importlib.metadata
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import (
    atomic_write_json,
    calibration_fingerprint,
    canonical_json_sha256,
    import_calibration,
    register_native_calibration,
)
from .identity import SourceIdentity, inspect_source, sha256_file
from .registry import SourceCatalog, SourceEntry


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


def _runtime() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "opencv-python", "mediapipe", "ultralytics")
    }


def _code_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in (
            root / "src/workflow/rally_pipeline.py",
            root / "src/temporal_selector/rally_inference.py",
            root / "src/temporal_selector/rally_dataset.py",
            root / "src/temporal_selector/dataset.py",
            root / "src/temporal_selector/rally_features.py",
            root / "src/temporal_selector/rally.py",
            root / "src/temporal_selector/batch.py",
            root / "src/temporal_selector/joint_inference.py",
            root / "src/court_projection.py",
            root / "src/rally_release.py",
            root / "InPlay/heuristic/players.py",
            root / "InPlay/heuristic/person_tracks.py",
            root / "InPlay/heuristic/player_interpretation.py",
            root / "src/pose_estimator.py",
        )
    }


def _reusable(
    receipt: Path, inputs: Mapping[str, Any], outputs: Sequence[Path], *, force: bool
) -> bool:
    if force:
        return False
    if not receipt.exists() and not any(path.exists() for path in outputs):
        return False
    try:
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        expected = {str(path.name): sha256_file(path) for path in outputs}
        if (
            previous.get("inputs") == dict(inputs)
            and previous.get("outputs") == expected
        ):
            return True
    except (OSError, ValueError, TypeError):
        pass
    raise ValueError(
        f"stale or incomplete artifacts at {receipt.parent}; rerun with --force"
    )


def _receipt(path: Path, inputs: Mapping[str, Any], outputs: Sequence[Path]) -> Path:
    return atomic_write_json(
        path,
        {
            "schema": "baddievision_pipeline_stage",
            "schema_version": 1,
            "inputs": dict(inputs),
            "outputs": {item.name: sha256_file(item) for item in outputs},
        },
    )


def _person_coverage(path: Path, identity: SourceIdentity) -> None:
    from InPlay.heuristic.person_tracks import load_person_tracks

    metadata, records = load_person_tracks(path)
    if (
        metadata.get("frame_count") != identity.frame_count
        or metadata.get("frame_size") != [identity.width, identity.height]
        or not math.isclose(
            float(metadata.get("fps", 0)),
            identity.fps_numerator / identity.fps_denominator,
            rel_tol=0,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("person observations do not match the source video")
    if any(not 0 <= int(row["frame"]) < identity.frame_count for row in records):
        raise ValueError("person observation frame is outside the source video")


def _assignment_coverage(path: Path, identity: SourceIdentity) -> None:
    with path.open(encoding="utf-8") as handle:
        metadata = json.loads(next(handle))
        frames = [int(json.loads(line)["frame"]) for line in handle if line.strip()]
    if metadata.get("schema") != "player_assignments" or frames != list(
        range(identity.frame_count)
    ):
        raise ValueError("player assignments must cover each source frame exactly once")


def _player_stage_valid(
    paths: Sequence[Path],
    source: SourceEntry,
    identity: SourceIdentity,
) -> None:
    """Validate staged player artifacts before publishing their completion receipt."""
    from InPlay.heuristic.player_interpretation import load_pose_cache

    assignments, poses, features = paths
    _assignment_coverage(assignments, identity)
    with assignments.open(encoding="utf-8") as handle:
        metadata = json.loads(next(handle))
    raw = "sha256:" + sha256_file(source.person_tracks_path)
    if (
        metadata.get("source_id") != source.source_id
        or metadata.get("source_identity_sha256") != identity.fingerprint
        or metadata.get("calibration_sha256") != sha256_file(source.calibration_path)
        or metadata.get("raw_artifact_fingerprint") != raw
        or metadata.get("frame_count") != identity.frame_count
        or metadata.get("frame_size") != [identity.width, identity.height]
        or not math.isclose(
            float(metadata.get("fps", 0)),
            identity.fps_numerator / identity.fps_denominator,
            rel_tol=0,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("player assignments do not match source/calibration lineage")
    pose_metadata, records = load_pose_cache(poses)
    if (
        pose_metadata is None
        or pose_metadata.get("raw_artifact_fingerprint") != raw
        or any(
            not str(pose_metadata.get(key, "")).startswith("sha256:")
            for key in ("pose_model_fingerprint", "preprocessing_fingerprint")
        )
        or any(not 0 <= int(row["frame"]) < identity.frame_count for row in records)
    ):
        raise ValueError("pose cache has invalid source lineage or frame coverage")
    with features.open(newline="", encoding="utf-8") as handle:
        frame_indices = [int(row["Frame"]) for row in csv.DictReader(handle)]
    if frame_indices != list(range(identity.frame_count)):
        raise ValueError(
            "player feature rows must cover every source frame exactly once"
        )


def _calibrate(source: SourceEntry, external: Path | None, *, force: bool) -> None:
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


def run_pipeline(
    catalog: SourceCatalog,
    *,
    source_id: str | None = None,
    video: Path | None = None,
    output_dir: Path | None = None,
    calibration: Path | None = None,
    model_id: str = "rally-six-source-v1",
    checkpoint: Path | None = None,
    model_manifest: Path | None = None,
    cache_dir: Path | None = None,
    device: str = "auto",
    batch_size: int = 16,
    overlay: bool = True,
    force: bool = False,
    person_extractor: Callable[..., Any] | None = None,
    player_interpreter: Callable[..., Any] | None = None,
) -> dict[str, Path]:
    """Generate or verify observations, then infer using the frozen release contract."""
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch size must be positive")
    if (checkpoint is None) != (model_manifest is None):
        raise ValueError("--checkpoint and --model-manifest must be supplied together")
    resolved_device = select_device(device)
    if video is None:
        if source_id is None:
            raise ValueError("provide --video FILE or --source ID")
        source = catalog.select([source_id])[0]
        if not source.enabled:
            raise ValueError(f"source is disabled: {source_id}")
        if output_dir is not None and output_dir.resolve() != source.artifact_root:
            raise ValueError(
                "--output-dir differs from the registered source artifact root"
            )
    else:
        if output_dir is None:
            raise ValueError("--output-dir is required with --video")
        previous_identity = output_dir.expanduser().resolve() / "source.json"
        if previous_identity.exists():
            requested_identity = inspect_source(source_id or video.stem, video)
            if SourceIdentity.read(previous_identity) != requested_identity:
                raise ValueError(
                    "output directory belongs to a different source; choose a new --output-dir"
                )
        source = catalog.add(video, source_id=source_id, output_dir=output_dir)
        if source.artifact_root != output_dir.expanduser().resolve():
            raise ValueError(
                "video is already registered with a different output directory"
            )
    identity = SourceIdentity.read(source.identity_path)
    if inspect_source(source.source_id, source.video_path) != identity:
        raise ValueError("source video identity changed; register it as a new source")
    _calibrate(source, calibration, force=force)

    from src.rally_release import resolve_release
    from src.temporal_selector.rally_dataset import RallySourceConfig
    from src.temporal_selector.rally_inference import (
        predict_source,
        write_prediction_artifacts,
    )

    model, manifest, assets = resolve_release(
        model_id, checkpoint=checkpoint, manifest=model_manifest, cache_dir=cache_dir
    )
    print(
        f"{source.source_id}: verified model assets; using {resolved_device}",
        flush=True,
    )
    preprocessing = {
        **manifest["preprocessing"],
        "feature_view": manifest["rally_feature_view"],
    }
    common = {
        "video_sha256": identity.video_sha256,
        "identity": identity.fingerprint,
        "runtime": _runtime(),
        "code": _code_identity(),
    }
    players_root = source.artifact_root / "players"
    players_root.mkdir(parents=True, exist_ok=True)
    person_inputs = {
        **common,
        "detector_sha256": sha256_file(assets["person_detector"]),
    }
    person_receipt = players_root / "person-tracks.manifest.json"
    if not _reusable(
        person_receipt, person_inputs, [source.person_tracks_path], force=force
    ):
        if person_extractor is None:
            from InPlay.heuristic.person_tracks import extract_person_tracks

            person_extractor = extract_person_tracks
        print(f"{source.source_id}: extracting person tracks", flush=True)
        with tempfile.TemporaryDirectory(
            prefix=".person-", dir=players_root
        ) as temporary:
            tracks = Path(temporary) / "person_tracks.jsonl"
            person_extractor(source.video_path, tracks, str(assets["person_detector"]))
            _person_coverage(tracks, identity)
            tracks.replace(source.person_tracks_path)
        _receipt(person_receipt, person_inputs, [source.person_tracks_path])
    _person_coverage(source.person_tracks_path, identity)
    player_inputs = {
        **common,
        "person_tracks_sha256": sha256_file(source.person_tracks_path),
        "calibration_sha256": calibration_fingerprint(source),
        "pose_sha256": sha256_file(assets["pose_landmarker"]),
        "preprocessing": preprocessing,
    }
    player_outputs = [
        source.assignments_path,
        source.pose_cache_path,
        players_root / "features.csv",
    ]
    player_receipt = players_root / "players.manifest.json"
    if not _reusable(player_receipt, player_inputs, player_outputs, force=force):
        if player_interpreter is None:
            from InPlay.heuristic.player_interpretation import interpret_players

            player_interpreter = interpret_players
        print(
            f"{source.source_id}: interpreting players and extracting pose", flush=True
        )
        with tempfile.TemporaryDirectory(
            prefix=".players-", dir=players_root
        ) as temporary:
            stage = Path(temporary)
            staged = [stage / path.name for path in player_outputs]
            player_interpreter(
                source.video_path,
                source.person_tracks_path,
                source.calibration_path,
                staged[1],
                staged[0],
                staged[2],
                pose_model_asset=assets["pose_landmarker"],
                pose_crop_padding=preprocessing["pose_crop_padding"],
            )
            _player_stage_valid(staged, source, identity)
            for original, destination in zip(staged, player_outputs):
                original.replace(destination)
        _receipt(player_receipt, player_inputs, player_outputs)
    _player_stage_valid(player_outputs, source, identity)
    prediction_inputs = {
        **player_inputs,
        "player_outputs": {path.name: sha256_file(path) for path in player_outputs},
        "release_manifest_sha256": canonical_json_sha256(manifest),
        "checkpoint_sha256": sha256_file(assets["checkpoint"]),
        "device": resolved_device,
        "batch_size": batch_size,
        "overlay": overlay,
    }
    rally_root = source.artifact_root / "rallies"
    outputs = {
        "predictions": rally_root / "predictions.jsonl",
        "rallies": rally_root / "rallies.csv",
    }
    if overlay:
        outputs["overlay"] = rally_root / "overlay.mp4"
    receipt = rally_root / "run.manifest.json"
    if _reusable(receipt, prediction_inputs, list(outputs.values()), force=force):
        return {**outputs, "manifest": receipt}
    print(f"{source.source_id}: predicting rallies and rendering outputs", flush=True)
    rally_source = RallySourceConfig(
        source.source_id,
        source.video_path,
        source.assignments_path,
        source.pose_cache_path,
        source.calibration_path,
    )
    rows = predict_source(
        rally_source,
        model,
        preprocessing=preprocessing,
        decoder=manifest["decoder_configuration"],
        device=resolved_device,
        batch_size=batch_size,
    )
    write_prediction_artifacts(
        rally_source,
        rows,
        rally_root,
        metadata={
            "model_id": manifest["model_id"],
            "input_fingerprints": prediction_inputs,
        },
        overlay=overlay,
    )
    _receipt(receipt, prediction_inputs, list(outputs.values()))
    return {**outputs, "manifest": receipt}
