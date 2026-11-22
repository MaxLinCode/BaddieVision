"""Shared extraction stages and durable completion policy for all entrypoints."""

from __future__ import annotations

import ast
import csv
import importlib.metadata
import inspect
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from InPlay.heuristic.person_tracks import BOTSORT_CONTINUOUS, BYTETRACK_PERIODIC

from .artifacts import (
    atomic_write_json,
    calibration_fingerprint,
    canonical_json_sha256,
    validate_person_tracks,
)
from .identity import SourceIdentity, sha256_file
from .registry import SourceEntry
from .publication import publish, recover, recovered_publication
from .source import select_device, verify_source as verify_source


STAGE_CODE = {
    "person": (
        "config/botsort-fixed-camera.yaml",
        "InPlay/heuristic/person_tracks.py",
        "InPlay/heuristic/players.py",
        "src/single_video/video.py",
    ),
    "pose": (
        "InPlay/heuristic/legacy_player_assignment.py",
        "InPlay/heuristic/player_interpretation.py",
        "src/pose_estimator.py",
        "src/court_projection.py",
    ),
    "shuttle": (
        "src/TrackNetV3/predictArgs.py",
        "src/TrackNetV3/dataset.py",
        "src/TrackNetV3/utils/general.py",
        "src/single_video/shuttle.py",
        "src/single_video/models.py",
        "src/TrackNetV3/test.py",
        "src/TrackNetV3/model.py",
        "src/workflow/extraction.py",
    ),
    "shuttle-links": ("src/single_video/shuttle.py",),
    "export": ("src/single_video/artifacts.py",),
    "predict": (
        "src/temporal_selector/rally_inference.py",
        "src/temporal_selector/rally_dataset.py",
        "src/temporal_selector/observations.py",
        "src/temporal_selector/rally_features.py",
        "src/temporal_selector/rally.py",
        "src/temporal_selector/batch.py",
        "src/court_projection.py",
    ),
    "decode": (
        "src/temporal_selector/joint_inference.py",
        "src/temporal_selector/rally_inference.py",
    ),
}
STAGE_PACKAGES = {
    "person": ("torch", "numpy", "opencv-python", "ultralytics", "lap"),
    "pose": ("numpy", "opencv-python", "mediapipe"),
    "shuttle": ("torch", "numpy", "opencv-python"),
    "predict": ("torch", "numpy"),
}


def implementation_identity(stage: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    packages = {}
    for name in STAGE_PACKAGES.get(stage, ()):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return {
        "contract_version": 1,
        "code": {
            name: code_fingerprint(root / name, stage)
            for name in STAGE_CODE.get(stage, ())
            if (root / name).is_file()
        },
        "runtime": packages,
    }


def code_fingerprint(path: Path, stage: str) -> str:
    # Rendering and decoding consume saved probabilities. Their implementation
    # must not invalidate the expensive forward pass.
    definitions = {
        ("extraction.py", "shuttle"): {"_tracknet_adapter"},
        ("artifacts.py", "export"): {"zip_result_dir"},
        ("players.py", "person"): {"ensure_person_detector", "detector_class_name"},
        ("video.py", "person"): {"probe_video", "read_video_info"},
        ("shuttle.py", "shuttle"): {
            "ShuttleCandidateCollector",
            "legacy_tracknet_bbox",
            "_is_sha256",
            "tracknet_candidate_frame_range",
        },
        ("rally_inference.py", "predict"): {
            "predict_windows",
            "predict_source",
            "_identity",
            "validate_prediction_rows",
        },
        ("rally_inference.py", "decode"): {
            "decode_prediction_rows",
            "_decoder",
            "write_prediction_artifacts",
        },
    }
    names = definitions.get((path.name, stage))
    if names is not None:
        tree = ast.parse(path.read_text())
        return canonical_json_sha256(
            {
                "definitions": [
                    ast.dump(node, include_attributes=False)
                    for node in tree.body
                    if not isinstance(node, (ast.FunctionDef, ast.ClassDef))
                    or node.name in names
                ]
            }
        )
    return sha256_file(path)


def backend_identity(
    callback: Callable[..., Any] | None, explicit: str | None = None
) -> str:
    if explicit:
        return explicit
    if callback is None:
        return "builtin"
    try:
        implementation = inspect.getsource(callback)
    except (OSError, TypeError):
        raise ValueError(
            "custom backends without inspectable source require a backend identity"
        )
    return canonical_json_sha256(
        {
            "module": callback.__module__,
            "name": callback.__qualname__,
            "implementation": implementation,
        }
    )


def pose_work_progress(source: SourceEntry) -> dict[str, Any]:
    """Summarize committed units across retained model/source partitions."""
    root = source.artifact_root / "players" / ".pose-work"
    units = list(root.glob("*/*/*/*.json"))
    result: dict[str, Any] = {
        "directory": str(root),
        "retained_unit_count": len(units),
        "last_committed_unit": None,
    }
    if units:
        try:
            latest = max(units, key=lambda path: path.stat().st_mtime_ns)
            record = json.loads(latest.read_text(encoding="utf-8"))
            result["last_committed_unit"] = {
                "path": str(latest.relative_to(root)),
                **{
                    key: record[key]
                    for key in (
                        "frame",
                        "track_id",
                        "raw_artifact_fingerprint",
                        "pose_model_fingerprint",
                        "preprocessing_fingerprint",
                    )
                },
            }
        except (OSError, ValueError, KeyError) as exc:
            result["progress_problem"] = str(exc)
    return result


def read_run_status(source: SourceEntry) -> dict[str, Any] | None:
    """Refresh crop progress even if an abrupt exit left the stage marked running."""
    path = source.artifact_root / "run-status.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if "pose" in value["stages"]:
        value["stages"]["pose"]["committed_work"] = pose_work_progress(source)
    return value


def stage_status(source: SourceEntry, stage: str, state: str, **details: Any) -> None:
    path = source.artifact_root / "run-status.json"
    value = (
        json.loads(path.read_text())
        if path.is_file()
        else {
            "schema": "baddievision_run_status",
            "schema_version": 1,
            "source_id": source.source_id,
            "stages": {},
        }
    )
    if stage == "pose":
        if state in {"completed", "failed"} and not details.get("reused"):
            details["committed_work"] = pose_work_progress(source)
        else:
            details["committed_work"] = (
                value["stages"].get(stage, {}).get("committed_work")
            )
    value["stages"][stage] = {"state": state, **details}
    atomic_write_json(path, value)


def reusable(
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
        f"stale or incomplete artifacts at {receipt.parent}; rerun with --rebuild for this stage or --force"
    )


def run_stage(
    source: SourceEntry,
    stage: str,
    inputs: Mapping[str, Any],
    outputs: Sequence[Path],
    receipt_path: Path,
    execute: Callable[[list[Path]], Any],
    validate: Callable[[Sequence[Path]], None],
    *,
    force: bool = False,
) -> bool:
    """Publish only validated outputs; preserve previous outputs and committed work on failure."""
    recover(receipt_path)
    try:
        if reusable(receipt_path, inputs, outputs, force=force):
            validate(outputs)
            stage_status(source, stage, "completed", reused=True)
            return False
    except (OSError, ValueError) as exc:
        stage_status(source, stage, "stale", reason=str(exc))
        raise
    stage_status(
        source,
        stage,
        "pending",
        reason="rebuild requested" if force else "no completion receipt",
    )
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    stage_status(source, stage, "running")
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{stage}-", dir=outputs[0].parent
        ) as temporary:
            staged = [Path(temporary) / path.name for path in outputs]
            execute(staged)
            validate(staged)
            publish(
                staged,
                outputs,
                receipt_path,
                {
                    "schema": "baddievision_pipeline_stage",
                    "schema_version": 1,
                    "inputs": dict(inputs),
                    "outputs": {
                        path.name: sha256_file(staged_path)
                        for path, staged_path in zip(outputs, staged)
                    },
                },
            )
    except BaseException as exc:
        stage_status(
            source,
            stage,
            "failed",
            reason=str(exc),
            committed_work=str(source.artifact_root / "players" / ".pose-work")
            if stage == "pose"
            else None,
        )
        raise
    stage_status(source, stage, "completed", reused=False)
    return True


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


def run_person_stage(
    source: SourceEntry,
    *,
    model: str | Path | None = None,
    device: str = "auto",
    force: bool = False,
    extractor: Callable[..., Any] | None = None,
    extractor_identity: str | None = None,
    tracking_profile: str = BOTSORT_CONTINUOUS,
) -> bool:
    identity = verify_source(source)
    imported_receipt = source.person_tracks_path.with_name(
        "person-tracks.manifest.json"
    )
    recover(imported_receipt)
    if imported_receipt.exists() and not force:
        previous = json.loads(imported_receipt.read_text())
        if previous.get("inputs", {}).get("origin") == "validated_import":
            if previous["inputs"].get("identity") != identity.fingerprint:
                raise ValueError(
                    "imported person observations have stale source identity"
                )
            if not reusable(
                imported_receipt,
                previous["inputs"],
                [source.person_tracks_path],
                force=False,
            ):
                raise ValueError("imported person observations are incomplete")
            validate_person_tracks(source.person_tracks_path, identity)
            stage_status(
                source, "person", "completed", reused=True, origin="validated_import"
            )
            return False
    from InPlay.heuristic.person_tracks import default_person_model

    if device == "auto":
        device = select_device(device)
    if model is None:
        model = (
            "yolov8n.pt" if tracking_profile == BYTETRACK_PERIODIC
            else default_person_model(device)
        )
    # Resolve an Ultralytics alias before hashing; receipts always describe model bytes.
    model_path = Path(model).expanduser()
    if not model_path.is_file() and model_path.suffix.lower() == ".engine":
        raise FileNotFoundError(
            f"person detector engine is missing: {model_path}; "
            "export YOLO26L on the target NVIDIA GPU (see docs/workflow.md)"
        )
    if not model_path.is_file():
        from ultralytics import YOLO

        model_path = Path(YOLO(str(model)).ckpt_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"person detector is missing: {model_path}")
    inputs = {
        "identity": identity.fingerprint,
        "detector_sha256": sha256_file(model_path),
        "implementation": implementation_identity("person"),
        "backend": backend_identity(extractor, extractor_identity),
        "device": device,
        "tracking_profile": tracking_profile,
    }

    def execute(paths):
        if extractor is None:
            from InPlay.heuristic.person_tracks import extract_person_tracks

            extract_person_tracks(
                source.video_path,
                paths[0],
                str(model_path),
                device=device,
                tracking_profile=tracking_profile,
            )
        else:
            extractor(source.video_path, paths[0], str(model_path))

    return run_stage(
        source,
        "person",
        inputs,
        [source.person_tracks_path],
        source.person_tracks_path.with_name("person-tracks.manifest.json"),
        execute,
        lambda paths: validate_person_tracks(paths[0], identity),
        force=force,
    )


def run_player_stage(
    source: SourceEntry,
    *,
    pose_model: Path | None = None,
    pose_crop_padding: float = 0.20,
    force: bool = False,
    interpreter: Callable[..., Any] | None = None,
    interpreter_identity: str | None = None,
    tracking_profile: str | None = None,
) -> bool:
    identity = verify_source(source)
    tracks = source.person_tracks_path
    receipt = tracks.with_name("person-tracks.manifest.json")
    with recovered_publication(receipt):
        try:
            committed = json.loads(receipt.read_text(encoding="utf-8"))
            tracks_sha256 = sha256_file(tracks)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"cannot verify committed person tracks at {tracks}; "
                "run stage person-tracks or import the tracks"
            ) from exc
        if (
            not isinstance(committed, dict)
            or committed.get("schema") != "baddievision_pipeline_stage"
            or committed.get("schema_version") != 1
            or not isinstance(committed.get("inputs"), dict)
            or committed["inputs"].get("identity") != identity.fingerprint
            or committed.get("outputs") != {tracks.name: tracks_sha256}
        ):
            raise ValueError(
                f"person tracks differ from the committed receipt at {receipt}; "
                "rebuild or reimport the tracks"
            )
        person_profile = committed["inputs"].get("tracking_profile")
        if tracking_profile is None:
            tracking_profile = person_profile or BOTSORT_CONTINUOUS
        elif person_profile is not None and person_profile != tracking_profile:
            raise ValueError("player assignment profile differs from person tracks")
        validate_person_tracks(tracks, identity)
        from src.pose_estimator import ensure_pose_model_asset

        pose_model = Path(ensure_pose_model_asset(pose_model))
        inputs = {
            "identity": identity.fingerprint,
            "person_tracks_sha256": tracks_sha256,
            "calibration_sha256": calibration_fingerprint(source),
            "pose_sha256": sha256_file(pose_model),
            "pose_crop_padding": pose_crop_padding,
            "tracking_profile": tracking_profile,
            "implementation": implementation_identity("pose"),
            "backend": backend_identity(interpreter, interpreter_identity),
        }
        outputs = [
            source.assignments_path,
            source.pose_cache_path,
            source.artifact_root / "players" / "features.csv",
        ]

        def execute(paths):
            callback = interpreter
            if callback is None:
                from InPlay.heuristic.player_interpretation import interpret_players

                callback = interpret_players
            if source.pose_cache_path.is_file():
                shutil.copyfile(source.pose_cache_path, paths[1])
            kwargs = {
                "pose_model_asset": pose_model,
                "pose_crop_padding": pose_crop_padding,
            }
            if "tracking_profile" in inspect.signature(callback).parameters:
                kwargs["tracking_profile"] = tracking_profile
            if "pose_work_dir" in inspect.signature(callback).parameters:
                kwargs["pose_work_dir"] = source.artifact_root / "players" / ".pose-work"
            if (
                interpreter is not None or interpreter_identity is not None
            ) and "pose_backend_identity" in inspect.signature(callback).parameters:
                kwargs["pose_backend_identity"] = {"interpreter": inputs["backend"]}
            callback(
                source.video_path,
                tracks,
                source.calibration_path,
                paths[1],
                paths[0],
                paths[2],
                **kwargs,
            )

        return run_stage(
            source,
            "pose",
            inputs,
            outputs,
            source.artifact_root / "players" / "players.manifest.json",
            execute,
            lambda paths: _player_stage_valid(paths, source, identity),
            force=force,
        )
