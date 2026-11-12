"""Sanitized, checksum-verified public rally inference model packages.

Research checkpoints remain private and retain their existing schema. Public
packages contain only an allowlisted inference contract and CPU tensors.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import torch

from src.temporal_selector.joint_inference import InPlayDecoderConfig
from src.temporal_selector.rally import (
    RALLY_CONTEXT_DIRECTION,
    RALLY_MODEL_TYPE,
    RallySegmenter,
    RallySegmenterConfig,
)
from src.temporal_selector.rally_features import (
    RALLY_FEATURE_SCHEMA,
    RALLY_FEATURE_VERSION,
    RALLY_FEATURE_VIEWS,
)

RELEASE_SCHEMA = "baddievision.rally.inference_release"
RELEASE_VERSION = 1
RELEASE_BASE_URL = "https://github.com/MaxLinCode/BaddieVision/releases/download"
ASSET_NAMES = ("checkpoint", "person_detector", "pose_landmarker")
DEFAULT_PREPROCESSING = {
    "pose_coordinate_mode": "player_relative",
    "pose_visibility_threshold": 0.5,
    "context_seconds": 1.0,
    "ownership_seconds": 1.0,
    "pose_crop_padding": 0.20,
    "pose_retry_padding": 0.25,
    "pose_quality_threshold": 0.55,
    "pose_backend": "mediapipe.tasks.pose_landmarker",
}
CONTRACT_FIELDS = (
    "schema",
    "schema_version",
    "model_id",
    "model_type",
    "code_revision",
    "rally_segmenter_config",
    "rally_feature_schema",
    "rally_feature_version",
    "rally_feature_view",
    "rally_feature_names",
    "context_direction",
    "decoder_configuration",
    "preprocessing",
    "runtime_versions",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", value):
        raise ValueError("model_id must be a simple immutable release identifier")
    return value


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "uninstalled"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Reject unsupported feature, preprocessing, decoder, and asset contracts."""
    value = dict(manifest)
    missing = set(CONTRACT_FIELDS) - set(value)
    if missing:
        raise ValueError(f"release manifest is missing fields: {sorted(missing)}")
    if set(value) != {*CONTRACT_FIELDS, "assets"}:
        raise ValueError("release manifest contains unsupported or private metadata")
    if value["schema"] != RELEASE_SCHEMA or value["schema_version"] != RELEASE_VERSION:
        raise ValueError("unsupported inference release schema/version")
    _model_id(value["model_id"])
    if value["model_type"] != RALLY_MODEL_TYPE:
        raise ValueError("release is not a RallySegmenter")
    if not re.fullmatch(r"[0-9a-f]{7,40}", value["code_revision"]):
        raise ValueError("code_revision must be a Git commit ID")
    if (
        value["rally_feature_schema"] != RALLY_FEATURE_SCHEMA
        or value["rally_feature_version"] != RALLY_FEATURE_VERSION
    ):
        raise ValueError("release feature schema/version is incompatible")
    view = RALLY_FEATURE_VIEWS.get(value["rally_feature_view"])
    config = RallySegmenterConfig(**value["rally_segmenter_config"])
    if not math.isfinite(config.layer_norm_eps):
        raise ValueError("model layer_norm_eps must be finite")
    if (
        view is None
        or value["rally_feature_names"] != list(view.names)
        or config.frame_feature_dim != view.dimension
    ):
        raise ValueError("release feature view/names/dimension is incompatible")
    if value["context_direction"] != RALLY_CONTEXT_DIRECTION:
        raise ValueError("release must use offline centered context")
    versions = value["runtime_versions"]
    if (
        not isinstance(versions, dict)
        or set(versions)
        != {"python", "torch", "ultralytics", "mediapipe", "numpy", "opencv-python"}
        or any(
            not isinstance(v, str) or not v or len(v) > 100 for v in versions.values()
        )
    ):
        raise ValueError(
            "runtime_versions must record the inference dependency versions"
        )
    preprocessing = value["preprocessing"]
    if set(preprocessing) != set(DEFAULT_PREPROCESSING):
        raise ValueError("release preprocessing fields are incompatible")
    for name in (
        "pose_coordinate_mode",
        "pose_backend",
        "pose_retry_padding",
        "pose_quality_threshold",
    ):
        if preprocessing[name] != DEFAULT_PREPROCESSING[name]:
            raise ValueError(f"unsupported preprocessing setting: {name}")
    for name in ("context_seconds", "ownership_seconds"):
        if not math.isfinite(preprocessing[name]) or preprocessing[name] <= 0:
            raise ValueError(f"preprocessing {name} must be finite and positive")
    for name in ("pose_visibility_threshold", "pose_crop_padding"):
        if not math.isfinite(preprocessing[name]) or not 0 <= preprocessing[name] <= 1:
            raise ValueError(f"preprocessing {name} must be in [0, 1]")
    decoder = value["decoder_configuration"]
    parsed = InPlayDecoderConfig(**decoder)
    if (
        not all(
            math.isfinite(getattr(parsed, name))
            for name in ("threshold", "max_gap_seconds", "minimum_duration_seconds")
        )
        or type(parsed.preserve_edge_runs) is not bool
    ):
        raise ValueError("release decoder settings must be finite and valid")
    assets = value.get("assets", {})
    if set(assets) != set(ASSET_NAMES):
        raise ValueError(
            "release requires checkpoint, person_detector and pose_landmarker assets"
        )
    for name, asset in assets.items():
        filename = asset.get("filename", "")
        if not filename or Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError(f"invalid asset filename: {name}")
        if not re.fullmatch("[0-9a-f]{64}", asset.get("sha256", "")):
            raise ValueError(f"invalid asset SHA-256: {name}")
        if type(asset.get("size_bytes")) is not int or asset["size_bytes"] <= 0:
            raise ValueError(f"invalid asset size: {name}")
        parsed_url = urlparse(asset.get("url", ""))
        if parsed_url.scheme not in {"https", "http", "file"}:
            raise ValueError(f"unsupported asset URL: {name}")
        if parsed_url.scheme == "http" and parsed_url.hostname not in {
            "127.0.0.1",
            "localhost",
        }:
            raise ValueError("HTTP assets are allowed only for local tests")
    return value


def verify_asset(path: Path, asset: Mapping[str, Any]) -> Path:
    path = Path(path)
    if (
        not path.is_file()
        or path.stat().st_size != asset["size_bytes"]
        or sha256_file(path) != asset["sha256"]
    ):
        raise ValueError(
            f"asset is missing, corrupt, or has an incorrect SHA-256: {path.name}"
        )
    return path


def fetch_asset(asset: Mapping[str, Any], cache_dir: Path) -> Path:
    """Download only to a temporary file; expose bytes after hash verification."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / asset["filename"]
    if destination.exists():
        return verify_asset(destination, asset)
    fd, temporary = tempfile.mkstemp(prefix=".download-", dir=cache_dir)
    try:
        with os.fdopen(fd, "wb") as output, urlopen(asset["url"], timeout=60) as source:
            total = 0
            while block := source.read(1024 * 1024):
                total += len(block)
                if total > asset["size_bytes"]:
                    raise ValueError("download exceeds release asset size")
                output.write(block)
        verify_asset(Path(temporary), asset)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def load_release(
    checkpoint: Path, manifest: Path | Mapping[str, Any]
) -> tuple[RallySegmenter, dict[str, Any]]:
    """Load only verified tensors/primitives, without the research provenance."""
    raw = (
        json.loads(Path(manifest).read_text())
        if isinstance(manifest, (str, Path))
        else manifest
    )
    value = validate_manifest(raw)
    verify_asset(Path(checkpoint), value["assets"]["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {*CONTRACT_FIELDS, "model_state_dict"}:
        raise ValueError("checkpoint contains unsupported or private metadata")
    if any(payload[name] != value[name] for name in CONTRACT_FIELDS):
        raise ValueError("checkpoint and manifest inference contracts differ")
    state = payload["model_state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint model_state_dict must contain tensors")
    if any(
        not isinstance(tensor, torch.Tensor) or not bool(torch.isfinite(tensor).all())
        for tensor in state.values()
    ):
        raise ValueError("checkpoint weights must be finite tensors")
    model = RallySegmenter(RallySegmenterConfig(**value["rally_segmenter_config"]))
    model.load_state_dict(state, strict=True)
    return model.eval(), value


def resolve_release(
    model_id: str = "rally-six-source-v1",
    *,
    checkpoint: Path | None = None,
    manifest: Path | None = None,
    cache_dir: Path | None = None,
) -> tuple[RallySegmenter, dict[str, Any], dict[str, Path]]:
    """Resolve a named GitHub release or an explicit local checkpoint/manifest."""
    if (checkpoint is None) != (manifest is None):
        raise ValueError(
            "local checkpoint requires both --checkpoint and --model-manifest"
        )
    cache = Path(
        cache_dir
        or os.environ.get("BADDIEVISION_CACHE_DIR")
        or Path.home() / ".cache" / "baddievision" / "models"
    )
    if manifest is not None:
        manifest_value = validate_manifest(json.loads(Path(manifest).read_text()))
    else:
        model_id = _model_id(model_id)
        manifest_path = cache / model_id / "manifest.json"
        if not manifest_path.exists():
            try:
                with urlopen(
                    f"{RELEASE_BASE_URL}/{model_id}/manifest.json", timeout=30
                ) as source:
                    data = source.read(1024 * 1024 + 1)
                if len(data) > 1024 * 1024:
                    raise ValueError("release manifest exceeds 1 MiB")
                manifest_value = validate_manifest(json.loads(data))
                if manifest_value["model_id"] != model_id:
                    raise ValueError(
                        "downloaded release ID does not match requested model"
                    )
                _atomic_json(manifest_path, manifest_value)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"release {model_id} is unavailable; provide --checkpoint and --model-manifest "
                    "from a verified export, or retry after the release is published"
                ) from error
        else:
            manifest_value = validate_manifest(json.loads(manifest_path.read_text()))
            if manifest_value["model_id"] != model_id:
                raise ValueError("cached release ID does not match requested model")
    asset_cache = cache / manifest_value["model_id"]
    paths = {}
    for name, asset in manifest_value["assets"].items():
        adjacent = (
            Path(manifest).resolve().parent / asset["filename"] if manifest else None
        )
        if name == "checkpoint" and checkpoint:
            paths[name] = verify_asset(Path(checkpoint), asset)
        elif adjacent is not None and adjacent.exists():
            # Validate local exports before their immutable release URL exists.
            # A corrupt adjacent asset is an error, never a silent download fallback.
            paths[name] = verify_asset(adjacent, asset)
        else:
            paths[name] = fetch_asset(asset, asset_cache)
    model, value = load_release(paths["checkpoint"], manifest_value)
    return model, value, paths


def export_release(
    research_checkpoint: Path,
    output_dir: Path,
    *,
    person_detector: Path,
    pose_landmarker: Path,
    code_revision: str,
    base_url: str,
    model_id: str = "rally-six-source-v1",
    preprocessing: Mapping[str, Any] | None = None,
) -> Path:
    """Export allowlisted metadata; never copy private research checkpoint extras.

    ``research_checkpoint`` must be a trusted locally generated checkpoint. The
    public loader deliberately uses ``weights_only=True`` instead.
    """
    import shutil

    model_id = _model_id(model_id)
    research = torch.load(research_checkpoint, map_location="cpu", weights_only=False)
    model = RallySegmenter.from_checkpoint(research).eval()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("release output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": RELEASE_SCHEMA,
        "schema_version": RELEASE_VERSION,
        "model_id": model_id,
        "model_type": RALLY_MODEL_TYPE,
        "code_revision": code_revision,
        **{
            name: research[name]
            for name in (
                "rally_segmenter_config",
                "rally_feature_schema",
                "rally_feature_version",
                "rally_feature_view",
                "rally_feature_names",
                "context_direction",
                "decoder_configuration",
            )
        },
        "preprocessing": dict(preprocessing or DEFAULT_PREPROCESSING),
        "runtime_versions": {
            "python": platform.python_version(),
            **{
                name: _package_version(name)
                for name in (
                    "torch",
                    "ultralytics",
                    "mediapipe",
                    "numpy",
                    "opencv-python",
                )
            },
        },
    }
    checkpoint_path = output / "rally-segmenter.pt"
    torch.save(
        {
            **contract,
            "model_state_dict": {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            },
        },
        checkpoint_path,
    )
    assets = {"checkpoint": checkpoint_path}
    for name, source, filename in (
        ("person_detector", person_detector, "person-detector.pt"),
        ("pose_landmarker", pose_landmarker, "pose-landmarker.task"),
    ):
        path = output / filename
        shutil.copyfile(source, path)
        assets[name] = path
    manifest = {
        **contract,
        "assets": {
            name: {
                "filename": path.name,
                "url": f"{base_url.rstrip('/')}/{path.name}",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in assets.items()
        },
    }
    validate_manifest(manifest)
    load_release(checkpoint_path, manifest)
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path
