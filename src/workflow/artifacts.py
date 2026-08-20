"""Canonical artifact discovery, fingerprints, and calibration import."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .identity import SourceIdentity, sha256_file
from .registry import SourceEntry


IMPORTABLE_ARTIFACTS = {
    "shuttle-candidates-pilot": ("candidates_path", "shuttle_candidates"),
    "shuttle-candidates-frozen": ("frozen_candidates_path", "shuttle_candidates"),
    "person-tracks": ("person_tracks_path", "person_tracks"),
    "player-poses": ("pose_cache_path", "pose_cache"),
}


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def import_artifact(
    source: SourceEntry, kind: str, external_path: str | Path, *, force: bool = False
) -> Path:
    """Copy a validated legacy artifact without modifying its original."""
    if kind not in IMPORTABLE_ARTIFACTS:
        raise ValueError(f"unsupported import artifact: {kind}")
    attribute, expected_schema = IMPORTABLE_ARTIFACTS[kind]
    destination = Path(getattr(source, attribute))
    external = Path(external_path).expanduser().resolve()
    if destination.exists() and not force:
        raise FileExistsError(f"canonical artifact already exists: {destination}")
    with external.open(encoding="utf-8") as handle:
        try:
            metadata = json.loads(next(handle))
        except (StopIteration, json.JSONDecodeError) as exc:
            raise ValueError("import artifact has no valid metadata record") from exc
    if metadata.get("schema") != expected_schema:
        raise ValueError(f"expected {expected_schema} artifact")
    identity = SourceIdentity.read(source.identity_path)
    artifact_video_sha = metadata.get("source_video_sha256") or metadata.get("video_sha256")
    if artifact_video_sha is not None and artifact_video_sha != identity.video_sha256:
        raise ValueError("artifact video fingerprint differs from source identity")
    frame_count = metadata.get("frame_count")
    if frame_count is not None and int(frame_count) != identity.frame_count:
        raise ValueError("artifact frame count differs from source identity")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(external, temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def import_calibration(
    source: SourceEntry,
    external_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Copy an external calibration into the source's sole canonical location."""
    external = Path(external_path).expanduser().resolve()
    if external == source.calibration_path.resolve():
        raise ValueError("calibration is already at the canonical artifact path")
    value = json.loads(external.read_text(encoding="utf-8"))
    image_size = value.get("image_size")
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError("calibration must include image_size [width, height]")
    identity = SourceIdentity.read(source.identity_path)
    if [int(image_size[0]), int(image_size[1])] != [identity.width, identity.height]:
        raise ValueError("calibration image_size does not match source identity")
    if source.calibration_path.exists() and not force:
        raise FileExistsError(f"canonical calibration already exists: {source.calibration_path}")
    # Metadata is embedded in the calibration so downstream consumers have one
    # file and one authority.  The legacy path is informational only.
    value["artifact_metadata"] = {
        "schema": "baddievision_court_calibration_artifact",
        "schema_version": 1,
        "source_id": source.source_id,
        "source_identity_sha256": identity.fingerprint,
        "source_frame": None,
        "source_frame_status": "legacy_unknown",
        "frame_indexing": "source_local_zero_based",
        "imported_from": str(external),
        "imported_file_sha256": sha256_file(external),
    }
    return atomic_write_json(source.calibration_path, value)


def register_native_calibration(source: SourceEntry, *, source_frame: int) -> Path:
    """Attach canonical artifact metadata after the native calibration UI writes."""
    value = json.loads(source.calibration_path.read_text(encoding="utf-8"))
    identity = SourceIdentity.read(source.identity_path)
    if value.get("image_size") != [identity.width, identity.height]:
        raise ValueError("calibration image_size does not match source identity")
    if not 0 <= int(source_frame) < identity.frame_count:
        raise ValueError("calibration frame lies outside the source-local frame domain")
    value["artifact_metadata"] = {
        "schema": "baddievision_court_calibration_artifact",
        "schema_version": 1,
        "source_id": source.source_id,
        "source_identity_sha256": identity.fingerprint,
        "source_frame": int(source_frame),
        "frame_indexing": "source_local_zero_based",
        "producer": "src.calibrate_court",
    }
    return atomic_write_json(source.calibration_path, value)


def calibration_fingerprint(source: SourceEntry) -> str:
    if not source.calibration_path.is_file():
        raise FileNotFoundError(f"canonical calibration is missing: {source.calibration_path}")
    value = json.loads(source.calibration_path.read_text(encoding="utf-8"))
    identity = SourceIdentity.read(source.identity_path)
    metadata = value.get("artifact_metadata", {})
    if (
        metadata.get("source_id") != source.source_id
        or metadata.get("source_identity_sha256") != identity.fingerprint
        or metadata.get("frame_indexing") != "source_local_zero_based"
    ):
        raise ValueError("calibration was produced for a different source identity")
    if value.get("image_size") != [identity.width, identity.height]:
        raise ValueError("calibration image_size does not match source identity")
    return canonical_json_sha256(value)
