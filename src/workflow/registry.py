"""Canonical local source catalog and generated downstream configurations."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .identity import SourceIdentity, inspect_source, sha256_file

CATALOG_SCHEMA = "baddievision_source_catalog"
CATALOG_VERSION = 1
DEFAULT_SEED = 1729
_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _portable_path(path: Path, parent: Path) -> str:
    path = path.expanduser().resolve()
    try:
        return os.path.relpath(path, parent.resolve())
    except ValueError:
        return str(path)


def _resolve(parent: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (parent / path).resolve() if not path.is_absolute() else path.resolve()


def _loso_manifest(source_ids: list[str], seed: int) -> dict[str, Any]:
    """Mirror the selector's deterministic LOSO format without importing torch."""
    folds = [
        {
            "fold_id": chr(ord("A") + index),
            "training_source_ids": [item for item in source_ids if item != held_out],
            "evaluation_source_ids": [held_out],
        }
        for index, held_out in enumerate(source_ids)
    ]
    payload = {"seed": int(seed), "folds": folds}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**payload, "fingerprint": fingerprint}


@dataclass(frozen=True)
class SourceEntry:
    source_id: str
    video_path: Path
    artifact_root: Path
    enabled: bool = True

    @property
    def identity_path(self) -> Path:
        return self.artifact_root / "source.json"

    @property
    def output_dir(self) -> Path:
        return self.artifact_root

    @property
    def candidates_path(self) -> Path:
        return self.artifact_root / "shuttle" / "candidates.pilot.jsonl"

    @property
    def frozen_candidates_path(self) -> Path:
        return self.artifact_root / "shuttle" / "candidates.frozen.jsonl"

    @property
    def person_tracks_path(self) -> Path:
        return self.artifact_root / "players" / "person_tracks.jsonl"

    @property
    def assignments_path(self) -> Path:
        return self.artifact_root / "players" / "assignments.jsonl"

    @property
    def pose_cache_path(self) -> Path:
        return self.artifact_root / "players" / "poses.jsonl"

    @property
    def calibration_path(self) -> Path:
        return self.artifact_root / "court" / "calibration.json"

    @property
    def rally_predictions_path(self) -> Path:
        return self.artifact_root / "rallies" / "predictions.jsonl"

    @property
    def rally_proposals_path(self) -> Path:
        return self.artifact_root / "rallies" / "proposals.json"

    def candidate_path(self, stage: str) -> Path:
        if stage == "pilot":
            return self.candidates_path
        if stage == "frozen":
            return self.frozen_candidates_path
        raise ValueError(f"unknown candidate stage: {stage}")


@dataclass(frozen=True)
class SourceStatus:
    source_id: str
    enabled: bool
    video: bool
    identity: bool
    candidates: bool
    frozen_candidates: bool
    person_tracks: bool
    assignments: bool
    pose_cache: bool
    calibration: bool
    problems: tuple[str, ...]

    @property
    def annotation_ready(self) -> bool:
        return self.enabled and self.video and self.identity and (self.frozen_candidates or self.candidates)

    @property
    def experiment_ready(self) -> bool:
        return (
            self.annotation_ready
            and self.frozen_candidates
            and self.person_tracks
            and self.assignments
            and self.pose_cache
            and self.calibration
            and not self.problems
        )

    @property
    def rally_ready(self) -> bool:
        """Rally inference deliberately has no shuttle dependency."""
        rally_problems = tuple(
            problem for problem in self.problems
            if not problem.startswith(("candidate artifact", "frozen candidate artifact"))
        )
        return (
            self.enabled and self.video and self.identity and self.person_tracks
            and self.assignments and self.pose_cache and self.calibration
            and not rally_problems
        )


class SourceCatalog:
    def __init__(self, path: Path, sources: Iterable[SourceEntry] = ()):
        self.path = Path(path).expanduser().resolve()
        self.sources = {source.source_id: source for source in sources}

    @classmethod
    def read(cls, path: str | Path) -> "SourceCatalog":
        path = Path(path).expanduser().resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != CATALOG_SCHEMA or value.get("schema_version") != 1:
            raise ValueError("unsupported source catalog schema")
        raw_sources = value.get("sources")
        if not isinstance(raw_sources, list):
            raise ValueError("source catalog requires a sources list")
        sources = []
        for raw in raw_sources:
            source_id = str(raw["source_id"])
            # schema v1 originally allowed every artifact path to be specified.
            # Read its output_dir as the artifact root, but never retain the
            # competing per-artifact path authorities when writing again.
            root_value = raw.get("artifact_root", raw.get("output_dir"))
            if root_value is None:
                raise ValueError(f"source {source_id} has no artifact_root")
            sources.append(
                SourceEntry(
                    source_id=source_id,
                    video_path=_resolve(path.parent, raw["video_path"]),
                    artifact_root=_resolve(path.parent, root_value),
                    enabled=bool(raw.get("enabled", True)),
                )
            )
        if len({source.source_id for source in sources}) != len(sources):
            raise ValueError("source catalog contains duplicate source IDs")
        return cls(path, sources)

    def add(
        self,
        video_path: str | Path,
        *,
        source_id: str | None = None,
        output_dir: str | Path | None = None,
    ) -> SourceEntry:
        video = Path(video_path).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f"video does not exist: {video}")
        source_id = source_id or video.stem
        if not _SOURCE_ID.fullmatch(source_id):
            raise ValueError("source ID may contain only letters, digits, '.', '_' and '-'")
        identity = inspect_source(source_id, video)
        if source_id in self.sources:
            current = self.sources[source_id]
            if (
                current.video_path == video
                and current.identity_path.is_file()
                and SourceIdentity.read(current.identity_path) == identity
            ):
                return current
            raise ValueError(f"source ID already refers to a different video: {source_id}")
        for current in self.sources.values():
            if (
                current.identity_path.is_file()
                and SourceIdentity.read(current.identity_path).video_sha256 == identity.video_sha256
            ):
                raise ValueError(f"video is already registered as {current.source_id}")
        output = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else (self.path.parent.parent / "outputs" / source_id).resolve()
        )
        entry = SourceEntry(
            source_id=source_id,
            video_path=video,
            artifact_root=output,
        )
        identity.write(entry.identity_path)
        self.sources[source_id] = entry
        self.write()
        return entry

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": CATALOG_SCHEMA,
            "schema_version": CATALOG_VERSION,
            "sources": [self._mapping(source) for source in self.sources.values()],
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _mapping(self, source: SourceEntry) -> dict[str, Any]:
        parent = self.path.parent
        return {
            "source_id": source.source_id,
            "enabled": source.enabled,
            "video_path": _portable_path(source.video_path, parent),
            "artifact_root": _portable_path(source.artifact_root, parent),
        }

    def select(self, source_ids: Iterable[str] | None = None) -> list[SourceEntry]:
        if source_ids is None:
            return [source for source in self.sources.values() if source.enabled]
        selected = []
        for source_id in source_ids:
            if source_id not in self.sources:
                raise KeyError(f"unknown source: {source_id}")
            selected.append(self.sources[source_id])
        return selected

    def status(self, source: SourceEntry) -> SourceStatus:
        problems = []
        identity = None
        if source.identity_path.is_file():
            try:
                identity = SourceIdentity.read(source.identity_path)
                if identity.source_id != source.source_id:
                    problems.append("source identity ID differs from catalog")
                if source.video_path.is_file() and _sha256(source.video_path) != identity.video_sha256:
                    problems.append("video fingerprint changed")
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                problems.append("source identity is invalid")
        if source.calibration_path.is_file():
            try:
                calibration = json.loads(source.calibration_path.read_text(encoding="utf-8"))
                if "image_size" not in calibration:
                    problems.append("calibration has no image_size")
                metadata = calibration.get("artifact_metadata", {})
                if identity is not None and (
                    calibration.get("image_size") != [identity.width, identity.height]
                    or metadata.get("source_id") != source.source_id
                    or metadata.get("source_identity_sha256") != identity.fingerprint
                    or metadata.get("frame_indexing") != "source_local_zero_based"
                ):
                    problems.append("calibration source identity/geometry differs")
            except (OSError, json.JSONDecodeError):
                problems.append("calibration is not valid JSON")
        assignments_valid = source.assignments_path.is_file()
        if assignments_valid and source.calibration_path.is_file():
            try:
                first_line = source.assignments_path.open(encoding="utf-8").readline()
                assignment_metadata = json.loads(first_line)
                if (
                    assignment_metadata.get("source_id") != source.source_id
                    or (
                        identity is not None
                        and assignment_metadata.get("source_identity_sha256")
                        != identity.fingerprint
                    )
                    or assignment_metadata.get("frame_indexing")
                    != "source_local_zero_based"
                    or assignment_metadata.get("calibration_sha256")
                    != _sha256(source.calibration_path)
                ):
                    assignments_valid = False
                    problems.append("player assignments source/calibration lineage differs")
            except (OSError, json.JSONDecodeError):
                assignments_valid = False
                problems.append("player assignments metadata is invalid")
        candidates_valid = source.candidates_path.is_file()
        frozen_valid = source.frozen_candidates_path.is_file()
        for path, label in (
            (source.candidates_path, "candidate artifact"),
            (source.frozen_candidates_path, "frozen candidate artifact"),
        ):
            if not path.is_file():
                continue
            try:
                metadata = json.loads(path.open(encoding="utf-8").readline())
                artifact_count = metadata.get("source_frame_count", metadata.get("frame_count"))
                if metadata.get("schema") != "shuttle_candidates":
                    raise ValueError("wrong schema")
                if identity is not None and artifact_count is not None and int(artifact_count) != identity.frame_count:
                    raise ValueError("frame count differs")
                if identity is not None and metadata.get("source_frame_range") not in (None, [0, identity.frame_count - 1]):
                    raise ValueError("frame range differs")
            except (OSError, ValueError, json.JSONDecodeError):
                if path == source.candidates_path:
                    candidates_valid = False
                else:
                    frozen_valid = False
                problems.append(f"{label} metadata is invalid or stale")
        person_tracks_valid = source.person_tracks_path.is_file()
        if person_tracks_valid and identity is not None:
            try:
                metadata = json.loads(source.person_tracks_path.open(encoding="utf-8").readline())
                if metadata.get("schema") != "person_tracks" or int(metadata.get("frame_count", -1)) != identity.frame_count:
                    raise ValueError("person-track identity differs")
            except (OSError, ValueError, json.JSONDecodeError):
                person_tracks_valid = False
                problems.append("person-track metadata is invalid or stale")
        pose_valid = source.pose_cache_path.is_file()
        if pose_valid and person_tracks_valid:
            try:
                metadata = json.loads(source.pose_cache_path.open(encoding="utf-8").readline())
                expected = f"sha256:{_sha256(source.person_tracks_path)}"
                if metadata.get("schema") != "pose_cache" or metadata.get("raw_artifact_fingerprint") != expected:
                    raise ValueError("pose input fingerprint differs")
            except (OSError, ValueError, json.JSONDecodeError):
                pose_valid = False
                problems.append("pose artifact metadata is invalid or stale")
        return SourceStatus(
            source_id=source.source_id,
            enabled=source.enabled,
            video=source.video_path.is_file(),
            identity=identity is not None,
            candidates=candidates_valid,
            frozen_candidates=frozen_valid,
            person_tracks=person_tracks_valid,
            assignments=assignments_valid,
            pose_cache=pose_valid,
            calibration=source.calibration_path.is_file(),
            problems=tuple(problems),
        )

    def write_annotation_config(
        self, output: str | Path, *, source_ids: Iterable[str] | None = None, stage: str = "frozen"
    ) -> Path:
        output = Path(output).expanduser().resolve()
        rows = []
        for source in self.select(source_ids):
            candidate = source.candidate_path(stage)
            if stage == "frozen" and not candidate.is_file() and source.candidates_path.is_file():
                candidate = source.candidates_path
            if not source.video_path.is_file() or not candidate.is_file():
                raise ValueError(f"source is not annotation-ready: {source.source_id}")
            rows.append(
                {
                    "source_id": source.source_id,
                    "video_path": _portable_path(source.video_path, output.parent),
                    "candidates_path": _portable_path(candidate, output.parent),
                }
            )
        if not rows:
            raise ValueError("no annotation-ready sources selected")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"sources": rows}, indent=2) + "\n", encoding="utf-8")
        return output

    def write_experiment(
        self,
        output: str | Path,
        *,
        runtime: str | Path,
        seed: int = DEFAULT_SEED,
        source_ids: Iterable[str] | None = None,
    ) -> tuple[Path, Path]:
        output = Path(output).expanduser().resolve()
        runtime = Path(runtime).expanduser().resolve()
        selected = self.select(source_ids)
        not_ready = [s.source_id for s in selected if not self.status(s).experiment_ready]
        if not_ready:
            raise ValueError(f"sources are not experiment-ready: {', '.join(not_ready)}")
        if len(selected) < 2:
            raise ValueError("an experiment requires at least two sources")
        annotations = runtime / "events" / "shuttle.jsonl"
        rallies = runtime / "rallies.csv"
        rally_manifest = runtime / "rallies.manifest.json"
        required = [annotations, rallies, rally_manifest]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError("missing experiment inputs: " + ", ".join(missing))
        queue_names = (
            "shuttle-adaptive.json",
            "shuttle-audit.json",
            "shuttle-rally-audit.json",
            "shuttle-refill.json",
        )
        queues = [runtime / "queues" / name for name in queue_names if (runtime / "queues" / name).is_file()]
        if not queues:
            raise ValueError("runtime has no annotation queues")
        manifest_value = _loso_manifest([source.source_id for source in selected], seed)
        manifest_path = output.with_name(output.stem + "-loso-manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest_value, indent=2) + "\n", encoding="utf-8")
        value = {
            "schema": "temporal_selector_experiment_config",
            "schema_version": 1,
            "seed": seed,
            "crossfit_manifest_path": _portable_path(manifest_path, output.parent),
            "dataset": {
                "minimum_cutoff": 0.05,
                "retention_k": 8,
                "pose_visibility_threshold": 0.5,
                "expected_annotation_sha256": _sha256(annotations),
                "annotations_path": _portable_path(annotations, output.parent),
                "rally_intervals_path": _portable_path(rallies, output.parent),
                "rally_manifest_path": _portable_path(rally_manifest, output.parent),
                "queue_paths": [_portable_path(path, output.parent) for path in queues],
                "sources": [
                    {
                        "source_id": source.source_id,
                        "video_path": _portable_path(source.video_path, output.parent),
                        "candidates_path": _portable_path(source.frozen_candidates_path, output.parent),
                        "assignments_path": _portable_path(source.assignments_path, output.parent),
                        "pose_cache_path": _portable_path(source.pose_cache_path, output.parent),
                        "calibration_path": _portable_path(source.calibration_path, output.parent),
                    }
                    for source in selected
                ],
            },
        }
        output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return output, manifest_path
