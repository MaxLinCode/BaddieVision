"""Canonical local source catalog and generated downstream configurations."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

CATALOG_SCHEMA = "baddievision_source_catalog"
CATALOG_VERSION = 1
DEFAULT_SEED = 1729
_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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
    output_dir: Path
    candidates_path: Path
    frozen_candidates_path: Path
    assignments_path: Path
    pose_cache_path: Path
    calibration_path: Path
    enabled: bool = True
    video_sha256: str | None = None

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
    candidates: bool
    frozen_candidates: bool
    assignments: bool
    pose_cache: bool
    calibration: bool
    problems: tuple[str, ...]

    @property
    def annotation_ready(self) -> bool:
        return self.enabled and self.video and (self.frozen_candidates or self.candidates)

    @property
    def experiment_ready(self) -> bool:
        return (
            self.annotation_ready
            and self.frozen_candidates
            and self.assignments
            and self.pose_cache
            and self.calibration
            and not self.problems
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
            output_dir = _resolve(path.parent, raw["output_dir"])
            artifacts = raw.get("artifacts", {})
            sources.append(
                SourceEntry(
                    source_id=source_id,
                    video_path=_resolve(path.parent, raw["video_path"]),
                    output_dir=output_dir,
                    candidates_path=_resolve(
                        path.parent,
                        artifacts.get("candidates", output_dir / "shuttle_candidates_pilot.jsonl"),
                    ),
                    frozen_candidates_path=_resolve(
                        path.parent,
                        artifacts.get("frozen_candidates", output_dir / "shuttle_candidates_frozen.jsonl"),
                    ),
                    assignments_path=_resolve(
                        path.parent,
                        artifacts.get("assignments", output_dir / "player_assignments.jsonl"),
                    ),
                    pose_cache_path=_resolve(
                        path.parent,
                        artifacts.get("pose_cache", output_dir / "pose_cache.jsonl"),
                    ),
                    calibration_path=_resolve(
                        path.parent,
                        artifacts.get(
                            "calibration",
                            path.parent.parent / "features" / "court" / f"{source_id}.json",
                        ),
                    ),
                    enabled=bool(raw.get("enabled", True)),
                    video_sha256=raw.get("video_sha256"),
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
        calibration_path: str | Path | None = None,
    ) -> SourceEntry:
        video = Path(video_path).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f"video does not exist: {video}")
        source_id = source_id or video.stem
        if not _SOURCE_ID.fullmatch(source_id):
            raise ValueError("source ID may contain only letters, digits, '.', '_' and '-'")
        digest = _sha256(video)
        if source_id in self.sources:
            current = self.sources[source_id]
            if current.video_path == video and current.video_sha256 == digest:
                return current
            raise ValueError(f"source ID already refers to a different video: {source_id}")
        for current in self.sources.values():
            if current.video_sha256 == digest:
                raise ValueError(f"video is already registered as {current.source_id}")
        output = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else (self.path.parent.parent / "outputs" / source_id).resolve()
        )
        calibration = (
            Path(calibration_path).expanduser().resolve()
            if calibration_path is not None
            else (self.path.parent.parent / "features" / "court" / f"{source_id}.json").resolve()
        )
        entry = SourceEntry(
            source_id=source_id,
            video_path=video,
            output_dir=output,
            candidates_path=output / "shuttle_candidates_pilot.jsonl",
            frozen_candidates_path=output / "shuttle_candidates_frozen.jsonl",
            assignments_path=output / "player_assignments.jsonl",
            pose_cache_path=output / "pose_cache.jsonl",
            calibration_path=calibration,
            video_sha256=digest,
        )
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
            "video_sha256": source.video_sha256,
            "output_dir": _portable_path(source.output_dir, parent),
            "artifacts": {
                "candidates": _portable_path(source.candidates_path, parent),
                "frozen_candidates": _portable_path(source.frozen_candidates_path, parent),
                "assignments": _portable_path(source.assignments_path, parent),
                "pose_cache": _portable_path(source.pose_cache_path, parent),
                "calibration": _portable_path(source.calibration_path, parent),
            },
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
        if source.video_path.is_file() and source.video_sha256:
            if _sha256(source.video_path) != source.video_sha256:
                problems.append("video fingerprint changed")
        if source.calibration_path.is_file():
            try:
                calibration = json.loads(source.calibration_path.read_text(encoding="utf-8"))
                if "image_size" not in calibration:
                    problems.append("calibration has no image_size")
            except (OSError, json.JSONDecodeError):
                problems.append("calibration is not valid JSON")
        return SourceStatus(
            source_id=source.source_id,
            enabled=source.enabled,
            video=source.video_path.is_file(),
            candidates=source.candidates_path.is_file(),
            frozen_candidates=source.frozen_candidates_path.is_file(),
            assignments=source.assignments_path.is_file(),
            pose_cache=source.pose_cache_path.is_file(),
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
