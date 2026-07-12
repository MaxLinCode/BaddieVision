"""Shuttle candidate-selection task plugin."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import LabelOption, SourceRegistration, file_sha256


SHUTTLE_TASK = "shuttle_selection"
SHUTTLE_LABELS = ("selected", "no_shuttle", "missing_proposal", "skip", "undo")


@dataclass(frozen=True)
class ShuttleCandidateArtifact:
    path: Path
    sha256: str
    size: int
    mtime_ns: int
    metadata: Mapping[str, Any]
    frames: Mapping[int, tuple[Mapping[str, Any], ...]]
    candidates: Mapping[str, tuple[int, Mapping[str, Any]]]

    @classmethod
    def load(cls, path: str | Path) -> "ShuttleCandidateArtifact":
        path = Path(path).expanduser().resolve()
        raw = path.read_bytes()
        lines = raw.splitlines()
        if not lines:
            raise ValueError(f"empty candidate artifact: {path}")
        try:
            metadata = json.loads(lines[0])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid candidate metadata: {path}") from exc
        if metadata.get("schema") != "shuttle_candidates" or metadata.get("schema_version") not in {1, 2}:
            raise ValueError(f"expected shuttle_candidates schema v1/v2: {path}")
        frames: dict[int, tuple[Mapping[str, Any], ...]] = {}
        candidates: dict[str, tuple[int, Mapping[str, Any]]] = {}
        frame_order: list[int] = []
        for line_number, raw_line in enumerate(lines[1:], start=2):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid candidate JSON at {path}:{line_number}") from exc
            if record.get("type") != "frame":
                raise ValueError(f"candidate artifact contains non-frame record at line {line_number}")
            frame = int(record["frame"])
            if frame in frames:
                raise ValueError(f"duplicate candidate frame: {frame}")
            frame_order.append(frame)
            items = tuple(record.get("candidates", ()))
            for candidate in items:
                candidate_id = str(candidate.get("candidate_id", ""))
                if not candidate_id:
                    raise ValueError(f"candidate without ID on frame {frame}")
                if candidate_id in candidates:
                    raise ValueError(f"duplicate candidate ID: {candidate_id}")
                candidates[candidate_id] = (frame, candidate)
            frames[frame] = items
        if metadata.get("schema_version") == 2:
            declared_range = metadata.get("source_frame_range")
            if (
                not isinstance(declared_range, list)
                or len(declared_range) != 2
                or any(isinstance(value, bool) for value in declared_range)
            ):
                raise ValueError("schema-v2 candidates require source_frame_range [start, end]")
            frame_start, frame_end = (int(value) for value in declared_range)
            if frame_start < 0 or frame_end < frame_start:
                raise ValueError("invalid schema-v2 source_frame_range")
            expected_frames = list(range(frame_start, frame_end + 1))
            if frame_order != expected_frames:
                raise ValueError(
                    "schema-v2 candidate frame records must be ordered, contiguous, "
                    "and exactly match source_frame_range"
                )
        stat = path.stat()
        return cls(
            path,
            hashlib.sha256(raw).hexdigest(),
            stat.st_size,
            stat.st_mtime_ns,
            metadata,
            frames,
            candidates,
        )


class ShuttleSelectionPlugin:
    """Candidate attribution with explicit absent and proposal-failure labels."""

    task_name = SHUTTLE_TASK
    display_name = "Shuttle candidate selection"
    artifact_role = "candidates"

    def __init__(self) -> None:
        self._artifacts: dict[str, ShuttleCandidateArtifact] = {}

    @property
    def label_options(self) -> Sequence[LabelOption]:
        return (
            LabelOption("no_shuttle", "No shuttle", "n"),
            LabelOption("missing_proposal", "Missing proposal", "m"),
            LabelOption("skip", "Skip", "s"),
        )

    def prepare_source(self, source: SourceRegistration) -> None:
        try:
            path = source.artifacts[self.artifact_role]
        except KeyError as exc:
            raise ValueError(
                f"source {source.source_id!r} has no {self.artifact_role!r} artifact"
            ) from exc
        artifact = ShuttleCandidateArtifact.load(path)
        artifact_fps = artifact.metadata.get("fps")
        if artifact_fps is not None and abs(float(artifact_fps) - source.fps) > 1e-6:
            raise ValueError(
                f"candidate/source FPS mismatch for {source.source_id}: "
                f"{artifact_fps} != {source.fps}"
            )
        if artifact.metadata.get("schema_version") == 2:
            artifact_image_size = artifact.metadata.get("image_size")
            if (
                not isinstance(artifact_image_size, list)
                or len(artifact_image_size) != 2
                or tuple(int(value) for value in artifact_image_size) != source.image_size
            ):
                raise ValueError(
                    f"candidate/source image_size mismatch for {source.source_id}: "
                    f"{artifact_image_size} != {list(source.image_size)}"
                )
        invalid_frames = [frame for frame in artifact.frames if not 0 <= frame < source.frame_count]
        if invalid_frames:
            raise ValueError(f"candidate frames outside source bounds: {invalid_frames[:3]}")
        self._artifacts[source.source_id] = artifact

    def _artifact(
        self,
        source: SourceRegistration,
        *,
        verify_fingerprint: bool = False,
    ) -> ShuttleCandidateArtifact:
        try:
            artifact = self._artifacts[source.source_id]
        except KeyError as exc:
            raise RuntimeError(f"source was not prepared: {source.source_id}") from exc
        stat = artifact.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (artifact.size, artifact.mtime_ns):
            raise ValueError(
                f"candidate artifact changed after registration for {source.source_id}; "
                "restart with a new queue/session"
            )
        if verify_fingerprint:
            # Queue creation and label writes are trust boundaries. Read-only
            # overlay/score calls use the immutable registration snapshot so a
            # long video does not hash the artifact once per frame.
            current_sha = file_sha256(artifact.path)
            if current_sha != artifact.sha256:
                raise ValueError(
                    f"candidate artifact changed after registration for {source.source_id}; "
                    "restart with a new queue/session"
                )
        return artifact

    def artifact_sha256(self, source: SourceRegistration) -> str:
        return self._artifact(source).sha256

    def verify_artifact_fingerprint(self, source: SourceRegistration) -> str:
        return self._artifact(source, verify_fingerprint=True).sha256

    def validate_label(
        self,
        source: SourceRegistration,
        *,
        frame: int,
        label_kind: str,
        candidate_id: str | None,
        candidate_artifact_sha256: str,
    ) -> None:
        artifact = self._artifact(source, verify_fingerprint=True)
        if not 0 <= int(frame) < source.frame_count:
            raise ValueError(f"frame outside source bounds: {frame}")
        if candidate_artifact_sha256 != artifact.sha256:
            raise ValueError("label candidate-artifact SHA-256 does not match the registered artifact")
        if label_kind not in SHUTTLE_LABELS:
            raise ValueError(f"invalid shuttle label kind: {label_kind}")
        if label_kind == "selected":
            if not candidate_id:
                raise ValueError("selected labels require a candidate ID")
            resolved = artifact.candidates.get(candidate_id)
            if resolved is None:
                raise ValueError(f"candidate ID cannot be resolved: {candidate_id}")
            if resolved[0] != int(frame):
                raise ValueError(
                    f"candidate {candidate_id!r} belongs to frame {resolved[0]}, not {frame}"
                )
        elif candidate_id is not None:
            raise ValueError(f"{label_kind} labels cannot carry a candidate ID")

    def overlays(self, source: SourceRegistration, frame: int) -> Sequence[Mapping[str, Any]]:
        artifact = self._artifact(source)
        frame = int(frame)
        if frame not in artifact.frames:
            raise ValueError(
                f"candidate artifact has no frame record for {source.source_id}:{frame}"
            )
        return artifact.frames[frame]

    def eligible_frames(self, source: SourceRegistration) -> Sequence[int]:
        return tuple(self._artifact(source).frames)

    def queue_score(self, source: SourceRegistration, frame: int) -> float:
        """Bootstrap ambiguity score: crowded frames and weak evidence first."""
        candidates = self.overlays(source, frame)
        peaks = [
            float(
                candidate.get(
                    "peak_activation",
                    candidate.get("peak_value", candidate.get("peak", 0.0)),
                )
            )
            for candidate in candidates
        ]
        strongest = max(peaks, default=0.0)
        weak_evidence = 1.0 - min(1.0, max(0.0, strongest))
        return float(len(candidates)) + weak_evidence

    def image_size(self, source: SourceRegistration) -> tuple[int, int]:
        return source.image_size
