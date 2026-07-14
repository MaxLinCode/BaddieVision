"""Shuttle candidate-selection task plugin."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from .core import AnnotationSuggestion, LabelOption, SourceRegistration, file_sha256


SHUTTLE_TASK = "shuttle_selection"
WRITABLE_SHUTTLE_LABELS = (
    "selected",
    "missing_proposal",
    "occluded_inferable",
    "no_in_frame_target",
    "unsure",
    "undo",
)
LEGACY_SHUTTLE_LABELS = ("no_shuttle",)
SHUTTLE_LABELS = WRITABLE_SHUTTLE_LABELS + LEGACY_SHUTTLE_LABELS
GROUPING_VERSION = "peak-contained-high-to-low-v1"
SELECTOR_NO_SHUTTLE = "NO_SHUTTLE"
LEGACY_EXTRACTION_VERSION = "tracknet-components-v2.0"
LEGACY_POLICY = {
    "model_stage": "tracknet_pre_inpaint",
    "threshold": 0.5,
    "threshold_comparator": ">",
    "contours": {"retrieval": "RETR_EXTERNAL", "approximation": "CHAIN_APPROX_SIMPLE"},
    "selection": "first_largest_bounding_box_area_strict_greater_than",
}


def _point(candidate: Mapping[str, Any], field: str) -> tuple[float, float] | None:
    value = candidate.get(field)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _bbox_contains_peak(group: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    bbox = candidate.get("bbox")
    peak = _point(group, "peak_position") or _point(group, "center")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or peak is None:
        return False
    try:
        left, top, right, bottom = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return False
    return left <= peak[0] < right and top <= peak[1] < bottom


def _centroid_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a = _point(left, "weighted_centroid") or _point(left, "center")
    b = _point(right, "weighted_centroid") or _point(right, "center")
    if a is None or b is None:
        return float("inf")
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def group_shuttle_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Group threshold components without bbox-union bridges.

    Thresholds are processed from high to low. A lower-threshold component can
    join only one existing group whose fixed representative peak is inside its
    bbox, with nearest weighted centroid and stable candidate ID as tie-breaks.
    A group can contain at most one member from any threshold.
    """
    ordered = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda item: (
            -float(item.get("threshold", 0.0)),
            str(item.get("candidate_id", "")),
        ),
    )
    groups: list[dict[str, Any]] = []
    for candidate in ordered:
        threshold = float(candidate.get("threshold", 0.0))
        compatible = [
            group
            for group in groups
            if threshold not in group["thresholds"]
            and _bbox_contains_peak(group["representative"], candidate)
        ]
        if compatible:
            group = min(
                compatible,
                key=lambda item: (
                    _centroid_distance(item["representative"], candidate),
                    str(item["representative"].get("candidate_id", "")),
                ),
            )
            group["members"].append(candidate)
            group["thresholds"].add(threshold)
        else:
            groups.append(
                {
                    "representative": candidate,
                    "members": [candidate],
                    "thresholds": {threshold},
                }
            )
    output: list[Mapping[str, Any]] = []
    for group in groups:
        representative = dict(group["representative"])
        members = sorted(
            group["members"], key=ShuttleSelectionPlugin._representative_key
        )
        member_ids = [str(item["candidate_id"]) for item in members]
        representative.update(
            {
                "candidate_group_id": "group:" + str(representative["candidate_id"]),
                "grouping_version": GROUPING_VERSION,
                "grouped_candidate_ids": member_ids,
                "raw_member_ids": member_ids,
                "grouped_candidate_count": len(member_ids),
            }
        )
        output.append(representative)
    return tuple(sorted(output, key=ShuttleSelectionPlugin._representative_key))


def selector_training_target(label_kind: str) -> str | None:
    """Map an annotation label to selector supervision; ``None`` means masked."""
    label_kind = str(label_kind).lower()
    if label_kind == "selected":
        return "SELECTED_PROPOSAL"
    if label_kind == "no_in_frame_target":
        return SELECTOR_NO_SHUTTLE
    if label_kind in {"missing_proposal", "occluded_inferable", "unsure", "no_shuttle"}:
        return None
    raise ValueError(f"unsupported selector label: {label_kind}")


@dataclass(frozen=True)
class ShuttleCandidateArtifact:
    path: Path
    sha256: str
    size: int
    mtime_ns: int
    metadata: Mapping[str, Any]
    frames: Mapping[int, tuple[Mapping[str, Any], ...]]
    candidates: Mapping[str, tuple[int, Mapping[str, Any]]]
    frozen_frames: Mapping[int, tuple[Mapping[str, Any], ...]]

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
        if metadata.get("schema") != "shuttle_candidates" or metadata.get(
            "schema_version"
        ) not in {1, 2}:
            raise ValueError(f"expected shuttle_candidates schema v1/v2: {path}")
        frames: dict[int, tuple[Mapping[str, Any], ...]] = {}
        candidates: dict[str, tuple[int, Mapping[str, Any]]] = {}
        frozen_frames: dict[int, tuple[Mapping[str, Any], ...]] = {}
        frame_order: list[int] = []
        for line_number, raw_line in enumerate(lines[1:], start=2):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid candidate JSON at {path}:{line_number}"
                ) from exc
            if record.get("type") != "frame":
                raise ValueError(
                    f"candidate artifact contains non-frame record at line {line_number}"
                )
            frame = int(record["frame"])
            if frame in frames:
                raise ValueError(f"duplicate candidate frame: {frame}")
            frame_order.append(frame)
            items = tuple(record.get("candidates", ()))
            frozen_items = record.get("frozen_candidates")
            if frozen_items is not None:
                if not isinstance(frozen_items, list):
                    raise ValueError(
                        f"frozen candidates must be a list on frame {frame}"
                    )
                frozen_frames[frame] = tuple(frozen_items)
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
                raise ValueError(
                    "schema-v2 candidates require source_frame_range [start, end]"
                )
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
            frozen_frames,
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
            LabelOption("occluded_inferable", "Occluded / inferable", "i"),
            LabelOption("no_in_frame_target", "No in-frame target", "n"),
            LabelOption("missing_proposal", "Missing proposal", "m"),
            LabelOption("unsure", "Unsure", "u"),
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
                or tuple(int(value) for value in artifact_image_size)
                != source.image_size
            ):
                raise ValueError(
                    f"candidate/source image_size mismatch for {source.source_id}: "
                    f"{artifact_image_size} != {list(source.image_size)}"
                )
        invalid_frames = [
            frame for frame in artifact.frames if not 0 <= frame < source.frame_count
        ]
        if invalid_frames:
            raise ValueError(
                f"candidate frames outside source bounds: {invalid_frames[:3]}"
            )
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
            raise ValueError(
                "label candidate-artifact SHA-256 does not match the registered artifact"
            )
        if label_kind not in SHUTTLE_LABELS:
            raise ValueError(f"invalid shuttle label kind: {label_kind}")
        if label_kind in LEGACY_SHUTTLE_LABELS:
            raise ValueError(
                f"legacy shuttle label {label_kind!r} is readable but cannot be written"
            )
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

    def resolve_candidate_position(
        self,
        source: SourceRegistration,
        *,
        frame: int,
        candidate_id: str | None,
        candidate_artifact_sha256: str,
    ) -> Mapping[str, object]:
        """Snapshot normalized coordinates from the fingerprint-verified artifact."""
        self.validate_label(
            source,
            frame=frame,
            label_kind="selected",
            candidate_id=candidate_id,
            candidate_artifact_sha256=candidate_artifact_sha256,
        )
        artifact = self._artifact(source, verify_fingerprint=True)
        candidate = artifact.candidates[str(candidate_id)][1]
        image_width, image_height = source.image_size

        def normalized(normalized_field: str, pixel_field: str) -> object:
            value = candidate.get(normalized_field)
            if value is not None:
                return value
            pixel_value = candidate.get(pixel_field)
            if not isinstance(pixel_value, (list, tuple)) or len(pixel_value) != 2:
                return None
            try:
                return [
                    float(pixel_value[0]) / image_width,
                    float(pixel_value[1]) / image_height,
                ]
            except (TypeError, ValueError, ZeroDivisionError):
                return None

        return {
            "coordinate_space": "normalized_image_xy",
            "canonical_field": "peak_position_normalized",
            "peak_position_normalized": normalized(
                "peak_position_normalized", "peak_position"
            ),
            "weighted_centroid_normalized": normalized(
                "weighted_centroid_normalized", "weighted_centroid"
            ),
            "center_normalized": normalized("center_normalized", "center"),
        }

    def overlays(
        self, source: SourceRegistration, frame: int
    ) -> Sequence[Mapping[str, Any]]:
        artifact = self._artifact(source)
        frame = int(frame)
        if frame not in artifact.frames:
            raise ValueError(
                f"candidate artifact has no frame record for {source.source_id}:{frame}"
            )
        return artifact.frames[frame]

    @staticmethod
    def _representative_key(
        candidate: Mapping[str, Any],
    ) -> tuple[float, float, float, float, str]:
        return (
            -float(candidate.get("threshold", 0.0)),
            -float(candidate.get("peak_activation", candidate.get("peak_value", 0.0))),
            -float(candidate.get("mean_activation", 0.0)),
            -float(candidate.get("area", 0.0)),
            str(candidate.get("candidate_id", "")),
        )

    def annotator_overlays(
        self, source: SourceRegistration, frame: int
    ) -> Sequence[Mapping[str, Any]]:
        """Return deterministic non-transitive cross-threshold proposal groups."""
        artifact = self._artifact(source)
        if "frozen_candidate_view" in artifact.metadata:
            try:
                return artifact.frozen_frames[int(frame)]
            except KeyError as exc:
                raise ValueError(
                    f"frozen artifact has no candidate view for {source.source_id}:{frame}"
                ) from exc
        return group_shuttle_candidates(self.overlays(source, frame))

    def display_overlays(
        self,
        source: SourceRegistration,
        frame: int,
        *,
        view: str = "grouped",
        minimum_threshold: float | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        candidates = tuple(
            candidate
            for candidate in self.overlays(source, frame)
            if minimum_threshold is None
            or float(candidate.get("threshold", 1.0)) >= float(minimum_threshold)
        )
        if view == "raw":
            return candidates
        if view != "grouped":
            raise ValueError("candidate view must be 'grouped' or 'raw'")
        artifact = self._artifact(source)
        if "frozen_candidate_view" in artifact.metadata:
            if minimum_threshold is not None:
                raise ValueError(
                    "frozen grouped views do not support runtime threshold filtering"
                )
            return self.annotator_overlays(source, frame)
        return group_shuttle_candidates(candidates)

    def representative_candidate_id(
        self, source: SourceRegistration, frame: int, candidate_id: str
    ) -> str:
        for candidate in self.annotator_overlays(source, frame):
            if candidate_id in candidate.get("grouped_candidate_ids", ()):
                return str(candidate["candidate_id"])
        return candidate_id

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

    def suggestion(
        self, source: SourceRegistration, frame: int
    ) -> AnnotationSuggestion | None:
        """Return the frozen legacy TrackNet decision only for verified v2 artifacts."""
        artifact = self._artifact(source)
        metadata = artifact.metadata
        hashes = (
            metadata.get("checkpoint_sha256"),
            metadata.get("inference_model_sha256"),
        )
        valid_hashes = all(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdefABCDEF" for char in value)
            for value in hashes
        )
        if not (
            metadata.get("schema_version") == 2
            and metadata.get("extraction_version") == LEGACY_EXTRACTION_VERSION
            and metadata.get("model_stage") == LEGACY_POLICY["model_stage"]
            and metadata.get("threshold_comparator") == ">"
            and metadata.get("legacy_compatibility_threshold") == 0.5
            and 0.5 in metadata.get("thresholds", [])
            and metadata.get("provenance_verified") is True
            and not metadata.get("nonproduction_unverified_provenance", False)
            and valid_hashes
        ):
            return None
        candidates = artifact.frames.get(int(frame))
        if candidates is None:
            return None
        marked = [
            item for item in candidates if item.get("legacy_largest_component") is True
        ]
        threshold_candidates = [
            item for item in candidates if item.get("threshold") == 0.5
        ]
        if len(marked) > 1 or any(item.get("threshold") != 0.5 for item in marked):
            return None
        if threshold_candidates and len(marked) != 1:
            return None
        candidate_id = str(marked[0]["candidate_id"]) if marked else None
        return AnnotationSuggestion(
            provider="legacy_tracknet",
            semantic_label="selected" if candidate_id else "no_in_frame_target",
            candidate_id=candidate_id,
            policy=LEGACY_POLICY,
            artifact_fingerprints={
                "candidate_artifact_sha256": artifact.sha256,
                "checkpoint_sha256": str(hashes[0]),
                "inference_model_sha256": str(hashes[1]),
            },
            verified=True,
            metadata={
                "opencv_version": cv2.__version__,
                "extraction_implementation_version": metadata["extraction_version"],
                "inference_model_artifact": metadata.get("inference_model_artifact"),
                "overlap_ensemble_mode": metadata.get("overlap_ensemble_mode"),
            },
        )
