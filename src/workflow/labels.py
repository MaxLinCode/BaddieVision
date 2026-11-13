"""Candidate-independent shuttle label projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.annotation_platform.events import AnnotationEvent


CANONICAL_VISIBILITY = ("visible", "occluded", "out_of_frame", "uncertain")


@dataclass(frozen=True)
class CanonicalShuttleLabel:
    source_id: str
    frame: int
    visibility: str
    position_normalized: tuple[float, float] | None
    position_kind: str | None
    coordinate_status: str
    candidate_provenance: Mapping[str, Any] | None = None

    @property
    def trains_selector(self) -> bool:
        return self.visibility == "visible" and self.position_normalized is not None


def _canonical_position(event: AnnotationEvent) -> tuple[float, float] | None:
    snapshot = event.candidate_position or {}
    value = snapshot.get("peak_position_normalized")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return float(value[0]), float(value[1])


def canonicalize_shuttle_event(event: AnnotationEvent) -> CanonicalShuttleLabel:
    """Project current/legacy events without making candidate ID the target."""
    provenance = {
        "candidate_id": event.candidate_id,
        "candidate_artifact_sha256": event.candidate_artifact_sha256,
    }
    if event.label_kind == "selected":
        position = _canonical_position(event)
        return CanonicalShuttleLabel(
            event.source_id,
            event.frame,
            "visible",
            position,
            "observed" if position is not None else None,
            "available" if position is not None else "legacy_unavailable",
            provenance,
        )
    if event.label_kind == "missing_proposal":
        position = _canonical_position(event)
        # Old events asserted a visible target but offered no way to click its
        # canonical location. Preserve that audit evidence while new events
        # carry a reusable position.
        return CanonicalShuttleLabel(
            event.source_id,
            event.frame,
            "visible",
            position,
            "observed" if position is not None else None,
            "available" if position is not None else "legacy_unavailable",
            provenance,
        )
    if event.label_kind in {"occluded_inferable"}:
        return CanonicalShuttleLabel(
            event.source_id, event.frame, "occluded", None, None, "unavailable", provenance
        )
    if event.label_kind in {"no_in_frame_target", "no_shuttle"}:
        return CanonicalShuttleLabel(
            event.source_id, event.frame, "out_of_frame", None, None, "not_applicable", provenance
        )
    if event.label_kind == "unsure":
        return CanonicalShuttleLabel(
            event.source_id, event.frame, "uncertain", None, None, "not_applicable", provenance
        )
    raise ValueError(f"unsupported shuttle event label: {event.label_kind}")


def match_candidate_by_position(
    label: CanonicalShuttleLabel,
    candidates: list[Mapping[str, Any]],
    *,
    maximum_distance: float = 0.02,
) -> int | None:
    """Match in normalized image space, independently of candidate IDs."""
    if label.position_normalized is None:
        return None
    x, y = label.position_normalized
    matches: list[tuple[float, str, int]] = []
    for index, candidate in enumerate(candidates):
        point = candidate.get("peak_position_normalized")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            point = candidate.get("weighted_centroid_normalized")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        distance = ((float(point[0]) - x) ** 2 + (float(point[1]) - y) ** 2) ** 0.5
        if distance <= maximum_distance:
            matches.append((distance, str(candidate.get("candidate_id", "")), index))
    return min(matches)[2] if matches else None
