"""Court-space player-anchor features for shot classification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from .court_projection import HALF_LENGTH, HALF_WIDTH, CourtHomography
except ImportError:  # Support direct execution from src/.
    from court_projection import HALF_LENGTH, HALF_WIDTH, CourtHomography


LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


@dataclass(frozen=True)
class Calibration:
    homography: CourtHomography
    image_size: tuple[int, int]


class CalibrationRegistry:
    """Resolve clip names to source-video calibrations."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("version") != 1:
            raise ValueError(f"unsupported calibration registry version: {data.get('version')}")
        self.sources = {
            str(key).lower(): value for key, value in data.get("sources", {}).items()
        }
        self.clip_overrides = {
            str(key).lower(): str(value).lower()
            for key, value in data.get("clip_overrides", {}).items()
        }
        self._calibration_cache: dict[str, Calibration] = {}
        if not self.sources:
            raise ValueError("calibration registry must define at least one source")

    @staticmethod
    def _clip_name(name: str) -> str:
        name = Path(name).stem.lower()
        for suffix in ("_pose", "_features"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name

    def source_for_clip(self, clip_name: str) -> str:
        clip = self._clip_name(clip_name)
        if clip in self.clip_overrides:
            source = self.clip_overrides[clip]
            if source not in self.sources:
                raise ValueError(
                    f"clip override for {clip!r} references unknown source {source!r}"
                )
            return source

        matches = [
            source
            for source in self.sources
            if re.search(rf"(?<![a-z0-9]){re.escape(source)}(?![a-z0-9])", clip)
        ]
        if len(matches) != 1:
            reason = "no source ID" if not matches else f"ambiguous source IDs {matches}"
            raise ValueError(f"{reason} found in clip name {clip!r}")
        return matches[0]

    def calibration_for_source(self, source: str) -> Calibration:
        source = source.lower()
        if source in self._calibration_cache:
            return self._calibration_cache[source]
        if source not in self.sources:
            raise ValueError(f"unknown calibration source {source!r}")
        entry = self.sources[source]
        calibration_value = (
            entry.get("calibration") if isinstance(entry, Mapping) else entry
        )
        if not calibration_value:
            raise ValueError(f"source {source!r} has no calibration path")
        path = (self.path.parent / str(calibration_value)).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        size = data.get("image_size")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(not isinstance(value, int) or value <= 0 for value in size)
        ):
            raise ValueError(f"calibration {path} must contain a positive image_size")
        calibration = Calibration(CourtHomography.load(path), (size[0], size[1]))
        self._calibration_cache[source] = calibration
        return calibration

    def calibration_for_clip(self, clip_name: str) -> Calibration:
        return self.calibration_for_source(self.source_for_clip(clip_name))


def _visible_point(
    keypoints: Mapping[str, Mapping[str, float]],
    landmark_ids: Sequence[int],
    visibility_threshold: float,
) -> np.ndarray | None:
    points = []
    weights = []
    for landmark_id in landmark_ids:
        landmark = keypoints.get(str(landmark_id), {})
        visibility = float(landmark.get("visibility", 0.0))
        x = landmark.get("x")
        y = landmark.get("y")
        if (
            visibility >= visibility_threshold
            and x is not None
            and y is not None
            and np.isfinite([x, y]).all()
        ):
            points.append([float(x), float(y)])
            weights.append(visibility)
    if not points:
        return None
    return np.average(np.asarray(points), axis=0, weights=np.asarray(weights))


def observed_court_anchor(
    keypoints: Mapping[str, Mapping[str, float]],
    calibration: Calibration,
    visibility_threshold: float = 0.5,
) -> np.ndarray | None:
    """Return normalized court x/y from visible ground-contact landmarks."""

    foot_specs = (
        ((LEFT_HEEL, LEFT_FOOT_INDEX), LEFT_ANKLE),
        ((RIGHT_HEEL, RIGHT_FOOT_INDEX), RIGHT_ANKLE),
    )
    image_anchors = []
    for ground_ids, ankle_id in foot_specs:
        point = _visible_point(keypoints, ground_ids, visibility_threshold)
        if point is None:
            point = _visible_point(keypoints, (ankle_id,), visibility_threshold)
        if point is not None:
            image_anchors.append(point)
    if not image_anchors:
        return None

    projected = calibration.homography.project_normalized_to_court(
        image_anchors, calibration.image_size
    )
    anchor = projected.mean(axis=0)
    return anchor / np.asarray([HALF_WIDTH, HALF_LENGTH])


def build_court_anchor_features(
    frames: Sequence[Mapping],
    calibration: Calibration,
    num_frames: int,
    max_missing_gap: int = 5,
    visibility_threshold: float = 0.5,
) -> np.ndarray:
    """Build [court_x, court_y, observed] features with bounded imputation."""

    coordinates = np.full((num_frames, 2), np.nan, dtype=np.float32)
    observed = np.zeros(num_frames, dtype=np.float32)
    for index, frame in enumerate(frames[:num_frames]):
        anchor = observed_court_anchor(
            frame.get("keypoints", {}), calibration, visibility_threshold
        )
        if anchor is not None:
            coordinates[index] = anchor
            observed[index] = 1.0

    valid_indices = np.flatnonzero(observed)
    if not len(valid_indices):
        raise ValueError("clip has no usable foot anchors")

    missing = observed == 0
    padded = np.pad(missing.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    gaps = [(start, end) for start, end in edges.reshape(-1, 2)]
    too_long = [(start, end) for start, end in gaps if end - start > max_missing_gap]
    if too_long:
        lengths = [end - start for start, end in too_long]
        raise ValueError(
            f"clip has missing anchor gaps longer than {max_missing_gap} frames: {lengths}"
        )

    frame_indices = np.arange(num_frames)
    for axis in range(2):
        coordinates[:, axis] = np.interp(
            frame_indices, valid_indices, coordinates[valid_indices, axis]
        )
    return np.column_stack([coordinates, observed]).astype(np.float32)
