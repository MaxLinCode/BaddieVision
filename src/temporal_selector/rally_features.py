"""Versioned rally feature views and their ordered inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

from .rally_audio import RALLY_AUDIO_FEATURE_NAMES

RALLY_FEATURE_SCHEMA: Final = "baddievision.rally.feature_views"
RALLY_FEATURE_VERSION: Final = 2

PLAYER_CONTEXT_FIELDS: Final = (
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "foot_x",
    "foot_y",
    "court_x",
    "court_y",
    "detection_confidence",
    "assignment_confidence",
    "activity",
)
FULL_RALLY_FEATURE_NAMES: Final = tuple(
    f"{role}_{field}"
    for role in ("p1", "p2")
    for field in PLAYER_CONTEXT_FIELDS
) + tuple(
    f"{role}_pose_{landmark}_{axis}"
    for role in ("p1", "p2")
    for landmark in range(33)
    for axis in ("x", "y")
)


@dataclass(frozen=True)
class RallyFeatureView:
    """A stable ordered projection of visual features and optional audio features."""

    name: str
    names: tuple[str, ...]
    full_indices: tuple[int, ...]
    uses_audio: bool = False

    @property
    def dimension(self) -> int:
        return len(self.names)

    def select(self, values: Tensor, validity: Tensor) -> tuple[Tensor, Tensor]:
        if values.shape != validity.shape:
            raise ValueError("rally values and validity must have identical shapes")
        input_dimension = len(FULL_RALLY_FEATURE_NAMES)
        if self.uses_audio:
            input_dimension += len(RALLY_AUDIO_FEATURE_NAMES)
        if values.shape[-1] != input_dimension:
            raise ValueError("feature-view projection received an incompatible rally vector")
        indices = torch.tensor(self.full_indices, dtype=torch.long, device=values.device)
        return values.index_select(-1, indices), validity.index_select(-1, indices)


def _names_for_fields(fields: Sequence[str], *, include_pose: bool = True) -> tuple[str, ...]:
    selected = tuple(
        f"{role}_{field}" for role in ("p1", "p2") for field in fields
    )
    if include_pose:
        selected += tuple(
            name for name in FULL_RALLY_FEATURE_NAMES if "_pose_" in name
        )
    return selected


def _view(name: str, names: Sequence[str], *, uses_audio: bool = False) -> RallyFeatureView:
    requested = tuple(names)
    if len(requested) != len(set(requested)):
        raise AssertionError(f"duplicate feature in rally view {name}")
    input_names = FULL_RALLY_FEATURE_NAMES + (RALLY_AUDIO_FEATURE_NAMES if uses_audio else ())
    positions = {feature: index for index, feature in enumerate(input_names)}
    missing = [feature for feature in requested if feature not in positions]
    if missing:
        raise AssertionError(f"unknown feature in rally view {name}: {missing}")
    return RallyFeatureView(name, requested, tuple(positions[item] for item in requested), uses_audio)


RALLY_FEATURE_VIEWS: Final[Mapping[str, RallyFeatureView]] = {
    "full": _view("full", FULL_RALLY_FEATURE_NAMES),
    "full_audio": _view(
        "full_audio", FULL_RALLY_FEATURE_NAMES + RALLY_AUDIO_FEATURE_NAMES, uses_audio=True
    ),
    "calibrated_canonical": _view(
        "calibrated_canonical",
        _names_for_fields(
            (
                "court_x",
                "court_y",
                "detection_confidence",
                "assignment_confidence",
                "activity",
            )
        ),
    ),
    "no_position": _view(
        "no_position",
        _names_for_fields(
            ("detection_confidence", "assignment_confidence", "activity")
        ),
    ),
    "image_position": _view(
        "image_position",
        _names_for_fields(
            (
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "foot_x",
                "foot_y",
                "detection_confidence",
                "assignment_confidence",
                "activity",
            )
        ),
    ),
}

RALLY_PROBE_GROUPS: Final[Mapping[str, tuple[str, ...]]] = {
    "full": FULL_RALLY_FEATURE_NAMES,
    "canonical": RALLY_FEATURE_VIEWS["calibrated_canonical"].names,
    "court_only": _names_for_fields(("court_x", "court_y"), include_pose=False),
    "image_position_only": _names_for_fields(
        ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "foot_x", "foot_y"),
        include_pose=False,
    ),
    "confidence_validity_only": _names_for_fields(
        ("detection_confidence", "assignment_confidence", "activity"),
        include_pose=False,
    ),
    "relative_pose_only": tuple(
        name for name in FULL_RALLY_FEATURE_NAMES if "_pose_" in name
    ),
}


def rally_feature_view(name: str) -> RallyFeatureView:
    try:
        return RALLY_FEATURE_VIEWS[name]
    except KeyError as error:
        raise ValueError(f"unknown rally feature view: {name}") from error


def rally_feature_indices(names: Sequence[str]) -> tuple[int, ...]:
    positions = {name: index for index, name in enumerate(FULL_RALLY_FEATURE_NAMES)}
    try:
        return tuple(positions[name] for name in names)
    except KeyError as error:
        raise ValueError(f"unknown rally feature: {error.args[0]}") from error


assert len(FULL_RALLY_FEATURE_NAMES) == 154
assert {name: view.dimension for name, view in RALLY_FEATURE_VIEWS.items()} == {
    "full": 154,
    "full_audio": 161,
    "calibrated_canonical": 142,
    "no_position": 138,
    "image_position": 150,
}
