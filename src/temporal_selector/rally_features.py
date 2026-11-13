"""Versioned rally feature views over one canonical full-frame vector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import torch
from torch import Tensor


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
    """A stable ordered projection of the full rally feature vector."""

    name: str
    names: tuple[str, ...]
    full_indices: tuple[int, ...]

    @property
    def dimension(self) -> int:
        return len(self.names)

    def select(self, values: Tensor, validity: Tensor) -> tuple[Tensor, Tensor]:
        if values.shape != validity.shape:
            raise ValueError("rally values and validity must have identical shapes")
        if values.shape[-1] != len(FULL_RALLY_FEATURE_NAMES):
            raise ValueError("feature-view projection requires the full rally vector")
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


def _view(name: str, names: Sequence[str]) -> RallyFeatureView:
    requested = tuple(names)
    if len(requested) != len(set(requested)):
        raise AssertionError(f"duplicate feature in rally view {name}")
    positions = {feature: index for index, feature in enumerate(FULL_RALLY_FEATURE_NAMES)}
    missing = [feature for feature in requested if feature not in positions]
    if missing:
        raise AssertionError(f"unknown feature in rally view {name}: {missing}")
    return RallyFeatureView(name, requested, tuple(positions[item] for item in requested))


RALLY_FEATURE_VIEWS: Final[Mapping[str, RallyFeatureView]] = {
    "full": _view("full", FULL_RALLY_FEATURE_NAMES),
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
    "calibrated_canonical": 142,
    "no_position": 138,
    "image_position": 150,
}
