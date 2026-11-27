"""Versioned rally feature views and their ordered inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

from src.court_projection import HALF_LENGTH, HALF_SINGLES_WIDTH, HALF_WIDTH

from .rally_audio import RALLY_AUDIO_FEATURE_NAMES

RALLY_FEATURE_SCHEMA: Final = "baddievision.rally.feature_views"
RALLY_FEATURE_VERSION: Final = 2
CANONICAL_RALLY_FEATURE_VIEW: Final = "full_audio_zscore_motion"
BASE_DISTANCE_RALLY_FEATURE_VIEW: Final = "full_audio_zscore_motion_base_distance"
BASE_RADIAL_RALLY_FEATURE_VIEW: Final = "full_audio_zscore_motion_base_radial"

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


MOTION_BASE_FEATURE_NAMES: Final = tuple(
    name.replace("_pose_", "_median_torso_pose_").replace("_court_", "_reliable_median_court_")
    for name in FULL_RALLY_FEATURE_NAMES
)
_POSE_START: Final = len(PLAYER_CONTEXT_FIELDS) * 2

_COURT_POSITION_INDICES: Final = tuple(
    index for index, name in enumerate(FULL_RALLY_FEATURE_NAMES)
    if name.endswith(("_court_x", "_court_y"))
)
MOTION_POSITION_INDICES: Final = _COURT_POSITION_INDICES + tuple(
    index for index, name in enumerate(FULL_RALLY_FEATURE_NAMES)
    if "_pose_" in name
)
MOTION_FEATURE_NAMES: Final = tuple(
    FULL_RALLY_FEATURE_NAMES[index].rsplit("_", 1)[0] + "_velocity_" +
    FULL_RALLY_FEATURE_NAMES[index][-1]
    for index in MOTION_POSITION_INDICES
)
BASE_DISTANCE_FEATURE_NAMES: Final = ("p1_base_distance", "p2_base_distance")
BASE_RADIAL_FEATURE_NAMES: Final = ("p1_base_radial_velocity", "p2_base_radial_velocity")
BASE_FEATURE_NAMES: Final = BASE_DISTANCE_FEATURE_NAMES + BASE_RADIAL_FEATURE_NAMES


# Two court coordinates per player, followed by 66 pose coordinates per player.
_MOTION_PLAYER_INDICES: Final = (0, 0, 1, 1) + (0,) * 66 + (1,) * 66


def _four_sample_slope(samples: Tensor) -> Tensor:
    return (3 * (samples[..., 3] - samples[..., 0]) + samples[..., 2] - samples[..., 1]) / 10


def rally_singles_base_features(
    values: Tensor, validity: Tensor, track_ids: Tensor, fps: float,
) -> tuple[Tensor, Tensor]:
    """Measure distance and its signed slope about each singles half-court midpoint."""
    positions = values[:, _COURT_POSITION_INDICES].reshape(-1, 2, 2)
    observed = validity[:, _COURT_POSITION_INDICES].reshape(-1, 2, 2).all(dim=-1)
    observed &= track_ids >= 0
    base = positions.new_tensor(((0.0, -0.5), (0.0, 0.5)))
    metres = (positions - base) * positions.new_tensor((HALF_WIDTH, HALF_LENGTH))
    distance = metres.norm(dim=-1) / (HALF_SINGLES_WIDTH**2 + (HALF_LENGTH / 2)**2)**0.5
    radial_velocity = torch.zeros_like(distance)
    radial_valid = torch.zeros_like(observed)
    if len(distance) >= 4:
        tracks = track_ids.unfold(0, 4, 1)
        radial_valid[3:] = (
            observed.unfold(0, 4, 1).all(dim=-1) & (tracks == tracks[..., :1]).all(dim=-1)
        )
        slope = _four_sample_slope(distance.unfold(0, 4, 1)) * fps
        radial_velocity[3:] = torch.where(radial_valid[3:], slope, 0.0)
    return (
        torch.cat((torch.where(observed, distance, 0.0), radial_velocity), dim=-1),
        torch.cat((observed, radial_valid), dim=-1),
    )


def normalize_rally_motion_pose(
    values: Tensor, validity: Tensor, track_ids: Tensor, image_size: tuple[int, int],
) -> None:
    """Normalize image pose in place using one median torso length per player track."""
    poses = values[:, _POSE_START:].reshape(-1, 2, 33, 2)
    valid = validity[:, _POSE_START:].reshape_as(poses)
    poses.mul_(poses.new_tensor(image_size))
    visible = valid.all(dim=-1)
    hips = poses[:, :, [23, 24]].mean(dim=-2, keepdim=True)
    shoulders = poses[:, :, [11, 12]].mean(dim=-2, keepdim=True)
    torso = (shoulders - hips).norm(dim=-1).squeeze(-1)
    has_hips = visible[:, :, 23] & visible[:, :, 24]
    has_torso = has_hips & visible[:, :, 11] & visible[:, :, 12] & (torso > 1.0)
    for player in range(2):
        scale = torch.zeros_like(torso[:, player])
        player_tracks = track_ids[:, player]
        for track in player_tracks[has_torso[:, player] & (player_tracks >= 0)].unique():
            frames = player_tracks == track
            scale[frames] = torso[:, player][frames & has_torso[:, player]].quantile(0.5)
        usable = has_hips[:, player] & (scale > 0)
        valid[:, player] &= usable[:, None, None]
        relative = (poses[:, player] - hips[:, player]) / scale.clamp_min(1)[:, None, None]
        poses[:, player] = torch.where(valid[:, player], relative, 0.0)


def smooth_rally_court_positions(
    values: Tensor, validity: Tensor, track_ids: Tensor, reliable_feet: Tensor,
) -> None:
    """Apply a centered three-sample median within a continuously observed track."""
    if len(values) < 3:
        return

    court = values[:, _COURT_POSITION_INDICES]
    samples = court.unfold(0, 3, 1)
    observed = validity[:, _COURT_POSITION_INDICES].unfold(0, 3, 1).all(dim=-1)
    tracks = track_ids.unfold(0, 3, 1)
    same_track = (tracks == tracks[..., :1]).all(dim=-1) & (tracks[..., 0] >= 0)
    reliable_window = reliable_feet.unfold(0, 3, 1).all(dim=-1)
    usable = observed & (same_track & reliable_window).repeat_interleave(2, dim=-1)
    values[1:-1, _COURT_POSITION_INDICES] = torch.where(
        usable, samples.median(dim=-1).values, court[1:-1]
    )


def rally_motion_features(
    values: Tensor, validity: Tensor, track_ids: Tensor, fps: float,
) -> tuple[Tensor, Tensor]:
    """Fit trailing slopes over continuously observed coordinates on the same track."""
    positions = values[:, MOTION_POSITION_INDICES]
    observed = validity[:, MOTION_POSITION_INDICES]
    velocity = torch.zeros_like(positions)
    valid = torch.zeros_like(observed)

    # Court motion uses four samples. Joints use three to retain faster motion.
    for columns, count in ((slice(0, 4), 4), (slice(4, None), 3)):
        if len(positions) < count:
            continue
        tracks = track_ids.unfold(0, count, 1)
        same_track = (tracks == tracks[..., :1]).all(dim=-1) & (tracks[..., 0] >= 0)
        valid[count - 1:, columns] = (
            observed[:, columns].unfold(0, count, 1).all(dim=-1)
            & same_track[:, _MOTION_PLAYER_INDICES[columns]]
        )
        samples = positions[:, columns].unfold(0, count, 1)
        slope = (samples[..., -1] - samples[..., 0]) / 2
        if count == 4:
            slope = _four_sample_slope(samples)
        velocity[count - 1:, columns] = torch.where(valid[count - 1:, columns], slope * fps, 0.0)
    return velocity, valid


@dataclass(frozen=True)
class RallyFeatureView:
    """A stable ordered projection of visual features and optional audio features."""

    name: str
    names: tuple[str, ...]
    full_indices: tuple[int, ...]
    uses_audio: bool = False
    audio_zscore: bool = False
    uses_motion: bool = False
    uses_singles_base: bool = False

    @property
    def dimension(self) -> int:
        return len(self.names)

    def select(self, values: Tensor, validity: Tensor) -> tuple[Tensor, Tensor]:
        if values.shape != validity.shape:
            raise ValueError("rally values and validity must have identical shapes")
        input_dimension = len(FULL_RALLY_FEATURE_NAMES)
        if self.uses_motion:
            input_dimension += len(MOTION_FEATURE_NAMES)
        if self.uses_singles_base:
            input_dimension += len(BASE_FEATURE_NAMES)
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


def _view(
    name: str, names: Sequence[str], *, uses_audio: bool = False, audio_zscore: bool = False,
    uses_motion: bool = False, uses_singles_base: bool = False,
) -> RallyFeatureView:
    requested = tuple(names)
    if len(requested) != len(set(requested)):
        raise AssertionError(f"duplicate feature in rally view {name}")
    input_names = (
        (MOTION_BASE_FEATURE_NAMES + MOTION_FEATURE_NAMES if uses_motion else FULL_RALLY_FEATURE_NAMES)
        + (BASE_FEATURE_NAMES if uses_singles_base else ())
        + (RALLY_AUDIO_FEATURE_NAMES if uses_audio else ())
    )
    positions = {feature: index for index, feature in enumerate(input_names)}
    missing = [feature for feature in requested if feature not in positions]
    if missing:
        raise AssertionError(f"unknown feature in rally view {name}: {missing}")
    return RallyFeatureView(
        name, requested, tuple(positions[item] for item in requested), uses_audio, audio_zscore,
        uses_motion, uses_singles_base,
    )


_MOTION_VISUAL_FEATURE_NAMES: Final = MOTION_BASE_FEATURE_NAMES + MOTION_FEATURE_NAMES


RALLY_FEATURE_VIEWS: Final[Mapping[str, RallyFeatureView]] = {
    "full": _view("full", FULL_RALLY_FEATURE_NAMES),
    "full_motion": _view(
        "full_motion", _MOTION_VISUAL_FEATURE_NAMES, uses_motion=True
    ),
    "full_audio_zscore_motion": _view(
        "full_audio_zscore_motion",
        _MOTION_VISUAL_FEATURE_NAMES + RALLY_AUDIO_FEATURE_NAMES,
        uses_audio=True, audio_zscore=True, uses_motion=True,
    ),
    BASE_DISTANCE_RALLY_FEATURE_VIEW: _view(
        BASE_DISTANCE_RALLY_FEATURE_VIEW,
        _MOTION_VISUAL_FEATURE_NAMES + BASE_DISTANCE_FEATURE_NAMES + RALLY_AUDIO_FEATURE_NAMES,
        uses_audio=True, audio_zscore=True, uses_motion=True, uses_singles_base=True,
    ),
    BASE_RADIAL_RALLY_FEATURE_VIEW: _view(
        BASE_RADIAL_RALLY_FEATURE_VIEW,
        _MOTION_VISUAL_FEATURE_NAMES + BASE_FEATURE_NAMES + RALLY_AUDIO_FEATURE_NAMES,
        uses_audio=True, audio_zscore=True, uses_motion=True, uses_singles_base=True,
    ),
    "full_audio": _view(
        "full_audio", FULL_RALLY_FEATURE_NAMES + RALLY_AUDIO_FEATURE_NAMES, uses_audio=True
    ),
    "full_audio_zscore": _view(
        "full_audio_zscore", FULL_RALLY_FEATURE_NAMES + RALLY_AUDIO_FEATURE_NAMES,
        uses_audio=True, audio_zscore=True,
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
assert CANONICAL_RALLY_FEATURE_VIEW in RALLY_FEATURE_VIEWS
assert {name: view.dimension for name, view in RALLY_FEATURE_VIEWS.items()} == {
    "full": 154,
    "full_motion": 290,
    "full_audio_zscore_motion": 297,
    BASE_DISTANCE_RALLY_FEATURE_VIEW: 299,
    BASE_RADIAL_RALLY_FEATURE_VIEW: 301,
    "full_audio": 161,
    "full_audio_zscore": 161,
    "calibrated_canonical": 142,
    "no_position": 138,
    "image_position": 150,
}
