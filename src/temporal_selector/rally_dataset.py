"""Candidate-free dataset adapter for the independent rally segmenter."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .batch import MASKED_TARGET
from .dataset import (
    POSE_REPRESENTATION_VERSION,
    _frame_context,
    _jsonl,
    _sha256,
    _validate_calibration,
    rally_boundary_targets,
)
from src.court_projection import CourtHomography, HALF_LENGTH, HALF_WIDTH
from src.workflow.identity import SourceIdentity
from .rally import (
    RALLY_FEATURE_SCHEMA,
    RALLY_FEATURE_VERSION,
    RallyBatch,
)
from .rally_features import (
    FULL_RALLY_FEATURE_NAMES,
    rally_feature_view,
)
from .rally_intervals import read_rally_intervals


RALLY_FEATURE_NAMES = FULL_RALLY_FEATURE_NAMES


@dataclass(frozen=True)
class RallySourceConfig:
    source_id: str
    video_path: Path
    assignments_path: Path
    pose_cache_path: Path
    calibration_path: Path

    def __post_init__(self) -> None:
        for name in (
            "video_path",
            "assignments_path",
            "pose_cache_path",
            "calibration_path",
        ):
            object.__setattr__(
                self, name, Path(getattr(self, name)).expanduser().resolve()
            )


@dataclass(frozen=True)
class RallyDataConfig:
    sources: tuple[RallySourceConfig, ...]
    rally_intervals_path: Path | None
    rally_manifest_path: Path | None
    pose_visibility_threshold: float = 0.5
    pose_coordinate_mode: str = "player_relative"
    context_seconds: float = 1.0
    ownership_seconds: float = 1.0
    feature_view: str = "full"
    cached_court_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if len({source.source_id for source in self.sources}) != len(self.sources):
            raise ValueError("rally source IDs must be unique")
        if self.pose_coordinate_mode != "player_relative":
            raise ValueError(
                "the independent rally feature schema requires player_relative pose"
            )
        if not 0.0 <= self.pose_visibility_threshold <= 1.0:
            raise ValueError("pose visibility threshold must be in [0, 1]")
        if self.context_seconds <= 0 or self.ownership_seconds <= 0:
            raise ValueError("rally context and ownership durations must be positive")
        rally_feature_view(self.feature_view)
        if self.cached_court_tolerance < 0:
            raise ValueError("cached court-coordinate tolerance cannot be negative")
        if (self.rally_intervals_path is None) != (self.rally_manifest_path is None):
            raise ValueError("rally interval CSV and manifest must be supplied together")
        for name in ("rally_intervals_path", "rally_manifest_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())


@dataclass(frozen=True)
class RallyWindow:
    source_id: str
    window_id: str
    anchor_frame: int
    frame_indices: tuple[int, ...]
    owned_frames: frozenset[int]
    relative_time_seconds: torch.Tensor
    frame_values: torch.Tensor
    frame_validity: torch.Tensor
    inplay_targets: torch.Tensor
    rally_start_targets: torch.Tensor
    rally_end_targets: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RallyWindowDataset(Dataset[RallyWindow]):
    """Compile dense source-local windows without loading shuttle artifacts."""

    def __init__(self, config: RallyDataConfig):
        self.config = config
        self.rally_index = (
            read_rally_intervals(config.rally_intervals_path, config.rally_manifest_path)
            if config.rally_intervals_path is not None
            and config.rally_manifest_path is not None
            else None
        )
        self.feature_view = rally_feature_view(config.feature_view)
        self.windows = self._compile()
        annotation_identity = (
            {
                "rally_intervals_sha256": self.rally_index.intervals_sha256,
                "rally_manifest_sha256": _sha256(config.rally_manifest_path),
            }
            if self.rally_index is not None and config.rally_manifest_path is not None
            else None
        )
        self.annotation_fingerprint = hashlib.sha256(
            json.dumps(
                annotation_identity, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest() if annotation_identity is not None else None
        identity = {
            "schema": "rally_window_dataset",
            "schema_version": 2,
            "rally_feature_schema": RALLY_FEATURE_SCHEMA,
            "rally_feature_version": RALLY_FEATURE_VERSION,
            "feature_view": self.feature_view.name,
            "feature_names": self.feature_view.names,
            "pose_coordinate_mode": config.pose_coordinate_mode,
            "pose_representation_version": POSE_REPRESENTATION_VERSION[
                config.pose_coordinate_mode
            ],
            "pose_visibility_threshold": config.pose_visibility_threshold,
            "context_seconds": config.context_seconds,
            "ownership_seconds": config.ownership_seconds,
            "annotation": annotation_identity,
            "sources": {
                source.source_id: {
                    "video_sha256": _sha256(source.video_path),
                    "assignments_sha256": _sha256(source.assignments_path),
                    "pose_cache_sha256": _sha256(source.pose_cache_path),
                    "calibration_sha256": _sha256(source.calibration_path),
                }
                for source in config.sources
            },
        }
        dataset_fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        owned_targets = [
            int(target)
            for window in self.windows
            for frame, target in zip(
                window.frame_indices, window.inplay_targets.tolist()
            )
            if frame in window.owned_frames
        ]
        self.manifest = {
            **identity,
            "dataset_fingerprint": dataset_fingerprint,
            "annotation_fingerprint": self.annotation_fingerprint,
            "window_count": len(self.windows),
            "source_window_counts": {
                source.source_id: sum(
                    window.source_id == source.source_id
                    for window in self.windows
                )
                for source in config.sources
            },
            "owned_target_counts": {
                "negative": owned_targets.count(0),
                "positive": owned_targets.count(1),
                "masked": owned_targets.count(MASKED_TARGET),
            },
        }

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> RallyWindow:
        return self.windows[index]

    def _compile(self) -> tuple[RallyWindow, ...]:
        source_ids = {source.source_id for source in self.config.sources}
        if self.rally_index is not None:
            unresolved = source_ids - set(self.rally_index.sources)
            if unresolved:
                raise ValueError(
                    f"rally manifest is missing sources: {sorted(unresolved)}"
                )
        return tuple(
            window
            for source in self.config.sources
            for window in self._compile_source(source)
        )

    def _compile_source(
        self, source: RallySourceConfig
    ) -> list[RallyWindow]:
        assignment_meta, assignment_records = _jsonl(source.assignments_path)
        pose_meta, pose_records = _jsonl(source.pose_cache_path)
        frame_count = int(assignment_meta["frame_count"])
        fps = float(assignment_meta["fps"])
        image_size = tuple(map(int, assignment_meta["frame_size"]))
        if frame_count <= 0 or fps <= 0:
            raise ValueError("rally source frame count and FPS must be positive")
        assignments = {
            int(record["frame"]): record for record in assignment_records
        }
        if sorted(assignments) != list(range(frame_count)):
            raise ValueError(
                "player assignments must be a complete zero-based source sequence"
            )
        if pose_meta.get("raw_artifact_fingerprint") != assignment_meta.get(
            "raw_artifact_fingerprint"
        ):
            raise ValueError(
                "pose-cache lineage does not match player assignments"
            )
        for key in ("pose_model_fingerprint", "preprocessing_fingerprint"):
            if not str(pose_meta.get(key, "")).startswith("sha256:"):
                raise ValueError(f"pose cache has invalid {key}")
        poses = {
            (int(record["frame"]), int(record["track_id"])): record
            for record in pose_records
        }
        calibration_sha256 = _sha256(source.calibration_path)
        identity_path = source.calibration_path.parent.parent / "source.json"
        if source.calibration_path.name != "calibration.json" or source.calibration_path.parent.name != "court":
            raise ValueError("rally calibration must use the canonical court/calibration.json path")
        if not identity_path.is_file():
            raise FileNotFoundError(f"source identity is required: {identity_path}")
        identity = SourceIdentity.read(identity_path)
        if identity.source_id != source.source_id:
            raise ValueError("source identity does not match rally source ID")
        identity_fps = identity.fps_numerator / identity.fps_denominator
        if (
            identity.frame_indexing != "source_local_zero_based"
            or (identity.width, identity.height) != image_size
            or identity.frame_count != frame_count
            or not math.isclose(identity_fps, fps, rel_tol=0.0, abs_tol=1e-6)
        ):
            raise ValueError("source identity geometry does not match player assignments")
        calibration_value = json.loads(source.calibration_path.read_text(encoding="utf-8"))
        calibration_metadata = calibration_value.get("artifact_metadata", {})
        if (
            calibration_metadata.get("source_id") != source.source_id
            or calibration_metadata.get("source_identity_sha256") != identity.fingerprint
        ):
            raise ValueError("calibration source identity is missing or ambiguous")
        if (
            assignment_meta.get("source_id") != source.source_id
            or assignment_meta.get("source_identity_sha256") != identity.fingerprint
            or assignment_meta.get("calibration_sha256") != calibration_sha256
        ):
            raise ValueError("player assignments do not fingerprint the canonical calibration lineage")
        _validate_calibration(source.calibration_path, image_size)
        homography = CourtHomography.load(source.calibration_path)

        capture = cv2.VideoCapture(str(source.video_path))
        try:
            video_fps = float(capture.get(cv2.CAP_PROP_FPS))
            video_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_size = (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        finally:
            capture.release()
        if (
            not math.isclose(video_fps, fps, rel_tol=0, abs_tol=1e-6)
            or video_count != frame_count
            or video_size != image_size
        ):
            raise ValueError(
                "video geometry/frame sequence does not match player artifacts"
            )

        if self.rally_index is not None:
            source_manifest = self.rally_index.sources[source.source_id]
            if (
                source_manifest.video_sha256 != _sha256(source.video_path)
                or not math.isclose(
                    source_manifest.fps, fps, rel_tol=0, abs_tol=1e-6
                )
                or source_manifest.frame_count != frame_count
            ):
                raise ValueError("rally annotation provenance does not match source")
            if identity.video_sha256 != source_manifest.video_sha256:
                raise ValueError("source identity video fingerprint does not match rally annotations")

        ownership = max(1, round(fps * self.config.ownership_seconds))
        radius = max(1, round(fps * self.config.context_seconds))
        windows: list[RallyWindow] = []
        for owned_start in range(0, frame_count, ownership):
            owned = frozenset(
                range(owned_start, min(frame_count, owned_start + ownership))
            )
            anchor = sorted(owned)[len(owned) // 2]
            start = max(0, anchor - radius)
            end = min(frame_count - 1, anchor + radius)
            frames = tuple(range(start, end + 1))
            values: list[list[float]] = []
            validity: list[list[bool]] = []
            targets: list[int] = []
            for frame in frames:
                frame_values, frame_validity = _frame_context(
                    assignments[frame],
                    poses,
                    frame,
                    image_size,
                    "full_context",
                    self.config.pose_visibility_threshold,
                    self.config.pose_coordinate_mode,
                )
                for player_index, role in enumerate(("P1", "P2")):
                    item = assignments[frame].get("slots", {}).get(role, {}).get("assignment")
                    if not isinstance(item, Mapping):
                        continue
                    projected = homography.project_to_court([item["foot"]])[0]
                    cached = np.asarray([item.get("court_x"), item.get("court_y")], dtype=float)
                    if not np.isfinite(cached).all() or not np.allclose(
                        cached,
                        projected,
                        rtol=0.0,
                        atol=self.config.cached_court_tolerance,
                    ):
                        raise ValueError(
                            f"cached assignment court coordinates disagree with canonical calibration at frame {frame} {role}"
                        )
                    offset = player_index * 11
                    frame_values[offset + 6] = float(projected[0]) / HALF_WIDTH
                    frame_values[offset + 7] = float(projected[1]) / HALF_LENGTH
                selected_values, selected_validity = self.feature_view.select(
                    torch.tensor(frame_values, dtype=torch.float32),
                    torch.tensor(frame_validity, dtype=torch.bool),
                )
                values.append(selected_values.tolist())
                validity.append(selected_validity.tolist())
                targets.append(
                    self.rally_index.target(source.source_id, frame)
                    if self.rally_index is not None and frame in owned
                    else MASKED_TARGET
                )
            if self.rally_index is not None:
                start_targets, end_targets = rally_boundary_targets(
                    self.rally_index.for_source(source.source_id),
                    self.rally_index.partial_start_rally_ids,
                    frames,
                    owned,
                    targets,
                    fps=fps,
                )
            else:
                start_targets = torch.full((len(frames),), float(MASKED_TARGET))
                end_targets = torch.full((len(frames),), float(MASKED_TARGET))
            windows.append(
                RallyWindow(
                    source_id=source.source_id,
                    window_id=f"rally-{source.source_id}-{anchor:09d}",
                    anchor_frame=anchor,
                    frame_indices=frames,
                    owned_frames=owned,
                    relative_time_seconds=torch.tensor(
                        [(frame - anchor) / fps for frame in frames],
                        dtype=torch.float32,
                    ),
                    frame_values=torch.tensor(
                        values, dtype=torch.float32
                    ).reshape(len(frames), self.feature_view.dimension),
                    frame_validity=torch.tensor(
                        validity, dtype=torch.bool
                    ).reshape(len(frames), self.feature_view.dimension),
                    inplay_targets=torch.tensor(targets, dtype=torch.long),
                    rally_start_targets=start_targets,
                    rally_end_targets=end_targets,
                    metadata={
                        "fps": fps,
                        "frame_count": frame_count,
                        "image_size": list(image_size),
                        "feature_view": self.feature_view.name,
                        "calibration_sha256": calibration_sha256,
                        "owns_rally_boundary": any(
                            frame in owned
                            and (
                                float(start_targets[index]) == 1.0
                                or float(end_targets[index]) == 1.0
                            )
                            for index, frame in enumerate(frames)
                        ),
                        "owns_short_rally": self.rally_index is not None and any(
                            interval.contains(frame)
                            and (
                                interval.end_frame
                                - interval.start_frame
                                + 1
                            )
                            / fps
                            <= 1.0
                            for frame in owned
                            for interval in self.rally_index.for_source(
                                source.source_id
                            )
                        ),
                    },
                )
            )
        return windows


def collate_rally_windows(windows: Sequence[RallyWindow]) -> RallyBatch:
    if not windows:
        raise ValueError("cannot collate an empty rally batch")
    batch_size = len(windows)
    max_frames = max(len(window.frame_indices) for window in windows)
    feature_dims = {int(window.frame_values.shape[-1]) for window in windows}
    if len(feature_dims) != 1:
        raise ValueError("cannot collate rally windows from different feature views")
    feature_dim = feature_dims.pop()
    values = torch.zeros(batch_size, max_frames, feature_dim)
    validity = torch.zeros(
        batch_size, max_frames, feature_dim, dtype=torch.bool
    )
    times = torch.zeros(batch_size, max_frames)
    frame_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    inplay = torch.full(
        (batch_size, max_frames), MASKED_TARGET, dtype=torch.long
    )
    starts = torch.full(
        (batch_size, max_frames), float(MASKED_TARGET), dtype=torch.float32
    )
    ends = torch.full(
        (batch_size, max_frames), float(MASKED_TARGET), dtype=torch.float32
    )
    for batch_index, window in enumerate(windows):
        frame_count = len(window.frame_indices)
        if window.frame_values.shape != (frame_count, feature_dim):
            raise ValueError("rally window has an incompatible feature shape")
        values[batch_index, :frame_count] = window.frame_values
        validity[batch_index, :frame_count] = window.frame_validity
        times[batch_index, :frame_count] = window.relative_time_seconds
        frame_mask[batch_index, :frame_count] = True
        inplay[batch_index, :frame_count] = window.inplay_targets
        starts[batch_index, :frame_count] = window.rally_start_targets
        ends[batch_index, :frame_count] = window.rally_end_targets
    return RallyBatch(
        frame_values=values,
        frame_validity=validity,
        relative_time_seconds=times,
        frame_mask=frame_mask,
        inplay_targets=inplay,
        rally_start_targets=starts,
        rally_end_targets=ends,
    ).validate(frame_feature_dim=feature_dim)


def rally_data_config_from_selector_mapping(
    config: Mapping[str, Any], *, base_dir: Path
) -> RallyDataConfig:
    """Read only player/pose/rally paths from an existing experiment config."""
    dataset = config["dataset"]

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (base_dir / path).resolve()

    return RallyDataConfig(
        sources=tuple(
            RallySourceConfig(
                source_id=str(source["source_id"]),
                video_path=resolve(source["video_path"]),
                assignments_path=resolve(source["assignments_path"]),
                pose_cache_path=resolve(source["pose_cache_path"]),
                calibration_path=resolve(source["calibration_path"]),
            )
            for source in dataset["sources"]
        ),
        rally_intervals_path=resolve(dataset["rally_intervals_path"]),
        rally_manifest_path=resolve(dataset["rally_manifest_path"]),
        pose_visibility_threshold=float(
            dataset.get("pose_visibility_threshold", 0.5)
        ),
        feature_view=str(dataset.get("rally_feature_view", "full")),
    )
