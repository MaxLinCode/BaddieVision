"""Validated annotation-window adapter for the temporal selector."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

from src.annotation_platform.events import AnnotationEvent, replay_events
from src.annotation_platform.queues import AnnotationQueue
from src.annotation_platform.shuttle import GROUPING_VERSION
from src.court_projection import HALF_LENGTH, HALF_WIDTH
from src.single_video.shuttle import (
    CANDIDATE_RETENTION_POLICY,
    read_shuttle_candidates,
)

from .batch import MASKED_TARGET, NULL_TARGET, SelectorBatch

FrameView = Literal["candidates_only", "players_court", "full_context"]
FRAME_DIMS = {"candidates_only": 0, "players_court": 30, "full_context": 162}
MASKED_LABELS = {
    "missing_proposal",
    "occluded_inferable",
    "unsure",
    "no_shuttle",
    "dropped_by_k",
}


def _frozen_candidate_frames(
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    retention_k: int,
) -> dict[int, tuple[Mapping[str, Any], ...]]:
    """Validate and return the artifact-owned grouped/ranked candidate view."""
    view = metadata.get("frozen_candidate_view")
    expected_view = {
        "schema_version": 1,
        "record_field": "frozen_candidates",
        "grouping_version": GROUPING_VERSION,
        "ordering_policy": list(CANDIDATE_RETENTION_POLICY),
        "retention_k": retention_k,
    }
    if view != expected_view:
        raise ValueError("candidate artifact has no compatible frozen candidate view")

    output: dict[int, tuple[Mapping[str, Any], ...]] = {}
    seen_group_ids: set[str] = set()
    for record in records:
        frame = int(record["frame"])
        raw = record.get("candidates")
        frozen = record.get("frozen_candidates")
        if not isinstance(raw, list) or not isinstance(frozen, list):
            raise ValueError(f"frame {frame} has an incomplete frozen candidate view")
        if len(frozen) > retention_k:
            raise ValueError(f"frame {frame} exceeds frozen retention K={retention_k}")
        raw_ids = {str(item.get("candidate_id", "")) for item in raw}
        if "" in raw_ids or len(raw_ids) != len(raw):
            raise ValueError(
                f"frame {frame} has invalid or duplicate raw candidate IDs"
            )
        covered_raw_ids: set[str] = set()
        for group in frozen:
            group_id = str(group.get("candidate_id", ""))
            members = group.get("raw_member_ids")
            if (
                not group_id
                or group_id in seen_group_ids
                or group.get("grouping_version") != GROUPING_VERSION
                or not isinstance(members, list)
                or not members
            ):
                raise ValueError(f"frame {frame} has an invalid frozen candidate group")
            member_ids = [str(item) for item in members]
            if len(set(member_ids)) != len(member_ids):
                raise ValueError(f"frame {frame} has duplicate frozen group members")
            if group_id not in member_ids or not set(member_ids) <= raw_ids:
                raise ValueError(
                    f"frame {frame} frozen group does not match raw candidates"
                )
            if covered_raw_ids.intersection(member_ids):
                raise ValueError(
                    f"frame {frame} assigns a raw candidate more than once"
                )
            covered_raw_ids.update(member_ids)
            seen_group_ids.add(group_id)
        if covered_raw_ids != raw_ids:
            raise ValueError(
                f"frame {frame} frozen view does not cover retained raw candidates"
            )
        output[frame] = tuple(frozen)
    return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("type") != "metadata":
        raise ValueError(f"artifact has no metadata record: {path}")
    return records[0], records[1:]


@dataclass(frozen=True)
class SelectorSourceConfig:
    source_id: str
    video_path: Path
    candidates_path: Path
    assignments_path: Path
    pose_cache_path: Path
    calibration_path: Path

    def __post_init__(self) -> None:
        for name in (
            "video_path",
            "candidates_path",
            "assignments_path",
            "pose_cache_path",
            "calibration_path",
        ):
            object.__setattr__(
                self, name, Path(getattr(self, name)).expanduser().resolve()
            )


@dataclass(frozen=True)
class SelectorDataConfig:
    sources: tuple[SelectorSourceConfig, ...]
    queue_paths: tuple[Path, ...]
    annotations_path: Path
    context_mode: FrameView = "full_context"
    minimum_cutoff: float = 0.05
    retention_k: int = 8
    pose_visibility_threshold: float = 0.5
    expected_annotation_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.context_mode not in FRAME_DIMS:
            raise ValueError(f"unsupported context mode: {self.context_mode}")
        if not math.isclose(float(self.minimum_cutoff), 0.05) or self.retention_k != 8:
            raise ValueError("this dataset version is frozen at cutoff 0.05 and K=8")
        if len({source.source_id for source in self.sources}) != len(self.sources):
            raise ValueError("selector source IDs must be unique")
        object.__setattr__(
            self,
            "queue_paths",
            tuple(Path(p).expanduser().resolve() for p in self.queue_paths),
        )
        object.__setattr__(
            self, "annotations_path", Path(self.annotations_path).expanduser().resolve()
        )


@dataclass(frozen=True)
class SelectorWindow:
    source_id: str
    burst_id: str
    queue_kind: str
    anchor_frame: int
    frame_indices: tuple[int, ...]
    owned_frames: frozenset[int]
    relative_time_seconds: torch.Tensor
    candidate_values: torch.Tensor
    candidate_validity: torch.Tensor
    candidate_frame_indices: torch.Tensor
    candidate_ids: tuple[str, ...]
    frame_values: torch.Tensor
    frame_validity: torch.Tensor
    targets: torch.Tensor
    target_status: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _candidate_values(candidate: Mapping[str, Any]) -> tuple[list[float], list[bool]]:
    fields = (
        ("weighted_centroid_normalized", 2),
        ("peak_position_normalized", 2),
        ("bbox_normalized", 4),
        ("area_normalized", 1),
        ("peak_activation_normalized", 1),
        ("mean_activation_normalized", 1),
        ("total_activation_normalized", 1),
    )
    values: list[float] = []
    valid: list[bool] = []
    for name, width in fields:
        raw = candidate.get(name)
        if width == 1:
            raw = [candidate.get(name, candidate.get(name.replace("_normalized", "")))]
        ok = isinstance(raw, (list, tuple)) and len(raw) == width
        parsed = [float(item) for item in raw] if ok else [0.0] * width
        ok = ok and all(math.isfinite(item) for item in parsed)
        values.extend(parsed if ok else [0.0] * width)
        valid.extend([ok] * width)
    return values, valid


def _calibration_values(path: Path, image_size: tuple[int, int]) -> list[float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if tuple(value.get("image_size", ())) != image_size:
        raise ValueError("calibration image_size does not match source artifacts")
    matrix = np.asarray(value.get("image_to_court"), dtype=float)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or abs(np.linalg.det(matrix)) < 1e-12
    ):
        raise ValueError("calibration homography is invalid")
    landmarks = value.get("image_landmarks", {})
    names = (
        "left_doubles_sideline__near_short_service",
        "left_doubles_sideline__far_short_service",
        "right_doubles_sideline__near_short_service",
        "right_doubles_sideline__far_short_service",
    )
    if any(name not in landmarks for name in names):
        raise ValueError("calibration is missing required court anchors")
    width, height = image_size
    return [
        coordinate / scale
        for name in names
        for coordinate, scale in zip(map(float, landmarks[name]), (width, height))
    ]


def _frame_context(
    assignment: Mapping[str, Any],
    poses: Mapping[tuple[int, int], Mapping[str, Any]],
    frame: int,
    image_size: tuple[int, int],
    court: Sequence[float],
    mode: FrameView,
    visibility_threshold: float,
) -> tuple[list[float], list[bool]]:
    if mode == "candidates_only":
        return [], []
    width, height = image_size
    values, validity = list(court), [True] * 8
    player_poses: list[tuple[list[float], list[bool]]] = []
    for role in ("P1", "P2"):
        slot = assignment.get("slots", {}).get(role, {})
        item = slot.get("assignment")
        if not isinstance(item, Mapping):
            values.extend([0.0] * 11)
            validity.extend([False] * 11)
            player_poses.append(([0.0] * 66, [False] * 66))
            continue
        bbox, foot = item.get("bbox"), item.get("foot")
        player = [
            float(bbox[0]) / width,
            float(bbox[1]) / height,
            float(bbox[2]) / width,
            float(bbox[3]) / height,
            float(foot[0]) / width,
            float(foot[1]) / height,
            float(item["court_x"]) / HALF_WIDTH,
            float(item["court_y"]) / HALF_LENGTH,
            float(item["detection_confidence"]),
            float(slot["confidence"]),
            float(item["activity"]),
        ]
        if not all(math.isfinite(value) for value in player):
            raise ValueError(f"non-finite player assignment at frame {frame}")
        values.extend(player)
        validity.extend([True] * 11)
        pose = poses.get((frame, int(item["track_id"])))
        pose_values, pose_valid = [0.0] * 66, [False] * 66
        if (
            pose is not None
            and pose.get("status") == "detected"
            and len(pose.get("pose_landmarks", ())) == 33
        ):
            pb = pose.get("pose_bbox")
            if not isinstance(pb, list) or len(pb) != 4:
                raise ValueError("accepted pose record has no pose_bbox")
            pose_values, pose_valid = [], []
            for landmark in pose["pose_landmarks"]:
                x = (
                    float(pb[0]) + float(landmark["x"]) * (float(pb[2]) - float(pb[0]))
                ) / width
                y = (
                    float(pb[1]) + float(landmark["y"]) * (float(pb[3]) - float(pb[1]))
                ) / height
                visible = (
                    math.isfinite(x)
                    and math.isfinite(y)
                    and float(landmark.get("visibility", 0)) >= visibility_threshold
                )
                pose_values.extend([x if visible else 0.0, y if visible else 0.0])
                pose_valid.extend([visible, visible])
        player_poses.append((pose_values, pose_valid))
    if mode == "full_context":
        for pose_values, pose_valid in player_poses:
            values.extend(pose_values)
            validity.extend(pose_valid)
    return values, validity


class SelectorWindowDataset(Dataset[SelectorWindow]):
    """Eagerly validates and exposes one two-second window per queue burst."""

    def __init__(self, config: SelectorDataConfig):
        self.config = config
        self.windows = self._compile()
        identity = {
            "schema": "selector_window_dataset",
            "schema_version": 1,
            "minimum_cutoff": config.minimum_cutoff,
            "retention_k": config.retention_k,
            "annotation_sha256": _sha256(config.annotations_path),
            "queue_sha256": [_sha256(path) for path in config.queue_paths],
            "candidate_sha256": {
                source.source_id: _sha256(source.candidates_path)
                for source in config.sources
            },
            "bursts": [
                [window.queue_kind, window.source_id, window.burst_id]
                for window in self.windows
            ],
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        statuses = [
            status
            for window in self.windows
            for frame, status in zip(window.frame_indices, window.target_status)
            if frame in window.owned_frames
        ]
        status_counts = dict(sorted(Counter(statuses).items()))
        self.manifest = {
            **identity,
            "dataset_fingerprint": fingerprint,
            "window_count": len(self.windows),
            "source_window_counts": {
                source.source_id: sum(
                    window.source_id == source.source_id for window in self.windows
                )
                for source in config.sources
            },
            "target_counts": {
                "selected_retained": statuses.count("selected_retained"),
                "null": statuses.count("null"),
                "masked": sum(
                    status not in {"selected_retained", "null"} for status in statuses
                ),
                "dropped_by_k": statuses.count("dropped_by_k"),
            },
            "target_reason_counts": status_counts,
        }

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> SelectorWindow:
        return self.windows[index]

    def _compile(self) -> tuple[SelectorWindow, ...]:
        if (
            self.config.expected_annotation_sha256
            and _sha256(self.config.annotations_path)
            != self.config.expected_annotation_sha256
        ):
            raise ValueError("annotation fingerprint mismatch")
        events = [
            AnnotationEvent.from_mapping(json.loads(line))
            for line in self.config.annotations_path.read_text().splitlines()
        ]
        active = replay_events(events).active
        queues = [AnnotationQueue.read(path) for path in self.config.queue_paths]
        bursts = [(queue.kind, burst) for queue in queues for burst in queue.bursts]
        if len({burst.burst_id for _, burst in bursts}) != len(bursts):
            raise ValueError("duplicate burst IDs across selector queues")
        source_map = {source.source_id: source for source in self.config.sources}
        windows = []
        for source_id in source_map:
            source_bursts = [
                (kind, burst) for kind, burst in bursts if burst.source_id == source_id
            ]
            windows.extend(
                self._compile_source(source_map[source_id], source_bursts, active)
            )
        unknown = {burst.source_id for _, burst in bursts} - set(source_map)
        if unknown:
            raise ValueError(f"queue references unknown source IDs: {sorted(unknown)}")
        return tuple(windows)

    def _compile_source(
        self, source: SelectorSourceConfig, bursts, active
    ) -> list[SelectorWindow]:
        candidate_meta, candidate_records = read_shuttle_candidates(
            source.candidates_path
        )
        if not math.isclose(
            float(candidate_meta.get("frozen_minimum_cutoff", -1)),
            self.config.minimum_cutoff,
        ):
            raise ValueError("candidate artifact does not carry the frozen cutoff")
        if int(candidate_meta.get("frozen_retention_k", -1)) != self.config.retention_k:
            raise ValueError("candidate artifact does not carry frozen K")
        if candidate_meta.get("grouping_version") != GROUPING_VERSION:
            raise ValueError("candidate grouping version mismatch")
        fps = float(candidate_meta["fps"])
        image_size = tuple(map(int, candidate_meta["image_size"]))
        candidate_frames = _frozen_candidate_frames(
            candidate_meta,
            candidate_records,
            retention_k=self.config.retention_k,
        )
        expected_frames = list(range(len(candidate_records)))
        if int(candidate_meta.get("source_frame_count", len(candidate_records))) != len(
            candidate_records
        ):
            raise ValueError("candidate metadata frame count mismatch")
        if sorted(candidate_frames) != expected_frames:
            raise ValueError(
                "candidate frame alignment is not a complete zero-based sequence"
            )
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
            or video_count != len(candidate_records)
            or video_size != image_size
        ):
            raise ValueError(
                "video FPS/frame count/image size does not align with candidate artifact"
            )
        assignment_meta, assignment_records = _jsonl(source.assignments_path)
        if (
            float(assignment_meta["fps"]) != fps
            or tuple(assignment_meta["frame_size"]) != image_size
            or int(assignment_meta["frame_count"]) != len(candidate_records)
        ):
            raise ValueError("player assignment geometry/frame alignment mismatch")
        assignments = {int(record["frame"]): record for record in assignment_records}
        if sorted(assignments) != expected_frames:
            raise ValueError("player assignment frames are incomplete or misaligned")
        pose_meta, pose_records = _jsonl(source.pose_cache_path)
        if pose_meta.get("raw_artifact_fingerprint") != assignment_meta.get(
            "raw_artifact_fingerprint"
        ):
            raise ValueError("pose-cache lineage does not match player assignments")
        for key in ("pose_model_fingerprint", "preprocessing_fingerprint"):
            if not str(pose_meta.get(key, "")).startswith("sha256:"):
                raise ValueError(f"pose cache has invalid {key}")
        poses = {
            (int(record["frame"]), int(record["track_id"])): record
            for record in pose_records
        }
        court = _calibration_values(source.calibration_path, image_size)
        windows = []
        for kind, burst in bursts:
            if burst.candidate_artifact_sha256 != _sha256(source.candidates_path):
                raise ValueError("queue candidate fingerprint mismatch")
            if burst.source_video_sha256 != _sha256(source.video_path):
                raise ValueError("queue video fingerprint mismatch")
            radius = int(math.floor(fps + 0.5))
            start = max(0, burst.anchor_frame - radius)
            end = min(len(candidate_records) - 1, burst.anchor_frame + radius)
            frame_indices = tuple(range(start, end + 1))
            owned = frozenset(burst.frames)
            candidate_values = []
            candidate_validity = []
            candidate_indices = []
            candidate_ids = []
            frame_values = []
            frame_validity = []
            targets = []
            statuses = []
            derived_reasons: dict[str, str] = {}
            for local_frame, frame in enumerate(frame_indices):
                groups = candidate_frames[frame]
                for group in groups:
                    value, valid = _candidate_values(group)
                    candidate_values.append(value)
                    candidate_validity.append(valid)
                    candidate_indices.append(local_frame)
                    candidate_ids.append(str(group["candidate_id"]))
                values, valid = _frame_context(
                    assignments[frame],
                    poses,
                    frame,
                    image_size,
                    court,
                    self.config.context_mode,
                    self.config.pose_visibility_threshold,
                )
                frame_values.append(values)
                frame_validity.append(valid)
                event = (
                    active.get(("shuttle_selection", source.source_id, frame))
                    if frame in owned
                    else None
                )
                target, status = MASKED_TARGET, "context"
                if event is not None:
                    if (
                        event.candidate_artifact_sha256
                        != burst.candidate_artifact_sha256
                        or event.source_video_sha256 != burst.source_video_sha256
                    ):
                        raise ValueError("annotation fingerprint mismatch")
                    if event.label_kind == "no_in_frame_target":
                        target, status = NULL_TARGET, "null"
                    elif event.label_kind in MASKED_LABELS:
                        status = event.label_kind
                        derived_reason = (event.annotation_metadata or {}).get(
                            "derived_reason"
                        )
                        if derived_reason is not None:
                            derived_reasons[str(frame)] = str(derived_reason)
                    elif event.label_kind == "selected":
                        members = set(
                            (event.annotation_metadata or {}).get(
                                "raw_member_ids", (event.candidate_id,)
                            )
                        )
                        matched = next(
                            (
                                i
                                for i, group in enumerate(groups)
                                if members.intersection(group["raw_member_ids"])
                            ),
                            None,
                        )
                        if matched is None:
                            status = "dropped_by_k"
                        else:
                            target, status = matched, "selected_retained"
                    else:
                        raise ValueError(
                            f"unsupported selector label {event.label_kind!r}"
                        )
                targets.append(target)
                statuses.append(status)
            cv = torch.tensor(candidate_values, dtype=torch.float32).reshape(-1, 12)
            valid = torch.tensor(candidate_validity, dtype=torch.bool).reshape(-1, 12)
            windows.append(
                SelectorWindow(
                    source.source_id,
                    burst.burst_id,
                    kind,
                    burst.anchor_frame,
                    frame_indices,
                    owned,
                    torch.tensor(
                        [(frame - burst.anchor_frame) / fps for frame in frame_indices],
                        dtype=torch.float32,
                    ),
                    cv,
                    valid,
                    torch.tensor(candidate_indices, dtype=torch.long),
                    tuple(candidate_ids),
                    torch.tensor(frame_values, dtype=torch.float32).reshape(
                        len(frame_indices), FRAME_DIMS[self.config.context_mode]
                    ),
                    torch.tensor(frame_validity, dtype=torch.bool).reshape(
                        len(frame_indices), FRAME_DIMS[self.config.context_mode]
                    ),
                    torch.tensor(targets, dtype=torch.long),
                    tuple(statuses),
                    {
                        "minimum_cutoff": self.config.minimum_cutoff,
                        "retention_k": self.config.retention_k,
                        "candidate_sha256": _sha256(source.candidates_path),
                        "annotation_sha256": _sha256(self.config.annotations_path),
                        "target_derived_reasons": derived_reasons,
                    },
                )
            )
        return windows


def collate_selector_windows(windows: Sequence[SelectorWindow]) -> SelectorBatch:
    if not windows:
        raise ValueError("cannot collate an empty selector batch")
    batch, max_frames = (
        len(windows),
        max(len(window.frame_indices) for window in windows),
    )
    max_candidates = max((len(window.candidate_ids) for window in windows), default=0)
    frame_dim = windows[0].frame_values.shape[1]
    if any(window.frame_values.shape[1] != frame_dim for window in windows):
        raise ValueError("cannot collate mixed frame views")
    candidate_values = torch.zeros(batch, max_candidates, 12)
    candidate_validity = torch.zeros(batch, max_candidates, 12, dtype=torch.bool)
    candidate_frame_indices = torch.full((batch, max_candidates), -1, dtype=torch.long)
    candidate_mask = torch.zeros(batch, max_candidates, dtype=torch.bool)
    frame_values = torch.zeros(batch, max_frames, frame_dim)
    frame_validity = torch.zeros(batch, max_frames, frame_dim, dtype=torch.bool)
    frame_mask = torch.zeros(batch, max_frames, dtype=torch.bool)
    times = torch.zeros(batch, max_frames)
    targets = torch.full((batch, max_frames), MASKED_TARGET, dtype=torch.long)
    for index, window in enumerate(windows):
        nc, nf = len(window.candidate_ids), len(window.frame_indices)
        candidate_values[index, :nc] = window.candidate_values
        candidate_validity[index, :nc] = window.candidate_validity
        candidate_frame_indices[index, :nc] = window.candidate_frame_indices
        candidate_mask[index, :nc] = True
        frame_values[index, :nf] = window.frame_values
        frame_validity[index, :nf] = window.frame_validity
        frame_mask[index, :nf] = True
        times[index, :nf] = window.relative_time_seconds
        targets[index, :nf] = window.targets
    return SelectorBatch(
        candidate_values,
        candidate_validity,
        candidate_frame_indices,
        candidate_mask,
        frame_values,
        frame_validity,
        frame_mask,
        times,
        targets,
    ).validate(frame_feature_dim=frame_dim)
