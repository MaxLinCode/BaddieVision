"""Plan chronological rally partitions, then extract each split recording independently."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from bisect import bisect_right
from dataclasses import dataclass, replace
from pathlib import Path
from itertools import accumulate
from typing import Any, Mapping, Sequence

import torch

from src.rtmpose import DEFAULT_RTMPOSE_MODEL_PATH, RTMPOSE_BACKEND_TYPE
from src.workflow.artifacts import atomic_write_json, import_calibration
from src.workflow.identity import ParentVideoProvenance, SourceIdentity, inspect_source, sha256_file
from src.workflow.publication import recovered_publication
from src.workflow.registry import SourceEntry
from src.workflow.stages import resolve_person_detector, run_person_stage, run_player_stage

from .batch import MASKED_TARGET
from .observations import rally_boundary_targets
from .rally_audio import video_frame_timestamps
from .rally_dataset import RallyDataConfig, RallySourceConfig, RallyWindow, RallyWindowDataset
from .rally_intervals import RallyIntervalIndex


@dataclass(frozen=True)
class RallySourceRange:
    source_id: str
    start: int
    stop: int


@dataclass(frozen=True)
class RallyCameraSplit:
    train: tuple[RallySourceRange, ...]
    validation: tuple[RallySourceRange, ...]
    cut: int
    total_frames: int


def plan_rally_camera_splits(
    index: RallyIntervalIndex, camera_sources: Mapping[str, Sequence[str]],
) -> dict[str, RallyCameraSplit]:
    """Prefer whole recordings near 80/20, otherwise cut near a reviewed break."""
    splits = {}
    for camera, sources in camera_sources.items():
        counts = [index.sources[source].frame_count for source in sources]
        total = sum(counts)
        if total < 2:
            raise ValueError(f"camera {camera!r} needs at least two frames to split")
        nominal = max(1, int(total * 0.8))
        boundaries = list(accumulate(counts))[:-1]
        nearby_boundaries = [cut for cut in boundaries if 0.7 <= cut / total <= 0.9]
        if nearby_boundaries:
            cut = min(nearby_boundaries, key=lambda frame: (abs(frame - nominal), frame))
        else:
            cut = nominal
            if cut not in boundaries:
                source_index = bisect_right(boundaries, cut)
                offset = boundaries[source_index - 1] if source_index else 0
                source_id = sources[source_index]
                source = index.sources[source_id]
                local = cut - offset
                radius = round(2 * source.fps)
                breaks = (
                    frame for frame in range(max(1, local - radius), min(source.frame_count, local + radius + 1))
                    if index.target(source_id, frame - 1) == index.target(source_id, frame) == 0
                )
                cut = offset + min(breaks, key=lambda frame: (abs(frame - local), frame), default=local)
        train, validation = [], []
        offset = 0
        for source, count in zip(sources, counts, strict=True):
            relative_cut = min(count, max(0, cut - offset))
            if relative_cut:
                train.append(RallySourceRange(source, 0, relative_cut))
            if relative_cut < count:
                validation.append(RallySourceRange(source, relative_cut, count))
            offset += count
        splits[camera] = RallyCameraSplit(tuple(train), tuple(validation), cut, total)
    return splits


def _write_partition_video(
    source: RallySourceConfig, block: RallySourceRange, parent: SourceIdentity, output: Path,
) -> SourceIdentity:
    times = video_frame_timestamps(source.video_path, parent.frame_count)
    fps = parent.fps_numerator / parent.fps_denominator
    start = float(times[block.start])
    stop = float(times[block.stop]) if block.stop < len(times) else float(times[-1]) + 1 / fps
    output.parent.mkdir(parents=True, exist_ok=True)
    # Trim by decoded frame number for video and by presentation time for audio.
    # PCM in Matroska preserves audio packet gaps without encoder buffering.
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / output.name
        result = subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-copyts", "-i", str(source.video_path),
            "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", f"trim=start_frame={block.start}:end_frame={block.stop},setpts=PTS-STARTPTS",
            "-af", f"atrim=start={start:.9f}:end={stop:.9f},asetpts=PTS-({start:.9f})/TB",
            "-map_metadata", "-1", "-map_chapters", "-1",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
            "-enc_time_base", "1:90000", "-fps_mode", "passthrough", "-c:a", "pcm_f32le", str(temporary),
        ], capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"could not extract rally partition for {source.source_id}: {result.stderr.strip()}")
        identity = inspect_source(output.parent.name, temporary)
        if (identity.frame_count, identity.width, identity.height) != (
            block.stop - block.start, parent.width, parent.height,
        ):
            raise ValueError("extracted rally partition differs from the planned frame range")
        temporary.replace(output)
    return replace(identity, parent_video=ParentVideoProvenance(
        video_sha256=parent.video_sha256,
        start_time_seconds=start,
        end_time_seconds=stop,
        handling="rally-partition",
    ))


def _metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.loads(handle.readline())


def prepare_rally_partition(
    config: RallyDataConfig,
    source: RallySourceConfig,
    block: RallySourceRange,
    index: RallyIntervalIndex,
    cache_dir: Path,
    *,
    device: str,
    person_model: str | None = None,
    pose_model: Path | None = None,
) -> tuple[tuple[RallyWindow, ...], str]:
    """Materialize a bounded source using fresh tracker and player stages."""
    parent = SourceIdentity.read(source.calibration_path.parent.parent / "source.json")
    if not 0 <= block.start < block.stop <= parent.frame_count:
        raise ValueError("rally partition is outside its source frame range")
    if parent.source_id != block.source_id or parent.video_sha256 != sha256_file(source.video_path):
        raise ValueError(f"source identity changed for {block.source_id}")
    cache = _metadata(source.pose_cache_path)
    backend = cache.get("pose_backend") or {}
    rtmpose = backend.get("backend_type") == RTMPOSE_BACKEND_TYPE
    recorded_pose_model = backend.get("model_asset_path")
    if pose_model is None and recorded_pose_model:
        pose_model = Path(recorded_pose_model).expanduser()
    if rtmpose:
        pose_model = pose_model or DEFAULT_RTMPOSE_MODEL_PATH
        if not pose_model.is_file() or sha256_file(pose_model) != cache.get("pose_model_sha256"):
            raise ValueError(f"{block.source_id}: provide the exact training asset with --pose-model")
    tracks_path = source.assignments_path.with_name("person_tracks.jsonl")
    tracks = _metadata(tracks_path) if tracks_path.is_file() else {}
    if person_model is None:
        person_model = (tracks.get("detector") or {}).get("model")
    if not person_model:
        raise ValueError(f"{block.source_id}: provide the original detector with --person-model")
    detector = resolve_person_detector(person_model)
    detector_digest = sha256_file(detector)
    receipt_path = tracks_path.with_name("person-tracks.manifest.json")
    if receipt_path.is_file():
        original = _metadata(receipt_path)["inputs"].get("detector_sha256")
        if original is not None and original != detector_digest:
            raise ValueError(f"{block.source_id}: --person-model must match the full-source detector")
    padding = (cache.get("preprocessing") or {}).get("pose_crop_padding")
    cache_inputs = [
        "rally-partition-v3", parent.fingerprint, sha256_file(source.calibration_path),
        block.start, block.stop, detector_digest, str(pose_model), device,
        cache.get("pose_model_fingerprint"), cache.get("preprocessing_fingerprint"),
    ]
    digest = hashlib.sha256(json.dumps(cache_inputs).encode()).hexdigest()
    root = cache_dir / f"partition-{digest}"
    entry = SourceEntry(root.name, root / "clip.mkv", root)
    with recovered_publication(entry.identity_path):
        if entry.identity_path.is_file():
            identity = SourceIdentity.read(entry.identity_path)
            if identity.video_sha256 != sha256_file(entry.video_path):
                raise ValueError(f"cached rally partition video has changed: {entry.video_path}")
        else:
            identity = _write_partition_video(source, block, parent, entry.video_path)
            atomic_write_json(entry.identity_path, identity.to_mapping())
        if not entry.calibration_path.is_file():
            import_calibration(entry, source.calibration_path)
    # The person stage creates a fresh detector/tracker for this bounded video.
    run_person_stage(entry, model=detector, device=device)
    if tracks.get("tracker") is not None and _metadata(entry.person_tracks_path).get("tracker") != tracks["tracker"]:
        raise ValueError(f"{block.source_id}: partition extraction used a different tracker")
    run_player_stage(
        entry, pose_model=pose_model, pose_crop_padding=padding,
        pose_backend="rtmpose" if rtmpose else "mediapipe",
        pose_device="cuda" if rtmpose and device == "cuda" else None,
    )
    child = RallySourceConfig(
        entry.source_id, entry.video_path, entry.assignments_path, entry.pose_cache_path, entry.calibration_path,
    )
    dataset = RallyWindowDataset(replace(
        config, sources=(child,), rally_intervals_path=None, rally_manifest_path=None,
    ))
    windows = tuple(_label_partition(window, block, index, parent.frame_count) for window in dataset)
    return windows, dataset.manifest["dataset_fingerprint"]


def _label_partition(
    window: RallyWindow, block: RallySourceRange, index: RallyIntervalIndex, source_count: int,
) -> RallyWindow:
    """Restore original frame keys and supervise only boundaries inside the partition."""
    intervals = index.for_source(block.source_id)
    frames = tuple(frame + block.start for frame in window.frame_indices)
    owned = frozenset(frame + block.start for frame in window.owned_frames)
    anchor = window.anchor_frame + block.start
    targets = [index.target(block.source_id, frame) if frame in owned else MASKED_TARGET for frame in frames]
    fps = float(window.metadata["fps"])
    bounds = range(block.start, block.stop)
    starts, ends = rally_boundary_targets(
        intervals, index.partial_start_rally_ids, frames, owned, targets, fps=fps, boundary_range=bounds,
    )
    partial = tuple(
        (
            max(interval.start_frame, block.start),
            min(interval.end_frame, block.stop - 1),
            interval.start_frame in bounds and interval.rally_id not in index.partial_start_rally_ids,
            interval.end_frame in bounds,
        )
        for interval in intervals
        if interval.start_frame < block.stop and interval.end_frame >= block.start
        and (interval.start_frame < block.start or interval.end_frame >= block.stop)
    )
    return replace(
        window, source_id=block.source_id, window_id=f"rally-{block.source_id}-{anchor:09d}",
        anchor_frame=anchor, frame_indices=frames, owned_frames=owned, inplay_targets=torch.tensor(targets),
        rally_start_targets=starts, rally_end_targets=ends,
        metadata={
            **window.metadata, "frame_count": source_count,
            "partial_rally_spans": partial,
            "owns_rally_boundary": bool(((starts == 1) | (ends == 1)).any()),
            "owns_short_rally": any(
                interval.end_frame - interval.start_frame + 1 <= fps
                and any(interval.contains(frame) for frame in owned)
                for interval in intervals
            ),
        },
    )
