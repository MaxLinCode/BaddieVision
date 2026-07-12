"""Headless view-model and frame-rendering helpers shared with Dash callbacks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from .core import AnnotationRegistry, SourceRegistration, file_sha256


@dataclass(frozen=True)
class PlaybackView:
    task: str
    source_id: str
    center_frame: int
    center_seconds: float
    clip_start_seconds: float
    clip_end_seconds: float
    candidate_artifact_sha256: str
    source_video_sha256: str
    overlays_by_frame: Mapping[int, tuple[Mapping[str, Any], ...]]

    @property
    def center_candidates(self) -> tuple[Mapping[str, Any], ...]:
        return self.overlays_by_frame.get(self.center_frame, ())


def validate_source_video(source: SourceRegistration, *, force: bool = False) -> None:
    if not source.video_path.is_file():
        raise ValueError(f"registered source video is missing: {source.video_path}")
    stat = source.video_path.stat()
    unchanged_stat = (stat.st_size, stat.st_mtime_ns) == (
        source.video_size,
        source.video_mtime_ns,
    )
    if unchanged_stat and not force:
        return
    current = file_sha256(source.video_path)
    if current != source.video_sha256:
        raise ValueError(
            f"source video changed after registration for {source.source_id}; "
            "restart with a new source registration"
        )


def build_playback_view(
    registry: AnnotationRegistry,
    *,
    task: str,
    source_id: str,
    center_frame: int,
    context_seconds: float = 1.0,
) -> PlaybackView:
    """Preload candidate overlays for every frame in the ±context clip."""
    plugin, source = registry.resolve(task, source_id)
    center_frame = int(center_frame)
    if not 0 <= center_frame < source.frame_count:
        raise ValueError(f"center frame outside source bounds: {center_frame}")
    if context_seconds <= 0:
        raise ValueError("context_seconds must be positive")
    validate_source_video(source)
    eligible = {int(frame) for frame in plugin.eligible_frames(source)}
    if center_frame not in eligible:
        raise ValueError(
            f"task {task!r} has no payload record for {source_id}:{center_frame}"
        )
    run_start = center_frame
    while run_start - 1 in eligible:
        run_start -= 1
    run_end = center_frame
    while run_end + 1 in eligible:
        run_end += 1
    center_seconds = center_frame / source.fps
    source_end = source.frame_count / source.fps
    start_seconds = max(0.0, run_start / source.fps, center_seconds - context_seconds)
    end_seconds = min(
        source_end,
        (run_end + 1) / source.fps,
        center_seconds + context_seconds,
    )
    start_frame = max(0, math.floor(start_seconds * source.fps))
    end_frame = min(run_end, source.frame_count - 1, math.ceil(end_seconds * source.fps))
    overlays = {
        frame: tuple(plugin.overlays(source, frame))
        for frame in range(start_frame, end_frame + 1)
    }
    return PlaybackView(
        task=task,
        source_id=source_id,
        center_frame=center_frame,
        center_seconds=center_seconds,
        clip_start_seconds=start_seconds,
        clip_end_seconds=end_seconds,
        candidate_artifact_sha256=plugin.artifact_sha256(source),
        source_video_sha256=source.video_sha256,
        overlays_by_frame=overlays,
    )


def decode_video_frame(source: SourceRegistration, frame: int) -> np.ndarray:
    """Decode the requested source frame exactly for center-frame selection."""
    validate_source_video(source)
    frame = int(frame)
    if not 0 <= frame < source.frame_count:
        raise ValueError(f"frame outside source bounds: {frame}")
    capture = cv2.VideoCapture(str(source.video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok or image is None:
        raise ValueError(f"could not decode {source.source_id} frame {frame}")
    return image


def render_center_frame(
    registry: AnnotationRegistry,
    *,
    task: str,
    source_id: str,
    frame: int,
    image_format: str = ".jpg",
) -> bytes:
    """Render a labeled center frame for UI and non-UI smoke tests."""
    plugin, source = registry.resolve(task, source_id)
    image = decode_video_frame(source, frame)
    for index, candidate in enumerate(plugin.overlays(source, frame), start=1):
        center = candidate.get("center")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            continue
        point = int(round(float(center[0]))), int(round(float(center[1])))
        cv2.circle(image, point, 9, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            str(index),
            (point[0] + 11, point[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 230, 255),
            2,
            cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(image_format, image)
    if not ok:
        raise ValueError(f"could not encode center frame as {image_format}")
    return encoded.tobytes()
