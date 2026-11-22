"""Read source frames and run a crop pose backend on assigned observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import cv2
import numpy as np
from tqdm.auto import tqdm

from src.pose_backends import CropPoseEstimator, PoseBackend

ObservationKey = tuple[int, int, tuple[float, ...]]
CropTask = tuple[ObservationKey, Iterable[float], np.ndarray]


def expand_pose_bbox(
    bbox: Iterable[float], frame_size: tuple[int, int], padding: float
) -> list[float]:
    """Expand every detector-box edge by a fraction of its original dimension."""
    x1, y1, x2, y2 = map(float, bbox)
    width, height = frame_size
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return [max(0.0, x1 - dx), max(0.0, y1 - dy), min(float(width), x2 + dx), min(float(height), y2 + dy)]


@dataclass(frozen=True)
class CropAttempt:
    bbox: list[float]
    landmarks: list[dict[str, float]] | None
    quality: float
    status: str


def _estimate_crops(
    estimator: CropPoseEstimator,
    tasks: list[CropTask],
    padding: float,
    quality: Callable[[list[dict[str, float]] | None], float],
) -> list[CropAttempt]:
    crops = []
    boxes = []
    valid_indices = []
    for index, (_, bbox, image) in enumerate(tasks):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = map(round, expand_pose_bbox(bbox, (width, height), padding))
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        crop = image[y1:y2, x1:x2]
        if x2 > x1 and y2 > y1 and crop.size:
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            valid_indices.append(index)

    attempts = [CropAttempt(box, None, 0.0, "invalid_crop") for box in boxes]
    if crops:
        estimate_batch = getattr(estimator, "estimate_batch", None)
        results = estimate_batch(crops) if estimate_batch else [estimator.estimate_pose(crop) for crop in crops]
        for index, estimate in zip(valid_indices, results, strict=True):
            landmarks = estimate.to_raw_landmarks() if estimate.detected else None
            attempts[index] = CropAttempt(
                boxes[index], landmarks, quality(landmarks), "detected" if landmarks else "no_pose",
            )
    return attempts


def _run_batch(estimator, tasks, backend, quality, threshold):
    first = _estimate_crops(estimator, tasks, backend.padding, quality)
    retries = {}
    if backend.retry_padding is not None and backend.retry_padding != backend.padding:
        indices = [index for index, attempt in enumerate(first) if attempt.quality < threshold]
        if indices:
            retried = _estimate_crops(estimator, [tasks[index] for index in indices], backend.retry_padding, quality)
            retries = dict(zip(indices, retried, strict=True))
    for index, ((key, _, _), attempt) in enumerate(zip(tasks, first, strict=True)):
        retry = retries.get(index)
        yield key, retry if retry is not None and retry.quality > attempt.quality else attempt, 2 if retry else 1


def iter_player_crop_poses(
    video_path: str | Path,
    by_frame: dict[int, list[tuple[ObservationKey, Iterable[float]]]],
    backend: PoseBackend,
    quality: Callable[[list[dict[str, float]] | None], float],
    threshold: float,
) -> Iterator[tuple[ObservationKey, CropAttempt, int]]:
    """Bound memory by backend batch size and retain source-local frame order."""
    if not by_frame:
        return
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"failed to open pose source: {video_path}")
        estimator = backend.create()
        try:
            tasks = []
            total = sum(map(len, by_frame.values()))
            for frame in tqdm(range(max(by_frame) + 1), desc=f"Pose enrichment ({total} crops)", unit="frame"):
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"video ended before required pose frame {frame}")
                for key, bbox in by_frame.get(frame, ()):
                    tasks.append((key, bbox, image))
                    if len(tasks) == backend.batch_size:
                        yield from _run_batch(estimator, tasks, backend, quality, threshold)
                        tasks.clear()
            if tasks:
                yield from _run_batch(estimator, tasks, backend, quality, threshold)
        finally:
            estimator.close()
    finally:
        cap.release()
