"""Reproducible temporal backgrounds for one static, source-local camera segment.

Alignment diagnostics are a guardrail, not camera segmentation: feature-poor
images can be inconclusive. The caller must not interpret missing motion evidence
as proof that a camera is static.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_STACK_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class BackgroundEstimate:
    image: np.ndarray
    image_size: tuple[int, int]
    frame_indices: list[int]
    frames: list[np.ndarray]
    scale: tuple[float, float]
    diagnostics: dict


def distributed_frame_indices(frame_count: int, sample_count: int = 31) -> list[int]:
    """Include both source-local endpoints without duplicate short-video samples."""
    if frame_count < 1 or sample_count < 1:
        raise ValueError("frame_count and sample_count must be positive")
    return np.unique(
        np.rint(np.linspace(0, frame_count - 1, min(sample_count, frame_count))).astype(int)
    ).tolist()


def _alignment_diagnostics(frames: list[np.ndarray]) -> dict:
    """Compare stable features against the middle sampled frame.

    A well-supported affine displacement greater than 1.2% of the image diagonal
    is material motion. Failed registration is explicitly inconclusive, including
    cuts that leave no feature correspondences.
    """
    if len(frames) < 2:
        return {"motion_detected": False, "alignment_status": "single_image",
                "alignment_checked_pairs": 0, "alignment_inconclusive_pairs": 0,
                "max_motion_fraction": 0.0}
    detector = cv2.ORB_create(nfeatures=1200)
    reference = frames[len(frames) // 2]
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    reference_points, reference_descriptors = detector.detectAndCompute(reference_gray, None)
    checked, inconclusive, largest = 0, 0, 0.0
    height, width = reference.shape[:2]
    anchors = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=float)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    for index, frame in enumerate(frames):
        if index == len(frames) // 2:
            continue
        points, descriptors = detector.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
        if descriptors is None or reference_descriptors is None or len(descriptors) < 2:
            inconclusive += 1
            continue
        matches = matcher.knnMatch(reference_descriptors, descriptors, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance]
        if len(good) < 12:
            inconclusive += 1
            continue
        source = np.float32([reference_points[m.queryIdx].pt for m in good])
        destination = np.float32([points[m.trainIdx].pt for m in good])
        matrix, inliers = cv2.estimateAffinePartial2D(source, destination, method=cv2.RANSAC,
                                                    ransacReprojThreshold=2.5)
        if matrix is None or inliers is None or inliers.sum() < 12 or inliers.mean() < .5:
            inconclusive += 1
            continue
        support = source[inliers.ravel().astype(bool)]
        if np.ptp(support[:, 0]) < .25 * width or np.ptp(support[:, 1]) < .25 * height:
            inconclusive += 1
            continue
        transformed = np.column_stack([anchors, np.ones(4)]) @ matrix.T
        displacement = np.linalg.norm(transformed - anchors, axis=1).max() / np.hypot(width, height)
        largest = max(largest, float(displacement))
        checked += 1
    motion = largest > .012
    return {"motion_detected": motion,
            "alignment_status": "motion" if motion else ("inconclusive" if inconclusive else "consistent"),
            "alignment_checked_pairs": checked, "alignment_inconclusive_pairs": inconclusive,
            "max_motion_fraction": largest}


def estimate_background(source: Path, sample_count: int = 31,
                        max_dimension: int = 960) -> BackgroundEstimate:
    """Compute a uint8 BGR median from bounded-resolution distributed samples.

    ``scale`` maps original pixels to working pixels. ``frames`` retains up to
    five distributed samples for independent visual checks (not statistical
    holdouts from the median). Decode failures and geometry changes fail closed.
    """
    source = Path(source)
    if not 1 <= sample_count <= 255:
        raise ValueError("sample_count must be between 1 and 255")
    if max_dimension < 16:
        raise ValueError("max_dimension must be at least 16")
    capture = None
    if source.suffix.lower() in IMAGE_SUFFIXES:
        initial = cv2.imread(str(source))
        if initial is None:
            raise ValueError(f"could not read image: {source}")
        indices = [0]
    else:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"could not open video: {source}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1:
            capture.release()
            raise ValueError(f"could not determine frame count: {source}")
        indices = distributed_frame_indices(count, sample_count)
        initial = None
    image_size = None
    stack = None
    actual_indices = []
    try:
        for sample_index, requested_index in enumerate(indices):
            if capture is None:
                frame, actual_index = initial, 0
            else:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, requested_index):
                    raise ValueError(f"could not seek to source-local frame {requested_index}")
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(f"could not decode source-local frame {requested_index}")
                position = capture.get(cv2.CAP_PROP_POS_FRAMES)
                if not np.isfinite(position) or position < 1:
                    raise ValueError("decoder did not report its source-local frame position")
                actual_index = int(round(position)) - 1
                if actual_index != requested_index:
                    raise ValueError(f"decoder returned frame {actual_index}, requested {requested_index}")
            height, width = frame.shape[:2]
            if image_size is None:
                image_size = (width, height)
                ratio = min(1., max_dimension / max(width, height))
                working_size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
                stack_bytes = len(indices) * working_size[0] * working_size[1] * 3
                if stack_bytes > MAX_STACK_BYTES:
                    raise ValueError("background sample stack exceeds 256 MiB; reduce sample_count or max_dimension")
                stack = np.empty((len(indices), working_size[1], working_size[0], 3), dtype=np.uint8)
            elif image_size != (width, height):
                raise ValueError("source image geometry changes between sampled frames")
            stack[sample_index] = cv2.resize(frame, working_size, interpolation=cv2.INTER_AREA)
            actual_indices.append(actual_index)
    finally:
        if capture is not None:
            capture.release()
    retained_indices = distributed_frame_indices(len(indices), min(5, len(indices)))
    frames = [stack[index].copy() for index in retained_indices]
    diagnostics = _alignment_diagnostics(frames)
    # overwrite_input prevents an additional full sample-stack copy. Retained
    # raw frames above are independent because partitioning mutates the stack.
    median = np.median(stack, axis=0, overwrite_input=True).astype(np.uint8)
    diagnostics.update({"requested_sample_count": sample_count,
                        "sample_count": len(actual_indices),
                        "max_dimension": max_dimension,
                        "working_size": list(working_size),
                        "frame_indices": actual_indices,
                        "verification_frame_indices": [actual_indices[index] for index in retained_indices],
                        "frame_indexing": "source_local_zero_based",
                        "stack_bytes": int(stack.nbytes)})
    return BackgroundEstimate(median, image_size, actual_indices, frames,
                              (working_size[0] / image_size[0], working_size[1] / image_size[1]),
                              diagnostics)
