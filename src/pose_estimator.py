"""Repo-local adapter for MediaPipe Tasks pose estimation."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.request import urlretrieve

import numpy as np

PoseRunningMode = Literal["image", "video"]

POSE_BACKEND_TYPE = "mediapipe.tasks.pose_landmarker"
DEFAULT_POSE_MODEL_NAME = "pose_landmarker_full.task"
DEFAULT_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DEFAULT_POSE_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / DEFAULT_POSE_MODEL_NAME

POSE_NUM_LANDMARKS = 33
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)


@dataclass(frozen=True)
class PoseLandmark:
    x: float
    y: float
    z: float
    visibility: float
    presence: float | None = None

    def to_raw_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "visibility": self.visibility,
        }


@dataclass(frozen=True)
class PoseEstimate:
    landmarks: list[PoseLandmark]

    @property
    def detected(self) -> bool:
        return bool(self.landmarks)

    def to_raw_landmarks(self) -> list[dict[str, float]]:
        return [landmark.to_raw_dict() for landmark in self.landmarks]


def pose_connections() -> tuple[tuple[int, int], ...]:
    return POSE_CONNECTIONS


def pose_backend_info(model_asset_path: str | Path | None = None) -> dict[str, str]:
    try:
        package_version = importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError:
        package_version = "uninstalled"
    return {
        "backend_type": POSE_BACKEND_TYPE,
        "package": "mediapipe",
        "package_version": package_version,
        "model_asset_path": str(model_asset_path or DEFAULT_POSE_MODEL_PATH),
    }


def ensure_pose_model_asset(model_asset_path: str | Path | None = None) -> Path:
    path = Path(model_asset_path) if model_asset_path is not None else DEFAULT_POSE_MODEL_PATH
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(DEFAULT_POSE_MODEL_URL, path)
    return path


def _normalize_running_mode(value: PoseRunningMode | str) -> PoseRunningMode:
    normalized = value.lower()
    if normalized not in {"image", "video"}:
        raise ValueError(f"unsupported pose running mode: {value!r}")
    return normalized  # type: ignore[return-value]


def _landmark_value(landmark: Any, name: str, default: float | None = None) -> float | None:
    value = getattr(landmark, name, default)
    return None if value is None else float(value)


def normalize_pose_landmarks(landmarks: Iterable[Any] | None) -> list[PoseLandmark]:
    if landmarks is None:
        return []
    normalized = []
    for landmark in landmarks:
        presence = _landmark_value(landmark, "presence")
        visibility = _landmark_value(landmark, "visibility", presence)
        normalized.append(
            PoseLandmark(
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                visibility=0.0 if visibility is None else visibility,
                presence=presence,
            )
        )
    return normalized


class MediaPipeTasksPoseEstimator:
    def __init__(
        self,
        *,
        model_asset_path: str | Path | None = None,
        running_mode: PoseRunningMode | str = "image",
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp

        self.model_asset_path = ensure_pose_model_asset(model_asset_path)
        self.running_mode = _normalize_running_mode(running_mode)
        self._mp = mp
        vision_running_mode = (
            mp.tasks.vision.RunningMode.IMAGE
            if self.running_mode == "image"
            else mp.tasks.vision.RunningMode.VIDEO
        )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_asset_path)),
            running_mode=vision_running_mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    @property
    def backend_info(self) -> dict[str, str]:
        return pose_backend_info(self.model_asset_path)

    def estimate_pose(self, image: np.ndarray, timestamp_ms: int | None = None) -> PoseEstimate:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image)
        if self.running_mode == "video":
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required for MediaPipe Tasks video mode")
            result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))
        else:
            result = self._landmarker.detect(mp_image)
        pose_landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        return PoseEstimate(normalize_pose_landmarks(pose_landmarks))

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "MediaPipeTasksPoseEstimator":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def create_pose_estimator(**kwargs: Any) -> MediaPipeTasksPoseEstimator:
    return MediaPipeTasksPoseEstimator(**kwargs)


def estimate_pose(
    estimator: MediaPipeTasksPoseEstimator,
    image: np.ndarray,
    timestamp_ms: int | None = None,
) -> PoseEstimate:
    return estimator.estimate_pose(image, timestamp_ms=timestamp_ms)
