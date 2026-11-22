"""Backend selection for pose estimation on already identified player crops."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from src.pose_estimator import (
    POSE_NUM_LANDMARKS,
    PoseEstimate,
    PoseLandmark,
)
from src.rtmpose import (
    DEFAULT_RTMPOSE_MODEL_PATH, RTMPOSE_BACKEND_TYPE, RTMPOSE_CROP_PADDING,
    open_session, predict, validate_fp16_halpe26_model,
)

POSE_BACKENDS = ("rtmpose", "mediapipe")
HALPE_TO_PLAYER_LAYOUT = {
    **dict(enumerate((0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28))),
    20: 31, 21: 32, 24: 29, 25: 30,
}


class CropPoseEstimator(Protocol):
    def estimate_pose(self, image: np.ndarray) -> PoseEstimate: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class PoseBackend:
    info: dict[str, str]
    create: Callable[[], CropPoseEstimator]
    padding: float
    retry_padding: float | None
    batch_size: int = 1


class RTMPoseCropEstimator:
    def __init__(self, model_path: Path, device: str) -> None:
        session, _ = open_session(model_path, device)
        validate_fp16_halpe26_model(session)
        self.session = session

    def estimate_pose(self, image: np.ndarray) -> PoseEstimate:
        return self.estimate_batch([image])[0]

    def estimate_batch(self, images: Sequence[np.ndarray]) -> list[PoseEstimate]:
        # The adapter accepts RGB, while the native RTMPose preprocessor accepts BGR.
        results = predict(self.session, [image[..., ::-1] for image in images])
        estimates = []
        for image, (points, scores) in zip(images, results, strict=True):
            height, width = image.shape[:2]
            landmarks = [PoseLandmark(0.0, 0.0, 0.0, 0.0) for _ in range(POSE_NUM_LANDMARKS)]
            for source, target in HALPE_TO_PLAYER_LAYOUT.items():
                x, y = points[source]
                score = scores[source]
                if 0 <= x < width and 0 <= y < height and np.isfinite(score) and score > 0:
                    landmarks[target] = PoseLandmark(
                        float(x / width), float(y / height), 0.0, min(float(score), 1.0),
                    )
            estimates.append(PoseEstimate(landmarks if any(p.visibility > 0 for p in landmarks) else []))
        return estimates

    def close(self) -> None:
        self.session = None


def resolve_crop_pose_backend(
    name: str,
    model_asset: Path | str | None,
    *,
    device: str | None,
    crop_padding: float | None,
    mediapipe_padding: float,
    mediapipe_retry_padding: float | None,
    estimator_factory: Callable[..., CropPoseEstimator] | None = None,
) -> PoseBackend:
    """Resolve metadata before loading the runtime, so cache hits skip inference setup."""
    if crop_padding is not None and (not np.isfinite(crop_padding) or crop_padding < 0):
        raise ValueError("pose crop padding must be finite and nonnegative")
    if name == "mediapipe":
        from src.pose_estimator import (
            create_pose_estimator, ensure_pose_model_asset, pose_backend_info,
        )

        if device not in (None, "cpu"):
            raise ValueError("the MediaPipe crop backend uses CPU")
        factory = estimator_factory or create_pose_estimator
        model = ensure_pose_model_asset(model_asset) if factory is create_pose_estimator else model_asset
        return PoseBackend(
            pose_backend_info(model),
            lambda: factory(model_asset_path=model, running_mode="image"),
            mediapipe_padding if crop_padding is None else crop_padding,
            mediapipe_retry_padding,
        )
    if name != "rtmpose":
        raise ValueError(f"unknown crop pose backend: {name}")
    model = Path(model_asset or DEFAULT_RTMPOSE_MODEL_PATH).expanduser()
    if not model.is_file():
        raise FileNotFoundError(f"RTMPose FP16 Halpe26 model is missing: {model}; supply a local model asset")
    if device not in (None, "cpu", "cuda"):
        raise ValueError(f"unsupported RTMPose device: {device}")
    provider = device or "cpu"
    try:
        runtime_version = version("onnxruntime-gpu" if provider == "cuda" else "onnxruntime")
    except PackageNotFoundError as exc:
        if provider == "cuda":
            raise RuntimeError("install onnxruntime-gpu for RTMPose on CUDA") from exc
        try:
            runtime_version = version("onnxruntime-gpu")
        except PackageNotFoundError:
            raise RuntimeError("install onnxruntime for RTMPose on CPU") from exc
    return PoseBackend(
        {
            "backend_type": RTMPOSE_BACKEND_TYPE,
            "package": "onnxruntime",
            "package_version": runtime_version,
            "model_asset_path": str(model),
            "device": provider,
            "precision": "fp16",
            "output_layout": "mediapipe33_halpe26_subset_v1",
        },
        lambda: RTMPoseCropEstimator(model, provider),
        RTMPOSE_CROP_PADDING if crop_padding is None else crop_padding,
        None,
        batch_size=8,
    )
