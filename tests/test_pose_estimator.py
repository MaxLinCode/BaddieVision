from types import SimpleNamespace

import cv2
import numpy as np

from src import pose_estimator
from src.pose_estimator import PoseLandmark, PoseEstimate, normalize_pose_landmarks
from src.visualize_features import draw_pose


def test_normalize_pose_landmarks_preserves_raw_schema() -> None:
    landmarks = [
        SimpleNamespace(x=0.1, y=0.2, z=-0.3, visibility=0.9, presence=0.8),
        SimpleNamespace(x=0.4, y=0.5, z=0.6, visibility=0.7),
    ]

    estimate = PoseEstimate(normalize_pose_landmarks(landmarks))

    assert estimate.detected
    assert estimate.to_raw_landmarks() == [
        {"x": 0.1, "y": 0.2, "z": -0.3, "visibility": 0.9},
        {"x": 0.4, "y": 0.5, "z": 0.6, "visibility": 0.7},
    ]
    assert estimate.landmarks[0].presence == 0.8
    assert estimate.landmarks[1].presence is None


def test_normalize_pose_landmarks_uses_presence_when_visibility_missing() -> None:
    landmark = SimpleNamespace(x=0.1, y=0.2, z=0.3, presence=0.65)

    normalized = normalize_pose_landmarks([landmark])

    assert normalized == [PoseLandmark(0.1, 0.2, 0.3, 0.65, 0.65)]


def test_draw_pose_uses_adapter_connections(monkeypatch) -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    pose_frame = {
        "keypoints": {
            "0": {"x": 0.25, "y": 0.25, "visibility": 1.0},
            "1": {"x": 0.75, "y": 0.75, "visibility": 1.0},
        }
    }
    monkeypatch.setattr(pose_estimator, "POSE_CONNECTIONS", ((0, 1),))

    visible = draw_pose(frame, pose_frame, visibility_threshold=0.5)

    assert visible == 2
    assert np.count_nonzero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 0
