import json

import numpy as np
import pytest

from src.court_features import (
    Calibration,
    CalibrationRegistry,
    build_court_anchor_features,
    observed_court_anchor,
)
from src.court_projection import HALF_LENGTH, HALF_WIDTH, CourtHomography
from src.extract_clip_features import (
    FRAMES_WINDOW_SIZE,
    load_pose_data,
    load_shuttle_positions,
)


def identity_calibration():
    return Calibration(CourtHomography(np.eye(3)), (100, 100))


def landmark(x, y, visibility=1.0):
    return {"x": x, "y": y, "visibility": visibility}


def feet(left=(0.25, 0.4), right=(0.75, 0.6), visibility=1.0):
    return {
        "29": landmark(*left, visibility),
        "31": landmark(*left, visibility),
        "30": landmark(*right, visibility),
        "32": landmark(*right, visibility),
    }


def test_observed_anchor_projects_feet_then_averages_and_normalizes():
    anchor = observed_court_anchor(feet(), identity_calibration())
    np.testing.assert_allclose(anchor, [50 / HALF_WIDTH, 50 / HALF_LENGTH])


def test_observed_anchor_uses_visibility_weights_and_ankle_fallback():
    keypoints = {
        "29": landmark(0.2, 0.4, 1.0),
        "31": landmark(0.4, 0.4, 0.5),
        "28": landmark(0.8, 0.6, 0.9),
    }
    anchor = observed_court_anchor(keypoints, identity_calibration())
    left_x = (0.2 * 1.0 + 0.4 * 0.5) / 1.5
    expected_pixels = np.mean([[left_x * 100, 40], [80, 60]], axis=0)
    np.testing.assert_allclose(
        anchor, expected_pixels / [HALF_WIDTH, HALF_LENGTH]
    )


def test_anchor_features_interpolate_short_gap_and_preserve_mask():
    frames = [{"keypoints": feet(left=(i / 100, 0.2), right=(i / 100, 0.2))}
              for i in range(8)]
    frames[3] = {"keypoints": {}}
    frames[4] = {"keypoints": {}}

    features = build_court_anchor_features(
        frames, identity_calibration(), num_frames=8, max_missing_gap=2
    )

    assert features.shape == (8, 3)
    np.testing.assert_allclose(
        features[3:5, 0],
        np.asarray([3, 4]) / HALF_WIDTH,
    )
    np.testing.assert_array_equal(features[:, 2], [1, 1, 1, 0, 0, 1, 1, 1])


def test_anchor_features_fill_short_edge_gap_with_nearest_value():
    frames = [{"keypoints": {}}, {"keypoints": {}}] + [
        {"keypoints": feet(left=(0.1, 0.2), right=(0.1, 0.2))}
        for _ in range(4)
    ]
    features = build_court_anchor_features(
        frames, identity_calibration(), num_frames=6, max_missing_gap=2
    )
    np.testing.assert_allclose(features[0, :2], features[2, :2])
    np.testing.assert_array_equal(features[:3, 2], [0, 0, 1])


def test_anchor_features_reject_long_or_entirely_missing_gaps():
    frames = [{"keypoints": feet()}] + [{"keypoints": {}}] * 3 + [
        {"keypoints": feet()}
    ]
    with pytest.raises(ValueError, match="longer than 2"):
        build_court_anchor_features(
            frames, identity_calibration(), num_frames=5, max_missing_gap=2
        )
    with pytest.raises(ValueError, match="no usable"):
        build_court_anchor_features(
            [{"keypoints": {}}] * 3, identity_calibration(), num_frames=3
        )


def write_calibration(path):
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "coordinate_system": {},
                "image_to_court": np.eye(3).tolist(),
                "image_size": [1280, 720],
            }
        ),
        encoding="utf-8",
    )


def test_registry_resolves_source_ids_overrides_and_calibration(tmp_path):
    write_calibration(tmp_path / "img_3214.json")
    write_calibration(tmp_path / "img_3212.json")
    registry_path = tmp_path / "calibrations.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {
                    "img_3214": {"calibration": "img_3214.json"},
                    "img_3212": {"calibration": "img_3212.json"},
                },
                "clip_overrides": {"clear_001": "img_3214"},
            }
        ),
        encoding="utf-8",
    )
    registry = CalibrationRegistry(registry_path)

    assert registry.source_for_clip("smash_IMG_3212_01494_pose.json") == "img_3212"
    assert registry.source_for_clip("clear_001_features.npy") == "img_3214"
    assert registry.calibration_for_clip("clear_001").image_size == (1280, 720)
    with pytest.raises(ValueError, match="no source ID"):
        registry.source_for_clip("unknown_001")


def test_combined_feature_order_retains_original_73_columns(tmp_path):
    frames = [{"frame": i, "keypoints": feet()} for i in range(FRAMES_WINDOW_SIZE)]
    pose_path = tmp_path / "smash_img_3214_1_pose.json"
    pose_path.write_text(json.dumps(frames), encoding="utf-8")

    pose = load_pose_data(pose_path, FRAMES_WINDOW_SIZE)
    shuttle = load_shuttle_positions(tmp_path, "smash_img_3214_1")
    court = build_court_anchor_features(
        frames, identity_calibration(), FRAMES_WINDOW_SIZE
    )
    combined = np.concatenate([pose, shuttle, court], axis=1)

    assert combined.shape == (36, 76)
    np.testing.assert_array_equal(combined[:, :66], pose)
    np.testing.assert_array_equal(combined[:, 66:73], shuttle)
    np.testing.assert_array_equal(combined[:, 73:76], court)
