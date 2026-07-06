import json

import cv2
import numpy as np
import pytest

from src.court_projection import (
    COURT_CALIBRATION_LINES,
    COURT_LANDMARKS,
    COURT_LENGTH,
    COURT_WIDTH,
    CourtHomography,
    court_line_segments,
    draw_court_overlay,
)


def synthetic_calibration():
    court = np.array(
        [
            COURT_LANDMARKS["near_left_doubles"],
            COURT_LANDMARKS["near_right_doubles"],
            COURT_LANDMARKS["far_right_doubles"],
            COURT_LANDMARKS["far_left_doubles"],
        ],
        dtype=np.float64,
    )
    image = np.array(
        [[70, 450], [570, 450], [440, 80], [200, 80]], dtype=np.float64
    )
    calibration, inliers = CourtHomography.from_points(image, court)
    return calibration, image, court, inliers


def test_court_dimensions_and_markings():
    assert COURT_WIDTH == pytest.approx(6.10)
    assert COURT_LENGTH == pytest.approx(13.40)
    assert len(court_line_segments()) == 13
    assert ((0.0, -6.7), (0.0, -1.98)) in court_line_segments()
    assert ((0.0, 1.98), (0.0, 6.7)) in court_line_segments()
    assert COURT_LANDMARKS["near_doubles_long_service_center"] == pytest.approx(
        (0.0, -5.94)
    )


def test_round_trip_projection():
    calibration, image, court, inliers = synthetic_calibration()
    np.testing.assert_allclose(calibration.project_to_court(image), court, atol=1e-6)
    np.testing.assert_allclose(calibration.project_to_image(court), image, atol=1e-4)
    np.testing.assert_allclose(
        calibration.project_normalized_to_court(image / [640, 480], (640, 480)),
        court,
        atol=1e-6,
    )
    assert calibration.project_to_court([]).shape == (0, 2)
    assert inliers.all()


def test_named_landmark_calibration():
    _, image, court, _ = synthetic_calibration()
    names = [
        "near_left_doubles",
        "near_right_doubles",
        "far_right_doubles",
        "far_left_doubles",
    ]
    calibration, _ = CourtHomography.from_landmarks(dict(zip(names, image)))
    np.testing.assert_allclose(calibration.project_to_court(image), court, atol=1e-6)


def test_ransac_rejects_bad_click():
    truth, _, _, _ = synthetic_calibration()
    names = [
        "near_left_doubles",
        "near_right_doubles",
        "far_right_doubles",
        "far_left_doubles",
        "near_short_service_left",
        "near_short_service_right",
    ]
    court = np.asarray([COURT_LANDMARKS[name] for name in names])
    image = truth.project_to_image(court)
    image[-1] += [80, -60]

    fitted, inliers = CourtHomography.from_points(
        image, court, ransac_threshold=3.0
    )

    assert inliers.tolist() == [True, True, True, True, True, False]
    np.testing.assert_allclose(
        fitted.project_to_image(court[:-1]),
        truth.project_to_image(court[:-1]),
        atol=1e-4,
    )


def test_line_calibration_infers_intersections():
    truth, _, _, _ = synthetic_calibration()
    image_lines = {}
    for name in (
        "left_doubles_sideline",
        "right_doubles_sideline",
        "near_short_service",
        "far_short_service",
    ):
        axis, value = COURT_CALIBRATION_LINES[name]
        court_endpoints = (
            [[value, -4.0], [value, 4.0]]
            if axis == "x"
            else [[-2.0, value], [2.0, value]]
        )
        image_lines[name] = truth.project_to_image(court_endpoints)

    fitted, inliers, intersections = CourtHomography.from_lines(image_lines)

    assert inliers.all()
    assert len(intersections) == 4
    sample = [[-3.05, -6.7], [0.0, 0.0], [3.05, 6.7]]
    np.testing.assert_allclose(
        fitted.project_to_image(sample),
        truth.project_to_image(sample),
        atol=1e-4,
    )


def test_line_calibration_requires_two_lines_per_axis():
    with pytest.raises(ValueError, match="two sidelines and two cross-court"):
        CourtHomography.from_lines(
            {
                "left_doubles_sideline": [[0, 0], [0, 10]],
                "right_doubles_sideline": [[10, 0], [10, 10]],
                "near_short_service": [[0, 2], [10, 2]],
            }
        )


def test_rejects_bad_correspondences():
    with pytest.raises(ValueError, match="at least four"):
        CourtHomography.from_points([[0, 0]] * 3, [[0, 0]] * 3)
    with pytest.raises(ValueError, match="unknown court"):
        CourtHomography.from_landmarks(
            {
                "a": [0, 0],
                "b": [1, 0],
                "c": [1, 1],
                "d": [0, 1],
            }
        )


def test_save_and_load(tmp_path):
    calibration, image, _, _ = synthetic_calibration()
    path = tmp_path / "camera.json"
    calibration.save(
        path,
        {"near_left_doubles": image[0]},
        image_size=(640, 480),
    )
    loaded = CourtHomography.load(path)
    np.testing.assert_allclose(loaded.image_to_court, calibration.image_to_court)
    data = json.loads(path.read_text())
    assert data["image_size"] == [640, 480]
    assert data["coordinate_system"]["units"] == "metres"


def test_overlay_draws_pixels():
    calibration, _, _, _ = synthetic_calibration()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay = draw_court_overlay(image, calibration)
    assert cv2.countNonZero(cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)) > 0
    assert not image.any()
