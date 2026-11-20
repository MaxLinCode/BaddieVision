"""Local line contrast should adapt to the surrounding court surface."""

import cv2
import numpy as np
import pytest

from src.court_detection import _surface_region, line_evidence


@pytest.mark.parametrize("floor", [(40, 120, 180), (150, 80, 35), (110, 110, 110)])
def test_pale_paint_survives_colored_reflection(floor):
    image = np.full((160, 240, 3), floor, dtype=np.uint8)
    cv2.line(image, (20, 45), (220, 45), (225, 225, 225), 3)
    cv2.line(image, (20, 115), (220, 115), (60, 180, 60), 3)
    evidence = line_evidence(image)
    assert np.mean(evidence[45, 30:210] > 0) > 0.9
    assert np.mean(evidence[115, 30:210] > 0) < 0.1


def test_faint_paint_on_bright_colored_floor():
    image = np.full((160, 240, 3), (145, 195, 205), dtype=np.uint8)
    cv2.line(image, (20, 80), (220, 80), (195, 202, 205), 3)
    evidence = line_evidence(image)
    assert np.mean(evidence[80, 30:210] > 0) > 0.9


@pytest.mark.parametrize("floor,wall", [((40, 120, 220), (70, 75, 80)),
                                       ((100, 170, 70), (120, 90, 60))])
def test_surface_prior_learns_lower_image_color(floor, wall):
    image = np.full((200, 300, 3), wall, dtype=np.uint8)
    image[90:] = floor
    cv2.line(image, (20, 130), (280, 130), (240, 240, 240), 3)
    region = _surface_region(image)
    assert region is not None
    assert region[160, 150] > 0
    assert region[30, 150] == 0
    assert region[130, 150] > 0


def test_surface_prior_declines_achromatic_scene():
    image = np.full((200, 300, 3), 120, dtype=np.uint8)
    assert _surface_region(image) is None


def test_surface_prior_keeps_two_floor_colors():
    image = np.full((200, 300, 3), (70, 75, 80), dtype=np.uint8)
    image[90:, :150] = (40, 120, 220)
    image[90:, 150:] = (100, 170, 70)
    cv2.line(image, (20, 130), (280, 130), (240, 240, 240), 3)
    region = _surface_region(image)
    assert region is not None
    assert region[160, 70] > 0
    assert region[160, 220] > 0
    assert region[30, 70] == 0
