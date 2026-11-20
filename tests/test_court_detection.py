"""End-to-end geometric tests for the classical review-only detector."""

import cv2
import numpy as np
import pytest

from src.court_detection import (
    _consistency, _court_line_endpoints, _distance, _extract_lines, _finite_overlap,
    _matched_lines, _refine_lines, detect_court, line_evidence,
)
from src.court_projection import (
    COURT_CALIBRATION_LINES, CourtHomography, floor_court_line_segments,
)


def synthetic_court():
    truth, _ = CourtHomography.from_landmarks({
        "near_left_doubles": [70, 450],
        "near_right_doubles": [570, 450],
        "far_right_doubles": [440, 80],
        "far_left_doubles": [200, 80],
    })
    image = np.full((480, 640, 3), [45, 110, 35], dtype=np.uint8)
    for line in floor_court_line_segments():
        points = np.rint(truth.project_to_image(line)).astype(int)
        cv2.line(image, tuple(points[0]), tuple(points[1]), (245, 245, 245), 3)
    return image, truth


@pytest.mark.parametrize("occlusion", ["none", "cropped", "corners"])
def test_recovers_template_with_missing_corners(occlusion):
    image, truth = synthetic_court()
    if occlusion == "cropped":
        image = image[:390].copy()
    elif occlusion == "corners":
        image[395:, :190] = [45, 110, 35]
        image[395:, 450:] = [45, 110, 35]
    result = detect_court(image, validation_frames=[image])
    assert result.status == "needs_review"
    assert result.calibration is not None
    points = [[-2, -4], [2, -4], [-2, 4], [2, 4], [0, 0]]
    error = np.linalg.norm(result.calibration.project_to_image(points)
                           - truth.project_to_image(points), axis=1)
    assert error.max() < 7
    assert result.diagnostics["matched_line_count"] >= 5
    assert len(result.diagnostics["validation_scores"]) == 1
    assert result.diagnostics["hypotheses_evaluated"] <= 8192
    assert "orientation_requires_review" in result.diagnostics["reasons"]
    assert not result.diagnostics["inferred_intersections_are_independent"]
    assert result.diagnostics["coverage"]["court_width_fraction"] > 0.8
    assert result.diagnostics["line_observations"]


def test_oblique_view_with_offscreen_near_corners():
    truth, _ = CourtHomography.from_landmarks({
        "near_left_doubles": [88, 558],
        "near_right_doubles": [988, 441],
        "far_right_doubles": [345, 330],
        "far_left_doubles": [8, 340],
    })
    image = np.full((540, 960, 3), [45, 110, 35], dtype=np.uint8)
    for line in floor_court_line_segments():
        points = np.rint(truth.project_to_image(line)).astype(int)
        cv2.line(image, tuple(points[0]), tuple(points[1]), (245, 245, 245), 3)
    result = detect_court(image)
    assert result.status == "needs_review"
    points = [[-2, -4], [2, -4], [-2, 4], [2, 4], [0, 0]]
    error = np.linalg.norm(result.calibration.project_to_image(points)
                           - truth.project_to_image(points), axis=1)
    assert error.max() < 7


def test_cropped_near_baseline_and_compressed_far_end_are_unobserved_not_contradictory():
    truth, _ = CourtHomography.from_landmarks({
        "near_left_doubles": [-80, 650], "near_right_doubles": [1040, 650],
        "far_right_doubles": [580, 260], "far_left_doubles": [380, 260],
    })
    image = np.full((540, 960, 3), [45, 110, 35], np.uint8)
    lines = []
    for segment in floor_court_line_segments():
        pixels = truth.project_to_image(segment)
        lines.append(pixels)
        a, b = np.rint(pixels).astype(int)
        cv2.line(image, tuple(a), tuple(b), (245, 245, 245), 3)
    # Paint on other courts beyond the far baseline must not masquerade as
    # continuation of this court through a fixed, oversized image-pixel probe.
    for y in (235, 243, 250):
        cv2.line(image, (250, y), (710, y), (245, 245, 245), 3)
    consistency = _consistency(truth.court_to_image, lines, _distance(line_evidence(image)), 6.48)
    probes = {probe["name"]: probe for probe in consistency["junction_probe_diagnostics"]}
    assert consistency["expected_baseline_support"]["near"] is None
    assert probes["near_baseline_center"]["status"] == "outside_observed_region"
    assert probes["far_baseline_center"]["status"] == "unresolved_perspective"
    assert not consistency["baseline_junctions"]
    near = next(j for j in consistency["center_service_junctions"] if j["name"] == "near_short_service_center")
    assert near["incoming_support"] > .7
    assert near["unexpected_continuation"] < .5


def test_refinement_uses_independent_observed_lines():
    _, truth = synthetic_court()
    observations = {name: truth.project_to_image(_court_line_endpoints(name)).tolist()
                    for name in COURT_CALIBRATION_LINES}
    disturbed = truth.court_to_image.copy()
    disturbed[0, 2] += 4
    disturbed[1, 2] -= 3
    refined = _refine_lines(disturbed, observations, 640, 480, 5)
    result = CourtHomography(np.linalg.inv(refined))
    points = [[-2, -4], [2, -4], [-2, 4], [2, 4], [0, 0]]
    np.testing.assert_allclose(result.project_to_image(points),
                               truth.project_to_image(points), atol=1e-3)


def test_observation_cannot_support_two_line_identities():
    _, truth = synthetic_court()
    # A deliberately huge tolerance makes one observation eligible for both
    # axes; it must still count as one independent source segment.
    lines = [truth.project_to_image([[-3.05, -6.7], [-3.05, 6.7]])]
    matched = _matched_lines(truth, lines, lines, tolerance=10000)
    assert len(matched) == 1


def test_neighboring_court_extension_is_not_a_landmark_observation():
    predicted = np.asarray([[0., 0.], [100., 0.]])
    neighbor = np.asarray([[120., 0.], [250., 0.]])
    assert _finite_overlap(predicted, neighbor, 2) is None
    partly_shared = np.asarray([[50., 0.], [250., 0.]])
    np.testing.assert_allclose(_finite_overlap(predicted, partly_shared, 2),
                               [[50, 0], [100, 0]])


def test_baseline_long_service_alias_leaves_distinctive_paint_unexplained():
    image, truth = synthetic_court()
    evidence = line_evidence(image)
    lines = _extract_lines(evidence)
    correct = _consistency(truth.court_to_image, lines, _distance(evidence), 5.76)
    # This transform places both baselines on the doubles long-service lines.
    alias = truth.court_to_image @ np.diag([1, 5.94 / 6.7, 1])
    wrong = _consistency(alias, lines, _distance(evidence), 5.76)
    assert correct["observed_paint_explained"] > wrong["observed_paint_explained"] + .15
    assert correct["topology_score"] > wrong["topology_score"] + .15
    assert wrong["unexplained_segments"]


def test_missing_distinctive_service_junctions_remains_unresolved():
    image, truth = synthetic_court()
    image[:] = [45, 110, 35]
    for line in floor_court_line_segments():
        if line[0][0] == 0 and line[1][0] == 0:
            continue
        points = np.rint(truth.project_to_image(line)).astype(int)
        cv2.line(image, tuple(points[0]), tuple(points[1]), (245, 245, 245), 3)
    result = detect_court(image)
    assert result.status == "failed"
    assert result.calibration is None
    assert "insufficient_distinctive_court_markings" in result.diagnostics["reasons"]


def test_court_region_contract():
    image, _ = synthetic_court()
    with pytest.raises(ValueError, match="court_region"):
        detect_court(image, court_region=np.zeros(image.shape[:2], np.uint8))
    with pytest.raises(ValueError, match="court_region"):
        detect_court(image, court_region=np.ones((10, 10), np.uint8))


def test_coherent_contradictory_interior_markings_are_rejected():
    image, _ = synthetic_court()
    for y in (190, 210, 300, 320, 350, 370):
        cv2.line(image, (180, y), (460, y), (245, 245, 245), 3)
    result = detect_court(image)
    assert result.status == "failed"
    assert result.calibration is None
    assert "contradictory_interior_court_markings" in result.diagnostics["reasons"]
    assert len(result.diagnostics["consistency"]["unexplained_segments"]) >= 3


def test_selected_region_recovers_cropped_court_beside_another_court():
    image, truth = synthetic_court()
    # The near baseline and both near corners are outside this cropped frame. The adjacent
    # court supplies similar, stronger paint and must not become target evidence.
    scene = np.full((390, 1120, 3), [45, 110, 35], dtype=np.uint8)
    scene[:, :640] = image[:390]
    for line in floor_court_line_segments():
        points = np.rint(truth.project_to_image(line) + [550, 0]).astype(int)
        cv2.line(scene, tuple(points[0]), tuple(points[1]), (245, 245, 245), 4)
    region = np.zeros(scene.shape[:2], np.uint8)
    corners = truth.project_to_image([[-3.05, -6.7], [3.05, -6.7],
                                      [3.05, 6.7], [-3.05, 6.7]])
    corners = corners.mean(axis=0) + (corners - corners.mean(axis=0)) * 1.1
    cv2.fillConvexPoly(region, np.rint(corners).astype(np.int32), 255)
    result = detect_court(scene, court_region=region)
    assert result.status == "needs_review"
    assert result.diagnostics["court_region"] == "provided"
    points = [[-2, -4], [2, -4], [-2, 4], [2, 4], [0, 0]]
    errors = np.linalg.norm(result.calibration.project_to_image(points)
                            - truth.project_to_image(points), axis=1)
    assert errors.max() < 7
    assert result.diagnostics["consistency"]["visible_template_inside_region"] >= .75


def test_blank_and_one_direction_fail_without_candidate():
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    for y in (40, 100, 220):
        cv2.line(image, (10, y), (390, y), (255, 255, 255), 3)
    for candidate in (image, np.zeros_like(image)):
        result = detect_court(candidate)
        assert result.status == "failed"
        assert result.calibration is None
        assert not result.image_lines


def test_repeatable_bounded_search():
    image, _ = synthetic_court()
    first = detect_court(image, max_hypotheses=512, seed=12)
    second = detect_court(image, max_hypotheses=512, seed=12)
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics["hypotheses_evaluated"] <= 512
    assert first.calibration is not None
    np.testing.assert_allclose(first.calibration.image_to_court,
                               second.calibration.image_to_court)


def test_validation_shape_and_input_contract():
    image, _ = synthetic_court()
    with pytest.raises(ValueError, match="must match"):
        detect_court(image, [image[:200]], max_hypotheses=512)
    with pytest.raises(ValueError, match="uint8 BGR"):
        line_evidence(image[:, :, 0])
    with pytest.raises(ValueError, match="positive"):
        detect_court(image, max_hypotheses=0)
