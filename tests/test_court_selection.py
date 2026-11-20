"""Geometry cannot be rescued by court-size or centrality priors."""

import numpy as np
import pytest
import copy
import cv2

from src.court_selection import (
    SELECTION_WEIGHTS, _diverse_shortlist, geometry_rejection,
    ranking_components, visible_footprint,
    polygon_iou,
)
from src.court_projection import CourtHomography, floor_court_line_segments


def fixture():
    calibration, _ = CourtHomography.from_landmarks({
        "near_left_doubles": [70, 450], "near_right_doubles": [570, 450],
        "far_right_doubles": [440, 80], "far_left_doubles": [200, 80],
    })
    names = ["left_doubles_sideline", "right_doubles_sideline", "left_singles_sideline",
             "right_singles_sideline", "near_baseline", "far_baseline",
             "near_short_service", "far_short_service"]
    matched = dict.fromkeys(names)
    consistency = {"observed_paint_explained": .95, "topology_score": .95,
                   "independent_segments_checked": 8, "unexplained_segments": [],
                   "visible_template_inside_region": 1.,
                   "center_service_junctions": [{"name": "near_short_service_center",
                     "incoming_support": 1., "crossbar_support": 1., "unexpected_continuation": 0.}]}
    return calibration.court_to_image, matched, consistency


def test_geometry_is_dominant_and_scores_are_normalized():
    matrix, matched, consistency = fixture()
    score, components, polygon = ranking_components(matrix, .95, matched, consistency, 640, 480)
    assert sum(SELECTION_WEIGHTS.values()) == pytest.approx(1)
    assert SELECTION_WEIGHTS["geometry"] > sum(v for k, v in SELECTION_WEIGHTS.items() if k != "geometry")
    assert 0 < score <= 1
    assert all(0 <= value <= 1 for value in components.values())
    assert len(polygon) >= 3


def test_offscreen_extrapolated_area_does_not_inflate_visible_area():
    first = np.array([[1000., 0, 320], [0, -1000, 240], [0, 0, 1]])
    second = first @ np.diag([10., 10., 1])
    polygon, area, center = visible_footprint(first, 640, 480)
    _, larger_area, _ = visible_footprint(second, 640, 480)
    assert larger_area == pytest.approx(area, abs=1e-5)
    assert 0 < area <= 1
    assert center > .99
    assert np.all(polygon >= 0)
    assert np.all(polygon <= [639, 479])


def test_contradictory_large_central_court_is_rejected_before_ranking():
    _, matched, consistency = fixture()
    consistency.update(observed_paint_explained=.55, unexplained_segments=[{}, {}, {}])
    assert geometry_rejection(matched, consistency) == "contradictory_interior_court_markings"
    consistency.update(observed_paint_explained=.95, unexplained_segments=[])
    assert geometry_rejection(matched, consistency) is None


def test_identity_aliases_cannot_exhaust_physical_court_shortlist():
    first = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    second = first + [200, 0]
    aliases = [{"polygon": first + offset, "quick": .99 - offset * .001} for offset in range(12)]
    other = {"polygon": second, "quick": .8}
    selected = _diverse_shortlist([*aliases, other], groups_limit=2, per_group=2)
    assert len(selected) == 3
    assert any(candidate is other for candidate in selected)


def test_cropped_court_overlap_is_symmetric_along_shared_image_edges():
    first = np.array([[-1e-14, 450], [358, 275], [571, 275], [959, 465],
                      [959, 536], [208, 539], [0, 539]])
    second = np.array([[-1e-15, 414], [331, 270], [588, 272], [959, 431],
                       [959, 533], [236, 539], [0, 539]])
    assert polygon_iou(first, second) > .8
    assert polygon_iou(first, second) == pytest.approx(polygon_iou(second, first), abs=1e-6)


def test_multiple_sidelines_continuing_past_a_baseline_reject_a_partial_court():
    _, matched, consistency = fixture()
    consistency["boundary_junctions"] = [
        {"baseline": "near", "incoming_support": 1., "unexpected_continuation": 1.,
         "crossbar_support": 0.} for _ in range(3)]
    assert geometry_rejection(matched, consistency) == "contradictory_interior_court_markings"


@pytest.mark.parametrize("rescue_by_refinement", [False, True])
def test_valid_runner_up_survives_a_higher_scoring_invalid_candidate(monkeypatch, rescue_by_refinement):
    from src import court_selection as selection

    matrix, matched, consistency = fixture()
    bad = matrix.copy()
    bad[0, 2] += 2
    polygon, _, _ = visible_footprint(matrix, 640, 480)
    candidates = [{"matrix": bad, "forward": 1., "support": [1.], "polygon": polygon,
                   "quick": .99, "stage": "global"},
                  {"matrix": matrix, "forward": .9, "support": [.9], "polygon": polygon,
                   "quick": .9, "stage": "global"}]
    lines = [np.array([[10., 10.], [200., 10.]]), np.array([[10., 100.], [200., 100.]]),
             np.array([[10., 10.], [10., 100.]]), np.array([[200., 10.], [200., 100.]])]
    monkeypatch.setattr(selection, "_extract_lines", lambda mask, *, max_lines, balanced: lines)
    monkeypatch.setattr(selection, "_generate", lambda *args, adaptive: (copy.deepcopy(candidates), {
        "stage": "global", "budget": 512, "attempted": 2, "evaluated": 2}))
    monkeypatch.setattr(selection, "_matched_lines", lambda *args: matched.copy())
    monkeypatch.setattr(selection, "_score", lambda *args: (.9, [.9]))
    monkeypatch.setattr(selection, "_line_diagnostics", lambda *args: ({}, {}))

    def evidence(candidate, *args):
        value = copy.deepcopy(consistency)
        if np.allclose(candidate, bad):
            value.update(observed_paint_explained=.4, unexplained_segments=[{}, {}, {}])
        return value

    def no_refinement(*args):
        if rescue_by_refinement:
            return matrix
        raise ValueError("retain initial fit")

    monkeypatch.setattr(selection, "_consistency", evidence)
    monkeypatch.setattr(selection, "_refine_lines", no_refinement)
    result = selection.select_court(np.zeros((480, 640, 3), np.uint8), max_hypotheses=512)
    assert result.status == "needs_review"
    np.testing.assert_allclose(result.calibration.court_to_image, matrix)
    assert result.diagnostics["selection"]["geometry_rejected"].get("contradictory_interior_court_markings", 0) == (0 if rescue_by_refinement else 1)
    assert result.diagnostics["selection"]["refinement_rescues"] == int(rescue_by_refinement)


def test_automatic_selection_finds_main_court_without_a_manual_region():
    from src.court_detection import detect_court

    image = np.full((512, 1000, 3), [45, 110, 35], np.uint8)
    names = ["near_left_doubles", "near_right_doubles", "far_right_doubles", "far_left_doubles"]
    for corners in ([[0, 440], [220, 440], [180, 140], [40, 140]],
                    [[240, 480], [940, 480], [760, 100], [400, 100]]):
        target, _ = CourtHomography.from_landmarks(dict(zip(names, corners)))
        for line in floor_court_line_segments():
            endpoints = np.rint(target.project_to_image(line)).astype(int)
            cv2.line(image, tuple(endpoints[0]), tuple(endpoints[1]), (245, 245, 245), 3)
    result = detect_court(image)
    assert result.status == "needs_review"
    points = [[-2, -4], [2, -4], [-2, 4], [2, 4], [0, 0]]
    error = np.linalg.norm(result.calibration.project_to_image(points) - target.project_to_image(points), axis=1)
    assert error.max() < 2
    assert result.diagnostics["court_region"] == "unrestricted"
    assert result.diagnostics["hypotheses_attempted"] <= 8192
    assert result.diagnostics["selection"]["candidates"][0]["components"]["coverage"] > .8
