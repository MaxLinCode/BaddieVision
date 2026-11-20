"""Geometry-first selection among independently validated court hypotheses.

Automatic proposal masks only control which lines generate hypotheses. Every
candidate is validated against the original evidence in its finite footprint.
Scores are ranking qualities, not calibrated acceptance probabilities.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, product

import cv2
import numpy as np

from src.court_projection import COURT_CALIBRATION_LINES, CourtHomography, _fit_homography_dlt
from src.court_detection import (
    CourtDetection, _consistency, _direction_families, _distance, _extract_lines,
    _legacy_line_evidence, _legacy_surface_region,
    _intersection, _line_diagnostics, _matched_lines, _plausible, _refine_lines,
    _sample_template, _score, _surface_region, line_evidence,
)


SELECTION_WEIGHTS = {"geometry": .70, "coverage": .20, "visible_area": .07, "centrality": .03}
COURT_CORNERS = np.array([[-3.05, -6.7], [3.05, -6.7], [3.05, 6.7], [-3.05, 6.7]])


def visible_footprint(matrix: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, float]:
    """Clip the finite court to the image; never reward offscreen extrapolation."""
    calibration = CourtHomography(np.linalg.inv(matrix))
    polygon = calibration.project_to_image(COURT_CORNERS).astype(np.float32)
    bounds = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    area, clipped = cv2.intersectConvexConvex(polygon, bounds)
    if clipped is None or area <= 0:
        return np.empty((0, 2)), 0., 0.
    clipped = clipped.reshape(-1, 2)
    moments = cv2.moments(clipped)
    center = np.array([moments["m10"], moments["m01"]]) / max(moments["m00"], 1e-12)
    centrality = float(np.clip(1 - np.linalg.norm((center - [width / 2, height / 2])
                                                / [width / 2, height / 2]) / np.sqrt(2), 0, 1))
    return clipped, float(np.clip(area / (width * height), 0, 1)), centrality


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3 or len(second) < 3:
        return 0.
    # Frame clipping leaves tiny negative coordinates and nearly collinear edge
    # vertices. Normalize subpixel noise before OpenCV's float32 intersection.
    first = cv2.convexHull(np.float32(np.round(first, 3))).reshape(-1, 2)
    second = cv2.convexHull(np.float32(np.round(second, 3))).reshape(-1, 2)
    first_area, second_area = cv2.contourArea(first), cv2.contourArea(second)
    intersection, _ = cv2.intersectConvexConvex(first, second)
    reverse_intersection, _ = cv2.intersectConvexConvex(second, first)
    if abs(intersection - reverse_intersection) > max(1e-3, min(first_area, second_area) * 1e-5):
        # Rare shared-edge numerical failure: use convex clipping in float64.
        clipped = np.float64(first)
        for a, b in zip(second, np.roll(second, -1, axis=0)):
            if not len(clipped):
                break
            edge = np.float64(b - a)

            def signed(point):
                return edge[0] * (point[1] - a[1]) - edge[1] * (point[0] - a[0])

            output = []
            previous = clipped[-1]
            previous_distance = signed(previous)
            for current in clipped:
                current_distance = signed(current)
                if (previous_distance >= -1e-8) != (current_distance >= -1e-8):
                    fraction = previous_distance / (previous_distance - current_distance)
                    output.append(previous + fraction * (current - previous))
                if current_distance >= -1e-8:
                    output.append(current)
                previous, previous_distance = current, current_distance
            clipped = np.asarray(output, dtype=np.float64).reshape(-1, 2)
        intersection = (abs(np.dot(clipped[:, 0], np.roll(clipped[:, 1], 1))
                            - np.dot(clipped[:, 1], np.roll(clipped[:, 0], 1))) / 2
                        if len(clipped) >= 3 else 0.)
    intersection = float(np.clip(intersection, 0, min(first_area, second_area)))
    union = first_area + second_area - intersection
    return float(intersection / max(union, 1e-12))


def ranking_components(matrix, forward, matched, consistency, width, height):
    polygon, area, centrality = visible_footprint(matrix, width, height)
    geometry = (.4 * forward + .35 * consistency["observed_paint_explained"]
                + .25 * consistency["topology_score"])
    axes = [COURT_CALIBRATION_LINES[name][0] for name in matched]
    # Count named independent markings, balancing both projective directions.
    coverage = .5 * min(1., axes.count("x") / 4) + .5 * min(1., axes.count("y") / 6)
    components = {"geometry": float(geometry), "coverage": float(coverage),
                  "visible_area": area, "centrality": centrality}
    quality = sum(SELECTION_WEIGHTS[name] * value for name, value in components.items())
    return float(quality), components, polygon


def geometry_rejection(matched: dict, consistency: dict) -> str | None:
    """Structural validity is a prerequisite, independent of size and center."""
    consistency.pop("rejection_detail", None)

    def reject(reason, gate, **values):
        consistency["rejection_detail"] = {"gate": gate, **values}
        return reason

    axes = [COURT_CALIBRATION_LINES[name][0] for name in matched]
    if axes.count("x") < 2 or axes.count("y") < 2 or len(matched) < 5:
        return reject("insufficient_independent_line_support", "independent_line_support",
                      sidelines=axes.count("x"), cross_lines=axes.count("y"))
    if consistency["visible_template_inside_region"] < .75:
        return reject("template_outside_selected_court_region", "target_region_coverage")
    if any(support is not None and support < .55
           for support in consistency.get("expected_short_service_support", {}).values()):
        return reject("contradictory_interior_court_markings", "short_service_paint_support",
                      support=consistency["expected_short_service_support"], threshold=.55)
    if any(support is not None and support < .3
           for support in consistency.get("expected_baseline_support", {}).values()):
        return reject("contradictory_interior_court_markings", "baseline_paint_support",
                      support=consistency["expected_baseline_support"], threshold=.3)
    baseline_contradictions = [junction for junction in consistency.get("baseline_junctions", [])
                               if junction["incoming_support"] >= .6
                               and junction["crossbar_support"] >= .6
                               and junction["unexpected_continuation"] >= .6]
    if baseline_contradictions:
        consistency["baseline_endpoint_contradictions"] = baseline_contradictions
        return reject("contradictory_interior_court_markings", "baseline_center_continuation")
    continuing_boundaries = Counter(junction["baseline"]
                                    for junction in consistency.get("boundary_junctions", [])
                                    if min(junction["incoming_support"],
                                           junction["unexpected_continuation"]) >= .6)
    if any(count >= 3 for count in continuing_boundaries.values()):
        consistency["truncated_court_boundary"] = dict(continuing_boundaries)
        return reject("contradictory_interior_court_markings", "multiple_sidelines_continue_beyond_baseline")
    service_contradictions = [junction for junction in consistency["center_service_junctions"]
                              if junction["incoming_support"] >= .6
                              and junction["unexpected_continuation"] >= .6]
    if service_contradictions:
        consistency["service_endpoint_contradictions"] = service_contradictions
        return reject("contradictory_interior_court_markings", "service_center_continuation")
    distinctive = [junction for junction in consistency["center_service_junctions"]
                   if junction["name"].replace("_center", "") in matched
                   and junction["incoming_support"] >= .55
                   and junction["crossbar_support"] >= .55
                   and junction["unexpected_continuation"] <= .5]
    if not distinctive:
        return reject("insufficient_distinctive_court_markings", "distinctive_service_junction")
    consistency["distinctive_center_junctions"] = len(distinctive)
    consistency["center_service_evidence"] = "independent_short_service_and_T_junction"
    explained = consistency["observed_paint_explained"]
    if ((explained < .6 and consistency["independent_segments_checked"] >= 5)
            or (explained < .7 and len(consistency["unexplained_segments"]) >= 3)):
        return reject("contradictory_interior_court_markings", "unexplained_paint",
                      explained_fraction=explained, unexplained_segments=len(consistency["unexplained_segments"]))
    return None


def _generate(mask, budget, seed, stage, *, adaptive=False):
    lines = _extract_lines(mask, max_lines=28 if adaptive else 20, balanced=adaptive)
    families = _direction_families(lines)
    meta = {"stage": stage, "budget": budget, "attempted": 0, "evaluated": 0,
            "detected_lines": len(lines), "direction_families": len(families)}
    if budget < 1 or len(families) < 2:
        return [], meta
    height, width = mask.shape
    distance = _distance(mask)
    points, slices = _sample_template()
    tolerance = max(3., min(width, height) * .012)
    x_values = sorted({v for axis, v in COURT_CALIBRATION_LINES.values() if axis == "x"})
    y_values = sorted({v for axis, v in COURT_CALIBRATION_LINES.values() if axis == "y"})
    x_pairs = list(combinations(x_values, 2))
    y_pairs = [(a, b) for a, b in combinations(y_values, 2) if b - a > 3]
    rectangles = set()
    for first, second in product(families, repeat=2):
        if first != second:
            for sides, crosses in product(combinations(first, 2), combinations(second, 2)):
                if len(set(sides + crosses)) == 4:
                    rectangles.add((sides, crosses))
    rectangles = sorted(rectangles)
    template_count = len(x_pairs) * len(y_pairs)
    total = len(rectangles) * template_count
    meta["available"] = total
    order = np.random.default_rng(seed).choice(total, min(total, budget), replace=False)
    candidates = []
    for index in order:
        meta["attempted"] += 1
        rectangle_index, template_index = divmod(int(index), template_count)
        x_index, y_index = divmod(template_index, len(y_pairs))
        sides, crosses = rectangles[rectangle_index]
        intersections = [_intersection(lines[a], lines[b]) for b in crosses for a in sides]
        if any(point is None for point in intersections):
            continue
        quad = np.asarray(intersections).reshape(2, 2, 2)
        quad = quad[np.argsort(-quad[:, :, 1].mean(axis=1))]
        quad = np.asarray([row[np.argsort(row[:, 0])] for row in quad])
        left, right = x_pairs[x_index]
        near, far = y_pairs[y_index]
        try:
            matrix = _fit_homography_dlt(np.array([[left, near], [right, near],
                                                  [left, far], [right, far]]), quad.reshape(4, 2))
            if not _plausible(matrix, width, height):
                continue
            forward, support = _score(matrix, distance, points, slices, tolerance)
            meta["evaluated"] += 1
            if forward > .25:
                matched = _matched_lines(CourtHomography(np.linalg.inv(matrix)), lines, lines, tolerance)
                axes = [COURT_CALIBRATION_LINES[name][0] for name in matched]
                if axes.count("x") >= 2 and axes.count("y") >= 2 and len(matched) >= 5:
                    polygon, area, center = visible_footprint(matrix, width, height)
                    if area < .10:
                        continue
                    candidates.append({"matrix": matrix, "forward": forward, "support": support,
                                       "polygon": polygon, "quick": .75 * forward + .20 * area + .05 * center,
                                       "stage": stage})
        except (ValueError, np.linalg.LinAlgError, cv2.error):
            continue
    return candidates, meta


def _groups(candidates):
    groups = []
    for candidate in sorted(candidates, key=lambda c: -c.get("quality", c["quick"])):
        group = next((g for g in groups if polygon_iou(candidate["polygon"], g[0]["polygon"]) >= .6), None)
        if group is None:
            groups.append([candidate])
        else:
            group.append(candidate)
    return groups


def _diverse_shortlist(candidates, groups_limit=8, per_group=24):
    return [candidate for group in _groups(candidates)[:groups_limit] for candidate in group[:per_group]]


def _proposal_regions(candidates, mask):
    """Derive local proposal windows from coarse footprints, without labels."""
    regions = []
    for group in _groups(candidates)[:3]:
        polygon = group[0]["polygon"]
        center = polygon.mean(axis=0)
        polygon = center + (polygon - center) * 1.12
        region = np.zeros_like(mask)
        cv2.fillConvexPoly(region, np.rint(polygon).astype(np.int32), 255)
        regions.append(region)
    return regions


def _select_court(image: np.ndarray, validation_frames: list[np.ndarray] | None = None,
                  *, max_hypotheses: int = 8192, seed: int = 0,
                  court_region: np.ndarray | None = None,
                  adaptive: bool = False) -> CourtDetection:
    if max_hypotheses < 1:
        raise ValueError("max_hypotheses must be positive")
    proposal_evidence = line_evidence if adaptive else _legacy_line_evidence
    mask = proposal_evidence(image)
    if court_region is not None:
        if court_region.shape != image.shape[:2] or court_region.dtype != np.uint8 or not np.any(court_region):
            raise ValueError("court_region must be a nonempty uint8 mask matching the image")
        mask = np.where(court_region > 0, mask, 0).astype(np.uint8)
    for frame in (validation_frames or [])[:8]:
        if frame.shape != image.shape:
            raise ValueError("validation frames must match background image shape")
    surface = _surface_region(image) if adaptive else _legacy_surface_region(image)
    if surface is not None:
        mask = np.where(surface > 0, mask, 0).astype(np.uint8)
    height, width = mask.shape
    full_lines = _extract_lines(mask, max_lines=28 if adaptive else 20, balanced=adaptive)
    # The floor estimate guides candidate generation. Paint validation uses the
    # broad original evidence so a second floor color or reflection cannot erase
    # a genuine court marking from the support score.
    scoring_mask = _legacy_line_evidence(image) if adaptive else mask
    if adaptive and court_region is not None:
        scoring_mask = np.where(court_region > 0, scoring_mask, 0).astype(np.uint8)
    distance = _distance(scoring_mask)
    tolerance = max(3., min(width, height) * .012)
    points, slices = _sample_template()
    diagnostics = {"method": "geometry_first_floor_lines_v4", "thresholds": "provisional_not_benchmark_validated",
                   "orientation": "near_at_image_bottom_left_to_right_requires_review",
                   "seed": seed, "max_hypotheses": max_hypotheses,
                   "detected_lines": len(full_lines), "direction_families": len(_direction_families(full_lines)),
                   "surface_prior": ("lower_half_dominant_color_proposals_only" if adaptive else
                                     "large_connected_chromatic_region") if surface is not None else "unused",
                   "proposal_mode": "adaptive" if adaptive else "baseline",
                   "court_region": "provided" if court_region is not None else "unrestricted", "reasons": []}
    # Preserve the useful exhaustive small-budget behavior. Larger unrestricted
    # searches reserve half the total budget for local proposal re-detection.
    global_budget = max_hypotheses if court_region is not None or max_hypotheses < 2048 else max_hypotheses // 2
    candidates, stage = _generate(mask, global_budget, seed, "global", adaptive=adaptive)
    stages = [stage]
    regions = _proposal_regions(candidates, mask) if global_budget < max_hypotheses else []
    remaining = max_hypotheses - stage["attempted"]
    for index, region in enumerate(regions):
        budget = remaining // (len(regions) - index)
        local, local_meta = _generate(np.where(region > 0, mask, 0).astype(np.uint8),
                                      budget, seed + index + 1, f"automatic_region_{index + 1}", adaptive=adaptive)
        candidates.extend(local)
        stages.append(local_meta)
        remaining -= local_meta["attempted"]
    diagnostics.update({"search_stages": stages, "hypotheses_evaluated": sum(s["evaluated"] for s in stages),
                        "hypotheses_attempted": sum(s["attempted"] for s in stages),
                        "hypotheses_available": sum(s.get("available", 0) for s in stages)})
    rejected = Counter()
    rejected_details = {}
    rejected_gates = Counter()
    rejected_candidates = []
    refinement_rescues = 0
    valid = []
    for candidate in _diverse_shortlist(candidates):
        matrix = candidate["matrix"]
        try:
            # Local masks do not participate here: validate original full evidence.
            calibration = CourtHomography(np.linalg.inv(matrix))
            matched = _matched_lines(calibration, full_lines, full_lines, tolerance)
            consistency = _consistency(matrix, full_lines, distance, tolerance, court_region)
            reason = geometry_rejection(matched, consistency)
            # Four-line hypotheses can be slightly displaced yet identify enough
            # real markings for refinement. Validate the refined fit before
            # discarding it; thresholds and final evidence remain unchanged.
            axes = [COURT_CALIBRATION_LINES[name][0] for name in matched]
            if reason and axes.count("x") >= 2 and axes.count("y") >= 2 and len(matched) >= 5:
                try:
                    refined = _refine_lines(matrix, matched, width, height, tolerance)
                    if _plausible(refined, width, height):
                        fit = CourtHomography(np.linalg.inv(refined))
                        refined_matches = _matched_lines(fit, full_lines, full_lines, tolerance)
                        refined_consistency = _consistency(refined, full_lines, distance, tolerance, court_region)
                        if geometry_rejection(refined_matches, refined_consistency) is None:
                            matrix, matched, consistency, reason = refined, refined_matches, refined_consistency, None
                            refinement_rescues += 1
                except (ValueError, np.linalg.LinAlgError):
                    pass
            if reason:
                rejected[reason] += 1
                detail = consistency.get("rejection_detail", {})
                rejected_gates[detail.get("gate", reason)] += 1
                if len(rejected_candidates) < 24:
                    rejected_candidates.append({"reason": reason, "detail": detail,
                                                "stage": candidate["stage"], "court_to_image": matrix.tolist(),
                                                "visible_polygon": candidate["polygon"].tolist(),
                                                "matched_lines": list(matched)})
                if reason not in rejected_details:
                    rejected_details[reason] = consistency
                continue
            forward, support = _score(matrix, distance, points, slices, tolerance)
            quality, components, polygon = ranking_components(matrix, forward, matched, consistency, width, height)
            if components["visible_area"] < .10:
                rejected["insufficient_visible_court_area"] += 1
                continue
            # Refine each valid court, so a lower-ranked valid candidate can win.
            try:
                refined = _refine_lines(matrix, matched, width, height, tolerance)
                if _plausible(refined, width, height):
                    fit = CourtHomography(np.linalg.inv(refined))
                    refined_matches = _matched_lines(fit, full_lines, full_lines, tolerance)
                    refined_consistency = _consistency(refined, full_lines, distance, tolerance, court_region)
                    if geometry_rejection(refined_matches, refined_consistency) is None:
                        f, sup = _score(refined, distance, points, slices, tolerance)
                        q, comp, poly = ranking_components(refined, f, refined_matches, refined_consistency, width, height)
                        if q >= quality and comp["visible_area"] >= .10:
                            matrix, matched, consistency = refined, refined_matches, refined_consistency
                            forward, support, quality, components, polygon = f, sup, q, comp, poly
            except (ValueError, np.linalg.LinAlgError):
                pass
            valid.append({**candidate, "matrix": matrix, "matched": matched, "consistency": consistency,
                          "forward": forward, "support": support, "quality": quality,
                          "components": components, "polygon": polygon})
        except (ValueError, np.linalg.LinAlgError, cv2.error):
            rejected["invalid_geometry"] += 1
    diagnostics["selection"] = {"weights": SELECTION_WEIGHTS.copy(), "score_type": "ranking_quality_not_probability",
                                "geometry_rejected": dict(rejected), "rejection_gates": dict(rejected_gates),
                                "rejected_candidates": rejected_candidates, "refinement_rescues": refinement_rescues,
                                "valid_hypotheses": len(valid), "candidates": []}
    if not valid:
        reason = ("contradictory_interior_court_markings" if rejected["contradictory_interior_court_markings"]
                  else rejected.most_common(1)[0][0] if rejected else "no_supported_court_template")
        diagnostics["reasons"].append(reason)
        diagnostics["reasons"].extend(other for other in rejected if other != reason)
        if reason in rejected_details:
            diagnostics["consistency"] = rejected_details[reason]
        return CourtDetection(None, {}, diagnostics, "failed")
    groups = _groups(valid)
    representatives = [group[0] for group in groups]
    best = representatives[0]
    for index, candidate in enumerate(representatives[:8]):
        diagnostics["selection"]["candidates"].append({
            "rank": index + 1, "court_group": index, "score": candidate["quality"],
            "components": candidate["components"], "visible_polygon": candidate["polygon"].tolist(),
            "matched_lines": list(candidate["matched"]), "stage": candidate["stage"], "accepted_geometry": True})
    court_margin = best["quality"] - representatives[1]["quality"] if len(representatives) > 1 else None
    check_points = np.array([[-2, -4], [2, -4], [-2, 4], [2, 4], [0, 0]])
    calibration = CourtHomography(np.linalg.inv(best["matrix"]))
    projected = calibration.project_to_image(check_points)
    aliases = [c for c in groups[0][1:] if np.linalg.norm(
        CourtHomography(np.linalg.inv(c["matrix"])).project_to_image(check_points) - projected, axis=1).mean() > tolerance * 3]
    identity_margin = best["quality"] - aliases[0]["quality"] if aliases else None
    diagnostics["selection"].update({"court_selection_margin": court_margin, "line_identity_margin": identity_margin,
                                      "physical_court_count": len(groups), "selected_group": 0})
    frame_scores = []
    for frame in (validation_frames or [])[:8]:
        evidence = _legacy_line_evidence(frame)
        if surface is not None and not adaptive:
            evidence = np.where(surface > 0, evidence, 0).astype(np.uint8)
        if court_region is not None:
            evidence = np.where(court_region > 0, evidence, 0).astype(np.uint8)
        f, _ = _score(best["matrix"], _distance(evidence), points, slices, tolerance)
        frame_scores.append(f)
    observations, coverage = _line_diagnostics(calibration, best["matched"], distance, tolerance)
    reasons = ["orientation_requires_review", "acceptance_thresholds_unvalidated"]
    if court_margin is not None and court_margin < .03:
        reasons.append("ambiguous_main_court")
    if identity_margin is not None and identity_margin < .03:
        reasons.append("ambiguous_template")
    if frame_scores and min(frame_scores) < best["forward"] * .6:
        reasons.append("weak_temporal_line_support")
    diagnostics.update({"score": best["quality"], "line_support": best["support"],
                        "matched_line_count": len(best["matched"]), "matched_lines": list(best["matched"]),
                        "alternative_score": aliases[0]["quality"] if aliases else None,
                        "alternative_margin": identity_margin, "validation_scores": frame_scores,
                        "reasons": reasons, "inferred_intersections_are_independent": False,
                        "line_observations": observations, "coverage": coverage, "consistency": best["consistency"]})
    return CourtDetection(calibration, best["matched"], diagnostics, "needs_review")


def select_court(image: np.ndarray, validation_frames: list[np.ndarray] | None = None,
                 *, max_hypotheses: int = 8192, seed: int = 0,
                 court_region: np.ndarray | None = None) -> CourtDetection:
    baseline = _select_court(image, validation_frames, max_hypotheses=max_hypotheses,
                             seed=seed, court_region=court_region)
    if baseline.status != "failed":
        return baseline
    adaptive = _select_court(image, validation_frames, max_hypotheses=max_hypotheses,
                             seed=seed, court_region=court_region, adaptive=True)
    if adaptive.status != "failed":
        adaptive.diagnostics["baseline_attempt"] = {
            "status": baseline.status, "reasons": baseline.diagnostics.get("reasons", [])}
        return adaptive
    baseline.diagnostics["adaptive_attempt"] = {
        "status": adaptive.status, "reasons": adaptive.diagnostics.get("reasons", []),
        "detected_lines": adaptive.diagnostics.get("detected_lines"),
        "selection": adaptive.diagnostics.get("selection", {})}
    return baseline
