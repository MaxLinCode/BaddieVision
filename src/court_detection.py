"""Classical, bounded court-template search on a static-camera background.

This initial detector proposes calibrations for review. Court geometry alone
cannot identify the physical near end of a symmetric court, and the quality
thresholds have not yet been calibrated on a source-disjoint real-video set.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from src.court_projection import (
    COURT_CALIBRATION_LINES,
    CourtHomography,
    _transform_numpy,
    floor_court_line_segments,
)


@dataclass
class CourtDetection:
    calibration: CourtHomography | None
    image_lines: dict[str, list[list[float]]]
    diagnostics: dict
    status: str


def line_evidence(image: np.ndarray) -> np.ndarray:
    """Bright local line contrast, independent of a particular surface color."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("court image must be a uint8 BGR image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = max(9, int(min(gray.shape) * 0.035) | 1)
    local = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT, np.ones((size, size), dtype=np.uint8)
    )
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    chroma = cv2.morphologyEx(
        255 - saturation, cv2.MORPH_TOPHAT, np.ones((size, size), dtype=np.uint8)
    )
    # Compare each candidate with its own surroundings. A colored reflection
    # can be bright in gray while adding little or no light to one color
    # channel; white paint generally brightens all channels, or is much less
    # saturated than the nearby floor. Neither cue assumes a floor hue.
    nearby = cv2.blur(image.astype(np.float32), (size, size))
    channel_gain = np.min(image.astype(np.float32) - nearby, axis=2)
    nearby_saturation = cv2.blur(saturation.astype(np.float32), (size, size))
    pale = saturation.astype(np.float32) < np.maximum(45.0, nearby_saturation * 0.65)
    contrast = ((local > 7) & ((channel_gain > 3) | pale)) | ((chroma > 12) & pale)
    return np.where(contrast & (gray > 105), 255, 0).astype(np.uint8)


def _legacy_line_evidence(image: np.ndarray) -> np.ndarray:
    """Broad paint evidence retained for geometry validation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = max(9, int(min(gray.shape) * 0.035) | 1)
    local = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((size, size), np.uint8))
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    chroma = cv2.morphologyEx(255 - saturation, cv2.MORPH_TOPHAT,
                              np.ones((size, size), np.uint8))
    return np.where(((local > 7) | (chroma > 12)) & (gray > 105), 255, 0).astype(np.uint8)


def _legacy_surface_region(image: np.ndarray) -> np.ndarray | None:
    """Original broad surface mask, used to preserve existing proposal results."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colored = ((hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 55)).astype(np.uint8)
    size = max(9, int(min(image.shape[:2]) * 0.05) | 1)
    colored = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(colored, 8)
    if count < 2:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == index).astype(np.uint8)
    area = int(region.sum())
    if area < .15 * region.size or region[image.shape[0] // 2:].sum() < area * .6:
        return None
    return cv2.dilate(region, np.ones((9, 9), np.uint8))


def _surface_region(image: np.ndarray) -> np.ndarray | None:
    """Find one or two floor colors learned from the lower half of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    colored = (saturation > 45) & (value > 55)
    lower = colored[image.shape[0] // 2:]
    if lower.sum() < 0.1 * lower.size:
        return None
    histogram = np.bincount(hue[image.shape[0] // 2:][lower], minlength=180)
    # A circular hue window tolerates both illumination changes and the hue
    # wraparound, without selecting a particular court color.
    smoothed = sum(np.roll(histogram, offset) for offset in range(-10, 11))
    peaks = []
    available = smoothed.copy()
    for _ in range(2):
        peak = int(np.argmax(available))
        if available[peak] < .18 * histogram.sum():
            break
        peaks.append(peak)
        for offset in range(-24, 25):
            available[(peak + offset) % 180] = 0
    if not peaks:
        return None
    region = np.zeros(hue.shape, dtype=np.uint8)
    for peak in peaks:
        hue_distance = np.abs(hue.astype(np.int16) - peak)
        hue_distance = np.minimum(hue_distance, 180 - hue_distance)
        similar = (hue_distance < 16) & colored
        lower_similar = similar[image.shape[0] // 2:]
        if lower_similar.sum() < .05 * lower_similar.size:
            continue
        # Each color has its own brightness floor; a bright outer court does
        # not force a dim inner court out of the proposal region.
        minimum_value = np.percentile(value[image.shape[0] // 2:][lower_similar], 10)
        region |= (similar & (value >= minimum_value)).astype(np.uint8)
    size = max(9, int(min(image.shape[:2]) * 0.05) | 1)
    opening = max(3, int(min(image.shape[:2]) * 0.017) | 1)
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, np.ones((opening, opening), np.uint8))
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(region, 8)
    if count < 2:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == index).astype(np.uint8)
    area = int(region.sum())
    if area < 0.15 * region.size or region[image.shape[0] // 2:].sum() < area * 0.65:
        return None
    return cv2.dilate(region, np.ones((9, 9), np.uint8))


def _coefficients(segment: np.ndarray) -> np.ndarray:
    points = np.column_stack((segment, np.ones(2)))
    line = np.cross(points[0], points[1])
    line /= np.linalg.norm(line[:2])
    if line[0] < 0 or (abs(line[0]) < 1e-9 and line[1] < 0):
        line *= -1
    return line


def _extract_lines(mask: np.ndarray, max_lines: int = 28, *, balanced: bool = True) -> list[np.ndarray]:
    height, width = mask.shape
    minimum = max(20, int(min(height, width) * 0.10))
    detected = cv2.HoughLinesP(
        mask, 1, np.pi / 720, max(18, minimum // 2),
        minLineLength=minimum, maxLineGap=max(8, minimum // 3),
    )
    if detected is None:
        return []
    segments = np.asarray(detected, dtype=np.float64).reshape(-1, 2, 2)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    groups: list[list[np.ndarray]] = []
    tolerance = max(4.0, min(height, width) * 0.0125)
    # Merge Hough fragments and both edges of a painted line. Distance is
    # evaluated on the observed segment, not at a remote vanishing point.
    # Keep a wider pool before merging: long net and wall edges can otherwise
    # consume the entire pool before shorter court sidelines are considered.
    pool_size = max(320, max_lines * 12) if balanced else max(160, max_lines * 4)
    for segment in segments[np.argsort(-lengths)[:pool_size]]:
        coeff = _coefficients(segment)
        for group in groups:
            reference = _coefficients(group[0])
            if abs(float(coeff[:2] @ reference[:2])) < np.cos(np.deg2rad(2.0)):
                continue
            residual = np.abs(segment @ reference[:2] + reference[2]).mean()
            if residual < tolerance:
                group.append(segment)
                break
        else:
            groups.append([segment])
    merged = []
    for group in groups:
        points = np.concatenate(group)
        center = points.mean(axis=0)
        _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
        direction = vectors[0]
        extent = (points - center) @ direction
        merged.append(np.array([center + extent.min() * direction,
                                center + extent.max() * direction]))
    if not balanced:
        merged.sort(key=lambda line: -float(np.linalg.norm(line[1] - line[0])))
        return merged[:max_lines]
    # A global longest-first cut favors one family of parallel lines (often the
    # net and far wall). Preserve several directions and image bands so a pair
    # of sidelines can reach the bounded court-template search.
    remaining = list(merged)
    selected = []
    direction_counts: dict[int, int] = {}
    band_counts: dict[tuple[int, int], int] = {}
    while remaining and len(selected) < max_lines:
        def priority(line: np.ndarray) -> float:
            vector = line[1] - line[0]
            length = float(np.linalg.norm(vector))
            angle = np.arctan2(vector[1], vector[0]) % np.pi
            direction = min(11, int(12 * angle / np.pi))
            band = min(2, int(3 * float(line[:, 1].mean()) / height))
            return length / (1 + .6 * direction_counts.get(direction, 0)
                             + .25 * band_counts.get((direction, band), 0))

        index = max(range(len(remaining)), key=lambda i: priority(remaining[i]))
        line = remaining.pop(index)
        angle = np.arctan2(*(line[1] - line[0])[::-1]) % np.pi
        direction = min(11, int(12 * angle / np.pi))
        band = min(2, int(3 * float(line[:, 1].mean()) / height))
        selected.append(line)
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
        band_counts[(direction, band)] = band_counts.get((direction, band), 0) + 1
    return selected


def _direction_families(lines: list[np.ndarray]) -> list[tuple[int, ...]]:
    """Propose projective parallel families, including vanishing points at infinity.

    Image-space angles alone are insufficient for oblique courts: two sidelines
    can have very different slopes while meeting at the same vanishing point.
    """
    if len(lines) < 4:
        return []
    coefficients = np.asarray([_coefficients(line) for line in lines])
    centers = np.asarray([line.mean(axis=0) for line in lines])
    lengths = np.asarray([np.linalg.norm(line[1] - line[0]) for line in lines])
    families = set()
    for first, second in combinations(range(len(lines)), 2):
        vanishing = np.cross(coefficients[first], coefficients[second])
        rays = vanishing[:2] - centers * vanishing[2]
        norms = np.linalg.norm(rays, axis=1)
        if not np.any(norms > 1e-8):
            continue
        sine = np.abs(coefficients @ vanishing) / np.maximum(norms, 1e-8)
        members = tuple(np.flatnonzero((sine < np.sin(np.deg2rad(2.0))) & (norms > 1e-8)))
        if len(members) >= 2:
            families.add(members)
    # Avoid many near-identical proposals consuming the hypothesis budget.
    ordered = sorted(families, key=lambda group: (-len(group), -float(lengths[list(group)].sum()), group))
    unique = []
    for group in ordered:
        if not any(set(group).issubset(set(other)) for other in unique):
            unique.append(group)
    return unique[:20]


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    point = np.cross(_coefficients(a), _coefficients(b))
    if abs(point[2]) < 1e-8:
        return None
    return point[:2] / point[2]


def _sample_template() -> tuple[np.ndarray, list[slice]]:
    points, slices = [], []
    for start, end in floor_court_line_segments():
        index = len(points)
        points.extend(np.linspace(start, end, 40))
        slices.append(slice(index, len(points)))
    return np.asarray(points), slices


def _score(matrix: np.ndarray, distance: np.ndarray, points: np.ndarray,
           slices: list[slice], tolerance: float) -> tuple[float, list[float]]:
    try:
        projected = _transform_numpy(points, matrix)
    except ValueError:
        return 0.0, []
    height, width = distance.shape
    visible = ((projected[:, 0] >= 0) & (projected[:, 0] < width - 1)
               & (projected[:, 1] >= 0) & (projected[:, 1] < height - 1))
    indexes = np.clip(np.rint(projected), [0, 0], [width - 1, height - 1]).astype(int)
    residuals = distance[indexes[:, 1], indexes[:, 0]]
    support = np.maximum(0, 1 - residuals / tolerance)
    fractions = []
    for section in slices:
        selected = visible[section]
        # One visible fragment must have enough samples to count as a marking.
        if selected.sum() >= 8:
            fractions.append(float(support[section][selected].mean()))
    if len(fractions) < 5:
        return 0.0, fractions
    # Balance distinct markings. Reward explaining more of the available model,
    # without forcing off-screen markings to be present.
    score = float(np.mean(fractions)) * min(1.0, len(fractions) / 8)
    return score, fractions


def _distance(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)


def _plausible(matrix: np.ndarray, width: int, height: int) -> bool:
    corners = np.array([[-3.05, -6.7], [3.05, -6.7], [3.05, 6.7], [-3.05, 6.7]])
    homogeneous = np.column_stack((corners, np.ones(4))) @ matrix.T
    if np.any(homogeneous[:, 2] * homogeneous[0, 2] <= 0):
        return False
    image = homogeneous[:, :2] / homogeneous[:, 2:]
    if not np.isfinite(image).all() or np.max(np.abs(image)) > 4 * max(width, height):
        return False
    edges = np.roll(image, -1, axis=0) - image
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if not (np.all(turns > 0) or np.all(turns < 0)):
        return False
    area = abs(np.dot(image[:, 0], np.roll(image[:, 1], 1))
               - np.dot(image[:, 1], np.roll(image[:, 0], 1))) / 2
    # Very thin slivers can explain many near-parallel paint fragments with an
    # arbitrary template and poorly constrain the orthogonal court direction.
    return 0.10 * width * height < area < 6 * width * height


def _matched_lines(calibration: CourtHomography, sides: list[np.ndarray],
                   crosses: list[np.ndarray], tolerance: float) -> dict:
    matched = {}
    proposals = []
    for name, (axis, value) in COURT_CALIBRATION_LINES.items():
        endpoints = ([[value, -6.7], [value, 6.7]] if axis == "x"
                     else [[-3.05, value], [3.05, value]])
        predicted = calibration.project_to_image(endpoints)
        line = _coefficients(predicted)
        candidates = sides if axis == "x" else crosses
        if not candidates:
            continue
        for index, segment in enumerate(candidates):
            overlap = _finite_overlap(predicted, segment, tolerance)
            if overlap is None:
                continue
            error = np.abs(overlap @ line[:2] + line[2]).mean()
            if error < tolerance:
                proposals.append((float(error), name, axis, index, overlap))
    # A detected paint line cannot independently support both the singles and
    # doubles identities, especially where they nearly meet in the far court.
    assigned = set()
    for _, name, axis, index, overlap in sorted(proposals, key=lambda item: item[:4]):
        if name not in matched and index not in assigned:
            matched[name] = overlap.tolist()
            assigned.add(index)
    return matched


def _finite_overlap(predicted: np.ndarray, observed: np.ndarray,
                    tolerance: float) -> np.ndarray | None:
    """Clip a supported observation to its finite court marking.

    Collinearity with an extension on a neighboring court is insufficient. The
    observed segment must overlap the actual projected marking, have a compatible
    direction, and contribute a meaningful length within those endpoints.
    """
    direction = predicted[1] - predicted[0]
    length = float(np.linalg.norm(direction))
    observed_direction = observed[1] - observed[0]
    observed_length = float(np.linalg.norm(observed_direction))
    if min(length, observed_length) < 1e-6:
        return None
    direction /= length
    cosine = abs(float(direction @ observed_direction / observed_length))
    if cosine < np.cos(np.deg2rad(7)):
        return None
    parameters = (observed - predicted[0]) @ direction
    low, high = max(0.0, float(parameters.min())), min(length, float(parameters.max()))
    if high - low < max(8.0, 0.12 * min(length, observed_length)):
        return None
    if abs(parameters[1] - parameters[0]) < 1e-8:
        return None
    clipped = observed[0] + ((np.array([low, high]) - parameters[0])
                             / (parameters[1] - parameters[0]))[:, None] * observed_direction
    if np.abs(clipped @ _coefficients(predicted)[:2] + _coefficients(predicted)[2]).mean() > tolerance:
        return None
    return clipped


def _finite_distances(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Exact image-pixel distance to finite template segments (no raster alias)."""
    directions = segments[:, 1] - segments[:, 0]
    length_squared = np.maximum(np.sum(directions ** 2, axis=1), 1e-12)
    offsets = points[:, None] - segments[None, :, 0]
    parameters = np.clip(np.einsum("nsi,si->ns", offsets, directions)
                         / length_squared[None], 0, 1)
    nearest = segments[None, :, 0] + parameters[:, :, None] * directions[None]
    return np.linalg.norm(points[:, None] - nearest, axis=2).min(axis=1)


def _consistency(matrix: np.ndarray, lines: list[np.ndarray], distance: np.ndarray,
                 tolerance: float, court_region: np.ndarray | None = None) -> dict:
    """Reverse evidence and center-junction checks for shortlisted hypotheses.

    Only coherent detected paint segments inside the predicted court contribute
    reverse evidence. Random texture and people are not treated as court labels.
    This is the analytic equivalent of a finite rendered-template distance map.
    """
    height, width = distance.shape
    calibration = CourtHomography(np.linalg.inv(matrix))
    template = _transform_numpy(np.asarray(floor_court_line_segments()).reshape(-1, 2), matrix).reshape(-1, 2, 2)
    points = np.concatenate([np.linspace(*line, 24) for line in lines])
    court = calibration.project_to_court(points)
    inside = ((np.abs(court[:, 0]) <= 3.05) & (np.abs(court[:, 1]) <= 6.7)
              & (points[:, 0] >= 0) & (points[:, 0] < width)
              & (points[:, 1] >= 0) & (points[:, 1] < height))
    indexes = np.clip(np.rint(points), [0, 0], [width - 1, height - 1]).astype(int)
    if court_region is not None:
        inside &= court_region[indexes[:, 1], indexes[:, 0]] > 0
    residuals = _finite_distances(points, template)
    explained, weights, unmodeled = [], [], []
    for index, line in enumerate(lines):
        section = slice(24 * index, 24 * (index + 1))
        selected = inside[section]
        if selected.sum() < 6:
            continue
        errors = residuals[section][selected]
        fraction = float(np.mean(np.minimum(1, errors / tolerance)))
        explained.append(1 - fraction)
        visible_length = float(np.linalg.norm(line[1] - line[0]) * selected.mean())
        weights.append(np.sqrt(max(visible_length, 1)))
        if fraction > 0.45:
            unmodeled.append({"segment": line.tolist(), "unexplained_fraction": fraction})
    reverse = float(np.average(explained, weights=weights)) if explained else 0.0
    samples, _ = _sample_template()
    projected = _transform_numpy(samples, matrix)
    visible = ((projected[:, 0] >= 0) & (projected[:, 0] < width)
               & (projected[:, 1] >= 0) & (projected[:, 1] < height))
    region_coverage = 1.0
    if court_region is not None and visible.any():
        indexes = np.clip(np.rint(projected[visible]), [0, 0], [width - 1, height - 1]).astype(int)
        region_coverage = float(np.mean(court_region[indexes[:, 1], indexes[:, 0]] > 0))
    junctions = []
    probe_diagnostics = []

    def support(samples, pixel_tolerance=None):
        valid = ((samples[:, 0] >= 0) & (samples[:, 0] < width)
                 & (samples[:, 1] >= 0) & (samples[:, 1] < height))
        indexes = np.clip(np.rint(samples), [0, 0], [width - 1, height - 1]).astype(int)
        if court_region is not None:
            valid &= court_region[indexes[:, 1], indexes[:, 0]] > 0
        if valid.sum() < len(samples) * 0.8:
            return None
        threshold = tolerance if pixel_tolerance is None else pixel_tolerance
        return float(np.maximum(0, 1 - distance[indexes[valid, 1], indexes[valid, 0]] / threshold).mean())

    def metric_junction(x, y, inward_sign, name, both_crossbar_sides=True):
        # Sample the same physical 15--70 cm neighborhood at every perspective.
        # Fixed image-pixel probes can cross several markings at the far end.
        origin, along_x, along_y = _transform_numpy(np.array([
            [x, y], [x + .05, y], [x, y + .05]]), matrix)
        dx, dy = (along_x - origin) / .05, (along_y - origin) / .05
        determinant = abs(float(np.linalg.det(np.array([dx, dy]))))
        x_perpendicular = determinant / max(np.linalg.norm(dy), 1e-8)
        y_perpendicular = determinant / max(np.linalg.norm(dx), 1e-8)
        y_tolerance = min(tolerance, max(1.25, .02 * x_perpendicular + .75))
        x_tolerance = min(tolerance, max(1.25, .02 * y_perpendicular + .75))
        offsets = np.linspace(.15, .70, 12)
        incoming = _transform_numpy(np.column_stack((np.full(12, x), y + inward_sign * offsets)), matrix)
        outgoing = _transform_numpy(np.column_stack((np.full(12, x), y - inward_sign * offsets)), matrix)
        if both_crossbar_sides:
            across = _transform_numpy(np.column_stack((x + np.r_[offsets, -offsets], np.full(24, y))), matrix)
        else:
            across = _transform_numpy(np.column_stack((x - np.sign(x) * offsets, np.full(12, y))), matrix)
        branch_length = float(min(np.linalg.norm(incoming[-1] - origin), np.linalg.norm(outgoing[-1] - origin)))
        crossbar_normal = np.array([-dx[1], dx[0]]) / max(np.linalg.norm(dx), 1e-8)
        perpendicular_span = float(min(abs((incoming[-1] - origin) @ crossbar_normal),
                                       abs((outgoing[-1] - origin) @ crossbar_normal)))
        probe = {"name": name, "branch_length_px": branch_length,
                 "perpendicular_span_px": perpendicular_span,
                 "line_tolerance_px": y_tolerance, "sample_distance_metres": [.15, .70]}
        if perpendicular_span < max(4., 2 * x_tolerance + 1):
            probe["status"] = "unresolved_perspective"
            probe_diagnostics.append(probe)
            return None
        incoming_support = support(incoming, y_tolerance)
        outgoing_support = support(outgoing, y_tolerance)
        crossbar_support = support(across, x_tolerance)
        if incoming_support is None or outgoing_support is None or crossbar_support is None:
            probe["status"] = "outside_observed_region"
            probe_diagnostics.append(probe)
            return None
        probe["status"] = "observed"
        probe_diagnostics.append(probe)
        return {"name": name, "incoming_support": incoming_support,
                "crossbar_support": crossbar_support, "unexpected_continuation": outgoing_support}

    for y in (-1.98, 1.98):
        name = "near_short_service_center" if y < 0 else "far_short_service_center"
        junction = metric_junction(0., y, np.sign(y), name)
        if junction is not None:
            junctions.append(junction)
    baseline_junctions = []
    boundary_junctions = []
    for y in (-6.7, 6.7):
        side = "near" if y < 0 else "far"
        junction = metric_junction(0., y, -np.sign(y), f"{side}_baseline_center")
        if junction is not None:
            baseline_junctions.append(junction)
        for x in (-3.05, -2.59, 2.59, 3.05):
            junction = metric_junction(x, y, -np.sign(y), f"{side}_baseline_x_{x}", False)
            if junction is not None:
                boundary_junctions.append({**junction, "baseline": side, "court_x": x})
    topology = 0.0
    if junctions:
        # A visible center-service line should meet a short-service crossbar and
        # stop toward the net. Missing incoming paint lowers confidence without
        # being mistaken for a contradictory continuation.
        qualities = [.5 + .5 * min(j["incoming_support"], j["crossbar_support"])
                     - .25 * min(j["incoming_support"], j["unexpected_continuation"])
                     for j in junctions]
        topology = float(np.mean(qualities))
    short_service_support = {}
    for name, y in (("near", -1.98), ("far", 1.98)):
        samples = _transform_numpy(np.column_stack((np.linspace(-3.05, 3.05, 40), np.full(40, y))), matrix)
        short_service_support[name] = support(samples)
    baseline_support = {}
    for name, y in (("near", -6.7), ("far", 6.7)):
        samples = _transform_numpy(np.column_stack((np.linspace(-3.05, 3.05, 40), np.full(40, y))), matrix)
        baseline_support[name] = support(samples)
    return {"observed_paint_explained": reverse, "unexplained_segments": unmodeled,
            "independent_segments_checked": len(explained),
            "visible_template_inside_region": region_coverage,
            "center_service_junctions": junctions, "baseline_junctions": baseline_junctions,
            "boundary_junctions": boundary_junctions,
            "junction_probe_diagnostics": probe_diagnostics,
            "expected_short_service_support": short_service_support,
            "expected_baseline_support": baseline_support,
            "topology_score": topology,
            "center_service_evidence": "sampled_not_verified" if junctions else "unobserved"}


def _court_line_endpoints(name: str) -> np.ndarray:
    axis, value = COURT_CALIBRATION_LINES[name]
    return np.asarray([[value, -6.7], [value, 6.7]] if axis == "x"
                      else [[-3.05, value], [3.05, value]])


def _refine_lines(matrix: np.ndarray, matched: dict, width: int, height: int,
                  tolerance: float) -> np.ndarray:
    """Bounded robust least squares on independent observed line segments.

    Each distinct marking contributes the same six samples regardless of its
    length or the number of inferred intersections. Normalize parameters for
    finite-difference conditioning; residuals stay in image pixels.
    """
    names = list(matched)
    observed = np.asarray([np.linspace(*matched[name], 6) for name in names])
    court = np.asarray([_court_line_endpoints(name) for name in names])
    image_scale = np.diag([width, height, 1.0])
    court_scale = np.diag([3.05, 6.7, 1.0])
    normalized = np.linalg.inv(image_scale) @ matrix @ court_scale
    normalized /= normalized[2, 2]
    parameters = normalized.reshape(-1)[:8].copy()

    def unpack(values):
        return image_scale @ np.append(values, 1.0).reshape(3, 3) @ np.linalg.inv(court_scale)

    def residual(values):
        projected = _transform_numpy(court.reshape(-1, 2), unpack(values)).reshape(-1, 2, 2)
        coefficients = np.cross(np.concatenate((projected[:, 0], np.ones((len(names), 1))), axis=1),
                                np.concatenate((projected[:, 1], np.ones((len(names), 1))), axis=1))
        lengths = np.linalg.norm(coefficients[:, :2], axis=1)
        if np.any(lengths < 1e-8):
            raise ValueError("degenerate projected line")
        coefficients /= lengths[:, None]
        return (np.einsum("nsi,ni->ns", observed, coefficients[:, :2])
                + coefficients[:, 2, None]).reshape(-1)

    for _ in range(8):
        errors = residual(parameters)
        weights = 1.0 / np.maximum(1.0, np.abs(errors) / tolerance)
        jacobian = np.empty((len(errors), 8))
        for index in range(8):
            changed = parameters.copy()
            changed[index] += 1e-5
            jacobian[:, index] = (residual(changed) - errors) / 1e-5
        normal = jacobian.T @ (weights[:, None] * jacobian)
        step = np.linalg.solve(normal + 1e-5 * np.diag(np.maximum(np.diag(normal), 1.0)),
                               -(jacobian.T @ (weights * errors)))
        # Backtracking keeps refinement stable when a candidate is only partial.
        objective = float(np.sum(weights * errors ** 2))
        improved = False
        for factor in (1.0, 0.5, 0.25, 0.125):
            proposal = parameters + factor * step
            trial = residual(proposal)
            if np.sum(weights * trial ** 2) < objective:
                parameters, improved = proposal, True
                break
        if not improved or np.linalg.norm(step) < 1e-7:
            break
    return unpack(parameters)


def _line_diagnostics(calibration: CourtHomography, matched: dict, distance: np.ndarray,
                      tolerance: float) -> tuple[dict, dict]:
    height, width = distance.shape
    details = {}
    for name, segment in matched.items():
        samples = np.linspace(*segment, 24)
        indexes = np.clip(np.rint(samples), [0, 0], [width - 1, height - 1]).astype(int)
        line = _coefficients(calibration.project_to_image(_court_line_endpoints(name)))
        errors = np.abs(samples @ line[:2] + line[2])
        visible_support = float(np.mean(distance[indexes[:, 1], indexes[:, 0]] < tolerance))
        details[name] = {
            "observed_segment": segment,
            "residual_mean_px": float(errors.mean()),
            "residual_max_px": float(errors.max()),
            "visible_support_fraction": visible_support,
            "confidence": float(visible_support * np.exp(-errors.mean() / tolerance)),
            "observation": "detected_line_segment",
        }
    observed = np.asarray(list(matched.values())).reshape(-1, 2)
    low, high = observed.min(axis=0), observed.max(axis=0)
    court = calibration.project_to_court(observed)
    court_low = np.maximum(court.min(axis=0), [-3.05, -6.7])
    court_high = np.minimum(court.max(axis=0), [3.05, 6.7])
    spans = np.maximum(0, court_high - court_low) / [6.1, 13.4]
    coverage = {
        "image_bbox_fraction": float(np.prod(np.minimum(high - low, [width, height]) / [width, height])),
        "court_width_fraction": float(spans[0]),
        "court_length_fraction": float(spans[1]),
        "description": "bounding spans of detected segment endpoints, not inferred corners",
    }
    return details, coverage


def detect_court(image: np.ndarray, validation_frames: list[np.ndarray] | None = None,
                 *, max_hypotheses: int = 8192, seed: int = 0,
                 court_region: np.ndarray | None = None) -> CourtDetection:
    """Select a geometrically validated main court; all proposals require review."""
    from src.court_selection import select_court

    return select_court(image, validation_frames, max_hypotheses=max_hypotheses,
                        seed=seed, court_region=court_region)
