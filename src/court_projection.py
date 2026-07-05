"""Badminton court geometry and image-to-court homography utilities.

Court coordinates are expressed in metres. The origin is the centre of the
court below the net, x runs across the court, and y runs from the near baseline
to the far baseline.
"""

from __future__ import annotations

import json
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np

COURT_WIDTH = 6.10
COURT_LENGTH = 13.40
SINGLES_WIDTH = 5.18
SHORT_SERVICE_DISTANCE = 1.98
DOUBLES_LONG_SERVICE_INSET = 0.76

HALF_WIDTH = COURT_WIDTH / 2
HALF_LENGTH = COURT_LENGTH / 2
HALF_SINGLES_WIDTH = SINGLES_WIDTH / 2

# Named intersections make calibration files readable and prevent ambiguity
# about point ordering.
COURT_LANDMARKS: dict[str, tuple[float, float]] = {
    "near_left_doubles": (-HALF_WIDTH, -HALF_LENGTH),
    "near_right_doubles": (HALF_WIDTH, -HALF_LENGTH),
    "far_left_doubles": (-HALF_WIDTH, HALF_LENGTH),
    "far_right_doubles": (HALF_WIDTH, HALF_LENGTH),
    "near_left_singles": (-HALF_SINGLES_WIDTH, -HALF_LENGTH),
    "near_right_singles": (HALF_SINGLES_WIDTH, -HALF_LENGTH),
    "far_left_singles": (-HALF_SINGLES_WIDTH, HALF_LENGTH),
    "far_right_singles": (HALF_SINGLES_WIDTH, HALF_LENGTH),
    "net_left": (-HALF_WIDTH, 0.0),
    "net_right": (HALF_WIDTH, 0.0),
    "net_left_singles": (-HALF_SINGLES_WIDTH, 0.0),
    "net_right_singles": (HALF_SINGLES_WIDTH, 0.0),
    "near_short_service_left": (-HALF_WIDTH, -SHORT_SERVICE_DISTANCE),
    "near_short_service_right": (HALF_WIDTH, -SHORT_SERVICE_DISTANCE),
    "near_short_service_left_singles": (
        -HALF_SINGLES_WIDTH,
        -SHORT_SERVICE_DISTANCE,
    ),
    "near_short_service_right_singles": (
        HALF_SINGLES_WIDTH,
        -SHORT_SERVICE_DISTANCE,
    ),
    "far_short_service_left": (-HALF_WIDTH, SHORT_SERVICE_DISTANCE),
    "far_short_service_right": (HALF_WIDTH, SHORT_SERVICE_DISTANCE),
    "far_short_service_left_singles": (
        -HALF_SINGLES_WIDTH,
        SHORT_SERVICE_DISTANCE,
    ),
    "far_short_service_right_singles": (
        HALF_SINGLES_WIDTH,
        SHORT_SERVICE_DISTANCE,
    ),
    "near_doubles_long_service_left": (
        -HALF_WIDTH,
        -HALF_LENGTH + DOUBLES_LONG_SERVICE_INSET,
    ),
    "near_doubles_long_service_right": (
        HALF_WIDTH,
        -HALF_LENGTH + DOUBLES_LONG_SERVICE_INSET,
    ),
    "near_doubles_long_service_left_singles": (
        -HALF_SINGLES_WIDTH,
        -HALF_LENGTH + DOUBLES_LONG_SERVICE_INSET,
    ),
    "near_doubles_long_service_center": (
        0.0,
        -HALF_LENGTH + DOUBLES_LONG_SERVICE_INSET,
    ),
    "near_doubles_long_service_right_singles": (
        HALF_SINGLES_WIDTH,
        -HALF_LENGTH + DOUBLES_LONG_SERVICE_INSET,
    ),
    "far_doubles_long_service_left": (
        -HALF_WIDTH,
        HALF_LENGTH - DOUBLES_LONG_SERVICE_INSET,
    ),
    "far_doubles_long_service_right": (
        HALF_WIDTH,
        HALF_LENGTH - DOUBLES_LONG_SERVICE_INSET,
    ),
    "far_doubles_long_service_left_singles": (
        -HALF_SINGLES_WIDTH,
        HALF_LENGTH - DOUBLES_LONG_SERVICE_INSET,
    ),
    "far_doubles_long_service_center": (
        0.0,
        HALF_LENGTH - DOUBLES_LONG_SERVICE_INSET,
    ),
    "far_doubles_long_service_right_singles": (
        HALF_SINGLES_WIDTH,
        HALF_LENGTH - DOUBLES_LONG_SERVICE_INSET,
    ),
    "near_short_service_center": (0.0, -SHORT_SERVICE_DISTANCE),
    "far_short_service_center": (0.0, SHORT_SERVICE_DISTANCE),
}

# Lines suitable for floor-plane calibration. The net cord is intentionally
# excluded because it is elevated above the plane described by the homography.
COURT_CALIBRATION_LINES: dict[str, tuple[str, float]] = {
    "left_doubles_sideline": ("x", -HALF_WIDTH),
    "left_singles_sideline": ("x", -HALF_SINGLES_WIDTH),
    "right_singles_sideline": ("x", HALF_SINGLES_WIDTH),
    "right_doubles_sideline": ("x", HALF_WIDTH),
    "near_baseline": ("y", -HALF_LENGTH),
    "near_doubles_long_service": (
        "y",
        -HALF_LENGTH + DOUBLES_LONG_SERVICE_INSET,
    ),
    "near_short_service": ("y", -SHORT_SERVICE_DISTANCE),
    "far_short_service": ("y", SHORT_SERVICE_DISTANCE),
    "far_doubles_long_service": (
        "y",
        HALF_LENGTH - DOUBLES_LONG_SERVICE_INSET,
    ),
    "far_baseline": ("y", HALF_LENGTH),
}


def _points_array(points: Iterable[Iterable[float]], name: str) -> np.ndarray:
    array = np.asarray(list(points), dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return array


def _validate_correspondences(image_points: np.ndarray, court_points: np.ndarray) -> None:
    if len(image_points) != len(court_points):
        raise ValueError("image_points and court_points must have the same length")
    if len(image_points) < 4:
        raise ValueError("at least four point correspondences are required")
    if len(np.unique(image_points, axis=0)) < 4:
        raise ValueError("image_points must contain at least four unique points")
    if len(np.unique(court_points, axis=0)) < 4:
        raise ValueError("court_points must contain at least four unique points")
    if np.linalg.matrix_rank(image_points - image_points.mean(axis=0)) < 2:
        raise ValueError("image_points are collinear")
    if np.linalg.matrix_rank(court_points - court_points.mean(axis=0)) < 2:
        raise ValueError("court_points are collinear")


def _fit_homography_dlt(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Fit source-to-destination homography with normalized NumPy DLT."""

    def normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = points.mean(axis=0)
        shifted = points - center
        mean_distance = np.linalg.norm(shifted, axis=1).mean()
        if mean_distance < 1e-12:
            raise ValueError("cannot normalize coincident calibration points")
        scale = np.sqrt(2.0) / mean_distance
        transform = np.array(
            [
                [scale, 0.0, -scale * center[0]],
                [0.0, scale, -scale * center[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        homogeneous = np.column_stack([points, np.ones(len(points))])
        normalized = (transform @ homogeneous.T).T[:, :2]
        return normalized, transform

    source_norm, source_transform = normalize(source)
    destination_norm, destination_transform = normalize(destination)
    rows = []
    for (x, y), (u, v) in zip(source_norm, destination_norm):
        rows.extend(
            [
                [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u],
                [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v],
            ]
        )
    _, _, vh = np.linalg.svd(np.asarray(rows))
    normalized_homography = vh[-1].reshape(3, 3)
    homography = (
        np.linalg.inv(destination_transform)
        @ normalized_homography
        @ source_transform
    )
    scale = homography[2, 2]
    if abs(scale) < 1e-12:
        scale = np.linalg.norm(homography)
    return homography / scale


def _transform_numpy(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = (matrix @ homogeneous.T).T
    denominator = projected[:, 2]
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("point projects to infinity")
    return projected[:, :2] / denominator[:, None]


def _fit_court_to_image(
    court: np.ndarray, image: np.ndarray, ransac_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fit robustly without depending on OpenCV's NumPy ABI bindings."""

    if len(image) == 4:
        return _fit_homography_dlt(court, image), np.ones(4, dtype=bool)

    # Calibration normally uses fewer than a dozen points, so evaluating all
    # four-point samples is deterministic and inexpensive.
    samples = combinations(range(len(image)), 4)
    best_matrix = None
    best_inliers = None
    best_score = (-1, float("-inf"))
    for sample in samples:
        indices = np.asarray(sample)
        sample_court = court[indices]
        sample_image = image[indices]
        if (
            np.linalg.matrix_rank(sample_court - sample_court.mean(axis=0)) < 2
            or np.linalg.matrix_rank(sample_image - sample_image.mean(axis=0)) < 2
        ):
            continue
        try:
            matrix = _fit_homography_dlt(sample_court, sample_image)
            errors = np.linalg.norm(_transform_numpy(court, matrix) - image, axis=1)
        except (ValueError, np.linalg.LinAlgError):
            continue
        inliers = errors <= ransac_threshold
        count = int(inliers.sum())
        mean_error = errors[inliers].mean() if count else float("inf")
        score = (count, -mean_error)
        if score > best_score:
            best_matrix, best_inliers, best_score = matrix, inliers, score

    if best_matrix is None or best_inliers is None or best_inliers.sum() < 4:
        raise ValueError("homography fit has fewer than four inliers")
    return _fit_homography_dlt(court[best_inliers], image[best_inliers]), best_inliers


def _line_intersections(
    image_lines: Mapping[str, Iterable[Iterable[float]]],
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    unknown = set(image_lines) - set(COURT_CALIBRATION_LINES)
    if unknown:
        raise ValueError(f"unknown court calibration lines: {sorted(unknown)}")

    coefficients = {}
    for name, endpoints in image_lines.items():
        points = _points_array(endpoints, f"image_lines[{name!r}]")
        if len(points) != 2 or np.linalg.norm(points[1] - points[0]) < 1e-6:
            raise ValueError(f"{name} must contain two distinct image points")
        homogeneous = np.column_stack([points, np.ones(2)])
        line = np.cross(homogeneous[0], homogeneous[1])
        coefficients[name] = line / np.linalg.norm(line[:2])

    x_lines = [
        name for name in image_lines if COURT_CALIBRATION_LINES[name][0] == "x"
    ]
    y_lines = [
        name for name in image_lines if COURT_CALIBRATION_LINES[name][0] == "y"
    ]
    if len(x_lines) < 2 or len(y_lines) < 2:
        raise ValueError(
            "line calibration requires at least two sidelines and two cross-court lines"
        )

    image_intersections = {}
    court_intersections = {}
    for x_name in x_lines:
        for y_name in y_lines:
            intersection = np.cross(coefficients[x_name], coefficients[y_name])
            if abs(intersection[2]) < 1e-9:
                raise ValueError(f"{x_name} and {y_name} are parallel in the image")
            point = intersection[:2] / intersection[2]
            key = f"{x_name}__{y_name}"
            image_intersections[key] = (float(point[0]), float(point[1]))
            court_intersections[key] = (
                COURT_CALIBRATION_LINES[x_name][1],
                COURT_CALIBRATION_LINES[y_name][1],
            )
    return image_intersections, court_intersections


@dataclass(frozen=True)
class CourtHomography:
    """A calibrated projective transform between pixels and court metres."""

    image_to_court: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.image_to_court, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("image_to_court must be a finite 3x3 matrix")
        if abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("image_to_court matrix is singular")
        scale = matrix[2, 2]
        if abs(scale) < 1e-12:
            scale = np.linalg.norm(matrix)
        object.__setattr__(self, "image_to_court", matrix / scale)

    @classmethod
    def from_points(
        cls,
        image_points: Iterable[Iterable[float]],
        court_points: Iterable[Iterable[float]],
        *,
        ransac_threshold: float = 3.0,
    ) -> tuple["CourtHomography", np.ndarray]:
        """Fit a homography and return it with an inlier mask.

        RANSAC is used when more than four correspondences are supplied.
        ``ransac_threshold`` is measured in image pixels by fitting the inverse
        (court-to-image) mapping, then inverting it.
        """

        image = _points_array(image_points, "image_points")
        court = _points_array(court_points, "court_points")
        _validate_correspondences(image, court)

        court_to_image, inliers = _fit_court_to_image(
            court, image, ransac_threshold
        )
        image_to_court = np.linalg.inv(court_to_image)
        return cls(image_to_court), inliers

    @classmethod
    def from_landmarks(
        cls,
        image_landmarks: Mapping[str, Iterable[float]],
        *,
        ransac_threshold: float = 3.0,
    ) -> tuple["CourtHomography", np.ndarray]:
        unknown = set(image_landmarks) - set(COURT_LANDMARKS)
        if unknown:
            raise ValueError(f"unknown court landmarks: {sorted(unknown)}")
        names = list(image_landmarks)
        return cls.from_points(
            [image_landmarks[name] for name in names],
            [COURT_LANDMARKS[name] for name in names],
            ransac_threshold=ransac_threshold,
        )

    @classmethod
    def from_lines(
        cls,
        image_lines: Mapping[str, Iterable[Iterable[float]]],
        *,
        ransac_threshold: float = 3.0,
    ) -> tuple["CourtHomography", np.ndarray, dict[str, tuple[float, float]]]:
        """Fit from draggable court lines, even when intersections are off-screen."""

        image_intersections, court_intersections = _line_intersections(image_lines)
        calibration, inliers = cls.from_points(
            image_intersections.values(),
            court_intersections.values(),
            ransac_threshold=ransac_threshold,
        )
        return calibration, inliers, image_intersections

    @property
    def court_to_image(self) -> np.ndarray:
        inverse = np.linalg.inv(self.image_to_court)
        scale = inverse[2, 2]
        return inverse / (scale if abs(scale) >= 1e-12 else np.linalg.norm(inverse))

    def project_to_court(
        self, image_points: Iterable[Iterable[float]]
    ) -> np.ndarray:
        return self._transform(image_points, self.image_to_court)

    def project_to_image(
        self, court_points: Iterable[Iterable[float]]
    ) -> np.ndarray:
        return self._transform(court_points, self.court_to_image)

    def project_normalized_to_court(
        self,
        normalized_points: Iterable[Iterable[float]],
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """Project 0..1 image coordinates, such as MediaPipe landmarks."""

        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must contain positive width and height")
        points = _points_array(normalized_points, "normalized_points")
        return self.project_to_court(points * np.array([width, height]))

    @staticmethod
    def _transform(
        points: Iterable[Iterable[float]], matrix: np.ndarray
    ) -> np.ndarray:
        array = _points_array(points, "points")
        if len(array) == 0:
            return np.empty((0, 2), dtype=np.float64)
        transformed = _transform_numpy(array, matrix)
        if not np.isfinite(transformed).all():
            raise ValueError("point projects to infinity")
        return transformed

    def reprojection_errors(
        self,
        image_points: Iterable[Iterable[float]],
        court_points: Iterable[Iterable[float]],
    ) -> np.ndarray:
        image = _points_array(image_points, "image_points")
        court = _points_array(court_points, "court_points")
        if len(image) != len(court):
            raise ValueError("image_points and court_points must have the same length")
        return np.linalg.norm(self.project_to_image(court) - image, axis=1)

    def to_dict(
        self,
        image_landmarks: Mapping[str, Iterable[float]] | None = None,
        image_size: tuple[int, int] | None = None,
        image_lines: Mapping[str, Iterable[Iterable[float]]] | None = None,
    ) -> dict:
        data: dict = {
            "version": 1,
            "coordinate_system": {
                "units": "metres",
                "origin": "court_center_below_net",
                "x_axis": "left_to_right",
                "y_axis": "near_to_far",
            },
            "image_to_court": self.image_to_court.tolist(),
        }
        if image_landmarks is not None:
            data["image_landmarks"] = {
                name: [float(value) for value in point]
                for name, point in image_landmarks.items()
            }
        if image_size is not None:
            data["image_size"] = [int(image_size[0]), int(image_size[1])]
        if image_lines is not None:
            data["image_lines"] = {
                name: [
                    [float(value) for value in point]
                    for point in endpoints
                ]
                for name, endpoints in image_lines.items()
            }
        return data

    def save(
        self,
        path: str | Path,
        image_landmarks: Mapping[str, Iterable[float]] | None = None,
        image_size: tuple[int, int] | None = None,
        image_lines: Mapping[str, Iterable[Iterable[float]]] | None = None,
    ) -> None:
        Path(path).write_text(
            json.dumps(
                self.to_dict(image_landmarks, image_size, image_lines), indent=2
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CourtHomography":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != 1:
            raise ValueError(f"unsupported calibration version: {data.get('version')}")
        return cls(np.asarray(data["image_to_court"], dtype=np.float64))


def court_line_segments() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return regulation court markings as metric line segments."""

    x_left, x_right = -HALF_WIDTH, HALF_WIDTH
    y_near, y_far = -HALF_LENGTH, HALF_LENGTH
    horizontal_y = {
        y_near,
        -HALF_LENGTH + DOUBLES_LONG_SERVICE_INSET,
        -SHORT_SERVICE_DISTANCE,
        0.0,
        SHORT_SERVICE_DISTANCE,
        HALF_LENGTH - DOUBLES_LONG_SERVICE_INSET,
        y_far,
    }
    segments = [((x_left, y), (x_right, y)) for y in sorted(horizontal_y)]
    for x in (x_left, -HALF_SINGLES_WIDTH, HALF_SINGLES_WIDTH, x_right):
        segments.append(((x, y_near), (x, y_far)))
    # Centre service lines run from each short-service line to the corresponding
    # baseline. They pass through the doubles long-service line.
    segments.extend(
        [
            ((0.0, y_near), (0.0, -SHORT_SERVICE_DISTANCE)),
            ((0.0, SHORT_SERVICE_DISTANCE), (0.0, y_far)),
        ]
    )
    return segments


def draw_court_overlay(
    image: np.ndarray,
    homography: CourtHomography,
    *,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw projected regulation court lines on a copy of an image."""

    output = image.copy()
    for start, end in court_line_segments():
        projected = homography.project_to_image([start, end])
        p1, p2 = np.rint(projected).astype(int)
        # Some OpenCV/NumPy combinations reject np.int64 inside point tuples.
        point1 = (int(p1[0]), int(p1[1]))
        point2 = (int(p2[0]), int(p2[1]))
        cv2.line(output, point1, point2, color, thickness, cv2.LINE_AA)
    return output
