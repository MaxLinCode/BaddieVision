"""Optional, tolerant court-region signal for heuristic segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tracks import FrameFeature


def add_court_signal(
    frames: list[FrameFeature],
    calibration_path: str | Path,
    tolerance: float,
    source_id: str | None = None,
) -> str | None:
    """Mark image points inside an expanded calibrated court polygon.

    This deliberately uses an image-region polygon, not a floor projection of
    the airborne shuttle.
    """
    try:
        path = Path(calibration_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if "sources" in data:
            sources = {str(key).lower(): value for key, value in data["sources"].items()}
            key = (source_id or "").lower()
            if key not in sources:
                raise ValueError("unresolved calibration source")
            entry = sources[key]
            relative = entry.get("calibration") if isinstance(entry, dict) else entry
            if not relative:
                raise ValueError("source has no calibration")
            path = path.parent / str(relative)
            data = json.loads(path.read_text(encoding="utf-8"))
        size = data["image_size"]
        if len(size) != 2 or min(size) <= 0:
            raise ValueError("invalid image_size")
        if "image_points" in data:
            points = np.asarray(data["image_points"], dtype=float)
        elif "image_landmarks" in data:
            points = np.asarray(list(data["image_landmarks"].values()), dtype=float)
        elif "points" in data:
            values = data["points"]
            points = np.asarray(
                [value["image"] if isinstance(value, dict) else value for value in values],
                dtype=float,
            )
        elif "image_to_court" in data:
            matrix = np.asarray(data["image_to_court"], dtype=float)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError("invalid homography")
            court_corners = np.asarray(
                [
                    [-3.05, -6.70, 1.0],
                    [3.05, -6.70, 1.0],
                    [3.05, 6.70, 1.0],
                    [-3.05, 6.70, 1.0],
                ]
            )
            projected = (np.linalg.inv(matrix) @ court_corners.T).T
            points = projected[:, :2] / projected[:, 2, None]
        else:
            raise ValueError("no image calibration points")
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
            raise ValueError("invalid calibration points")
        # A tolerant bounding region is stable for this intentionally rough signal.
        lo, hi = points.min(axis=0), points.max(axis=0)
        margin = (hi - lo) * tolerance
        lo, hi = lo - margin, hi + margin
        width, height = size
        for item in frames:
            if item.cleaned:
                point = np.asarray([item.raw_x, item.raw_y])
                item.inside_courtish = bool(np.all(point >= lo) and np.all(point <= hi))
        return None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return "court_projection_unstable"
