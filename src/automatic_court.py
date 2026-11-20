"""Automatic calibration orchestration; unbenchmarked candidates require review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.court_background import estimate_background
from src.court_detection import detect_court
from src.court_projection import COURT_CALIBRATION_LINES, CourtHomography, draw_court_overlay
from src.workflow.artifacts import atomic_write_json


class CalibrationNeedsReview(RuntimeError):
    """No authoritative calibration was published."""


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write calibration diagnostic: {path}")


def _source_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def court_region_mask(value: dict, image_size: tuple[int, int],
                      working_size: tuple[int, int]) -> np.ndarray:
    """Rasterize an explicitly selected court in original-image coordinates."""
    if not isinstance(value, dict) or value.get("image_size") != list(image_size):
        raise ValueError("court region image_size must match the source")
    points = np.asarray(value.get("polygon", []), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("court region polygon requires at least three finite pixel pairs")
    if np.any(points < 0) or np.any(points > image_size):
        raise ValueError("court region polygon lies outside the source image")
    if not cv2.isContourConvex(points.astype(np.float32)):
        raise ValueError("court region must be a nondegenerate convex polygon in boundary order")
    scaled = np.rint(points * (np.asarray(working_size) / image_size)).astype(np.int32)
    mask = np.zeros((working_size[1], working_size[0]), np.uint8)
    cv2.fillConvexPoly(mask, scaled, 255)
    if np.count_nonzero(mask) < mask.size * .01:
        raise ValueError("court region must cover at least one percent of the image")
    return mask


def _line_assignment_overlay(image: np.ndarray, lines: dict) -> np.ndarray:
    """Expose the selected identities instead of showing an unlabeled grid."""
    display = image.copy()
    height, width = display.shape[:2]
    for name, endpoints in lines.items():
        points = np.rint(endpoints).astype(int)
        color = (255, 220, 40) if "baseline" in name else (40, 220, 255)
        cv2.line(display, tuple(points[0]), tuple(points[1]), color, 3, cv2.LINE_AA)
        position = points.mean(axis=0).astype(int)
        position = np.clip(position, [8, 24], [max(8, width - 400), max(24, height - 12)])
        label = name.replace("_", " ")
        cv2.putText(display, label, tuple(position), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(display, label, tuple(position), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, color, 1, cv2.LINE_AA)
    return display


def _court_selection_overlay(image: np.ndarray, selection: dict,
                             scale: np.ndarray) -> np.ndarray:
    """Display ranked, clipped court footprints in original-image pixels."""
    display = image.copy()
    for candidate in reversed(selection.get("candidates", [])[:8]):
        polygon = np.asarray(candidate.get("visible_polygon", []), dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            continue
        polygon = np.rint(polygon / scale).astype(np.int32)
        rank = candidate.get("rank", "?")
        color = (60, 240, 60) if rank == 1 else (0, 165, 255)
        cv2.polylines(display, [polygon], True, color, 3, cv2.LINE_AA)
        center = polygon.mean(axis=0).astype(int)
        label = f"#{rank} quality {candidate.get('score', 0):.3f}"
        position = tuple(np.maximum(center, [8, 24]))
        cv2.putText(display, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                    .7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(display, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                    .7, color, 1, cv2.LINE_AA)
    return display


def calibrate_automatically(args) -> int:
    """Generate diagnostics, optionally review, and atomically publish a result."""
    from src.calibrate_court import collect_browser_selection

    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(f"calibration already exists: {output}; use --force to replace")
    if args.ui != "browser":
        raise ValueError("automatic calibration review requires the browser UI")
    source_signature = _source_signature(Path(args.source))
    region_path = getattr(args, "court_region", None)
    region = json.loads(Path(region_path).read_text(encoding="utf-8")) if region_path else None
    if region_path and not isinstance(region, dict):
        raise ValueError("court region file must contain an object with image_size and polygon")
    protected = {Path(args.source).resolve(), output.resolve()}
    if region_path:
        protected.add(Path(region_path).resolve())
    if args.preview and Path(args.preview).resolve() in protected:
        raise ValueError("preview must not overlap the source, calibration output, or court region")
    diagnostics_dir = args.diagnostics or output.parent / "automatic"
    diagnostics_dir = Path(diagnostics_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_paths = [diagnostics_dir / name for name in
                        ("candidate.json", "overlay.jpg", "line-assignments.jpg", "court-selection.jpg",
                         "background.jpg", "diagnostics.json")]
    diagnostic_paths.extend(diagnostics_dir.glob("validation-*.jpg"))
    if any(path.resolve() in protected for path in diagnostic_paths):
        raise ValueError("diagnostic files must not overlap the source or calibration output")
    # Diagnostics have fixed names for convenience. Remove only our own files so
    # a failed re-run cannot display an earlier candidate as its current result.
    for path in diagnostic_paths:
        path.unlink(missing_ok=True)
    try:
        background = estimate_background(
            Path(args.source), sample_count=args.samples, max_dimension=args.max_dimension
        )
    except ValueError as error:
        atomic_write_json(diagnostics_dir / "diagnostics.json", {
            "schema": "baddievision_automatic_court_diagnostics", "version": 1,
            "status": "failed", "reason": str(error),
        })
        raise CalibrationNeedsReview(f"{error}; see {diagnostics_dir}") from error
    _write_image(diagnostics_dir / "background.jpg", background.image)
    report = {
        "schema": "baddievision_automatic_court_diagnostics",
        "version": 1,
        "method": "classical_lines_temporal_median_v4",
        "status": "failed",
        "image_size": list(background.image_size),
        "working_scale": list(background.scale),
        "detection_coordinate_system": "working_image_pixels",
        "frame_indices": list(background.frame_indices),
        "frame_indexing": "source_local_zero_based",
        "parameters": {"samples": args.samples, "max_dimension": args.max_dimension,
                       "max_hypotheses": getattr(args, "max_hypotheses", 65536)},
        "background": background.diagnostics,
        "court_region": region,
        "review_status": "unreviewed",
    }
    # Source bytes identify diagnostic caches independently of local path aliases.
    digest = hashlib.sha256()
    with Path(args.source).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    report["source_sha256"] = digest.hexdigest()
    if _source_signature(Path(args.source)) != source_signature:
        report["reason"] = "source changed during background estimation"
        atomic_write_json(diagnostics_dir / "diagnostics.json", report)
        raise CalibrationNeedsReview(report["reason"])
    if background.diagnostics.get("motion_detected"):
        report["reason"] = "camera movement/cut detected; split into static source segments"
        atomic_write_json(diagnostics_dir / "diagnostics.json", report)
        raise CalibrationNeedsReview(f"{report['reason']}; see {diagnostics_dir}")

    mask = None
    if region is not None:
        mask = court_region_mask(region, background.image_size,
                                 (background.image.shape[1], background.image.shape[0]))
    detection = detect_court(background.image, validation_frames=background.frames,
                             max_hypotheses=getattr(args, "max_hypotheses", 65536),
                             court_region=mask)
    report["detection"] = detection.diagnostics
    report["status"] = detection.status
    # Detection coordinates are in the resized working image. Persist and review
    # only original-image coordinates so downstream readers need no special case.
    scale = np.asarray(background.scale, dtype=float)
    lines = {
        name: (np.asarray(endpoints, dtype=float) / scale).tolist()
        for name, endpoints in detection.image_lines.items()
    }
    width, height = background.image_size
    full_background = cv2.resize(background.image, (width, height))
    if detection.diagnostics.get("selection", {}).get("candidates"):
        _write_image(diagnostics_dir / "court-selection.jpg", _court_selection_overlay(
            full_background, detection.diagnostics["selection"], scale))
    candidate = None
    if detection.calibration is not None:
        candidate = CourtHomography(detection.calibration.image_to_court @ np.diag([*scale, 1.0]))
        value = candidate.to_dict(image_size=background.image_size, image_lines=lines)
        value["automatic_calibration"] = report.copy()
        atomic_write_json(diagnostics_dir / "candidate.json", value)
        _write_image(diagnostics_dir / "overlay.jpg", draw_court_overlay(full_background, candidate))
        _write_image(diagnostics_dir / "line-assignments.jpg", _line_assignment_overlay(full_background, lines))
        for index, frame in enumerate(background.frames):
            _write_image(
                diagnostics_dir / f"validation-{index}.jpg",
                draw_court_overlay(cv2.resize(frame, (width, height)), candidate),
            )
    else:
        # Do not leave an earlier candidate behind after a failed re-run.
        (diagnostics_dir / "candidate.json").unlink(missing_ok=True)
        (diagnostics_dir / "overlay.jpg").unlink(missing_ok=True)
    atomic_write_json(diagnostics_dir / "diagnostics.json", report)
    print(f"Automatic calibration diagnostics: {diagnostics_dir}")
    if not args.review:
        raise CalibrationNeedsReview(
            f"Automatic calibration is {detection.status}; rerun with --review to correct/confirm. "
            "No calibration published."
        )

    selection, frame, selected_index = collect_browser_selection(
        Path(args.source), getattr(args, "frame", background.frame_indices[0]), mode="lines",
        landmark_names=[], line_names=list(COURT_CALIBRATION_LINES),
        open_browser=args.open_browser, initial_lines=lines,
        background=full_background, review=True,
    )
    calibration, _, landmarks = CourtHomography.from_lines(selection)
    if _source_signature(Path(args.source)) != source_signature:
        report["status"] = "failed"
        report["reason"] = "source changed during calibration review"
        atomic_write_json(diagnostics_dir / "diagnostics.json", report)
        raise CalibrationNeedsReview(report["reason"])
    report["status"] = "accepted"
    report["review_status"] = "manually_reviewed"
    report["review_frame"] = selected_index
    report["corrected"] = selection != lines
    value = calibration.to_dict(landmarks, background.image_size, image_lines=selection)
    value["automatic_calibration"] = report
    # Recheck immediately before publication, including after a long review.
    if output.exists() and not args.force:
        raise FileExistsError(f"calibration already exists: {output}; use --force to replace")
    atomic_write_json(output, value)
    atomic_write_json(diagnostics_dir / "diagnostics.json", report)
    if args.preview:
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        _write_image(Path(args.preview), draw_court_overlay(frame, calibration))
    print(f"Saved reviewed calibration {output}")
    return int(selected_index)
