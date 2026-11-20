"""Offline evaluation of court candidates; no calibration is published.

Manifest paths are relative to the manifest. Sources must have unique IDs and a
``group`` of ``development`` or ``evaluation``. Evaluation is descriptive: this
command neither tunes thresholds nor promotes provisional candidates to accepted
calibrations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.court_background import estimate_background
from src.court_detection import detect_court
from src.court_projection import COURT_LANDMARKS, CourtHomography
from src.workflow.artifacts import atomic_write_json


def _summary(rows: list[dict]) -> dict:
    courts = [row for row in rows if row.get("expected") == "court"]
    negatives = [row for row in rows if row.get("expected") == "no_court"]
    candidates = [row for row in courts if row.get("candidate_available")]
    heldout = [row["heldout_landmarks"] for row in courts if row.get("heldout_landmarks", {}).get("count")]
    return {
        "source_count": len(rows),
        "court_source_count": len(courts),
        "candidate_count": len(candidates),
        "detection_coverage_rate": len(candidates) / len(courts) if courts else None,
        "negative_source_count": len(negatives),
        "false_candidate_negative_count": sum(bool(row.get("candidate_available")) for row in negatives),
        "failed_source_count": sum(row.get("status") in {"error", "failed", "rejected_motion"} for row in rows),
        "automatic_acceptance_count": 0,
        "automatic_acceptance_rate": 0.0 if rows else None,
        "heldout_scored_source_count": len(heldout),
        "heldout_image_error_diagonal_fraction_source_mean": (
            float(np.mean([value["image_error_diagonal_fraction_mean"] for value in heldout]))
            if heldout else None
        ),
        "heldout_court_error_metres_source_mean": (
            float(np.mean([value["court_error_metres_mean"] for value in heldout]))
            if heldout else None
        ),
    }


def _point_metrics(calibration: CourtHomography, landmarks: dict,
                   image_size: tuple[int, int]) -> dict:
    names = list(landmarks)
    invalid = (set(names) - set(COURT_LANDMARKS)) | {name for name in names if name.startswith("net_")}
    if invalid:
        raise ValueError(f"heldout_landmarks must name floor landmarks; invalid names: {sorted(invalid)}")
    if not names:
        return {"count": 0}
    image = np.asarray([landmarks[name] for name in names], dtype=float)
    if image.shape != (len(names), 2) or not np.isfinite(image).all():
        raise ValueError("heldout_landmarks must contain finite pixel coordinate pairs")
    court = np.asarray([COURT_LANDMARKS[name] for name in names], dtype=float)
    image_errors = np.linalg.norm(calibration.project_to_image(court) - image, axis=1)
    court_errors = np.linalg.norm(calibration.project_to_court(image) - court, axis=1)
    diagonal = float(np.hypot(*image_size))
    return {
        "count": len(names), "names": names,
        "image_error_px_mean": float(image_errors.mean()),
        "image_error_px_max": float(image_errors.max()),
        "image_error_diagonal_fraction_mean": float(image_errors.mean() / diagonal),
        "court_error_metres_mean": float(court_errors.mean()),
        "court_error_metres_max": float(court_errors.max()),
        "per_landmark": {
            name: {"image_error_px": float(image_error), "court_error_metres": float(court_error)}
            for name, image_error, court_error in zip(names, image_errors, court_errors)
        },
    }


def _reference_metrics(calibration: CourtHomography, reference: CourtHomography,
                       image_size: tuple[int, int]) -> dict:
    grid = np.asarray([(x, y) for x in np.linspace(-2.8, 2.8, 5)
                       for y in np.linspace(-6.0, 6.0, 9)])
    reference_image = reference.project_to_image(grid)
    error = np.linalg.norm(calibration.project_to_image(grid) - reference_image, axis=1)
    court_error = np.linalg.norm(calibration.project_to_court(reference_image) - grid, axis=1)
    return {
        "label": "reference_agreement_not_ground_truth",
        "sample_count": len(grid),
        "image_error_px_mean": float(error.mean()),
        "image_error_diagonal_fraction_mean": float(error.mean() / np.hypot(*image_size)),
        "court_error_metres_mean": float(court_error.mean()),
        "court_error_metres_max": float(court_error.max()),
    }


def evaluate_manifest(manifest_path: Path, *, samples: int = 31,
                      max_dimension: int = 960, max_hypotheses: int = 65536) -> dict:
    """Evaluate each source independently; preserve failures in the report."""
    manifest_path = Path(manifest_path)
    if not 1 <= samples <= 255 or max_dimension < 16 or max_hypotheses < 1:
        raise ValueError("samples must be 1..255, max_dimension at least 16, and max_hypotheses positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("manifest requires a nonempty sources list")
    ids = [source.get("source_id") for source in sources if isinstance(source, dict)]
    if len(ids) != len(sources) or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("every source requires a nonempty source_id")
    if len(set(ids)) != len(ids):
        raise ValueError("source IDs must be unique across development and evaluation groups")

    def path(value: str) -> Path:
        selected = Path(value)
        return selected if selected.is_absolute() else manifest_path.parent / selected

    video_groups: dict[Path, str] = {}
    for source in sources:
        if not isinstance(source.get("video"), str) or source.get("group") not in {"development", "evaluation"}:
            continue
        video = path(source["video"]).resolve()
        previous = video_groups.setdefault(video, source["group"])
        if previous != source["group"]:
            raise ValueError("the same video path cannot occur in both development and evaluation groups")

    rows = []
    for source in sources:
        started = perf_counter()
        row = {"source_id": source["source_id"], "group": source.get("group"),
               "expected": source.get("expected", "court"), "status": "error",
               "candidate_available": False, "automatic_accepted": False}
        if "notes" in source:
            row["notes"] = source["notes"]
        try:
            if row["group"] not in {"development", "evaluation"}:
                raise ValueError("group must be development or evaluation")
            if row["expected"] not in {"court", "no_court"}:
                raise ValueError("expected must be court or no_court")
            video = path(source["video"])
            row["video"] = str(video)
            background = estimate_background(video, sample_count=samples, max_dimension=max_dimension)
            row["background_seconds"] = perf_counter() - started
            row["image_size"] = list(background.image_size)
            row["frame_indices"] = background.frame_indices
            row["background"] = background.diagnostics
            if background.diagnostics.get("motion_detected"):
                row["status"] = "rejected_motion"
                row["error"] = "camera movement/cut detected; split into static sources"
                continue
            detection_started = perf_counter()
            result = detect_court(background.image, validation_frames=background.frames,
                                  max_hypotheses=max_hypotheses)
            row["detection_seconds"] = perf_counter() - detection_started
            row["status"] = result.status
            row["detection"] = result.diagnostics
            row["candidate_available"] = result.calibration is not None
            if result.calibration is not None:
                calibration = CourtHomography(
                    result.calibration.image_to_court @ np.diag([*background.scale, 1.0])
                )
                if "heldout_landmarks" in source:
                    row["heldout_landmarks"] = _point_metrics(
                        calibration, source["heldout_landmarks"], background.image_size
                    )
                if source.get("reference_calibration"):
                    reference_path = path(source["reference_calibration"])
                    value = json.loads(reference_path.read_text(encoding="utf-8"))
                    if value.get("image_size") != list(background.image_size):
                        raise ValueError("reference calibration image_size differs from source")
                    row["reference_agreement"] = _reference_metrics(
                        calibration, CourtHomography.load(reference_path), background.image_size
                    )
        except Exception as error:
            row["status"] = "error"
            row["error"] = f"{type(error).__name__}: {error}"
        finally:
            row["runtime_seconds"] = perf_counter() - started
            rows.append(row)
    return {
        "schema": "baddievision_court_calibration_evaluation", "version": 1,
        "manifest": str(manifest_path), "parameters": {"samples": samples, "max_dimension": max_dimension,
                                                       "max_hypotheses": max_hypotheses},
        "thresholds": "provisional_not_benchmark_validated",
        "acceptance_policy": "all_candidates_require_manual_review",
        "split_note": "Source IDs and resolved video paths are disjoint across groups; shared-camera and duplicate-file provenance must be verified by the manifest author.",
        "summary": _summary(rows),
        "groups": {group: _summary([row for row in rows if row["group"] == group])
                   for group in ("development", "evaluation")},
        "sources": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--max-dimension", type=int, default=960)
    parser.add_argument("--max-hypotheses", type=int, default=65536)
    args = parser.parse_args(argv)
    report = evaluate_manifest(args.manifest, samples=args.samples, max_dimension=args.max_dimension,
                               max_hypotheses=args.max_hypotheses)
    atomic_write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
