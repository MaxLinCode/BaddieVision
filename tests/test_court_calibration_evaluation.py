import json
from types import SimpleNamespace

import numpy as np
import pytest

from src import evaluate_court_calibration as evaluation
from src.court_projection import COURT_LANDMARKS, CourtHomography


def write_manifest(tmp_path, sources):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"sources": sources}))
    return path


def mock_background(**updates):
    values = dict(image=np.zeros((50, 100, 3), np.uint8), image_size=(200, 100),
                  frame_indices=[0, 5, 10], frames=[], scale=(0.5, 0.5), diagnostics={})
    values.update(updates)
    return SimpleNamespace(**values)


def mock_detector(calibration):
    return SimpleNamespace(calibration=calibration, status="needs_review" if calibration else "failed",
                           diagnostics={"matched_line_count": 6, "score": .8})


def test_metrics_original_resolution_and_split_summary(tmp_path, monkeypatch):
    original = CourtHomography(np.array([[.1, 0, -10], [0, .1, -5], [0, 0, 1.]]))
    working = CourtHomography(original.image_to_court @ np.diag([2., 2., 1.]))
    landmark = "near_short_service_center"
    pixels = original.project_to_image([COURT_LANDMARKS[landmark]])[0]
    manifest = write_manifest(tmp_path, [
        {"source_id": "dev", "video": "dev.mp4", "group": "development",
         "heldout_landmarks": {landmark: pixels.tolist()}, "reference_calibration": "reference.json"},
        {"source_id": "negative", "video": "negative.mp4", "expected": "no_court", "group": "evaluation"},
    ])
    original.save(tmp_path / "reference.json", image_size=(200, 100))
    paths = []
    def background(path, **kwargs):
        paths.append(path)
        return mock_background()
    monkeypatch.setattr(evaluation, "estimate_background", background)
    monkeypatch.setattr(evaluation, "detect_court", lambda *args, **kwargs: mock_detector(working))
    report = evaluation.evaluate_manifest(manifest)
    scored = report["sources"][0]
    assert paths[0] == tmp_path / "dev.mp4"
    assert scored["heldout_landmarks"]["court_error_metres_mean"] == pytest.approx(0, abs=1e-10)
    assert scored["heldout_landmarks"]["image_error_diagonal_fraction_mean"] == pytest.approx(0, abs=1e-10)
    assert scored["reference_agreement"]["label"] == "reference_agreement_not_ground_truth"
    assert scored["reference_agreement"]["court_error_metres_mean"] == pytest.approx(0, abs=1e-10)
    assert report["summary"]["detection_coverage_rate"] == 1
    assert report["summary"]["false_candidate_negative_count"] == 1
    assert report["summary"]["automatic_acceptance_count"] == 0
    assert report["groups"]["development"]["source_count"] == 1
    assert report["groups"]["evaluation"]["source_count"] == 1


def test_decode_failure_does_not_skip_later_sources(tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path, [
        {"source_id": "bad", "video": "missing.mp4", "group": "evaluation"},
        {"source_id": "good", "video": "good.mp4", "group": "evaluation"},
    ])
    def background(path, **kwargs):
        if path.name == "missing.mp4":
            raise ValueError("decode failed")
        return mock_background()
    monkeypatch.setattr(evaluation, "estimate_background", background)
    monkeypatch.setattr(evaluation, "detect_court", lambda *args, **kwargs: mock_detector(None))
    report = evaluation.evaluate_manifest(manifest)
    assert [row["status"] for row in report["sources"]] == ["error", "failed"]
    assert report["summary"]["failed_source_count"] == 2
    assert report["summary"]["detection_coverage_rate"] == 0


def test_motion_rejection_skips_detector_and_writes_report(tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path, [{"source_id": "moving", "video": "moving.mp4", "group": "evaluation"}])
    monkeypatch.setattr(evaluation, "estimate_background", lambda *args, **kwargs: mock_background(
        diagnostics={"motion_detected": True}))
    def no_detection(*args, **kwargs):
        pytest.fail("detector must not run on a moving source")
    monkeypatch.setattr(evaluation, "detect_court", no_detection)
    output = tmp_path / "reports" / "evaluation.json"
    assert evaluation.main(["--manifest", str(manifest), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["sources"][0]["status"] == "rejected_motion"
    assert report["sources"][0]["runtime_seconds"] >= 0


def test_duplicate_source_ids_rejected_across_groups(tmp_path):
    manifest = write_manifest(tmp_path, [{"source_id": "same", "group": group, "video": "same.mp4"}
                                         for group in ("development", "evaluation")])
    with pytest.raises(ValueError, match="unique"):
        evaluation.evaluate_manifest(manifest)


def test_floor_landmark_validation():
    calibration = CourtHomography(np.eye(3))
    with pytest.raises(ValueError, match="floor landmarks"):
        evaluation._point_metrics(calibration, {"net_left": [0, 0]}, (200, 100))
    assert evaluation._point_metrics(calibration, {}, (200, 100)) == {"count": 0}
    result = evaluation._point_metrics(calibration, {"near_short_service_center": [1, -1.98]}, (200, 100))
    assert result["image_error_px_mean"] == pytest.approx(1)
    assert result["image_error_diagonal_fraction_mean"] == pytest.approx(1 / np.hypot(200, 100))
    assert result["court_error_metres_mean"] == pytest.approx(1)


def test_same_video_alias_cannot_cross_groups(tmp_path):
    manifest = write_manifest(tmp_path, [
        {"source_id": "first", "group": "development", "video": "same.mp4"},
        {"source_id": "second", "group": "evaluation", "video": "nested/../same.mp4"},
    ])
    with pytest.raises(ValueError, match="same video path"):
        evaluation.evaluate_manifest(manifest)
