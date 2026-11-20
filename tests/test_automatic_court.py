"""Automatic calibration integration and real browser endpoint contracts."""

import hashlib
import json
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from src.automatic_court import CalibrationNeedsReview, calibrate_automatically, court_region_mask
from src.calibrate_court import collect_browser_selection
from src.court_background import BackgroundEstimate
from src.court_detection import CourtDetection
from src.court_projection import COURT_CALIBRATION_LINES, COURT_LANDMARKS, CourtHomography
from src.workflow.artifacts import atomic_write_json


def calibration_fixture():
    names = ["near_left_doubles", "near_right_doubles", "far_right_doubles", "far_left_doubles"]
    calibration, _ = CourtHomography.from_points(
        [[70, 450], [570, 450], [440, 80], [200, 80]],
        [COURT_LANDMARKS[name] for name in names],
    )
    lines = {}
    for name in ("left_doubles_sideline", "right_doubles_sideline",
                 "near_short_service", "far_short_service"):
        axis, value = COURT_CALIBRATION_LINES[name]
        endpoints = [[value, -4], [value, 4]] if axis == "x" else [[-2, value], [2, value]]
        lines[name] = calibration.project_to_image(endpoints).tolist()
    return calibration, lines


class AutomaticCourtTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "video.mp4"
        self.source.write_bytes(b"source bytes only; decoder is mocked")
        self.output = self.root / "court" / "calibration.json"
        self.args = SimpleNamespace(source=self.source, output=self.output, force=False,
                                    ui="browser", diagnostics=None, samples=3,
                                    max_dimension=320, review=False, open_browser=False,
                                    preview=None)
        self.truth, self.lines = calibration_fixture()
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        self.background = BackgroundEstimate(image, (640, 480), [0, 10, 20],
                                             [image.copy()], (.5, .5),
                                             {"motion_detected": False})
        working = CourtHomography(self.truth.image_to_court @ np.diag([2., 2., 1.]))
        self.detection = CourtDetection(working,
                                       {name: (np.asarray(line) * .5).tolist()
                                        for name, line in self.lines.items()},
                                       {"reason": "review thresholds unbenchmarked"}, "needs_review")

    def run_mocked(self):
        with patch("src.automatic_court.estimate_background", return_value=self.background), \
             patch("src.automatic_court.detect_court", return_value=self.detection):
            return calibrate_automatically(self.args)

    def test_headless_candidates_use_original_coordinates_and_never_publish(self):
        with self.assertRaises(CalibrationNeedsReview):
            self.run_mocked()
        self.assertFalse(self.output.exists())
        candidate_path = self.output.parent / "automatic" / "candidate.json"
        candidate = CourtHomography.load(candidate_path)
        data = json.loads(candidate_path.read_text())
        self.assertEqual(data["image_size"], [640, 480])
        np.testing.assert_allclose(candidate.project_to_image([[0, 0], [3, 5]]),
                                   self.truth.project_to_image([[0, 0], [3, 5]]), atol=1e-6)
        for name, line in self.lines.items():
            np.testing.assert_allclose(data["image_lines"][name], line, atol=1e-6)
        metadata = data["automatic_calibration"]
        self.assertEqual(metadata["frame_indices"], [0, 10, 20])
        self.assertEqual(metadata["source_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(metadata["review_status"], "unreviewed")

    def test_existing_output_requires_force_and_headless_force_preserves_it(self):
        self.output.parent.mkdir()
        self.output.write_text("existing calibration")
        with patch("src.automatic_court.estimate_background") as estimate:
            with self.assertRaises(FileExistsError):
                calibrate_automatically(self.args)
            estimate.assert_not_called()
        self.args.force = True
        with self.assertRaises(CalibrationNeedsReview):
            self.run_mocked()
        self.assertEqual(self.output.read_text(), "existing calibration")

    def test_camera_movement_rejects_before_detection_or_review(self):
        self.background.diagnostics["motion_detected"] = True
        self.args.review = True
        diagnostics = self.output.parent / "automatic"
        diagnostics.mkdir(parents=True)
        for name in ("candidate.json", "overlay.jpg", "validation-0.jpg"):
            (diagnostics / name).write_text("stale")
        with patch("src.automatic_court.estimate_background", return_value=self.background), \
             patch("src.automatic_court.detect_court") as detect, \
             patch("src.calibrate_court.collect_browser_selection") as review:
            with self.assertRaises(CalibrationNeedsReview):
                calibrate_automatically(self.args)
        detect.assert_not_called()
        review.assert_not_called()
        self.assertFalse(self.output.exists())
        for name in ("candidate.json", "overlay.jpg", "validation-0.jpg"):
            self.assertFalse((diagnostics / name).exists())
        report = json.loads((self.output.parent / "automatic" / "diagnostics.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("movement", report["reason"])

    def test_reviewed_result_is_atomically_published_with_multi_frame_provenance(self):
        self.args.review = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("src.calibrate_court.collect_browser_selection", return_value=(self.lines, frame, 10)) as review, \
             patch("src.automatic_court.atomic_write_json", wraps=atomic_write_json) as atomic:
            self.assertEqual(self.run_mocked(), 10)
        self.assertTrue(any(call.args[0] == self.output for call in atomic.call_args_list))
        data = json.loads(self.output.read_text())
        self.assertEqual(data["automatic_calibration"]["review_status"], "manually_reviewed")
        self.assertEqual(data["automatic_calibration"]["status"], "accepted")
        self.assertEqual(data["automatic_calibration"]["frame_indices"], [0, 10, 20])
        self.assertEqual(data["automatic_calibration"]["review_frame"], 10)
        self.assertEqual(data["image_size"], [640, 480])
        initial = review.call_args.kwargs["initial_lines"]
        for name, line in self.lines.items():
            np.testing.assert_allclose(initial[name], line)
        np.testing.assert_allclose(CourtHomography.load(self.output).project_to_image([[0, 0]]),
                                   self.truth.project_to_image([[0, 0]]), atol=1e-6)
        self.assertEqual(list(self.output.parent.glob(".calibration.json.*")), [])

    def test_existing_output_created_during_review_is_not_overwritten(self):
        self.args.review = True

        def review(*args, **kwargs):
            self.output.write_text("another writer")
            return self.lines, np.zeros((480, 640, 3), np.uint8), 0

        with patch("src.calibrate_court.collect_browser_selection", side_effect=review):
            with self.assertRaises(FileExistsError):
                self.run_mocked()
        self.assertEqual(self.output.read_text(), "another writer")

    def test_failed_detection_cleans_old_candidate_and_leaves_output_unresolved(self):
        diagnostics = self.output.parent / "automatic"
        diagnostics.mkdir(parents=True)
        (diagnostics / "candidate.json").write_text("stale")
        (diagnostics / "overlay.jpg").write_text("stale")
        self.detection = CourtDetection(None, {}, {"reason": "no lines"}, "failed")
        with self.assertRaises(CalibrationNeedsReview):
            self.run_mocked()
        self.assertFalse((diagnostics / "candidate.json").exists())
        self.assertFalse((diagnostics / "overlay.jpg").exists())
        self.assertFalse(self.output.exists())

    def test_source_changed_during_review_cannot_publish(self):
        self.args.review = True

        def review(*args, **kwargs):
            self.source.write_bytes(b"replacement video")
            return self.lines, np.zeros((480, 640, 3), np.uint8), 0

        with patch("src.calibrate_court.collect_browser_selection", side_effect=review):
            with self.assertRaisesRegex(CalibrationNeedsReview, "source changed"):
                self.run_mocked()
        self.assertFalse(self.output.exists())

    def test_diagnostic_cleanup_cannot_remove_input_or_output(self):
        self.args.diagnostics = self.root
        source = self.root / "background.jpg"
        source.write_bytes(b"source image")
        self.args.source = source
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.run_mocked()
        self.assertEqual(source.read_bytes(), b"source image")

    def test_explicit_court_region_is_scaled_and_recorded(self):
        polygon = {"image_size": [640, 480],
                   "polygon": [[0, 100], [640, 100], [640, 480], [0, 480]]}
        path = self.root / "region.json"
        path.write_text(json.dumps(polygon))
        self.args.court_region = path
        with patch("src.automatic_court.estimate_background", return_value=self.background), \
             patch("src.automatic_court.detect_court", return_value=self.detection) as detect:
            with self.assertRaises(CalibrationNeedsReview):
                calibrate_automatically(self.args)
        mask = detect.call_args.kwargs["court_region"]
        self.assertEqual(mask.shape, (240, 320))
        self.assertEqual(mask[10, 20], 0)
        self.assertEqual(mask[100, 20], 255)
        report = json.loads((self.output.parent / "automatic" / "diagnostics.json").read_text())
        self.assertEqual(report["court_region"], polygon)

    def test_court_region_rejects_mismatched_geometry_and_self_crossing_polygon(self):
        for value in [
            {"image_size": [320, 240], "polygon": [[0, 0], [640, 0], [640, 480]]},
            {"image_size": [640, 480], "polygon": [[0, 0], [640, 480], [640, 0], [0, 480]]},
            {"image_size": [640, 480], "polygon": [[0, 0], [800, 0], [640, 480]]},
        ]:
            with self.subTest(region=value), self.assertRaises(ValueError):
                court_region_mask(value, (640, 480), (320, 240))

    def test_explicit_null_region_is_rejected_instead_of_ignored(self):
        path = self.root / "region.json"
        path.write_text("null")
        self.args.court_region = path
        with self.assertRaisesRegex(ValueError, "must contain an object"):
            self.run_mocked()

    def test_preview_cannot_overwrite_source_or_calibration(self):
        for path in (self.source, self.output):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "preview must not overlap"):
                self.args.preview = path
                self.run_mocked()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.source.read_bytes(), b"source bytes only; decoder is mocked")


class BrowserReviewHTTPTests(unittest.TestCase):
    def test_optional_lines_preview_and_submit_require_valid_fit_and_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "court.png"
            image = np.zeros((480, 640, 3), np.uint8)
            cv2.imwrite(str(source), image)
            _, lines = calibration_fixture()
            ready = threading.Event()
            servers, result, errors = [], [], []

            class ObservedServer(HTTPServer):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    servers.append(self)
                    ready.set()

            def collect():
                try:
                    result.append(collect_browser_selection(
                        source, 0, mode="lines", landmark_names=[],
                        line_names=list(COURT_CALIBRATION_LINES), initial_lines=lines,
                        background=image, review=True))
                except BaseException as error:
                    errors.append(error)
                    ready.set()

            with patch("src.calibrate_court.HTTPServer", ObservedServer):
                worker = threading.Thread(target=collect, daemon=True)
                worker.start()
                self.assertTrue(ready.wait(5), "browser server did not start")
                if errors and isinstance(errors[0], PermissionError):
                    self.skipTest("local HTTP binding unavailable")
                self.assertFalse(errors, errors)
                server = servers[0]
                base = f"http://127.0.0.1:{server.server_port}"

                def post(path, payload):
                    request = Request(base + path, data=json.dumps(payload).encode(),
                                      headers={"Content-Type": "application/json"})
                    try:
                        with urlopen(request, timeout=5) as response:
                            return response.status, json.load(response)
                    except HTTPError as response:
                        with response:
                            return response.code, json.load(response)

                try:
                    with urlopen(base, timeout=5) as response:
                        page = response.read().decode()
                        self.assertIn('"optionalLines": true', page)
                        self.assertIn('"initialLines":', page)
                    with urlopen(base + "/background.jpg", timeout=5) as response:
                        self.assertEqual(response.headers.get_content_type(), "image/jpeg")
                    code, preview = post("/preview", {"lines": lines})
                    self.assertEqual(code, 200)
                    self.assertTrue(preview["segments"])
                    payload = {"mode": "lines", "frame_index": 0, "lines": lines}
                    self.assertEqual(post("/submit", payload)[0], 400)
                    payload["orientation_confirmed"] = True
                    self.assertEqual(post("/submit", {**payload, "lines": {"not_a_line": [[0, 0], [1, 1]]}})[0], 400)
                    self.assertEqual(post("/submit", {**payload, "lines": dict(list(lines.items())[:3])})[0], 422)
                    self.assertEqual(post("/submit", payload)[0], 200)
                    worker.join(5)
                    self.assertFalse(worker.is_alive())
                    self.assertFalse(errors, errors)
                    self.assertEqual(result[0][0], lines)
                    self.assertEqual(result[0][2], 0)
                finally:
                    if worker.is_alive():
                        server.shutdown()
                        worker.join(5)


if __name__ == "__main__":
    unittest.main()
