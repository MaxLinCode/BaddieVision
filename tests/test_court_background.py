"""Background extraction contracts; runnable with unittest without extra tools."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.court_background import (
    _alignment_diagnostics,
    distributed_frame_indices,
    estimate_background,
)


class FakeCapture:
    def __init__(self, frames):
        self.frames, self.position, self.released = frames, 0, False

    def isOpened(self):
        return True

    def get(self, property_id):
        return len(self.frames) if property_id == cv2.CAP_PROP_FRAME_COUNT else self.position

    def set(self, property_id, value):
        self.position = value
        return True

    def read(self):
        frame = self.frames[self.position]
        self.position += 1
        return frame is not None, frame

    def release(self):
        self.released = True


class BackgroundTests(unittest.TestCase):
    def test_distributed_sampling_is_source_local_unique_and_deterministic(self):
        self.assertEqual(distributed_frame_indices(100, 5), [0, 25, 50, 74, 99])
        self.assertEqual(distributed_frame_indices(3, 31), [0, 1, 2])
        with self.assertRaises(ValueError):
            distributed_frame_indices(0)

    def test_median_removes_distributed_occluders_and_preserves_raw_frames(self):
        base = np.full((80, 120, 3), 40, np.uint8)
        base[:, 60:63] = 240
        frames = [base.copy() for _ in range(5)]
        for index, frame in enumerate(frames):
            frame[10:30, index * 20:index * 20 + 15] = 255
        capture = FakeCapture(frames)
        with patch("src.court_background.cv2.VideoCapture", return_value=capture):
            estimate = estimate_background(Path("sample.mp4"), sample_count=5, max_dimension=120)
        np.testing.assert_array_equal(estimate.image, base)
        np.testing.assert_array_equal(estimate.frames[0], frames[0])
        self.assertEqual(estimate.frame_indices, [0, 1, 2, 3, 4])
        self.assertEqual(estimate.image_size, (120, 80))
        self.assertEqual(estimate.scale, (1., 1.))
        self.assertTrue(capture.released)

    def test_image_is_resized_without_upscaling_and_reports_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "court.png"
            cv2.imwrite(str(path), np.full((100, 200, 3), 60, np.uint8))
            estimate = estimate_background(path, max_dimension=100)
        self.assertEqual(estimate.image.shape, (50, 100, 3))
        self.assertEqual(estimate.scale, (.5, .5))
        self.assertEqual(estimate.diagnostics["alignment_status"], "single_image")

    def test_real_video_sampling_reports_decoded_source_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "court.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (120, 80))
            if not writer.isOpened():
                self.skipTest("MJPG writer unavailable")
            try:
                for value in range(9):
                    writer.write(np.full((80, 120, 3), value * 20, np.uint8))
            finally:
                writer.release()
            estimate = estimate_background(path, sample_count=3, max_dimension=120)
        self.assertEqual(estimate.frame_indices, [0, 4, 8])
        self.assertLess(abs(float(np.mean(estimate.image)) - 80), 3)

    def test_geometry_change_and_decode_failure_release_capture(self):
        for frames in ([np.zeros((80, 120, 3), np.uint8), np.zeros((81, 120, 3), np.uint8)],
                       [np.zeros((80, 120, 3), np.uint8), None]):
            capture = FakeCapture(frames)
            with patch("src.court_background.cv2.VideoCapture", return_value=capture):
                with self.assertRaises(ValueError):
                    estimate_background(Path("sample.mp4"), sample_count=2)
            self.assertTrue(capture.released)

    def test_motion_diagnostics_distinguish_movement_and_missing_evidence(self):
        rng = np.random.default_rng(44)
        frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        shifted = cv2.warpAffine(frame, np.float32([[1, 0, 20], [0, 1, 0]]), (320, 240))
        static = _alignment_diagnostics([frame, frame.copy(), frame.copy()])
        moving = _alignment_diagnostics([frame, shifted, shifted.copy()])
        blank = _alignment_diagnostics([np.zeros_like(frame), np.zeros_like(frame)])
        self.assertFalse(static["motion_detected"])
        self.assertEqual(static["alignment_status"], "consistent")
        self.assertTrue(moving["motion_detected"])
        self.assertEqual(blank["alignment_status"], "inconclusive")

    def test_memory_budget_is_checked_before_allocating_sample_stack(self):
        capture = FakeCapture([np.zeros((100, 200, 3), np.uint8)] * 5)
        with patch("src.court_background.cv2.VideoCapture", return_value=capture):
            with patch("src.court_background.MAX_STACK_BYTES", 100):
                with self.assertRaisesRegex(ValueError, "256 MiB"):
                    estimate_background(Path("sample.mp4"), sample_count=5)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
