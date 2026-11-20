"""Automatic workflow stages publish reviewed data with identity in one write."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.automatic_court import CalibrationNeedsReview
from src.workflow.__main__ import main
from src.workflow.identity import SourceIdentity
from src.workflow.registry import SourceEntry


class AutomaticWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.source = SourceEntry("camera", root / "video.mp4", root / "artifacts")
        SourceIdentity("camera", "a" * 64, 30, 640, 480, 30, 1).write(self.source.identity_path)
        self.catalog = SimpleNamespace(select=lambda ids: [self.source])
        self.command = ["stage", "calibrate", "--source", "camera", "--mode", "auto"]

    def test_reviewed_stage_publishes_identity_without_temporary_authority(self):
        def calibrate(argv):
            destination = Path(argv[1])
            self.assertNotEqual(destination, self.source.calibration_path)
            self.assertFalse(self.source.calibration_path.exists())
            self.assertIn("--review", argv)
            self.assertEqual(argv[argv.index("--samples") + 1], "7")
            self.assertEqual(argv[argv.index("--court-region") + 1], "target-region.json")
            self.assertEqual(Path(argv[argv.index("--diagnostics") + 1]),
                             self.source.calibration_path.parent / "automatic")
            destination.write_text(json.dumps({
                "version": 1, "image_size": [640, 480],
                "image_to_court": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "automatic_calibration": {
                    "frame_indices": [0, 15, 29], "source_sha256": "a" * 64,
                    "status": "accepted", "review_status": "manually_reviewed",
                    "method": "classical_lines_temporal_median_v1",
                },
            }))
            return 15

        with patch("src.workflow.__main__._load_or_create", return_value=self.catalog), \
             patch("src.calibrate_court.main", side_effect=calibrate):
            self.assertEqual(main([*self.command, "--review", "--samples", "7",
                                   "--court-region", "target-region.json"]), 0)
        value = json.loads(self.source.calibration_path.read_text())
        self.assertEqual(value["artifact_metadata"]["source_frames"], [0, 15, 29])
        self.assertEqual(value["artifact_metadata"]["source_id"], "camera")

    def test_unattended_uncertain_stage_exits_two_without_publication(self):
        with patch("src.workflow.__main__._load_or_create", return_value=self.catalog), \
             patch("src.calibrate_court.main", side_effect=CalibrationNeedsReview("review required")):
            self.assertEqual(main(self.command), 2)
        self.assertFalse(self.source.calibration_path.exists())

    def test_replacement_requires_force_before_running_detector(self):
        self.source.calibration_path.parent.mkdir()
        self.source.calibration_path.write_text("original")
        with patch("src.workflow.__main__._load_or_create", return_value=self.catalog), \
             patch("src.calibrate_court.main") as calibrate:
            with self.assertRaises(FileExistsError):
                main(self.command)
            calibrate.assert_not_called()
        self.assertEqual(self.source.calibration_path.read_text(), "original")

    def test_staged_preview_cannot_overwrite_canonical_calibration(self):
        with patch("src.workflow.__main__._load_or_create", return_value=self.catalog), \
             patch("src.calibrate_court.main") as calibrate:
            with self.assertRaisesRegex(ValueError, "preview must not overlap"):
                main([*self.command, "--preview", str(self.source.calibration_path)])
            calibrate.assert_not_called()
        self.assertFalse(self.source.calibration_path.exists())


if __name__ == "__main__":
    unittest.main()
