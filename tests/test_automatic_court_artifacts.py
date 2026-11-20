"""Canonical calibration metadata supports automatic multi-frame provenance."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.workflow.artifacts import calibration_fingerprint, register_native_calibration
from src.workflow.identity import ParentVideoProvenance, SourceIdentity
from src.workflow.registry import SourceEntry


class AutomaticCalibrationArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.source = SourceEntry("source", root / "video.mp4", root / "artifacts")
        self.identity = SourceIdentity(
            "source", "a" * 64, 100, 640, 480, 30, 1,
            parent_video=ParentVideoProvenance(start_time_seconds=60, end_time_seconds=64))
        self.identity.write(self.source.identity_path)
        self.value = {
            "version": 1, "image_size": [640, 480],
            "image_to_court": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "automatic_calibration": {
                "frame_indices": [0, 50, 99], "source_sha256": "a" * 64,
                "method": "classical_lines_temporal_median_v1",
                "status": "accepted", "review_status": "manually_reviewed",
            },
        }

    def test_multi_frame_registration_keeps_source_local_provenance(self):
        result = register_native_calibration(self.source, source_frame=50, calibration_data=self.value)
        self.assertEqual(result, self.source.calibration_path)
        data = json.loads(result.read_text())
        metadata = data["artifact_metadata"]
        self.assertEqual(metadata["source_frames"], [0, 50, 99])
        self.assertEqual(metadata["source_frame"], 50)
        self.assertEqual(metadata["frame_indexing"], "source_local_zero_based")
        self.assertEqual(metadata["source_identity_sha256"], self.identity.fingerprint)
        self.assertEqual(metadata["method"], "classical_lines_temporal_median_v1")
        self.assertTrue(calibration_fingerprint(self.source))

    def test_source_mismatch_and_unreviewed_results_cannot_replace_existing_calibration(self):
        self.source.calibration_path.parent.mkdir()
        self.source.calibration_path.write_text("existing")
        cases = [{"source_sha256": "b" * 64}, {"status": "needs_review"},
                 {"review_status": "unreviewed"}, {"frame_indices": []},
                 {"frame_indices": [0, 100]}, {"frame_indices": [False, 50]},
                 {"frame_indices": [-1, 50]}]
        for mutation in cases:
            with self.subTest(mutation=mutation):
                value = copy.deepcopy(self.value)
                value["automatic_calibration"].update(mutation)
                with self.assertRaises(ValueError):
                    register_native_calibration(self.source, source_frame=50, calibration_data=value)
                self.assertEqual(self.source.calibration_path.read_text(), "existing")

    def test_legacy_manual_registration_remains_compatible(self):
        value = copy.deepcopy(self.value)
        del value["automatic_calibration"]
        register_native_calibration(self.source, source_frame=0, calibration_data=value)
        data = json.loads(self.source.calibration_path.read_text())
        self.assertNotIn("source_frames", data["artifact_metadata"])
        self.assertTrue(calibration_fingerprint(self.source))

    def test_geometry_and_representative_frame_must_match_identity(self):
        for size, index in [([320, 240], 0), ([640, 480], 100), ([640, 480], -1)]:
            with self.subTest(size=size, index=index):
                value = {**self.value, "image_size": size}
                with self.assertRaises(ValueError):
                    register_native_calibration(self.source, source_frame=index, calibration_data=value)
        self.assertFalse(self.source.calibration_path.exists())


if __name__ == "__main__":
    unittest.main()
