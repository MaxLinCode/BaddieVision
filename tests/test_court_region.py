"""Explicit target selection and preview paths must fail closed."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.automatic_court import calibrate_automatically


class CourtRegionInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "court.jpg"
        self.source.write_bytes(b"source image")
        self.output = self.root / "calibration.json"
        self.args = SimpleNamespace(source=self.source, output=self.output, force=False,
                                    ui="browser", diagnostics=None, samples=3,
                                    max_dimension=320, review=False, open_browser=False,
                                    preview=None, court_region=None)

    def test_explicit_null_target_region_cannot_be_silently_ignored(self):
        region = self.root / "region.json"
        region.write_text("null")
        self.args.court_region = region
        with patch("src.automatic_court.estimate_background") as estimate:
            with self.assertRaises(ValueError):
                calibrate_automatically(self.args)
        estimate.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_preview_cannot_overwrite_source_output_or_region(self):
        region = self.root / "region.json"
        region.write_text('{"image_size": [640, 480], "polygon": [[0,0],[640,0],[640,480],[0,480]]}')
        self.args.court_region = region
        for protected in (self.source, self.output, region):
            with self.subTest(path=protected):
                self.args.preview = protected
                with patch("src.automatic_court.estimate_background") as estimate:
                    with self.assertRaises(ValueError):
                        calibrate_automatically(self.args)
                estimate.assert_not_called()
        self.assertEqual(self.source.read_bytes(), b"source image")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
