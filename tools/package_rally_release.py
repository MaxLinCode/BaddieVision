"""Validate an explicit public asset allowlist and build a deterministic ZIP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.court_projection import CourtHomography
from src.rally_release import load_release, sha256_file, validate_manifest, verify_asset
from src.single_video.video import probe_video, read_video_info
from src.workflow.identity import SourceIdentity, inspect_source

REQUIRED_FILES = frozenset(
    {
        "manifest.json",
        "rally-segmenter.pt",
        "person-detector.pt",
        "pose-landmarker.task",
        "MODEL_CARD.md",
        "requirements-inference.txt",
        "sample.mp4",
        "sample-calibration.json",
        "sample-source.json",
        "sample-overlay.mp4",
        "sample-poster.jpg",
    }
)
OPTIONAL_FILES = frozenset({"sample-rallies.csv"})
PUBLIC_CALIBRATION_FIELDS = frozenset(
    {"version", "coordinate_system", "image_to_court", "image_size"}
)
PRIVATE_HOME = re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/")
PRIVATE_JSON_KEYS = frozenset(
    {
        "parent_video",
        "annotations",
        "annotation_rows",
        "training_rows",
        "split_manifest",
        "training_window_ids",
        "source_video",
        "video_path",
        "imported_from",
        "artifact_metadata",
        "annotator",
        "ground_truth",
        "labels",
        "targets",
        "private_source_video",
    }
)


def _check_json(value: object) -> None:
    if isinstance(value, dict):
        if set(value) & PRIVATE_JSON_KEYS:
            raise ValueError("public asset contains private annotation/provenance keys")
        for child in value.values():
            _check_json(child)
    elif isinstance(value, list):
        for child in value:
            _check_json(child)
    elif isinstance(value, str) and PRIVATE_HOME.search(value):
        raise ValueError("public asset contains a private absolute home path")


def _check_video(path: Path, identity: SourceIdentity) -> None:
    info = read_video_info(path)
    if (
        int(info["frame_count"]) != identity.frame_count
        or [int(info["width"]), int(info["height"])]
        != [identity.width, identity.height]
        or not math.isclose(
            float(info["fps"]),
            identity.fps_numerator / identity.fps_denominator,
            rel_tol=1e-5,
        )
    ):
        raise ValueError("public video geometry, FPS, or frame count is inconsistent")
    cap = cv2.VideoCapture(str(path))
    try:
        count = 0
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    if count != identity.frame_count:
        raise ValueError("public video does not decode every advertised frame")


def validate_bundle(bundle: Path) -> list[Path]:
    bundle = Path(bundle).resolve()
    paths = list(bundle.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError(
            "public release directory may contain only regular allowlisted files"
        )
    names = {path.name for path in paths}
    if names - REQUIRED_FILES - OPTIONAL_FILES:
        raise ValueError(
            f"unexpected release files: {sorted(names - REQUIRED_FILES - OPTIONAL_FILES)}"
        )
    if REQUIRED_FILES - names:
        raise ValueError(f"missing release files: {sorted(REQUIRED_FILES - names)}")
    for path in paths:
        if path.suffix in {".json", ".md", ".txt", ".csv"}:
            text = path.read_text(encoding="utf-8")
            if not text.strip() or PRIVATE_HOME.search(text):
                raise ValueError(f"empty or private text asset: {path.name}")
            if path.suffix == ".json":
                _check_json(json.loads(text))
    manifest = validate_manifest(json.loads((bundle / "manifest.json").read_text()))
    for asset in manifest["assets"].values():
        if asset["filename"] not in REQUIRED_FILES:
            raise ValueError("manifest points outside the release allowlist")
        verify_asset(bundle / asset["filename"], asset)
        if not asset["url"].startswith("https://"):
            raise ValueError("public release model assets must use HTTPS download URLs")
    load_release(bundle / "rally-segmenter.pt", manifest)
    identity = SourceIdentity.read(bundle / "sample-source.json")
    if identity.source_id != "rally-demo" or identity.parent_video is not None:
        raise ValueError(
            "sample identity must use rally-demo without private parent provenance"
        )
    actual = inspect_source("rally-demo", bundle / "sample.mp4")
    if actual.to_mapping() != identity.to_mapping():
        raise ValueError("sample identity differs from the actual public video")
    if (
        identity.frame_count / (identity.fps_numerator / identity.fps_denominator)
        > 10.1
    ):
        raise ValueError("public sample exceeds the approved short duration")
    for filename in ("sample.mp4", "sample-overlay.mp4"):
        _check_video(bundle / filename, identity)
        # The exporter removes source metadata; reject diagnostic captions in stream tags.
        tags = probe_video(bundle / filename).get("tags", {})
        if set(tags) - {"language", "handler_name", "vendor_id", "encoder"}:
            raise ValueError(
                "public sample contains unexpected embedded stream metadata"
            )
    calibration = json.loads((bundle / "sample-calibration.json").read_text())
    if set(calibration) - PUBLIC_CALIBRATION_FIELDS or calibration.get(
        "image_size"
    ) != [identity.width, identity.height]:
        raise ValueError(
            "sample calibration contains private metadata or wrong geometry"
        )
    CourtHomography.load(bundle / "sample-calibration.json")
    poster = cv2.imread(str(bundle / "sample-poster.jpg"))
    if poster is None or poster.shape[:2] != (identity.height, identity.width):
        raise ValueError("sample poster geometry differs from the video")
    if "sample-rallies.csv" in names:
        previous_end = -1
        with (bundle / "sample-rallies.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != [
                "source_id",
                "rally_id",
                "start_frame",
                "end_frame",
            ]:
                raise ValueError(
                    "sample rally CSV does not use the canonical inference schema"
                )
            for row in reader:
                start, end = int(row["start_frame"]), int(row["end_frame"])
                if (
                    row["source_id"] != identity.source_id
                    or not previous_end < start <= end < identity.frame_count
                ):
                    raise ValueError(
                        "sample rally intervals are not source-local or ordered"
                    )
                previous_end = end
    return sorted(paths, key=lambda path: path.name)


def package_release(bundle: Path, output_zip: Path) -> Path:
    """Build a ZIP only after all public assets pass validation; never glob a repo."""
    paths = validate_bundle(bundle)
    output_zip = Path(output_zip).resolve()
    if output_zip.is_relative_to(Path(bundle).resolve()):
        raise ValueError("release ZIP must be outside the validated asset directory")
    if output_zip.exists():
        raise FileExistsError("release ZIP already exists")
    checksums = "".join(f"{sha256_file(path)}  {path.name}\n" for path in paths)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".release-", dir=output_zip.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name, data in [
                *((path.name, path.read_bytes()) for path in paths),
                ("SHA256SUMS", checksums.encode()),
            ]:
                member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = 0o100644 << 16
                archive.writestr(member, data)
        os.replace(temporary, output_zip)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output_zip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()
    print(package_release(args.bundle_dir, args.output_zip))


if __name__ == "__main__":
    main()
