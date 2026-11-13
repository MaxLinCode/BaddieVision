"""Public sample and package boundaries use generated media, never private data."""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import asdict

import cv2
import numpy as np
import pytest
import torch

from src.rally_release import export_release, sha256_file
from src.temporal_selector.joint_inference import InPlayDecoderConfig
from src.temporal_selector.rally import (
    RallySegmenter,
    RallySegmenterConfig,
    RallyTrainingRecipe,
    rally_checkpoint,
)
from tools.package_rally_release import REQUIRED_FILES, package_release
from tools.prepare_rally_sample import prepare_sample


@pytest.fixture
def sample(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg required for public sample export")
    video = tmp_path / "private-recording.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 6, (64, 48))
    assert writer.isOpened()
    for index in range(24):
        writer.write(np.full((48, 64, 3), index * 8, dtype=np.uint8))
    writer.release()
    calibration = tmp_path / "private-calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "version": 1,
                "image_size": [64, 48],
                "image_to_court": [[0.1, 0, -3.2], [0, 0.2, -4.8], [0, 0, 1]],
                "artifact_metadata": {
                    "source_id": "private-person",
                    "imported_from": "/private/private-video",
                },
            }
        )
    )
    public = prepare_sample(
        video, calibration, tmp_path / "sample", center_frame=12, duration_seconds=2
    )
    return public, video, calibration


def test_sample_is_frame_exact_rebased_and_sanitized(sample):
    public, _, _ = sample
    assert {p.name for p in public.iterdir()} == {
        "sample.mp4",
        "sample-source.json",
        "sample-calibration.json",
    }
    identity = json.loads((public / "sample-source.json").read_text())
    assert identity["source_id"] == "rally-demo"
    assert identity["frame_count"] == 12
    assert identity["frame_range_start"] == 0
    assert identity["frame_range_end_inclusive"] == 11
    assert identity["fps_numerator"] / identity["fps_denominator"] == 6
    assert "parent_video" not in identity
    public_text = (public / "sample-source.json").read_text() + (
        public / "sample-calibration.json"
    ).read_text()
    assert "private" not in public_text
    assert "artifact_metadata" not in public_text
    private = json.loads(
        (public.parent / ".sample.private-provenance.json").read_text()
    )
    assert private["source_start_frame"] == 6
    assert private["source_end_frame_inclusive"] == 17
    cap = cv2.VideoCapture(str(public / "sample.mp4"))
    try:
        success, first = cap.read()
        assert success
        assert abs(float(first.mean()) - 6 * 8) < 8
    finally:
        cap.release()


def test_sample_rejects_outside_frame_or_public_provenance(sample, tmp_path):
    _, video, calibration = sample
    with pytest.raises(ValueError, match="outside"):
        prepare_sample(video, calibration, tmp_path / "bad", center_frame=999)
    with pytest.raises(ValueError, match="at most ten"):
        prepare_sample(
            video, calibration, tmp_path / "bad", center_frame=0, duration_seconds=11
        )
    with pytest.raises(ValueError, match="outside the public"):
        prepare_sample(
            video,
            calibration,
            tmp_path / "bad",
            center_frame=0,
            private_provenance=tmp_path / "bad" / "private.json",
        )


@pytest.fixture
def bundle(sample, tmp_path):
    public, _, _ = sample
    model = RallySegmenter(
        RallySegmenterConfig(
            token_size=8,
            num_layers=1,
            num_attention_heads=2,
            feed_forward_size=16,
            dropout=0,
        )
    )
    research = tmp_path / "research.pt"
    torch.save(
        rally_checkpoint(
            model,
            dataset_fingerprint="private-dataset",
            annotation_fingerprint="private-annotations",
            split_manifest={"private": "private-video"},
            training_recipe=RallyTrainingRecipe(),
            chosen_epoch=1,
            decoder_configuration=asdict(InPlayDecoderConfig()),
        ),
        research,
    )
    detector, pose = tmp_path / "person.pt", tmp_path / "pose.task"
    detector.write_bytes(b"generated-model-fixture")
    pose.write_bytes(b"generated-pose-fixture")
    output = tmp_path / "release"
    export_release(
        research,
        output,
        person_detector=detector,
        pose_landmarker=pose,
        code_revision="a" * 40,
        base_url="https://example.com/rally-v1",
    )
    for path in public.iterdir():
        shutil.copyfile(path, output / path.name)
    shutil.copyfile(public / "sample.mp4", output / "sample-overlay.mp4")
    cap = cv2.VideoCapture(str(public / "sample.mp4"))
    success, image = cap.read()
    cap.release()
    assert success
    assert cv2.imwrite(str(output / "sample-poster.jpg"), image)
    (output / "MODEL_CARD.md").write_text(
        "# Synthetic model fixture\nNo accuracy claim.\n"
    )
    (output / "requirements-inference.txt").write_text("torch==2.7.1\n")
    return output


def test_release_zip_is_allowlisted_reproducible_and_checksummed(bundle, tmp_path):
    first = package_release(bundle, tmp_path / "one.zip")
    second = package_release(bundle, tmp_path / "two.zip")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert set(archive.namelist()) == REQUIRED_FILES | {"SHA256SUMS"}
        checksums = archive.read("SHA256SUMS").decode()
        assert f"{sha256_file(bundle / 'sample.mp4')}  sample.mp4" in checksums
        assert "private-recording" not in checksums


def test_release_rejects_extra_private_files(bundle, tmp_path):
    (bundle / "rallies.csv").write_text("private training labels")
    with pytest.raises(ValueError, match="unexpected release files"):
        package_release(bundle, tmp_path / "bad.zip")
    assert not (tmp_path / "bad.zip").exists()


def test_release_rejects_private_calibration_metadata(bundle, tmp_path):
    path = bundle / "sample-calibration.json"
    value = json.loads(path.read_text())
    value["artifact_metadata"] = {"source_id": "private-person"}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="private"):
        package_release(bundle, tmp_path / "bad.zip")


def test_release_rejects_bad_geometry_and_corrupt_weights(bundle, tmp_path):
    source = bundle / "sample-source.json"
    original = source.read_text()
    value = json.loads(original)
    value["width"] += 1
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="differs"):
        package_release(bundle, tmp_path / "bad.zip")
    source.write_text(original)
    (bundle / "rally-segmenter.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        package_release(bundle, tmp_path / "bad.zip")


def test_release_rejects_symlinks(bundle, tmp_path):
    path = bundle / "MODEL_CARD.md"
    path.unlink()
    outside = tmp_path / "outside.md"
    outside.write_text("external")
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="regular allowlisted"):
        package_release(bundle, tmp_path / "bad.zip")
