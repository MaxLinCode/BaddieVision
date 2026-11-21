"""Exercise unlabeled production inference on generated video and observations."""

import csv
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.temporal_selector.rally import RallySegmenter, RallySegmenterConfig
from src.temporal_selector.rally_audio import _centered_flux_excess, _rolling_zscore
from src.temporal_selector.rally_dataset import (
    RallyDataConfig,
    RallySourceConfig,
    RallyWindowDataset,
)
from src.temporal_selector.rally_features import rally_feature_view
from src.temporal_selector.rally_inference import (
    predict_source,
    predict_windows,
    write_prediction_artifacts,
)
from src.workflow.identity import SourceIdentity, sha256_file


def _jsonl(path, metadata, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in [metadata, *rows]) + "\n")


@pytest.fixture
def source(tmp_path, request):
    frame_count = getattr(request, "param", 11)
    video = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (128, 96))
    assert writer.isOpened()
    for number in range(frame_count):
        writer.write(np.full((96, 128, 3), number * 10, np.uint8))
    writer.release()
    identity = SourceIdentity("generated", sha256_file(video), frame_count, 128, 96, 5, 1)
    identity.write(tmp_path / "source.json")
    calibration = tmp_path / "court" / "calibration.json"
    calibration.parent.mkdir()
    calibration.write_text(json.dumps({
        "version": 1, "image_size": [128, 96],
        "image_to_court": [[.05, 0, -3.2], [0, .1, -4.8], [0, 0, 1]],
        "artifact_metadata": {
            "source_id": identity.source_id, "source_identity_sha256": identity.fingerprint,
        },
    }))
    assignments = tmp_path / "players" / "assignments.jsonl"
    poses = tmp_path / "players" / "poses.jsonl"
    lineage = "sha256:" + "a" * 64
    assignment_rows = [{
        "type": "frame", "frame": frame,
        "slots": {
            "P1": {"confidence": .9, "assignment": {
                "track_id": 1, "bbox": [20, 10, 40, 40], "foot": [30, 40],
                "court_x": -1.7, "court_y": -.8,
                "detection_confidence": .9, "activity": .05,
            }} if frame != 4 else {"assignment": None},
            "P2": {"assignment": None},
        },
    } for frame in range(frame_count)]
    _jsonl(assignments, {
        "type": "metadata", "frame_count": frame_count, "fps": 5, "frame_size": [128, 96],
        "source_id": identity.source_id, "source_identity_sha256": identity.fingerprint,
        "calibration_sha256": sha256_file(calibration), "raw_artifact_fingerprint": lineage,
    }, assignment_rows)
    _jsonl(poses, {
        "type": "metadata", "raw_artifact_fingerprint": lineage,
        "pose_model_fingerprint": lineage, "preprocessing_fingerprint": lineage,
    }, [{
        "frame": frame, "track_id": 1, "status": "detected", "bbox": [20, 10, 40, 40],
        "landmarks": [{"x": .5, "y": .5, "visibility": 1.} for _ in range(33)],
    } for frame in range(frame_count)])
    return RallySourceConfig(identity.source_id, video, assignments, poses, calibration)


def _model(feature_view="full"):
    model = RallySegmenter(RallySegmenterConfig(
        frame_feature_dim=rally_feature_view(feature_view).dimension,
        token_size=8, num_layers=1, num_attention_heads=2, feed_forward_size=16, dropout=0,
    ))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def _predict(source, *, threshold=.49):
    return predict_source(source, _model(), preprocessing={}, decoder={
        "threshold": threshold, "max_gap_seconds": 0., "minimum_duration_seconds": 0.,
    }, batch_size=2)


def test_unlabeled_model_inference_csv_and_full_video_overlay(source, tmp_path, monkeypatch):
    # Fail if inference accidentally reads teachers, shuttle artifacts, or labels.
    original_open = Path.open

    def checked_open(path, *args, **kwargs):
        assert "shuttle" not in str(path) and ".annotation" not in str(path)
        assert path.name not in {"rallies.csv", "rallies.manifest.json"}
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "open", checked_open)
        rows = _predict(source)
    assert [row["frame"] for row in rows] == list(range(11))
    assert all(row["inplay_probability"] == .5 for row in rows)
    assert all(row["decoded_inplay"] and row["rally_id"] == "generated-0001" for row in rows)
    output = write_prediction_artifacts(source, rows, tmp_path / "predictions")
    with output["rallies"].open() as handle:
        assert list(csv.DictReader(handle)) == [{
            "source_id": "generated", "rally_id": "generated-0001", "start_frame": "0", "end_frame": "10",
        }]
    stored = [json.loads(line) for line in output["predictions"].read_text().splitlines()]
    assert stored[0]["frame_indexing"] == "source_local_zero_based"
    assert len(stored) == 12
    capture = cv2.VideoCapture(str(output["overlay"]))
    count = 0
    while capture.read()[0]:
        count += 1
    assert count == 11
    assert capture.get(cv2.CAP_PROP_FPS) == 5
    capture.release()
    assert not list((tmp_path / "predictions").glob(".rally-output-*"))


def test_no_rally_video_writes_header_only_csv(source, tmp_path):
    rows = _predict(source, threshold=.51)
    output = write_prediction_artifacts(source, rows, tmp_path / "result", overlay=False)
    assert output["rallies"].read_text().strip() == "source_id,rally_id,start_frame,end_frame"
    assert "overlay" not in output
    assert all(row["rally_id"] is None for row in rows)


def test_targets_remain_masked_and_duplicate_ownership_is_rejected(source):
    dataset = RallyWindowDataset(RallyDataConfig((source,), None, None))
    assert all(torch.all(window.inplay_targets == -100) for window in dataset.windows)
    assert len(predict_windows(_model(), dataset.windows)) == 11
    with pytest.raises(ValueError, match="duplicate owned"):
        predict_windows(_model(), [dataset.windows[0], dataset.windows[0]])
    with pytest.raises(ValueError, match="one source"):
        predict_windows(_model(), [dataset.windows[0], replace(dataset.windows[0], source_id="other")])


@pytest.mark.parametrize("decoder", [
    {"threshold": float("nan")}, {"max_gap_seconds": float("inf")},
    {"minimum_duration_seconds": float("nan")}, {"preserve_edge_runs": "yes"},
])
def test_reject_invalid_decoder_before_inference(source, decoder):
    with pytest.raises((TypeError, ValueError)):
        predict_source(source, _model(), preprocessing={}, decoder=decoder)


def test_reject_stale_video_identity(source):
    source.video_path.write_bytes(source.video_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="identity"):
        _predict(source)


def test_reject_nonfinite_logits_and_batch_size(source):
    dataset = RallyWindowDataset(RallyDataConfig((source,), None, None))
    with pytest.raises(ValueError, match="batch size"):
        predict_windows(_model(), dataset.windows, batch_size=0)
    model = _model()
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        predict_windows(model, dataset.windows)


def test_failed_overlay_does_not_publish_partial_artifacts(source, tmp_path, monkeypatch):
    rows = _predict(source)

    def fail(*args):
        raise RuntimeError("encoder unavailable")

    monkeypatch.setattr("src.temporal_selector.rally_inference._render_overlay", fail)
    destination = tmp_path / "result"
    with pytest.raises(RuntimeError, match="encoder unavailable"):
        write_prediction_artifacts(source, rows, destination)
    assert list(destination.iterdir()) == []


def test_artifact_writer_rejects_missing_or_duplicate_frame(source, tmp_path):
    rows = _predict(source)
    for invalid in (rows[:-1], [rows[0], *rows]):
        with pytest.raises(ValueError, match="every source frame"):
            write_prediction_artifacts(source, invalid, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_reject_calibration_geometry_mismatch(source):
    calibration = json.loads(source.calibration_path.read_text())
    calibration["image_size"] = [640, 480]
    source.calibration_path.write_text(json.dumps(calibration))
    lines = source.assignments_path.read_text().splitlines()
    metadata = json.loads(lines[0])
    metadata["calibration_sha256"] = sha256_file(source.calibration_path)
    source.assignments_path.write_text("\n".join([json.dumps(metadata), *lines[1:]]) + "\n")
    with pytest.raises(ValueError, match="image_size"):
        _predict(source)


@pytest.mark.parametrize("source", [1], indirect=True)
def test_single_frame_video_is_not_lost_to_context_or_duration(source, tmp_path):
    rows = _predict(source)
    assert len(rows) == 1 and rows[0]["frame"] == 0
    output = write_prediction_artifacts(source, rows, tmp_path / "one")
    capture = cv2.VideoCapture(str(output["overlay"]))
    assert capture.read()[0]
    assert not capture.read()[0]
    capture.release()


def test_missing_all_players_and_poses_still_covers_every_frame(source):
    lines = [json.loads(line) for line in source.assignments_path.read_text().splitlines()]
    for row in lines[1:]:
        row["slots"] = {"P1": {"assignment": None}, "P2": {"assignment": None}}
    source.assignments_path.write_text("\n".join(json.dumps(row) for row in lines) + "\n")
    pose_metadata = source.pose_cache_path.read_text().splitlines()[0]
    source.pose_cache_path.write_text(pose_metadata + "\n")
    assert len(_predict(source)) == 11


@pytest.mark.skipif(
    not shutil.which("ffprobe") or not shutil.which("ffmpeg"), reason="requires FFmpeg"
)
def test_audio_view_keeps_visual_features_and_aligns_delayed_track(source, tmp_path):
    visual = RallyWindowDataset(RallyDataConfig((source,), None, None))
    silent = RallyWindowDataset(RallyDataConfig((source,), None, None, feature_view="full_audio"))
    visual_dim = visual.feature_view.dimension
    for baseline, window in zip(visual, silent, strict=True):
        torch.testing.assert_close(window.frame_values[:, :visual_dim], baseline.frame_values)
        torch.testing.assert_close(window.frame_validity[:, :visual_dim], baseline.frame_validity)
        assert not window.frame_validity[:, visual_dim:].any()
        assert not window.frame_values[:, visual_dim:].any()

    video = tmp_path / "delayed-audio.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(source.video_path),
        "-itsoffset", "0.6", "-f", "lavfi", "-i", "sine=frequency=1000:duration=0.6",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "alac", str(video),
    ], check=True, capture_output=True)
    source = replace(source, video_path=video)
    identity_path = tmp_path / "source.json"
    identity = replace(SourceIdentity.read(identity_path), video_sha256=sha256_file(video))
    identity.write(identity_path)
    calibration = json.loads(source.calibration_path.read_text())
    calibration["artifact_metadata"]["source_identity_sha256"] = identity.fingerprint
    source.calibration_path.write_text(json.dumps(calibration))
    metadata, *observations = map(json.loads, source.assignments_path.read_text().splitlines())
    metadata.update(
        source_identity_sha256=identity.fingerprint,
        calibration_sha256=sha256_file(source.calibration_path),
    )
    _jsonl(source.assignments_path, metadata, observations)

    with_audio = RallyWindowDataset(
        RallyDataConfig((source,), None, None, feature_view="full_audio")
    )
    assert with_audio.manifest["audio_coverage"][source.source_id] == {
        "frames": 11, "energy_frames": 3, "flux_frames": 3,
    }
    for window in with_audio:
        for index, frame in enumerate(window.frame_indices):
            valid = window.frame_validity[index, visual_dim:]
            if frame in (3, 4, 5):
                assert valid.all()
                bands = window.frame_values[index, visual_dim + 2:visual_dim + 6]
                assert bands.argmax().item() == 1
            else:
                assert not valid.any()
                assert not window.frame_values[index, visual_dim:].any()
    normalized = RallyWindowDataset(
        RallyDataConfig((source,), None, None, feature_view="full_audio_zscore")
    )
    assert normalized.manifest["audio_coverage"] == with_audio.manifest["audio_coverage"]
    for raw, standardized in zip(with_audio, normalized, strict=True):
        torch.testing.assert_close(standardized.frame_validity, raw.frame_validity)
        torch.testing.assert_close(standardized.frame_values[:, :visual_dim], raw.frame_values[:, :visual_dim])
        if 3 in standardized.frame_indices:
            first_observed = standardized.frame_indices.index(3)
            assert not standardized.frame_values[first_observed, visual_dim:].any()

    centered_flux = RallyWindowDataset(
        RallyDataConfig((source,), None, None, feature_view="full_audio_centered_flux")
    )
    assert centered_flux.manifest["audio_coverage"] == with_audio.manifest["audio_coverage"]
    for raw, centered in zip(with_audio, centered_flux, strict=True):
        torch.testing.assert_close(centered.frame_validity, raw.frame_validity)
        torch.testing.assert_close(centered.frame_values[:, :-1], raw.frame_values[:, :-1], rtol=0, atol=0)

    rows = predict_source(
        source, _model("full_audio_zscore"), preprocessing={"feature_view": "full_audio_zscore"},
        decoder={"threshold": 0.49, "max_gap_seconds": 0, "minimum_duration_seconds": 0},
        batch_size=2,
    )
    assert [row["frame"] for row in rows] == list(range(11))


def test_audio_zscore_uses_trailing_observed_values_for_all_seven_features():
    times = np.array([0.0, 1.0, 2.0, 11.0, 11.5, 22.0])
    values = np.arange(42, dtype=np.float32).reshape(6, 7) ** 2
    values[:, -1] = 1.0
    valid = np.ones_like(values, dtype=bool)
    valid[2, ::2] = False
    valid[3, 1::2] = False
    expected = np.zeros_like(values)
    for frame, timestamp in enumerate(times):
        for feature in range(values.shape[1]):
            if valid[frame, feature]:
                series = values[
                    (times > timestamp - 10.0) & (times <= timestamp) & valid[:, feature],
                    feature,
                ]
                if len(series) > 1 and series.std() > 1e-6:
                    expected[frame, feature] = (values[frame, feature] - series.mean()) / series.std()

    actual = _rolling_zscore(values, valid, times)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(_rolling_zscore(values * 2 + 30, valid, times), actual, atol=1e-6)
    normalized_prefix = _rolling_zscore(values[:4], valid[:4], times[:4])
    np.testing.assert_array_equal(normalized_prefix, actual[:4])


def test_centered_flux_uses_future_background_and_excludes_missing_samples():
    times = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 1.34, 1.36, 1.5, 2.4, 4, 4.1, 4.2])
    flux = np.array([0, 0.2, 0.4, 0.9, 9, 0.6, 0.7, 0.8, 9, 0.1, 10, 0.1], dtype=np.float32)
    valid = np.array([True, True, True, True, False, True, True, True, True, True, True, True])
    result = _centered_flux_excess(flux, valid, times)
    # At 0.6s the valid neighborhood has median 0.4 and MAD 0.2.
    assert result[3] == pytest.approx((0.9 - 0.4) / (1.4826 * 0.2))
    assert result[4] == 0
    assert result[8] == 0  # An isolated observation defines its own baseline.
    assert result[10] == 8  # A transient above flat background is bounded.
    assert (result >= 0).all()
