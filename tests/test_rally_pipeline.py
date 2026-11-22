"""End-to-end pipeline tests without private media or pretrained model downloads."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.rally_release import DEFAULT_PREPROCESSING, export_release, resolve_release
from src.temporal_selector.joint_inference import InPlayDecoderConfig
from src.temporal_selector.rally import (
    RallySegmenter,
    RallySegmenterConfig,
    RallyTrainingRecipe,
    rally_checkpoint,
)
from src.temporal_selector.rally_features import RALLY_FEATURE_VIEWS
from src.workflow.identity import SourceIdentity, inspect_source, sha256_file
from src.workflow.rally_pipeline import run_pipeline, select_device
from src.workflow.registry import SourceCatalog


def _write_jsonl(path, metadata, rows):
    path.write_text("\n".join(json.dumps(item) for item in [metadata, *rows]) + "\n")


@pytest.fixture
def pipeline(tmp_path):
    video = tmp_path / "match.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 6, (128, 96))
    assert writer.isOpened()
    for frame in range(9):
        writer.write(np.full((96, 128, 3), frame * 8, dtype=np.uint8))
    writer.release()
    calibration = tmp_path / "external-court.json"
    calibration.write_text(
        json.dumps(
            {
                "version": 1,
                "image_size": [128, 96],
                "image_to_court": [[0.05, 0, -3.2], [0, 0.1, -4.8], [0, 0, 1]],
            }
        )
    )
    model = RallySegmenter(
        RallySegmenterConfig(
            token_size=8,
            num_layers=1,
            num_attention_heads=2,
            feed_forward_size=16,
            dropout=0,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    research = tmp_path / "research.pt"
    torch.save(
        rally_checkpoint(
            model,
            dataset_fingerprint="test",
            annotation_fingerprint="test",
            split_manifest={},
            training_recipe=RallyTrainingRecipe(),
            chosen_epoch=1,
            decoder_configuration=asdict(
                InPlayDecoderConfig(threshold=0.49, minimum_duration_seconds=0)
            ),
        ),
        research,
    )
    detector, pose = tmp_path / "detector.pt", tmp_path / "pose.task"
    detector.write_bytes(b"deterministic-person-model")
    pose.write_bytes(b"deterministic-pose-model")
    release = tmp_path / "release"
    manifest = export_release(
        research,
        release,
        person_detector=detector,
        pose_landmarker=pose,
        code_revision="b" * 40,
        base_url=release.as_uri(),
    )
    calls = {"person": 0, "players": 0}

    def extract(video_path, destination, model_path):
        assert Path(model_path).read_bytes() == detector.read_bytes()
        identity = inspect_source("test", video_path)
        calls["person"] += 1
        _write_jsonl(
            destination,
            {
                "type": "metadata",
                "schema": "person_tracks",
                "schema_version": 1,
                "frame_count": identity.frame_count,
                "frame_size": [identity.width, identity.height],
                "fps": identity.fps_numerator / identity.fps_denominator,
            },
            [],
        )

    def interpret(
        video_path,
        person_path,
        calibration_path,
        poses,
        assignments,
        features,
        *,
        pose_model_asset,
        pose_crop_padding,
    ):
        assert Path(pose_model_asset).read_bytes() == pose.read_bytes()
        assert pose_crop_padding == 0.20
        identity = SourceIdentity.read(calibration_path.parent.parent / "source.json")
        raw = "sha256:" + sha256_file(person_path)
        calls["players"] += 1
        _write_jsonl(
            assignments,
            {
                "type": "metadata",
                "schema": "player_assignments",
                "schema_version": 3,
                "source_id": identity.source_id,
                "source_identity_sha256": identity.fingerprint,
                "calibration_sha256": sha256_file(calibration_path),
                "raw_artifact_fingerprint": raw,
                "frame_count": identity.frame_count,
                "fps": identity.fps_numerator / identity.fps_denominator,
                "frame_size": [identity.width, identity.height],
            },
            [
                {
                    "type": "frame",
                    "frame": frame,
                    "slots": {"P1": {"assignment": None}, "P2": {"assignment": None}},
                }
                for frame in range(identity.frame_count)
            ],
        )
        _write_jsonl(
            poses,
            {
                "type": "metadata",
                "schema": "pose_cache",
                "schema_version": 1,
                "raw_artifact_fingerprint": raw,
                "pose_model_fingerprint": "sha256:" + sha256_file(pose_model_asset),
                "preprocessing_fingerprint": "sha256:" + "e" * 64,
            },
            [],
        )
        features.write_text(
            "Frame\n" + "\n".join(map(str, range(identity.frame_count))) + "\n"
        )

    catalog = SourceCatalog(tmp_path / "catalog.json")
    return {
        "catalog": catalog,
        "video": video,
        "calibration": calibration,
        "output_dir": tmp_path / "result",
        "source_id": "generated",
        "checkpoint": release / "rally-segmenter.pt",
        "model_manifest": manifest,
        "cache_dir": tmp_path / "cache",
        "device": "cpu",
        "batch_size": 2,
        "person_extractor": extract,
        "player_interpreter": interpret,
    }, calls


def test_rtmpose_release_converts_fps_before_extraction_and_reuses_source(
    pipeline, tmp_path, monkeypatch,
):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe are required")
    kwargs, calls = pipeline
    raw_video = kwargs["video"]
    audio_video = tmp_path / "with-audio.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(raw_video),
         "-f", "lavfi", "-i", "sine=frequency=660", "-shortest",
         "-vf", "fps=60", "-c:v", "libx264", "-c:a", "aac", str(audio_video)],
        check=True,
    )
    pose = tmp_path / "rtmpose.onnx"
    pose.write_bytes(b"test pose model")
    research = tmp_path / "rtm-research.pt"
    model = RallySegmenter(RallySegmenterConfig(
        frame_feature_dim=RALLY_FEATURE_VIEWS["full_audio_zscore"].dimension,
        audio_attention=True,
        token_size=8, num_layers=1, num_attention_heads=2, feed_forward_size=16,
    ))
    torch.save(rally_checkpoint(
        model, dataset_fingerprint="test", annotation_fingerprint="test",
        split_manifest={}, training_recipe=RallyTrainingRecipe(temporal_smoothing_weight=0.1),
        chosen_epoch=1, feature_view="full_audio_zscore",
        decoder_configuration=asdict(InPlayDecoderConfig(continuation_threshold=0.4)),
        pose_training_metadata={
            "pose_model_sha256": sha256_file(pose),
            "preprocessing": DEFAULT_PREPROCESSING,
        },
    ), research)
    monkeypatch.setattr("src.rally_release.open_session", lambda *args: (object(), "CPU"))
    monkeypatch.setattr("src.rally_release.validate_fp16_halpe26_model", lambda _: None)
    monkeypatch.setattr("src.pose_backends.version", lambda _: "test")
    release = tmp_path / "rtm-release"
    manifest = export_release(
        research, release, pose_model=pose, code_revision="a" * 40,
        person_detector=tmp_path / "detector.pt", base_url=release.as_uri(),
    )
    kwargs.update(
        video=audio_video, checkpoint=release / "rally-segmenter.pt",
        model_manifest=manifest, player_interpreter=None, overlay=False,
    )
    outputs = run_pipeline(**kwargs)
    source = kwargs["catalog"].select(["generated"])[0]
    identity = SourceIdentity.read(source.identity_path)
    assert (identity.fps_numerator, identity.fps_denominator) == (30, 1)
    assert identity.frame_count == 45
    assert identity.parent_video.video_sha256 == sha256_file(audio_video)
    assert source.video_path != audio_video
    with outputs["predictions"].open() as handle:
        assert len(handle.readlines()) == identity.frame_count + 1
    from src.temporal_selector.rally_audio import read_rally_audio

    validity = read_rally_audio(source.video_path, frame_count=45, fps=30).validity
    assert validity[:, 0].any()
    assert run_pipeline(**kwargs) == outputs
    assert calls["person"] == 1

    manifest_value = json.loads(manifest.read_text())
    pose_entry = manifest_value["assets"]["pose_model"]
    replacement = release / pose_entry["filename"]
    replacement.write_bytes(b"different pose model")
    pose_entry.update(sha256=sha256_file(replacement), size_bytes=replacement.stat().st_size)
    manifest.write_text(json.dumps(manifest_value))
    resolve_kwargs = {
        "checkpoint": kwargs["checkpoint"], "manifest": manifest, "cache_dir": kwargs["cache_dir"],
    }
    with pytest.raises(ValueError, match="RTMPose asset SHA-256 differs from release preprocessing"):
        resolve_release(**resolve_kwargs)

    manifest_value["preprocessing"]["pose_model_sha256"] = pose_entry["sha256"]
    manifest.write_text(json.dumps(manifest_value))
    with pytest.raises(ValueError, match="checkpoint and manifest inference contracts differ"):
        resolve_release(**resolve_kwargs)


def test_complete_pipeline_and_verified_cached_rerun(pipeline, monkeypatch):
    from InPlay.heuristic.person_tracks import BYTETRACK_PERIODIC

    kwargs, calls = pipeline
    outputs = run_pipeline(**kwargs)
    for filename in ("person-tracks.manifest.json", "players.manifest.json"):
        receipt = json.loads((kwargs["output_dir"] / "players" / filename).read_text())
        assert receipt["inputs"]["tracking_profile"] == BYTETRACK_PERIODIC
    assert set(outputs) == {"manifest", "predictions", "rallies", "overlay"}
    with outputs["rallies"].open() as handle:
        assert list(csv.DictReader(handle)) == [
            {
                "source_id": "generated",
                "rally_id": "generated-0001",
                "start_frame": "0",
                "end_frame": "8",
            }
        ]
    rows = [
        json.loads(line) for line in outputs["predictions"].read_text().splitlines()
    ]
    assert [row["frame"] for row in rows[1:]] == list(range(9))
    assert all(row["inplay_probability"] == 0.5 for row in rows[1:])
    capture = cv2.VideoCapture(str(outputs["overlay"]))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 9
    assert capture.get(cv2.CAP_PROP_FPS) == 6
    capture.release()
    hashes = {name: sha256_file(path) for name, path in outputs.items()}
    monkeypatch.setattr(
        "src.rally_release.urlopen", lambda *args, **kw: pytest.fail("network used")
    )
    registered = dict(kwargs, video=None, output_dir=None, calibration=None)
    assert run_pipeline(**registered) == outputs
    assert {name: sha256_file(path) for name, path in outputs.items()} == hashes
    assert calls == {"person": 1, "players": 1}


@pytest.mark.parametrize(
    "relative",
    [
        "players/person_tracks.jsonl",
        "players/assignments.jsonl",
        "players/poses.jsonl",
        "players/features.csv",
        "rallies/predictions.jsonl",
        "rallies/overlay.mp4",
    ],
)
def test_corrupt_outputs_require_force_and_regenerate(pipeline, relative):
    kwargs, calls = pipeline
    run_pipeline(**kwargs)
    changed = kwargs["output_dir"] / relative
    changed.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="stale or incomplete"):
        run_pipeline(**kwargs)
    outputs = run_pipeline(**kwargs, force=True)
    assert outputs["manifest"].is_file()
    assert calls == {"person": 2, "players": 2}
    assert changed.read_bytes() != b"corrupted"


def test_changed_calibration_invalidates_descendants(pipeline):
    kwargs, calls = pipeline
    run_pipeline(**kwargs)
    value = json.loads(kwargs["calibration"].read_text())
    value["image_to_court"][0][2] = -3.1
    kwargs["calibration"].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="calibration changed"):
        run_pipeline(**kwargs)
    run_pipeline(**kwargs, force=True)
    assert calls == {"person": 2, "players": 2}


def test_corrupt_asset_never_reaches_extraction(pipeline):
    kwargs, calls = pipeline
    run_pipeline(**kwargs)
    manifest = json.loads(kwargs["model_manifest"].read_text())
    filename = manifest["assets"]["person_detector"]["filename"]
    asset = kwargs["model_manifest"].parent / filename
    asset.write_bytes(b"wrong-model")
    with pytest.raises(ValueError, match="SHA-256"):
        run_pipeline(**kwargs, force=True)
    assert calls == {"person": 1, "players": 1}


@pytest.mark.parametrize("stage", ["person", "players"])
def test_interrupted_stage_is_not_published_and_can_resume(pipeline, stage):
    kwargs, calls = pipeline
    replacement = "person_extractor" if stage == "person" else "player_interpreter"

    def interrupt(*args, **kw):
        destination = args[1] if stage == "person" else args[4]
        destination.write_text("partially written")
        raise RuntimeError("interrupted extraction")

    with pytest.raises(RuntimeError, match="interrupted"):
        run_pipeline(**dict(kwargs, **{replacement: interrupt}))
    root = kwargs["output_dir"]
    assert not (
        root
        / "players"
        / ("person_tracks.jsonl" if stage == "person" else "assignments.jsonl")
    ).exists()
    assert not (root / "rallies" / "run.manifest.json").exists()
    assert not [path for path in (root / "players").glob(".person-*") if path.is_dir()]
    assert not [path for path in (root / "players").glob(".players-*") if path.is_dir()]
    run_pipeline(**kwargs)
    assert calls == {"person": 1, "players": 1}


def test_missing_pose_output_never_receives_completed_player_receipt(pipeline):
    kwargs, _ = pipeline
    original = kwargs["player_interpreter"]

    def incomplete(*args, **kw):
        original(*args, **kw)
        args[3].unlink()

    with pytest.raises(ValueError, match="pose cache"):
        run_pipeline(**dict(kwargs, player_interpreter=incomplete))
    assert not (kwargs["output_dir"] / "players" / "players.manifest.json").exists()
    assert not (kwargs["output_dir"] / "players" / "assignments.jsonl").exists()
    run_pipeline(**kwargs)


def test_no_overlay_pipeline_does_not_create_video(pipeline):
    kwargs, _ = pipeline
    outputs = run_pipeline(**kwargs, overlay=False)
    assert "overlay" not in outputs
    assert not (kwargs["output_dir"] / "rallies" / "overlay.mp4").exists()


@pytest.mark.parametrize("stage", ["external_geometry", "canonical_geometry"])
def test_calibration_geometry_rejected_before_extraction(pipeline, stage):
    kwargs, calls = pipeline
    if stage == "canonical_geometry":
        run_pipeline(**kwargs)
        path = kwargs["output_dir"] / "court" / "calibration.json"
        kwargs = dict(kwargs, calibration=None)
    else:
        path = kwargs["calibration"]
    value = json.loads(path.read_text())
    value["image_size"] = [1920, 1080]
    path.write_text(json.dumps(value))
    before = dict(calls)
    with pytest.raises(ValueError, match="image_size"):
        run_pipeline(**kwargs)
    assert calls == before


def test_calibration_cancellation_is_clear_and_does_not_extract(
    pipeline, monkeypatch, capsys
):
    kwargs, calls = pipeline

    def cancel(argv):
        assert "--open-browser" in argv
        raise KeyboardInterrupt

    monkeypatch.setattr("src.calibrate_court.main", cancel)
    with pytest.raises(ValueError, match="calibration was not completed"):
        run_pipeline(**dict(kwargs, calibration=None))
    assert "calibration is missing" in capsys.readouterr().err
    assert calls == {"person": 0, "players": 0}


def test_explicit_device_failure_and_auto_fallback(pipeline, monkeypatch):
    kwargs, calls = pipeline
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device("auto") == "cpu"
    with pytest.raises(ValueError, match="requested device is unavailable"):
        run_pipeline(**dict(kwargs, device="cuda"))
    assert calls == {"person": 0, "players": 0}


@pytest.mark.parametrize("view", ["no_position", "full_audio", "full_motion", "full_audio_zscore_motion"])
def test_pipeline_uses_manifest_feature_view(pipeline, view):
    kwargs, _ = pipeline
    config = RallySegmenterConfig(
        frame_feature_dim=RALLY_FEATURE_VIEWS[view].dimension,
        token_size=8,
        num_layers=1,
        num_attention_heads=2,
        feed_forward_size=16,
        dropout=0,
    )
    research = kwargs["checkpoint"].parent.parent / "view-research.pt"
    model = RallySegmenter(config)
    checkpoint = rally_checkpoint(
        model,
        dataset_fingerprint="test",
        annotation_fingerprint="test",
        split_manifest={},
        training_recipe=RallyTrainingRecipe(),
        chosen_epoch=1,
        decoder_configuration=asdict(InPlayDecoderConfig()),
        feature_view=view,
    )
    torch.save(checkpoint, research)
    original_manifest = json.loads(kwargs["model_manifest"].read_text())
    original_root = kwargs["model_manifest"].parent
    output = original_root.parent / "view-release"
    manifest = export_release(
        research,
        output,
        person_detector=original_root
        / original_manifest["assets"]["person_detector"]["filename"],
        pose_landmarker=original_root
        / original_manifest["assets"]["pose_landmarker"]["filename"],
        code_revision="b" * 40,
        base_url=output.as_uri(),
        model_id="view-v1",
    )
    result = run_pipeline(
        **dict(
            kwargs, checkpoint=output / "rally-segmenter.pt", model_manifest=manifest
        )
    )
    assert result["predictions"].exists()


def test_cli_raw_and_registered_source_run_actual_pipeline(
    pipeline, monkeypatch, capsys
):
    from src.workflow.__main__ import main

    kwargs, calls = pipeline
    original = run_pipeline

    def with_fixtures(catalog, **arguments):
        return original(
            catalog,
            **arguments,
            person_extractor=kwargs["person_extractor"],
            player_interpreter=kwargs["player_interpreter"],
        )

    monkeypatch.setattr("src.workflow.rally_pipeline.run_pipeline", with_fixtures)
    common = [
        "--catalog",
        str(kwargs["catalog"].path),
        "rally",
        "predict",
        "--checkpoint",
        str(kwargs["checkpoint"]),
        "--model-manifest",
        str(kwargs["model_manifest"]),
        "--cache-dir",
        str(kwargs["cache_dir"]),
        "--device",
        "cpu",
        "--batch-size",
        "2",
    ]
    assert (
        main(
            [
                *common,
                "--video",
                str(kwargs["video"]),
                "--source-id",
                "generated",
                "--output-dir",
                str(kwargs["output_dir"]),
                "--calibration",
                str(kwargs["calibration"]),
            ]
        )
        == 0
    )
    assert "overlay:" in capsys.readouterr().out
    assert main([*common, "--source", "generated"]) == 0
    assert calls == {"person": 1, "players": 1}


def test_new_catalog_cannot_overwrite_another_source_identity(pipeline):
    kwargs, _ = pipeline
    run_pipeline(**kwargs)
    identity_path = kwargs["output_dir"] / "source.json"
    before = identity_path.read_bytes()
    different = dict(
        kwargs,
        catalog=SourceCatalog(kwargs["output_dir"].parent / "other-catalog.json"),
        source_id="another-source",
        force=True,
    )
    with pytest.raises(ValueError, match="different source"):
        run_pipeline(**different)
    assert identity_path.read_bytes() == before


def test_manual_and_pipeline_stages_share_completion_policy(pipeline):
    from src.rally_release import release_tracking_profile
    from src.workflow.extraction import resolve_source
    from src.workflow.rally_pipeline import _calibrate
    from src.workflow.stages import run_person_stage, run_player_stage

    kwargs, calls = pipeline
    source = resolve_source(
        kwargs["catalog"],
        source_id=kwargs["source_id"],
        video=kwargs["video"],
        output_dir=kwargs["output_dir"],
    )
    _calibrate(source, kwargs["calibration"], force=False)
    manifest = json.loads(kwargs["model_manifest"].read_text())
    assets = {
        name: kwargs["model_manifest"].parent / item["filename"]
        for name, item in manifest["assets"].items()
    }
    run_person_stage(
        source,
        model=assets["person_detector"],
        device="cpu",
        extractor=kwargs["person_extractor"],
        tracking_profile=release_tracking_profile(manifest),
    )
    run_player_stage(
        source,
        pose_model=assets["pose_landmarker"],
        interpreter=kwargs["player_interpreter"],
    )
    run_pipeline(**kwargs)
    assert calls == {"person": 1, "players": 1}


def test_players_reports_rollback_and_uses_committed_tracks(pipeline, capsys):
    from src.workflow.stages import run_player_stage

    kwargs, _ = pipeline
    run_pipeline(**kwargs, overlay=False)
    source = kwargs["catalog"].select([kwargs["source_id"]])[0]
    tracks = source.person_tracks_path
    receipt = tracks.with_name("person-tracks.manifest.json")
    committed_hash = sha256_file(tracks)
    replacement = tracks.with_name("replacement.jsonl")
    replacement.write_text(
        json.dumps({**json.loads(tracks.read_text()), "revision": 2}) + "\n"
    )
    script = """
import os, sys
from pathlib import Path
from src.workflow.publication import publish

replacement, tracks, receipt = map(Path, sys.argv[1:])
replace = Path.replace

def interrupt(path, destination):
    result = replace(path, destination)
    if destination == tracks:
        os._exit(17)
    return result

Path.replace = interrupt
publish([replacement], [tracks], receipt, {"revision": 2})
"""
    process = subprocess.run(
        [sys.executable, "-c", script, replacement, tracks, receipt], check=False
    )
    assert process.returncode == 17
    assert sha256_file(tracks) != committed_hash

    run_player_stage(
        source,
        pose_model=kwargs["output_dir"].parent / "pose.task",
        interpreter=kwargs["player_interpreter"],
        force=True,
    )
    message = capsys.readouterr().err
    assert "Rolled back an interrupted publication" in message
    assert str(receipt) in message
    assert sha256_file(tracks) == committed_hash
    assignments = json.loads(source.assignments_path.read_text().splitlines()[0])
    assert assignments["raw_artifact_fingerprint"] == f"sha256:{committed_hash}"


@pytest.mark.parametrize("problem", ["missing_receipt", "wrong_source", "wrong_hash"])
def test_players_refuses_unverified_person_tracks(pipeline, problem):
    from src.workflow.stages import run_player_stage

    kwargs, calls = pipeline
    run_pipeline(**kwargs, overlay=False)
    source = kwargs["catalog"].select([kwargs["source_id"]])[0]
    tracks = source.person_tracks_path
    receipt = tracks.with_name("person-tracks.manifest.json")
    if problem == "missing_receipt":
        receipt.unlink()
    elif problem == "wrong_source":
        value = json.loads(receipt.read_text())
        value["inputs"]["identity"] = "different-source"
        receipt.write_text(json.dumps(value))
    else:
        tracks.write_text(tracks.read_text() + "\n")

    with pytest.raises(ValueError, match="committed (person tracks|receipt)"):
        run_player_stage(source, interpreter=kwargs["player_interpreter"], force=True)
    assert calls["players"] == 1


def test_inference_edit_and_pose_backend_rebuild_preserve_person_cache(
    pipeline, monkeypatch
):
    import src.workflow.stages as stages

    kwargs, calls = pipeline
    run_pipeline(**kwargs)
    original_identity = stages.implementation_identity

    def changed(stage):
        return {
            **original_identity(stage),
            **({"revision": "new-inference"} if stage == "predict" else {}),
        }

    monkeypatch.setattr(stages, "implementation_identity", changed)
    monkeypatch.setattr("src.workflow.rally_pipeline.implementation_identity", changed)
    run_pipeline(**kwargs, rebuild=["predict"])
    assert calls == {"person": 1, "players": 1}
    run_pipeline(
        **kwargs, pose_backend_identity="another-pose-backend", rebuild=["pose"]
    )
    assert calls == {"person": 1, "players": 2}


def test_render_failure_keeps_predictions_and_retries_without_inference(
    pipeline, monkeypatch
):
    import src.temporal_selector.rally_inference as inference

    kwargs, calls = pipeline
    original = inference.render_overlay

    def fail(*args):
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(inference, "render_overlay", fail)
    with pytest.raises(RuntimeError, match="encoder failed"):
        run_pipeline(**kwargs)
    root = kwargs["output_dir"]
    assert (root / "rallies" / "run.manifest.json").exists()
    status = json.loads((root / "run-status.json").read_text())
    assert status["stages"]["render"]["state"] == "failed"
    monkeypatch.setattr(inference, "render_overlay", original)
    monkeypatch.setattr(
        inference,
        "predict_source",
        lambda *args, **kw: pytest.fail("inference repeated"),
    )
    run_pipeline(**kwargs)
    assert calls == {"person": 1, "players": 1}


def test_decoder_rebuild_uses_saved_probabilities(pipeline, monkeypatch):
    import src.temporal_selector.rally_inference as inference

    kwargs, calls = pipeline
    run_pipeline(**kwargs)
    monkeypatch.setattr(
        inference,
        "predict_source",
        lambda *args, **kw: pytest.fail("inference repeated"),
    )
    result = run_pipeline(**kwargs, rebuild=["decode"])
    assert result["overlay"].is_file()
    assert calls == {"person": 1, "players": 1}


def test_shared_observation_runner_reuses_both_branches_and_optional_outputs(
    pipeline, monkeypatch
):
    from src.single_video.shuttle import ShuttleCandidateCollector
    from src.workflow.extraction import run_extraction
    import src.single_video.artifacts as artifacts

    kwargs, calls = pipeline
    root = kwargs["output_dir"].parent
    models = root / "bundle"
    for name in (
        "models/TrackNet_torchscript.pt",
        "src/TrackNetV3/ckpts/TrackNet_best.pt",
    ):
        path = models / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tracknet")
    manifest = json.loads(kwargs["model_manifest"].read_text())
    assets = {
        name: kwargs["model_manifest"].parent / item["filename"]
        for name, item in manifest["assets"].items()
    }
    shuttle_calls = []

    def shuttle(video, tracks, candidates):
        shuttle_calls.append(video)
        identity = inspect_source("generated", video)
        collector = ShuttleCandidateCollector(
            fps=6,
            image_size=(128, 96),
            heatmap_size=(2, 2),
            checkpoint_sha256=sha256_file(
                models / "src/TrackNetV3/ckpts/TrackNet_best.pt"
            ),
            inference_model_sha256=sha256_file(
                models / "models/TrackNet_torchscript.pt"
            ),
            tracknet_config={"sequence_length": 3},
            source_frame_range=(0, 8),
        )
        for frame in range(9):
            collector.add(frame, np.zeros((2, 2), dtype=np.float32))
        collector.write(candidates)
        lines = candidates.read_text().splitlines()
        header = json.loads(lines[0])
        header["source_video_sha256"] = identity.video_sha256
        candidates.write_text(json.dumps(header) + "\n" + "\n".join(lines[1:]) + "\n")
        tracks.write_text(
            "Frame,X,Y,Visibility\n" + "".join(f"{frame},0,0,0\n" for frame in range(9))
        )

    arguments = dict(
        catalog=kwargs["catalog"],
        source_id="generated",
        video=kwargs["video"],
        output_dir=kwargs["output_dir"],
        calibration=kwargs["calibration"],
        person_model=assets["person_detector"],
        pose_model=assets["pose_landmarker"],
        model_root=models,
        device="cpu",
        person_extractor=kwargs["person_extractor"],
        player_interpreter=kwargs["player_interpreter"],
        shuttle_extractor=shuttle,
    )
    result = run_extraction(**arguments)
    before = result["manifest"].read_bytes()
    assert run_extraction(**arguments) == result
    assert len(shuttle_calls) == 1
    assert calls == {"person": 1, "players": 1}

    def fail(*args, **kw):
        raise RuntimeError("preview encoder failed")

    monkeypatch.setattr(artifacts, "render_player_preview", fail)
    with pytest.raises(RuntimeError, match="preview encoder failed"):
        run_extraction(**arguments, preview=True)
    assert result["manifest"].read_bytes() == before
    assert calls == {"person": 1, "players": 1}
    assert len(shuttle_calls) == 1
    zip_path = root / "portable.zip"
    run_extraction(**arguments, export_zip=zip_path)
    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        assert not any(
            ".pose-work" in path or "publication" in path for path in archive.namelist()
        )
        assert any(path.endswith("source.json") for path in archive.namelist())


def test_explicit_person_import_is_reused_without_loading_detector(pipeline):
    from src.workflow.artifacts import import_artifact
    from src.workflow.stages import run_person_stage

    kwargs, calls = pipeline
    run_pipeline(**kwargs)
    source = kwargs["catalog"].select(["generated"])[0]
    external = kwargs["output_dir"].parent / "imported-person-tracks.jsonl"
    external.write_bytes(source.person_tracks_path.read_bytes())
    import_artifact(source, "person-tracks", external, force=True)
    # An explicitly imported source layer does not claim to be generated by a
    # currently configured detector and needs no model download to reuse it.
    assert not run_person_stage(source, model="nonexistent-detector.pt", device="cpu")
    run_pipeline(**kwargs)
    assert calls == {"person": 1, "players": 1}


def test_tracknet_adapter_passes_verified_models_and_local_frame_domain(
    pipeline, monkeypatch
):
    import sys
    import types
    import src.single_video.models as model_loader
    import src.workflow.extraction as extraction
    from src.single_video.shuttle import (
        ShuttleCandidateCollector,
        read_shuttle_candidates,
    )

    kwargs, _ = pipeline
    kwargs["video"] = kwargs["video"].rename(kwargs["video"].with_suffix(".webm"))
    source = extraction.resolve_source(
        kwargs["catalog"],
        source_id="generated",
        video=kwargs["video"],
        output_dir=kwargs["output_dir"],
    )
    staging = kwargs["output_dir"].parent / "staged-shuttle"
    staging.mkdir()
    models = {
        "tracknet": object(),
        "tracknet_checkpoint": "verified-checkpoint",
        "tracknet_model": "verified-script",
        "inpaintnet_enabled": False,
    }
    backend = types.ModuleType("TrackNetV3.predictArgs")
    seen = []

    def predict(**options):
        assert options["reuse_models"] is models
        assert options["num_workers"] == 0
        assert options["video_file"] == str(source.video_path)
        assert options["large_video"] and not options["output_video"]
        seen.append(options)
        directory = Path(options["save_dir"])
        (directory / f"{source.video_path.name[:-4].lower()}_ball.csv").write_text(
            "Frame,X,Y,Visibility\n" + "".join(f"{frame},0,0,0\n" for frame in range(9))
        )
        collector = ShuttleCandidateCollector(
            fps=6,
            image_size=(128, 96),
            heatmap_size=(2, 2),
            source_frame_range=(0, 8),
            checkpoint_sha256="a" * 64,
            inference_model_sha256="b" * 64,
        )
        for frame in range(9):
            collector.add(frame, np.zeros((2, 2), dtype=np.float32))
        collector.write(options["candidate_output_path"])

    backend.run_prediction = predict

    def child_process(command, **options):
        package = types.ModuleType("TrackNetV3")
        package.predictArgs = backend
        with monkeypatch.context() as child:
            child.setitem(sys.modules, "TrackNetV3", package)
            child.setattr(sys, "argv", ["-c", *command[3:]])
            child.setattr(model_loader, "load_track_models", lambda root: models)
            exec(compile(command[2], "tracknet-adapter", "exec"), {})

    monkeypatch.setattr(extraction.subprocess, "run", child_process)
    tracks, candidates = staging / "tracks.csv", staging / "candidates.jsonl"
    extraction._tracknet_adapter(
        source,
        tracks,
        candidates,
        model_root=staging,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )
    assert len(seen) == 1
    assert backend.DEVICE == torch.device("cpu")
    header, frames = read_shuttle_candidates(candidates)
    assert header["source_id"] == "generated"
    assert header["source_video_sha256"] == sha256_file(source.video_path)
    assert header["checkpoint_sha256"] == "a" * 64
    assert header["inference_model_sha256"] == "b" * 64
    assert [frame["frame"] for frame in frames] == list(range(9))
    assert tracks.is_file()


def test_registration_preserves_parent_provenance_and_rejects_wrong_root(pipeline):
    from dataclasses import replace
    from src.workflow.extraction import resolve_source
    from src.workflow.identity import ParentVideoProvenance

    kwargs, _ = pipeline
    source = resolve_source(
        kwargs["catalog"],
        source_id="generated",
        video=kwargs["video"],
        output_dir=kwargs["output_dir"],
    )
    identity = SourceIdentity.read(source.identity_path)
    identity = replace(
        identity,
        parent_video=ParentVideoProvenance(
            video_sha256="a" * 64,
            start_time_seconds=120,
            end_time_seconds=122,
            handling="trim",
        ),
    )
    identity.write(source.identity_path)
    original = source.identity_path.read_bytes()
    assert (
        resolve_source(
            kwargs["catalog"],
            source_id="generated",
            video=kwargs["video"],
            output_dir=kwargs["output_dir"],
        )
        == source
    )
    another_catalog = SourceCatalog(
        kwargs["output_dir"].parent / "another-catalog.json"
    )
    assert (
        another_catalog.add(
            kwargs["video"], source_id="generated", output_dir=kwargs["output_dir"]
        )
        == source
    )
    assert source.identity_path.read_bytes() == original
    with pytest.raises(ValueError, match="different source identity"):
        SourceCatalog(kwargs["output_dir"].parent / "wrong-catalog.json").add(
            kwargs["video"], source_id="another-id", output_dir=kwargs["output_dir"]
        )
    assert source.identity_path.read_bytes() == original


def test_status_reports_committed_pose_work_after_interruption(pipeline, capsys):
    from src.workflow.__main__ import main

    kwargs, _ = pipeline

    def interrupt(
        video,
        person,
        calibration,
        poses,
        assignments,
        features,
        *,
        pose_model_asset,
        pose_crop_padding,
        pose_work_dir,
        pose_backend_identity,
    ):
        unit = pose_work_dir / "raw" / "backend" / "preprocessing" / "observation.json"
        unit.parent.mkdir(parents=True)
        unit.write_text(
            json.dumps(
                {
                    "frame": 3,
                    "track_id": 7,
                    "raw_artifact_fingerprint": "raw",
                    "pose_model_fingerprint": "model",
                    "preprocessing_fingerprint": "preprocessing",
                }
            )
        )
        # Abandoned temporary files are never counted as committed observations.
        (unit.parent / ".interrupted.tmp").write_text("partial")
        raise RuntimeError("interrupted after commit")

    with pytest.raises(RuntimeError, match="after commit"):
        run_pipeline(**{**kwargs, "player_interpreter": interrupt})
    capsys.readouterr()
    assert (
        main(
            [
                "--catalog",
                str(kwargs["catalog"].path),
                "source",
                "status",
                "--source",
                "generated",
                "--json",
            ]
        )
        == 0
    )
    stage = json.loads(capsys.readouterr().out)[0]["run"]["stages"]["pose"]
    assert stage["state"] == "failed"
    assert stage["committed_work"]["retained_unit_count"] == 1
    assert stage["committed_work"]["last_committed_unit"]["frame"] == 3
    assert stage["committed_work"]["last_committed_unit"]["track_id"] == 7


def test_retracking_resume_verifies_inputs_and_committed_outputs(pipeline, monkeypatch):
    from InPlay.heuristic import person_tracks, player_interpretation
    from src.workflow.source import calibrate_source, resolve_source
    from tools import retrack_rally_experiment

    kwargs, calls = pipeline
    original = resolve_source(
        kwargs["catalog"], source_id=kwargs["source_id"],
        video=kwargs["video"], output_dir=kwargs["output_dir"],
    )
    calibrate_source(original, kwargs["calibration"], force=False)
    config = original.artifact_root.parent / "experiment-input.json"
    config.write_text(json.dumps({"dataset": {
        "rally_intervals_path": "labels.json",
        "rally_manifest_path": "labels.manifest.json",
        "sources": [{
            "source_id": original.source_id,
            "video_path": str(original.video_path),
            "calibration_path": str(original.calibration_path),
        }],
    }}))
    release = kwargs["model_manifest"].parent
    detector = release / "person-detector.pt"
    pose_model = release / "pose-landmarker.task"
    output = original.artifact_root.parent / "retracked"
    command = [
        "retrack", "--config", str(config), "--output-dir", str(output),
        "--model", str(detector), "--pose-model", str(pose_model), "--device", "cpu",
    ]

    def extract(video, path, model, **_options):
        kwargs["person_extractor"](video, path, model)

    def interrupt(*args, **options):
        kwargs["player_interpreter"](*args, **options)
        args[-1].write_text("Frame\n0\n")
        raise RuntimeError("interrupted writing features")

    monkeypatch.setattr(person_tracks, "extract_person_tracks", extract)
    monkeypatch.setattr(player_interpretation, "interpret_players", interrupt)
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(RuntimeError, match="interrupted writing features"):
        retrack_rally_experiment.main()
    assert not (output / "experiment.json").exists()

    monkeypatch.setattr(player_interpretation, "interpret_players", kwargs["player_interpreter"])
    monkeypatch.setattr(sys, "argv", [*command, "--resume"])
    assert retrack_rally_experiment.main() == 0
    assert calls == {"person": 1, "players": 2}
    assert retrack_rally_experiment.main() == 0
    assert calls == {"person": 1, "players": 2}

    player_root = output / "sources" / original.source_id / "players"
    experiment = json.loads((output / "experiment.json").read_text())
    assert experiment["dataset"]["sources"][0]["assignments_path"] == str(
        player_root / "assignments.jsonl"
    )
    for model in (detector, pose_model):
        original_bytes = model.read_bytes()
        model.write_bytes(b"changed model")
        with pytest.raises(ValueError, match="stale or incomplete artifacts"):
            retrack_rally_experiment.main()
        model.write_bytes(original_bytes)
    (player_root / "features.csv").write_text("Frame\n0\n")
    with pytest.raises(ValueError, match="stale or incomplete artifacts"):
        retrack_rally_experiment.main()
