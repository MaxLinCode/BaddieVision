"""End-to-end pipeline tests without private media or pretrained model downloads."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.rally_release import export_release
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


def test_complete_pipeline_and_verified_cached_rerun(pipeline, monkeypatch):
    kwargs, calls = pipeline
    outputs = run_pipeline(**kwargs)
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
    assert not list((root / "players").glob(".person-*"))
    assert not list((root / "players").glob(".players-*"))
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


def test_pipeline_uses_manifest_feature_view(pipeline):
    kwargs, _ = pipeline
    view = "no_position"
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
