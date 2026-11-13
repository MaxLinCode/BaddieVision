from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from src.rally_release import (
    CONTRACT_FIELDS,
    export_release,
    fetch_asset,
    load_release,
    resolve_release,
    sha256_file,
    validate_manifest,
)
from src.temporal_selector.joint_inference import InPlayDecoderConfig
from src.temporal_selector.rally import (
    RallyBatch,
    RallySegmenter,
    RallySegmenterConfig,
    RallyTrainingRecipe,
    rally_checkpoint,
)


@pytest.fixture
def exported(tmp_path):
    torch.manual_seed(17)
    model = RallySegmenter(
        RallySegmenterConfig(
            token_size=8,
            num_layers=1,
            num_attention_heads=2,
            feed_forward_size=16,
            dropout=0,
        )
    ).eval()
    checkpoint = rally_checkpoint(
        model,
        dataset_fingerprint="private-dataset-id",
        annotation_fingerprint="private-labels-id",
        split_manifest={"private-recording": ["/private/video.mp4"]},
        training_recipe=RallyTrainingRecipe(),
        chosen_epoch=3,
        decoder_configuration=asdict(InPlayDecoderConfig()),
        extra={"training_rows": [{"source_id": "private-person", "target": 1}]},
    )
    research = tmp_path / "research.pt"
    torch.save(checkpoint, research)
    detector, pose = tmp_path / "detector.pt", tmp_path / "pose.task"
    detector.write_bytes(b"detector-weights")
    pose.write_bytes(b"pose-weights")
    output = tmp_path / "export"
    manifest = export_release(
        research,
        output,
        person_detector=detector,
        pose_landmarker=pose,
        code_revision="a" * 40,
        base_url=output.as_uri(),
    )
    return model, manifest, json.loads(manifest.read_text())


def test_export_retains_predictions_without_private_metadata(exported):
    original, manifest_path, manifest = exported
    checkpoint = manifest_path.parent / "rally-segmenter.pt"
    restored, value = load_release(checkpoint, manifest_path)
    batch = RallyBatch(
        frame_values=torch.randn(1, 3, 154),
        frame_validity=torch.ones((1, 3, 154), dtype=torch.bool),
        relative_time_seconds=torch.tensor([[-0.1, 0.0, 0.1]]),
        frame_mask=torch.ones((1, 3), dtype=torch.bool),
        inplay_targets=torch.full((1, 3), -100, dtype=torch.long),
    )
    with torch.inference_mode():
        torch.testing.assert_close(
            original(batch).inplay_logits, restored(batch).inplay_logits
        )
    payload = torch.load(checkpoint, weights_only=True)
    assert set(payload) == {*CONTRACT_FIELDS, "model_state_dict"}
    public_text = json.dumps(
        {key: val for key, val in payload.items() if key != "model_state_dict"}
    )
    assert "private" not in public_text
    assert "target" not in public_text
    assert value == manifest
    assert manifest["assets"]["checkpoint"]["sha256"] == sha256_file(checkpoint)


def test_local_resolution_download_and_offline_reuse(exported, tmp_path, monkeypatch):
    _, manifest_path, _ = exported
    kwargs = {
        "checkpoint": manifest_path.parent / "rally-segmenter.pt",
        "manifest": manifest_path,
        "cache_dir": tmp_path / "cache",
    }
    _, _, paths = resolve_release(**kwargs)
    assert paths["person_detector"].read_bytes() == b"detector-weights"
    monkeypatch.setattr(
        "src.rally_release.urlopen", lambda *a, **k: pytest.fail("network accessed")
    )
    _, _, cached = resolve_release(**kwargs)
    assert paths == cached


def test_bad_download_is_atomic(exported, tmp_path):
    _, _, manifest = exported
    asset = dict(manifest["assets"]["person_detector"], sha256="0" * 64)
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_asset(asset, cache)
    assert not list(cache.iterdir())


def test_corrupt_cached_asset_rejected(exported, tmp_path):
    _, _, manifest = exported
    asset = manifest["assets"]["person_detector"]
    cache = tmp_path / "cache"
    path = fetch_asset(asset, cache)
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_asset(asset, cache)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(rally_feature_version=999),
        lambda m: m.update(private_training_labels=[1, 0]),
        lambda m: m.update(context_direction="causal"),
        lambda m: m["rally_feature_names"].reverse(),
        lambda m: m["preprocessing"].update(context_seconds=float("nan")),
        lambda m: m["decoder_configuration"].update(max_gap_seconds=float("nan")),
        lambda m: m["assets"]["checkpoint"].update(filename="../bad.pt"),
        lambda m: m["assets"]["person_detector"].update(
            url="http://example.com/model.pt"
        ),
    ],
)
def test_incompatible_manifest_rejected(exported, mutation):
    _, _, manifest = exported
    changed = copy.deepcopy(manifest)
    mutation(changed)
    with pytest.raises(ValueError):
        validate_manifest(changed)


def test_payload_and_manifest_must_match(exported):
    _, manifest_path, manifest = exported
    checkpoint = manifest_path.parent / "rally-segmenter.pt"
    payload = torch.load(checkpoint, weights_only=True)
    payload["decoder_configuration"]["threshold"] = 0.99
    torch.save(payload, checkpoint)
    manifest["assets"]["checkpoint"].update(
        sha256=sha256_file(checkpoint), size_bytes=checkpoint.stat().st_size
    )
    with pytest.raises(ValueError, match="contracts differ"):
        load_release(checkpoint, manifest)


def test_nonfinite_weights_rejected(exported):
    _, manifest_path, manifest = exported
    checkpoint = manifest_path.parent / "rally-segmenter.pt"
    payload = torch.load(checkpoint, weights_only=True)
    next(iter(payload["model_state_dict"].values())).fill_(float("nan"))
    torch.save(payload, checkpoint)
    manifest["assets"]["checkpoint"].update(
        sha256=sha256_file(checkpoint), size_bytes=checkpoint.stat().st_size
    )
    with pytest.raises(ValueError, match="finite tensors"):
        load_release(checkpoint, manifest)


def test_named_release_id_must_be_safe(tmp_path):
    with pytest.raises(ValueError, match="release identifier"):
        resolve_release("../../outside", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="requires both"):
        resolve_release(checkpoint=Path("anything.pt"))


def test_complete_local_export_needs_no_network_before_publication(
    exported, tmp_path, monkeypatch
):
    _, manifest_path, manifest = exported
    for asset in manifest["assets"].values():
        asset["url"] = "https://example.com/not-published/" + asset["filename"]
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "src.rally_release.urlopen", lambda *a, **kw: pytest.fail("network accessed")
    )
    _, _, assets = resolve_release(
        checkpoint=manifest_path.parent / "rally-segmenter.pt",
        manifest=manifest_path,
        cache_dir=tmp_path / "empty-cache",
    )
    assert assets["person_detector"] == manifest_path.parent / "person-detector.pt"
    assert assets["pose_landmarker"] == manifest_path.parent / "pose-landmarker.task"
