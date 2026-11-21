from __future__ import annotations

import copy
import random
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from src.temporal_selector.rally import (
    RallySegmenter,
    RallySegmenterConfig,
    RallyTrainingRecipe,
)
from src.temporal_selector.rally_dataset import RallyWindow, collate_rally_windows
from src.temporal_selector.rally_features import FULL_RALLY_FEATURE_NAMES, rally_feature_view
from src.temporal_selector import rally_experiment as training


@pytest.fixture(autouse=True)
def _isolate_training_randomness():
    """Keep CPU recovery checks from changing later tests' RNG/thread settings."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    thread_count = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_num_threads(1)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_num_threads(thread_count)


def _windows(dim=2):
    return [
        RallyWindow(
            source_id="camera",
            window_id=f"window-{i}",
            anchor_frame=1,
            frame_indices=(0, 1, 2),
            owned_frames=frozenset((0, 1, 2)),
            relative_time_seconds=torch.tensor([-1.0, 0.0, 1.0]),
            frame_values=torch.linspace(0, 1, 3 * dim).reshape(3, dim) + i,
            frame_validity=torch.ones((3, dim), dtype=torch.bool),
            inplay_targets=torch.tensor([0, 1, 0]),
            rally_start_targets=torch.tensor([0.0, 1.0, 0.0]),
            rally_end_targets=torch.tensor([0.0, 1.0, 0.0]),
        )
        for i in range(2)
    ]


def _model(dim=2):
    return RallySegmenter(
        RallySegmenterConfig(
            frame_feature_dim=dim,
            token_size=8,
            num_layers=1,
            num_attention_heads=2,
            feed_forward_size=16,
            dropout=0.2,
        )
    )


def _seed():
    random.seed(17)
    np.random.seed(17)
    torch.random.default_generator.manual_seed(17)


def test_cpu_configuration_and_historical_preset():
    recipe = RallyTrainingRecipe.from_preset(
        "default",
        device="cpu",
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        gradient_clip_norm=0.5,
        seed=8,
    )
    assert recipe.device == "cpu" and recipe.epochs == 1
    assert asdict(RallyTrainingRecipe.from_preset("benchmark")) == asdict(
        RallyTrainingRecipe()
    )
    with pytest.raises(ValueError, match="exact historical"):
        RallyTrainingRecipe.from_preset("benchmark", device="cpu")
    with pytest.raises(ValueError, match="learning_rate"):
        RallyTrainingRecipe(learning_rate=float("nan"))


@pytest.mark.parametrize("validation", [True, False])
@pytest.mark.parametrize("audio_dropout", [0, 0.4])
def test_epoch_resume_matches_uninterrupted_training(tmp_path, monkeypatch, validation, audio_dropout):
    dim = rally_feature_view("full_audio_zscore").dimension if audio_dropout else 2
    windows = _windows(dim)
    options = dict(
        device=torch.device("cpu"),
        epochs=3,
        batch_size=1,
        seed=17,
        learning_rate=0.002,
        gradient_clip_norm=0.3,
        audio_dropout=audio_dropout,
        recovery_identity={"dataset": "v1", "split": "camera-held-out"},
    )
    _seed()
    uninterrupted = _model(dim)
    expected = training.train_rally_segmenter(
        uninterrupted, windows, windows[:1] if validation else None, **options
    )
    _seed()
    interrupted = _model(dim)
    path = tmp_path / "recovery.pt"
    save = training._atomic_torch_save

    def stop_after_first_epoch(payload, destination):
        save(payload, destination)
        if payload["completed_epoch"] == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(training, "_atomic_torch_save", stop_after_first_epoch)
    with pytest.raises(RuntimeError, match="simulated"):
        training.train_rally_segmenter(
            interrupted,
            windows,
            windows[:1] if validation else None,
            recovery_path=path,
            **options,
        )
    saved = torch.load(path, weights_only=True)
    assert saved["completed_epoch"] == saved["sampler_epoch"] == 1
    assert saved["optimizer_state_dict"]["param_groups"][0]["lr"] == 0.002
    if not audio_dropout:
        del saved["identity"]["recipe"]["audio_dropout"]
        save(saved, path)
    monkeypatch.setattr(training, "_atomic_torch_save", save)
    # A fresh process/model can consume arbitrary RNG before restoring a checkpoint.
    resumed = _model(dim)
    torch.rand(13)
    actual = training.train_rally_segmenter(
        resumed,
        windows,
        windows[:1] if validation else None,
        recovery_path=path,
        resume=True,
        **options,
    )
    assert actual == expected
    for name, expected_value in uninterrupted.state_dict().items():
        assert torch.equal(resumed.state_dict()[name], expected_value), name
    assert torch.load(path, weights_only=True)["completed_epoch"] == 3


def test_resume_rejects_data_split_or_configuration_change(tmp_path):
    path = tmp_path / "recovery.pt"
    windows = _windows()
    options = dict(
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        seed=3,
        recovery_path=path,
        recovery_identity={"split": "A"},
    )
    training.train_rally_segmenter(_model(), windows, windows[:1], **options)
    with pytest.raises(FileExistsError, match="use resume"):
        training.train_rally_segmenter(_model(), windows, windows[:1], **options)
    with pytest.raises(ValueError, match="identity has changed"):
        training.train_rally_segmenter(
            _model(),
            windows,
            windows[:1],
            resume=True,
            **{**options, "recovery_identity": {"split": "B"}},
        )
    changed = copy.deepcopy(windows)
    changed[0].frame_values[0, 0] += 1
    with pytest.raises(ValueError, match="identity has changed"):
        training.train_rally_segmenter(
            _model(), changed, windows[:1], resume=True, **options
        )
    with pytest.raises(ValueError, match="identity has changed"):
        training.train_rally_segmenter(
            _model(),
            windows,
            windows[:1],
            resume=True,
            **{**options, "learning_rate": 0.01},
        )

    changed_labels = copy.deepcopy(windows)
    changed_labels[0].inplay_targets[0] = 1
    with pytest.raises(ValueError, match="identity has changed"):
        training.train_rally_segmenter(
            _model(), changed_labels, windows[:1], resume=True, **options
        )
    changed_sampling = [replace(windows[0], metadata={"owns_short_rally": True}), windows[1]]
    with pytest.raises(ValueError, match="identity has changed"):
        training.train_rally_segmenter(
            _model(), changed_sampling, windows[:1], resume=True, **options
        )
    changed_model = RallySegmenter(replace(_model().config, dropout=0.1))
    with pytest.raises(ValueError, match="identity has changed"):
        training.train_rally_segmenter(
            changed_model, windows, windows[:1], resume=True, **options
        )


def test_cli_resolves_config_and_command_line_overrides(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "training": {"device": "cpu", "epochs": 2, "batch_size": 3},
                "camera_sources": {"A": ["a"], "B": ["b"]},
            }
        )
    )
    dataset = SimpleNamespace(
        config=SimpleNamespace(
            sources=[SimpleNamespace(source_id=i) for i in ("a", "b")]
        )
    )
    monkeypatch.setattr(
        training, "rally_data_config_from_selector_mapping", lambda *a, **k: None
    )
    monkeypatch.setattr(training, "RallyWindowDataset", lambda _: dataset)
    called = {}
    monkeypatch.setattr(
        training, "run_clean_rally_experiment", lambda *a, **k: called.update(k)
    )
    assert (
        training.main(
            [
                "--config",
                str(config),
                "--output-dir",
                str(tmp_path / "output"),
                "--clean-only",
                "--epochs",
                "1",
                "--learning-rate",
                "0.001",
                "--resume",
            ]
        )
        == 0
    )
    assert called["recipe"].epochs == 1
    assert called["recipe"].batch_size == 3
    assert called["recipe"].learning_rate == 0.001
    assert called["device"] == torch.device("cpu")
    assert called["resume"]


def test_run_records_config_and_rejects_changed_resume_identity(tmp_path):
    import json
    from types import SimpleNamespace

    dataset = SimpleNamespace(
        manifest={"dataset_fingerprint": "data-v1"}, annotation_fingerprint="labels-v1"
    )
    recipe = RallyTrainingRecipe(device="cpu", epochs=1)
    output = tmp_path / "run"
    training._prepare_rally_run(output, dataset, recipe, {"A": ["a"]}, resume=False)
    assert json.loads((output / "resolved-config.json").read_text())[
        "training_recipe"
    ] == asdict(recipe)
    training._prepare_rally_run(output, dataset, recipe, {"A": ["a"]}, resume=True)
    dataset.annotation_fingerprint = "labels-v2"
    with pytest.raises(ValueError, match="identity has changed"):
        training._prepare_rally_run(output, dataset, recipe, {"A": ["a"]}, resume=True)


def test_camera_key_order_preserves_fold_paths_on_resume(tmp_path):
    from types import SimpleNamespace

    dataset = SimpleNamespace(
        manifest={"dataset_fingerprint": "data-v1"}, annotation_fingerprint="labels-v1"
    )
    recipe = RallyTrainingRecipe(device="cpu", epochs=1)
    groups = {"South": ["s1"], "North": ["n1", "n2"]}
    output, ordered = training._prepare_rally_run(
        tmp_path / "run", dataset, recipe, groups, resume=False
    )
    folds = training.camera_held_out_folds(ordered)
    assert [(fold.fold_id, fold.held_out_camera) for fold in folds] == [
        ("A", "North"), ("B", "South")
    ]

    _, reordered = training._prepare_rally_run(
        output, dataset, recipe, dict(reversed(list(groups.items()))), resume=True
    )
    assert training.camera_held_out_folds(reordered) == folds

    for changed in (
        {"South": ["s1"], "North": ["n2", "n1"]},
        {"South": ["s1", "n2"], "North": ["n1"]},
        {**groups, "East": ["e1"]},
    ):
        with pytest.raises(ValueError, match="identity has changed"):
            training._prepare_rally_run(output, dataset, recipe, changed, resume=True)


def test_audio_dropout_masks_values_and_validity_for_entire_training_window():
    dim = rally_feature_view("full_audio_zscore").dimension
    windows = _windows(dim) * 3
    batch = collate_rally_windows(windows)
    before = copy.deepcopy(batch)
    generator = torch.Generator().manual_seed(0)
    training._drop_rally_audio(batch, 0.4, generator)
    audio_start = len(FULL_RALLY_FEATURE_NAMES)
    dropped = [2, 3, 4]
    kept = [0, 1, 5]
    assert not batch.frame_validity[dropped, :, audio_start:].any()
    assert not batch.frame_values[dropped, :, audio_start:].any()
    torch.testing.assert_close(batch.frame_values[kept], before.frame_values[kept])
    torch.testing.assert_close(batch.frame_validity[kept], before.frame_validity[kept])
    torch.testing.assert_close(batch.frame_values[..., :audio_start], before.frame_values[..., :audio_start])
    torch.testing.assert_close(batch.frame_validity[..., :audio_start], before.frame_validity[..., :audio_start])
    torch.testing.assert_close(batch.inplay_targets, before.inplay_targets)
    torch.testing.assert_close(collate_rally_windows(windows).frame_values, before.frame_values)
