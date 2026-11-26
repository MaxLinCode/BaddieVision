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
from src.temporal_selector.rally_dataset import RallyWindow
from src.temporal_selector import rally_experiment as training
from src.temporal_selector.rally_features import (
    CANONICAL_RALLY_FEATURE_VIEW,
    RALLY_FEATURE_VIEWS,
)


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


def _windows():
    return [
        RallyWindow(
            source_id="camera",
            window_id=f"window-{i}",
            anchor_frame=1,
            frame_indices=(0, 1, 2),
            owned_frames=frozenset((0, 1, 2)),
            relative_time_seconds=torch.tensor([-1.0, 0.0, 1.0]),
            frame_values=torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]) + i,
            frame_validity=torch.ones((3, 2), dtype=torch.bool),
            inplay_targets=torch.tensor([0, 1, 0]),
            rally_start_targets=torch.tensor([0.0, 1.0, 0.0]),
            rally_end_targets=torch.tensor([0.0, 1.0, 0.0]),
        )
        for i in range(2)
    ]


def _model():
    return RallySegmenter(
        RallySegmenterConfig(
            frame_feature_dim=2,
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


def test_ema_update_and_validation_weight_swap():
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(2.0)
    ema = training._ModelEMA(model, 0.9)
    with torch.no_grad():
        model.weight.fill_(4.0)
        model.bias.fill_(4.0)
    ema.update(model)
    assert ema.updates == 1
    # The first update keeps only 2/11 of the previous average so the EMA
    # follows the newly trained weights closely during warm-up.
    expected = 2.0 * (2.0 / 11.0) + 4.0 * (9.0 / 11.0)
    torch.testing.assert_close(
        ema.shadow["weight"], torch.full((2, 2), expected)
    )
    live = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    with training._ema_model_weights(model, ema):
        for name, value in ema.shadow.items():
            assert torch.equal(model.state_dict()[name], value), name
    for name, value in live.items():
        assert torch.equal(model.state_dict()[name], value), name


def test_validation_runs_on_ema_weights_and_exports_the_selected_average(monkeypatch):
    windows = _windows()
    model = _model()
    raw_states = []
    ema_states = []
    original_update = training._ModelEMA.update

    def recording_update(ema, target):
        original_update(ema, target)
        raw_states.append(
            {
                name: value.detach().clone()
                for name, value in target.state_dict().items()
            }
        )
        ema_states.append(ema.state_dict())

    seen = []

    def fake_predict(target, windows, **kwargs):
        seen.append(
            {
                name: value.detach().clone()
                for name, value in target.state_dict().items()
            }
        )
        return [
            {
                "source_id": "camera",
                "frame": 0,
                "inplay_target": 0,
                "inplay_probability": 0.1,
            }
        ]

    monkeypatch.setattr(training._ModelEMA, "update", recording_update)
    monkeypatch.setattr(training, "predict_rally_windows", fake_predict)
    monkeypatch.setattr(training, "rally_metrics", lambda rows: {"raw_bce": 0.5})
    history, epoch = training.train_rally_segmenter(
        model,
        windows,
        windows[:1],
        device=torch.device("cpu"),
        epochs=2,
        batch_size=1,
        seed=3,
        learning_rate=0.01,
    )
    assert len(seen) == len(history) == 2
    assert len(ema_states) == len(raw_states) == history[-1]["ema_updates"]
    selected_update = history[0]["ema_updates"] - 1
    for snapshot, entry in zip(seen, history, strict=True):
        ema_snapshot = ema_states[entry["ema_updates"] - 1]
        raw_snapshot = raw_states[entry["ema_updates"] - 1]
        for name, value in snapshot.items():
            assert torch.equal(value, ema_snapshot[name]), name
        assert any(
            not torch.equal(value, raw_snapshot[name])
            for name, value in snapshot.items()
            if value.is_floating_point()
        )
    assert [entry["validation_weights"] for entry in history] == ["ema", "ema"]
    assert [entry["ema_updates"] for entry in history] == [2, 4]
    # Constant validation BCE keeps the first epoch as the selected state.
    assert epoch == 1
    for name, value in model.state_dict().items():
        assert torch.equal(value, ema_states[selected_update][name]), name


def test_pooled_validation_selection_pools_then_smooths():
    fold_a = [
        {"epoch": epoch, "validation_raw_bce": value}
        for epoch, value in enumerate((0.58, 0.66, 0.48, 0.66, 0.58), start=1)
    ]
    fold_b = [
        {"epoch": epoch, "validation_raw_bce": value}
        for epoch, value in enumerate((0.62, 0.64, 0.52, 0.64, 0.62), start=1)
    ]
    selection = training.select_production_epoch(
        [fold_a, fold_b], fold_ids=["A", "B"]
    )
    assert selection.pooled_validation_raw_bce == pytest.approx(
        (0.60, 0.65, 0.50, 0.65, 0.60)
    )
    assert selection.smoothed_validation_raw_bce == pytest.approx(
        (0.625, 1.75 / 3.0, 0.60, 1.75 / 3.0, 0.625)
    )
    # The raw pooled minimum is epoch 3, but smoothing removes that spike.
    assert selection.epoch == 2
    assert selection.fold_validation_raw_bce["A"] == pytest.approx(
        (0.58, 0.66, 0.48, 0.66, 0.58)
    )
    metadata = selection.metadata()
    assert metadata["epoch_selection"] == training.POOLED_PRODUCTION_EPOCH_SELECTION
    assert metadata["selected_epoch"] == 2
    assert metadata["moving_average_window_epochs"] == 3
    assert metadata["pooled_validation_raw_bce"] == pytest.approx(
        [0.60, 0.65, 0.50, 0.65, 0.60]
    )


def test_pooled_validation_selection_rejects_incomplete_or_mismatched_curves():
    with pytest.raises(ValueError, match="every epoch"):
        training.select_production_epoch(
            [[{"epoch": 1, "validation_raw_bce": 0.5}, {"epoch": 2}]]
        )
    with pytest.raises(ValueError, match="same number"):
        training.select_production_epoch(
            [
                [{"epoch": 1, "validation_raw_bce": 0.5}],
                [
                    {"epoch": 1, "validation_raw_bce": 0.4},
                    {"epoch": 2, "validation_raw_bce": 0.3},
                ],
            ]
        )
    with pytest.raises(ValueError, match="unique"):
        training.select_production_epoch(
            [
                [{"epoch": 1, "validation_raw_bce": 0.5}],
                [{"epoch": 1, "validation_raw_bce": 0.4}],
            ],
            fold_ids=["A", "A"],
        )


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
    with pytest.raises(ValueError, match="ema_decay"):
        RallyTrainingRecipe(ema_decay=1.0)


@pytest.mark.parametrize("validation", [True, False])
def test_epoch_resume_matches_uninterrupted_training(tmp_path, monkeypatch, validation):
    windows = _windows()
    options = dict(
        device=torch.device("cpu"),
        epochs=3,
        batch_size=1,
        seed=17,
        learning_rate=0.002,
        gradient_clip_norm=0.3,
        recovery_identity={"dataset": "v1", "split": "camera-held-out"},
    )
    _seed()
    uninterrupted = _model()
    expected = training.train_rally_segmenter(
        uninterrupted, windows, windows[:1] if validation else None, **options
    )
    _seed()
    interrupted = _model()
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
    assert saved["version"] == 2
    assert saved["ema"]["updates"] == 2
    assert saved["optimizer_state_dict"]["param_groups"][0]["lr"] == 0.002
    monkeypatch.setattr(training, "_atomic_torch_save", save)
    # A fresh process/model can consume arbitrary RNG before restoring a checkpoint.
    resumed = _model()
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
    completed = torch.load(path, weights_only=True)
    assert completed["completed_epoch"] == 3
    assert completed["ema"]["updates"] == 6


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
    config_stub = SimpleNamespace(sources=[SimpleNamespace(source_id=i) for i in ("a", "b")])
    monkeypatch.setattr(
        training, "rally_data_config_from_selector_mapping", lambda *a, **k: config_stub
    )
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
                "--ema-decay",
                "0.9",
                "--resume",
            ]
        )
        == 0
    )
    assert called["recipe"].epochs == 1
    assert called["recipe"].batch_size == 3
    assert called["recipe"].learning_rate == 0.001
    assert called["recipe"].ema_decay == 0.9
    assert called["device"] == torch.device("cpu")
    assert called["resume"]


@pytest.mark.parametrize(
    ("dataset_overrides", "expected_seconds"), [({}, 2.0), ({"rally_context_seconds": 3.0}, 3.0)]
)
@pytest.mark.parametrize(
    ("cli_view", "expected_view"),
    [(None, CANONICAL_RALLY_FEATURE_VIEW), ("full_audio_zscore", "full_audio_zscore")],
)
def test_rtmpose_preset_defaults_to_canonical_motion_and_allows_audio_baseline(
    tmp_path, monkeypatch, dataset_overrides, expected_seconds, cli_view, expected_view,
):
    import json

    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "dataset": {
            "sources": [{
                "source_id": "a", "video_path": "video.mp4", "assignments_path": "assignments.jsonl",
                "pose_cache_path": "poses.jsonl", "calibration_path": "calibration.json",
            }],
            "rally_intervals_path": "rallies.csv", "rally_manifest_path": "rallies.json",
            **dataset_overrides,
        },
        "camera_sources": {"A": ["a"]},
    }))
    called = {}
    monkeypatch.setattr(training, "run_clean_rally_experiment", lambda config, *a, **k: called.update(
        context=config.context_seconds,
        feature_view=config.feature_view,
        smoothing_weight=k["recipe"].temporal_smoothing_weight,
        require_rtmpose=k["require_rtmpose"],
    ))
    arguments = [
        "--config", str(config), "--output-dir", str(tmp_path / "output"),
        "--clean-only", "--preset", "rtmpose-audio", "--device", "cpu",
    ]
    if cli_view is not None:
        arguments.extend(("--feature-view", cli_view))
    assert training.main(arguments) == 0
    assert called == {
        "context": expected_seconds, "feature_view": expected_view, "smoothing_weight": 0.1,
        "require_rtmpose": True,
    }


def test_example_experiment_uses_canonical_motion_view():
    import json
    from pathlib import Path

    config = json.loads(
        (Path(__file__).parents[1] / "config/rally-experiment.example.json").read_text()
    )
    assert config["dataset"]["rally_feature_view"] == CANONICAL_RALLY_FEATURE_VIEW
    canonical = RALLY_FEATURE_VIEWS[CANONICAL_RALLY_FEATURE_VIEW]
    assert canonical.uses_audio and canonical.audio_zscore and canonical.uses_motion


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
