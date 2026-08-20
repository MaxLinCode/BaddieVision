from __future__ import annotations

import copy
from dataclasses import fields

import pytest
import torch

from src.temporal_selector import (
    MASKED_TARGET,
    RALLY_CONTEXT_DIRECTION,
    RALLY_FEATURE_DIM,
    RALLY_FEATURE_SCHEMA,
    RALLY_FEATURE_VERSION,
    RallyBatch,
    RallySegmenter,
    RallySegmenterConfig,
    RallyTrainingRecipe,
    RallyWindow,
    collate_rally_windows,
    rally_checkpoint,
)
from src.temporal_selector.rally_evaluation import (
    camera_held_out_folds,
    chronological_camera_train_validation_split,
    independent_model_acceptance,
    paired_bce_bootstrap,
    rally_metrics,
    stable_track_fusion_acceptance,
)


def _config() -> RallySegmenterConfig:
    return RallySegmenterConfig(
        frame_feature_dim=4,
        token_size=16,
        num_layers=1,
        num_attention_heads=2,
        feed_forward_size=32,
        dropout=0,
    )


def _batch() -> RallyBatch:
    return RallyBatch(
        frame_values=torch.randn(2, 4, 4),
        frame_validity=torch.tensor(
            [
                [[True] * 4, [False] * 4, [True] * 4, [True] * 4],
                [[True] * 4, [True] * 4, [False] * 4, [False] * 4],
            ]
        ),
        relative_time_seconds=torch.tensor(
            [[-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 0.0, 0.0]]
        ),
        frame_mask=torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        ),
        inplay_targets=torch.tensor(
            [[0, 1, 1, 0], [0, 1, MASKED_TARGET, MASKED_TARGET]]
        ),
        rally_start_targets=torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, MASKED_TARGET, MASKED_TARGET],
            ]
        ),
        rally_end_targets=torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, MASKED_TARGET, MASKED_TARGET],
            ]
        ),
    ).validate(frame_feature_dim=4)


def test_rally_batch_and_model_have_no_candidate_contract() -> None:
    assert not {
        field.name
        for field in fields(RallyBatch)
        if "candidate" in field.name or "selector" in field.name
    }
    model = RallySegmenter(_config())
    assert not any(
        "candidate" in name or "selector" in name
        for name, _ in model.named_modules()
    )
    with pytest.raises(TypeError):
        RallyBatch(candidate_values=torch.zeros(1))  # type: ignore[call-arg]


def test_padding_missing_features_and_boundary_masks_produce_finite_cpu_loss() -> None:
    batch = _batch()
    model = RallySegmenter(_config())
    output = model(batch)
    assert output.inplay_logits.shape == (2, 4)
    assert torch.isneginf(output.inplay_logits[1, 2:]).all()
    losses = model.losses(batch, output)
    assert losses.inplay_frames == 6
    assert losses.boundary_frames == 6
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert model.frame_projection.network[0].weight.grad is not None


def test_padding_targets_and_centered_time_are_validated() -> None:
    batch = _batch()
    bad_padding = copy.deepcopy(batch)
    bad_padding.inplay_targets[1, 2] = 0
    with pytest.raises(ValueError, match="padding frames"):
        bad_padding.validate(frame_feature_dim=4)
    bad_time = copy.deepcopy(batch)
    bad_time.relative_time_seconds[0] += 0.25
    with pytest.raises(ValueError, match="relative time zero"):
        bad_time.validate(frame_feature_dim=4)


def test_checkpoint_reconstruction_requires_complete_provenance() -> None:
    model = RallySegmenter(_config())
    checkpoint = rally_checkpoint(
        model,
        dataset_fingerprint="dataset",
        annotation_fingerprint="annotation",
        split_manifest={"fold_id": "A"},
        training_recipe=RallyTrainingRecipe(),
        chosen_epoch=7,
        decoder_configuration={
            "threshold": 0.5,
            "max_gap_seconds": 0.2,
            "minimum_duration_seconds": 0.3,
            "preserve_edge_runs": True,
        },
    )
    restored = RallySegmenter.from_checkpoint(checkpoint)
    assert restored.config == model.config
    assert checkpoint["rally_feature_schema"] == RALLY_FEATURE_SCHEMA
    assert checkpoint["rally_feature_version"] == RALLY_FEATURE_VERSION
    assert checkpoint["context_direction"] == RALLY_CONTEXT_DIRECTION
    incomplete = dict(checkpoint)
    incomplete.pop("annotation_fingerprint")
    with pytest.raises(ValueError, match="annotation_fingerprint"):
        RallySegmenter.from_checkpoint(incomplete)


def _window(
    source: str,
    frame: int,
    *,
    fps: float = 10.0,
    frame_count: int = 100,
) -> RallyWindow:
    return RallyWindow(
        source_id=source,
        window_id=f"{source}-{frame}",
        anchor_frame=frame,
        frame_indices=(frame,),
        owned_frames=frozenset({frame}),
        relative_time_seconds=torch.tensor([0.0]),
        frame_values=torch.zeros(1, RALLY_FEATURE_DIM),
        frame_validity=torch.zeros(1, RALLY_FEATURE_DIM, dtype=torch.bool),
        inplay_targets=torch.tensor([frame % 2]),
        rally_start_targets=torch.tensor([0.0]),
        rally_end_targets=torch.tensor([0.0]),
        metadata={"fps": fps, "frame_count": frame_count},
    )


def test_rally_collation_preserves_source_local_padding_and_masks() -> None:
    first = _window("a", 0)
    second = RallyWindow(
        **{
            **vars(_window("b", 0)),
            "frame_indices": (0, 1),
            "owned_frames": frozenset({0, 1}),
            "relative_time_seconds": torch.tensor([0.0, 0.1]),
            "frame_values": torch.zeros(2, RALLY_FEATURE_DIM),
            "frame_validity": torch.zeros(
                2, RALLY_FEATURE_DIM, dtype=torch.bool
            ),
            "inplay_targets": torch.tensor([0, 1]),
            "rally_start_targets": torch.tensor([0.0, 1.0]),
            "rally_end_targets": torch.tensor([0.0, 0.0]),
        }
    )
    batch = collate_rally_windows((first, second))
    assert batch.frame_mask.tolist() == [[True, False], [True, True]]
    assert batch.inplay_targets[0].tolist() == [0, MASKED_TARGET]
    assert first.frame_indices == (0,)
    assert second.frame_indices == (0, 1)


def test_camera_splits_are_contained_guarded_and_deterministic() -> None:
    windows = [
        *(_window("a1", frame) for frame in range(100)),
        *(_window("a2", frame) for frame in range(100)),
        *(_window("b1", frame) for frame in range(100)),
    ]
    groups = {"camera-a": ("a1", "a2"), "camera-b": ("b1",)}
    first = chronological_camera_train_validation_split(
        windows, groups, guard_seconds=1.0
    )
    second = chronological_camera_train_validation_split(
        windows, groups, guard_seconds=1.0
    )
    assert first[2] == second[2]
    assert {
        window.window_id for window in first[0]
    }.isdisjoint(window.window_id for window in first[1])
    assert first[2]["test_labels_used_for_selection"] is False
    assert first[2]["cameras"]["camera-a"]["train_global_range"] == [0, 149]
    assert first[2]["cameras"]["camera-a"]["validation_global_range"] == [
        170,
        199,
    ]
    folds = camera_held_out_folds(
        {"camera-a": ("a1", "a2"), "camera-b": ("b1",)}
    )
    assert set(folds[0].training_source_ids).isdisjoint(
        folds[0].test_source_ids
    )


def _rows(clean: bool) -> list[dict]:
    rows = []
    for source in ("a", "b"):
        for frame in range(20):
            target = int(4 <= frame <= 12)
            probability = (
                (0.9 if target else 0.1)
                if clean
                else (0.7 if target else 0.3)
            )
            rows.append(
                {
                    "source_id": source,
                    "frame": frame,
                    "inplay_target": target,
                    "inplay_probability": probability,
                    "decoded_inplay": probability >= 0.5,
                }
            )
    return rows


def test_metrics_and_stratified_bootstrap_are_deterministic() -> None:
    clean, legacy = _rows(True), _rows(False)
    metrics = rally_metrics(clean)
    assert metrics["raw_bce"] < rally_metrics(legacy)["raw_bce"]
    assert metrics["intervals"]["f1"] == 1.0
    first = paired_bce_bootstrap(clean, legacy, samples=100, seed=1729)
    second = paired_bce_bootstrap(clean, legacy, samples=100, seed=1729)
    assert first == second
    assert first["confidence_interval_95"][0] > 0


def test_predeclared_acceptance_and_track_follow_up_gates() -> None:
    clean = rally_metrics(_rows(True))
    legacy = rally_metrics(_rows(False))
    acceptance = independent_model_acceptance(
        {"a": clean, "b": clean},
        {"a": legacy, "b": legacy},
        {"confidence_interval_95": [0.1, 0.2]},
    )
    assert acceptance["accepted"] is True
    rejected = stable_track_fusion_acceptance(
        {"raw_bce": 0.20},
        {"raw_bce": 0.205},
        {"raw_bce": 0.22},
        {"confidence_interval_95": [0.01, 0.02]},
        interval_and_boundary_guardrails_satisfied=True,
    )
    assert rejected["accepted"] is False


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable in this process"
)
def test_mps_loss_is_finite_when_host_exposes_mps() -> None:
    device = torch.device("mps")
    model = RallySegmenter(_config()).to(device)
    losses = model.losses(_batch().to(device))
    assert bool(torch.isfinite(losses.total))
