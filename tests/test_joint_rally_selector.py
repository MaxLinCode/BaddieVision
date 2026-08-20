from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from src.temporal_selector import (
    MASKED_TARGET,
    InPlayDecoderConfig,
    JointRallyShuttleModel,
    RallyInterval,
    RallySourceManifest,
    SelectorBatch,
    SelectorConfig,
    SelectorWindow,
    TemporalShuttleSelector,
    build_two_source_crossfit,
    calibrate_inplay_decoder,
    decode_inplay_probabilities,
    decoded_intervals,
    read_rally_intervals,
    refill_frames,
    write_joint_artifacts,
    write_rally_intervals,
)
from src.temporal_selector.experiment import aggregate_joint_predictions, run_experiment


def _joint_batch() -> SelectorBatch:
    return SelectorBatch(
        candidate_values=torch.randn(1, 2, 12),
        candidate_validity=torch.ones(1, 2, 12, dtype=torch.bool),
        candidate_frame_indices=torch.tensor([[1, 2]]),
        candidate_mask=torch.ones(1, 2, dtype=torch.bool),
        frame_values=torch.randn(1, 3, 4),
        frame_validity=torch.ones(1, 3, 4, dtype=torch.bool),
        frame_mask=torch.ones(1, 3, dtype=torch.bool),
        relative_time_seconds=torch.tensor([[-1.0, 0.0, 1.0]]),
        targets=torch.tensor([[MASKED_TARGET, 0, 0]]),
        inplay_targets=torch.tensor([[0, 1, 1]]),
    ).validate(frame_feature_dim=4)


def test_joint_head_losses_and_zero_candidate_state_path() -> None:
    config = SelectorConfig(
        context_mode="full_context",
        frame_feature_dim=4,
        token_size=16,
        num_layers=1,
        num_attention_heads=2,
        feed_forward_size=32,
        dropout=0,
    )
    model = JointRallyShuttleModel(config)
    batch = _joint_batch()
    output = model(batch)
    assert output.inplay_logits.shape == (1, 3)
    losses = model.joint_losses(batch, output, inplay_pos_weight=torch.tensor(0.5))
    assert losses.selection_frames == 2
    assert losses.inplay_frames == 3
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert model.inplay_head.projection.weight.grad is not None

    output.inplay_logits = torch.tensor([[0.0, 0.0, -5.0]])
    unweighted = model.joint_losses(batch, output).inplay
    weighted = model.joint_losses(
        batch,
        output,
        boundary_weight=3.0,
        boundary_window_seconds=1.0,
    ).inplay
    assert weighted < unweighted
    with pytest.raises(ValueError, match="boundary weight"):
        model.joint_losses(batch, output, boundary_weight=0.5)

    batch.frame_mask[0, 2] = False
    batch.inplay_targets[0, 2] = MASKED_TARGET
    batch.targets[0, 2] = MASKED_TARGET
    output.inplay_logits = torch.tensor([[0.0, 0.0, float("nan")]])
    padded = model.joint_losses(batch, output, boundary_weight=3.0)
    assert torch.isfinite(padded.inplay)

    batch.candidate_mask[:] = False
    batch.candidate_frame_indices[:] = -1
    batch.targets[:] = MASKED_TARGET
    zero_candidate = model.joint_losses(batch)
    assert torch.isfinite(zero_candidate.inplay)


def test_optional_boundary_heads_are_finite_and_checkpoint_compatible() -> None:
    config = SelectorConfig(
        context_mode="full_context",
        frame_feature_dim=4,
        token_size=16,
        num_layers=1,
        num_attention_heads=2,
        feed_forward_size=32,
        dropout=0,
    )
    batch = _joint_batch()
    batch.rally_start_targets = torch.tensor([[0.0, 1.0, MASKED_TARGET]])
    batch.rally_end_targets = torch.tensor([[MASKED_TARGET, 0.5, 1.0]])
    model = JointRallyShuttleModel(config, boundary_heads=True)
    output = model(batch)
    losses = model.joint_losses(
        batch,
        output,
        boundary_start_pos_weight=torch.tensor(10.0),
        boundary_end_pos_weight=torch.tensor(10.0),
    )
    assert output.rally_start_logits.shape == (1, 3)
    assert output.rally_end_logits.shape == (1, 3)
    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.rally_start)
    assert torch.isfinite(losses.rally_end)
    losses.total.backward()
    assert model.rally_start_head.projection.weight.grad is not None
    assert JointRallyShuttleModel(config).load_state_dict(
        JointRallyShuttleModel(config).state_dict()
    ) is not None


def test_legacy_selector_checkpoint_contract_has_no_joint_head() -> None:
    config = SelectorConfig(
        token_size=16,
        num_layers=1,
        num_attention_heads=2,
        feed_forward_size=32,
    )
    legacy_keys = TemporalShuttleSelector(config).state_dict()
    joint_keys = JointRallyShuttleModel(config).state_dict()
    assert not any(key.startswith("inplay_head.") for key in legacy_keys)
    assert any(key.startswith("inplay_head.") for key in joint_keys)


@pytest.mark.parametrize(
    "conditioning_mode",
    (
        "soft_detached",
        "soft_joint",
        "presence_only",
        "hard_shared",
        "hard_isolated",
        "presence_isolated",
    ),
)
def test_conditioned_selection_probabilities_normalize_with_null_only_frames(
    conditioning_mode: str,
) -> None:
    config = SelectorConfig(
        context_mode="full_context",
        frame_feature_dim=4,
        token_size=16,
        num_layers=1,
        num_attention_heads=2,
        feed_forward_size=32,
        dropout=0,
    )
    model = JointRallyShuttleModel(
        config, boundary_heads=True, conditioning_mode=conditioning_mode
    ).eval()
    batch = _joint_batch()
    output = model(batch)
    candidate_probabilities, null_probabilities = (
        model.frame_selection_probabilities(output)
    )
    for frame in range(batch.frame_mask.shape[1]):
        slots = batch.candidate_frame_indices[0] == frame
        assert torch.allclose(
            candidate_probabilities[0, slots].sum() + null_probabilities[0, frame],
            torch.tensor(1.0),
        )
    assert null_probabilities[0, 0] == 1
    assert torch.isfinite(output.inplay_logits).all()
    assert torch.isfinite(output.rally_start_logits).all()
    assert torch.isfinite(output.rally_end_logits).all()


def test_conditioning_is_padding_safe_and_candidate_permutation_invariant() -> None:
    config = SelectorConfig(
        context_mode="full_context",
        frame_feature_dim=4,
        token_size=16,
        num_layers=1,
        num_attention_heads=2,
        feed_forward_size=32,
        dropout=0,
    )
    model = JointRallyShuttleModel(
        config, conditioning_mode="soft_detached"
    ).eval()
    batch = _joint_batch()
    batch.candidate_frame_indices[:] = torch.tensor([[1, 1]])
    batch.targets[:] = torch.tensor([[MASKED_TARGET, 0, MASKED_TARGET]])
    permuted = copy.deepcopy(batch)
    permutation = torch.tensor([1, 0])
    permuted.candidate_values = permuted.candidate_values[:, permutation]
    permuted.candidate_validity = permuted.candidate_validity[:, permutation]
    permuted.candidate_frame_indices = permuted.candidate_frame_indices[:, permutation]
    permuted.targets[0, 1] = 1
    with torch.no_grad():
        original_output = model(batch)
        permuted_output = model(permuted)
    assert torch.allclose(
        original_output.inplay_logits, permuted_output.inplay_logits, atol=1e-6
    )

    padded = copy.deepcopy(batch)
    padded.frame_mask[0, 2] = False
    padded.targets[0, 2] = MASKED_TARGET
    padded.inplay_targets[0, 2] = MASKED_TARGET
    padded_output = model(padded)
    assert torch.isfinite(padded_output.inplay_logits[0, :2]).all()
    assert torch.isneginf(padded_output.inplay_logits[0, 2])
    losses = model.joint_losses(padded, padded_output)
    assert torch.isfinite(losses.total)


def test_detached_and_joint_modes_control_selector_probability_gradients() -> None:
    config = SelectorConfig(
        context_mode="full_context",
        frame_feature_dim=4,
        token_size=16,
        num_layers=1,
        num_attention_heads=2,
        feed_forward_size=32,
        dropout=0,
    )
    for mode, expects_gradient in (("soft_detached", False), ("soft_joint", True)):
        model = JointRallyShuttleModel(config, conditioning_mode=mode)
        output = model(_joint_batch())
        output.inplay_logits[0, 1:].sum().backward()
        candidate_gradient = model.selection_head.projection.weight.grad
        null_gradient = model.null_head.projection.weight.grad
        if expects_gradient:
            assert candidate_gradient is not None
            assert null_gradient is not None
            assert candidate_gradient.abs().sum() > 0
            assert null_gradient.abs().sum() > 0
        else:
            assert candidate_gradient is None
            assert null_gradient is None


@pytest.mark.parametrize("mode", ("hard_shared", "hard_isolated"))
def test_hard_routing_decision_is_detached_from_rally_loss(mode: str) -> None:
    config = SelectorConfig(
        context_mode="full_context",
        frame_feature_dim=4,
        token_size=16,
        num_layers=1,
        num_attention_heads=2,
        feed_forward_size=32,
        dropout=0,
    )
    model = JointRallyShuttleModel(config, conditioning_mode=mode)
    output = model(_joint_batch())
    output.inplay_logits[:, 1:].sum().backward()
    assert model.selection_head.projection.weight.grad is None
    assert model.null_head.projection.weight.grad is None


def test_hard_isolated_ignores_unselected_candidate_content() -> None:
    config = SelectorConfig(
        context_mode="full_context",
        frame_feature_dim=4,
        token_size=16,
        num_layers=1,
        num_attention_heads=2,
        feed_forward_size=32,
        dropout=0,
    )
    model = JointRallyShuttleModel(
        config, boundary_heads=True, conditioning_mode="hard_isolated"
    ).eval()
    batch = _joint_batch()
    batch.candidate_frame_indices[:] = torch.tensor([[1, 1]])
    # Freeze the detached selector decision so this test isolates the rally
    # branch contract rather than testing whether a counterfactual changes the
    # selector's decision.
    selected = torch.tensor([[-1, 0, -1]])
    model._hard_selected_candidate_slots = lambda output: selected  # type: ignore[method-assign]
    changed = copy.deepcopy(batch)
    changed.candidate_values[0, 1] += 1000
    changed.candidate_validity[0, 1] = ~changed.candidate_validity[0, 1]
    with torch.no_grad():
        original = model(batch)
        counterfactual = model(changed)
    assert torch.equal(original.inplay_logits, counterfactual.inplay_logits)
    assert torch.equal(original.rally_start_logits, counterfactual.rally_start_logits)
    assert torch.equal(original.rally_end_logits, counterfactual.rally_end_logits)


def test_presence_isolated_ignores_all_candidate_coordinates() -> None:
    config = SelectorConfig(
        context_mode="full_context",
        frame_feature_dim=4,
        token_size=16,
        num_layers=1,
        num_attention_heads=2,
        feed_forward_size=32,
        dropout=0,
    )
    model = JointRallyShuttleModel(
        config, boundary_heads=True, conditioning_mode="presence_isolated"
    ).eval()
    batch = _joint_batch()
    changed = copy.deepcopy(batch)
    changed.candidate_values = torch.randn_like(changed.candidate_values) * 1000
    changed.candidate_validity = ~changed.candidate_validity
    with torch.no_grad():
        original = model(batch)
        counterfactual = model(changed)
    assert torch.equal(original.inplay_logits, counterfactual.inplay_logits)
    assert torch.equal(original.rally_start_logits, counterfactual.rally_start_logits)
    assert torch.equal(original.rally_end_logits, counterfactual.rally_end_logits)


def test_boundary_heads_use_conditioned_readout() -> None:
    config = SelectorConfig(
        context_mode="full_context",
        frame_feature_dim=4,
        token_size=16,
        num_layers=1,
        num_attention_heads=2,
        feed_forward_size=32,
        dropout=0,
    )
    model = JointRallyShuttleModel(
        config, boundary_heads=True, conditioning_mode="soft_detached"
    )
    output = model(_joint_batch())
    (output.rally_start_logits[:, 1:].sum() + output.rally_end_logits[:, 1:].sum()).backward()
    assert model.conditioning_fusion.weight.grad is not None
    assert model.conditioning_fusion.weight.grad.abs().sum() > 0


def test_conditioning_checkpoint_mode_is_recorded_and_required() -> None:
    config = SelectorConfig(
        context_mode="full_context",
        frame_feature_dim=4,
        token_size=16,
        num_layers=1,
        num_attention_heads=2,
        feed_forward_size=32,
        dropout=0,
    )
    legacy = JointRallyShuttleModel(config)
    legacy_checkpoint = {
        "selector_config": vars(config),
        "model_state_dict": legacy.state_dict(),
    }
    restored_legacy = JointRallyShuttleModel.from_checkpoint(legacy_checkpoint)
    assert restored_legacy.conditioning_mode == "none"

    conditioned = JointRallyShuttleModel(
        config, conditioning_mode="soft_joint"
    )
    conditioned_checkpoint = {
        "selector_config": vars(config),
        "model_state_dict": conditioned.state_dict(),
        "conditioning_mode": "soft_joint",
    }
    restored = JointRallyShuttleModel.from_checkpoint(conditioned_checkpoint)
    assert restored.conditioning_mode == "soft_joint"
    del conditioned_checkpoint["conditioning_mode"]
    with pytest.raises(ValueError, match="missing required conditioning_mode"):
        JointRallyShuttleModel.from_checkpoint(conditioned_checkpoint)

    isolated = JointRallyShuttleModel(
        config, boundary_heads=True, conditioning_mode="hard_isolated"
    )
    isolated_checkpoint = {
        "selector_config": vars(config),
        "model_state_dict": isolated.state_dict(),
        "boundary_heads": True,
        "conditioning_mode": "hard_isolated",
    }
    restored_isolated = JointRallyShuttleModel.from_checkpoint(isolated_checkpoint)
    assert restored_isolated.conditioning_mode == "hard_isolated"


def test_strict_interval_round_trip_fingerprint_and_refill(tmp_path: Path) -> None:
    intervals_path = tmp_path / "rallies.csv"
    manifest_path = tmp_path / "rallies.manifest.json"
    source = RallySourceManifest("match", "a" * 64, 30.0, 300)
    intervals = (
        RallyInterval("match", "match-0001", 30, 89),
        RallyInterval("match", "match-0002", 150, 239),
    )
    written = write_rally_intervals(
        intervals_path,
        manifest_path,
        intervals,
        (source,),
        annotation_revision="revision-1",
    )
    assert written.targets("match", (0, 30, 89, 90)) == (0, 1, 1, 0)
    assert read_rally_intervals(intervals_path, manifest_path) == written
    refill = refill_frames(intervals[1], fps=30, already_labeled=(150, 239))
    assert 150 not in refill and 239 not in refill
    assert 151 in refill and 238 in refill

    intervals_path.write_text(intervals_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        read_rally_intervals(intervals_path, manifest_path)


def test_interval_overlap_is_rejected_before_write(tmp_path: Path) -> None:
    source = RallySourceManifest("match", "b" * 64, 30.0, 100)
    with pytest.raises(ValueError, match="overlapping"):
        write_rally_intervals(
            tmp_path / "rallies.csv",
            tmp_path / "manifest.json",
            (
                RallyInterval("match", "a", 10, 30),
                RallyInterval("match", "b", 30, 40),
            ),
            (source,),
            annotation_revision="revision",
        )


def test_decoder_calibration_and_joint_artifacts(tmp_path: Path) -> None:
    probabilities = [0.1, 0.9, 0.8, 0.2, 0.85, 0.1]
    targets = [0, 1, 1, 1, 1, 0]
    config = calibrate_inplay_decoder(probabilities, targets, fps=10)
    decoded = decode_inplay_probabilities(probabilities, fps=10, config=config)
    assert len(decoded) == len(targets)
    frames = list(range(100, 106))
    intervals = decoded_intervals(frames, decoded)
    assert intervals

    rows = [
        {
            "prediction_source_id": "match",
            "frame": frame,
            "decoded_inplay": state,
            "in_play_probability": probability,
            "predicted_outcome": "selected" if state else "not_required",
        }
        for frame, state, probability in zip(frames, decoded, probabilities)
    ]
    frame_path, rally_path = write_joint_artifacts(rows, tmp_path)
    assert len(frame_path.read_text().splitlines()) == len(rows) + 1
    assert rally_path.read_text().splitlines()[0] == (
        "source_id,rally_id,start_frame,end_frame"
    )


def test_decoder_can_preserve_short_open_boundary_runs() -> None:
    probabilities = [0.9, 0.1, 0.1, 0.9]
    ordinary = decode_inplay_probabilities(
        probabilities,
        fps=10,
        config=InPlayDecoderConfig(
            threshold=0.5,
            max_gap_seconds=0,
            minimum_duration_seconds=0.5,
        ),
    )
    preserved = decode_inplay_probabilities(
        probabilities,
        fps=10,
        config=InPlayDecoderConfig(
            threshold=0.5,
            max_gap_seconds=0,
            minimum_duration_seconds=0.5,
            preserve_edge_runs=True,
        ),
    )
    assert ordinary == (False, False, False, False)
    assert preserved == (True, False, False, True)


def test_joint_aggregation_uses_candidate_ids_not_slots() -> None:
    position = {"peak_position_normalized": [0.2, 0.3]}
    common = {
        "prediction_source_id": "match",
        "frame": 10,
        "inplay_target": 1,
        "true_status": "selected_retained",
        "target_candidate_id": "a",
        "null_logit": -1.0,
        "inplay_logit": 2.0,
    }
    rows = [
        {
            **common,
            "owned": True,
            "aggregation_weight": 1.0,
            "candidate_ids": ["a", "b"],
            "candidate_logits": {"a": 1.0, "b": 0.0},
            "candidate_positions": {"a": position, "b": position},
        },
        {
            **common,
            "owned": False,
            "aggregation_weight": 0.5,
            "candidate_ids": ["b", "a"],
            "candidate_logits": {"b": 4.0, "a": 3.0},
            "candidate_positions": {"b": position, "a": position},
        },
    ]
    result = aggregate_joint_predictions(rows)[0]
    assert result["aggregation_observation_count"] == 2
    assert result["predicted_candidate_id"] == "a"
    assert result["candidate_logits"]["a"] == pytest.approx(5 / 3)
    assert result["candidate_logits"]["b"] == pytest.approx(4 / 3)


def test_joint_aggregation_skips_selection_for_masked_candidate_inputs() -> None:
    row = {
        "prediction_source_id": "match",
        "frame": 10,
        "inplay_target": 1,
        "true_status": "candidate_inputs_masked",
        "original_true_status": "selected_retained",
        "target_candidate_id": None,
        "null_logit": -1.0,
        "inplay_logit": 2.0,
        "owned": True,
        "aggregation_weight": 1.0,
        "candidate_ids": [],
        "candidate_logits": {},
        "candidate_positions": {},
        "candidate_inputs_masked": True,
    }
    result = aggregate_joint_predictions([row])[0]
    assert result["candidate_ids"] == []
    assert result["predicted_candidate_id"] is None
    assert result["selection_loss"] is None


class _JointDataset:
    def __init__(self) -> None:
        windows = []
        for source in ("one", "two"):
            for frame, inplay in enumerate((0, 1, 1, 0, MASKED_TARGET)):
                windows.append(
                    SelectorWindow(
                        source_id=source,
                        burst_id=f"dense-{source}-{frame}",
                        queue_kind="dense",
                        anchor_frame=frame,
                        frame_indices=(frame,),
                        owned_frames=frozenset({frame}),
                        relative_time_seconds=torch.tensor([0.0]),
                        candidate_values=torch.zeros(1, 12),
                        candidate_validity=torch.ones(1, 12, dtype=torch.bool),
                        candidate_frame_indices=torch.tensor([0]),
                        candidate_ids=(f"{source}-{frame}",),
                        frame_values=torch.full((1, 154), float(inplay)),
                        frame_validity=torch.ones(1, 154, dtype=torch.bool),
                        targets=torch.tensor([
                            0 if inplay == 1 else MASKED_TARGET
                        ]),
                        target_status=(
                            "selected_retained"
                            if inplay == 1
                            else "out_of_play"
                            if inplay == 0
                            else "out_of_coverage",
                        ),
                        inplay_targets=torch.tensor([inplay]),
                        metadata={"fps": 2.0},
                    )
                )
        self.windows = tuple(windows)
        self.manifest = {"dataset_fingerprint": "joint-synthetic"}


def test_joint_runner_writes_conditional_outputs(tmp_path: Path) -> None:
    dataset = _JointDataset()
    manifest = build_two_source_crossfit(("one", "two"), seed=1729)
    metrics = run_experiment(
        dataset,
        manifest,
        tmp_path / "run",
        context_mode="full_context",
        epochs=1,
        batch_size=2,
        device=torch.device("cpu"),
    )
    output = tmp_path / "run"
    assert (output / "rally_shuttle_predictions.jsonl").is_file()
    assert (output / "rallies.csv").is_file()
    assert metrics["schema_version"] == 2
    assert metrics["task_definition"] == (
        "strict_inplay_with_conditional_shuttle_selection"
    )
    rows = [
        json.loads(line)
        for line in (output / "rally_shuttle_predictions.jsonl").read_text().splitlines()
    ][1:]
    assert len(rows) == 10
    masked = [row for row in rows if row["inplay_target"] == MASKED_TARGET]
    assert len(masked) == 2
    assert all(row["inplay_loss"] is None for row in masked)
    assert {row["predicted_outcome"] for row in rows} <= {
        "selected",
        "no_shuttle",
        "not_required",
    }
