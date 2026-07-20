from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.temporal_selector import (
    MASKED_TARGET,
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

    batch.candidate_mask[:] = False
    batch.candidate_frame_indices[:] = -1
    batch.targets[:] = MASKED_TARGET
    zero_candidate = model.joint_losses(batch)
    assert torch.isfinite(zero_candidate.inplay)


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


class _JointDataset:
    def __init__(self) -> None:
        windows = []
        for source in ("one", "two"):
            for frame, inplay in enumerate((0, 1, 1, 0)):
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
                        targets=torch.tensor([0 if inplay else MASKED_TARGET]),
                        target_status=(
                            "selected_retained" if inplay else "out_of_play",
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
    assert len(rows) == 8
    assert {row["predicted_outcome"] for row in rows} <= {
        "selected",
        "no_shuttle",
        "not_required",
    }
