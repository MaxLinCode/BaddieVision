import pytest
import torch

from src.temporal_selector import SelectorBatch, SelectorConfig, TemporalShuttleSelector
from src.temporal_selector.dataset import _frozen_candidate_frames
from src.annotation_platform.shuttle import GROUPING_VERSION
from src.single_video.shuttle import CANDIDATE_RETENTION_POLICY


def _batch(*, batch_size=1, frame_dim=0, targets=None, candidate_padding=0):
    # Candidate slots deliberately interleave frames to exercise packed mapping.
    frame_times = torch.tensor([[-1.0, 0.0, 1.0]]).expand(batch_size, -1).clone()
    candidate_frames = torch.tensor([[1, 0, 1, 2]]).expand(batch_size, -1).clone()
    candidate_values = torch.randn(batch_size, 4 + candidate_padding, 12)
    candidate_validity = torch.ones_like(candidate_values, dtype=torch.bool)
    candidate_mask = torch.zeros(batch_size, 4 + candidate_padding, dtype=torch.bool)
    candidate_mask[:, :4] = True
    if candidate_padding:
        candidate_frames = torch.cat(
            (candidate_frames, torch.full((batch_size, candidate_padding), -1)), dim=1
        )
    if targets is None:
        targets = torch.tensor([[-100, 1, -1]]).expand(batch_size, -1).clone()
    return SelectorBatch(
        candidate_values=candidate_values,
        candidate_validity=candidate_validity,
        candidate_frame_indices=candidate_frames,
        candidate_mask=candidate_mask,
        frame_values=torch.randn(batch_size, 3, frame_dim),
        frame_validity=torch.ones(batch_size, 3, frame_dim, dtype=torch.bool),
        frame_mask=torch.ones(batch_size, 3, dtype=torch.bool),
        relative_time_seconds=frame_times,
        targets=targets,
    )


def test_dataset_consumes_complete_frozen_candidate_view_without_reordering():
    metadata = {
        "frozen_candidate_view": {
            "schema_version": 1,
            "record_field": "frozen_candidates",
            "grouping_version": GROUPING_VERSION,
            "ordering_policy": list(CANDIDATE_RETENTION_POLICY),
            "retention_k": 8,
        }
    }
    records = [
        {
            "frame": 0,
            "candidates": [
                {"candidate_id": "lower-ranked"},
                {"candidate_id": "higher-ranked"},
            ],
            "frozen_candidates": [
                {
                    "candidate_id": "higher-ranked",
                    "grouping_version": GROUPING_VERSION,
                    "raw_member_ids": ["higher-ranked"],
                },
                {
                    "candidate_id": "lower-ranked",
                    "grouping_version": GROUPING_VERSION,
                    "raw_member_ids": ["lower-ranked"],
                },
            ],
        }
    ]
    frames = _frozen_candidate_frames(metadata, records, retention_k=8)
    assert [item["candidate_id"] for item in frames[0]] == [
        "higher-ranked",
        "lower-ranked",
    ]

    records[0].pop("frozen_candidates")
    with pytest.raises(ValueError, match="incomplete frozen candidate view"):
        _frozen_candidate_frames(metadata, records, retention_k=8)


def _small_config(**kwargs):
    values = dict(
        token_size=16,
        num_layers=1,
        num_attention_heads=4,
        feed_forward_size=24,
        dropout=0.0,
    )
    values.update(kwargs)
    return SelectorConfig(**values)


def test_baseline_defaults_and_configurable_capacity():
    config = SelectorConfig()
    assert (config.token_size, config.num_layers, config.num_attention_heads) == (
        128,
        4,
        4,
    )
    assert config.feed_forward_size == 256
    model = TemporalShuttleSelector(
        _small_config(activation="relu", norm_first=False, final_norm=False)
    )
    layer = model.encoder.transformer.layers[0]
    assert layer.self_attn.batch_first
    assert not layer.norm_first


@pytest.mark.parametrize(
    "mode,frame_dim",
    [("candidates_only", 0), ("players_court", 5), ("full_context", 9)],
)
def test_context_modes_shapes_and_cpu_backprop(mode, frame_dim):
    model = TemporalShuttleSelector(
        _small_config(context_mode=mode, frame_feature_dim=frame_dim)
    )
    batch = _batch(batch_size=2, frame_dim=frame_dim)
    output = model(batch)
    assert output.candidate_logits.shape == (2, 4)
    assert output.null_logits.shape == (2, 3)
    loss = model.loss(batch, output)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_frame_local_mapping_null_and_noncenter_supervision():
    batch = _batch(targets=torch.tensor([[0, -100, -1]]))
    model = TemporalShuttleSelector(_small_config())
    output = model(batch)
    # Packed order is F0,C(slot1),F1,C(slot0),C(slot2),F2,C(slot3).
    assert output.encoded.frame_token_indices.tolist() == [[0, 2, 5]]
    assert output.encoded.candidate_token_indices.tolist() == [[3, 1, 4, 6]]
    assert output.encoded.packed_relative_time_seconds.tolist() == [
        [-1, -1, 0, 0, 0, 1, 1]
    ]
    assert torch.isfinite(model.loss(batch, output))


def test_zero_candidates_valid_null_and_completely_unsupervised():
    batch = _batch(targets=torch.tensor([[-100, -1, -100]]))
    batch.candidate_mask[:] = False
    batch.candidate_frame_indices[:] = -1
    model = TemporalShuttleSelector(_small_config())
    loss = model.loss(batch)
    loss.backward()
    batch.targets[:] = -100
    unsupervised = model.loss(batch)
    assert unsupervised.item() == 0
    unsupervised.backward()


def test_invalid_candidate_target_and_frame_association_rejected():
    batch = _batch(targets=torch.tensor([[-100, 2, -100]]))
    with pytest.raises(ValueError, match="references no candidate"):
        batch.validate()
    batch = _batch()
    batch.candidate_frame_indices[0, 0] = 9
    with pytest.raises(ValueError, match="nonexistent frame"):
        batch.validate()


def test_batch_padding_invariance_and_candidate_permutation_equivariance():
    torch.manual_seed(4)
    model = TemporalShuttleSelector(_small_config()).eval()
    batch = _batch()
    padded = _batch(candidate_padding=3)
    for name in (
        "candidate_values",
        "candidate_validity",
        "candidate_frame_indices",
        "candidate_mask",
    ):
        getattr(padded, name)[:, :4] = getattr(batch, name)
    padded.frame_values.copy_(batch.frame_values)
    padded.relative_time_seconds.copy_(batch.relative_time_seconds)
    with torch.no_grad():
        expected = model(batch)
        actual = model(padded)
    torch.testing.assert_close(
        expected.candidate_logits, actual.candidate_logits[:, :4]
    )
    torch.testing.assert_close(expected.null_logits, actual.null_logits)

    permutation = torch.tensor([2, 3, 0, 1])
    permuted = _batch()
    for name in (
        "candidate_values",
        "candidate_validity",
        "candidate_frame_indices",
        "candidate_mask",
    ):
        setattr(permuted, name, getattr(batch, name)[:, permutation])
    permuted.frame_values.copy_(batch.frame_values)
    permuted.relative_time_seconds.copy_(batch.relative_time_seconds)
    with torch.no_grad():
        result = model(permuted)
    torch.testing.assert_close(
        expected.candidate_logits[:, permutation], result.candidate_logits
    )
    torch.testing.assert_close(expected.null_logits, result.null_logits)


def test_noncausal_frames_on_both_sides_influence_center():
    torch.manual_seed(8)
    model = TemporalShuttleSelector(
        _small_config(context_mode="players_court", frame_feature_dim=2)
    ).eval()
    batch = _batch(frame_dim=2)
    with torch.no_grad():
        baseline = model(batch).null_logits[0, 1]
        before = _batch(frame_dim=2)
        before.candidate_values.copy_(batch.candidate_values)
        before.frame_values.copy_(batch.frame_values)
        before.frame_values[0, 0] += 10
        after = _batch(frame_dim=2)
        after.candidate_values.copy_(batch.candidate_values)
        after.frame_values.copy_(batch.frame_values)
        after.frame_values[0, 2] += 10
        assert not torch.isclose(baseline, model(before).null_logits[0, 1])
        assert not torch.isclose(baseline, model(after).null_logits[0, 1])


def test_validity_is_an_explicit_input_and_time_must_be_centered():
    model = TemporalShuttleSelector(_small_config())
    batch = _batch()
    invalid = _batch()
    invalid.candidate_values.copy_(batch.candidate_values)
    invalid.candidate_validity.copy_(batch.candidate_validity)
    invalid.candidate_validity[0, 0, 0] = False
    with torch.no_grad():
        assert not torch.isclose(
            model(batch).candidate_logits[0, 0], model(invalid).candidate_logits[0, 0]
        )
    batch.relative_time_seconds += 2
    with pytest.raises(ValueError, match="centered"):
        model(batch)


def test_tiny_temporal_association_problem_can_overfit():
    torch.manual_seed(12)
    # Each frame has two ambiguous observations. Across time the selected
    # observations form the smooth left-to-right path (local slots 0, 1, 0).
    candidate_values = torch.zeros(1, 6, 12)
    candidate_values[0, :, :2] = torch.tensor(
        [
            [0.15, 0.4],
            [0.85, 0.4],
            [0.65, 0.4],
            [0.35, 0.4],
            [0.55, 0.4],
            [0.45, 0.4],
        ]
    )
    batch = SelectorBatch(
        candidate_values=candidate_values,
        candidate_validity=torch.ones_like(candidate_values, dtype=torch.bool),
        candidate_frame_indices=torch.tensor([[0, 0, 1, 1, 2, 2]]),
        candidate_mask=torch.ones(1, 6, dtype=torch.bool),
        frame_values=torch.empty(1, 3, 0),
        frame_validity=torch.empty(1, 3, 0, dtype=torch.bool),
        frame_mask=torch.ones(1, 3, dtype=torch.bool),
        relative_time_seconds=torch.tensor([[-1.0, 0.0, 1.0]]),
        targets=torch.tensor([[0, 1, 0]]),
    )
    model = TemporalShuttleSelector(
        _small_config(token_size=16, num_attention_heads=2, feed_forward_size=32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(300):
        optimizer.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        optimizer.step()
    assert model.loss(batch).item() < 0.03
