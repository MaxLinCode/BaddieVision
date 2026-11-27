from __future__ import annotations

import copy
from dataclasses import fields, replace

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
from src.temporal_selector.rally_audio import RALLY_AUDIO_FEATURE_NAMES


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


def test_temporal_smoothing_uses_only_adjacent_owned_same_state_frames() -> None:
    batch = _batch()
    model = RallySegmenter(_config())
    output = model(batch)
    # Only the middle pair of the first row has matching supervised labels.
    output.inplay_logits = torch.tensor(
        [[-2.0, -1.0, 1.0, -2.0], [0.0, 2.0, -torch.inf, -torch.inf]],
        requires_grad=True,
    )
    baseline = model.losses(batch, output)
    smoothed = model.losses(batch, output, temporal_smoothing_weight=0.5)
    probabilities = output.inplay_logits.detach().sigmoid()
    expected = 0.5 * (probabilities[0, 2] - probabilities[0, 1])
    penalty = smoothed.total - baseline.total
    torch.testing.assert_close(penalty, expected)
    penalty.backward()
    gradient = output.inplay_logits.grad
    assert gradient is not None
    assert gradient[0, 1] < 0 < gradient[0, 2]
    assert gradient.count_nonzero() == 2


def test_attention_masks_missing_audio_without_masking_its_video_frames() -> None:
    audio_dim = len(RALLY_AUDIO_FEATURE_NAMES)
    model = RallySegmenter(replace(
        _config(), frame_feature_dim=RALLY_FEATURE_DIM + audio_dim, audio_attention=True,
    )).eval()
    batch = _batch()
    batch.frame_values = torch.zeros(2, 4, model.config.frame_feature_dim)
    batch.frame_validity = torch.ones_like(batch.frame_values, dtype=torch.bool)
    audio_validity = batch.frame_validity[..., -audio_dim:]
    audio_validity[0, 0] = False
    audio_validity[0, 2, 1:] = False
    audio_validity[1, :2] = False
    masks = []
    handle = model.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: masks.append(kwargs["src_key_padding_mask"]),
        with_kwargs=True,
    )
    with torch.no_grad():
        expected = model(batch).inplay_logits
        batch.frame_values[..., -audio_dim:].masked_fill_(~audio_validity, 1000)
        actual = model(batch).inplay_logits
    handle.remove()
    assert masks[0].tolist() == [
        [False, False, False, False, True, False, False, False],
        [False, False, True, True, True, True, True, True],
    ]
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual[batch.frame_mask]).all()


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


def test_mixed_fps_camera_split_keeps_guard_in_seconds() -> None:
    windows = [
        *(_window("slow", frame, fps=10) for frame in range(100)),
        *(_window("fast", frame, fps=20) for frame in range(100)),
    ]
    train, validation, split = chronological_camera_train_validation_split(
        windows, {"camera": ("slow", "fast")}, guard_seconds=1.0,
    )

    assert len(train) == 120
    assert train[-1].window_id == "fast-19"
    assert len(validation) == 40
    assert validation[0].window_id == "fast-60"
    assert split["cameras"]["camera"]["cut_seconds"] == 12


def test_mixed_fps_camera_split_includes_exact_fractional_boundary() -> None:
    windows = [
        *(_window("early", frame, fps=24, frame_count=120) for frame in range(120)),
        *(_window("late", frame, fps=25, frame_count=175) for frame in range(175)),
    ]
    train, validation, _ = chronological_camera_train_validation_split(
        windows, {"camera": ("early", "late")}, guard_seconds=1.0,
    )

    assert [window.window_id for window in train] == [
        *(f"early-{frame}" for frame in range(120)),
        *(f"late-{frame}" for frame in range(90)),
    ]
    assert [window.window_id for window in validation] == [
        f"late-{frame}" for frame in range(140, 175)
    ]


@pytest.mark.parametrize("fps", [25.0, 30.0, 60.0])
def test_rally_motion_uses_elapsed_time_and_masks_gaps_and_track_changes(fps) -> None:
    from src.temporal_selector.rally_features import (
        FULL_RALLY_FEATURE_NAMES,
        MOTION_FEATURE_NAMES,
        rally_feature_view,
        rally_motion_features,
    )

    positions = torch.zeros(9, len(FULL_RALLY_FEATURE_NAMES))
    observed = torch.ones_like(positions, dtype=torch.bool)
    tracks = torch.ones(9, 2, dtype=torch.long)
    tracks[4:, 0] = 2
    time = torch.arange(9) / fps
    positions[:, FULL_RALLY_FEATURE_NAMES.index("p1_court_x")] = -2 * time
    joint = FULL_RALLY_FEATURE_NAMES.index("p2_pose_0_y")
    positions[:, joint] = 3 * time
    observed[4, joint] = False

    velocity, valid = rally_motion_features(positions, observed, tracks, fps)
    court_x = MOTION_FEATURE_NAMES.index("p1_court_velocity_x")
    joint_y = MOTION_FEATURE_NAMES.index("p2_pose_0_velocity_y")
    expected_valid = [False, False, False, True, False, False, False, True, True]
    assert valid[:, court_x].tolist() == expected_valid
    assert valid[:, joint_y].tolist() == [False, False, True, True, False, False, False, True, True]
    assert torch.allclose(velocity[valid[:, court_x], court_x], torch.full((3,), -2.0))
    assert torch.allclose(velocity[valid[:, joint_y], joint_y], torch.full((4,), 3.0))
    assert torch.count_nonzero(velocity[~valid]) == 0

    view = rally_feature_view("full_motion")
    values, mask = view.select(
        torch.cat((positions, velocity), dim=-1),
        torch.cat((observed, valid), dim=-1),
    )
    assert values.shape == mask.shape == (9, 290)
    assert torch.equal(values[:, :positions.shape[1]], positions)


def test_rally_singles_base_features_use_metric_distance_and_mask_discontinuous_motion() -> None:
    from src.court_projection import HALF_LENGTH, HALF_SINGLES_WIDTH, HALF_WIDTH
    from src.temporal_selector.rally_features import (
        FULL_RALLY_FEATURE_NAMES,
        SinglesBaseObservations,
        rally_singles_base_features,
    )

    values = torch.zeros(8, len(FULL_RALLY_FEATURE_NAMES))
    observed = torch.ones_like(values, dtype=torch.bool)
    tracks = torch.ones(8, 2, dtype=torch.long)
    near_x, near_y, far_y = (
        FULL_RALLY_FEATURE_NAMES.index(name)
        for name in ("p1_court_x", "p1_court_y", "p2_court_y")
    )
    distance_metres = torch.arange(8) * 0.25
    values[:, near_x] = distance_metres / HALF_WIDTH
    values[:, near_y] = -0.5
    values[:, far_y] = 0.5 + distance_metres.flip((0,)) / HALF_LENGTH
    tracks[4:, 1] = 2
    observed[4, near_y] = False
    observations = SinglesBaseObservations.from_features(values, observed, tracks)
    features, valid = rally_singles_base_features(observations, fps=30)

    normalization = (HALF_SINGLES_WIDTH**2 + (HALF_LENGTH / 2)**2)**0.5
    assert torch.allclose(features[:4, 0], distance_metres[:4] / normalization)
    assert torch.allclose(features[:, 1], distance_metres.flip((0,)) / normalization)
    assert torch.isclose(features[3, 2], torch.tensor(7.5 / normalization))
    assert torch.isclose(features[3, 3], torch.tensor(-7.5 / normalization))
    assert valid[:, 2].tolist() == [False, False, False, True, False, False, False, False]
    assert valid[:, 3].tolist() == [False, False, False, True, False, False, False, True]
    assert not valid[4, 0]
    assert valid[-1, 1] and features[-1, 1] == 0
    assert torch.count_nonzero(features[~valid]) == 0


def test_fit_singles_base_uses_owned_inplay_positions_and_mirrors_depth() -> None:
    from src.court_projection import HALF_LENGTH, HALF_WIDTH
    from src.temporal_selector.rally_dataset import fit_rally_singles_base, with_rally_singles_base
    from src.temporal_selector.rally_features import (
        BASE_FEATURE_NAMES,
        BASE_RADIAL_RALLY_FEATURE_VIEW,
        SinglesBaseObservations,
        rally_feature_view,
        rally_singles_base_features,
    )

    view = rally_feature_view(BASE_RADIAL_RALLY_FEATURE_VIEW)
    metres = torch.tensor([
        [[2.0, -6.0], [2.0, 6.0]],
        [[-2.0, -5.0], [-2.0, 5.0]],
        [[1.0, -4.0], [1.0, 4.0]],
        [[-1.0, -2.0], [1.0, 4.0]],
        [[0.4, -3.0], [3.0, 5.0]],
        [[-2.0, -5.0], [-2.0, 5.0]],
    ])
    observations = SinglesBaseObservations(
        metres / torch.tensor([HALF_WIDTH, HALF_LENGTH]),
        torch.ones((6, 2), dtype=torch.bool),
        torch.ones((6, 2), dtype=torch.long),
    )
    window = replace(
        _window("train", 4),
        frame_indices=(3, 4, 5),
        owned_frames=frozenset((3, 4)),
        inplay_targets=torch.ones(3, dtype=torch.long),
        frame_values=torch.ones(3, view.dimension),
        frame_validity=torch.ones(3, view.dimension, dtype=torch.bool),
        metadata={"fps": 30.0, "feature_view": view.name},
        singles_base_observations=observations,
    )
    base = fit_rally_singles_base([window])
    assert base == pytest.approx((0.4, 3.0))
    with pytest.raises(ValueError, match="observed in-play"):
        fit_rally_singles_base([replace(window, inplay_targets=torch.zeros(3))])

    fitted = with_rally_singles_base([window], base)[0]
    expected_values, expected_validity = rally_singles_base_features(observations, 30, base)
    columns = [view.names.index(name) for name in BASE_FEATURE_NAMES]
    assert torch.equal(fitted.frame_values[:, columns], expected_values[3:])
    assert torch.equal(fitted.frame_validity[:, columns], expected_validity[3:])
    assert fitted.frame_validity[0, columns[2:]].all()
    assert torch.all(fitted.frame_values[:, :min(columns)] == 1)
    assert torch.all(fitted.frame_values[:, max(columns) + 1:] == 1)


def test_rally_motion_pose_uses_crop_geometry_and_track_median_torso() -> None:
    from src.temporal_selector.observations import frame_context
    from src.temporal_selector.rally_features import (
        FULL_RALLY_FEATURE_NAMES,
        normalize_rally_motion_pose,
    )

    image_size = (640, 480)
    crop = [85, 70, 215, 330]
    assignment = {"slots": {"P1": {"confidence": 1, "assignment": {
        "track_id": 1, "bbox": [100, 100, 200, 300], "foot": [150, 300],
        "court_x": 0, "court_y": 0, "detection_confidence": 1, "activity": 0,
    }}}}
    observations = []
    for frame, torso in enumerate((20, 20, 100, 40, 40)):
        pixels = [(160, 200)] * 33
        for left, right, y in ((11, 12, 200 - torso), (23, 24, 200)):
            pixels[left], pixels[right] = (140, y), (160, y)
        landmarks = [
            {"x": (x - crop[0]) / (crop[2] - crop[0]),
             "y": (y - crop[1]) / (crop[3] - crop[1]),
             "visibility": 0 if (frame, index) == (4, 24) else 1}
            for index, (x, y) in enumerate(pixels)
        ]
        track = 1 if frame < 3 else 2
        assignment["slots"]["P1"]["assignment"]["track_id"] = track
        poses = {(frame, track): {
            "status": "detected", "pose_bbox": crop, "pose_landmarks": landmarks,
        }}
        observations.append(frame_context(assignment, poses, frame, image_size, "full_context", 0.5, "image"))
    values, valid = map(torch.tensor, zip(*observations))
    track_ids = torch.tensor([[1, -1], [1, -1], [1, -1], [2, -1], [2, -1]])
    normalize_rally_motion_pose(values, valid, track_ids, image_size)
    wrist_x = FULL_RALLY_FEATURE_NAMES.index("p1_pose_15_x")
    assert values[:, wrist_x].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.25, 0.0])
    assert valid[:, wrist_x].tolist() == [True, True, True, True, False]


def test_court_median_gates_low_quality_frames_without_filtering_velocity() -> None:
    from src.temporal_selector.rally_features import (
        FULL_RALLY_FEATURE_NAMES,
        MOTION_FEATURE_NAMES,
        rally_motion_features,
        smooth_rally_court_positions,
    )

    values = torch.zeros(9, len(FULL_RALLY_FEATURE_NAMES))
    valid = torch.ones_like(values, dtype=torch.bool)
    tracks = torch.ones(9, 2, dtype=torch.long)
    tracks[7:, 0] = 2
    court_x = FULL_RALLY_FEATURE_NAMES.index("p1_court_x")
    other_x = FULL_RALLY_FEATURE_NAMES.index("p2_court_x")
    values[:, court_x] = torch.tensor([0., 0., 10., 0., 0., 8., 0., 7., 0.])
    valid[6, court_x] = False
    values[:, other_x] = torch.arange(len(values))
    original_mask = valid.clone()

    reliable = torch.ones_like(tracks, dtype=torch.bool)
    reliable[2, 0] = False
    velocity, mask = rally_motion_features(values, valid, tracks, fps=30)
    smooth_rally_court_positions(values, valid, tracks, reliable)
    assert values[2, court_x] == 10

    reliable[2, 0] = True
    smooth_rally_court_positions(values, valid, tracks, reliable)
    assert values[:, court_x].tolist() == [0., 0., 0., 0., 0., 8., 0., 7., 0.]
    assert torch.equal(values[:, other_x], torch.arange(len(values)))
    assert torch.equal(valid, original_mask)

    court_vx = MOTION_FEATURE_NAMES.index("p1_court_velocity_x")
    assert mask[3, court_vx]
    assert velocity[3, court_vx] == 30
    assert not mask[6:, court_vx].any()



def test_court_foot_reliability_requires_two_visible_feet_and_accepted_pose() -> None:
    from src.temporal_selector.rally_dataset import _reliable_court_foot

    landmarks = [{"visibility": 0.0, "x": 0.4, "y": 0.8} for _ in range(33)]
    pose = {"status": "detected", "pose_landmarks": landmarks}
    landmarks[31]["visibility"] = 0.8
    assert not _reliable_court_foot({}, pose, 0.5)
    landmarks[32]["visibility"] = 0.8
    assert _reliable_court_foot({}, pose, 0.5)
    assert not _reliable_court_foot({}, pose, 0.9)
    assert not _reliable_court_foot({"foot_diagnostic": {"status": "disagrees"}}, pose, 0.5)
    assert _reliable_court_foot({}, dict(pose, temporal_status="validated"), 0.5)
    assert not _reliable_court_foot({}, dict(pose, temporal_status="interpolated_single_gap"), 0.5)
