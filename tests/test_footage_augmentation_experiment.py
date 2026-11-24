import pytest
import torch

from experiments.temporal_selector.footage_augmentation_experiment import (
    FootageAugmentationSampler,
    source_train_stop,
    variant_train_windows,
)
from src.temporal_selector.rally_dataset import RallyWindow


def _window(
    source_id: str,
    anchor: int,
    *,
    frames: tuple[int, ...] | None = None,
    frame_count: int = 100,
    owns_boundary: bool = False,
) -> RallyWindow:
    frames = (
        tuple(frames)
        if frames is not None
        else tuple(range(max(0, anchor - 2), min(frame_count, anchor + 3)))
    )
    return RallyWindow(
        source_id=source_id,
        window_id=f"rally-{source_id}-{anchor:09d}",
        anchor_frame=anchor,
        frame_indices=frames,
        owned_frames=frozenset(frames),
        relative_time_seconds=torch.zeros(len(frames)),
        frame_values=torch.zeros(len(frames), 4),
        frame_validity=torch.ones(len(frames), 4, dtype=torch.bool),
        inplay_targets=torch.zeros(len(frames), dtype=torch.long),
        rally_start_targets=torch.zeros(len(frames)),
        rally_end_targets=torch.zeros(len(frames)),
        metadata={
            "fps": 30.0,
            "frame_count": frame_count,
            "owns_rally_boundary": owns_boundary,
            "owns_short_rally": False,
        },
    )


def test_source_train_stop_uses_global_cut() -> None:
    split = {
        "cameras": {
            "A": {
                "source_order": ["a1", "a2"],
                "source_frame_counts": {"a1": 100, "a2": 50},
                "source_offsets": {"a1": 0, "a2": 100},
                "train_global_range": [0, 129],
            }
        }
    }
    stops = source_train_stop(split, {"A": ("a1", "a2")})
    assert stops == {"a1": 100, "a2": 30}


def test_source_train_stop_rejects_mixed_fps() -> None:
    split = {"cameras": {"A": {"source_fps": {"a1": 30.0}}}}
    with pytest.raises(ValueError, match="uniform-fps"):
        source_train_stop(split, {"A": ("a1",)})


def test_variant_train_windows_stay_inside_parent_train_range() -> None:
    windows = [
        _window("parent__hflip", frames[0], frames=frames)
        for frames in ((8, 9, 10, 11, 12), (38, 39, 40, 41, 42), (78, 79, 80, 81, 82), (98, 99, 100, 101))
    ]
    selected = variant_train_windows(
        windows,
        variant_parents={"parent__hflip": "parent"},
        train_sources={"parent"},
        stops={"parent": 100},
    )
    assert [window.anchor_frame for window in selected] == [8, 38, 78]


def test_variant_train_windows_requires_training_parent() -> None:
    windows = [_window("parent__hflip", 10)]
    assert (
        variant_train_windows(
            windows,
            variant_parents={"parent__hflip": "parent"},
            train_sources={"other"},
            stops={"parent": 100},
        )
        == []
    )


def _paired_windows() -> tuple[list[RallyWindow], list[RallyWindow], dict[str, str]]:
    common = [_window("a", anchor) for anchor in (10, 30, 50, 70, 90)]
    common += [_window("b", anchor) for anchor in (10, 30, 50, 70, 90)]
    variants = [_window("a__v", anchor) for anchor in (10, 30, 50, 70, 90)]
    variants += [_window("b__v", anchor) for anchor in (10, 30, 50, 70, 90)]
    return common, variants, {"a__v": "a", "b__v": "b"}


def _batches(sampler: FootageAugmentationSampler) -> list[list[RallyWindow]]:
    result = []
    for batch in sampler:
        result.append([sampler.windows[index] for index in batch])
    return result


def test_sampler_pairs_common_draws_and_substitutes_variants() -> None:
    common, variants, variant_parents = _paired_windows()
    baseline = FootageAugmentationSampler(
        common,
        steps=4,
        batch_size=8,
        seed=1729,
        original_windows=common,
        variant_parents=variant_parents,
        variant_share=0.5,
        include_variants=False,
    )
    augmented = FootageAugmentationSampler(
        [*common, *variants],
        steps=4,
        batch_size=8,
        seed=1729,
        original_windows=common,
        variant_parents=variant_parents,
        variant_share=0.5,
        include_variants=True,
    )
    baseline_batches = _batches(baseline)
    augmented_batches = _batches(augmented)
    assert len(baseline_batches) == len(augmented_batches) == 4
    substituted = 0
    for baseline_batch, augmented_batch in zip(baseline_batches, augmented_batches):
        for base_window, aug_window in zip(baseline_batch, augmented_batch):
            if aug_window.source_id == base_window.source_id:
                assert aug_window.window_id == base_window.window_id
            else:
                assert aug_window.source_id == f"{base_window.source_id}__v"
                assert aug_window.anchor_frame == base_window.anchor_frame
                substituted += 1
    assert substituted > 0
    assert baseline.last_sampling_stats["realized_source_counts"].keys() == {"a", "b"}
    augmented_counts = augmented.last_sampling_stats["realized_source_counts"]
    assert augmented_counts["a__v"] > 0 and augmented_counts["b__v"] > 0


def test_sampler_requires_available_variants() -> None:
    common, variants, variant_parents = _paired_windows()
    with pytest.raises(ValueError, match="no sampled variants"):
        FootageAugmentationSampler(
            common,
            steps=1,
            batch_size=2,
            seed=1729,
            original_windows=common,
            variant_parents=variant_parents,
            variant_share=0.5,
            include_variants=True,
        )
