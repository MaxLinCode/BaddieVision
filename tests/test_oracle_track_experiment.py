from __future__ import annotations

from dataclasses import replace

import torch

from src.annotation_platform.events import AnnotationEvent
from src.temporal_selector.dataset import SelectorWindow
from src.temporal_selector.oracle_track_experiment import (
    corrupt_oracle_track_windows,
    oracle_track_windows,
    shuffled_coordinate_map,
    windows_fingerprint,
)


def _window(frames: tuple[int, ...] = (10, 11, 12, 13)) -> SelectorWindow:
    count = len(frames)
    return SelectorWindow(
        source_id="camera", burst_id="burst", queue_kind="dense",
        anchor_frame=frames[0], frame_indices=frames, owned_frames=frozenset(frames),
        relative_time_seconds=torch.arange(count) / 10,
        candidate_values=torch.zeros(count, 12),
        candidate_validity=torch.zeros(count, 12, dtype=torch.bool),
        candidate_frame_indices=torch.arange(count),
        candidate_ids=tuple(f"raw-{frame}" for frame in frames),
        frame_values=torch.zeros(count, 154),
        frame_validity=torch.ones(count, 154, dtype=torch.bool),
        targets=torch.full((count,), -100), target_status=("x",) * count,
        inplay_targets=torch.tensor([0, 1, 1, 0][:count]), metadata={"fps": 10.0},
    )


def _event(frame: int, kind: str, xy: tuple[float, float] | None) -> AnnotationEvent:
    position = None if xy is None else {
        "coordinate_space": "normalized_image_xy",
        "canonical_field": "peak_position_normalized",
        "peak_position_normalized": list(xy),
        "weighted_centroid_normalized": list(xy),
        "center_normalized": list(xy),
    }
    return AnnotationEvent(
        revision_id=f"r-{frame}", task="shuttle_selection", source_id="camera",
        frame=frame, label_kind=kind, candidate_id=f"raw-{frame}" if kind == "selected" else None,
        candidate_artifact_sha256="a" * 64, source_video_sha256="b" * 64,
        annotator="a", session_id="s", timestamp="2026-01-01T00:00:00Z",
        superseded_revision=None, candidate_position=position,
    )


def _events() -> dict[tuple[str, int], AnnotationEvent]:
    return {
        ("camera", 10): _event(10, "selected", (0.2, 0.3)),
        ("camera", 11): _event(11, "missing_proposal", (0.4, 0.5)),
        ("camera", 12): _event(12, "occluded_inferable", None),
        ("camera", 13): _event(13, "no_in_frame_target", None),
    }


def test_selected_and_coordinate_bearing_missing_proposal_share_treatment() -> None:
    [result] = oracle_track_windows([_window()], _events(), variant="corrected_coordinates")
    assert torch.allclose(
        result.candidate_values[:2, :2], torch.tensor([[0.2, 0.3], [0.4, 0.5]])
    )
    assert result.candidate_validity[:2, :2].all()
    assert not result.candidate_validity[2:, :].any()
    assert not result.candidate_values[2:, :].any()


def test_every_variant_has_one_token_per_frame_and_controls_zero_coordinates() -> None:
    for variant in (
        "neutral_track", "observation_only", "corrected_coordinates",
        "shuffled_coordinates", "legacy_selected_only_mask",
    ):
        [result] = oracle_track_windows([_window()], _events(), variant=variant, shuffle_seed=7)
        assert len(result.candidate_ids) == len(result.frame_indices)
        assert result.candidate_frame_indices.tolist() == list(range(len(result.frame_indices)))
        if variant in {"neutral_track", "observation_only"}:
            assert not result.candidate_values.any()
    [observation] = oracle_track_windows([_window()], _events(), variant="observation_only")
    assert observation.candidate_validity[:, :2].all(dim=1).tolist() == [True, True, False, False]


def test_shuffle_is_deterministic_and_preserves_coordinates_and_missingness() -> None:
    coordinates = {("c", frame): (frame / 10, frame / 20) for frame in range(5)}
    first = shuffled_coordinate_map(coordinates, 42)
    assert first == shuffled_coordinate_map(coordinates, 42)
    assert set(first) == set(coordinates)
    assert sorted(first.values()) == sorted(coordinates.values())


def test_corruption_is_deterministic_noop_safe_and_does_not_mutate_source() -> None:
    base = oracle_track_windows([_window()], _events(), variant="corrected_coordinates")
    before = windows_fingerprint(base)
    noop = corrupt_oracle_track_windows(base, family="gap", seed=3, amount=0, gap_lengths=(1, 3))
    assert windows_fingerprint(noop) == before
    first = corrupt_oracle_track_windows(base, family="gap", seed=3, amount=0.5, gap_lengths=(1, 1))
    second = corrupt_oracle_track_windows(base, family="gap", seed=3, amount=0.5, gap_lengths=(1, 1))
    assert windows_fingerprint(first) == windows_fingerprint(second)
    assert windows_fingerprint(base) == before
    # Half of the two observed frames is exactly one dropped frame.
    assert int(first[0].candidate_validity[:, :2].all(dim=1).sum()) == 1


def test_lag_has_exact_frame_offset_and_distractors_record_provenance() -> None:
    events = {("camera", frame): _event(frame, "selected", (frame / 20, frame / 25)) for frame in range(10, 14)}
    base = oracle_track_windows([_window()], events, variant="corrected_coordinates")
    [lagged] = corrupt_oracle_track_windows(base, family="lag", seed=1, amount=1)
    assert not lagged.candidate_validity[0, :2].any()
    assert torch.equal(lagged.candidate_values[1:, :2], base[0].candidate_values[:-1, :2])

    raw = _window()
    raw_values = raw.candidate_values.clone()
    raw_validity = raw.candidate_validity.clone()
    raw_values[:, 2:4] = torch.tensor([[0.9, 0.8]] * 4)
    raw_validity[:, 2:4] = True
    raw = replace(raw, candidate_values=raw_values, candidate_validity=raw_validity)
    [switched] = corrupt_oracle_track_windows(base, family="distractor", seed=9, amount=1.0, raw_windows=[raw])
    assert switched.metadata["oracle_track_corruption"]["affected_frame_count"] == 4
    assert len(switched.metadata["oracle_track_corruption"]["candidate_ids"]) == 4
    assert torch.allclose(
        switched.candidate_values[:, :2], torch.tensor([[0.9, 0.8]] * 4)
    )
