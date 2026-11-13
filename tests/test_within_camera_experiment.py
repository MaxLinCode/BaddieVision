from __future__ import annotations

from dataclasses import replace

import torch
import pytest

from src.temporal_selector.dataset import (
    SelectorWindow,
    _frame_context,
    rally_boundary_targets,
)
from src.temporal_selector.experiment import _TokenBucketBatchSampler
from src.temporal_selector.within_camera_experiment import (
    _parser,
    chronological_window_split,
)
from src.temporal_selector.within_camera_decoder import (
    build_rally_diagnostics,
    calibrate_validation_decoder,
    contiguous_row_groups,
    decode_rows,
)
from src.temporal_selector.rally_intervals import RallyInterval
from src.temporal_selector.rally_video_diagnostics import selection_summary
from src.temporal_selector.partial_oracle_experiment import oracle_candidate_windows
from src.annotation_platform.events import AnnotationEvent


def _window(source: str, frame: int, *, fps: float = 10.0) -> SelectorWindow:
    return SelectorWindow(
        source_id=source,
        burst_id=f"dense-{source}-{frame}",
        queue_kind="dense",
        anchor_frame=frame,
        frame_indices=(frame,),
        owned_frames=frozenset({frame}),
        relative_time_seconds=torch.tensor([0.0]),
        candidate_values=torch.empty(0, 12),
        candidate_validity=torch.empty(0, 12, dtype=torch.bool),
        candidate_frame_indices=torch.empty(0, dtype=torch.long),
        candidate_ids=(),
        frame_values=torch.zeros(1, 154),
        frame_validity=torch.ones(1, 154, dtype=torch.bool),
        targets=torch.tensor([-100]),
        target_status=("out_of_play",),
        inplay_targets=torch.tensor([0]),
        metadata={"fps": fps},
    )


def test_chronological_split_preserves_source_frames_and_guard_bands() -> None:
    windows = [
        *(_window("early", frame) for frame in range(100)),
        *(_window("late", frame) for frame in range(100)),
    ]
    train, validation, test, metadata = chronological_window_split(
        windows, ("early", "late"), guard_seconds=1.0
    )
    assert metadata["source_offsets"] == {"early": 0, "late": 100}
    assert metadata["global_ranges"] == {
        "train": [0, 109], "validation": [130, 149], "test": [170, 199]
    }
    assert {(item.source_id, item.anchor_frame) for item in train} == {
        *{("early", frame) for frame in range(100)},
        *{("late", frame) for frame in range(10)},
    }
    assert [item.anchor_frame for item in validation] == list(range(30, 50))
    assert [item.anchor_frame for item in test] == list(range(70, 100))


def test_boundary_targets_taper_mask_context_and_suppress_partial_start() -> None:
    starts, ends = rally_boundary_targets(
        (
            RallyInterval("camera", "partial", 0, 2),
            RallyInterval("camera", "short", 4, 5),
        ),
        frozenset({"partial"}),
        tuple(range(7)),
        frozenset(range(6)),
        (1, 1, 1, 0, 1, 1, -100),
        fps=4,
    )
    assert starts.tolist() == [0, 0, 0, 0, 1, 0, -100]
    assert ends.tolist() == [0, 0, 1, 0, 0, 1, -100]


def test_boundary_balanced_sampler_is_deterministic_and_length_preserving() -> None:
    windows = [
        *(_window("source", frame) for frame in range(8)),
        *(
            replace(
                _window("source", 8 + frame),
                metadata={
                    "fps": 10.0,
                    "owns_rally_boundary": frame < 2,
                    "owns_short_rally": frame >= 2,
                },
            )
            for frame in range(4)
        ),
    ]
    samplers = [
        _TokenBucketBatchSampler(
            windows,
            batch_size=3,
            generator=torch.Generator().manual_seed(1729),
            sampling_mode="boundary_balanced",
        )
        for _ in range(2)
    ]
    draws = [list(sampler) for sampler in samplers]
    assert draws[0] == draws[1]
    assert sum(map(len, draws[0])) == len(windows)
    assert samplers[0].last_sampling_stats == samplers[1].last_sampling_stats
    assert samplers[0].last_sampling_stats["unique_sampled_window_count"] <= len(windows)


def test_within_camera_cli_can_explicitly_defer_test_evaluation() -> None:
    args = _parser().parse_args(
        [
            "--config", "config.json",
            "--output-dir", "run",
            "--defer-test",
        ]
    )
    assert args.defer_test is True


def _context(pose_bbox: list[float], mode: str) -> list[float]:
    landmarks = [
        {"x": 0.2 + (index % 5) * 0.1, "y": 0.1 + (index % 7) * 0.1, "visibility": 1.0}
        for index in range(33)
    ]
    assignment = {
        "slots": {
            "P1": {
                "confidence": 1.0,
                "assignment": {
                    "track_id": 1,
                    "bbox": pose_bbox,
                    "foot": [pose_bbox[2], pose_bbox[3]],
                    "court_x": 0.0,
                    "court_y": 0.0,
                    "detection_confidence": 1.0,
                    "activity": 0.5,
                },
            }
        }
    }
    poses = {(0, 1): {"status": "detected", "pose_bbox": pose_bbox, "pose_landmarks": landmarks}}
    values, _ = _frame_context(
        assignment, poses, 0, (1000, 800), "full_context", 0.5, mode
    )
    return values[22:88]


def test_player_relative_pose_is_translation_and_scale_invariant() -> None:
    first_relative = _context([100, 100, 200, 300], "player_relative")
    second_relative = _context([300, 200, 500, 600], "player_relative")
    assert second_relative == pytest.approx(first_relative)
    assert _context([100, 100, 200, 300], "image") != _context(
        [300, 200, 500, 600], "image"
    )


def _prediction_row(source: str, frame: int, probability: float, target: int) -> dict:
    return {
        "prediction_source_id": source,
        "frame": frame,
        "in_play_probability": probability,
        "inplay_target": target,
        "true_status": "out_of_play",
        "selection_loss": None,
        "inplay_loss": None,
        "candidate_count": 0,
    }


def test_decoder_groups_do_not_cross_sources_or_missing_frames() -> None:
    rows = [
        _prediction_row("a", 0, 0.9, 1),
        _prediction_row("a", 1, 0.1, 0),
        _prediction_row("a", 3, 0.9, 1),
        _prediction_row("b", 0, 0.9, 1),
    ]
    groups = contiguous_row_groups(reversed(rows))
    assert [[int(row["frame"]) for row in group] for group in groups] == [
        [0, 1],
        [3],
        [0],
    ]


def test_validation_calibration_removes_fragmentation() -> None:
    rows = []
    for frame in range(100):
        target = int(10 <= frame <= 39 or 60 <= frame <= 89)
        probability = 0.8 if target else 0.05
        if frame in (20, 21):
            probability = 0.05
        if frame in (47, 48):
            probability = 0.8
        rows.append(_prediction_row("camera", frame, probability, target))
    config, candidates = calibrate_validation_decoder(rows, fps=30)
    decoded = decode_rows(rows, fps=30, config=config)
    transitions = sum(
        bool(left["decoded_inplay"]) != bool(right["decoded_inplay"])
        for left, right in zip(decoded, decoded[1:])
    )
    assert candidates[0]["metrics"]["interval_f1"] == 1.0
    assert transitions == 4


def test_rally_diagnostics_measure_internal_gap_and_split() -> None:
    rows = []
    for frame in range(20):
        row = _prediction_row("camera", frame, 0.8, int(4 <= frame <= 15))
        row["frame_threshold_inplay"] = frame not in (9, 10)
        row["decoded_inplay"] = 4 <= frame <= 8 or 11 <= frame <= 15
        if frame in (9, 10):
            row["in_play_probability"] = 0.05
        rows.append(row)
    diagnostics = build_rally_diagnostics(
        rows,
        [RallyInterval("camera", "rally-1", 4, 15)],
        threshold=0.1,
    )
    assert diagnostics[0]["longest_internal_low_gap_frames"] == 2
    assert diagnostics[0]["overlapping_prediction_count"] == 2
    assert "split" in diagnostics[0]["status"]


def test_selection_summary_includes_null_in_score_and_margin() -> None:
    outcome, score, margin = selection_summary(
        {
            "candidate_logits": {"candidate-a": 2.0, "candidate-b": 0.0},
            "null_logit": 1.0,
        }
    )
    probabilities = torch.softmax(torch.tensor([2.0, 0.0, 1.0]), dim=0)
    assert outcome == "candidate-a"
    assert score == pytest.approx(float(probabilities[0]))
    assert margin == pytest.approx(float(probabilities[0] - probabilities[2]))


def test_selection_summary_can_report_null_as_winner() -> None:
    outcome, score, margin = selection_summary(
        {"candidate_logits": {"candidate-a": -2.0}, "null_logit": 2.0}
    )
    assert outcome == "NULL"
    assert score > 0.98
    assert margin > 0.96


def test_oracle_candidate_windows_keep_only_human_selected_candidate() -> None:
    window = replace(
        _window("camera", 10),
        frame_indices=(10, 11),
        owned_frames=frozenset({10, 11}),
        relative_time_seconds=torch.tensor([0.0, 0.1]),
        candidate_values=torch.arange(36, dtype=torch.float32).reshape(3, 12),
        candidate_validity=torch.ones(3, 12, dtype=torch.bool),
        candidate_frame_indices=torch.tensor([0, 0, 1]),
        candidate_ids=("wrong", "true", "distractor"),
        frame_values=torch.zeros(2, 154),
        frame_validity=torch.ones(2, 154, dtype=torch.bool),
        targets=torch.tensor([1, -100]),
        target_status=("selected_retained", "out_of_play"),
        inplay_targets=torch.tensor([1, 0]),
    )

    def event(frame: int, kind: str, candidate_id: str | None) -> AnnotationEvent:
        return AnnotationEvent(
            revision_id=f"revision-{frame}",
            task="shuttle_selection",
            source_id="camera",
            frame=frame,
            label_kind=kind,
            candidate_id=candidate_id,
            candidate_artifact_sha256="a" * 64,
            source_video_sha256="b" * 64,
            annotator="max",
            session_id="session",
            timestamp="2026-07-24T00:00:00Z",
            superseded_revision=None,
        )

    [oracle] = oracle_candidate_windows(
        [window],
        {
            ("camera", 10): event(10, "selected", "true"),
            ("camera", 11): event(11, "missing_proposal", None),
        },
    )
    assert oracle.candidate_ids == ("true",)
    assert oracle.candidate_frame_indices.tolist() == [0]
    assert oracle.targets.tolist() == [-100, -100]
