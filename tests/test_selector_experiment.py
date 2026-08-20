import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.temporal_selector import MASKED_TARGET, build_two_source_crossfit
from src.temporal_selector.dataset import SelectorWindow, collate_selector_windows
from src.temporal_selector.experiment import (
    _binary_roc_auc,
    _mask_candidate_inputs,
    NULL_SELECTION,
    _TokenBucketBatchSampler,
    _predict_window,
    _predict_windows,
    compile_metrics,
    prepare_output_directory,
    resolve_retained_candidate,
    run_experiment,
    select_device,
    windows_for_queue,
    windows_for_sources,
)


def test_device_selection_warns_when_mps_build_cannot_use_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    with pytest.warns(RuntimeWarning, match="training will use CPU"):
        assert select_device() == torch.device("cpu")


def _prediction(status, predicted, *, target=None, loss=None):
    return {
        "true_status": status,
        "predicted_outcome": predicted,
        "predicted_candidate_id": None if predicted == NULL_SELECTION else predicted,
        "target_candidate_id": target,
        "selection_loss": loss,
    }


def test_metrics_cover_selection_null_scoring_and_zero_denominators():
    metrics = compile_metrics(
        [
            _prediction("selected_retained", "a", target="a", loss=0.2),
            _prediction("selected_retained", NULL_SELECTION, target="b", loss=0.4),
            _prediction("null", NULL_SELECTION, loss=0.6),
            _prediction("null", "c", loss=0.8),
            _prediction("missing_proposal", "c"),
            _prediction("dropped_by_k", NULL_SELECTION),
        ]
    )
    assert metrics["mean_selection_loss"] == pytest.approx(0.5)
    assert metrics["proposal_coverage"] == {
        "value": 0.5,
        "selected_retained": 2,
        "missing_proposal": 1,
        "dropped_by_k": 1,
        "denominator": 4,
    }
    assert metrics["retained_target_accuracy"]["value"] == 0.5
    assert metrics["overall_accuracy"]["value"] == 0.5
    assert metrics["null_selection"]["precision"] == 0.5
    assert metrics["null_selection"]["recall"] == 0.5
    assert metrics["null_selection"]["f1"] == 0.5

    empty = compile_metrics([])
    assert empty["mean_selection_loss"] is None
    assert empty["proposal_coverage"]["value"] is None
    assert empty["retained_target_accuracy"]["value"] is None
    assert empty["overall_accuracy"]["value"] is None
    assert empty["null_selection"]["precision"] is None
    assert empty["null_selection"]["recall"] is None
    assert empty["null_selection"]["f1"] is None


def test_binary_roc_auc_is_tie_aware_and_handles_single_class():
    assert _binary_roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert _binary_roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert _binary_roc_auc([1, 1], [0.2, 0.8]) is None


def _window(source, queue, target, status, candidate_id):
    return SelectorWindow(
        source_id=source,
        burst_id=f"{queue}-{source}",
        queue_kind=queue,
        anchor_frame=10,
        frame_indices=(10,),
        owned_frames=frozenset({10}),
        relative_time_seconds=torch.tensor([0.0]),
        candidate_values=torch.zeros(1, 12),
        candidate_validity=torch.ones(1, 12, dtype=torch.bool),
        candidate_frame_indices=torch.tensor([0]),
        candidate_ids=(candidate_id,),
        frame_values=torch.empty(1, 0),
        frame_validity=torch.empty(1, 0, dtype=torch.bool),
        targets=torch.tensor([target]),
        target_status=(status,),
    )


def test_source_queue_filtering_and_retained_candidate_resolution():
    windows = [
        _window("malaysia", "adaptive", 0, "selected_retained", "m"),
        _window("max-vs-nik", "audit", -1, "null", "n"),
    ]
    assert windows_for_sources(windows, ("malaysia",)) == [windows[0]]
    assert windows_for_queue(windows, "audit") == [windows[1]]
    assert resolve_retained_candidate(windows[0], 0, 0) == "m"
    with pytest.raises(ValueError, match="does not resolve"):
        resolve_retained_candidate(windows[0], 0, 1)


def test_token_bucket_sampler_is_deterministic_and_groups_similar_sizes():
    windows = []
    for count in range(1, 9):
        base = _window("source", "dense", 0, "selected_retained", "c0")
        windows.append(
            replace(
                base,
                candidate_values=torch.zeros(count, 12),
                candidate_validity=torch.ones(count, 12, dtype=torch.bool),
                candidate_frame_indices=torch.zeros(count, dtype=torch.long),
                candidate_ids=tuple(f"c{slot}" for slot in range(count)),
            )
        )
    first = list(
        _TokenBucketBatchSampler(
            windows, batch_size=2, generator=torch.Generator().manual_seed(9)
        )
    )
    second = list(
        _TokenBucketBatchSampler(
            windows, batch_size=2, generator=torch.Generator().manual_seed(9)
        )
    )
    assert first == second
    assert sorted(index for batch in first for index in batch) == list(range(8))
    assert all(abs(len(windows[a].candidate_ids) - len(windows[b].candidate_ids)) <= 1 for a, b in first)
    collated = collate_selector_windows((windows[0], windows[-1]))
    assert collated.packed_token_count == 9
    assert collated.packed_frame_indices.tolist() == [[0], [0]]
    assert collated.packed_candidate_indices[1, :8].tolist() == list(range(1, 9))
    assert collated.candidate_local_indices[1, :8].tolist() == list(range(8))


def test_batched_prediction_matches_single_window_prediction():
    windows = [
        _window("malaysia", "adaptive", 0, "selected_retained", "m"),
        _window("malaysia", "audit", -1, "null", "n"),
    ]
    fold = build_two_source_crossfit(("malaysia", "max-vs-nik"), seed=1729).folds[1]
    from src.temporal_selector.model import TemporalShuttleSelector

    model = TemporalShuttleSelector().eval()
    expected = [
        row
        for window in windows
        for row in _predict_window(
            model, window, fold, device=torch.device("cpu")
        )
    ]
    actual = _predict_windows(
        model,
        windows,
        fold,
        device=torch.device("cpu"),
        batch_size=2,
    )
    numeric_fields = {"null_logit", "selection_loss", "inplay_logit", "in_play_probability", "inplay_loss"}
    for batched, single in zip(actual, expected):
        assert {
            key: value for key, value in batched.items() if key not in numeric_fields | {"candidate_logits"}
        } == {
            key: value for key, value in single.items() if key not in numeric_fields | {"candidate_logits"}
        }
        assert batched["candidate_logits"] == pytest.approx(single["candidate_logits"])
        for field in numeric_fields:
            if single[field] is None:
                assert batched[field] is None
            else:
                assert batched[field] == pytest.approx(single[field], abs=1e-6)


def test_no_shuttle_input_mask_removes_candidates_from_model_and_predictions():
    window = _window("malaysia", "adaptive", 0, "selected_retained", "m")
    fold = build_two_source_crossfit(("malaysia", "max-vs-nik"), seed=1729).folds[1]
    from src.temporal_selector.model import TemporalShuttleSelector

    batch = _mask_candidate_inputs(collate_selector_windows([window]))
    assert not bool(batch.candidate_mask.any())
    assert bool((batch.candidate_frame_indices == -1).all())
    assert bool((batch.targets == MASKED_TARGET).all())

    rows = _predict_windows(
        TemporalShuttleSelector().eval(),
        [window],
        fold,
        device=torch.device("cpu"),
        batch_size=1,
        mask_candidates=True,
    )
    assert rows[0]["candidate_inputs_masked"] is True
    assert rows[0]["candidate_count"] == 0
    assert rows[0]["candidate_ids"] == []
    assert rows[0]["candidate_logits"] == {}
    assert rows[0]["selection_loss"] is None
    assert rows[0]["true_status"] == "candidate_inputs_masked"
    assert rows[0]["original_true_status"] == "selected_retained"
def test_output_directory_refuses_nonempty_path(tmp_path: Path):
    output = prepare_output_directory(tmp_path / "run")
    (output / "existing").write_text("owned")
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_output_directory(output)


class _SyntheticDataset:
    def __init__(self):
        self.windows = tuple(
            _window(source, queue, target, status, f"{source}-{queue}")
            for source in ("malaysia", "max-vs-nik")
            for queue, target, status in (
                ("adaptive", 0, "selected_retained"),
                ("audit", -1, "null"),
            )
        )
        self.manifest = {"dataset_fingerprint": "synthetic-fingerprint"}


def test_lightweight_runner_writes_crossfit_outputs(tmp_path: Path):
    manifest = build_two_source_crossfit(("malaysia", "max-vs-nik"), seed=1729)
    metrics = run_experiment(
        _SyntheticDataset(),
        manifest,
        tmp_path / "run",
        epochs=1,
        device=torch.device("cpu"),
    )
    output = tmp_path / "run"
    assert (output / "fold-A.pt").is_file()
    assert (output / "fold-B.pt").is_file()
    assert (output / "metrics.json").is_file()
    predictions = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    assert len(predictions) == 4
    assert {row["label_queue"] for row in predictions} == {"adaptive", "audit"}
    assert all(
        row["prediction_source_id"] not in row["training_source_ids"]
        for row in predictions
    )
    assert metrics["two_fold_macro"]["fold_count"] == 2
    with pytest.raises(FileExistsError, match="not empty"):
        run_experiment(
            _SyntheticDataset(),
            manifest,
            output,
            epochs=1,
            device=torch.device("cpu"),
        )
