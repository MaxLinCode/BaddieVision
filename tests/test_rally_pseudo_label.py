import json
from pathlib import Path

import pytest

from InPlay.heuristic.label_intervals import load_review_draft, write_review_draft
from src.temporal_selector.joint_inference import InPlayDecoderConfig
from src.temporal_selector.rally_pseudo_label import (
    bridge_short_majority_gaps,
    select_snap_configuration,
    snap_intervals,
)


def test_snap_uses_strongest_peak_with_deterministic_ties_and_floor() -> None:
    starts = [0.0] * 20
    ends = [0.0] * 20
    starts[3] = starts[7] = 0.9
    ends[12] = 0.7
    assert snap_intervals([(5, 10)], starts, ends, fps=4, radius_seconds=1,
                          confidence_floor=0.5, minimum_duration_seconds=0.5) == ((3, 12),)
    assert snap_intervals([(5, 10)], starts, ends, fps=4, radius_seconds=.25,
                          confidence_floor=.95, minimum_duration_seconds=.5) == ((5, 10),)


def test_snap_reverts_short_inverted_and_overlapping_pairs() -> None:
    starts = [0.0] * 30
    ends = [0.0] * 30
    ends[12] = starts[11] = 1.0
    assert snap_intervals([(5, 10), (13, 20)], starts, ends, fps=4, radius_seconds=1,
                          confidence_floor=.5, minimum_duration_seconds=.5) == ((5, 10), (13, 20))
    starts[9] = ends[8] = 1.0
    assert snap_intervals([(8, 9)], starts, ends, fps=4, radius_seconds=1,
                          confidence_floor=.5, minimum_duration_seconds=1) == ((8, 9),)


def test_snap_selection_never_receives_heldout_rows() -> None:
    rows = [{"inplay_probability": value, "rally_start_probability": 0.0,
             "rally_end_probability": 0.0} for value in (0.0, 1.0, 1.0, 0.0)]
    calls = []
    selected = select_snap_configuration(rows, fps=4,
        decoder=InPlayDecoderConfig(threshold=.5, minimum_duration_seconds=0),
        score=lambda intervals: calls.append(intervals) or (len(intervals),))
    assert len(calls) == 16
    assert set(selected) == {"radius_seconds", "confidence_floor"}


def test_review_draft_is_fingerprinted_and_bound_to_proposals(tmp_path: Path) -> None:
    path = write_review_draft(tmp_path / "draft.json", source_id="one", annotator="ann",
                              proposal_fingerprint="abc", intervals=[(0, 4), (8, 9)],
                              partial_starts={(0, 4)})
    assert load_review_draft(path, "one", "abc") == ([(0, 4), (8, 9)], {(0, 4)})
    value = json.loads(path.read_text())
    value["intervals"] = [[1, 4]]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fingerprint"):
        load_review_draft(path, "one", "abc")


def test_majority_exit_hold_only_bridges_short_interior_gaps() -> None:
    states = (False, False, True, True, False, False, True, False, False, False, True,
              False, False)
    assert bridge_short_majority_gaps(states, fps=4, max_gap_seconds=.5) == (
        False, False, True, True, True, True, True, False, False, False, True, False, False
    )
    assert bridge_short_majority_gaps(states, fps=4, max_gap_seconds=0) == states
