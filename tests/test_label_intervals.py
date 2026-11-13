from pathlib import Path

import pytest

from InPlay.heuristic.label_intervals import (
    adjacent_boundary,
    adjusted_boundary,
    load_existing_annotations,
    load_existing_labels,
    local_timeline_frame,
    merge_across_cursor,
    nearest_boundary,
    selected_interval,
    timeline_frame,
    write_labels,
)
from src.temporal_selector.rally_intervals import RallySourceManifest


def test_labeler_resumes_saved_source_and_preserves_other_sources(tmp_path: Path) -> None:
    intervals_path = tmp_path / "rallies.csv"
    manifest_path = tmp_path / "rallies.manifest.json"
    source_one = RallySourceManifest("one", "a" * 64, 30.0, 100)
    source_two = RallySourceManifest("two", "b" * 64, 30.0, 200)

    write_labels(
        intervals_path,
        "one",
        [(10, 20)],
        manifest_path=manifest_path,
        source_manifest=source_one,
    )
    write_labels(
        intervals_path,
        "two",
        [(30, 40), (60, 70)],
        manifest_path=manifest_path,
        source_manifest=source_two,
    )

    assert load_existing_labels(
        intervals_path, manifest_path, "two", source_two
    ) == [(30, 40), (60, 70)]
    assert load_existing_labels(
        intervals_path, manifest_path, "one", source_one
    ) == [(10, 20)]


def test_labeler_rejects_resume_with_a_different_video(tmp_path: Path) -> None:
    intervals_path = tmp_path / "rallies.csv"
    manifest_path = tmp_path / "rallies.manifest.json"
    source = RallySourceManifest("one", "a" * 64, 30.0, 100)
    write_labels(
        intervals_path,
        "one",
        [(10, 20)],
        manifest_path=manifest_path,
        source_manifest=source,
    )

    with pytest.raises(ValueError, match="does not match video"):
        load_existing_labels(
            intervals_path,
            manifest_path,
            "one",
            RallySourceManifest("one", "b" * 64, 30.0, 100),
        )


def test_timeline_frame_maps_and_clamps_scrubber_positions() -> None:
    assert timeline_frame(30, width=1030, frame_count=101) == 0
    assert timeline_frame(515, width=1030, frame_count=101) == 50
    assert timeline_frame(1000, width=1030, frame_count=101) == 100
    assert timeline_frame(-100, width=1030, frame_count=101) == 0
    assert timeline_frame(2000, width=1030, frame_count=101) == 100
    assert local_timeline_frame(30, 1030, 500, 1001, 30) == 410
    assert local_timeline_frame(515, 1030, 500, 1001, 30) == 500
    assert local_timeline_frame(1000, 1030, 500, 1001, 30) == 590


def test_boundary_navigation_and_selected_rally_are_deterministic() -> None:
    intervals = [(10, 20), (30, 40)]
    assert selected_interval(10, intervals) == (10, 20)
    assert selected_interval(15, intervals) == (10, 20)
    assert selected_interval(25, intervals) is None
    assert nearest_boundary(25, intervals) == 20
    assert nearest_boundary(28, intervals) == 30
    assert adjacent_boundary(20, intervals, 1) == 30
    assert adjacent_boundary(30, intervals, -1) == 20
    assert adjacent_boundary(40, intervals, 1) is None


def test_boundary_adjustment_extends_trims_and_rejects_missing_target() -> None:
    intervals = [(10, 20), (30, 40)]
    assert adjusted_boundary(intervals, 5, "start") == ((10, 20), (5, 20))
    assert adjusted_boundary(intervals, 25, "end") == ((10, 20), (10, 25))
    assert adjusted_boundary(intervals, 15, "start") == ((10, 20), (15, 20))
    assert adjusted_boundary(intervals, 25, "start") == ((30, 40), (25, 40))
    with pytest.raises(ValueError, match="no rally start"):
        adjusted_boundary(intervals, 50, "start")
    with pytest.raises(ValueError, match="no rally end"):
        adjusted_boundary(intervals, 5, "end")


def test_merge_requires_cursor_in_gap_and_uses_adjacent_rallies() -> None:
    intervals = [(10, 20), (30, 40), (50, 60)]
    assert merge_across_cursor(intervals, 25) == ((10, 20), (30, 40), (10, 40))
    with pytest.raises(ValueError, match="gap"):
        merge_across_cursor(intervals, 35)
    with pytest.raises(ValueError, match="between two"):
        merge_across_cursor(intervals, 5)


def test_partial_start_round_trips_and_other_source_metadata_is_preserved(tmp_path: Path) -> None:
    intervals_path = tmp_path / "rallies.csv"
    manifest_path = tmp_path / "rallies.manifest.json"
    one = RallySourceManifest("one", "a" * 64, 30.0, 100)
    two = RallySourceManifest("two", "b" * 64, 30.0, 100)
    write_labels(
        intervals_path, "one", [(0, 20)], manifest_path=manifest_path,
        source_manifest=one, partial_starts={(0, 20)},
    )
    write_labels(
        intervals_path, "two", [(0, 10)], manifest_path=manifest_path,
        source_manifest=two, partial_starts={(0, 10)},
    )

    assert load_existing_annotations(
        intervals_path, manifest_path, "one", one
    ) == ([(0, 20)], {(0, 20)})
    assert load_existing_annotations(
        intervals_path, manifest_path, "two", two
    ) == ([(0, 10)], {(0, 10)})


def test_partial_start_requires_frame_zero(tmp_path: Path) -> None:
    source = RallySourceManifest("one", "a" * 64, 30.0, 100)
    with pytest.raises(ValueError, match="must begin at frame 0"):
        write_labels(
            tmp_path / "rallies.csv", "one", [(10, 20)],
            manifest_path=tmp_path / "rallies.manifest.json",
            source_manifest=source, partial_starts={(10, 20)},
        )
