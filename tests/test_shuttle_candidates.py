import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.single_video.shuttle import (
    ShuttleCandidateCollector,
    ShuttleHypothesisConfig,
    ShuttleLinkConfig,
    link_shuttle_candidates,
    link_shuttle_hypotheses,
)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_hypothesis_inputs(tmp_path: Path, tracklets: list[tuple[str, list[tuple[int, float, float, float]]]], *, fps: float = 10) -> tuple[Path, Path]:
    """Create compact, hand-authored immutable layers for decoder tests."""
    candidates_path, tracklets_path = tmp_path / "candidates.jsonl", tmp_path / "tracklets.jsonl"
    frames: dict[int, list[dict]] = {}
    tracklet_records = []
    for tracklet_id, points in tracklets:
        candidate_ids, tracklet_frames = [], []
        for index, (frame, x, y, peak) in enumerate(points):
            candidate_id = f"{tracklet_id}-c{index}"
            candidate_ids.append(candidate_id)
            tracklet_frames.append(frame)
            frames.setdefault(frame, []).append({"candidate_id": candidate_id, "center": [x, y], "peak_value": peak})
        tracklet_records.append({"type": "tracklet", "tracklet_id": tracklet_id, "candidate_ids": candidate_ids, "frames": tracklet_frames})
    metadata = {"type": "metadata", "schema": "shuttle_candidates", "schema_version": 1, "fps": fps, "image_size": [100, 100]}
    with candidates_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n")
        for frame in sorted(frames):
            handle.write(json.dumps({"type": "frame", "frame": frame, "candidates": frames[frame]}, sort_keys=True, separators=(",", ":")) + "\n")
    tracklet_metadata = {
        "type": "metadata", "schema": "shuttle_tracklets", "schema_version": 1,
        "candidate_artifact": candidates_path.name, "candidate_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
    }
    with tracklets_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(tracklet_metadata, sort_keys=True, separators=(",", ":")) + "\n")
        for record in tracklet_records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return candidates_path, tracklets_path


def test_candidate_collection_keeps_all_components_and_empty_frames(tmp_path: Path) -> None:
    collector = ShuttleCandidateCollector(image_size=(40, 20), heatmap_size=(4, 8), fps=25, threshold=0.5)
    heatmap = np.zeros((4, 8), dtype=np.float32)
    heatmap[0, 0:2] = [0.7, 0.9]
    heatmap[2:4, 4:7] = 0.6
    collector.add(2, heatmap)
    collector.add(3, np.zeros((4, 8), dtype=np.float32))
    output = collector.write(tmp_path / "candidates.jsonl")

    metadata, frame, empty = _records(output)
    assert metadata["schema"] == "shuttle_candidates"
    assert metadata["model_stage"] == "tracknet_pre_inpaint"
    assert metadata["heatmap_size"] == [8, 4]
    assert frame["frame"] == 2
    assert [candidate["candidate_id"] for candidate in frame["candidates"]] == ["f000002-c000", "f000002-c001"]
    assert frame["candidates"][0]["peak_value"] == pytest.approx(0.9)
    assert frame["candidates"][0]["center"] == [5.0, 2.5]
    assert frame["candidates"][1]["legacy_largest_component"] is True
    assert empty == {"type": "frame", "frame": 3, "candidates": []}


def test_linker_is_replayable_and_covers_every_candidate_once(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.jsonl"
    collector = ShuttleCandidateCollector(image_size=(100, 100), heatmap_size=(10, 10), fps=30)
    for frame, x in ((0, 1), (1, 2), (3, 4)):
        heatmap = np.zeros((10, 10), dtype=np.float32)
        heatmap[5, x] = 0.9
        collector.add(frame, heatmap)
    collector.add(2, np.zeros((10, 10), dtype=np.float32))
    collector.write(candidate_path)
    first, second = tmp_path / "tracklets-a.jsonl", tmp_path / "tracklets-b.jsonl"
    link_shuttle_candidates(candidate_path, first)
    link_shuttle_candidates(candidate_path, second)

    assert first.read_bytes() == second.read_bytes()
    metadata, *tracklets = _records(first)
    assert metadata["candidate_sha256"] == hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    assert metadata["link_config"]["max_missing_frames"] == 1
    candidate_ids = [candidate["candidate_id"] for record in _records(candidate_path)[1:] for candidate in record["candidates"]]
    linked_ids = [candidate_id for tracklet in tracklets for candidate_id in tracklet["candidate_ids"]]
    assert linked_ids == candidate_ids
    assert len(tracklets) == 1
    assert tracklets[0]["frames"] == [0, 1, 3]


def test_linker_splits_ambiguous_and_motion_gated_candidates(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.jsonl"
    collector = ShuttleCandidateCollector(image_size=(100, 100), heatmap_size=(10, 10), fps=30)
    for frame, positions in ((0, [(1, 5)]), (1, [(2, 4), (2, 6)]), (2, [(9, 9)])):
        heatmap = np.zeros((10, 10), dtype=np.float32)
        for x, y in positions:
            heatmap[y, x] = 0.9
        collector.add(frame, heatmap)
    collector.write(candidate_path)
    tracklet_path = tmp_path / "tracklets.jsonl"
    link_shuttle_candidates(
        candidate_path, tracklet_path,
        ShuttleLinkConfig(max_speed_image_diagonals_per_second=0.1),
    )
    _, *tracklets = _records(tracklet_path)
    assert all(len(tracklet["candidate_ids"]) == 1 for tracklet in tracklets)
    assert len(tracklets) == 4


def test_all_single_video_notebooks_bundle_layered_shuttle_artifacts() -> None:
    notebook_dir = Path(__file__).parents[1] / "notebooks"
    for variant in ("Local", "Colab", "Kaggle"):
        notebook = json.loads((notebook_dir / f"Single_Video_Feature_Extraction_{variant}.ipynb").read_text(encoding="utf-8"))
        source = "".join(line for cell in notebook["cells"] for line in cell.get("source", []))
        assert "shuttle_candidates.jsonl" in source
        assert "shuttle_tracklets.jsonl" in source
        assert "shuttle_hypotheses.jsonl" in source
        assert "candidate_output_path=str(candidates_path)" in source
        assert "link_shuttle_candidates(candidates_path, tracklets_path)" in source
        assert "link_shuttle_hypotheses(candidates_path, tracklets_path, hypotheses_path)" in source


def test_hypothesis_decoder_is_replayable_validates_inputs_and_preserves_raw_layers(tmp_path: Path) -> None:
    candidates, tracklets = _write_hypothesis_inputs(tmp_path, [("t0", [(0, 10, 10, 0.8)]), ("t1", [(1, 10, 10, 0.9)])])
    before = (candidates.read_bytes(), tracklets.read_bytes())
    first, second = tmp_path / "hypotheses-a.jsonl", tmp_path / "hypotheses-b.jsonl"
    link_shuttle_hypotheses(candidates, tracklets, first)
    link_shuttle_hypotheses(candidates, tracklets, second)
    assert first.read_bytes() == second.read_bytes()
    assert (candidates.read_bytes(), tracklets.read_bytes()) == before
    metadata, *records = _records(first)
    assert metadata["schema"] == "shuttle_hypotheses"
    assert metadata["candidate_sha256"] == hashlib.sha256(candidates.read_bytes()).hexdigest()
    assert metadata["tracklet_sha256"] == hashlib.sha256(tracklets.read_bytes()).hexdigest()
    assert records[0]["tracklet_ids"] == ["t0", "t1"]
    broken = json.loads(tracklets.read_text().splitlines()[0])
    broken["candidate_sha256"] = "wrong"
    tracklets.write_text(json.dumps(broken) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact candidate"):
        link_shuttle_hypotheses(candidates, tracklets, tmp_path / "invalid.jsonl")


def test_hypothesis_decoder_uses_legal_fps_scaled_links_and_regions(tmp_path: Path) -> None:
    candidates, tracklets = _write_hypothesis_inputs(tmp_path, [
        ("t0", [(0, 10, 10, 0.9), (1, 10, 10, 0.9)]),
        ("t1", [(6, 10, 10, 0.9)]),  # Four missing frames: 0.4 seconds at 10 FPS.
        ("t2", [(7, 10, 10, 0.9)]),
        ("isolated", [(20, 90, 90, 0.2)]),
        ("overlap", [(1, 100, 100, 0.5)]),
    ])
    output = tmp_path / "hypotheses.jsonl"
    link_shuttle_hypotheses(candidates, tracklets, output, ShuttleHypothesisConfig(max_speed_image_diagonals_per_second=1.0))
    metadata, *records = _records(output)
    regions = {tuple(region["tracklet_ids"]) for region in metadata["association_regions"]}
    assert ("t0", "t1", "t2") in regions
    assert ("isolated",) in regions
    assert ("overlap",) in regions
    assert any(record["tracklet_ids"] == ["t0", "t1", "t2"] for record in records)


def test_hypothesis_decoder_caps_diverse_paths_and_keeps_weak_singletons(tmp_path: Path) -> None:
    tracklets = [("root", [(0, 10, 10, 0.9)])]
    tracklets.extend((f"branch{index}", [(1, 10, 10, 0.95 - index * 0.05)]) for index in range(6))
    tracklets.append(("weak", [(1, 10, 10, 0.1)]))
    candidates, tracklets_path = _write_hypothesis_inputs(tmp_path, tracklets)
    output = tmp_path / "hypotheses.jsonl"
    link_shuttle_hypotheses(candidates, tracklets_path, output)
    _, *records = _records(output)
    main_region = [record for record in records if record["region_id"] == "r0000"]
    assert len(main_region) == 5
    assert all(record["rank"] == index for index, record in enumerate(main_region, start=1))
    assert len({tuple(record["tracklet_ids"]) for record in main_region}) == 5
    rich_output = tmp_path / "hypotheses-rich.jsonl"
    link_shuttle_hypotheses(candidates, tracklets_path, rich_output, ShuttleHypothesisConfig(max_hypotheses_per_region=20))
    _, *rich_records = _records(rich_output)
    weak = next(record for record in rich_records if record["tracklet_ids"] == ["weak"])
    assert weak["rank"] > 1
