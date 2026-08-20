import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.single_video.artifacts import _load_rank_one_hypothesis_segments, _load_shuttle_evidence
from src.single_video.shuttle import (
    CANDIDATE_RETENTION_POLICY,
    ShuttleCandidateCollector,
    ShuttleHypothesisConfig,
    ShuttleLinkConfig,
    evaluate_candidate_retention_recall,
    link_shuttle_candidates,
    link_shuttle_hypotheses,
    legacy_tracknet_bbox,
    rank_shuttle_candidates,
    read_shuttle_candidates,
    tracknet_candidate_frame_range,
)


def _untouched_tracknet_bbox(heatmap):
    import cv2

    mask = (heatmap > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    rects = [cv2.boundingRect(contour) for contour in contours]
    best, max_area = 0, rects[0][2] * rects[0][3]
    for index in range(1, len(rects)):
        area = rects[index][2] * rects[index][3]
        if area > max_area:
            best, max_area = index, area
    return rects[best]


@pytest.mark.parametrize("heatmap", [
    np.zeros((8, 8), dtype=np.float32),
    np.pad(np.ones((2, 3), dtype=np.float32), ((2, 4), (1, 4))),
    np.diag(np.ones(8, dtype=np.float32)),
    np.array([[0, 0, 0, 0, 0], [0, .9, 0, .9, 0], [0, .9, 0, .9, 0]], dtype=np.float32),
])
def test_legacy_suggestion_bbox_matches_untouched_tracknet(heatmap) -> None:
    assert legacy_tracknet_bbox(heatmap) == _untouched_tracknet_bbox(heatmap)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _collector(**kwargs) -> ShuttleCandidateCollector:
    kwargs.setdefault("checkpoint_sha256", "a" * 64)
    kwargs.setdefault("inference_model_sha256", "b" * 64)
    kwargs.setdefault("inference_model_artifact", "TrackNet_torchscript.pt")
    return ShuttleCandidateCollector(**kwargs)


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
    collector = _collector(image_size=(40, 20), heatmap_size=(4, 8), fps=25, threshold=0.5)
    heatmap = np.zeros((4, 8), dtype=np.float32)
    heatmap[0, 0:2] = [0.7, 0.9]
    heatmap[2:4, 4:7] = 0.6
    collector.add(2, heatmap)
    collector.add(3, np.zeros((4, 8), dtype=np.float32))
    output = collector.write(tmp_path / "candidates.jsonl")

    metadata, frame, empty = _records(output)
    assert metadata["schema"] == "shuttle_candidates"
    assert metadata["schema_version"] == 2
    assert metadata["model_stage"] == "tracknet_pre_inpaint"
    assert metadata["thresholds"] == [0.5]
    assert metadata["heatmap_size"] == [8, 4]
    assert metadata["source_frame_range"] == [2, 3]
    assert frame["frame"] == 2
    assert [candidate["candidate_id"] for candidate in frame["candidates"]] == ["f000002-t050-c000", "f000002-t050-c001"]
    assert frame["candidates"][0]["peak_value"] == pytest.approx(0.9)
    assert frame["candidates"][0]["center"] == [5.0, 2.5]
    assert frame["candidates"][0]["peak_position"] == [7.5, 2.5]
    assert frame["candidates"][0]["weighted_centroid"] == pytest.approx([5.3125, 2.5])
    assert frame["candidates"][0]["bbox_normalized"] == [0.0, 0.0, 0.25, 0.25]
    assert frame["candidates"][0]["area_normalized"] == pytest.approx(2 / 32)
    assert frame["candidates"][0]["mean_activation"] == pytest.approx(0.8)
    assert frame["candidates"][0]["total_activation"] == pytest.approx(1.6)
    assert frame["candidates"][0]["total_activation_normalized"] == pytest.approx(1.6 / 32)
    assert frame["candidates"][1]["legacy_largest_component"] is True
    assert empty == {"type": "frame", "frame": 3, "candidates": []}
    with pytest.raises(FileExistsError, match="non-overwriting output path"):
        collector.write(output)


def test_multi_threshold_boundaries_connectivity_and_ids_are_stable(tmp_path: Path) -> None:
    boundary = np.zeros((3, 15), dtype=np.float32)
    boundary[0, ::2] = [0.2, 0.2001, 0.3, 0.3001, 0.4, 0.4001, 0.5, 0.5001]
    collector = _collector(image_size=(150, 30), heatmap_size=(3, 15), fps=30)
    collector.add(0, boundary)
    records = collector._frames[0]
    assert {threshold: sum(item["threshold"] == threshold for item in records) for threshold in (0.2, 0.3, 0.4, 0.5)} == {
        0.2: 7, 0.3: 5, 0.4: 3, 0.5: 1,
    }

    diagonal = np.zeros((3, 3), dtype=np.float32)
    diagonal[0, 0] = diagonal[1, 1] = 0.9
    all_thresholds = _collector(image_size=(30, 30), heatmap_size=(3, 3), fps=30)
    all_thresholds.add(4, diagonal)
    assert len(all_thresholds._frames[4]) == 7
    assert all(item["area"] == 2 for item in all_thresholds._frames[4])
    single_threshold = _collector(
        image_size=(30, 30), heatmap_size=(3, 3), fps=30, threshold=0.5,
    )
    single_threshold.add(4, diagonal)
    assert [item["candidate_id"] for item in single_threshold._frames[4]] == ["f000004-t050-c000"]
    assert [item["candidate_id"] for item in all_thresholds._frames[4] if item["threshold"] == 0.5] == [
        "f000004-t050-c000"
    ]


def test_v2_provenance_and_deterministic_retention_ranking(tmp_path: Path) -> None:
    collector = _collector(
        image_size=(100, 50), heatmap_size=(5, 10), fps=25,
        checkpoint_sha256="c" * 64,
        inference_model_sha256="d" * 64,
        inference_model_artifact="TrackNet_torchscript.pt",
        tracknet_config={"sequence_length": 8, "background_mode": "subtract", "preprocessing": {"value_range": [0, 1]}},
        overlap_ensemble_mode="weight",
        source_frame_range=(0, 99),
    )
    for frame in range(100):
        collector.add(frame, np.zeros((5, 10), dtype=np.float32))
    metadata, *_ = _records(collector.write(tmp_path / "v2.jsonl"))
    assert metadata["checkpoint_sha256"] == "c" * 64
    assert metadata["inference_model_sha256"] == "d" * 64
    assert metadata["inference_model_artifact"] == "TrackNet_torchscript.pt"
    assert metadata["provenance_verified"] is True
    assert metadata["nonproduction_unverified_provenance"] is False
    assert metadata["tracknet_config"]["sequence_length"] == 8
    assert metadata["tracknet_config"]["background_mode"] == "subtract"
    assert metadata["overlap_ensemble_mode"] == "weight"
    assert metadata["source_frame_range"] == [0, 99]
    assert metadata["source_frame_count"] == 100
    assert metadata["source_frame_index_space"] == "zero_based_working_video"
    assert metadata["threshold_comparator"] == ">"
    assert metadata["retention_policy"] == list(CANDIDATE_RETENTION_POLICY)
    assert metadata["pixel_position_convention"] == "heatmap_pixel_centers_scaled_to_image_space"

    candidates = [
        {"candidate_id": "d", "peak_activation": 0.8, "mean_activation": 0.7, "area_normalized": 0.2},
        {"candidate_id": "c", "peak_activation": 0.9, "mean_activation": 0.5, "area_normalized": 0.1},
        {"candidate_id": "b", "peak_activation": 0.9, "mean_activation": 0.6, "area_normalized": 0.1},
        {"candidate_id": "a", "peak_activation": 0.9, "mean_activation": 0.6, "area_normalized": 0.1},
    ]
    expected = ["a", "b", "c", "d"]
    assert [item["candidate_id"] for item in rank_shuttle_candidates(candidates)] == expected
    assert [item["candidate_id"] for item in rank_shuttle_candidates(reversed(candidates), 2)] == expected[:2]
    assert rank_shuttle_candidates(candidates, 0) == []
    with pytest.raises(ValueError, match="non-negative"):
        rank_shuttle_candidates(candidates, -1)


def test_v2_requires_verified_checkpoint_and_inference_model_hashes() -> None:
    common = {"image_size": (10, 10), "heatmap_size": (2, 2), "fps": 30}
    with pytest.raises(ValueError, match="checkpoint_sha256.*inference_model_sha256"):
        ShuttleCandidateCollector(**common)
    with pytest.raises(ValueError, match="inference_model_sha256"):
        ShuttleCandidateCollector(**common, checkpoint_sha256="a" * 64)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        ShuttleCandidateCollector(**common, inference_model_sha256="b" * 64)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        ShuttleCandidateCollector(
            **common, checkpoint_sha256="not-a-sha", inference_model_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("frames", "missing"),
    [((0, 2), "missing=\\[1\\]"), ((0, 1), "missing=\\[2\\]")],
)
def test_v2_rejects_missing_middle_or_tail_frames(
    tmp_path: Path, frames: tuple[int, ...], missing: str,
) -> None:
    collector = _collector(
        image_size=(10, 10), heatmap_size=(2, 2), fps=30,
        source_frame_range=(0, 2),
    )
    for frame in frames:
        collector.add(frame, np.zeros((2, 2), dtype=np.float32))
    output = tmp_path / f"missing-{'-'.join(map(str, frames))}.jsonl"
    with pytest.raises(ValueError, match=missing):
        collector.write(output)
    assert not output.exists()


def test_tracknet_expected_frame_range_rejects_empty_and_short_overlap() -> None:
    assert tracknet_candidate_frame_range(8, 8, "weight") == (0, 7)
    assert tracknet_candidate_frame_range(2, 8, "nonoverlap") == (0, 1)
    with pytest.raises(ValueError, match="empty or unreadable"):
        tracknet_candidate_frame_range(0, 8, "weight")
    with pytest.raises(ValueError, match="at least one complete sequence"):
        tracknet_candidate_frame_range(7, 8, "average")


def test_linker_is_replayable_and_covers_every_candidate_once(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.jsonl"
    collector = _collector(image_size=(100, 100), heatmap_size=(10, 10), fps=30, threshold=0.5)
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
    collector = _collector(image_size=(100, 100), heatmap_size=(10, 10), fps=30, threshold=0.5)
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


def test_v1_reader_and_v2_legacy_link_view_avoid_threshold_duplicates(tmp_path: Path) -> None:
    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    v1_candidates, _ = _write_hypothesis_inputs(v1_dir, [("legacy", [(0, 5, 5, 0.8)])])
    v1_metadata, v1_frames = read_shuttle_candidates(v1_candidates)
    assert v1_metadata["schema_version"] == 1
    assert len(v1_frames) == 1

    candidates_path = tmp_path / "v2.jsonl"
    collector = _collector(image_size=(100, 100), heatmap_size=(10, 10), fps=30)
    for frame in (0, 1):
        heatmap = np.zeros((10, 10), dtype=np.float32)
        heatmap[5, 5] = 0.9
        collector.add(frame, heatmap)
    collector.write(candidates_path)
    tracklets_path = tmp_path / "v2-tracklets.jsonl"
    link_shuttle_candidates(candidates_path, tracklets_path)
    metadata, *tracklets = _records(tracklets_path)
    assert metadata["candidate_view"] == {
        "schema_version": 2, "threshold": 0.5, "purpose": "v1_compatibility",
    }
    linked_ids = [candidate_id for tracklet in tracklets for candidate_id in tracklet["candidate_ids"]]
    assert linked_ids == ["f000000-t050-c000", "f000001-t050-c000"]
    assert len(tracklets) == 1

    hypotheses_path = tmp_path / "v2-hypotheses.jsonl"
    link_shuttle_hypotheses(candidates_path, tracklets_path, hypotheses_path)
    _, *hypotheses = _records(hypotheses_path)
    assert hypotheses[0]["candidate_ids"] == linked_ids
    assert len(_load_shuttle_evidence(candidates_path, tracklets_path)[0]) == 7
    assert _load_rank_one_hypothesis_segments(candidates_path, hypotheses_path)[1]


def test_exact_candidate_retention_recall_uses_present_denominator(tmp_path: Path) -> None:
    candidates_path = tmp_path / "recall.jsonl"
    collector = _collector(image_size=(100, 100), heatmap_size=(10, 10), fps=30)
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[5, 5] = 0.9
    collector.add(0, heatmap)
    collector.add(1, np.zeros((10, 10), dtype=np.float32))
    collector.add(2, np.zeros((10, 10), dtype=np.float32))
    collector.write(candidates_path)
    report = evaluate_candidate_retention_recall(candidates_path, [
        {"frame": 0, "label_kind": "selected", "candidate_id": "f000000-t040-c000"},
        {"frame": 1, "label_kind": "MISSING_PROPOSAL"},
        {"frame": 2, "label_kind": "NO_SHUTTLE"},
    ])
    assert report["label_equivalence"] == "exact_candidate_id"
    assert report["threshold_comparison_valid"] is False
    assert report["present_shuttle_frames"] == 2
    assert report["missing_proposal_frames"] == 1
    assert report["no_shuttle_frames"] == 1
    assert report["recall_at_k"] == {
        "1": 0.0, "2": 0.0, "3": 0.0, "5": 0.0,
        "8": 0.5, "12": 0.5, "all": 0.5,
    }
    assert report["candidates_per_frame"] == pytest.approx(7 / 3)
    assert report["maximum_candidates_per_frame"] == 7


def test_notebooks_forward_tracknet_model_provenance() -> None:
    notebook_dir = Path(__file__).parents[1] / "notebooks"
    for variant in ("Local", "Colab", "Kaggle"):
        notebook = json.loads((notebook_dir / f"Single_Video_Feature_Extraction_{variant}.ipynb").read_text(encoding="utf-8"))
        source = "".join(line for cell in notebook["cells"] for line in cell.get("source", []))
        # The complete conditional bundle carries TrackNet provenance through
        # the same reuse path and includes InpaintNet only on explicit opt-in.
        assert "models = load_track_models(REPO_DIR, use_inpaintnet=USE_INPAINTNET)" in source
        assert "reuse_models=models" in source
    predict_source = (Path(__file__).parents[1] / "src" / "TrackNetV3" / "predictArgs.py").read_text(encoding="utf-8")
    assert "CAP_PROP_FRAME_COUNT" in predict_source
    assert "source_frame_range=candidate_source_range" in predict_source


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
