import hashlib
import json
from pathlib import Path

import pytest

from src.workflow.registry import SourceCatalog


def _touch_inputs(root: Path, source_id: str, *, ready: bool = False) -> Path:
    video = root / "videos" / f"{source_id}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(f"video-{source_id}".encode())
    if ready:
        output = root / "outputs" / source_id
        output.mkdir(parents=True)
        for name in (
            "shuttle_candidates_pilot.jsonl",
            "shuttle_candidates_frozen.jsonl",
            "player_assignments.jsonl",
            "pose_cache.jsonl",
        ):
            (output / name).write_text("{}\n")
        calibration = root / "features" / "court" / f"{source_id}.json"
        calibration.parent.mkdir(parents=True, exist_ok=True)
        calibration.write_text(json.dumps({"image_size": [1280, 720]}))
    return video


def test_add_source_derives_paths_and_is_portable(tmp_path):
    root = tmp_path / "repo"
    video = _touch_inputs(root, "match-a")
    catalog_path = root / "config" / "sources.local.json"
    catalog = SourceCatalog(catalog_path)

    source = catalog.add(video, source_id="match-a")

    assert source.output_dir == root / "outputs" / "match-a"
    assert source.frozen_candidates_path.name == "shuttle_candidates_frozen.jsonl"
    value = json.loads(catalog_path.read_text())
    assert value["sources"][0]["video_path"] == "../videos/match-a.mp4"
    assert value["sources"][0]["video_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert SourceCatalog.read(catalog_path).sources["match-a"] == source


def test_add_rejects_duplicate_video_under_another_source(tmp_path):
    video = _touch_inputs(tmp_path, "match-a")
    catalog = SourceCatalog(tmp_path / "config" / "sources.local.json")
    catalog.add(video, source_id="match-a")
    with pytest.raises(ValueError, match="already registered"):
        catalog.add(video, source_id="match-b")


def test_status_checks_readiness_and_calibration_schema(tmp_path):
    video = _touch_inputs(tmp_path, "match-a", ready=True)
    catalog = SourceCatalog(tmp_path / "config" / "sources.local.json")
    source = catalog.add(video, source_id="match-a")
    status = catalog.status(source)
    assert status.annotation_ready
    assert status.experiment_ready

    source.calibration_path.write_text("{}")
    status = catalog.status(source)
    assert not status.experiment_ready
    assert status.problems == ("calibration has no image_size",)


def test_annotation_config_prefers_frozen_and_falls_back_to_pilot(tmp_path):
    video = _touch_inputs(tmp_path, "match-a")
    catalog = SourceCatalog(tmp_path / "config" / "sources.local.json")
    source = catalog.add(video, source_id="match-a")
    source.candidates_path.parent.mkdir(parents=True)
    source.candidates_path.write_text("{}\n")

    output = catalog.write_annotation_config(tmp_path / "runtime" / "sources.json")
    value = json.loads(output.read_text())
    assert value["sources"][0]["candidates_path"].endswith("shuttle_candidates_pilot.jsonl")

    source.frozen_candidates_path.write_text("{}\n")
    catalog.write_annotation_config(output)
    value = json.loads(output.read_text())
    assert value["sources"][0]["candidates_path"].endswith("shuttle_candidates_frozen.jsonl")


def test_experiment_prepare_generates_hash_and_loso_manifest(tmp_path):
    catalog = SourceCatalog(tmp_path / "config" / "sources.local.json")
    for source_id in ("one", "two", "three"):
        catalog.add(_touch_inputs(tmp_path, source_id, ready=True), source_id=source_id)
    runtime = tmp_path / ".annotation-final"
    (runtime / "events").mkdir(parents=True)
    annotations = runtime / "events" / "shuttle.jsonl"
    annotations.write_text("{}\n")
    (runtime / "rallies.csv").write_text("source_id,rally_id,start_frame,end_frame\n")
    (runtime / "rallies.manifest.json").write_text("{}\n")
    (runtime / "queues").mkdir()
    (runtime / "queues" / "shuttle-audit.json").write_text("{}\n")

    output, manifest_path = catalog.write_experiment(
        tmp_path / "config" / "experiment.json", runtime=runtime, seed=71
    )

    config = json.loads(output.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert config["dataset"]["expected_annotation_sha256"] == hashlib.sha256(annotations.read_bytes()).hexdigest()
    assert len(config["dataset"]["sources"]) == 3
    assert [fold["evaluation_source_ids"] for fold in manifest["folds"]] == [["one"], ["two"], ["three"]]


def test_experiment_prepare_reports_unready_sources(tmp_path):
    catalog = SourceCatalog(tmp_path / "config" / "sources.local.json")
    catalog.add(_touch_inputs(tmp_path, "one"), source_id="one")
    catalog.add(_touch_inputs(tmp_path, "two"), source_id="two")
    with pytest.raises(ValueError, match="not experiment-ready"):
        catalog.write_experiment(tmp_path / "experiment.json", runtime=tmp_path / "runtime")
