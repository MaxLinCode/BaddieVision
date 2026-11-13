from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.check_private_assets import check_paths, private_path_reason


@pytest.mark.parametrize("name", [
    ".annotation-private/events/shuttle.jsonl",
    "config/match.local.json",
    "config/experiment.local-loso-manifest.json",
    "config/folds-manifest.json",
    "InPlay/in_play_frame_labels.csv",
    "yolov8/shuttle-yolo-dataset/labels/train/example.txt",
    "yolov8/shuttle-yolo-dataset/images/train/example.jpg",
    "outputs/sample/predictions.jsonl",
    "tests/fixtures/golden/real-source.json",
    "releases/private-model.pt",
    "private-recording.mp4",
    ".sample.private-provenance.json",
])
def test_private_data_is_rejected_even_if_force_added(name: str) -> None:
    assert private_path_reason(name) is not None


@pytest.mark.parametrize("name", [
    "config/rally-experiment.example.json",
    "tests/fixtures/heuristic_labels.csv",
    "tests/fixtures/heuristic_tracks.csv",
    "src/rally_release.py",
    "releases/rally-six-source-v1/manifest.example.json",
])
def test_source_and_synthetic_examples_are_allowed(name: str) -> None:
    assert private_path_reason(name) is None


def test_notebook_outputs_and_labels_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "example.ipynb"
    path.write_text(json.dumps({"cells": [{
        "source": ["serve: 12, 34"], "outputs": [{"text": "private"}],
        "execution_count": 1,
    }]}))
    problems = check_paths(tmp_path, [path.name])
    assert any("saved notebook output" in item for item in problems)
    assert any("hit annotations" in item for item in problems)
    path.write_text(json.dumps({"cells": [{"source": ["label_string = ''"], "outputs": [], "execution_count": None}]}))
    assert not check_paths(tmp_path, [path.name])


def test_guard_uses_tracked_files_not_ignored_local_data(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "video.mp4").write_bytes(b"private")
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_private_assets.py"
    clean = subprocess.run([sys.executable, str(script), "--root", str(tmp_path)], capture_output=True, text=True, check=False)
    assert clean.returncode == 0
    subprocess.run(["git", "add", "video.mp4"], cwd=tmp_path, check=True)
    dirty = subprocess.run([sys.executable, str(script), "--root", str(tmp_path)], capture_output=True, text=True, check=False)
    assert dirty.returncode == 1
    assert "video.mp4" in dirty.stdout


def test_embedded_source_intervals_are_private_but_local_loading_is_allowed(tmp_path: Path) -> None:
    source = tmp_path / "training.py"
    source.write_text("rally_intervals = [(10, 20), (40, 50)]\n")
    assert any("embedded recording-specific rally annotations" in issue for issue in check_paths(tmp_path, [source.name]))
    source.write_text("rally_intervals = read_private_intervals()\n")
    assert not check_paths(tmp_path, [source.name])
