import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from InPlay.heuristic.person_tracks import (
    BYTETRACK_PERIODIC,
    default_person_model,
    extract_person_tracks,
)


def test_explicit_device_reaches_person_detector(monkeypatch, tmp_path):
    calls = {}

    class Detector:
        names = {0: "person"}

        def __init__(self, model):
            pass

        def track(self, **kwargs):
            calls.update(kwargs)
            return [SimpleNamespace(boxes=SimpleNamespace(id=None))]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Detector))
    monkeypatch.setattr(
        "InPlay.heuristic.person_tracks.read_video_info",
        lambda p: {"width": 64, "height": 48, "fps": 30, "frame_count": 1},
    )
    extract_person_tracks(
        tmp_path / "input.mp4", tmp_path / "people.jsonl", "model.pt", device="cpu"
    )
    assert calls["device"] == "cpu"
    assert calls["half"] is False
    tracker = Path(calls["tracker"])
    assert tracker.name == "botsort-fixed-camera.yaml"
    tracker_settings = tracker.read_text()
    assert "gmc_method: none" in tracker_settings
    assert "track_buffer: 30" in tracker_settings
    assert "track_low_thresh: 0.10" in tracker_settings
    assert "track_high_thresh: 0.25" in tracker_settings
    assert "new_track_thresh: 0.25" in tracker_settings
    assert calls["conf"] == 0.10
    assert "with_reid: false" in tracker.read_text()
    metadata = json.loads((tmp_path / "people.jsonl").read_text().splitlines()[0])
    assert metadata["detector"]["conf"] == 0.10
    assert metadata["tracker"] == tracker.name
    assert metadata["tracker_config_fingerprint"].startswith("sha256:")
    assert calls["imgsz"] == 960


def test_cuda_person_detector_uses_fp16(monkeypatch, tmp_path):
    calls = {}

    class Detector:
        def __init__(self, model):
            pass

        def track(self, **kwargs):
            calls.update(kwargs)
            return [SimpleNamespace(names={0: "person"}, boxes=SimpleNamespace(id=None))]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Detector))
    monkeypatch.setattr(
        "InPlay.heuristic.person_tracks.read_video_info",
        lambda p: {"width": 64, "height": 48, "fps": 30, "frame_count": 1},
    )
    extract_person_tracks(
        tmp_path / "input.mp4", tmp_path / "people.jsonl", "model.engine", device="cuda"
    )
    assert calls["half"] is True
    assert calls["imgsz"] == 960


def test_bytetrack_release_uses_original_detector_options(monkeypatch, tmp_path):
    calls = {}

    class Detector:
        names = {0: "person"}

        def __init__(self, model):
            pass

        def track(self, **kwargs):
            calls.update(kwargs)
            return [SimpleNamespace(boxes=SimpleNamespace(id=None))]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Detector))
    monkeypatch.setattr(
        "InPlay.heuristic.person_tracks.read_video_info",
        lambda p: {"width": 64, "height": 48, "fps": 30, "frame_count": 1},
    )
    video = tmp_path / "input.mp4"
    extract_person_tracks(
        video, tmp_path / "people.jsonl", "model.pt", device="cuda",
        tracking_profile=BYTETRACK_PERIODIC,
    )
    assert calls == {
        "source": str(video),
        "classes": [0],
        "tracker": "bytetrack.yaml",
        "stream": True,
        "persist": True,
        "verbose": False,
        "device": "cuda",
    }


@pytest.mark.parametrize(
    ("device", "engine_present", "expected"),
    [
        ("cpu", True, "yolo26l.pt"),
        ("mps", True, "yolo26l.pt"),
        ("cuda", True, "models/yolo26l.engine"),
        ("cuda", False, "yolo26l.pt"),
    ],
)
def test_default_person_model(device, engine_present, expected, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    if engine_present:
        engine = Path("models/yolo26l.engine")
        engine.parent.mkdir()
        engine.touch()
    assert default_person_model(device) == Path(expected)
