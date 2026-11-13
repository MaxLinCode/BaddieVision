import sys
from types import SimpleNamespace

from InPlay.heuristic.person_tracks import extract_person_tracks


def test_explicit_device_reaches_person_detector(monkeypatch, tmp_path):
    calls = {}

    class Detector:
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
    monkeypatch.setattr(
        "InPlay.heuristic.players.ensure_person_detector", lambda *a: None
    )
    extract_person_tracks(
        tmp_path / "input.mp4", tmp_path / "people.jsonl", "model.pt", device="cpu"
    )
    assert calls["device"] == "cpu"
