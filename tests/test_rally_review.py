"""Exercising the local viewer with two saved runs on one synthetic recording."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from threading import Thread
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from src.rally_review.__main__ import Server
from src.rally_review.experiments import discover, recording_metrics, recording_overlays


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def create_review_fixture(root: Path) -> Path:
    source = root / "synthetic"
    (source / "players").mkdir(parents=True)
    (source / "shuttle").mkdir()
    video = source / "video.webm"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=navy:s=96x72:r=10:d=1.2",
         "-c:v", "libvpx-vp9", "-threads", "1", str(video)],
        check=True,
    )
    (root / "config").mkdir()
    (root / "config/sources.local.json").write_text(json.dumps({"sources": [{
        "source_id": "synthetic", "video_path": str(video), "artifact_root": str(source),
    }]}))
    assignments = [{
        "type": "metadata", "source_id": "synthetic", "frame_count": 12,
        "frame_size": [96, 72], "raw_artifact_fingerprint": "synthetic-tracks",
    }]
    evidence = {
        "peak_position_normalized": [0.6, 0.4],
        "bbox_normalized": [0.5, 0.3, 0.7, 0.5],
        "peak_activation": 0.9,
    }
    for frame in range(12):
        assignments.append({
            "type": "frame", "frame": frame, "slots": {
                "P1": {"assignment": {"track_id": 1, "bbox": [10, 10, 45, 60]}},
                "P2": {"assignment": None},
            },
        })
    write_jsonl(source / "players/assignments.jsonl", assignments)
    write_jsonl(source / "players/poses.jsonl", [
        {"type": "metadata", "raw_artifact_fingerprint": "synthetic-tracks"},
        {"type": "pose", "frame": 2, "track_id": 1, "status": "detected",
         "pose_bbox": [10, 10, 45, 60], "pose_landmarks": [
             {"x": 0.2 + joint / 66, "y": 0.3, "z": 0.0, "visibility": 0.9}
             for joint in range(33)
         ]},
    ])
    write_jsonl(source / "shuttle/candidates.frozen.jsonl", [
        {"type": "metadata", "source_id": "synthetic", "source_frame_count": 12},
        {"type": "frame", "frame": 0, "frozen_candidates": []},
        {"type": "frame", "frame": 2, "frozen_candidates": [evidence]},
    ])

    for run in ("primary", "second"):
        run_dir = root / "outputs" / "experiment" / run
        run_dir.mkdir(parents=True)
        rows = []
        for frame in range(12):
            decoded = 2 <= frame <= 6 or (run == "second" and frame >= 9)
            row = {
                "source_id": "synthetic", "frame": frame, "inplay_probability": 0.9 if decoded else 0.1,
                "decoded_inplay": decoded, "inplay_target": int(2 <= frame <= 6),
            }
            if run == "second" and frame == 2:
                row.update(candidate_count=2, selected_position=evidence)
            rows.append(row)
        write_jsonl(run_dir / "predictions.jsonl", rows)
    return source


@pytest.fixture
def source(tmp_path):
    return create_review_fixture(tmp_path)


def test_discovery_comparison_and_http_overlays(tmp_path, source):
    with Server(tmp_path, 0) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            def read(path):
                with urlopen(server.origin + path, timeout=5) as response:
                    return json.load(response)

            recording, = read("/api/catalog")["recordings"]
            assert recording["source"] == "synthetic"
            runs = {run["name"]: run["id"] for run in recording["runs"]}
            query = {"recording": recording["id"], "run": [runs["primary"], runs["second"]]}
            data = read("/api/metrics?" + urlencode(query, doseq=True))["data"]
            assert data["labels"] == [[2, 6]]
            assert data["models"][1]["decoded"].endswith("111")
            assert data["playerOverlays"] and data["shuttleOverlays"]
            for run, provenance in (("primary", "external_diagnostic"), ("second", "model_input")):
                query = urlencode({"recording": recording["id"], "run": runs[run]})
                chunk = read("/api/overlays?" + query)
                assert chunk["shuttle"][2]["provenance"] == provenance
                assert chunk["shuttle"][2]["position"] == [0.6, 0.4]
                assert chunk["shuttle"][2]["count"] == (1 if run == "primary" else 2)
                assert chunk["frames"][2][0][0] == [10, 10, 45, 60, 1]
                assert len(chunk["frames"][2][0][1]) == 66
            request = Request(server.origin + "/api/video?" + query, headers={"Range": "bytes=0-7"})
            with urlopen(request, timeout=5) as response:
                assert response.status == 206
                assert response.read() == (source / "video.webm").read_bytes()[:8]
        finally:
            server.shutdown()
            thread.join()


def test_optional_shuttle_overlay_does_not_depend_on_player_data(tmp_path, source):
    (source / "players/assignments.jsonl").unlink()
    recording, = discover(tmp_path).values()
    run = next(run for run in recording.runs if run.name == "primary")
    data, _ = recording_metrics(recording, [run])
    assert not data.get("playerOverlays")
    assert data["shuttleOverlays"]
    chunk = recording_overlays(recording, run, 0)
    assert chunk["frames"] is None
    assert chunk["shuttle"][0]["count"] == 0
    assert chunk["shuttle"][2]["peak"] == 0.9


def test_unaligned_shuttle_sidecar_keeps_player_overlays_and_scores(tmp_path, source):
    write_jsonl(source / "shuttle/candidates.frozen.jsonl", [
        {"type": "metadata", "source_id": "different-video", "source_frame_count": 12},
    ])
    recording, = discover(tmp_path).values()
    run = next(run for run in recording.runs if run.name == "primary")
    data, warnings = recording_metrics(recording, [run])
    assert data["models"] and data["playerOverlays"]
    assert not data.get("shuttleOverlays")
    assert any("Shuttle sidecar unavailable" in warning for warning in warnings)
    assert "shuttle" not in recording_overlays(recording, run, 0)
