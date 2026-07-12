from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from InPlay.heuristic.clip_export import padded_bounds
from InPlay.heuristic.rally_v2 import V2Inputs, decode_rallies, write_state_events
from InPlay.heuristic.segment import Rally


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in records))


def bundle(tmp_path: Path, *, eof: bool = False, ambiguous: bool = False) -> V2Inputs:
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidates, tracklets, hypotheses = (tmp_path / name for name in ("candidates.jsonl", "tracklets.jsonl", "hypotheses.jsonl"))
    assignments, poses = tmp_path / "assignments.jsonl", tmp_path / "poses.jsonl"
    metadata, calibration = tmp_path / "metadata.json", tmp_path / "court.json"
    count = 90 if eof else 140
    candidate_rows = [{"type": "metadata", "schema": "shuttle_candidates", "schema_version": 1,
                       "fps": 30.0, "image_size": [640, 360]}]
    ids = []
    end = count if eof else 65
    for frame in range(16, end):
        cid = f"f{frame:06d}-c000"; ids.append(cid)
        candidate_rows.append({"type": "frame", "frame": frame, "candidates": [{"candidate_id": cid,
            "center": [100 + 8 * (frame - 16), 120], "peak_value": .9}]})
    _write(candidates, candidate_rows)
    candidate_hash = hashlib.sha256(candidates.read_bytes()).hexdigest()
    _write(tracklets, [{"type": "metadata", "schema": "shuttle_tracklets", "schema_version": 1,
                        "candidate_artifact": candidates.name, "candidate_sha256": candidate_hash},
                       {"type": "tracklet", "tracklet_id": "t0", "candidate_ids": ids,
                        "frames": list(range(16, end))}])
    tracklet_hash = hashlib.sha256(tracklets.read_bytes()).hexdigest()
    hypothesis_rows = [{"type": "metadata", "schema": "shuttle_hypotheses", "schema_version": 1,
                        "candidate_sha256": candidate_hash, "tracklet_sha256": tracklet_hash}]
    hypothesis_rows.append({"type": "hypothesis", "region_id": "r0", "rank": 1,
                            "candidate_ids": ids, "total_score": .9})
    if ambiguous:
        hypothesis_rows.append({"type": "hypothesis", "region_id": "r0", "rank": 2,
                                "candidate_ids": ids, "total_score": .85})
    _write(hypotheses, hypothesis_rows)
    raw_fp = "sha256:raw"
    player_rows = [{"type": "metadata", "schema": "player_assignments", "schema_version": 3,
                    "raw_artifact_fingerprint": raw_fp, "frame_size": [640, 360],
                    "fps": 30.0, "frame_count": count}]
    for frame in range(count):
        p1x = 0 if frame < 16 else (-1 if frame == 16 else (1 if frame == 22 else 0))
        player_rows.append({"type": "frame", "frame": frame, "slots": {
            "P1": {"assignment": {"court_x": p1x, "court_y": -3, "bbox": [1, 1, 2, 2]}, "ambiguity_reason": None},
            "P2": {"assignment": {"court_x": 0, "court_y": 3, "bbox": [3, 3, 4, 4]}, "ambiguity_reason": None}}})
    _write(assignments, player_rows)
    _write(poses, [{"type": "metadata", "schema": "pose_cache", "schema_version": 2,
                    "raw_artifact_fingerprint": raw_fp}])
    metadata.write_text(json.dumps({"source_id": "camera", "fps": 30.0,
                                    "image_size": [640, 360], "frame_count": count}))
    calibration.write_text(json.dumps({"image_size": [640, 360], "image_to_court": [[1,0,0],[0,1,0],[0,0,1]]}))
    return V2Inputs(candidates, tracklets, hypotheses, assignments, poses, metadata, calibration)


def test_accepts_confirmed_exchange_backdates_and_cools_down(tmp_path: Path) -> None:
    rallies, events, meta = decode_rallies(bundle(tmp_path))
    assert len(rallies) == 1 and rallies[0].status == "accepted"
    assert rallies[0].start_frame == 16 and rallies[0].end_frame == 64
    transitions = {item["transition"] for item in events}
    assert {"readiness_confirmed", "serve_evidence", "opponent_response_confirmed",
            "live_support_lost", "end_confirmed", "cooldown_complete"} <= transitions
    assert meta["input_fingerprints"]["candidates"].startswith("sha256:")


def test_eof_ambiguity_and_degraded_are_review_only(tmp_path: Path) -> None:
    eof, _, _ = decode_rallies(bundle(tmp_path / "eof", eof=True))
    assert eof and eof[0].status == "review" and "eof_unresolved" in eof[0].flags
    ambiguous, _, _ = decode_rallies(bundle(tmp_path / "ambiguous", ambiguous=True))
    assert all(item.status != "accepted" for item in ambiguous)
    degraded, _, _ = decode_rallies(bundle(tmp_path / "degraded"), degraded=True)
    assert degraded[0].status == "review" and "degraded_inputs" in degraded[0].flags


def test_fingerprint_validation_and_jsonl_determinism(tmp_path: Path) -> None:
    inputs = bundle(tmp_path)
    rallies, events, metadata = decode_rallies(inputs)
    one, two = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    write_state_events(one, metadata, events); write_state_events(two, metadata, events)
    assert one.read_bytes() == two.read_bytes()
    inputs.candidates.write_text(inputs.candidates.read_text() + "\n")
    with pytest.raises(ValueError, match="fingerprint"):
        decode_rallies(inputs)


def test_export_padding_does_not_mutate_canonical_boundaries() -> None:
    rally = Rally("s", "r", 5, 90, 0.5, 9.0, "accepted", .9, "high", "")
    assert padded_bounds(rally, fps=10, frame_count=100, before_seconds=1, after_seconds=2) == (0, 99)
    assert (rally.start_frame, rally.end_frame) == (5, 90)
