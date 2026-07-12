"""Lossless-above-threshold TrackNet shuttle evidence and conservative links."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


CANDIDATE_SCHEMA = "shuttle_candidates"
TRACKLET_SCHEMA = "shuttle_tracklets"
HYPOTHESES_SCHEMA = "shuttle_hypotheses"


class ShuttleCandidateCollector:
    """Collect one deterministic component record per TrackNet output frame."""

    def __init__(self, *, image_size: tuple[int, int], heatmap_size: tuple[int, int], fps: float, threshold: float = 0.5):
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.fps = float(fps)
        self.threshold = float(threshold)
        self._frames: dict[int, list[dict]] = {}

    def add(self, frame: int, heatmap: np.ndarray) -> None:
        """Add an unthresholded, single-frame TrackNet heatmap exactly once."""
        frame = int(frame)
        if frame in self._frames:
            return
        heatmap = np.asarray(heatmap, dtype=np.float32)
        if heatmap.shape != self.heatmap_size:
            raise ValueError(f"expected heatmap {self.heatmap_size}, got {heatmap.shape}")
        mask = (heatmap > self.threshold).astype(np.uint8)
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        heatmap_h, heatmap_w = self.heatmap_size
        image_w, image_h = self.image_size
        sx, sy = image_w / heatmap_w, image_h / heatmap_h
        components = []
        for label in range(1, labels_count):
            x, y, width, height, area = [int(value) for value in stats[label]]
            peak = float(heatmap[labels == label].max())
            components.append((x, y, width, height, area, peak))
        # This ordering is explicit rather than relying on OpenCV label internals.
        components.sort(key=lambda item: (item[1], item[0], item[3], item[2], -item[4], -item[5]))
        legacy_index = None
        if components:
            # TrackNet's existing path picks the largest *bounding-box* area.
            legacy_index = max(range(len(components)), key=lambda index: components[index][2] * components[index][3])
        records = []
        for index, (x, y, width, height, area, peak) in enumerate(components):
            records.append({
                "candidate_id": f"f{frame:06d}-c{index:03d}",
                "center": [float((x + width / 2) * sx), float((y + height / 2) * sy)],
                "bbox": [float(x * sx), float(y * sy), float((x + width) * sx), float((y + height) * sy)],
                "area": area,
                "peak_value": peak,
                "legacy_largest_component": index == legacy_index,
            })
        self._frames[frame] = records

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image_w, image_h = self.image_size
        heatmap_h, heatmap_w = self.heatmap_size
        metadata = {
            "type": "metadata", "schema": CANDIDATE_SCHEMA, "schema_version": 1,
            "model_stage": "tracknet_pre_inpaint", "threshold": self.threshold, "fps": self.fps,
            "image_size": [image_w, image_h], "heatmap_size": [heatmap_w, heatmap_h],
            "coordinate_scaling": {"x": image_w / heatmap_w, "y": image_h / heatmap_h},
        }
        with path.open("w", encoding="utf-8") as output:
            output.write(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n")
            for frame in sorted(self._frames):
                output.write(json.dumps({"type": "frame", "frame": frame, "candidates": self._frames[frame]}, sort_keys=True, separators=(",", ":")) + "\n")
        return path


@dataclass(frozen=True)
class ShuttleLinkConfig:
    max_missing_frames: int = 1
    max_speed_image_diagonals_per_second: float = 6.0
    ambiguity_ratio: float = 0.7


@dataclass(frozen=True)
class ShuttleHypothesisConfig:
    """Configuration for the diagnostic, tracklet-level association decoder."""

    max_gap_seconds: float = 0.5
    max_speed_image_diagonals_per_second: float = 6.0
    max_hypotheses_per_region: int = 5
    # The legal-edge graph is a DAG, but the number of paths in it can still
    # be exponential.  Keep decoding bounded; this is a diagnostic decoder,
    # not an exhaustive graph enumerator.
    max_path_candidates_per_region: int = 4096
    min_symmetric_node_difference: float = 0.25
    score_version: str = "tracknet_motion_v1"


def _read_jsonl(path: Path, schema: str) -> tuple[dict, list[dict]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty artifact: {path}")
    metadata = json.loads(lines[0])
    if metadata.get("schema") != schema or metadata.get("schema_version") != 1:
        raise ValueError(f"expected {schema} schema v1 artifact: {path}")
    return metadata, [json.loads(line) for line in lines[1:] if line.strip()]


def _unambiguous_best(distance: float, alternatives: Iterable[float], ratio: float) -> bool:
    next_best = min(alternatives, default=None)
    return next_best is None or (distance < next_best and distance <= ratio * next_best)


def link_shuttle_candidates(candidate_path: str | Path, tracklet_path: str | Path, config: ShuttleLinkConfig | None = None) -> Path:
    """Link candidate components conservatively without selecting a rally shuttle."""
    config = config or ShuttleLinkConfig()
    candidate_path, tracklet_path = Path(candidate_path), Path(tracklet_path)
    metadata, frames = _read_jsonl(candidate_path, CANDIDATE_SCHEMA)
    fps = float(metadata["fps"])
    width, height = (float(value) for value in metadata["image_size"])
    motion_gate_per_frame = config.max_speed_image_diagonals_per_second * math.hypot(width, height) / fps
    candidates_by_frame: dict[int, list[dict]] = {}
    for frame_record in frames:
        if frame_record.get("type") != "frame":
            raise ValueError("candidate artifact contains a non-frame record")
        frame = int(frame_record["frame"])
        for candidate in frame_record.get("candidates", []):
            candidates_by_frame.setdefault(frame, []).append(candidate)

    tracklets: list[dict] = []
    active: dict[str, dict] = {}
    next_tracklet = 0

    def start(frame: int, candidate: dict) -> None:
        nonlocal next_tracklet
        tracklet_id = f"t{next_tracklet:06d}"
        next_tracklet += 1
        item = {"type": "tracklet", "tracklet_id": tracklet_id, "candidate_ids": [candidate["candidate_id"]], "frames": [frame]}
        tracklets.append(item)
        active[tracklet_id] = {"item": item, "points": [(frame, candidate["center"])]}

    for frame in sorted(candidates_by_frame):
        current = candidates_by_frame[frame]
        active = {key: state for key, state in active.items() if frame - state["points"][-1][0] <= config.max_missing_frames + 1}
        distances: dict[tuple[str, int], float] = {}
        for tracklet_id, state in active.items():
            points = state["points"]
            last_frame, last = points[-1]
            delta = frame - last_frame
            predicted = list(last)
            if len(points) >= 2:
                previous_frame, previous = points[-2]
                observed_delta = last_frame - previous_frame
                if observed_delta:
                    predicted = [last[axis] + (last[axis] - previous[axis]) * delta / observed_delta for axis in (0, 1)]
            for index, candidate in enumerate(current):
                distance = math.dist(predicted, candidate["center"])
                if distance <= motion_gate_per_frame * delta:
                    distances[(tracklet_id, index)] = distance

        accepted: list[tuple[str, int]] = []
        for (tracklet_id, index), distance in distances.items():
            track_alternatives = [value for (other_tracklet, other_index), value in distances.items() if other_tracklet == tracklet_id and other_index != index]
            candidate_alternatives = [value for (other_tracklet, other_index), value in distances.items() if other_index == index and other_tracklet != tracklet_id]
            best_for_track = min((value for (other_tracklet, _), value in distances.items() if other_tracklet == tracklet_id), default=math.inf)
            best_for_candidate = min((value for (_, other_index), value in distances.items() if other_index == index), default=math.inf)
            if distance == best_for_track == best_for_candidate and _unambiguous_best(distance, track_alternatives, config.ambiguity_ratio) and _unambiguous_best(distance, candidate_alternatives, config.ambiguity_ratio):
                accepted.append((tracklet_id, index))

        accepted_tracks = {tracklet_id for tracklet_id, _ in accepted}
        # A tracklet with gated candidates but no uniquely safe continuation ends here.
        ambiguous_tracks = {tracklet_id for tracklet_id, _ in distances} - accepted_tracks
        for tracklet_id in ambiguous_tracks:
            active.pop(tracklet_id, None)
        used = set()
        for tracklet_id, index in accepted:
            candidate = current[index]
            state = active[tracklet_id]
            state["item"]["candidate_ids"].append(candidate["candidate_id"])
            state["item"]["frames"].append(frame)
            state["points"].append((frame, candidate["center"]))
            used.add(index)
        for index, candidate in enumerate(current):
            if index not in used:
                start(frame, candidate)

    fingerprint = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    output_metadata = {
        "type": "metadata", "schema": TRACKLET_SCHEMA, "schema_version": 1,
        "candidate_artifact": candidate_path.name, "candidate_sha256": fingerprint,
        "link_config": asdict(config),
    }
    tracklet_path.parent.mkdir(parents=True, exist_ok=True)
    with tracklet_path.open("w", encoding="utf-8") as output:
        output.write(json.dumps(output_metadata, sort_keys=True, separators=(",", ":")) + "\n")
        for item in tracklets:
            output.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
    return tracklet_path


def link_shuttle_hypotheses(
    candidate_path: str | Path,
    tracklet_path: str | Path,
    hypotheses_path: str | Path,
    config: ShuttleHypothesisConfig | None = None,
) -> Path:
    """Decode diversified, replayable shuttle association hypotheses.

    This intentionally consumes only immutable candidate/tracklet evidence.  It
    does not alter the legacy TrackNet CSV path or attempt camera-specific
    ballistic modelling.
    """
    config = config or ShuttleHypothesisConfig()
    if config.max_gap_seconds < 0 or config.max_speed_image_diagonals_per_second <= 0:
        raise ValueError("hypothesis gap and speed limits must be positive")
    if config.max_hypotheses_per_region < 1:
        raise ValueError("max_hypotheses_per_region must be at least one")
    if config.max_path_candidates_per_region < config.max_hypotheses_per_region:
        raise ValueError("max_path_candidates_per_region must cover the requested hypotheses")
    if not 0 <= config.min_symmetric_node_difference <= 1:
        raise ValueError("min_symmetric_node_difference must be between zero and one")

    candidate_path, tracklet_path, hypotheses_path = map(Path, (candidate_path, tracklet_path, hypotheses_path))
    candidate_metadata, candidate_frames = _read_jsonl(candidate_path, CANDIDATE_SCHEMA)
    tracklet_metadata, tracklet_records = _read_jsonl(tracklet_path, TRACKLET_SCHEMA)
    candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if tracklet_metadata.get("candidate_artifact") != candidate_path.name or tracklet_metadata.get("candidate_sha256") != candidate_hash:
        raise ValueError("tracklet artifact does not reference this exact candidate artifact")

    fps = float(candidate_metadata["fps"])
    width, height = (float(value) for value in candidate_metadata["image_size"])
    speed_gate = config.max_speed_image_diagonals_per_second * math.hypot(width, height) / fps
    candidates = {
        item["candidate_id"]: (int(frame_record["frame"]), item)
        for frame_record in candidate_frames
        if frame_record.get("type") == "frame"
        for item in frame_record.get("candidates", [])
    }

    nodes: list[dict] = []
    for record in tracklet_records:
        if record.get("type") != "tracklet":
            raise ValueError("tracklet artifact contains a non-tracklet record")
        candidate_ids = list(record.get("candidate_ids", []))
        frames = [int(frame) for frame in record.get("frames", [])]
        if not candidate_ids or len(candidate_ids) != len(frames) or frames != sorted(frames) or len(set(frames)) != len(frames):
            raise ValueError(f"invalid tracklet geometry: {record.get('tracklet_id')}")
        points = []
        for candidate_id, frame in zip(candidate_ids, frames):
            candidate_frame, candidate = candidates.get(candidate_id, (None, None))
            if candidate is None or candidate_frame != frame:
                raise ValueError(f"tracklet references missing or mismatched candidate: {candidate_id}")
            points.append((frame, tuple(float(value) for value in candidate["center"])))
        nodes.append({
            "tracklet_id": record["tracklet_id"], "candidate_ids": candidate_ids, "frames": frames,
            "points": points,
            "tracknet_confidence": sum(float(candidates[item][1]["peak_value"]) for item in candidate_ids) / len(candidate_ids),
        })
    nodes.sort(key=lambda node: (node["frames"][0], node["frames"][-1], node["tracklet_id"]))
    if len({node["tracklet_id"] for node in nodes}) != len(nodes):
        raise ValueError("tracklet IDs must be unique")

    edges: dict[int, list[tuple[int, dict]]] = {index: [] for index in range(len(nodes))}
    for source_index, source in enumerate(nodes):
        tail_frame, tail_point = source["points"][-1]
        if len(source["points"]) > 1:
            previous_frame, previous_point = source["points"][-2]
            elapsed = tail_frame - previous_frame
            velocity = tuple((tail_point[axis] - previous_point[axis]) / elapsed for axis in (0, 1))
        else:
            velocity = (0.0, 0.0)
        for target_index, target in enumerate(nodes):
            head_frame, head_point = target["points"][0]
            if head_frame <= tail_frame:
                continue
            delta_frames = head_frame - tail_frame
            gap_seconds = (delta_frames - 1) / fps
            if gap_seconds > config.max_gap_seconds:
                continue
            predicted = tuple(tail_point[axis] + velocity[axis] * delta_frames for axis in (0, 1))
            distance = math.dist(predicted, head_point)
            gate = speed_gate * delta_frames
            if distance > gate:
                continue
            motion_compatibility = max(0.0, 1.0 - distance / gate)
            gap_penalty = gap_seconds / config.max_gap_seconds if config.max_gap_seconds else 0.0
            edge_score = motion_compatibility * (1.0 - gap_penalty)
            edges[source_index].append((target_index, {
                "motion_compatibility": motion_compatibility,
                "gap_penalty": gap_penalty,
                "edge_score": edge_score,
            }))
    for values in edges.values():
        values.sort(key=lambda item: (nodes[item[0]]["frames"][0], nodes[item[0]]["tracklet_id"]))

    # Connected components of the legal-edge graph are independent association regions.
    adjacent = {index: set() for index in range(len(nodes))}
    for source_index, values in edges.items():
        for target_index, _ in values:
            adjacent[source_index].add(target_index)
            adjacent[target_index].add(source_index)
    regions: list[list[int]] = []
    unseen = set(range(len(nodes)))
    while unseen:
        pending = [min(unseen)]
        region = []
        while pending:
            index = pending.pop()
            if index not in unseen:
                continue
            unseen.remove(index)
            region.append(index)
            pending.extend(sorted(adjacent[index] & unseen, reverse=True))
        regions.append(sorted(region, key=lambda item: (nodes[item]["frames"][0], nodes[item]["tracklet_id"])))

    def score_path(path: list[int], path_edges: list[dict]) -> tuple[float, dict]:
        confidence = sum(nodes[index]["tracknet_confidence"] for index in path) / len(path)
        motion = sum(edge["motion_compatibility"] for edge in path_edges) / len(path_edges) if path_edges else 0.0
        gap = sum(edge["gap_penalty"] for edge in path_edges) / len(path_edges) if path_edges else 0.0
        edge_score = sum(edge["edge_score"] for edge in path_edges) / len(path_edges) if path_edges else 0.0
        total = confidence if not path_edges else (confidence + edge_score) / 2
        return total, {
            "tracknet_confidence": confidence,
            "motion_compatibility": motion,
            "gap_penalty": gap,
            "edge_score": edge_score,
            "player_contact": None,
        }

    hypothesis_records = []
    region_metadata = []
    for region_number, region in enumerate(regions):
        region_id = f"r{region_number:04d}"
        region_set = set(region)
        candidates_for_region: list[tuple[float, list[int], list[dict], dict]] = []
        # Exhaustively enumerating paths is unsafe here: a region with only a
        # few plausible alternatives can contain tens of millions of paths.
        # Seed every singleton so isolated/weak evidence remains observable,
        # then use a deterministic beam per starting node for longer paths.
        for node in region:
            total, breakdown = score_path([node], [])
            candidates_for_region.append((total, [node], [], breakdown))

        beam_width = max(32, config.max_hypotheses_per_region * 8)
        path_budget = config.max_path_candidates_per_region - len(region)
        for node in region:
            if path_budget <= 0:
                break
            frontier = [([node], [])]
            while frontier and path_budget > 0:
                expanded: list[tuple[float, list[int], list[dict], dict]] = []
                for path, path_edges in frontier:
                    for target, edge in edges[path[-1]]:
                        if target not in region_set:
                            continue
                        extended_path = path + [target]
                        extended_edges = path_edges + [edge]
                        total, breakdown = score_path(extended_path, extended_edges)
                        expanded.append((total, extended_path, extended_edges, breakdown))
                if not expanded:
                    break
                expanded.sort(key=lambda item: (
                    -item[0], -len(item[1]),
                    tuple(nodes[index]["tracklet_id"] for index in item[1]),
                ))
                expanded = expanded[:min(beam_width, path_budget)]
                candidates_for_region.extend(expanded)
                path_budget -= len(expanded)
                frontier = [(item[1], item[2]) for item in expanded]
        candidates_for_region.sort(key=lambda item: (-item[0], -len(item[1]), tuple(nodes[index]["tracklet_id"] for index in item[1])))
        retained: list[tuple[float, list[int], list[dict], dict]] = []
        for candidate in candidates_for_region:
            candidate_nodes = set(candidate[1])
            if all(
                len(candidate_nodes.symmetric_difference(set(existing[1]))) / max(len(candidate_nodes), len(existing[1])) >= config.min_symmetric_node_difference
                for existing in retained
            ):
                retained.append(candidate)
                if len(retained) == config.max_hypotheses_per_region:
                    break
        region_metadata.append({
            "region_id": region_id,
            "tracklet_ids": [nodes[index]["tracklet_id"] for index in region],
            "frame_start": min(nodes[index]["frames"][0] for index in region),
            "frame_end": max(nodes[index]["frames"][-1] for index in region),
        })
        for rank, (total, path, _, breakdown) in enumerate(retained, start=1):
            hypothesis_records.append({
                "type": "hypothesis", "region_id": region_id, "rank": rank,
                "tracklet_ids": [nodes[index]["tracklet_id"] for index in path],
                "candidate_ids": [candidate_id for index in path for candidate_id in nodes[index]["candidate_ids"]],
                "frame_start": nodes[path[0]]["frames"][0], "frame_end": nodes[path[-1]]["frames"][-1],
                "total_score": total, "score_breakdown": breakdown,
            })

    hypotheses_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "metadata", "schema": HYPOTHESES_SCHEMA, "schema_version": 1,
        "candidate_artifact": candidate_path.name, "candidate_sha256": candidate_hash,
        "tracklet_artifact": tracklet_path.name, "tracklet_sha256": hashlib.sha256(tracklet_path.read_bytes()).hexdigest(),
        "decoder_config": asdict(config), "score_version": config.score_version,
        "association_regions": region_metadata,
    }
    with hypotheses_path.open("w", encoding="utf-8") as output:
        output.write(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n")
        for record in hypothesis_records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return hypotheses_path
