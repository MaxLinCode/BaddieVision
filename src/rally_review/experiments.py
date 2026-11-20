"""Discover and read local, source-aligned rally experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from struct import Struct

from src.temporal_selector.rally_intervals import read_rally_intervals


PREDICTION_NAMES = {"predictions.jsonl", "test-predictions.jsonl", "val-predictions.jsonl", "scored-predictions.jsonl"}
CONFIG_NAMES = ("sources.local.json", "experiment.json", "resolved-config.json", "config.json")
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}
SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
READ_ERRORS = (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError)


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected object rows in the JSONL artifact")
                yield row


def local_file(root, directory, value):
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    path = (directory / Path(value).expanduser()).resolve()
    return path if path.is_relative_to(root) and path.is_file() else None


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload


def stamp(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def key(*parts):
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:20]


def source_id(row, metadata):
    sources = metadata.get("source_ids", [])
    singleton = sources[0] if isinstance(sources, list) and len(sources) == 1 else None
    return row.get("source_id") or row.get("prediction_source_id") or metadata.get("source_id") or singleton


def has_rally_score(row):
    return row.get("inplay_probability") is not None or row.get("inplay_logit") is not None


@lru_cache(maxsize=256)
def prediction_sources(path, version):
    metadata = {}
    sources = set()
    try:
        for row in records(path):
            if row.get("type") == "metadata":
                metadata = row
            elif has_rally_score(row):
                sid = source_id(row, metadata)
                if isinstance(sid, str) and SOURCE_ID.fullmatch(sid):
                    sources.add(sid)
    except READ_ERRORS:
        return frozenset()
    return frozenset(sources)


@dataclass(frozen=True)
class Source:
    video: Path | None = None
    assignments: Path | None = None
    poses: Path | None = None
    label_pairs: tuple[tuple[Path, Path], ...] = ()
    shuttle: Path | None = None


@dataclass(frozen=True)
class Run:
    id: str
    name: str
    predictions: Path
    source: Source


@dataclass(frozen=True)
class Recording:
    id: str
    experiment: str
    source_id: str
    runs: tuple[Run, ...]


def label_pair(root, directory, csv="rallies.csv", manifest="rallies.manifest.json"):
    labels = local_file(root, directory, csv)
    metadata = local_file(root, directory, manifest)
    return (labels, metadata) if labels and metadata else None


def catalog_sources(path):
    raw = read_json(path)
    dataset = raw.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("Invalid experiment dataset configuration")
    entries = raw.get("sources", dataset.get("sources", []))
    if not isinstance(entries, list):
        raise ValueError("Expected a list of configured sources")
    return entries, dataset


def resolve_source(root, directory, sid, catalogs, configs):
    # Use the run's source definition first. Recovered catalogs supply portable
    # paths for runs copied back from a remote machine.
    nearby = []
    for parent in (directory, *directory.parents):
        if not parent.is_relative_to(root):
            break
        nearby.extend(path for name in CONFIG_NAMES if (path := local_file(root, parent, name)))
    video = assignments = poses = shuttle = None
    labels = []
    for parent in (directory, *directory.parents):
        if not parent.is_relative_to(root):
            break
        for subdir in (parent / "frozen-labels", parent):
            pair = label_pair(root, subdir)
            if pair:
                labels.append(pair)
    for path in dict.fromkeys([*nearby, *configs, *catalogs]):
        try:
            entries, dataset = catalog_sources(path)
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("source_id") != sid or not entry.get("enabled", True):
                    continue
                parent = path.parent
                artifact = entry.get("artifact_root")
                if artifact:
                    artifact = (parent / artifact).resolve()
                if video is None:
                    candidate = local_file(root, parent, entry.get("video_path"))
                    if candidate and candidate.suffix.lower() in VIDEO_SUFFIXES:
                        video = candidate
                if assignments is None:
                    assignments = local_file(root, parent, entry.get("assignments_path"))
                    if assignments is None and artifact:
                        assignments = local_file(root, artifact, "players/assignments.jsonl")
                if poses is None:
                    poses = local_file(root, parent, entry.get("pose_cache_path"))
                    if poses is None and artifact:
                        poses = local_file(root, artifact, "players/poses.jsonl")
                if shuttle is None:
                    shuttle = local_file(root, parent, entry.get("candidates_path"))
                    if shuttle is None and artifact:
                        shuttle = local_file(root, artifact, "shuttle/candidates.frozen.jsonl")
                        if shuttle is None:
                            shuttle = local_file(root, artifact, "shuttle/candidates.pilot.jsonl")
                pair = label_pair(root, parent, dataset.get("rally_intervals_path"), dataset.get("rally_manifest_path"))
                if pair:
                    labels.append(pair)
                for candidate in (parent / "review", parent):
                    pair = label_pair(root, candidate)
                    if pair:
                        labels.append(pair)
        except READ_ERRORS:
            continue
    if video is None:
        # Standalone inference directories can keep video.mp4 alongside predictions.
        # For deeper layouts use the source catalog, which disambiguates recordings.
        video = local_file(root, directory, f"{sid}_input.mp4") or local_file(root, directory, "video.mp4")
    return Source(video, assignments, poses, tuple(dict.fromkeys(labels)), shuttle)


def discover(root):
    local = root / ".local-data"
    paths = (local / "experiments", root / "experiments", root / "outputs", local / "rally-inference")
    catalog_candidates = [
        root / "config/sources.local.json",
        *sorted((local / "annotation").glob("*/sources.local.json")),
        local / "original-checkout/config/sources.local.json",
    ]
    catalogs = [path for candidate in catalog_candidates if (path := local_file(root, root, candidate))]
    grouped = defaultdict(list)
    for base in paths:
        if not base.is_dir():
            continue
        predictions = sorted({path for name in PREDICTION_NAMES for path in base.rglob(name)})
        seen = set()
        for candidate in predictions:
            path = local_file(root, root, candidate)
            if path is None or not path.is_relative_to(base) or path in seen:
                continue
            seen.add(path)
            relative = path.relative_to(base)
            experiment_dir = base / relative.parts[0] if len(relative.parts) > 1 else base
            experiment = str(experiment_dir.relative_to(root)).removeprefix(".local-data/")
            try:
                sources = prediction_sources(path, stamp(path))
            except OSError:
                continue
            for sid in sorted(sources):
                run_name = path.parent.relative_to(experiment_dir).as_posix()
                if path.name != "predictions.jsonl":
                    run_name = f"{run_name} / {path.stem}" if run_name != "." else path.stem
                elif run_name == ".":
                    run_name = path.stem
                if run_name.startswith(sid + "/"):
                    run_name = run_name[len(sid)+1:]
                config_candidates = sorted(experiment_dir.glob("*/experiment.json"))
                configs = [c for candidate in config_candidates if (c := local_file(root, root, candidate))]
                source = resolve_source(root, path.parent, sid, catalogs, configs)
                grouped[(experiment, sid)].append(Run(key(path.relative_to(root)), run_name, path, source))
    items = [
        Recording(key(experiment, sid), experiment, sid, tuple(runs))
        for (experiment, sid), runs in sorted(grouped.items(), reverse=True)
    ]
    # Open a frozen-label review first when one is present.
    def initial_priority(item):
        playable = [run for run in item.runs if run.source.video]
        has_frozen_labels = any(
            csv.parent.name == "frozen-labels" for run in playable for csv, _ in run.source.label_pairs
        )
        return not bool(playable), not has_frozen_labels

    items.sort(key=initial_priority)
    return {item.id: item for item in items}


def normalize_score(row, stem):
    value = row.get(stem + "_probability")
    if value is None and row.get(stem + "_logit") is not None:
        logit = float(row[stem + "_logit"])
        if math.isfinite(logit):
            value = 1 / (1 + math.exp(-max(-700, min(700, logit))))
    if value is None:
        return None
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f"Invalid probability for {stem}")
    return round(value, 4)


def normalized_coordinates(value, size):
    if isinstance(value, list) and len(value) == size and all(
        type(item) in (int, float) and 0 <= item <= 1 for item in value
    ):
        return value
    return None


def shuttle_evidence(provenance, count, candidate, peak=None):
    candidate = candidate if isinstance(candidate, dict) else {}
    position = candidate.get("peak_position_normalized") or candidate.get("center_normalized")
    return {
        "provenance": provenance,
        "count": count,
        "position": normalized_coordinates(position, 2),
        "box": normalized_coordinates(candidate.get("bbox_normalized"), 4),
        "peak": peak,
    }


@lru_cache(maxsize=24)
def load_prediction(path, version, sid, count):
    metadata = {}
    layers = {name: [None] * count for name in ("inplay", "start", "end")}
    decoded = ["?"] * count
    targets = [None] * count
    thresholds = set()
    frames = set()
    shuttle = {}
    for row in records(path):
        if row.get("type") == "metadata":
            metadata = row
            if metadata.get("source_id") == sid and metadata.get("frame_count", count) != count:
                raise ValueError("Prediction and video frame counts differ")
            continue
        if source_id(row, metadata) != sid or not has_rally_score(row):
            continue
        frame = row["frame"]
        if type(frame) is not int or not 0 <= frame < count or frame in frames:
            raise ValueError(f"Invalid or duplicate source-local prediction frame {frame}")
        frames.add(frame)
        decoder = row.get("decoder_configuration", metadata.get("decoder")) or {}
        if decoder.get("threshold") is not None:
            thresholds.add(float(decoder["threshold"]))
        for name, stem in (("inplay", "inplay"), ("start", "rally_start"), ("end", "rally_end")):
            layers[name][frame] = normalize_score(row, stem)
        if isinstance(row.get("decoded_inplay"), bool):
            decoded[frame] = "1" if row["decoded_inplay"] else "0"
        value = row.get("inplay_target")
        if value in (0, 1):
            targets[frame] = int(value)
        candidate_count = row.get("candidate_count")
        if type(candidate_count) is int and candidate_count >= 0:
            shuttle[frame] = shuttle_evidence("model_input", candidate_count, row.get("selected_position"))
    if not frames:
        raise ValueError("No frame scores for this recording")
    if len(thresholds) > 1 or any(not 0 <= t <= 1 for t in thresholds):
        raise ValueError("Inconsistent run decoder thresholds")
    payload = {name: values for name, values in layers.items() if name == "inplay" or any(v is not None for v in values)}
    payload.update(decoded="".join(decoded), threshold=next(iter(thresholds), None))
    hashes = metadata.get("input_artifact_sha256") or {}
    return payload, targets, hashes, shuttle


@lru_cache(maxsize=12)
def video_timing(path, version):
    """Use one presentation timestamp per stored frame, with a decode fallback."""
    common = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json"]
    fields = "stream=avg_frame_rate,nb_frames"
    probe = subprocess.run(
        [*common, "-show_packets", "-show_entries", f"{fields}:packet=pts_time,flags", str(path)],
        capture_output=True, text=True, check=True,
    )
    parsed = json.loads(probe.stdout)
    stream = parsed["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    packets = parsed.get("packets", [])
    count = stream.get("nb_frames")
    times = sorted(float(packet["pts_time"]) for packet in packets if "pts_time" in packet and "D" not in packet.get("flags", ""))
    if count != str(len(times)) or len(set(times)) != len(times):
        probe = subprocess.run(
            [*common, "-threads", "4", "-show_frames", "-show_entries",
             f"{fields}:frame=best_effort_timestamp_time", str(path)], capture_output=True, text=True, check=True,
        )
        times = [float(frame["best_effort_timestamp_time"]) for frame in json.loads(probe.stdout).get("frames", [])]
    if fps <= 0 or not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Video frame timings are missing or ambiguous")
    return fps, [round(t - times[0], 6) for t in times]


@lru_cache(maxsize=16)
def digest(path, version):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=8)
def load_shuttle_sidecar(path, version, sid, count, video, video_version):
    rows = iter(records(path))
    metadata = next(rows, {})
    video_hash = metadata.get("source_video_sha256") or metadata.get("video_sha256")
    if (
        metadata.get("type") != "metadata"
        or metadata.get("source_id", sid) != sid
        or metadata.get("source_frame_index_space", "zero_based_working_video") != "zero_based_working_video"
        or metadata.get("source_frame_count", metadata.get("frame_count", count)) != count
        or (video_hash and video_hash != digest(video, video_version))
    ):
        raise ValueError("Shuttle candidates do not match the source-local video")

    frames = {}
    for row in rows:
        frame = row.get("frame")
        if row.get("type") != "frame" or type(frame) is not int or not 0 <= frame < count or frame in frames:
            raise ValueError("Invalid or duplicate shuttle candidate frame")
        candidates = row.get("frozen_candidates", row.get("candidates"))
        if not isinstance(candidates, list) or not all(isinstance(candidate, dict) for candidate in candidates):
            raise ValueError("Expected a list of shuttle candidates")

        def peak(candidate):
            value = candidate.get("peak_activation", candidate.get("peak_value", 0))
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError("Invalid shuttle peak activation")
            return value

        top = max(candidates, key=peak, default=None)
        frames[frame] = shuttle_evidence(
            "external_diagnostic", len(candidates), top, peak(top) if top is not None else None
        )
    return frames


def external_shuttle(run, sid, count, video, video_version, warnings):
    path = run.source.shuttle
    if path is None:
        return {}
    try:
        return load_shuttle_sidecar(path, stamp(path), sid, count, video, video_version)
    except READ_ERRORS as error:
        warnings.append(f"Shuttle sidecar unavailable: {error}")
        return {}


@lru_cache(maxsize=12)
def read_labels(pair, versions, sid, video, video_version, count, fps):
    index = read_rally_intervals(*pair)
    manifest = index.sources.get(sid)
    if manifest is None:
        return None
    if manifest.frame_count != count or not math.isclose(manifest.fps, fps, abs_tol=.0001):
        raise ValueError("Label and video frame geometry differs")
    if manifest.video_sha256 != digest(video, video_version):
        raise ValueError("Label recording fingerprint differs from the local video")
    return (
        [[interval.start_frame, interval.end_frame] for interval in index.for_source(sid)],
        [[span.start_frame, span.end_frame] for span in index.reviewed_coverage if span.source_id == sid],
    )


POSE_LANDMARKS = 33
HIDDEN_POINT = -32768
POSE_POINTS = Struct("<66h")
OVERLAY_CHUNK = 240


def pixel_box(value):
    if not isinstance(value, (tuple, list)) or len(value) != 4 or not all(math.isfinite(v) for v in value):
        return None
    x1, y1, x2, y2 = value
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def project_pose(row):
    """Convert visible crop-relative landmarks to native video pixels."""
    points = row.get("pose_landmarks", [])
    if len(points) != POSE_LANDMARKS:
        return False, None
    box = pixel_box(row.get("pose_bbox"))
    coordinates = []
    has_visible = False
    for point in points:
        x, y, z = (point.get(key, math.nan) for key in ("x", "y", "z"))
        visible = point.get("visibility", 0) >= .5 and all(math.isfinite(v) for v in (x, y, z))
        has_visible |= visible
        if visible and box:
            left, top, right, bottom = box
            px = left + x * (right - left)
            py = top + y * (bottom - top)
            if all(math.isfinite(v) and -32767 <= v <= 32767 for v in (px, py)):
                coordinates.extend((round(px), round(py)))
                continue
        coordinates.extend((HIDDEN_POINT, HIDDEN_POINT))
    return has_visible, POSE_POINTS.pack(*coordinates) if has_visible and box else None


@dataclass
class PlayerReview:
    states: list[str]
    boxes: list[list]
    poses: list[list]
    image_size: tuple[int, int] | None

    @property
    def available(self):
        return any(state in "12" for series in self.states for state in series)

    def chunk(self, start):
        count = len(self.states[0])
        frames = []
        for frame in range(start, min(start + OVERLAY_CHUNK, count)):
            overlay = []
            for role in range(2):
                box = self.boxes[role][frame]
                pose = self.poses[role][frame]
                overlay.append([box, POSE_POINTS.unpack(pose) if pose else None] if box or pose else None)
            frames.append(overlay)
        return {"start": start, "imageSize": self.image_size, "frames": frames}


@lru_cache(maxsize=4)
def load_player_review(assignments, assignments_version, poses, poses_version, sid, count):
    role_states = [["?"] * count for _ in range(2)]
    boxes = [[None] * count for _ in range(2)]
    joints = [[None] * count for _ in range(2)]
    tracked = {}
    assignment_meta = {}
    image_size = None
    for row in records(assignments):
        if row.get("type") == "metadata":
            assignment_meta = row
            if row.get("source_id") != sid or row.get("frame_count") != count:
                raise ValueError("Player assignments do not match the source-local video")
            size = row.get("frame_size")
            if isinstance(size, list) and len(size) == 2 and all(type(v) is int and v > 0 for v in size):
                image_size = tuple(size)
            continue
        frame = row["frame"]
        if type(frame) is not int or not 0 <= frame < count or role_states[0][frame] != "?":
            raise ValueError("Invalid or duplicate player assignment frame")
        for index, role in enumerate(("P1", "P2")):
            assignment = row["slots"][role]["assignment"]
            role_states[index][frame] = "1" if assignment is not None else "0"
            if assignment is None:
                continue
            track = assignment.get("track_id")
            if track is not None:
                tracked.setdefault((frame, track), []).append(index)
            box = pixel_box(assignment.get("bbox"))
            if box:
                boxes[index][frame] = (*map(round, box), track)
    if poses:
        for row in records(poses):
            if row.get("type") == "metadata":
                if row.get("raw_artifact_fingerprint") != assignment_meta.get("raw_artifact_fingerprint"):
                    raise ValueError("Pose cache and player assignments have different lineage")
                continue
            frame = row.get("frame")
            roles = tracked.get((frame, row.get("track_id")))
            if not roles or row.get("status") != "detected":
                continue
            visible, pose = project_pose(row)
            if visible:
                for role in roles:
                    role_states[role][frame] = "2"
                    joints[role][frame] = pose
    return PlayerReview(["".join(role) for role in role_states], boxes, joints, image_size)


def source_player_review(source, sid, count):
    if source.assignments is None:
        raise ValueError("No saved player assignments for this run")
    return load_player_review(
        source.assignments, stamp(source.assignments),
        source.poses, stamp(source.poses) if source.poses else None, sid, count,
    )


def recording_overlays(recording, run, start):
    video = run.source.video
    if video is None:
        raise ValueError("Source video is unavailable")
    video_version = stamp(video)
    _, times = video_timing(video, video_version)
    count = len(times)
    if not 0 <= start < count or start % OVERLAY_CHUNK:
        raise ValueError(f"Overlay start must be a multiple of {OVERLAY_CHUNK} inside the source")
    sid = recording.source_id
    warnings = []
    output = {"start": start, "frames": None, "warnings": warnings}
    if run.source.assignments:
        try:
            output.update(source_player_review(run.source, sid, count).chunk(start))
        except READ_ERRORS as error:
            warnings.append(f"Player overlays unavailable: {error}")

    _, _, _, inline = load_prediction(run.predictions, stamp(run.predictions), sid, count)
    sidecar = external_shuttle(run, sid, count, video, video_version, warnings)
    if inline or sidecar:
        output["shuttle"] = [
            inline.get(frame, sidecar.get(frame)) for frame in range(start, min(start + OVERLAY_CHUNK, count))
        ]
    return output


def target_intervals(values, valid):
    intervals = []
    start = None
    for frame, value in enumerate([*values, None]):
        if valid(value) and start is None:
            start = frame
        elif not valid(value) and start is not None:
            intervals.append([start, frame - 1])
            start = None
    return intervals


def recording_metrics(recording, selected):
    sid = recording.source_id
    source = selected[0].source
    video = source.video
    if video is None:
        raise ValueError("No source video found. Add this recording to config/sources.local.json.")
    video_version = stamp(video)
    fps, times = video_timing(video, video_version)
    count = len(times)
    models = []
    target_values = [None] * count
    warnings = []
    primary_inline = {}
    for index, run in enumerate(selected):
        if run.source.video != video:
            raise ValueError("Selected runs resolve to different source videos")
        model, targets, hashes, inline = load_prediction(run.predictions, stamp(run.predictions), sid, count)
        if hashes.get("video") and hashes["video"] != digest(video, video_version):
            raise ValueError(f"{run.name}: prediction video fingerprint differs from the source")
        models.append(dict(model, name=run.name))
        if index == 0:
            primary_inline = inline
        for i, target in enumerate(targets):
            if target is not None:
                if target_values[i] is not None and target_values[i] != target:
                    raise ValueError("Selected runs have conflicting saved rally targets")
                target_values[i] = target
    output = {
        "schema": "baddievision-rally-diagnostics-v1",
        "sourceId": sid, "fps": fps, "frameCount": count, "timestamps": times, "models": models,
    }
    for pair in source.label_pairs:
        try:
            labels = read_labels(pair, tuple(map(stamp, pair)), sid, video, video_version, count, fps)
        except READ_ERRORS as error:
            warnings.append(f"Skipped incompatible rally labels: {error}")
            continue
        if labels is not None:
            output["labels"], output["coverage"] = labels
            break
    else:
        if any(target is not None for target in target_values):
            output["labels"] = target_intervals(target_values, lambda value: value == 1)
            output["coverage"] = target_intervals(target_values, lambda value: value is not None)
            warnings.append("Rally labels come from the saved evaluation targets.")
    if source.assignments:
        try:
            player_review = source_player_review(source, sid, count)
            output["players"] = player_review.states
            if player_review.available:
                output["playerOverlays"] = True
        except READ_ERRORS as error:
            warnings.append(f"Player lanes unavailable: {error}")
        else:
            if len(selected) > 1 and any(
                (run.source.assignments, run.source.poses) != (source.assignments, source.poses) for run in selected[1:]
            ):
                warnings.append(f"Player lanes show observations from {selected[0].name}.")
    sidecar = external_shuttle(selected[0], sid, count, video, video_version, warnings)
    if primary_inline or sidecar:
        output["shuttleOverlays"] = True
    if output.get("playerOverlays") or output.get("shuttleOverlays"):
        output["overlayChunkSize"] = OVERLAY_CHUNK
    return output, warnings
