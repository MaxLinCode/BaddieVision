"""Deterministic burst queues with adaptive/audit sampling separation."""

from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .core import AnnotationRegistry
from .views import validate_source_video


def native_fps_burst(
    anchor_frame: int,
    *,
    fps: float,
    frame_count: int,
    duration_seconds: float = 1.0,
) -> tuple[int, ...]:
    """Return a centered native-frame burst, shifted to fit source bounds.

    The frame count is ``floor(fps * duration + 0.5)`` (round-half-up), made
    explicit so fractional-FPS sources do not depend on Python's banker rounding.
    """
    if fps <= 0 or duration_seconds <= 0 or frame_count < 1:
        raise ValueError("fps, duration, and frame_count must be positive")
    anchor_frame = int(anchor_frame)
    if not 0 <= anchor_frame < frame_count:
        raise ValueError(f"anchor frame outside source bounds: {anchor_frame}")
    length = max(1, math.floor(float(fps) * float(duration_seconds) + 0.5))
    length = min(length, int(frame_count))
    start = anchor_frame - length // 2
    start = min(max(0, start), frame_count - length)
    return tuple(range(start, start + length))


@dataclass(frozen=True)
class QueueBurst:
    burst_id: str
    task: str
    source_id: str
    anchor_frame: int
    frames: tuple[int, ...]
    candidate_artifact_sha256: str
    source_video_sha256: str
    score: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "QueueBurst":
        return cls(
            burst_id=str(value["burst_id"]),
            task=str(value["task"]),
            source_id=str(value["source_id"]),
            anchor_frame=int(value["anchor_frame"]),
            frames=tuple(int(frame) for frame in value["frames"]),  # type: ignore[arg-type]
            candidate_artifact_sha256=str(value["candidate_artifact_sha256"]),
            source_video_sha256=str(value["source_video_sha256"]),
            score=None if value.get("score") is None else float(value["score"]),
        )


@dataclass(frozen=True)
class AnnotationQueue:
    queue_id: str
    kind: str
    task: str
    seed: int | None
    bursts: tuple[QueueBurst, ...]
    construction: Mapping[str, object]

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema": "annotation_queue",
            "schema_version": 1,
            "queue_id": self.queue_id,
            "kind": self.kind,
            "task": self.task,
            "seed": self.seed,
            "construction": dict(self.construction),
            "bursts": [asdict(burst) for burst in self.bursts],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AnnotationQueue":
        if value.get("schema") != "annotation_queue" or value.get("schema_version") != 1:
            raise ValueError("expected annotation_queue schema v1")
        raw_bursts = value.get("bursts")
        if not isinstance(raw_bursts, list):
            raise ValueError("annotation queue bursts must be a list")
        construction = value.get("construction", {})
        if not isinstance(construction, dict):
            raise ValueError("annotation queue construction must be an object")
        return cls(
            queue_id=str(value["queue_id"]),
            kind=str(value["kind"]),
            task=str(value["task"]),
            seed=None if value.get("seed") is None else int(value["seed"]),
            bursts=tuple(QueueBurst.from_mapping(item) for item in raw_bursts),
            construction=construction,
        )

    def write(self, path: str | Path, *, immutable: bool | None = None) -> Path:
        path = Path(path)
        immutable = self.kind == "audit" if immutable is None else immutable
        encoded = json.dumps(self.to_mapping(), sort_keys=True, indent=2) + "\n"
        if path.exists() and immutable:
            if path.read_text(encoding="utf-8") != encoded:
                raise FileExistsError(f"immutable queue already exists with different content: {path}")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path) -> "AnnotationQueue":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("annotation queue must be a JSON object")
        return cls.from_mapping(value)

    def frame_keys(self) -> set[tuple[str, int]]:
        return {
            (burst.source_id, frame)
            for burst in self.bursts
            for frame in burst.frames
        }


def _burst(
    registry: AnnotationRegistry,
    task: str,
    source_id: str,
    anchor: int,
    *,
    kind: str,
    score: float | None,
    artifact_sha256: str,
    frames: tuple[int, ...],
) -> QueueBurst:
    _, source = registry.resolve(task, source_id)
    return QueueBurst(
        burst_id=f"{kind}-{source_id}-{anchor:09d}",
        task=task,
        source_id=source_id,
        anchor_frame=anchor,
        frames=frames,
        candidate_artifact_sha256=artifact_sha256,
        source_video_sha256=source.video_sha256,
        score=score,
    )


def _eligible_runs(registry: AnnotationRegistry, task: str, source_id: str) -> tuple[tuple[int, ...], ...]:
    plugin, source = registry.resolve(task, source_id)
    eligible = tuple(sorted({int(frame) for frame in plugin.eligible_frames(source)}))
    if not eligible:
        raise ValueError(f"task {task!r} has no eligible frames for source {source_id!r}")
    if eligible[0] < 0 or eligible[-1] >= source.frame_count:
        raise ValueError(f"task eligible frames exceed source bounds: {source_id}")
    runs: list[list[int]] = []
    for frame in eligible:
        if not runs or frame != runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    return tuple(tuple(run) for run in runs)


def _eligible_bursts(
    registry: AnnotationRegistry,
    task: str,
    source_id: str,
) -> dict[int, tuple[int, ...]]:
    """Map every eligible anchor to a burst contained in its coverage run."""
    source = registry.sources[source_id]
    bursts: dict[int, tuple[int, ...]] = {}
    for run in _eligible_runs(registry, task, source_id):
        run_start = run[0]
        for frame in run:
            local = native_fps_burst(
                frame - run_start,
                fps=source.fps,
                frame_count=len(run),
            )
            bursts[frame] = tuple(run_start + item for item in local)
    return bursts


def _uniform_nonoverlap_slots(
    registry: AnnotationRegistry,
    task: str,
    source_id: str,
    rng: random.Random,
) -> list[tuple[int, tuple[int, ...]]]:
    """Build a randomly shifted maximum tiling over each eligible frame run."""
    source = registry.sources[source_id]
    target_length = max(1, math.floor(source.fps + 0.5))
    slots: list[tuple[int, tuple[int, ...]]] = []
    for run in _eligible_runs(registry, task, source_id):
        length = min(target_length, len(run))
        capacity = len(run) // length
        slack = len(run) - capacity * length
        offset = rng.randrange(slack + 1)
        for index in range(capacity):
            start = run[0] + offset + index * length
            frames = tuple(range(start, start + length))
            anchor = start + length // 2
            slots.append((anchor, frames))
    return slots


def build_uniform_audit_queue(
    registry: AnnotationRegistry,
    task: str,
    *,
    seed: int,
    anchor_count: int,
    manifest_path: str | Path | None = None,
) -> AnnotationQueue:
    """Create a separately seeded uniform queue without consulting task scores."""
    if anchor_count < 1:
        raise ValueError("audit anchor_count must be positive")
    registration = registry.tasks[task]
    artifact_hashes = {}
    for source_id in registration.source_ids:
        plugin, source = registry.resolve(task, source_id)
        validate_source_video(source, force=True)
        artifact_hashes[source_id] = plugin.verify_artifact_fingerprint(source)
    rng = random.Random(int(seed))
    population = [
        (source_id, frame, frames)
        for source_id in registration.source_ids
        for frame, frames in _uniform_nonoverlap_slots(
            registry, task, source_id, rng
        )
    ]
    if anchor_count > len(population):
        raise ValueError(
            "audit anchor_count exceeds non-overlapping burst capacity: "
            f"requested {anchor_count}, available {len(population)}"
        )
    sampled = rng.sample(population, anchor_count)
    bursts = tuple(
        _burst(
            registry,
            task,
            source_id,
            frame,
            kind="audit",
            score=None,
            artifact_sha256=artifact_hashes[source_id],
            frames=frames,
        )
        for source_id, frame, frames in sampled
    )
    audit_identity = json.dumps(
        {
            "version": 2,
            "task": task,
            "seed": int(seed),
            "anchor_count": anchor_count,
            "sources": [
                {
                    "source_id": source_id,
                    "artifact_sha256": artifact_hashes[source_id],
                    "video_sha256": registry.sources[source_id].video_sha256,
                }
                for source_id in registration.source_ids
            ],
            "bursts": [
                [burst.source_id, burst.anchor_frame, list(burst.frames)] for burst in bursts
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    queue = AnnotationQueue(
        queue_id=str(uuid.uuid5(uuid.NAMESPACE_URL, "annotation-audit:" + audit_identity)),
        kind="audit",
        task=task,
        seed=int(seed),
        bursts=bursts,
        construction={
            "sampling": "seeded_uniform_nonoverlap_tiling",
            "anchor_count": anchor_count,
        },
    )
    if manifest_path is not None:
        queue.write(manifest_path, immutable=True)
    return queue


def build_adaptive_queue(
    registry: AnnotationRegistry,
    task: str,
    *,
    anchor_count: int,
    audit_queue: AnnotationQueue | None = None,
    manifest_path: str | Path | None = None,
) -> AnnotationQueue:
    """Prioritize plugin-scored anchors while reserving all audit burst frames."""
    if anchor_count < 1:
        raise ValueError("adaptive anchor_count must be positive")
    if audit_queue is not None and (audit_queue.kind != "audit" or audit_queue.task != task):
        raise ValueError("adaptive exclusion queue must be an audit queue for the same task")
    registration = registry.tasks[task]
    artifact_hashes = {}
    for source_id in registration.source_ids:
        plugin, source = registry.resolve(task, source_id)
        validate_source_video(source, force=True)
        artifact_hashes[source_id] = plugin.verify_artifact_fingerprint(source)
    scored: list[tuple[float, str, int]] = []
    eligible_bursts: dict[str, dict[int, tuple[int, ...]]] = {}
    for source_id in registration.source_ids:
        plugin, source = registry.resolve(task, source_id)
        eligible_bursts[source_id] = _eligible_bursts(registry, task, source_id)
        for frame in eligible_bursts[source_id]:
            scored.append((float(plugin.queue_score(source, frame)), source_id, frame))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    reserved = audit_queue.frame_keys() if audit_queue is not None else set()
    used: set[tuple[str, int]] = set()
    selected: list[QueueBurst] = []
    for score, source_id, frame in scored:
        burst = _burst(
            registry,
            task,
            source_id,
            frame,
            kind="adaptive",
            score=score,
            artifact_sha256=artifact_hashes[source_id],
            frames=eligible_bursts[source_id][frame],
        )
        keys = {(source_id, item) for item in burst.frames}
        if keys & reserved or keys & used:
            continue
        selected.append(burst)
        used.update(keys)
        if len(selected) == anchor_count:
            break
    queue = AnnotationQueue(
        queue_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "annotation-adaptive:"
                + task
                + ":"
                + str(anchor_count)
                + ":"
                + (audit_queue.queue_id if audit_queue else "no-audit")
                + ":"
                + ":".join(
                    f"{source_id}={artifact_hashes[source_id]}="
                    f"{registry.sources[source_id].video_sha256}"
                    for source_id in registration.source_ids
                ),
            )
        ),
        kind="adaptive",
        task=task,
        seed=None,
        bursts=tuple(selected),
        construction={
            "sampling": "plugin_score_descending",
            "requested_anchor_count": anchor_count,
            "audit_queue_id": audit_queue.queue_id if audit_queue else None,
        },
    )
    if manifest_path is not None:
        queue.write(manifest_path, immutable=False)
    return queue


def validate_queue(registry: AnnotationRegistry, queue: AnnotationQueue) -> None:
    """Validate a persisted queue against current source/artifact fingerprints."""
    if queue.kind not in {"adaptive", "audit"}:
        raise ValueError(f"invalid queue kind: {queue.kind}")
    registration = registry.tasks.get(queue.task)
    if registration is None:
        raise ValueError(f"queue references unknown task: {queue.task}")
    fingerprints: dict[str, tuple[str, str]] = {}
    eligible_bursts: dict[str, dict[int, tuple[int, ...]]] = {}
    for source_id in registration.source_ids:
        plugin, source = registry.resolve(queue.task, source_id)
        validate_source_video(source, force=True)
        fingerprints[source_id] = plugin.verify_artifact_fingerprint(source), source.video_sha256
        eligible_bursts[source_id] = _eligible_bursts(registry, queue.task, source_id)
    used: set[tuple[str, int]] = set()
    for burst in queue.bursts:
        if burst.task != queue.task or burst.source_id not in fingerprints:
            raise ValueError(f"queue burst has an invalid task/source: {burst.burst_id}")
        artifact_sha, video_sha = fingerprints[burst.source_id]
        if burst.candidate_artifact_sha256 != artifact_sha:
            raise ValueError(f"queue candidate fingerprint mismatch: {burst.burst_id}")
        if burst.source_video_sha256 != video_sha:
            raise ValueError(f"queue source-video fingerprint mismatch: {burst.burst_id}")
        expected_frames = eligible_bursts[burst.source_id].get(burst.anchor_frame)
        if burst.frames != expected_frames:
            raise ValueError(
                f"queue burst frames do not match eligible native-FPS coverage: {burst.burst_id}"
            )
        keys = {(burst.source_id, frame) for frame in burst.frames}
        if queue.kind == "audit" and keys & used:
            raise ValueError(f"audit queue contains overlapping evaluation frames: {burst.burst_id}")
        used.update(keys)
    if queue.kind == "audit":
        declared_count = queue.construction.get("anchor_count")
        if declared_count != len(queue.bursts):
            raise ValueError(
                "audit queue burst count does not match its immutable construction contract"
            )
