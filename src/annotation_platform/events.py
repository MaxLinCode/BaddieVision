"""Immutable JSONL revision events, replay, audit, and current-label export."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

from .core import AnnotationRegistry


LabelKey = tuple[str, str, int]


@dataclass(frozen=True)
class AnnotationEvent:
    revision_id: str
    task: str
    source_id: str
    frame: int
    label_kind: str
    candidate_id: str | None
    candidate_artifact_sha256: str
    source_video_sha256: str
    annotator: str
    session_id: str
    timestamp: str
    superseded_revision: str | None

    @property
    def key(self) -> LabelKey:
        return self.task, self.source_id, self.frame

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AnnotationEvent":
        required = {
            "revision_id",
            "task",
            "source_id",
            "frame",
            "label_kind",
            "candidate_artifact_sha256",
            "source_video_sha256",
            "annotator",
            "session_id",
            "timestamp",
            "superseded_revision",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"annotation event missing fields: {missing}")
        candidate_id = value.get("candidate_id")
        return cls(
            revision_id=str(value["revision_id"]),
            task=str(value["task"]),
            source_id=str(value["source_id"]),
            frame=int(value["frame"]),
            label_kind=str(value["label_kind"]),
            candidate_id=None if candidate_id is None else str(candidate_id),
            candidate_artifact_sha256=str(value["candidate_artifact_sha256"]),
            source_video_sha256=str(value["source_video_sha256"]),
            annotator=str(value["annotator"]),
            session_id=str(value["session_id"]),
            timestamp=str(value["timestamp"]),
            superseded_revision=(
                None
                if value["superseded_revision"] is None
                else str(value["superseded_revision"])
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayState:
    events: tuple[AnnotationEvent, ...]
    heads: Mapping[LabelKey, AnnotationEvent]
    active: Mapping[LabelKey, AnnotationEvent]
    ignored_interrupted_tail: bool = False
    recovered_interrupted_tails: int = 0


def _parse_jsonl(raw: bytes) -> tuple[list[Mapping[str, object]], bool]:
    lines = raw.splitlines(keepends=True)
    records: list[Mapping[str, object]] = []
    ignored_tail = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        is_final_unterminated = index == len(lines) - 1 and not line.endswith((b"\n", b"\r"))
        if is_final_unterminated:
            # JSONL durability is record-delimiter based. Even syntactically
            # complete JSON without its newline may be the prefix of a write
            # that was interrupted immediately before more bytes arrived.
            ignored_tail = True
            break
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid annotation JSONL at line {index + 1}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"annotation event at line {index + 1} is not an object")
        records.append(value)
    return records, ignored_tail


def replay_events(events: Iterable[AnnotationEvent]) -> ReplayState:
    ordered: list[AnnotationEvent] = []
    heads: dict[LabelKey, AnnotationEvent] = {}
    active: dict[LabelKey, AnnotationEvent] = {}
    by_revision: dict[str, AnnotationEvent] = {}
    revision_ids: set[str] = set()
    for event in events:
        if not event.revision_id or event.revision_id in revision_ids:
            raise ValueError(f"duplicate or empty revision ID: {event.revision_id!r}")
        revision_ids.add(event.revision_id)
        by_revision[event.revision_id] = event
        current = heads.get(event.key)
        expected = current.revision_id if current else None
        if event.superseded_revision != expected:
            raise ValueError(
                f"revision {event.revision_id} supersedes {event.superseded_revision!r}; "
                f"expected {expected!r}"
            )
        if event.label_kind == "undo" and current is None:
            raise ValueError("undo cannot be the first revision for a frame")
        heads[event.key] = event
        if event.label_kind == "undo":
            previous = by_revision.get(current.superseded_revision) if current else None
            if previous is None or previous.label_kind == "undo":
                active.pop(event.key, None)
            else:
                active[event.key] = previous
        else:
            active[event.key] = event
        ordered.append(event)
    return ReplayState(tuple(ordered), heads, active)


class EventStore:
    """Append and replay revision events after task-plugin validation."""

    def __init__(self, path: str | Path, registry: AnnotationRegistry):
        self.path = Path(path)
        self.registry = registry

    def replay(self) -> ReplayState:
        paths = self._log_paths()
        if not paths:
            return replay_events(())
        records: list[Mapping[str, object]] = []
        recovered = 0
        ignored_tail = False
        for index, path in enumerate(paths):
            segment_records, segment_ignored = _parse_jsonl(path.read_bytes())
            records.extend(segment_records)
            if not segment_ignored:
                continue
            if index == len(paths) - 1:
                ignored_tail = True
                continue
            next_path = paths[index + 1]
            recovery = self._recovery_path(next_path)
            if not recovery.exists():
                raise ValueError(f"interrupted log segment has no recovery audit record: {path}")
            details = json.loads(recovery.read_text(encoding="utf-8"))
            if (
                details.get("recovered_from") != path.name
                or details.get("recovered_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                raise ValueError(f"invalid recovery audit record: {recovery}")
            recovered += 1
        state = replay_events(AnnotationEvent.from_mapping(record) for record in records)
        return ReplayState(
            state.events,
            state.heads,
            state.active,
            ignored_tail,
            recovered,
        )

    def recover_interrupted_tail(self) -> Path:
        """Preserve a corrupt segment and continue in a new append-only segment."""
        paths = self._log_paths()
        if not paths:
            raise ValueError("there is no annotation log to recover")
        corrupted = paths[-1]
        _, ignored_tail = _parse_jsonl(corrupted.read_bytes())
        if not ignored_tail:
            raise ValueError("the final annotation segment has no interrupted tail")
        self._segment_directory.mkdir(parents=True, exist_ok=True)
        next_number = len(paths)
        next_path = self._segment_directory / f"{next_number:04d}.jsonl"
        recovery_path = self._recovery_path(next_path)
        if next_path.exists() or recovery_path.exists():
            raise FileExistsError("annotation recovery segment already exists")
        recovery_record = {
            "type": "interrupted_tail_recovery",
            "recovered_from": corrupted.name,
            "recovered_sha256": hashlib.sha256(corrupted.read_bytes()).hexdigest(),
            "recovered_bytes": corrupted.stat().st_size,
            "new_segment": next_path.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        recovery_path.write_text(
            json.dumps(recovery_record, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        next_path.touch(exist_ok=False)
        return next_path

    def record(
        self,
        *,
        task: str,
        source_id: str,
        frame: int,
        label_kind: str,
        candidate_artifact_sha256: str,
        annotator: str,
        session_id: str,
        candidate_id: str | None = None,
        revision_id: str | None = None,
        timestamp: str | None = None,
    ) -> AnnotationEvent:
        plugin, source = self.registry.resolve(task, source_id)
        annotator = str(annotator).strip()
        session_id = str(session_id).strip()
        if not annotator or not session_id:
            raise ValueError("annotator and session_id are required")
        state = self.replay()
        if state.ignored_interrupted_tail:
            raise ValueError(
                "annotation log has an interrupted final write; preserve it for audit and "
                "start a new event segment before recording more labels"
            )
        key = task, source_id, int(frame)
        head = state.heads.get(key)
        if label_kind == "undo" and head is None:
            raise ValueError("there is no revision to undo")
        # Registration/startup performs a full video fingerprint. Label writes
        # cheaply verify the cached file stat and re-hash only if it changed.
        from .views import validate_source_video

        validate_source_video(source)
        plugin.validate_label(
            source,
            frame=int(frame),
            label_kind=label_kind,
            candidate_id=candidate_id,
            candidate_artifact_sha256=candidate_artifact_sha256,
        )
        event = AnnotationEvent(
            revision_id=revision_id or str(uuid.uuid4()),
            task=task,
            source_id=source_id,
            frame=int(frame),
            label_kind=label_kind,
            candidate_id=candidate_id,
            candidate_artifact_sha256=candidate_artifact_sha256,
            source_video_sha256=source.video_sha256,
            annotator=annotator,
            session_id=session_id,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            superseded_revision=head.revision_id if head else None,
        )
        # Validate the chain before touching disk.
        replay_events((*state.events, event))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_mapping(), sort_keys=True, separators=(",", ":")) + "\n"
        output_path = self._log_paths()[-1] if self._log_paths() else self.path
        with output_path.open("a", encoding="utf-8") as output:
            output.write(line)
            output.flush()
            os.fsync(output.fileno())
        return event

    def export_current(
        self,
        output_path: str | Path,
        *,
        task: str | None = None,
        include_skips: bool = False,
    ) -> Path:
        """Export the replayed current view while keeping the event log canonical."""
        state = self.replay()
        output_path = Path(output_path)
        selected = [
            event
            for event in state.active.values()
            if (task is None or event.task == task)
            and (include_skips or event.label_kind != "skip")
        ]
        selected.sort(key=lambda event: (event.task, event.source_id, event.frame))
        metadata = {
            "type": "metadata",
            "schema": "annotation_export",
            "schema_version": 1,
            "event_log": self.path.name,
            "event_segments": [
                {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in self._log_paths()
            ],
            "revision_count": len(state.events),
            "label_count": len(selected),
            "ignored_interrupted_tail": state.ignored_interrupted_tail,
            "recovered_interrupted_tails": state.recovered_interrupted_tails,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output:
            output.write(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n")
            for event in selected:
                record = event.to_mapping()
                record["type"] = "label"
                output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        return output_path

    def undo_last(self, *, annotator: str, session_id: str) -> AnnotationEvent:
        """Append an undo revision for this session's latest still-current edit."""
        state = self.replay()
        for previous in reversed(state.events):
            if previous.annotator != annotator or previous.session_id != session_id:
                continue
            if previous.label_kind == "undo":
                continue
            head = state.heads.get(previous.key)
            if head is None or head.revision_id != previous.revision_id:
                continue
            return self.record(
                task=previous.task,
                source_id=previous.source_id,
                frame=previous.frame,
                label_kind="undo",
                candidate_artifact_sha256=previous.candidate_artifact_sha256,
                annotator=annotator,
                session_id=session_id,
            )
        raise ValueError("this session has no current revision to undo")

    def audit_view(self, queue: "AnnotationQueue") -> list[dict[str, object]]:
        """Return audit-queue completion rows without changing queue order."""
        if queue.kind != "audit":
            raise ValueError("audit_view requires an audit queue")
        state = self.replay()
        rows: list[dict[str, object]] = []
        for burst in queue.bursts:
            for frame in burst.frames:
                key = burst.task, burst.source_id, frame
                event = state.active.get(key)
                rows.append(
                    {
                        "burst_id": burst.burst_id,
                        "task": burst.task,
                        "source_id": burst.source_id,
                        "frame": frame,
                        "completed": event is not None and event.label_kind != "skip",
                        "label_kind": event.label_kind if event else None,
                        "revision_id": event.revision_id if event else None,
                    }
                )
        return rows

    @property
    def _segment_directory(self) -> Path:
        return self.path.parent / f"{self.path.name}.segments"

    def _log_paths(self) -> list[Path]:
        paths = [self.path] if self.path.exists() else []
        if self._segment_directory.exists():
            paths.extend(sorted(self._segment_directory.glob("[0-9][0-9][0-9][0-9].jsonl")))
        return paths

    def _recovery_path(self, new_segment: Path) -> Path:
        return new_segment.with_suffix(".recovery.json")


if TYPE_CHECKING:
    from .queues import AnnotationQueue
