"""Small durable session cursors over annotation queues."""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from .events import LabelKey
from .queues import AnnotationQueue, QueueBurst


@dataclass(frozen=True)
class SessionState:
    session_id: str
    annotator: str
    queue_id: str
    burst_index: int
    frame_index: int
    started_at: str
    updated_at: str
    # Added after the first session schema shipped.  A default keeps every
    # existing JSON session readable and gives browser callbacks an optimistic
    # concurrency token.
    cursor_revision: int = 0


class SessionConflictError(ValueError):
    """The caller acted on a cursor which is no longer current."""


class SessionManager:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._lock = threading.RLock()

    @contextmanager
    def serialized(self) -> Iterator[None]:
        """Serialize event-plus-cursor actions in this local server process."""
        with self._lock:
            yield

    def create(self, annotator: str, queue: AnnotationQueue, *, session_id: str | None = None) -> SessionState:
        annotator = str(annotator).strip()
        if not annotator:
            raise ValueError("annotator must not be empty")
        if not queue.bursts:
            raise ValueError("cannot start a session on an empty queue")
        now = datetime.now(timezone.utc).isoformat()
        state = SessionState(
            session_id=session_id or str(uuid.uuid4()),
            annotator=annotator,
            queue_id=queue.queue_id,
            burst_index=0,
            frame_index=0,
            started_at=now,
            updated_at=now,
        )
        with self._lock:
            if self._path(state.session_id).exists():
                raise ValueError(f"session already exists: {state.session_id}")
            self._write(state)
        return state

    def load(self, session_id: str) -> SessionState:
        with self._lock:
            value = json.loads(self._path(session_id).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"invalid session JSON: {session_id}")
        value["cursor_revision"] = int(value.get("cursor_revision") or 0)
        return SessionState(**value)

    def load_compatible(self, session_id: str, annotator: str, queue: AnnotationQueue) -> SessionState:
        state = self.load(session_id)
        self._validate_compatible(state, annotator, queue)
        return state

    def select(
        self,
        annotator: str,
        queue: AnnotationQueue,
        active_labels: Iterable[LabelKey] = (),
    ) -> SessionState | None:
        """Return the furthest compatible session, then the most recent one."""
        candidates: list[SessionState] = []
        if not self.directory.exists():
            return None
        for path in sorted(self.directory.glob("*.json")):
            try:
                state = self.load(path.stem)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if state.annotator == annotator and state.queue_id == queue.queue_id:
                candidates.append(state)
        if not candidates:
            return None
        completed = set(active_labels)
        return max(
            candidates,
            key=lambda state: (self._reconciled_position(state, queue, completed), state.updated_at),
        )

    def _reconciled_position(
        self, state: SessionState, queue: AnnotationQueue, completed: set[LabelKey]
    ) -> int:
        position = self.position(state, queue)
        flattened = [
            (burst.task, burst.source_id, frame)
            for burst in queue.bursts
            for frame in burst.frames
        ]
        while position < len(flattened) and flattened[position] in completed:
            position += 1
        return position

    def current(self, state: SessionState, queue: AnnotationQueue) -> tuple[QueueBurst, int] | None:
        self._validate_queue(state, queue)
        if state.burst_index >= len(queue.bursts):
            return None
        burst = queue.bursts[state.burst_index]
        if state.frame_index >= len(burst.frames):
            return None
        return burst, burst.frames[state.frame_index]

    def position(self, state: SessionState, queue: AnnotationQueue) -> int:
        """Return the zero-based flattened cursor (queue length when complete)."""
        self._validate_queue(state, queue)
        preceding = sum(len(burst.frames) for burst in queue.bursts[:state.burst_index])
        return min(preceding + state.frame_index, self.queue_length(queue))

    @staticmethod
    def queue_length(queue: AnnotationQueue) -> int:
        return sum(len(burst.frames) for burst in queue.bursts)

    def advance(self, state: SessionState, queue: AnnotationQueue) -> SessionState:
        with self._lock:
            self._require_current(state)
            return self._move(state, queue, 1)

    def retreat(self, state: SessionState, queue: AnnotationQueue) -> SessionState:
        with self._lock:
            self._require_current(state)
            return self._move(state, queue, -1)

    def reconcile(
        self,
        state: SessionState,
        queue: AnnotationQueue,
        active_labels: Iterable[LabelKey],
    ) -> SessionState:
        """Durably skip the labeled prefix at the current cursor."""
        completed = set(active_labels)
        with self._lock:
            self._require_current(state)
            updated = state
            while True:
                target = self.current(updated, queue)
                if target is None:
                    break
                burst, frame = target
                if (burst.task, burst.source_id, frame) not in completed:
                    break
                updated = self._moved_state(updated, queue, 1)
            if updated != state:
                self._write(updated)
            return updated

    def seek(self, state: SessionState, queue: AnnotationQueue, source_id: str, frame: int) -> SessionState:
        """Move the durable cursor to an exact queue frame."""
        with self._lock:
            self._require_current(state)
            self._validate_queue(state, queue)
            for burst_index, burst in enumerate(queue.bursts):
                if burst.source_id != source_id:
                    continue
                for frame_index, queued_frame in enumerate(burst.frames):
                    if queued_frame == int(frame):
                        updated = self._updated(state, burst_index, frame_index)
                        self._write(updated)
                        return updated
        raise ValueError(f"frame is not present in session queue: {source_id}:{frame}")

    def _move(self, state: SessionState, queue: AnnotationQueue, delta: int) -> SessionState:
        self._validate_queue(state, queue)
        updated = self._moved_state(state, queue, delta)
        self._write(updated)
        return updated

    def _moved_state(self, state: SessionState, queue: AnnotationQueue, delta: int) -> SessionState:
        flat = max(0, min(self.position(state, queue) + delta, self.queue_length(queue)))
        remaining = flat
        for burst_index, burst in enumerate(queue.bursts):
            if remaining < len(burst.frames):
                return self._updated(state, burst_index, remaining)
            remaining -= len(burst.frames)
        return self._updated(state, len(queue.bursts), 0)

    @staticmethod
    def _updated(state: SessionState, burst_index: int, frame_index: int) -> SessionState:
        return SessionState(**{
            **asdict(state),
            "burst_index": burst_index,
            "frame_index": frame_index,
            "cursor_revision": state.cursor_revision + 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    def _require_current(self, state: SessionState) -> None:
        durable = self.load(state.session_id)
        if durable.cursor_revision != state.cursor_revision:
            raise SessionConflictError(
                f"stale session cursor revision {state.cursor_revision}; "
                f"current revision is {durable.cursor_revision}"
            )

    def _validate_compatible(self, state: SessionState, annotator: str, queue: AnnotationQueue) -> None:
        if state.annotator != str(annotator).strip():
            raise ValueError(
                f"session annotator mismatch: expected {annotator!r}, found {state.annotator!r}"
            )
        self._validate_queue(state, queue)

    def _validate_queue(self, state: SessionState, queue: AnnotationQueue) -> None:
        if state.queue_id != queue.queue_id:
            raise ValueError("session queue fingerprint does not match")

    def _path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"

    def _write(self, state: SessionState) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(state.session_id)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(state), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
