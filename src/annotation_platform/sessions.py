"""Small durable session cursors over annotation queues."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

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


class SessionManager:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

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
        if self._path(state.session_id).exists():
            raise ValueError(f"session already exists: {state.session_id}")
        self._write(state)
        return state

    def load(self, session_id: str) -> SessionState:
        value = json.loads(self._path(session_id).read_text(encoding="utf-8"))
        return SessionState(**value)

    def current(self, state: SessionState, queue: AnnotationQueue) -> tuple[QueueBurst, int] | None:
        self._validate_queue(state, queue)
        if state.burst_index >= len(queue.bursts):
            return None
        burst = queue.bursts[state.burst_index]
        if state.frame_index >= len(burst.frames):
            return None
        return burst, burst.frames[state.frame_index]

    def advance(self, state: SessionState, queue: AnnotationQueue) -> SessionState:
        self._validate_queue(state, queue)
        burst_index, frame_index = state.burst_index, state.frame_index + 1
        if burst_index < len(queue.bursts) and frame_index >= len(queue.bursts[burst_index].frames):
            burst_index += 1
            frame_index = 0
        updated = SessionState(
            **{
                **asdict(state),
                "burst_index": burst_index,
                "frame_index": frame_index,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._write(updated)
        return updated

    def retreat(self, state: SessionState, queue: AnnotationQueue) -> SessionState:
        self._validate_queue(state, queue)
        burst_index, frame_index = state.burst_index, state.frame_index - 1
        if frame_index < 0 and burst_index > 0:
            burst_index -= 1
            frame_index = len(queue.bursts[burst_index].frames) - 1
        else:
            frame_index = max(0, frame_index)
        updated = SessionState(
            **{
                **asdict(state),
                "burst_index": burst_index,
                "frame_index": frame_index,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._write(updated)
        return updated

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
