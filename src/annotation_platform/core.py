"""Registries and plugin contracts for local annotation tasks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import cv2


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading large videos into memory."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True)
class SourceRegistration:
    """A video source and the immutable artifacts used to annotate it."""

    source_id: str
    video_path: Path
    fps: float
    frame_count: int
    video_sha256: str
    video_size: int
    video_mtime_ns: int
    image_size: tuple[int, int]
    artifacts: Mapping[str, Path]

    @classmethod
    def create(
        cls,
        source_id: str,
        video_path: str | Path,
        *,
        fps: float,
        frame_count: int,
        image_size: tuple[int, int] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
    ) -> "SourceRegistration":
        source_id = str(source_id).strip()
        if not source_id:
            raise ValueError("source_id must not be empty")
        if fps <= 0:
            raise ValueError("source fps must be positive")
        if frame_count < 1:
            raise ValueError("source frame_count must be positive")
        resolved_video = Path(video_path).expanduser().resolve()
        if not resolved_video.is_file():
            raise ValueError(f"source video does not exist: {resolved_video}")
        capture = cv2.VideoCapture(str(resolved_video))
        try:
            probed_image_size = (
                int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
                int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            )
        finally:
            capture.release()
        if image_size is None:
            image_size = probed_image_size
        elif min(probed_image_size) > 0 and tuple(image_size) != probed_image_size:
            raise ValueError(
                f"declared source image_size {tuple(image_size)} does not match "
                f"decoded video size {probed_image_size}: {resolved_video}"
            )
        image_size = tuple(int(value) for value in image_size)
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError(f"could not resolve positive source image_size: {resolved_video}")
        artifact_paths = {
            str(name): Path(path).expanduser().resolve()
            for name, path in (artifacts or {}).items()
        }
        return cls(
            source_id=source_id,
            video_path=resolved_video,
            fps=float(fps),
            frame_count=int(frame_count),
            video_sha256=file_sha256(resolved_video),
            video_size=resolved_video.stat().st_size,
            video_mtime_ns=resolved_video.stat().st_mtime_ns,
            image_size=image_size,
            artifacts=MappingProxyType(artifact_paths),
        )


@dataclass(frozen=True)
class LabelOption:
    kind: str
    title: str
    hotkey: str | None = None


class TaskPlugin(Protocol):
    """Interface implemented by an annotation task.

    Plugins own artifact interpretation and task-specific validation. The core
    owns event history, queue/session state, and export.
    """

    task_name: str
    display_name: str

    @property
    def label_options(self) -> Sequence[LabelOption]: ...

    def prepare_source(self, source: SourceRegistration) -> None: ...

    def artifact_sha256(self, source: SourceRegistration) -> str: ...

    def verify_artifact_fingerprint(self, source: SourceRegistration) -> str: ...

    def eligible_frames(self, source: SourceRegistration) -> Sequence[int]: ...

    def image_size(self, source: SourceRegistration) -> tuple[int, int]: ...

    def validate_label(
        self,
        source: SourceRegistration,
        *,
        frame: int,
        label_kind: str,
        candidate_id: str | None,
        candidate_artifact_sha256: str,
    ) -> None: ...

    def overlays(self, source: SourceRegistration, frame: int) -> Sequence[Mapping[str, Any]]: ...

    def queue_score(self, source: SourceRegistration, frame: int) -> float: ...


@dataclass(frozen=True)
class TaskRegistration:
    task_name: str
    plugin: TaskPlugin
    source_ids: tuple[str, ...]


class AnnotationRegistry:
    """Shared source/task registry used by queues, event validation, and UI."""

    def __init__(self) -> None:
        self._sources: dict[str, SourceRegistration] = {}
        self._tasks: dict[str, TaskRegistration] = {}

    @property
    def sources(self) -> Mapping[str, SourceRegistration]:
        return MappingProxyType(self._sources)

    @property
    def tasks(self) -> Mapping[str, TaskRegistration]:
        return MappingProxyType(self._tasks)

    def register_source(
        self,
        source_id: str,
        video_path: str | Path,
        *,
        fps: float,
        frame_count: int,
        image_size: tuple[int, int] | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
    ) -> SourceRegistration:
        source = SourceRegistration.create(
            source_id,
            video_path,
            fps=fps,
            frame_count=frame_count,
            image_size=image_size,
            artifacts=artifacts,
        )
        if source.source_id in self._sources:
            raise ValueError(f"source already registered: {source.source_id}")
        self._sources[source.source_id] = source
        return source

    def register_task(
        self,
        plugin: TaskPlugin,
        *,
        source_ids: Sequence[str] | None = None,
    ) -> TaskRegistration:
        task_name = str(plugin.task_name).strip()
        if not task_name:
            raise ValueError("task plugin must define a non-empty task_name")
        if task_name in self._tasks:
            raise ValueError(f"task already registered: {task_name}")
        ids = tuple(source_ids if source_ids is not None else self._sources)
        if not ids:
            raise ValueError("a task must be registered against at least one source")
        if len(set(ids)) != len(ids):
            raise ValueError("task source IDs must be unique")
        for source_id in ids:
            if source_id not in self._sources:
                raise KeyError(f"unknown source: {source_id}")
            plugin.prepare_source(self._sources[source_id])
        registration = TaskRegistration(task_name, plugin, ids)
        self._tasks[task_name] = registration
        return registration

    def resolve(self, task_name: str, source_id: str) -> tuple[TaskPlugin, SourceRegistration]:
        try:
            task = self._tasks[task_name]
        except KeyError as exc:
            raise KeyError(f"unknown task: {task_name}") from exc
        if source_id not in task.source_ids:
            raise KeyError(f"task {task_name!r} is not registered for source {source_id!r}")
        return task.plugin, self._sources[source_id]
