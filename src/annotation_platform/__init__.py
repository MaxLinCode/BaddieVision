"""Reusable local annotation platform.

The package core has no Dash dependency. Import ``create_dash_app`` explicitly
to construct the optional browser UI.
"""

from .core import AnnotationRegistry, LabelOption, SourceRegistration, TaskPlugin
from .events import AnnotationEvent, EventStore, ReplayState, replay_events
from .queues import (
    AnnotationQueue,
    QueueBurst,
    build_adaptive_queue,
    build_uniform_audit_queue,
    native_fps_burst,
    validate_queue,
)
from .sessions import SessionManager, SessionState
from .shuttle import SHUTTLE_TASK, ShuttleCandidateArtifact, ShuttleSelectionPlugin
from .views import PlaybackView, build_playback_view, render_center_frame

__all__ = [
    "AnnotationEvent",
    "AnnotationQueue",
    "AnnotationRegistry",
    "EventStore",
    "LabelOption",
    "QueueBurst",
    "PlaybackView",
    "ReplayState",
    "SHUTTLE_TASK",
    "SessionManager",
    "SessionState",
    "ShuttleCandidateArtifact",
    "ShuttleSelectionPlugin",
    "SourceRegistration",
    "TaskPlugin",
    "build_adaptive_queue",
    "build_playback_view",
    "build_uniform_audit_queue",
    "native_fps_burst",
    "render_center_frame",
    "replay_events",
    "validate_queue",
]
