"""Reusable local annotation platform.

The package core has no Dash dependency. Import ``create_dash_app`` explicitly
to construct the optional browser UI.
"""

from .core import AnnotationRegistry, AnnotationSuggestion, LabelOption, SourceRegistration, TaskPlugin
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
from .pilot import (
    PILOT_THRESHOLDS,
    evaluate_threshold_pilot,
    filter_pilot_artifact,
    freeze_threshold_policy,
    materialize_final_runtime,
    migrate_v1_runtime,
    rebind_queue,
    validate_artifact_lineage,
    wilson_interval,
)
from .views import PlaybackView, build_playback_view, draw_candidates, render_center_frame

__all__ = [
    "AnnotationEvent",
    "AnnotationQueue",
    "AnnotationRegistry",
    "AnnotationSuggestion",
    "EventStore",
    "LabelOption",
    "QueueBurst",
    "PlaybackView",
    "PILOT_THRESHOLDS",
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
    "draw_candidates",
    "evaluate_threshold_pilot",
    "filter_pilot_artifact",
    "freeze_threshold_policy",
    "materialize_final_runtime",
    "migrate_v1_runtime",
    "native_fps_burst",
    "render_center_frame",
    "replay_events",
    "rebind_queue",
    "validate_artifact_lineage",
    "validate_queue",
    "wilson_interval",
]
