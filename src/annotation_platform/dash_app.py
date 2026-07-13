"""Optional Dash browser UI for the annotation core."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .core import AnnotationRegistry
from .events import EventStore
from .queues import AnnotationQueue
from .sessions import SessionManager, SessionState
from .views import build_playback_view, render_center_frame, validate_source_video


def create_dash_app(
    registry: AnnotationRegistry,
    queue: AnnotationQueue,
    event_store: EventStore,
    session_manager: SessionManager,
    session: SessionState,
) -> Any:
    """Create a one-annotator local app; Dash is imported only on demand."""
    try:
        from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update
        from flask import Response, send_file
    except ImportError as exc:  # pragma: no cover - exercised in environments without UI deps
        raise RuntimeError("Dash UI dependencies are unavailable; install requirements.txt") from exc

    if session.queue_id != queue.queue_id:
        raise ValueError("session and queue do not match")
    task_plugin = registry.tasks[queue.task].plugin
    for source_id in sorted({burst.source_id for burst in queue.bursts}):
        source = registry.sources[source_id]
        validate_source_video(source, force=True)
        task_plugin.verify_artifact_fingerprint(source)
    app = Dash(__name__, suppress_callback_exceptions=True)

    @app.server.get("/annotation/video/<source_id>")
    def source_video(source_id: str) -> Any:
        source = registry.sources[source_id]
        validate_source_video(source)
        return send_file(source.video_path, conditional=True)

    @app.server.get("/annotation/frame/<source_id>/<int:frame>.jpg")
    def center_frame(source_id: str, frame: int) -> Any:
        image = render_center_frame(
            registry,
            task=queue.task,
            source_id=source_id,
            frame=frame,
        )
        return Response(image, mimetype="image/jpeg")

    app.layout = html.Div(
        [
            dcc.Store(id="session-state", data=asdict(session)),
            dcc.Store(id="playback-view"),
            dcc.Store(id="playback-time", data=0.0),
            dcc.Store(id="visible-suggestion"),
            dcc.Interval(id="playback-clock", interval=100, n_intervals=0),
            html.H2(task_plugin.display_name),
            html.Div(id="progress"),
            html.Div(
                [
                    html.Video(
                        id="context-video",
                        controls=True,
                        muted=True,
                        style={"width": "100%", "display": "block"},
                    ),
                    html.Div(
                        id="context-overlays",
                        style={
                            "position": "absolute",
                            "inset": "0",
                            "width": "100%",
                            "height": "100%",
                            "pointerEvents": "none",
                        },
                    ),
                ],
                style={"position": "relative", "maxWidth": "960px"},
            ),
            html.P("Playback covers one second before and after the selected center frame."),
            html.Img(id="center-image", style={"maxWidth": "960px", "width": "100%"}),
            html.Div(id="candidate-buttons"),
            html.Div(
                [
                    *[
                        html.Button(
                            option.title + (f" [{option.hotkey.upper()}]" if option.hotkey else ""),
                            id={"type": "label-option", "kind": option.kind},
                            **({"data-hotkey": option.hotkey.lower()} if option.hotkey else {}),
                        )
                        for option in task_plugin.label_options
                    ],
                    html.Button("Accept suggestion [Space]", id="accept-suggestion"),
                    html.Button("Undo [Backspace]", id="undo"),
                    html.Button("Previous [←]", id="previous"),
                    html.Button("Next [→]", id="next"),
                ],
                style={"display": "flex", "gap": "0.5rem", "flexWrap": "wrap"},
            ),
            html.Pre(id="status"),
            html.H3("Audit / revision history"),
            html.Pre(id="history"),
        ],
        style={"fontFamily": "sans-serif", "margin": "1rem auto", "maxWidth": "1000px"},
    )

    # Browser-level keyboard controls click the same visible buttons used by
    # Dash callbacks. Digit hotkeys select center-frame candidates.
    keyboard_script = """
    <script>
    let annotationLocked = false;
    document.addEventListener('keydown', function(e) {
      if (e.target && ['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;
      if (e.repeat) { e.preventDefault(); return; }
      const fixed = {' ':'accept-suggestion', Backspace:'undo', ArrowLeft:'previous', ArrowRight:'next'};
      const id = fixed[e.key] || fixed[e.key.toLowerCase()] ||
                 null;
      const element = id ? document.getElementById(id) :
                      (/^[1-9]$/.test(e.key) ? document.querySelector('[data-shortcut="' + e.key + '"]') :
                       document.querySelector('[data-hotkey="' + e.key.toLowerCase() + '"]'));
      if (element) {
        e.preventDefault();
        if (annotationLocked && !['previous','next'].includes(element.id)) return;
        if (!['previous','next'].includes(element.id)) {
          annotationLocked = true;
          window.setTimeout(() => { annotationLocked = false; }, 350);
        }
        element.click();
      }
    });
    </script>
    """
    app.index_string = app.index_string.replace("</body>", keyboard_script + "</body>")

    app.clientside_callback(
        """
        function(n, view) {
          const video = document.getElementById('context-video');
          if (!video) return 0;
          if (view && video.currentTime >= view.clip_end_seconds) video.pause();
          return video.currentTime;
        }
        """,
        Output("playback-time", "data"),
        Input("playback-clock", "n_intervals"),
        State("playback-view", "data"),
    )

    @app.callback(
        Output("playback-view", "data"),
        Output("context-video", "src"),
        Output("center-image", "src"),
        Output("candidate-buttons", "children"),
        Output("progress", "children"),
        Output("history", "children"),
        Output("visible-suggestion", "data"),
        Input("session-state", "data"),
    )
    def show_target(raw_state: dict[str, Any]) -> tuple[Any, ...]:
        state = SessionState(**raw_state)
        target = session_manager.current(state, queue)
        if target is None:
            return None, no_update, no_update, [], "Queue complete", "", None
        burst, frame = target
        view = build_playback_view(
            registry, task=burst.task, source_id=burst.source_id, center_frame=frame
        )
        overlay_data = {
            "source_id": burst.source_id,
            "fps": registry.sources[burst.source_id].fps,
            "image_size": task_plugin.image_size(registry.sources[burst.source_id]),
            "clip_start_seconds": view.clip_start_seconds,
            "clip_end_seconds": view.clip_end_seconds,
            "overlays": {str(key): list(value) for key, value in view.overlays_by_frame.items()},
        }
        suggestion = (
            task_plugin.suggestion(registry.sources[burst.source_id], frame)
            if queue.kind == "adaptive" and hasattr(task_plugin, "suggestion") else None
        )
        buttons = [
            html.Button(
                f"{index}: {candidate['candidate_id']}",
                id={"type": "candidate", "candidate_id": candidate["candidate_id"]},
                **({"data-shortcut": str(index)} if index <= 9 else {}),
                style=(
                    {"outline": "4px solid #ffb300", "fontWeight": "bold"}
                    if suggestion and suggestion.candidate_id == candidate["candidate_id"] else {}
                ),
            )
            for index, candidate in enumerate(
                [item for item in view.center_candidates if item.get("candidate_id")],
                start=1,
            )
        ]
        suggestion_text = ""
        if suggestion:
            suggestion_text = (
                f" · suggestion: {suggestion.semantic_label}"
                + (f" {suggestion.candidate_id}" if suggestion.candidate_id else "")
            )
        progress = html.Div(
            f"{queue.kind} queue · burst {state.burst_index + 1}/{len(queue.bursts)} · "
            f"frame {state.frame_index + 1}/{len(burst.frames)} · "
            f"{burst.source_id}:{frame}{suggestion_text}",
            style=(
                {"background": "#8b1e1e", "color": "white", "padding": "0.75rem",
                 "fontWeight": "bold"}
                if suggestion and suggestion.semantic_label == "no_shuttle" else {}
            ),
        )
        replayed = event_store.replay()
        key = burst.task, burst.source_id, frame
        revisions = [event.to_mapping() for event in replayed.events if event.key == key]
        active = replayed.active.get(key)
        history = {
            "queue": {"kind": queue.kind, "seed": queue.seed, "queue_id": queue.queue_id},
            "active_label": active.to_mapping() if active else None,
            "frame_revisions": revisions,
        }
        if queue.kind == "audit":
            rows = event_store.audit_view(queue)
            history["audit_completion"] = {
                "complete": sum(bool(row["completed"]) for row in rows),
                "total": len(rows),
            }
        return (
            overlay_data,
            f"/annotation/video/{burst.source_id}",
            f"/annotation/frame/{burst.source_id}/{frame}.jpg",
            buttons,
            progress,
            json.dumps(history, indent=2, sort_keys=True),
            asdict(suggestion) if suggestion else None,
        )

    app.clientside_callback(
        """
        function(view) {
          if (!view) return '';
          const video = document.getElementById('context-video');
          if (!video) return '';
          const start = function() {
            video.currentTime = view.clip_start_seconds;
            const play = video.play(); if (play) play.catch(() => {});
          };
          if (video.readyState >= 1) start(); else video.onloadedmetadata = start;
          return view.source_id + ':' + view.clip_start_seconds;
        }
        """,
        Output("context-video", "title"),
        Input("playback-view", "data"),
    )

    @app.callback(
        Output("context-overlays", "children"),
        Input("playback-time", "data"),
        State("playback-view", "data"),
    )
    def playback_overlays(seconds: float, view: dict[str, Any] | None) -> Any:
        if not view or not view.get("image_size"):
            return []
        frame = int(round(float(seconds) * float(view["fps"])))
        candidates = view["overlays"].get(str(frame), [])
        children = []
        width, height = view["image_size"]
        for candidate in candidates:
            center = candidate.get("center")
            if isinstance(center, (list, tuple)) and len(center) == 2:
                children.append(
                    html.Div(
                        style={
                            "position": "absolute",
                            "left": f"{100.0 * float(center[0]) / width}%",
                            "top": f"{100.0 * float(center[1]) / height}%",
                            "width": "16px",
                            "height": "16px",
                            "border": "3px solid #00e5ff",
                            "borderRadius": "50%",
                            "transform": "translate(-50%, -50%)",
                            "boxSizing": "border-box",
                        }
                    )
                )
        return children

    @app.callback(
        Output("session-state", "data"),
        Output("status", "children"),
        Input({"type": "label-option", "kind": ALL}, "n_clicks"),
        Input("undo", "n_clicks"),
        Input("accept-suggestion", "n_clicks"),
        Input("previous", "n_clicks"),
        Input("next", "n_clicks"),
        Input({"type": "candidate", "candidate_id": ALL}, "n_clicks"),
        State("session-state", "data"),
        prevent_initial_call=True,
    )
    def edit_target(
        label_clicks: list[int | None],
        undo: int | None,
        accept_suggestion: int | None,
        previous: int | None,
        next_: int | None,
        candidate_clicks: list[int | None],
        raw_state: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        state = SessionState(**raw_state)
        triggered = callback_context.triggered_id
        if triggered == "previous":
            state = session_manager.retreat(state, queue)
            return asdict(state), "Moved to previous frame"
        if triggered == "next":
            state = session_manager.advance(state, queue)
            return asdict(state), "Moved to next frame"
        if triggered == "undo":
            event = event_store.undo_last(annotator=state.annotator, session_id=state.session_id)
            state = session_manager.seek(state, queue, event.source_id, event.frame)
            return asdict(state), f"UNDO · prior frame {event.frame} · next frame {event.frame}"
        target = session_manager.current(state, queue)
        if target is None:
            return asdict(state), "Queue complete"
        burst, frame = target
        candidate_id = None
        label_kind = None
        suggestion = (
            task_plugin.suggestion(registry.sources[burst.source_id], frame)
            if queue.kind == "adaptive" and hasattr(task_plugin, "suggestion") else None
        )
        if triggered == "accept-suggestion":
            if suggestion is None:
                return asdict(state), "Space ignored: no visible suggestion"
            label_kind = suggestion.semantic_label
            candidate_id = suggestion.candidate_id
        if isinstance(triggered, dict) and triggered.get("type") == "label-option":
            label_kind = str(triggered["kind"])
        if isinstance(triggered, dict) and triggered.get("type") == "candidate":
            candidate_id = str(triggered["candidate_id"])
            label_kind = "selected"
        if label_kind is None:
            return asdict(state), "No annotation action"
        event = event_store.record(
            task=burst.task,
            source_id=burst.source_id,
            frame=frame,
            label_kind=label_kind,
            candidate_id=candidate_id,
            candidate_artifact_sha256=burst.candidate_artifact_sha256,
            annotator=state.annotator,
            session_id=state.session_id,
            annotation_suggestion=suggestion,
        )
        state = session_manager.advance(state, queue)
        next_target = session_manager.current(state, queue)
        next_text = "complete" if next_target is None else f"{next_target[0].source_id}:{next_target[1]}"
        return asdict(state), (
            f"{event.review_action}: {event.label_kind} · prior frame "
            f"{burst.source_id}:{frame} · next frame {next_text}"
        )

    return app
