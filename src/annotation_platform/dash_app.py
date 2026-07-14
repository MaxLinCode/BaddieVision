"""Optional Dash browser UI for the annotation core."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .core import AnnotationRegistry
from .events import EventStore
from .queues import AnnotationQueue
from .sessions import SessionManager, SessionState
from .views import candidate_display_items, render_center_frame, render_raw_frame, validate_source_video

IMAGE_ONLY_HOTKEY = "o"


def step_preview_frame(current: int, center: int, delta: int, frame_count: int) -> int:
    """Step a motion preview without moving its annotation center."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    current = min(max(int(current), 0), frame_count - 1)
    center = min(max(int(center), 0), frame_count - 1)
    if delta == 0:
        return center
    return min(max(current + int(delta), 0), frame_count - 1)


def candidate_click_matches_target(
    triggered: Any, source_id: str, frame: int
) -> bool:
    """Return whether a dynamic candidate control belongs to the active target."""
    if not isinstance(triggered, dict) or triggered.get("type") != "candidate":
        return False
    try:
        clicked_frame = int(triggered.get("frame", -1))
    except (TypeError, ValueError):
        return False
    return triggered.get("source_id") == source_id and clicked_frame == int(frame)


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
        from flask import Response, request
    except ImportError as exc:  # pragma: no cover - exercised in environments without UI deps
        raise RuntimeError("Dash UI dependencies are unavailable; install requirements.txt") from exc

    if session.queue_id != queue.queue_id:
        raise ValueError("session and queue do not match")
    task_plugin = registry.tasks[queue.task].plugin
    assigned_hotkeys = {
        option.hotkey.lower() for option in task_plugin.label_options if option.hotkey
    }
    if IMAGE_ONLY_HOTKEY in assigned_hotkeys:
        raise ValueError(
            f"image-only hotkey {IMAGE_ONLY_HOTKEY.upper()!r} conflicts with a task label hotkey"
        )
    for source_id in sorted({burst.source_id for burst in queue.bursts}):
        source = registry.sources[source_id]
        validate_source_video(source, force=True)
        task_plugin.verify_artifact_fingerprint(source)
    app = Dash(__name__, suppress_callback_exceptions=True)
    # A correction token is deliberately process-local: Previous permits a
    # labeled target during this live interaction, while a refresh/startup
    # returns to normal event reconciliation.
    correction_cursors: set[tuple[str, int]] = set()

    @app.server.get("/annotation/frame/<source_id>/<int:frame>.jpg")
    def center_frame(source_id: str, frame: int) -> Any:
        verbose = request.args.get("verbose", "0").lower() in {"1", "true", "yes", "on"}
        candidate_view = request.args.get("view", "grouped")
        raw_minimum = request.args.get("minimum_threshold")
        minimum_threshold = None if raw_minimum in {None, "", "all"} else float(raw_minimum)
        source = registry.sources[source_id]
        suggestion = (
            task_plugin.suggestion(source, frame)
            if queue.kind == "adaptive" and hasattr(task_plugin, "suggestion") else None
        )
        highlighted_candidate_id = (
            task_plugin.representative_candidate_id(source, frame, suggestion.candidate_id)
            if suggestion and suggestion.candidate_id else None
        )
        image = render_center_frame(
            registry,
            task=queue.task,
            source_id=source_id,
            frame=frame,
            verbose=verbose,
            highlighted_candidate_id=highlighted_candidate_id,
            candidate_view=candidate_view,
            minimum_threshold=minimum_threshold,
        )
        return Response(image, mimetype="image/jpeg")

    @app.server.get("/annotation/preview/<source_id>/<int:frame>.jpg")
    def preview_frame(source_id: str, frame: int) -> Any:
        image = render_raw_frame(registry.sources[source_id], frame)
        return Response(image, mimetype="image/jpeg")

    def serve_layout() -> Any:
        # A browser refresh must never resurrect the server-start cursor.  It
        # also closes the crash window where the event reached disk but the
        # following session write did not.
        with session_manager.serialized():
            durable = session_manager.load(session.session_id)
            durable = session_manager.reconcile(durable, queue, event_store.replay().active)
        return html.Div(
            [
            dcc.Store(id="session-state", data=asdict(durable)),
            dcc.Store(id="preview-frame"),
            dcc.Store(id="visible-suggestion"),
            html.H2(task_plugin.display_name),
            html.Div(id="progress"),
            html.Div(
                [
                    html.Img(
                        id="center-image",
                        style={"display": "block", "height": "auto", "width": "100%"},
                    ),
                ],
                style={
                    "background": "#111",
                    "width": "100%",
                },
            ),
            html.P("Browse one native frame at a time. Numbers and colors match the controls below."),
            html.Div(id="preview-status"),
            html.Small(
                "Hold O for a clean image. I = inferable, N = no in-frame target, "
                "M = missing proposal, U = unsure."
            ),
            dcc.Checklist(
                id="verbose-candidate-labels",
                options=[{"label": "Verbose candidate labels", "value": "verbose"}],
                value=[],
                inline=True,
                style={"fontSize": "0.9rem", "margin": "0.35rem 0"},
            ),
            dcc.Dropdown(
                id="candidate-view",
                options=[
                    {"label": "Grouped candidates", "value": "grouped"},
                    {"label": "Raw candidates", "value": "raw"},
                ],
                value="grouped",
                clearable=False,
                style={"maxWidth": "20rem", "margin": "0.35rem 0"},
            ),
            dcc.Dropdown(
                id="minimum-threshold",
                options=[{"label": "All thresholds", "value": "all"}] + [
                    {"label": f"Minimum {value:.2f}", "value": f"{value:.2f}"}
                    for value in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
                ],
                value="all",
                clearable=False,
                style={"maxWidth": "20rem", "margin": "0.35rem 0"},
            ),
            html.Div(id="candidate-legend"),
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
                    html.Button("Earlier preview [,]", id="preview-previous"),
                    html.Button("Center frame [/]", id="preview-center"),
                    html.Button("Later preview [.]", id="preview-next"),
                    html.Button("Previous target [←]", id="previous"),
                    html.Button("Next target [→]", id="next"),
                ],
                style={"display": "flex", "gap": "0.5rem", "flexWrap": "wrap"},
            ),
            html.Pre(id="status"),
            html.H3("Audit / revision history"),
            html.Pre(id="history"),
            ],
            style={
                "fontFamily": "sans-serif",
                "fontSize": "18px",
                "margin": "0.75rem auto",
                "maxWidth": "1500px",
                "padding": "0 1rem",
            },
        )

    app.layout = serve_layout

    # Browser-level keyboard controls click the same visible buttons used by
    # Dash callbacks. Digit hotkeys select center-frame candidates.
    keyboard_script = """
    <script>
    let annotationLocked = false;
    let imageOnlyTimer = null;
    let imageOnlyActive = false;
    let imageOnlyAnnotatedSrc = null;
    let imageOnlyCleanSrc = null;

    function activateAnnotationControl(element) {
      if (!element) return;
      const navigation = ['previous','next','preview-previous','preview-center','preview-next'];
      if (annotationLocked && !navigation.includes(element.id)) return;
      if (!navigation.includes(element.id)) {
        annotationLocked = true;
        window.setTimeout(() => { annotationLocked = false; }, 350);
      }
      element.click();
    }

    function restoreImageOnly() {
      if (imageOnlyTimer) window.clearTimeout(imageOnlyTimer);
      imageOnlyTimer = null;
      if (imageOnlyActive) {
        const image = document.getElementById('center-image');
        if (image && imageOnlyAnnotatedSrc && image.src === imageOnlyCleanSrc) {
          image.src = imageOnlyAnnotatedSrc;
        }
      }
      imageOnlyActive = false;
      imageOnlyAnnotatedSrc = null;
      imageOnlyCleanSrc = null;
    }

    document.addEventListener('keydown', function(e) {
      if (e.target && ['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;
      if (e.key.toLowerCase() === 'o') {
        e.preventDefault();
        if (e.repeat || imageOnlyTimer || imageOnlyActive) return;
        imageOnlyTimer = window.setTimeout(() => {
          imageOnlyTimer = null;
          imageOnlyActive = true;
          const image = document.getElementById('center-image');
          if (!image) return;
          imageOnlyAnnotatedSrc = image.src;
          const clean = new URL(image.src, window.location.href);
          clean.pathname = clean.pathname.replace('/annotation/frame/', '/annotation/preview/');
          clean.search = '';
          imageOnlyCleanSrc = clean.toString();
          image.src = imageOnlyCleanSrc;
        }, 250);
        return;
      }
      if (e.repeat) { e.preventDefault(); return; }
      const fixed = {
        ' ':'accept-suggestion', Backspace:'undo',
        ArrowLeft:'previous', ArrowRight:'next',
        ',':'preview-previous', '<':'preview-previous',
        '.':'preview-next', '>':'preview-next',
        '/':'preview-center'
      };
      const id = fixed[e.key] || fixed[e.key.toLowerCase()] ||
                 null;
      const element = id ? document.getElementById(id) :
                      (/^[1-9]$/.test(e.key) ? document.querySelector('[data-shortcut="' + e.key + '"]') :
                       document.querySelector('[data-hotkey="' + e.key.toLowerCase() + '"]'));
      if (element) {
        e.preventDefault();
        activateAnnotationControl(element);
      }
    });
    document.addEventListener('keyup', function(e) {
      if (e.target && ['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;
      if (e.key.toLowerCase() !== 'o') return;
      e.preventDefault();
      restoreImageOnly();
    });
    window.addEventListener('blur', restoreImageOnly);
    </script>
    """
    app.index_string = app.index_string.replace("</body>", keyboard_script + "</body>")

    @app.callback(
        Output("preview-frame", "data"),
        Input("session-state", "data"),
        Input("preview-previous", "n_clicks"),
        Input("preview-center", "n_clicks"),
        Input("preview-next", "n_clicks"),
        State("preview-frame", "data"),
    )
    def navigate_preview(
        raw_state: dict[str, Any],
        previous: int | None,
        center_clicks: int | None,
        next_: int | None,
        preview: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        state = session_manager.load(str(raw_state["session_id"]))
        target = session_manager.current(state, queue)
        if target is None:
            return None
        burst, center = target
        source = registry.sources[burst.source_id]
        triggered = callback_context.triggered_id
        same_target = bool(
            preview
            and preview.get("source_id") == burst.source_id
            and int(preview.get("center", -1)) == center
        )
        current = int(preview["frame"]) if same_target else center
        delta = -1 if triggered == "preview-previous" else 1 if triggered == "preview-next" else 0
        frame = step_preview_frame(current, center, delta, source.frame_count)
        return {"source_id": burst.source_id, "center": center, "frame": frame}

    @app.callback(
        Output("center-image", "src"),
        Output("preview-status", "children"),
        Input("preview-frame", "data"),
        Input("verbose-candidate-labels", "value"),
        Input("candidate-view", "value"),
        Input("minimum-threshold", "value"),
    )
    def show_preview(
        preview: dict[str, Any] | None,
        verbose_labels: list[str] | None,
        candidate_view: str = "grouped",
        minimum_threshold: str = "all",
    ) -> tuple[Any, str]:
        if not preview:
            return no_update, ""
        source_id = str(preview["source_id"])
        center = int(preview["center"])
        frame = int(preview["frame"])
        if frame == center:
            verbose = 1 if verbose_labels and "verbose" in verbose_labels else 0
            src = (
                f"/annotation/frame/{source_id}/{frame}.jpg?verbose={verbose}"
                f"&view={candidate_view}&minimum_threshold={minimum_threshold}"
            )
            status = f"Center frame {frame} · annotations apply here"
        else:
            src = f"/annotation/preview/{source_id}/{frame}.jpg"
            offset = frame - center
            status = (
                f"Motion preview frame {frame} · {offset:+d} from center {center} · "
                "labels still apply to center"
            )
        return src, status

    @app.callback(
        Output("candidate-buttons", "children"),
        Output("candidate-legend", "children"),
        Output("progress", "children"),
        Output("history", "children"),
        Output("visible-suggestion", "data"),
        Input("session-state", "data"),
        Input("candidate-view", "value"),
        Input("minimum-threshold", "value"),
    )
    def show_target(
        raw_state: dict[str, Any],
        candidate_view: str = "grouped",
        minimum_threshold: str = "all",
    ) -> tuple[Any, ...]:
        state = session_manager.load(str(raw_state["session_id"]))
        target = session_manager.current(state, queue)
        if target is None:
            total = session_manager.queue_length(queue)
            return (
                [],
                [],
                f"session {state.session_id} · {queue.kind} queue · position {total}/{total} · complete",
                "",
                None,
            )
        burst, frame = target
        suggestion = (
            task_plugin.suggestion(registry.sources[burst.source_id], frame)
            if queue.kind == "adaptive" and hasattr(task_plugin, "suggestion") else None
        )
        source = registry.sources[burst.source_id]
        threshold = None if minimum_threshold in {None, "all"} else float(minimum_threshold)
        if hasattr(task_plugin, "display_overlays"):
            overlays = task_plugin.display_overlays(
                source, frame, view=candidate_view, minimum_threshold=threshold
            )
        else:
            overlays = task_plugin.overlays(source, frame)
        display_items = candidate_display_items(overlays)
        buttons = [
            html.Button(
                f"{item['number']} — {item['candidate_id']}",
                id={
                    "type": "candidate",
                    "source_id": burst.source_id,
                    "frame": frame,
                    "candidate_id": item["candidate_id"],
                },
                **({"data-shortcut": str(item["number"])} if item["number"] <= 9 else {}),
                style={
                    "backgroundColor": f"rgb{item['color']}", "color": "white",
                    "fontWeight": "bold", "textShadow": "0 1px 2px #000",
                    **(
                        {"border": "3px solid #ffd54f"}
                        if suggestion and (
                            task_plugin.representative_candidate_id(
                                source, frame, suggestion.candidate_id or ""
                            ) == item["candidate_id"]
                        ) else {}
                    ),
                },
            )
            for item in display_items
        ]
        legend = html.Div(
            [html.Span(
                f"{item['number']} = {item['candidate_id']}",
                style={
                    "backgroundColor": f"rgb{item['color']}", "color": "white",
                    "padding": "0.35rem 0.6rem", "fontWeight": "bold",
                    "border": "2px solid white", "textShadow": "0 1px 2px #000",
                },
            ) for item in display_items],
            style={"display": "flex", "gap": "0.5rem", "flexWrap": "wrap", "margin": "0.75rem 0"},
        )
        suggestion_text = ""
        if suggestion:
            representative_id = (
                task_plugin.representative_candidate_id(source, frame, suggestion.candidate_id)
                if suggestion.candidate_id else None
            )
            suggested_item = next(
                (item for item in display_items if item["candidate_id"] == representative_id),
                None,
            )
            suggestion_name = (
                f"candidate #{suggested_item['number']}" if suggested_item
                else suggestion.semantic_label.replace("_", " ").lower()
            )
            suggestion_text = f" · suggestion: {suggestion_name} · Space accepts"
        flat_position = session_manager.position(state, queue)
        progress = html.Div(
            f"session {state.session_id} · {queue.kind} queue · "
            f"position {flat_position + 1}/{session_manager.queue_length(queue)} · "
            f"burst {state.burst_index + 1}/{len(queue.bursts)} · "
            f"frame {state.frame_index + 1}/{len(burst.frames)} · "
            f"{burst.source_id}:{frame}{suggestion_text}",
            style=(
                {"background": "#8b1e1e", "color": "white", "padding": "0.75rem",
                 "fontWeight": "bold"}
                if suggestion and suggestion.semantic_label == "no_in_frame_target" else {}
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
            buttons,
            legend,
            progress,
            json.dumps(history, indent=2, sort_keys=True),
            asdict(suggestion) if suggestion else None,
        )

    @app.callback(
        Output("session-state", "data"),
        Output("status", "children"),
        Input({"type": "label-option", "kind": ALL}, "n_clicks"),
        Input("undo", "n_clicks"),
        Input("accept-suggestion", "n_clicks"),
        Input("previous", "n_clicks"),
        Input("next", "n_clicks"),
        Input(
            {
                "type": "candidate",
                "source_id": ALL,
                "frame": ALL,
                "candidate_id": ALL,
            },
            "n_clicks",
        ),
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
        submitted = SessionState(**raw_state)
        triggered = callback_context.triggered_id
        with session_manager.serialized():
            state = session_manager.load(submitted.session_id)
            correction_cursor = (state.session_id, state.cursor_revision)
            if correction_cursor not in correction_cursors:
                state = session_manager.reconcile(state, queue, event_store.replay().active)
            if (
                submitted.cursor_revision != state.cursor_revision
                or submitted.burst_index != state.burst_index
                or submitted.frame_index != state.frame_index
            ):
                return asdict(state), (
                    "Ignored stale action; reloaded the current durable cursor "
                    f"(revision {state.cursor_revision})"
                )
            if triggered == "previous":
                state = session_manager.retreat(state, queue)
                correction_cursors.add((state.session_id, state.cursor_revision))
                return asdict(state), "Moved to previous frame for correction"
            if triggered == "next":
                correction_cursors.discard(correction_cursor)
                state = session_manager.advance(state, queue)
                state = session_manager.reconcile(state, queue, event_store.replay().active)
                return asdict(state), "Moved to next unlabeled frame"
            if triggered == "undo":
                correction_cursors.discard(correction_cursor)
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
                if suggestion.candidate_id:
                    source = registry.sources[burst.source_id]
                    candidate_id = task_plugin.representative_candidate_id(
                        source, frame, suggestion.candidate_id
                    )
            if isinstance(triggered, dict) and triggered.get("type") == "label-option":
                label_kind = str(triggered["kind"])
            if isinstance(triggered, dict) and triggered.get("type") == "candidate":
                if not candidate_click_matches_target(triggered, burst.source_id, frame):
                    clicked_source = triggered.get("source_id", "unknown")
                    clicked_frame = triggered.get("frame", "unknown")
                    return asdict(state), (
                        f"Ignored stale candidate from {clicked_source}:{clicked_frame}; "
                        f"current target is {burst.source_id}:{frame}"
                    )
                candidate_id = str(triggered["candidate_id"])
                candidate_id = task_plugin.representative_candidate_id(
                    registry.sources[burst.source_id], frame, candidate_id
                )
                label_kind = "selected"
            if label_kind is None:
                return asdict(state), "No annotation action"
            annotation_metadata = None
            if label_kind == "selected":
                source = registry.sources[burst.source_id]
                overlays = getattr(task_plugin, "annotator_overlays", task_plugin.overlays)(source, frame)
                selected_overlay = next(
                    (item for item in overlays if item.get("candidate_id") == candidate_id),
                    None,
                )
                if selected_overlay is None:
                    return asdict(state), (
                        f"Ignored unavailable candidate {candidate_id!r}; "
                        f"current target is {burst.source_id}:{frame}"
                    )
                grouped = selected_overlay.get("grouped_candidate_ids", [candidate_id])
                annotation_metadata = {
                    "grouped_candidate_ids": list(grouped),
                    "raw_member_ids": list(
                        selected_overlay.get("raw_member_ids", grouped)
                    ),
                    "grouping_version": selected_overlay.get("grouping_version"),
                    "representative_candidate_id": candidate_id,
                }
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
                annotation_metadata=annotation_metadata,
            )
            correction_cursors.discard(correction_cursor)
            state = session_manager.advance(state, queue)
            state = session_manager.reconcile(state, queue, event_store.replay().active)
            next_target = session_manager.current(state, queue)
            next_text = "complete" if next_target is None else f"{next_target[0].source_id}:{next_target[1]}"
            return asdict(state), (
                f"{event.review_action}: {event.label_kind} · prior frame "
                f"{burst.source_id}:{frame} · next frame {next_text}"
            )

    return app
