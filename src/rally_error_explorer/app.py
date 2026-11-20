"""Dash browser UI for the rally error explorer."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .data import ExplorerData, derive_gt_intervals
from .render import render_frame_jpeg


PRIMARY_COLOR = "#1f77b4"
COMPARE_COLOR = "#d62728"
GT_COLOR = "#2ca02c"
START_COLOR = "#17becf"
END_COLOR = "#9467bd"

LAYER_DEFS = (
    ("gt", "Ground truth intervals"),
    ("decoded", "Decoded intervals"),
    ("inplay", "Raw InPlay probability"),
    ("start", "Start head probability"),
    ("end", "End head probability"),
    ("detection", "Player detection"),
    ("pose", "Pose validity"),
    ("activity", "Activity"),
    ("shuttle", "Shuttle evidence"),
)


_INDEX_STRING = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        <!--[if IE]><script>
        alert("Dash v2.7+ does not support Internet Explorer. Please use a newer browser.");
        </script><![endif]-->
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
            <script>
            (function () {
                function isEditable(element) {
                    var tag = element && element.tagName;
                    return tag === "INPUT" || tag === "TEXTAREA" ||
                        tag === "SELECT" || (element && element.isContentEditable);
                }
                document.addEventListener("keydown", function (event) {
                    if (isEditable(event.target)) {
                        return;
                    }
                    var button = null;
                    if (event.key === "ArrowLeft") {
                        button = document.getElementById("prev-frame");
                    } else if (event.key === "ArrowRight") {
                        button = document.getElementById("next-frame");
                    }
                    if (!button) {
                        return;
                    }
                    event.preventDefault();
                    button.dispatchEvent(new MouseEvent("click", {
                        bubbles: true,
                        cancelable: true
                    }));
                });
            })();
            </script>
        </footer>
    </body>
</html>"""


def _runs(values: Sequence[bool]) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    start: int | None = None
    previous = 0
    for frame, value in zip(range(len(values)), values):
        if value and start is None:
            start = frame
        if not value and start is not None:
            output.append((start, previous))
            start = None
        previous = frame
    if start is not None:
        output.append((start, previous))
    return output


def _decimate(
    frames: Sequence[int], values: Sequence[float], max_points: int = 20000
) -> tuple[list[int], list[float], list[int]]:
    if len(frames) <= max_points:
        return list(frames), [float(value) for value in values], list(frames)
    per_bin = math.ceil(len(frames) / max(1, max_points // 2))
    xs: list[int] = []
    ys: list[float] = []
    custom: list[int] = []
    for start in range(0, len(frames), per_bin):
        chunk_frames = frames[start:start + per_bin]
        chunk_values = values[start:start + per_bin]
        if not chunk_frames:
            continue
        indices = sorted(
            {
                min(range(len(chunk_values)), key=lambda i: chunk_values[i]),
                max(range(len(chunk_values)), key=lambda i: chunk_values[i]),
            }
        )
        for index in indices:
            xs.append(int(chunk_frames[index]))
            ys.append(float(chunk_values[index]))
            custom.append(int(chunk_frames[index]))
    return xs, ys, custom


def _select_by_frames(
    full_frames: Sequence[int],
    values: Sequence[float],
    target_frames: Sequence[int],
) -> list[float]:
    mapping = dict(zip(full_frames, values))
    return [float(mapping.get(int(frame), 0.0)) for frame in target_frames]


def _shuttle_evidence(
    data: ExplorerData,
    source_id: str,
) -> tuple[str | None, list[float]]:
    series = data.runs["primary"].series.get(source_id)
    if series is None:
        return None, []
    sidecar = data.shuttle_sidecar.get(source_id, {})
    inline = bool(series.shuttle_rows)
    if inline:
        return "model_input", [
            1.0 if int(frame) in series.shuttle_rows else 0.0
            for frame in series.frames
        ]
    if sidecar:
        return "external_diagnostic", [
            float(sidecar.get(int(frame), {}).get("top_peak", 0.0))
            for frame in series.frames
        ]
    return None, []


def _add_interval_trace(
    figure: Any,
    frames: Sequence[int],
    intervals: Sequence[tuple[int, int]],
    *,
    name: str,
    color: str,
    row: int,
    y: float,
) -> None:
    if not intervals:
        return
    xs: list[int | None] = []
    ys: list[float] = []
    for start, end in intervals:
        xs.extend([int(start), int(end), None])
        ys.extend([y, y, y])
    figure.add_trace(
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": ys,
            "line": {"color": color, "width": 8},
            "name": name,
            "hoverinfo": "skip",
            "legendgroup": name,
        },
        row=row,
        col=1,
    )


def _add_probability_trace(
    figure: Any,
    frames: Sequence[int],
    values: Sequence[float],
    *,
    name: str,
    color: str,
    row: int,
    fps: float,
    dash: str | None = None,
    width: float = 1.8,
) -> None:
    xs, ys, custom = _decimate(frames, values)
    text = [f"{frame / fps:.2f}s" for frame in custom]
    figure.add_trace(
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": ys,
            "customdata": custom,
            "text": text,
            "line": {"color": color, "width": width, "dash": dash or "solid"},
            "name": name,
            "legendgroup": name,
            "hovertemplate": (
                "frame=%{customdata}<br>t=%{text}<br>"
                + name
                + "=%{y:.3f}<extra></extra>"
            ),
        },
        row=row,
        col=1,
    )


def build_figure(
    data: ExplorerData,
    source_id: str,
    *,
    show_compare: bool,
    visible_layers: Sequence[str] | None = None,
    current_frame: int | None = None,
) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    enabled = (
        set(visible_layers)
        if visible_layers is not None
        else {key for key, _ in LAYER_DEFS}
    )
    bundle = data.bundles[source_id]
    current_frame = (
        min(max(int(current_frame), 0), bundle.frame_count - 1)
        if current_frame is not None
        else None
    )
    primary = data.runs["primary"].series[source_id]
    compare = data.runs["compare"].series.get(source_id) if "compare" in data.runs else None
    fps = bundle.fps
    shuttle_provenance, shuttle_values = _shuttle_evidence(data, source_id)

    feature_specs: list[tuple[str, str, list[tuple[str, list[float], str, str]]]] = []
    player = data.player_series[source_id]
    if "detection" in enabled:
        feature_specs.append(
            (
                "detection",
                "Player detection",
                [
                    ("P1 detected", player.p1_detected, "#1f77b4", "%{y}"),
                    ("P2 detected", player.p2_detected, "#ff7f0e", "%{y}"),
                ],
            )
        )
    if "pose" in enabled:
        feature_specs.append(
            (
                "pose",
                "Pose validity",
                [
                    ("P1 pose valid", player.p1_pose_valid, "#1f77b4", "%{y:.2f}"),
                    ("P2 pose valid", player.p2_pose_valid, "#ff7f0e", "%{y:.2f}"),
                ],
            )
        )
    if "activity" in enabled:
        feature_specs.append(
            (
                "activity",
                "Activity",
                [
                    ("P1 activity", player.p1_activity, "#1f77b4", "%{y:.3f}"),
                    ("P2 activity", player.p2_activity, "#ff7f0e", "%{y:.3f}"),
                ],
            )
        )
    if shuttle_provenance and "shuttle" in enabled:
        label = (
            "Shuttle evidence (model input)"
            if shuttle_provenance == "model_input"
            else "Shuttle evidence (external diagnostic)"
        )
        feature_specs.append(
            ("shuttle", "Shuttle evidence", [(label, shuttle_values, "#e377c2", "%{y:.3f}")])
        )

    row_count = 1 + len(feature_specs)
    row_heights = [0.44] + (
        [0.50 / len(feature_specs)] * len(feature_specs) if feature_specs else []
    )
    titles = ["Probabilities and intervals"] + [title for _, title, _ in feature_specs]
    figure = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
        subplot_titles=titles,
    )

    if "gt" in enabled and any(target != -1 for target in primary.targets):
        gt_intervals = derive_gt_intervals(primary.frames, primary.targets)
        _add_interval_trace(
            figure, primary.frames, gt_intervals,
            name="Ground truth", color=GT_COLOR, row=1, y=0.04,
        )
    if "decoded" in enabled:
        primary_intervals = [
            (primary.frames[start], primary.frames[end])
            for start, end in _runs(primary.decoded)
        ]
        _add_interval_trace(
            figure, primary.frames, primary_intervals,
            name="Decoded primary", color=PRIMARY_COLOR, row=1, y=0.10,
        )
        if compare is not None and show_compare:
            compare_intervals = [
                (compare.frames[start], compare.frames[end])
                for start, end in _runs(compare.decoded)
            ]
            _add_interval_trace(
                figure, compare.frames, compare_intervals,
                name="Decoded compare", color=COMPARE_COLOR, row=1, y=0.16,
            )

    if "inplay" in enabled:
        _add_probability_trace(
            figure, primary.frames, primary.inplay_probability,
            name="P(InPlay) primary", color=PRIMARY_COLOR, row=1, fps=fps, width=3.0,
        )
        if compare is not None and show_compare:
            _add_probability_trace(
                figure, compare.frames, compare.inplay_probability,
                name="P(InPlay) compare", color=COMPARE_COLOR, row=1, fps=fps, width=3.0,
            )
    if "start" in enabled and primary.start_available:
        _add_probability_trace(
            figure, primary.frames, primary.start_probability,
            name="Start head primary", color=START_COLOR, row=1, fps=fps, dash="dot", width=2.2,
        )
        if compare is not None and show_compare and compare.start_available:
            _add_probability_trace(
                figure, compare.frames, compare.start_probability,
                name="Start head compare", color=START_COLOR, row=1, fps=fps, dash="dashdot", width=2.2,
            )
    if "end" in enabled and primary.end_available:
        _add_probability_trace(
            figure, primary.frames, primary.end_probability,
            name="End head primary", color=END_COLOR, row=1, fps=fps, dash="dash", width=2.2,
        )
        if compare is not None and show_compare and compare.end_available:
            _add_probability_trace(
                figure, compare.frames, compare.end_probability,
                name="End head compare", color=END_COLOR, row=1, fps=fps, dash="longdash", width=2.2,
            )

    if "inplay" in enabled or "start" in enabled or "end" in enabled:
        figure.add_hline(
            y=primary.decoder_threshold,
            line={"color": "#666666", "width": 1, "dash": "dot"},
            row=1,
            col=1,
            annotation_text=f"threshold {primary.decoder_threshold:.2f}",
            annotation_position="top right",
        )
        if (
            compare is not None
            and show_compare
            and abs(compare.decoder_threshold - primary.decoder_threshold) > 1e-6
        ):
            figure.add_hline(
                y=compare.decoder_threshold,
                line={"color": "#bbbbbb", "width": 1, "dash": "dot"},
                row=1,
                col=1,
            )

    for row_index, (spec_key, _, traces) in enumerate(feature_specs, start=2):
        for name, values, color, value_format in traces:
            if spec_key == "shuttle":
                xs, ys, custom = _decimate(primary.frames, values)
            else:
                selected = _select_by_frames(player.frames, values, primary.frames)
                xs, ys, custom = _decimate(primary.frames, selected)
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    customdata=custom,
                    mode="lines",
                    line={"color": color, "width": 1.8},
                    name=name,
                    showlegend=True,
                    hovertemplate="frame=%{customdata}<br>" + name + "=" + value_format + "<extra></extra>",
                ),
                row=row_index,
                col=1,
            )

    if current_frame is not None:
        for row_index in range(1, row_count + 1):
            figure.add_vline(
                x=current_frame,
                row=row_index,
                col=1,
                line={"color": "#111111", "width": 1.4, "dash": "dot"},
            )

    figure.update_xaxes(rangeslider={"visible": False}, row=row_count, col=1)
    figure.update_layout(
        height=760,
        hovermode="x unified",
        clickmode="event+select",
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 55, "r": 20, "t": 70, "b": 50},
        uirevision="constant",
    )
    return figure


def _resolve_shuttle(
    data: ExplorerData,
    source_id: str,
    frame: int,
    run_name: str,
) -> Mapping[str, Any] | None:
    run = data.runs.get(run_name)
    series = run.series.get(source_id) if run is not None else None
    if series is not None and frame in series.shuttle_rows:
        row = series.shuttle_rows[frame]
        selected = row.get("selected_position")
        candidates = row.get("candidate_positions") or {}
        position = selected or (next(iter(candidates.values())) if candidates else None)
        if not isinstance(position, Mapping):
            return None
        return {
            "top_position_normalized": position.get("peak_position_normalized")
            or position.get("center_normalized"),
            "top_bbox_normalized": position.get("bbox_normalized"),
            "candidate_count": int(row.get("candidate_count", 0)),
            "top_peak": 0.0,
        }
    sidecar = data.shuttle_sidecar.get(source_id, {})
    return sidecar.get(frame)


def _frame_info(data: ExplorerData, source_id: str, frame: int, shuttle_run: str) -> str:
    bundle = data.bundles[source_id]
    fps = bundle.fps
    lines = [
        f"{source_id}  frame {frame}  t={frame / fps:.2f}s  fps={fps:.3f}",
    ]
    primary = data.runs["primary"].series[source_id]
    index = primary.frames.index(frame) if frame in primary.frames else None
    for name, run in data.runs.items():
        series = run.series[source_id]
        idx = series.frames.index(frame) if frame in series.frames else None
        if idx is None:
            lines.append(f"{name}: no prediction")
            continue
        lines.append(
            f"{name}: P(InPlay)={series.inplay_probability[idx]:.3f} "
            f"decoded={'IN' if series.decoded[idx] else 'OUT'} "
            f"thr={series.decoder_threshold:.2f}"
        )
        if series.start_available:
            lines.append(f"{name}: start P={series.start_probability[idx]:.3f}")
        if series.end_available:
            lines.append(f"{name}: end P={series.end_probability[idx]:.3f}")
    if index is not None and primary.targets[index] != -1:
        lines.append(f"ground truth: {'IN PLAY' if primary.targets[index] == 1 else 'OUT OF PLAY'}")
    assignment = bundle.assignments.get(frame, {})
    for role in ("P1", "P2"):
        slot = assignment.get("slots", {}).get(role, {})
        item = slot.get("assignment")
        if not isinstance(item, Mapping) or not item.get("bbox"):
            lines.append(f"{role}: missing")
            continue
        pose = bundle.poses.get((frame, int(item["track_id"])))
        pose_status = pose.get("status") if isinstance(pose, Mapping) else "missing"
        lines.append(
            f"{role}: act={float(item.get('activity', 0.0)):.2f} "
            f"conf={float(slot.get('confidence', 0.0)):.2f} pose={pose_status}"
        )
    shuttle = _resolve_shuttle(data, source_id, frame, shuttle_run)
    if shuttle is not None:
        provenance = (
            "model input"
            if data.runs.get(shuttle_run, data.runs["primary"]).series.get(source_id) is not None
            and frame in data.runs.get(shuttle_run, data.runs["primary"]).series[source_id].shuttle_rows
            else "external diagnostic"
        )
        lines.append(
            f"shuttle ({provenance}): count={shuttle.get('candidate_count', 0)} "
            f"peak={float(shuttle.get('top_peak', 0.0)):.3f}"
        )
    return "\n".join(lines)


def create_app(data: ExplorerData) -> Any:
    try:
        from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
        from flask import Response, request
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Dash UI dependencies are unavailable; install requirements.txt") from exc

    if not data.source_ids:
        raise ValueError("explorer data has no sources")
    has_compare = "compare" in data.runs
    has_shuttle = any(_shuttle_evidence(data, source_id)[0] for source_id in data.source_ids)
    available_layers = [
        key for key, _ in LAYER_DEFS if key != "shuttle" or has_shuttle
    ]
    layer_options = [
        {"label": label, "value": key}
        for key, label in LAYER_DEFS
        if key in available_layers
    ]
    initial_source = data.source_ids[0]
    initial_figure = build_figure(
        data, initial_source, show_compare=has_compare, current_frame=0
    )
    initial_src = (
        f"/rally-explorer/frame/{initial_source}/0.jpg"
        f"?pose=1&court=0&shuttle={int(has_shuttle)}&run=primary"
    )
    initial_info = _frame_info(data, initial_source, 0, "primary")
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.index_string = _INDEX_STRING

    @app.server.get("/rally-explorer/frame/<source_id>/<int:frame>.jpg")
    def frame_image(source_id: str, frame: int):
        if source_id not in data.bundles:
            return Response("unknown source", status=404)
        bundle = data.bundles[source_id]
        frame = min(max(int(frame), 0), bundle.frame_count - 1)
        pose = request.args.get("pose", "1").lower() in {"1", "true", "yes", "on"}
        court = request.args.get("court", "0").lower() in {"1", "true", "yes", "on"}
        shuttle_on = request.args.get("shuttle", "1").lower() in {"1", "true", "yes", "on"}
        shuttle_run = request.args.get("run", "primary")
        shuttle = _resolve_shuttle(data, source_id, frame, shuttle_run) if shuttle_on else None
        try:
            payload = render_frame_jpeg(
                bundle,
                frame,
                show_pose=pose,
                show_court=court,
                show_shuttle=shuttle_on,
                shuttle=shuttle,
            )
            return Response(payload, mimetype="image/jpeg")
        except ValueError as exc:
            return Response(str(exc), status=404)

    source_options = [{"label": source_id, "value": source_id} for source_id in data.source_ids]
    app.layout = html.Div(
        [
            html.H2("Rally error explorer"),
            html.Div(
                [
                    dcc.Dropdown(
                        id="source-dropdown",
                        options=source_options,
                        value=data.source_ids[0],
                        clearable=False,
                        style={"maxWidth": "30rem", "marginBottom": "0.4rem"},
                    ),
                    dcc.Checklist(
                        id="compare-toggle",
                        options=[{"label": "Show comparison run", "value": "show"}],
                        value=["show"] if has_compare else [],
                        style={"display": "block" if has_compare else "none"},
                    ),
                    dcc.Checklist(
                        id="layer-toggle",
                        options=layer_options,
                        value=available_layers,
                        inline=True,
                        style={"marginTop": "0.4rem"},
                    ),
                ],
                style={"marginBottom": "0.6rem"},
            ),
            html.Div(
                [
                    html.P(warning, style={"color": "#b00020", "margin": "0.2rem 0"})
                    for warning in data.warnings
                ],
                id="warnings",
            ),
            html.H3("Frame viewer"),
            html.Small("Use ← / → arrow keys to step frames (click the page first)."),
            html.Div(
                [
                    html.Button("◀", id="prev-frame", n_clicks=0),
                    dcc.Input(
                        id="frame-input",
                        type="number",
                        min=0,
                        value=0,
                        style={"width": "9rem", "margin": "0 0.5rem"},
                    ),
                    html.Button("▶", id="next-frame", n_clicks=0),
                    dcc.Checklist(
                        id="pose-toggle",
                        options=[{"label": "Pose", "value": "pose"}],
                        value=["pose"],
                        inline=True,
                        style={"marginLeft": "1rem"},
                    ),
                    dcc.Checklist(
                        id="court-toggle",
                        options=[{"label": "Court", "value": "court"}],
                        value=[],
                        inline=True,
                        style={"marginLeft": "1rem"},
                    ),
                    dcc.Checklist(
                        id="shuttle-toggle",
                        options=[{"label": "Shuttle", "value": "shuttle"}],
                        value=["shuttle"] if has_shuttle else [],
                        inline=True,
                        style={"marginLeft": "1rem", "display": "inline-block" if has_shuttle else "none"},
                    ),
                    dcc.Dropdown(
                        id="shuttle-run",
                        options=[
                            {"label": "primary", "value": "primary"},
                            *([{"label": "compare", "value": "compare"}] if has_compare else []),
                        ],
                        value="primary",
                        clearable=False,
                        style={"maxWidth": "10rem", "marginLeft": "1rem", "display": "inline-block" if has_shuttle and has_compare else "none"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginBottom": "0.6rem"},
            ),
            html.Img(
                id="frame-image",
                src=initial_src,
                style={"display": "block", "maxWidth": "100%", "height": "auto", "background": "#111"},
            ),
            html.Pre(id="frame-info", children=initial_info, style={"whiteSpace": "pre-wrap"}),
            dcc.Graph(id="timeline", figure=initial_figure),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "sans-serif"},
    )

    @app.callback(
        Output("timeline", "figure"),
        Input("source-dropdown", "value"),
        Input("compare-toggle", "value"),
        Input("layer-toggle", "value"),
        Input("frame-input", "value"),
    )
    def update_timeline(
        source_id: str,
        compare_toggle: list[str] | None,
        layer_toggle: list[str] | None,
        frame: int | None,
    ) -> Any:
        return build_figure(
            data,
            source_id,
            show_compare=bool(compare_toggle and "show" in compare_toggle),
            visible_layers=layer_toggle,
            current_frame=frame,
        )

    @app.callback(
        Output("frame-input", "value"),
        Input("timeline", "clickData"),
        Input("prev-frame", "n_clicks"),
        Input("next-frame", "n_clicks"),
        State("frame-input", "value"),
        State("source-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_frame_input(
        click_data: Mapping[str, Any] | None,
        prev_clicks: int | None,
        next_clicks: int | None,
        frame: int | None,
        source_id: str,
    ):
        if not callback_context.triggered:
            raise no_update
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
        bundle = data.bundles[source_id]
        if triggered == "timeline":
            if not click_data or not click_data.get("points"):
                raise no_update
            point = click_data["points"][0]
            if point.get("customdata") is not None:
                return int(point["customdata"])
            return int(point.get("x", 0))
        if triggered in {"prev-frame", "next-frame"}:
            current = int(frame) if frame is not None else 0
            delta = -1 if triggered == "prev-frame" else 1
            return min(max(current + delta, 0), bundle.frame_count - 1)
        raise no_update

    @app.callback(
        Output("frame-image", "src"),
        Output("frame-info", "children"),
        Input("source-dropdown", "value"),
        Input("frame-input", "value"),
        Input("pose-toggle", "value"),
        Input("court-toggle", "value"),
        Input("shuttle-toggle", "value"),
        Input("shuttle-run", "value"),
    )
    def update_frame_view(
        source_id: str,
        frame: int | None,
        pose_toggle: list[str] | None,
        court_toggle: list[str] | None,
        shuttle_toggle: list[str] | None,
        shuttle_run: str,
    ):
        if source_id is None or frame is None:
            raise no_update
        bundle = data.bundles[source_id]
        frame = min(max(int(frame), 0), bundle.frame_count - 1)
        pose = bool(pose_toggle and "pose" in pose_toggle)
        court = bool(court_toggle and "court" in court_toggle)
        shuttle_on = bool(shuttle_toggle and "shuttle" in shuttle_toggle)
        shuttle_run = shuttle_run or "primary"
        src = (
            f"/rally-explorer/frame/{source_id}/{frame}.jpg"
            f"?pose={int(pose)}&court={int(court)}&shuttle={int(shuttle_on)}&run={shuttle_run}"
        )
        return src, _frame_info(data, source_id, frame, shuttle_run)

    return app
