"""Run the local shuttle annotation app and maintenance commands."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from pathlib import Path

import cv2

from .core import AnnotationRegistry
from .dash_app import create_dash_app
from .events import EventStore
from .queues import (
    AnnotationQueue,
    build_adaptive_queue,
    build_uniform_audit_queue,
    validate_queue,
)
from .pilot import (
    evaluate_threshold_pilot,
    filter_pilot_artifact,
    freeze_threshold_policy,
    materialize_final_runtime,
    migrate_v1_runtime,
    wilson_interval,
)
from .sessions import SessionManager, SessionState
from .shuttle import SHUTTLE_TASK, ShuttleSelectionPlugin
from .views import build_playback_view, render_center_frame


DEFAULT_RUNTIME = Path(".annotation")


def _video_geometry(path: Path) -> tuple[float, int, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        image_size = (
            int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
    finally:
        capture.release()
    if fps <= 0 or frame_count < 1 or min(image_size) <= 0:
        raise ValueError(f"could not probe video geometry: {path}")
    return fps, frame_count, image_size


def load_registry(config_path: str | Path) -> AnnotationRegistry:
    """Load a reusable source catalog; paths are relative to its JSON file."""
    config_path = Path(config_path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("sources"), list):
        raise ValueError("annotation config requires a sources list")
    registry = AnnotationRegistry()
    for raw in config["sources"]:
        source_id = str(raw["source_id"])
        video_path = (config_path.parent / raw["video_path"]).resolve()
        candidates_path = (config_path.parent / raw["candidates_path"]).resolve()
        if raw.get("fps") is None or raw.get("frame_count") is None:
            fps, frame_count, image_size = _video_geometry(video_path)
        else:
            fps, frame_count = float(raw["fps"]), int(raw["frame_count"])
            raw_image_size = raw.get("image_size")
            image_size = (
                tuple(int(value) for value in raw_image_size)
                if raw_image_size is not None
                else None
            )
        registry.register_source(
            source_id,
            video_path,
            fps=fps,
            frame_count=frame_count,
            image_size=image_size,
            artifacts={"candidates": candidates_path},
        )
    registry.register_task(ShuttleSelectionPlugin())
    return registry


def _queues(
    registry: AnnotationRegistry,
    runtime: Path,
    *,
    audit_seed: int,
    audit_count: int,
    adaptive_count: int,
) -> tuple[AnnotationQueue, AnnotationQueue]:
    queue_dir = runtime / "queues"
    audit_path = queue_dir / "shuttle-audit.json"
    adaptive_path = queue_dir / "shuttle-adaptive.json"
    if audit_path.exists():
        audit = AnnotationQueue.read(audit_path)
        validate_queue(registry, audit)
        if audit.seed != audit_seed or audit.construction.get("anchor_count") != audit_count:
            raise ValueError("persisted audit queue does not match requested seed/count")
    else:
        audit = build_uniform_audit_queue(
            registry,
            SHUTTLE_TASK,
            seed=audit_seed,
            anchor_count=audit_count,
            manifest_path=audit_path,
        )
    if adaptive_path.exists():
        adaptive = AnnotationQueue.read(adaptive_path)
        validate_queue(registry, adaptive)
        if adaptive.construction.get("requested_anchor_count") != adaptive_count:
            raise ValueError("persisted adaptive queue does not match requested count")
        if adaptive.construction.get("audit_queue_id") != audit.queue_id:
            raise ValueError("persisted adaptive queue references a different audit queue")
    else:
        adaptive = build_adaptive_queue(
            registry,
            SHUTTLE_TASK,
            anchor_count=adaptive_count,
            audit_queue=audit,
            manifest_path=adaptive_path,
        )
    return adaptive, audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Source catalog JSON")
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    subparsers = parser.add_subparsers(dest="command", required=True)

    queue = subparsers.add_parser("build-queues")
    queue.add_argument("--audit-seed", type=int, default=1729)
    queue.add_argument("--audit-count", type=int, default=10)
    queue.add_argument("--adaptive-count", type=int, default=20)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--annotator", required=True)
    serve.add_argument("--queue", choices=("adaptive", "audit"), default="adaptive")
    serve.add_argument("--audit-seed", type=int, default=1729)
    serve.add_argument("--audit-count", type=int, default=10)
    serve.add_argument("--adaptive-count", type=int, default=20)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8050)
    serve.add_argument("--debug", action="store_true")
    session_choice = serve.add_mutually_exclusive_group()
    session_choice.add_argument("--session-id")
    session_choice.add_argument("--new-session", action="store_true")

    export = subparsers.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--include-unsure", action="store_true")

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-dir", type=Path)

    report = subparsers.add_parser("pilot-report")
    report.add_argument("--labels", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)

    freeze = subparsers.add_parser("freeze-pilot")
    freeze.add_argument("--report", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--target-recall", type=float, default=0.99)

    migrate = subparsers.add_parser("migrate-v1")
    migrate.add_argument("--v1-runtime", type=Path, required=True)
    migrate.add_argument("--v1-config", type=Path, required=True)

    final = subparsers.add_parser("finalize-runtime")
    final.add_argument("--pilot-runtime", type=Path, required=True)
    final.add_argument("--pilot-config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = load_registry(args.config)
    runtime = args.runtime.expanduser().resolve()
    event_store = EventStore(runtime / "events" / "shuttle.jsonl", registry)
    if args.command == "pilot-report":
        records = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines()]
        labels = [record for record in records if record.get("type") != "metadata"]
        reports = {}
        for source_id, source in registry.sources.items():
            source_labels = [record for record in labels if record.get("source_id") == source_id]
            reports[source_id] = evaluate_threshold_pilot(
                source.artifacts["candidates"], source_labels
            )
        combined_rows = []
        first_rows = next(iter(reports.values()))["threshold_results"] if reports else []
        for index, first in enumerate(first_rows):
            rows = [report["threshold_results"][index] for report in reports.values()]
            total = sum(int(row["observed_target_frames"]) for row in rows)
            hits = sum(int(row["observed_hits"]) for row in rows)
            hits_at_k = {
                key: sum(int(row["hits_at_k"][key]) for row in rows)
                for key in first["hits_at_k"]
            }
            def combined_counts(field: str) -> list[int]:
                return [
                    int(count)
                    for row in rows
                    for value, frequency in row[field].items()
                    for count in [value] * int(frequency)
                ]

            raw_counts = combined_counts("raw_candidate_count_histogram")
            grouped_counts = combined_counts("grouped_candidate_count_histogram")
            def summary(values: list[int]) -> dict[str, float | int]:
                values.sort()
                def percentile(q: float) -> float:
                    if not values:
                        return 0.0
                    position = (len(values) - 1) * q
                    lower, upper = int(position), math.ceil(position)
                    return values[lower] + (values[upper] - values[lower]) * (position - lower)
                return {"mean": sum(values) / len(values) if values else 0.0,
                        "p50": percentile(.5), "p90": percentile(.9),
                        "p95": percentile(.95), "p99": percentile(.99),
                        "maximum": max(values, default=0)}
            combined_rows.append({
                **first,
                "observed_target_frames": total,
                "observed_hits": hits,
                **{
                    field: sum(int(row[field]) for row in rows)
                    for field in (
                        "selected_label_frames",
                        "missing_proposal_frames",
                        "annotated_missing_proposal_frames",
                        "selected_lost_at_cutoff_frames",
                        "occluded_inferable_frames",
                        "no_in_frame_target_frames",
                        "unsure_frames",
                        "legacy_no_shuttle_frames",
                    )
                },
                "observed_proposal_recall": hits / total if total else None,
                "wilson_95": wilson_interval(hits, total),
                "hits_at_k": hits_at_k,
                "recall_at_k": {key: value / total if total else None for key, value in hits_at_k.items()},
                "raw_candidates_per_frame": summary(raw_counts),
                "grouped_candidates_per_frame": summary(grouped_counts),
                "source_results": {source_id: rows[position] for position, source_id in enumerate(reports)},
            })
        output = {
            "schema": "shuttle_threshold_pilot_report",
            "schema_version": 1,
            "sources": reports,
            "threshold_results": combined_rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(args.output)
        return 0
    if args.command == "freeze-pilot":
        report = json.loads(args.report.read_text(encoding="utf-8"))
        freeze = freeze_threshold_policy(report, args.target_recall)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for source_id, source in registry.sources.items():
            output = args.output_dir / f"{source_id}-shuttle-candidates-frozen.jsonl"
            filter_pilot_artifact(source.artifacts["candidates"], output, freeze)
            print(output)
        freeze_path = args.output_dir / "threshold-freeze.json"
        freeze_path.write_text(json.dumps(freeze, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(freeze_path)
        return 0
    if args.command == "migrate-v1":
        old_registry = load_registry(args.v1_config)
        old_artifacts = {
            source_id: source.artifacts["candidates"]
            for source_id, source in old_registry.sources.items()
        }
        result = migrate_v1_runtime(args.v1_runtime, runtime, registry, old_artifacts)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "finalize-runtime":
        pilot_registry = load_registry(args.pilot_config)
        pilot_artifacts = {
            source_id: source.artifacts["candidates"]
            for source_id, source in pilot_registry.sources.items()
        }
        result = materialize_final_runtime(args.pilot_runtime, runtime, registry, pilot_artifacts)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "build-queues":
        adaptive, audit = _queues(
            registry,
            runtime,
            audit_seed=args.audit_seed,
            audit_count=args.audit_count,
            adaptive_count=args.adaptive_count,
        )
        print(f"adaptive bursts: {len(adaptive.bursts)}")
        print(f"audit bursts: {len(audit.bursts)}")
        return 0
    if args.command == "export":
        event_store.export_current(args.output, task=SHUTTLE_TASK, include_unsure=args.include_unsure)
        print(args.output)
        return 0
    if args.command == "smoke":
        output_dir = (args.output_dir or runtime / "smoke").expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        for source_id, source in registry.sources.items():
            center = source.frame_count // 2
            view = build_playback_view(
                registry, task=SHUTTLE_TASK, source_id=source_id, center_frame=center
            )
            output = output_dir / f"{source_id}-frame-{center}.jpg"
            output.write_bytes(
                render_center_frame(
                    registry, task=SHUTTLE_TASK, source_id=source_id, frame=center
                )
            )
            print(
                f"{source_id}: {len(view.overlays_by_frame)} context frames, "
                f"{len(view.center_candidates)} center candidates -> {output}"
            )
        return 0
    adaptive, audit = _queues(
        registry,
        runtime,
        audit_seed=args.audit_seed,
        audit_count=args.audit_count,
        adaptive_count=args.adaptive_count,
    )
    selected_queue = adaptive if args.queue == "adaptive" else audit
    sessions = SessionManager(runtime / "sessions")
    replayed = event_store.replay()
    if args.session_id:
        session = sessions.load_compatible(args.session_id, args.annotator, selected_queue)
        status = "resumed explicit session"
    elif args.new_session:
        session = sessions.create(args.annotator, selected_queue)
        status = "created new session"
    else:
        session = sessions.select(args.annotator, selected_queue, replayed.active)
        if session is None:
            session = sessions.create(args.annotator, selected_queue)
            status = "created new session (no compatible session found)"
        else:
            status = "auto-resumed session"
    session = sessions.reconcile(session, selected_queue, replayed.active)
    resume_command = _resume_command(args, runtime, session.session_id)
    _print_session_status(status, session, sessions, selected_queue, resume_command)
    app = create_dash_app(registry, selected_queue, event_store, sessions, session)
    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    finally:
        durable = sessions.load(session.session_id)
        print("Annotation server stopped. Durable recovery state:")
        _print_session_status("saved session", durable, sessions, selected_queue, resume_command)
    return 0


def _resume_command(args: argparse.Namespace, runtime: Path, session_id: str) -> str:
    parts = [
        sys.executable,
        "-m",
        "src.annotation_platform",
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--runtime",
        str(runtime),
        "serve",
        "--annotator",
        args.annotator,
        "--queue",
        args.queue,
        "--audit-seed",
        str(args.audit_seed),
        "--audit-count",
        str(args.audit_count),
        "--adaptive-count",
        str(args.adaptive_count),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--session-id",
        session_id,
    ]
    if args.debug:
        parts.append("--debug")
    return shlex.join(parts)


def _print_session_status(
    status: str,
    session: SessionState,
    sessions: SessionManager,
    queue: AnnotationQueue,
    resume_command: str,
) -> None:
    # Kept as a small helper so startup and shutdown cannot drift apart.
    state = session
    position = sessions.position(state, queue)
    target = sessions.current(state, queue)
    current = "complete" if target is None else f"{target[0].source_id}:{target[1]}"
    print(f"Session: {state.session_id} ({status})")
    print(f"Queue position: {position + (0 if target is None else 1)}/{sessions.queue_length(queue)}")
    print(f"Current frame: {current}")
    print(f"Resume command: {resume_command}")


if __name__ == "__main__":
    raise SystemExit(main())
