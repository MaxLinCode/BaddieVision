"""Run the local shuttle annotation app and maintenance commands."""

from __future__ import annotations

import argparse
import json
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
from .sessions import SessionManager
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
    serve.add_argument("--session-id")

    export = subparsers.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--include-skips", action="store_true")

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = load_registry(args.config)
    runtime = args.runtime.expanduser().resolve()
    event_store = EventStore(runtime / "events" / "shuttle.jsonl", registry)
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
        event_store.export_current(args.output, task=SHUTTLE_TASK, include_skips=args.include_skips)
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
    if args.session_id:
        session = sessions.load(args.session_id)
    else:
        session = sessions.create(args.annotator, selected_queue)
    app = create_dash_app(registry, selected_queue, event_store, sessions, session)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
