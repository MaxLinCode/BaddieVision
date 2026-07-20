"""Low-friction source registration, annotation, and experiment setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import SourceCatalog


DEFAULT_CATALOG = Path("config/sources.local.json")
DEFAULT_RUNTIME = Path(".annotation-final")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser("source")
    source_commands = source.add_subparsers(dest="source_command", required=True)
    add = source_commands.add_parser("add")
    add.add_argument("--video", type=Path, required=True)
    add.add_argument("--source-id")
    add.add_argument("--output-dir", type=Path)
    add.add_argument("--calibration", type=Path)
    status = source_commands.add_parser("status", aliases=["doctor"])
    status.add_argument("--source", action="append", dest="sources")
    status.add_argument("--json", action="store_true")

    annotate = commands.add_parser("annotate")
    annotate.add_argument("--annotator", required=True)
    annotate.add_argument("--source", action="append", dest="sources")
    annotate.add_argument("--runtime", type=Path)
    annotate.add_argument("--queue", choices=("adaptive", "audit", "rally-audit", "refill"), default="adaptive")
    annotate.add_argument("--stage", choices=("pilot", "frozen"), default="frozen")
    annotate.add_argument("--host", default="127.0.0.1")
    annotate.add_argument("--port", type=int, default=8050)
    annotate.add_argument("--new-session", action="store_true")

    experiment = commands.add_parser("experiment")
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)
    prepare = experiment_commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, default=Path("config/selector-experiment.generated.json"))
    prepare.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    prepare.add_argument("--source", action="append", dest="sources")
    prepare.add_argument("--seed", type=int, default=1729)
    return parser


def _load_or_create(path: Path) -> SourceCatalog:
    path = path.expanduser().resolve()
    return SourceCatalog.read(path) if path.exists() else SourceCatalog(path)


def _status_mapping(catalog: SourceCatalog, source_ids: list[str] | None) -> list[dict[str, object]]:
    rows = []
    for source in catalog.select(source_ids):
        status = catalog.status(source)
        rows.append(
            {
                "source_id": status.source_id,
                "enabled": status.enabled,
                "video": status.video,
                "candidates": status.candidates,
                "frozen_candidates": status.frozen_candidates,
                "assignments": status.assignments,
                "pose_cache": status.pose_cache,
                "calibration": status.calibration,
                "annotation_ready": status.annotation_ready,
                "experiment_ready": status.experiment_ready,
                "problems": list(status.problems),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    catalog = _load_or_create(args.catalog)
    if args.command == "source" and args.source_command == "add":
        source = catalog.add(
            args.video,
            source_id=args.source_id,
            output_dir=args.output_dir,
            calibration_path=args.calibration,
        )
        print(f"registered {source.source_id}")
        print(f"catalog: {catalog.path}")
        return 0
    if args.command == "source":
        rows = _status_mapping(catalog, args.sources)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                state = "experiment-ready" if row["experiment_ready"] else "annotation-ready" if row["annotation_ready"] else "blocked"
                print(f"{row['source_id']}: {state}")
                for field in ("video", "candidates", "frozen_candidates", "assignments", "pose_cache", "calibration"):
                    print(f"  {field.replace('_', ' '):18} {'ready' if row[field] else 'missing'}")
                for problem in row["problems"]:
                    print(f"  problem: {problem}")
        return 0
    if args.command == "annotate":
        # Keep source/status/experiment commands usable without OpenCV and Dash.
        from src.annotation_platform.__main__ import main as annotation_main

        selected = catalog.select(args.sources)
        if not selected:
            raise ValueError("catalog has no enabled sources")
        if args.runtime is not None:
            runtime = args.runtime
        elif len(selected) == 1:
            runtime = Path(f".annotation-{selected[0].source_id}")
        else:
            runtime = DEFAULT_RUNTIME
        generated = runtime / "generated" / "annotation-sources.json"
        catalog.write_annotation_config(generated, source_ids=[s.source_id for s in selected], stage=args.stage)
        forwarded = [
            "--config", str(generated), "--runtime", str(runtime), "serve",
            "--annotator", args.annotator, "--queue", args.queue,
            "--host", args.host, "--port", str(args.port),
        ]
        if args.new_session:
            forwarded.append("--new-session")
        return annotation_main(forwarded)
    output, manifest = catalog.write_experiment(
        args.output,
        runtime=args.runtime,
        seed=args.seed,
        source_ids=args.sources,
    )
    print(output)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
