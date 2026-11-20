"""Command-line entry point for the rally error explorer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .app import create_app
from .data import load_explorer_data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--sources", type=Path, help="JSON with 'sources' or 'dataset.sources'")
    parser.add_argument("--artifact-root", type=Path, help="root containing one directory per source")
    parser.add_argument("--rallies", type=Path, help="strict rally interval CSV")
    parser.add_argument("--manifest", type=Path, help="rally interval manifest JSON")
    parser.add_argument("--shuttle-artifacts", type=Path, help="JSON mapping source_id -> shuttle candidate JSONL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    data = load_explorer_data(
        args.predictions,
        compare_path=args.compare,
        sources_path=args.sources,
        artifact_root=args.artifact_root,
        rallies_path=args.rallies,
        manifest_path=args.manifest,
        shuttle_artifacts_path=args.shuttle_artifacts,
    )
    for warning in data.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    app = create_app(data)
    print(f"Rally error explorer: http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
