"""Copy footage-augmentation results into a job workspace for retrieval."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--include-checkpoints", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.run_root.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "training-summary.json",
        "training-report.md",
        "comparison-report.md",
        "comparison.json",
    ):
        _copy_file(source / name, destination / name)
    verification = source / "verification"
    if verification.is_dir():
        for path in verification.rglob("*"):
            if path.is_file():
                _copy_file(
                    path, destination / "verification" / path.relative_to(verification)
                )
    for arm_dir in sorted(source.iterdir()):
        if not arm_dir.is_dir() or not (arm_dir / "experiment.json").is_file():
            continue
        for name in ("experiment.json", "build.json"):
            _copy_file(arm_dir / name, destination / arm_dir.name / name)
        annotations = arm_dir / "annotations"
        if annotations.is_dir():
            for path in annotations.iterdir():
                if path.is_file():
                    _copy_file(
                        path,
                        destination / arm_dir.name / "annotations" / path.name,
                    )
        training = arm_dir / "training"
        if training.is_dir():
            for path in training.rglob("*"):
                if not path.is_file():
                    continue
                relative = path.relative_to(training)
                if path.name in {
                    "metrics.json",
                    "summary.json",
                    "split.json",
                    "design.json",
                    "test-predictions.jsonl",
                } or (args.include_checkpoints and path.suffix == ".pt"):
                    _copy_file(
                        path,
                        destination / arm_dir.name / "training" / relative,
                    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
