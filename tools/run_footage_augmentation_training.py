"""Run paired baseline/augmented training for every built footage arm.

Each arm config is produced by ``tools/build_footage_augmentation.py``.  This
runner trains the paired harness for every requested arm and feature view,
reusing completed fold/arm metrics, and writes a combined summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument(
        "--feature-views", nargs="+", default=["full_motion"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--boundary-aux-weight", type=float, default=0.25)
    parser.add_argument("--temporal-smoothing-weight", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--variant-share", type=float, default=0.5)
    parser.add_argument("--guard-seconds", type=float, default=1.0)
    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--held-out-camera", action="append", default=[])
    parser.add_argument("--max-folds", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    summary_path = run_root / "training-summary.json"
    index: dict[str, Any] = (
        _read(summary_path) if summary_path.is_file() else {"arms": {}}
    )
    index.setdefault("arms", {})
    index["feature_views"] = sorted(
        set(index.get("feature_views", ())) | set(args.feature_views)
    )
    for arm in args.arms:
        config = run_root / arm / "experiment.json"
        if not config.is_file():
            raise FileNotFoundError(
                f"arm has not been built yet: {config}"
            )
        arm_index: dict[str, Any] = {}
        for view in args.feature_views:
            output = run_root / arm / "training" / view
            command = [
                sys.executable,
                "-m",
                "experiments.temporal_selector.footage_augmentation_experiment",
                "--config",
                str(config),
                "--output-dir",
                str(output),
                "--arm",
                arm,
                "--feature-view",
                view,
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--boundary-aux-weight",
                str(args.boundary_aux_weight),
                "--temporal-smoothing-weight",
                str(args.temporal_smoothing_weight),
                "--gradient-clip-norm",
                str(args.gradient_clip_norm),
                "--variant-share",
                str(args.variant_share),
                "--guard-seconds",
                str(args.guard_seconds),
                "--context-seconds",
                str(args.context_seconds),
                "--seeds",
                *map(str, args.seeds),
            ]
            if args.held_out_camera:
                for camera in args.held_out_camera:
                    command += ["--held-out-camera", camera]
            if args.max_folds is not None:
                command += ["--max-folds", str(args.max_folds)]
            print(f"=== {arm} / {view} ===", flush=True)
            subprocess.run(command, check=True)
            summary_path = output / "summary.json"
            if not summary_path.is_file():
                raise RuntimeError(f"training did not write a summary: {summary_path}")
            arm_index[view] = _read(summary_path)
        index["arms"].setdefault(arm, {}).update(arm_index)
    _write(run_root / "training-summary.json", index)
    _render_report(run_root, index)
    return 0


def _render_report(run_root: Path, index: Mapping[str, Any]) -> None:
    lines = ["# Footage augmentation training report", ""]
    for arm, views in index["arms"].items():
        lines.append(f"## {arm}")
        for view, summary in views.items():
            macro = summary.get("macro", {})
            arms = macro.get("arms", {})
            delta = macro.get("delta_augmented_minus_baseline", {})
            baseline = arms.get("baseline", {})
            augmented = arms.get("augmented", {})
            interval = baseline.get("interval_f1")
            interval_aug = augmented.get("interval_f1")
            lines.append(
                f"- {view}: baseline interval F1 {interval!s}; "
                f"augmented {interval_aug!s}; delta {delta.get('interval_f1')!s}"
            )
            lines.append(
                f"  - baseline frame F1 {baseline.get('frame_f1')!s}, "
                f"augmented {augmented.get('frame_f1')!s}"
            )
            lines.append(
                f"  - baseline boundary MAE {baseline.get('boundary_mae')!s}, "
                f"augmented {augmented.get('boundary_mae')!s}"
            )
        lines.append("")
    (run_root / "training-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
