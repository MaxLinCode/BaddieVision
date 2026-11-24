"""Summarize footage-augmentation training arms against the current best model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _delta(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-label", default="current best model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    summary = _read(run_root / "training-summary.json")
    reference = _read(args.reference) if args.reference else None
    reference_macro = (reference or {}).get("macro", {})

    lines = ["# Footage augmentation comparison", ""]
    if reference:
        lines += [
            f"Reference: **{args.reference_label}** "
            f"(`{args.reference}`)",
            "",
        ]
        lines.append(
            "| Reference metric | Value |"
            "\n|---|---:|"
            f"\n| Macro interval F1 | {_fmt(reference_macro.get('interval_f1'))} |"
            f"\n| Macro frame F1 | {_fmt(reference_macro.get('f1'))} |"
            f"\n| Macro ROC AUC | {_fmt(reference_macro.get('roc_auc'))} |"
            f"\n| Macro PR AUC | {_fmt(reference_macro.get('pr_auc'))} |"
            f"\n| Macro raw BCE | {_fmt(reference_macro.get('raw_bce'))} |"
            f"\n| Macro boundary MAE | {_fmt(reference_macro.get('boundary_mae'))} |"
        )
        lines.append("")

    results: dict[str, Any] = {}
    for arm, views in summary.get("arms", {}).items():
        for view, view_summary in views.items():
            macro = view_summary.get("macro", {})
            arms = macro.get("arms", {})
            delta = macro.get("delta_augmented_minus_baseline", {})
            by_camera = view_summary.get("by_camera", {})
            baseline = arms.get("baseline", {})
            augmented = arms.get("augmented", {})
            improved = [
                camera
                for camera, values in by_camera.items()
                if (values.get("delta_augmented_minus_baseline", {}).get("interval_f1") or 0) > 0
            ]
            results[f"{arm}/{view}"] = {
                "baseline": baseline,
                "augmented": augmented,
                "delta": delta,
                "cameras_improved_interval_f1": sorted(improved),
                "camera_count": len(by_camera),
            }
            lines += [
                f"## {arm} / {view}",
                "",
                "| Metric | Baseline | Augmented | Delta |",
                "|---|---:|---:|---:|",
            ]
            for name in (
                "interval_f1",
                "frame_f1",
                "roc_auc",
                "pr_auc",
                "raw_bce",
                "boundary_mae",
                "signed_start_error",
                "signed_end_error",
                "false_splits",
                "false_merges",
            ):
                lines.append(
                    f"| {name} | {_fmt(baseline.get(name))} | "
                    f"{_fmt(augmented.get(name))} | {_delta(delta.get(name))} |"
                )
            lines += [
                "",
                f"Cameras with higher augmented interval F1: "
                f"{', '.join(sorted(improved)) or 'none'} "
                f"({len(improved)}/{len(by_camera)})",
                "",
                "| Camera | Baseline interval F1 | Augmented interval F1 | Delta |",
                "|---|---:|---:|---:|",
            ]
            for camera in sorted(by_camera):
                values = by_camera[camera]
                camera_arms = values.get("arms", {})
                camera_delta = values.get("delta_augmented_minus_baseline", {})
                lines.append(
                    f"| {camera} | "
                    f"{_fmt(camera_arms.get('baseline', {}).get('interval_f1'))} | "
                    f"{_fmt(camera_arms.get('augmented', {}).get('interval_f1'))} | "
                    f"{_delta(camera_delta.get('interval_f1'))} |"
                )
            lines.append("")
            if reference:
                augmented_interval = augmented.get("interval_f1")
                reference_interval = reference_macro.get("interval_f1")
                if augmented_interval is not None and reference_interval is not None:
                    lines.append(
                        f"Augmented macro interval F1 minus reference: "
                        f"{_delta(augmented_interval - reference_interval)}"
                    )
                    lines.append("")

    lines += ["## Arm ranking by augmented macro interval F1", ""]
    ranked = sorted(
        (
            (key, value["augmented"].get("interval_f1"))
            for key, value in results.items()
            if value["augmented"].get("interval_f1") is not None
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    lines.append("| Rank | Arm / view | Augmented interval F1 | Delta vs its baseline |")
    lines.append("|---:|---|---:|---:|")
    for index, (key, value) in enumerate(ranked, start=1):
        delta = results[key]["delta"].get("interval_f1")
        lines.append(f"| {index} | {key} | {_fmt(value)} | {_delta(delta)} |")
    lines.append("")

    control_suffix = "/h264-control"
    controls = {
        key: value
        for key, value in results.items()
        if key.endswith(control_suffix)
    }
    if controls:
        lines += [
            "## Relative to the high-quality H.264 control",
            "",
            "Positive values favor the arm over the control after each arm is "
            "compared with its own original baseline.",
            "",
            "| Arm / view | Augmented interval F1 | Control augmented | Delta |",
            "|---|---:|---:|---:|",
        ]
        for key, value in sorted(results.items()):
            if key in controls:
                continue
            view = key.rsplit("/", 1)[-1]
            control_key = f"h264-control/{view}"
            control = results.get(control_key)
            if control is None:
                continue
            arm_value = value["augmented"].get("interval_f1")
            control_value = control["augmented"].get("interval_f1")
            if arm_value is None or control_value is None:
                continue
            lines.append(
                f"| {key} | {_fmt(arm_value)} | {_fmt(control_value)} | "
                f"{_delta(arm_value - control_value)} |"
            )
        lines.append("")

    output = args.output or (run_root / "comparison-report.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_root / "comparison.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
