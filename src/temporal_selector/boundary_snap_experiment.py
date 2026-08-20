"""Prediction-only cross-camera evaluation of operational rally boundary heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .rally_evaluation import rally_metrics
from .rally_pseudo_label import _split_merge_counts


RADII_SECONDS = (0.25, 0.5, 0.75)
CONFIDENCE_FLOORS = (0.25, 0.5, 0.75)
MAXIMUM_SHIFTS_SECONDS = (0.25, 0.5, 0.75)
INTERVAL_F1_TOLERANCE = 0.01


@dataclass(frozen=True)
class OperationalSnapConfig:
    start_radius_seconds: float
    end_radius_seconds: float
    start_confidence_floor: float
    end_confidence_floor: float
    maximum_shift_seconds: float


def _probability(row: Mapping[str, Any], boundary: str) -> float:
    key = f"rally_{boundary}_probability"
    if key in row:
        return float(row[key])
    logit = float(row[f"rally_{boundary}_logit"])
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _runs(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, row in enumerate(rows):
        state = bool(row["decoded_inplay"])
        if state and start is None:
            start = index
        elif not state and start is not None:
            result.append((start, index - 1))
            start = None
    if start is not None:
        result.append((start, len(rows) - 1))
    return result


def _peak(probabilities: Sequence[float], edge: int, radius: int,
          floor: float, maximum_shift: int) -> int:
    reach = min(radius, maximum_shift)
    left, right = max(0, edge - reach), min(len(probabilities) - 1, edge + reach)
    candidates = [i for i in range(left, right + 1) if probabilities[i] >= floor]
    # Stable tie-break: confidence, distance from current edge, then earlier frame.
    return min(candidates, key=lambda i: (-probabilities[i], abs(i - edge), i)) if candidates else edge


def snap_prediction_rows(rows: Iterable[Mapping[str, Any]], *,
                         fps_by_source: Mapping[str, float],
                         config: OperationalSnapConfig) -> list[dict[str, Any]]:
    """Adjust existing decoded edges without changing the interval count."""
    by_source: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        by_source.setdefault(str(row["source_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    serialized_config = asdict(config)
    for source, source_rows in sorted(by_source.items()):
        source_rows.sort(key=lambda row: int(row["frame"]))
        frames = [int(row["frame"]) for row in source_rows]
        if any(b != a + 1 for a, b in zip(frames, frames[1:])):
            raise ValueError(f"prediction frames are incomplete for {source!r}")
        fps = float(fps_by_source[source])
        starts = [_probability(row, "start") for row in source_rows]
        ends = [_probability(row, "end") for row in source_rows]
        start_radius = round(config.start_radius_seconds * fps)
        end_radius = round(config.end_radius_seconds * fps)
        maximum_shift = round(config.maximum_shift_seconds * fps)
        original = _runs(source_rows)
        proposed = [
            (_peak(starts, left, start_radius, config.start_confidence_floor, maximum_shift),
             _peak(ends, right, end_radius, config.end_confidence_floor, maximum_shift))
            for left, right in original
        ]
        invalid: set[int] = set()
        for i, ((old_left, old_right), (left, right)) in enumerate(zip(original, proposed)):
            minimum_seconds = float(source_rows[old_left]["decoder_configuration"]["minimum_duration_seconds"])
            minimum_frames = max(1, round(minimum_seconds * fps))
            if right < left or right - left + 1 < minimum_frames:
                invalid.add(i)
        for i in range(len(proposed) - 1):
            if proposed[i][1] >= proposed[i + 1][0]:
                invalid.update((i, i + 1))
        final = [old if i in invalid else new for i, (old, new) in enumerate(zip(original, proposed))]
        states = [False] * len(source_rows)
        for left, right in final:
            states[left:right + 1] = [True] * (right - left + 1)
        output.extend({**row, "decoded_inplay": state,
                       "boundary_snap_configuration": serialized_config}
                      for row, state in zip(source_rows, states))
    return output


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Different held-out folds legitimately have independently frozen decoder
    # thresholds. This experiment evaluates only decoded states and boundaries;
    # remove the raw-threshold metadata when several folds form a tuning pool.
    metric_rows = [{k: v for k, v in row.items() if k != "decoder_configuration"}
                   for row in rows]
    metrics = rally_metrics(metric_rows)
    intervals = metrics["intervals"]
    splits, merges = _split_merge_counts(rows)
    return {
        "signed_start_error": intervals["mean_start_boundary_error"],
        "signed_end_error": intervals["mean_end_boundary_error"],
        "start_mae": _boundary_mae(metrics, "start_error"),
        "end_mae": _boundary_mae(metrics, "end_error"),
        "boundary_mae": intervals["mean_absolute_boundary_error"],
        "interval_f1": intervals["f1"],
        "frame_f1": metrics["decoded_frame"]["f1"],
        "false_splits": splits,
        "false_merges": merges,
        "matched_intervals": intervals["matched_count"],
    }


def _boundary_mae(metrics: Mapping[str, Any], field: str) -> float | None:
    values = [abs(float(row[field])) for row in metrics.get("interval_matches", ())
              if row.get("match_status") == "matched" and row.get(field) not in (None, "")]
    return sum(values) / len(values) if values else None


def _configs() -> Iterable[OperationalSnapConfig]:
    for sr in RADII_SECONDS:
        for er in RADII_SECONDS:
            for st in CONFIDENCE_FLOORS:
                for et in CONFIDENCE_FLOORS:
                    for shift in MAXIMUM_SHIFTS_SECONDS:
                        yield OperationalSnapConfig(sr, er, st, et, shift)


def select_configuration(rows: Sequence[Mapping[str, Any]], *,
                         fps_by_source: Mapping[str, float]) -> tuple[OperationalSnapConfig, dict[str, Any]]:
    baseline = _summary(rows)
    candidates = []
    for config in _configs():
        summary = _summary(snap_prediction_rows(rows, fps_by_source=fps_by_source, config=config))
        safe = (summary["interval_f1"] >= baseline["interval_f1"] - INTERVAL_F1_TOLERANCE
                and summary["false_splits"] <= baseline["false_splits"]
                and summary["false_merges"] <= baseline["false_merges"])
        boundary = summary["boundary_mae"]
        score = (int(safe), -float(boundary if boundary is not None else math.inf),
                 float(summary["interval_f1"]), -config.maximum_shift_seconds,
                 -config.start_radius_seconds, -config.end_radius_seconds,
                 -config.start_confidence_floor, -config.end_confidence_floor)
        candidates.append((score, config, summary, safe))
    _, config, summary, safe = max(candidates, key=lambda item: item[0])
    return config, {"baseline": baseline, "selected": summary, "guardrails_satisfied": safe,
                    "candidate_count": len(candidates)}


def run_experiment(input_dir: Path, output: Path, *, source_root: Path) -> dict[str, Any]:
    root_metrics = json.loads((input_dir / "metrics.json").read_text())
    camera_sources = {str(k): tuple(map(str, v)) for k, v in root_metrics["camera_sources"].items()}
    if len(camera_sources) < 3:
        raise ValueError("boundary snapping needs at least three independent camera groups")
    rows_by_camera: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for fold_dir in sorted(input_dir.glob("fold-*")):
        path = fold_dir / "test-predictions.jsonl"
        if not path.is_file():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        sources = {str(row["source_id"]) for row in rows}
        matches = [camera for camera, expected in camera_sources.items() if sources == set(expected)]
        if len(matches) != 1:
            raise ValueError(f"cannot resolve held-out camera for {path}")
        rows_by_camera[matches[0]] = rows
        hashes[matches[0]] = hashlib.sha256(path.read_bytes()).hexdigest()
    if set(rows_by_camera) != set(camera_sources):
        raise ValueError("out-of-fold predictions do not cover every camera")
    fps_by_source = {}
    for source in {s for values in camera_sources.values() for s in values}:
        identity = json.loads((source_root / source / "source.json").read_text())
        fps_by_source[source] = float(identity["fps_numerator"]) / float(identity["fps_denominator"])
    reports = {}
    for heldout in camera_sources:
        tuning = [row for camera, rows in rows_by_camera.items() if camera != heldout for row in rows]
        config, selection = select_configuration(tuning, fps_by_source=fps_by_source)
        baseline = _summary(rows_by_camera[heldout])
        snapped_rows = snap_prediction_rows(rows_by_camera[heldout], fps_by_source=fps_by_source, config=config)
        snapped = _summary(snapped_rows)
        reports[heldout] = {
            "tuning_cameras": [camera for camera in camera_sources if camera != heldout],
            "heldout_camera": heldout, "selected_configuration": asdict(config),
            "selection": selection, "baseline": baseline, "snapped": snapped,
            "delta": {key: (None if baseline[key] is None or snapped[key] is None
                            else snapped[key] - baseline[key]) for key in baseline},
        }
    numeric = ("signed_start_error", "signed_end_error", "start_mae", "end_mae",
               "boundary_mae", "interval_f1", "frame_f1", "false_splits", "false_merges")
    macro = {}
    for variant in ("baseline", "snapped", "delta"):
        macro[variant] = {key: sum(float(reports[c][variant][key]) for c in reports
                                  if reports[c][variant][key] is not None) /
                               sum(reports[c][variant][key] is not None for c in reports)
                          for key in numeric}
    improved = sum(reports[c]["delta"]["boundary_mae"] < 0 for c in reports)
    macro_delta = macro["delta"]
    success = (improved > len(reports) / 2 and macro_delta["boundary_mae"] <= -3.0
               and macro_delta["interval_f1"] >= -INTERVAL_F1_TOLERANCE
               and macro_delta["false_splits"] <= 0 and macro_delta["false_merges"] <= 0)
    result = {
        "schema": "operational_boundary_snap_cross_camera", "schema_version": 1,
        "input_directory": str(input_dir.resolve()), "prediction_sha256": hashes,
        "camera_sources": {k: list(v) for k, v in camera_sources.items()},
        "selection_partition": "all_other_out_of_fold_camera_groups",
        "evaluation_partition": "heldout_out_of_fold_camera_group_once",
        "search_space": {"start_radius_seconds": RADII_SECONDS, "end_radius_seconds": RADII_SECONDS,
                         "start_confidence_floor": CONFIDENCE_FLOORS,
                         "end_confidence_floor": CONFIDENCE_FLOORS,
                         "maximum_shift_seconds": MAXIMUM_SHIFTS_SECONDS},
        "per_camera": reports, "macro": macro,
        "success_criterion": {"boundary_improved_camera_count": improved,
                              "required_majority": math.floor(len(reports) / 2) + 1,
                              "minimum_macro_boundary_improvement_frames": 3.0,
                              "maximum_interval_f1_regression": INTERVAL_F1_TOLERANCE,
                              "passed": success},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    result = run_experiment(args.input_dir, args.output, source_root=args.source_root)
    print(json.dumps({"output": str(args.output), "macro": result["macro"],
                      "success_criterion": result["success_criterion"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
