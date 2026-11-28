"""Preregistered decoder factorial on the five-seed development folds.

Grid: max gap {0.75, 1.0, 1.5, 2.0} s x minimum duration {0.3, 0.5} s, with the
production threshold/continuation/edge settings held fixed. Selection is
leave-one-camera-out within each seed: a config is chosen on the other three
held-out cameras and evaluated on the fourth. The deployment config is the modal
LOCO choice. Primary model arm is the production recipe (triangle BCE).
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from itertools import groupby
from pathlib import Path

from src.temporal_selector.joint_inference import InPlayDecoderConfig
from src.temporal_selector.rally_evaluation import decode_prediction_rows, rally_metrics

SEEDS = (1729, 1730, 1731, 1732, 1733)
ARM = "baseline"
GAPS = (0.75, 1.0, 1.5, 2.0)
MINS = (0.3, 0.5)
PRODUCTION = (0.5, 0.3)
FPS = 30.0
FIXED = dict(threshold=0.40, continuation_threshold=0.35, preserve_edge_runs=True)


def config(gap, minimum):
    return InPlayDecoderConfig(max_gap_seconds=gap, minimum_duration_seconds=minimum, **FIXED)


CONFIGS = {(gap, minimum): config(gap, minimum) for gap in GAPS for minimum in MINS}
CONFIGS[PRODUCTION] = config(*PRODUCTION)
GRID = tuple((gap, minimum) for gap in GAPS for minimum in MINS)


def false_starts(rows):
    ordered = sorted(rows, key=lambda row: (row["source_id"], row["frame"]))
    key = lambda row: (row["source_id"], row["decoded_inplay"])
    count = 0
    for (_, active), group in groupby(ordered, key):
        if not active:
            continue
        in_break = True
        previous = None
        for row in group:
            if previous is not None and row["frame"] != previous + 1:
                count += in_break
                in_break = True
            in_break &= row["inplay_target"] == 0
            previous = row["frame"]
        count += in_break
    minutes = sum(row["inplay_target"] == 0 for row in ordered) / FPS / 60
    return count, minutes


def evaluate(rows, decoder):
    source_ids = {row["source_id"] for row in rows}
    decoded = decode_prediction_rows(rows, fps_by_source={s: FPS for s in source_ids}, config=decoder)
    metrics = rally_metrics(decoded)
    intervals = metrics["intervals"]
    extras, minutes = false_starts(decoded)
    return {
        "interval_f1": intervals["f1"],
        "interval_precision": intervals["precision"],
        "interval_recall": intervals["recall"],
        "false_starts": extras,
        "false_starts_per_break_minute": extras / minutes,
        "false_splits": intervals["false_split_count"],
        "false_merges": intervals["false_merge_count"],
        "matched_rallies": intervals["matched_count"],
        "labeled_rallies": intervals["label_count"],
        "matched": {
            (match["source_id"], match["label_rally_id"]): match
            for match in metrics["interval_matches"] if match["match_status"] == "matched"
        },
    }


def main():
    results = Path(sys.argv[1])
    output = Path(sys.argv[2])
    cells = {}
    for seed in SEEDS:
        seed_dir = results / f"seed-{seed}" / ARM
        for fold_dir in sorted(seed_dir.glob("fold-*")):
            camera = fold_dir.name.split("-", 2)[-1]
            rows = [json.loads(line) for line in (fold_dir / "test-predictions.jsonl").open()]
            rows.sort(key=lambda row: (row["source_id"], row["frame"]))
            by_config = {name: evaluate(rows, decoder) for name, decoder in CONFIGS.items()}
            shared = set.intersection(*(set(value["matched"]) for value in by_config.values()))
            for value in by_config.values():
                matched = value.pop("matched")
                value["common_end_mae_frames"] = statistics.fmean(
                    abs(matched[key]["end_error"]) for key in shared
                ) if shared else None
                value["common_start_mae_frames"] = statistics.fmean(
                    abs(matched[key]["start_error"]) for key in shared
                ) if shared else None
                value["common_matched_rallies"] = len(shared)
            cells[(seed, camera)] = by_config

    # Leave-one-camera-out selection within each seed.
    choices = {}
    selected = {}
    for seed in SEEDS:
        cameras = [camera for (s, camera) in cells if s == seed]
        for held in cameras:
            others = [camera for camera in cameras if camera != held]
            candidates = {}
            for name in GRID:
                f1 = statistics.fmean(cells[(seed, camera)][name]["interval_f1"] for camera in others)
                starts = statistics.fmean(cells[(seed, camera)][name]["false_starts_per_break_minute"] for camera in others)
                end = statistics.fmean(cells[(seed, camera)][name]["common_end_mae_frames"] for camera in others)
                candidates[name] = (f1, starts, end)
            best = max(
                GRID,
                key=lambda name: (
                    candidates[name][0], -candidates[name][1], -candidates[name][2],
                    -name[0], -name[1],
                ),
            )
            choices[(seed, held)] = best
            selected[(seed, held)] = cells[(seed, held)][best]

    def macro(values, field):
        return statistics.fmean(value[field] for value in values)

    def paired(field):
        diffs = [selected[key][field] - cells[key][PRODUCTION][field] for key in cells]
        mean = statistics.fmean(diffs)
        error = statistics.stdev(diffs) / math.sqrt(len(diffs))
        t = 2.093  # two-sided 95%, df=19
        return {"mean": mean, "se": error, "ci95": [mean - t * error, mean + t * error]}

    outcomes = ("interval_f1", "interval_recall", "false_starts_per_break_minute", "common_end_mae_frames")
    per_config = {
        f"{gap:g}/{minimum:g}": {
            "max_gap_seconds": gap, "minimum_duration_seconds": minimum,
            **{outcome: macro([cells[key][(gap, minimum)] for key in cells], outcome) for outcome in outcomes},
        }
        for gap, minimum in GRID
    }
    frequency = {}
    for choice in choices.values():
        key = f"{choice[0]:g}/{choice[1]:g}"
        frequency[key] = frequency.get(key, 0) + 1
    ranked = sorted(frequency, key=lambda key: (-frequency[key], -per_config[key]["interval_f1"]))
    deployment = ranked[0]
    gap, minimum = (float(part) for part in deployment.split("/"))

    result = {
        "schema": "decoder_factorial_development",
        "schema_version": 1,
        "seeds": SEEDS,
        "arm": ARM,
        "grid": [{"max_gap_seconds": g, "minimum_duration_seconds": m} for g, m in GRID],
        "production": {"max_gap_seconds": PRODUCTION[0], "minimum_duration_seconds": PRODUCTION[1]},
        "fixed": FIXED,
        "selection": "leave-one-camera-out within each seed, maximize mean interval F1 on the other three cameras",
        "deployment_rule": "modal LOCO choice, ties by development interval F1",
        "deployment": {"max_gap_seconds": gap, "minimum_duration_seconds": minimum},
        "choices": {f"{seed}/{camera}": f"{choice[0]:g}/{choice[1]:g}" for (seed, camera), choice in choices.items()},
        "selection_frequency": frequency,
        "production_macro": {outcome: macro([cells[key][PRODUCTION] for key in cells], outcome) for outcome in outcomes},
        "selected_macro": {outcome: macro([selected[key] for key in cells], outcome) for outcome in outcomes},
        "per_config_macro": per_config,
        "paired_selected_vs_production": {outcome: paired(outcome) for outcome in outcomes},
        "cells": {
            f"{seed}/{camera}": {
                "selected": f"{choices[(seed, camera)][0]:g}/{choices[(seed, camera)][1]:g}",
                "production": cells[(seed, camera)][PRODUCTION],
                "selected_metrics": selected[(seed, camera)],
            }
            for seed, camera in cells
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1) + "\n")

    print("config gap/min   F1     recall fs/min  endMAE")
    for name, value in sorted(per_config.items(), key=lambda item: -item[1]["interval_f1"]):
        print(f"{name:14} {value['interval_f1']:.4f} {value['interval_recall']:.3f} "
              f"{value['false_starts_per_break_minute']:.2f} {value['common_end_mae_frames']:.1f}")
    print("\nproduction macro", {k: round(v, 4) for k, v in result["production_macro"].items()})
    print("LOCO selected   ", {k: round(v, 4) for k, v in result["selected_macro"].items()})
    print("choices", frequency, "deployment", deployment)
    for outcome, value in result["paired_selected_vs_production"].items():
        print(f"delta {outcome:32} {value['mean']:+.4f} CI [{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
