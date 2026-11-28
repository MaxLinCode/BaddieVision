"""Development-only decoder sweep over saved five-seed baseline predictions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import accumulate, groupby, product
import json
from pathlib import Path
import statistics

from tools.run_decoder_factorial import FPS, SEEDS, config, evaluate
from src.temporal_selector.joint_inference import _runs, decode_inplay_probabilities
from InPlay.heuristic.evaluate import Interval, evaluate as evaluate_intervals

GAPS = (0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25,
        3.5, 3.75, 4, 4.5, 5, 6, 8, 10)
MINIMUMS = (.1, .2, .3, .4, .5, .6, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4)
PRODUCTION = (.5, .3)
INCUMBENT = (2, .5)


def score_cell(task):
    path, gaps, minimums = task
    with path.open() as stream:
        rows = sorted(map(json.loads, stream), key=lambda row: (row["source_id"], row["frame"]))
    if any(row.get("partial_rally") or row["inplay_target"] not in (0, 1) for row in rows):
        raise ValueError("This sweep expects fully labeled rows with no partial-rally flags.")

    chunks = []
    truth = []
    seconds_of_break = sum(row["inplay_target"] == 0 for row in rows) / FPS
    truth_frames = sum(row["inplay_target"] for row in rows)
    for source, source_rows in groupby(rows, key=lambda row: row["source_id"]):
        interval_count = 0
        for _, group in groupby(enumerate(source_rows), key=lambda item: item[1]["frame"] - item[0]):
            sequence = [row for _, row in group]
            offset = sequence[0]["frame"]
            labels = [row["inplay_target"] for row in sequence]
            for start, end in _runs(labels):
                interval_count += 1
                truth.append(Interval(source, f"truth_state-{interval_count:04d}", offset + start, offset + end))
            chunks.append((source, offset, [r["inplay_probability"] for r in sequence],
                           list(accumulate(labels, initial=0))))

    measured = {}
    for gap in gaps:
        runs = []
        for source, offset, probabilities, prefix in chunks:
            bridged = decode_inplay_probabilities(probabilities, fps=FPS, config=config(gap, 0))
            for start, end in _runs(bridged):
                length = end - start + 1
                overlaps = prefix[end + 1] - prefix[start]
                preserve = start == 0 or end == len(bridged) - 1
                runs.append((source, offset + start, offset + end, length, overlaps, preserve))
        for minimum in minimums:
            minimum_frames = max(1, round(minimum * FPS))
            predictions = []
            counts = {}
            true_positive_frames = predicted_frames = false_starts = 0
            for source, start, end, length, overlaps, preserve in runs:
                if length < minimum_frames and not preserve:
                    continue
                counts[source] = counts.get(source, 0) + 1
                predictions.append(Interval(source, f"prediction_state-{counts[source]:04d}", start, end))
                true_positive_frames += overlaps
                predicted_frames += length
                false_starts += overlaps == 0
            intervals, details = evaluate_intervals(predictions, truth, threshold=.5)
            matches = {
                (m["source_id"], m["label_rally_id"]): m["end_error"]
                for m in details if m["match_status"] == "matched"
            }
            measured[gap, minimum] = {
                "interval_f1": intervals["f1"],
                "interval_precision": intervals["precision"],
                "interval_recall": intervals["recall"],
                "false_starts_per_break_minute": false_starts / (seconds_of_break / 60),
                "false_splits": intervals["false_split_count"],
                "false_merges": intervals["false_merge_count"],
                "missed_rallies": intervals["label_count"] - intervals["matched_count"],
                "frame_f1": 2 * true_positive_frames / (truth_frames + predicted_frames),
                "end_errors": matches,
            }

    for baseline in (PRODUCTION, INCUMBENT):
        reference = evaluate(rows, config(*baseline))
        actual = measured[baseline]
        for field in ("interval_f1", "interval_precision", "interval_recall",
                      "false_starts_per_break_minute", "false_merges", "false_splits"):
            if abs(reference[field] - actual[field]) > 1e-12:
                raise ValueError(f"Reference mismatch for {path.parent.name} {baseline} {field}")
        if {key: row["end_error"] for key, row in reference["matched"].items()} != actual["end_errors"]:
            raise ValueError(f"Matched end errors differ from reference for {path.parent.name} {baseline}")

    incumbent = measured[INCUMBENT]["end_errors"]
    for result in measured.values():
        errors = result.pop("end_errors")
        shared = incumbent.keys() & errors.keys()
        result["matched_vs_incumbent"] = len(shared)
        result["delta_signed_end_frames"] = statistics.fmean(
            errors[key] - incumbent[key] for key in shared
        ) if shared else None
        result["delta_end_mae_frames"] = statistics.fmean(
            abs(errors[key]) - abs(incumbent[key]) for key in shared
        ) if shared else None
    return measured


def mean(values):
    valid = [value for value in values if value is not None]
    return statistics.fmean(valid) if valid else None


def macro(results):
    return {field: mean([row[field] for row in results]) for field in results[0]}


def name(pair):
    return f"{pair[0]:g}/{pair[1]:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gaps", nargs="+", type=float, default=GAPS)
    parser.add_argument("--minimums", nargs="+", type=float, default=MINIMUMS)
    args = parser.parse_args()
    gaps = tuple(sorted(set(args.gaps) | {PRODUCTION[0], INCUMBENT[0]}))
    minimums = tuple(sorted(set(args.minimums) | {PRODUCTION[1], INCUMBENT[1]}))
    grid = tuple(product(gaps, minimums))
    files = {}
    for seed in SEEDS:
        paths = sorted((args.input / f"seed-{seed}" / "baseline").glob("fold-*/test-predictions.jsonl"))
        if len(paths) != 4:
            raise ValueError(f"Expected four complete development folds for seed {seed}, found {len(paths)}.")
        for path in paths:
            files[seed, path.parent.name.split("-", 2)[1]] = path
    with ProcessPoolExecutor(max_workers=5) as workers:
        cells = dict(zip(files, workers.map(score_cell, ((path, gaps, minimums) for path in files.values())),
                         strict=True))
    per_config = {pair: macro([cell[pair] for cell in cells.values()]) for pair in grid}

    def rank(metrics, pair):
        end_mae = metrics["delta_end_mae_frames"]
        return (metrics["interval_f1"], -metrics["false_starts_per_break_minute"],
                -end_mae if end_mae is not None else -float("inf"), -pair[0], -pair[1])

    selections = {}
    selected = {}
    for held in cells:
        others = [key for key in cells if key[0] == held[0] and key[1] != held[1]]
        candidate = max(grid, key=lambda pair: rank(macro([cells[key][pair] for key in others]), pair))
        selections[held] = candidate
        selected[held] = cells[held][candidate]
    frequency = {pair: list(selections.values()).count(pair) for pair in set(selections.values())}
    fixed_best = max(grid, key=lambda pair: rank(per_config[pair], pair))
    modal = max(frequency, key=lambda pair: (frequency[pair], *rank(per_config[pair], pair)))
    output = {
        "seeds": SEEDS, "gaps_seconds": gaps, "minimums_seconds": minimums,
        "fps": FPS, "production": name(PRODUCTION), "incumbent": name(INCUMBENT),
        "best_fixed": name(fixed_best), "modal_choice": name(modal),
        "selected_macro": macro(list(selected.values())),
        "selection_frequency": {name(pair): frequency[pair] for pair in sorted(frequency)},
        "selected_by_cell": {f"{seed}/{fold}": name(pair) for (seed, fold), pair in selections.items()},
        "configurations": {name(pair): metrics for pair, metrics in per_config.items()},
        "cells": {f"{seed}/{fold}": {name(pair): value for pair, value in metrics.items()}
                  for (seed, fold), metrics in cells.items()},
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"{len(grid)} configurations across {len(cells)} development cells")
    print("fixed best", name(fixed_best), "modal", name(modal), "LOCO F1", output["selected_macro"]["interval_f1"])
    for pair in sorted(grid, key=lambda pair: rank(per_config[pair], pair), reverse=True)[:20]:
        data = per_config[pair]
        print(name(pair), *(round(data[field], 4) if data[field] is not None else None for field in (
            "interval_f1", "false_starts_per_break_minute", "false_merges", "missed_rallies",
            "delta_signed_end_frames", "delta_end_mae_frames", "frame_f1")),
            frequency.get(pair, 0))


if __name__ == "__main__":
    main()
