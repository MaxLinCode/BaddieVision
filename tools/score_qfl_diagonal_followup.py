"""Score the preregistered five-seed QFL x diagonal development factorial."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from itertools import groupby
from pathlib import Path
from statistics import fmean

from src.temporal_selector.joint_inference import InPlayDecoderConfig
from src.temporal_selector.rally_evaluation import decode_prediction_rows, rally_metrics

SEEDS = (1729, 1730, 1731, 1732, 1733)
ARMS = {
    "baseline": ("full_audio_zscore_motion", "triangle_bce"),
    "qfl": ("full_audio_zscore_motion", "gaussian_qfl"),
    "diagonal": ("serve_boxes_diagonal", "triangle_bce"),
    "qfl-diagonal": ("serve_boxes_diagonal", "gaussian_qfl"),
}
DECODER = InPlayDecoderConfig(
    threshold=0.40,
    continuation_threshold=0.35,
    max_gap_seconds=0.5,
    minimum_duration_seconds=0.3,
    preserve_edge_runs=True,
)
FPS = 30.0


def read(path):
    return json.loads(path.read_text())


def mean_or_none(values):
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


def false_starts(rows):
    """Count complete predicted runs wholly inside reviewed break frames."""
    ordered = sorted(rows, key=lambda row: (row["source_id"], row["frame"]))
    def key(row):
        return (row["source_id"], row["decoded_inplay"])
    count = 0
    for (_, active), group in groupby(ordered, key):
        if not active:
            continue
        reviewed_break = True
        previous = None
        for row in group:
            if previous is not None and row["frame"] != previous + 1:
                count += reviewed_break
                reviewed_break = True
            reviewed_break &= row["inplay_target"] == 0
            previous = row["frame"]
        count += reviewed_break
    denominator = sum(row["inplay_target"] == 0 for row in ordered) / FPS / 60
    return count, denominator


def annotation_key(row):
    return (
        row["source_id"], row["frame"], row["window_id"], row["inplay_target"],
        row.get("partial_rally"), row["rally_start_target"] == 1, row["rally_end_target"] == 1,
        row["rally_start_target"] >= 0, row["rally_end_target"] >= 0,
    )


def collect_arm(path, seed, name):
    run = read(path / "metrics.json")
    recipe = dict(run["training_recipe"])
    manifest = run["dataset_manifest"]
    view, supervision = ARMS[name]
    assert manifest["feature_view"] == view, path
    assert recipe.pop("boundary_supervision", "triangle_bce") == supervision, path
    assert recipe["seed"] == seed and recipe["epochs"] == 25, recipe
    identity = {
        "data": {key: manifest[key] for key in (
            "sources", "annotation_fingerprint", "context_seconds",
            "ownership_seconds", "pose_visibility_threshold", "pose_representation_version",
            "audio_coverage",
        )},
        "camera_sources": run["camera_sources"],
        "recipe": recipe,
        "folds": {},
    }
    by_camera = {}
    partition_fingerprints = {}
    for folder in sorted(path.glob("fold-*")):
        fold = read(folder / "metrics.json")
        split = fold["split_manifest"]
        camera = split["held_out_camera"]
        assert camera not in by_camera, folder
        assert not split["test_labels_used_for_selection"], folder
        assert not set(split["test_source_ids"]).intersection(split["training_source_ids"]), folder
        with (folder / "test-predictions.jsonl").open() as handle:
            rows = [json.loads(line) for line in handle]
        rows.sort(key=lambda row: (row["source_id"], row["frame"]))
        assert {row["source_id"] for row in rows} == set(split["test_source_ids"]), folder
        assert len(rows) == len({(row["source_id"], row["frame"]) for row in rows}), folder
        assert all(row["partition"] == "test" for row in rows), folder
        train_validation = dict(split["train_validation"])
        partition_fingerprints[camera] = train_validation.pop("partition_fingerprints", {})
        identity["folds"][camera] = {
            "split": {key: split[key] for key in (
                "test_source_ids", "training_source_ids", "validation_source_ids",
                "train_window_ids", "validation_window_ids", "test_window_ids",
            )},
            "train_validation": train_validation,
            "partition_ranges": sorted(partition_fingerprints[camera]),
            "annotations": [annotation_key(row) for row in rows],
        }
        decoded = decode_prediction_rows(
            rows, fps_by_source={source: FPS for source in split["test_source_ids"]}, config=DECODER
        )
        metrics = rally_metrics(decoded)
        intervals = metrics["intervals"]
        matched = {
            (row["source_id"], row["label_rally_id"]): row
            for row in metrics["interval_matches"] if row["match_status"] == "matched"
        }
        extras, break_minutes = false_starts(decoded)
        assert break_minutes > 0, folder
        by_camera[camera] = {
            "chosen_epoch": fold["chosen_epoch"],
            "interval_f1": intervals["f1"],
            "interval_precision": intervals["precision"],
            "interval_recall": intervals["recall"],
            "matched_rallies": intervals["matched_count"],
            "labeled_rallies": intervals["label_count"],
            "false_starts": extras,
            "reviewed_break_minutes": break_minutes,
            "false_starts_per_break_minute": extras / break_minutes,
            "false_splits": intervals["false_split_count"],
            "false_merges": intervals["false_merge_count"],
            "start_mae_frames": mean_or_none([abs(row["start_error"]) for row in matched.values()]),
            "end_mae_frames": mean_or_none([abs(row["end_error"]) for row in matched.values()]),
            "decoded_frame_f1": metrics["decoded_frame"]["f1"],
            "matched": matched,
        }
    assert len(by_camera) == len(run["folds"]) == len(run["camera_sources"]) == 4, path
    return identity, by_camera, partition_fingerprints


def macro(folds):
    return {
        field: mean_or_none([fold[field] for fold in folds])
        for field in folds[0]
    }


def paired_effect(cells):
    base, qfl, diagonal, combined = (cells[name] for name in ARMS)
    return {
        "qfl_vs_baseline": qfl - base,
        "diagonal_vs_baseline": diagonal - base,
        "combined_vs_baseline": combined - base,
        "combined_vs_qfl": combined - qfl,
        "combined_vs_diagonal": combined - diagonal,
        "interaction": combined - qfl - diagonal + base,
    }


def collect(root):
    per_seed = {}
    shared_data = None
    partitions_by_view = {}
    for seed in SEEDS:
        reference = None
        arms = {}
        for name in ARMS:
            identity, arms[name], partitions = collect_arm(
                root / f"seed-{seed}" / name, seed, name
            )
            # A partition dataset fingerprint incorporates the feature view.
            view = ARMS[name][0]
            assert partitions_by_view.setdefault(view, partitions) == partitions, (seed, name)
            if reference is None:
                reference = identity
            else:
                assert reference == identity, f"Data, frames, or training settings differ for {seed}/{name}"
        data_identity = {key: value for key, value in reference.items() if key != "recipe"}
        if shared_data is None:
            shared_data = data_identity
        else:
            assert shared_data == data_identity, f"Development folds changed across seeds: {seed}"
        for camera in reference["camera_sources"]:
            shared_matches = set.intersection(*(set(out[camera]["matched"]) for out in arms.values()))
            for out in arms.values():
                fold = out[camera]
                matched = fold.pop("matched")
                fold["common_start_mae_frames"] = mean_or_none(
                    [abs(matched[key]["start_error"]) for key in shared_matches]
                )
                fold["common_end_mae_frames"] = mean_or_none(
                    [abs(matched[key]["end_error"]) for key in shared_matches]
                )
                fold["common_matched_rallies"] = len(shared_matches)
        per_seed[seed] = {
            name: {"macro": macro(list(cameras.values())), "by_camera": cameras}
            for name, cameras in arms.items()
        }
    overall = {
        name: macro([
            camera for values in per_seed.values()
            for camera in values[name]["by_camera"].values()
        ])
        for name in ARMS
    }
    outcomes = ("interval_f1", "interval_recall", "false_starts_per_break_minute", "common_end_mae_frames")
    comparisons = {}
    for field in outcomes:
        cells = {name: values[field] for name, values in overall.items()}
        assert all(value is not None and math.isfinite(value) for value in cells.values()), field
        comparisons[field] = paired_effect(cells)
    per_cell_effects = {
        seed: {
            camera: {
                field: paired_effect({name: per_seed[seed][name]["by_camera"][camera][field] for name in ARMS})
                for field in outcomes
                if all(per_seed[seed][name]["by_camera"][camera][field] is not None for name in ARMS)
            }
            for camera in shared_data["camera_sources"]
        }
        for seed in SEEDS
    }
    return {
        "seeds": SEEDS,
        "fixed_decoder": asdict(DECODER),
        "fps": FPS,
        "training_sources_per_camera": shared_data["camera_sources"],
        "annotation_fingerprint": shared_data["data"]["annotation_fingerprint"],
        "arms": ARMS,
        "overall": overall,
        "by_seed": per_seed,
        "paired_effects": comparisons,
        "cell_effects": per_cell_effects,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.results / "protocol.json")
    if (
        protocol["holdout_used"]
        or protocol["seed_order"] != list(SEEDS)
        or protocol["held_out_camera_folds"] != 4
        or protocol["production_decoder"] != asdict(DECODER)
    ):
        raise ValueError("The run protocol differs from the frozen comparison")
    if protocol["arms"] != {
        name: {"feature_view": view, "boundary_supervision": supervision}
        for name, (view, supervision) in ARMS.items()
    }:
        raise ValueError("The run protocol has different ablation arms")
    result = collect(args.results)
    if protocol["development_annotation_fingerprint"] != result["annotation_fingerprint"]:
        raise ValueError("The predictions use different development annotations")
    output = args.output.with_suffix(".tmp")
    output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    output.replace(args.output)
    for name, metrics in result["overall"].items():
        print(
            f"{name:14} F1 {metrics['interval_f1']:.3f} "
            f"recall {metrics['interval_recall']:.3f} "
            f"false starts/min {metrics['false_starts_per_break_minute']:.2f} "
            f"common end MAE {metrics['common_end_mae_frames']:.1f} frames"
        )


if __name__ == "__main__":
    main()
