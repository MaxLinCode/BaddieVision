"""One-shot holdout confirmation of the frozen decoder factorial choice."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from InPlay.heuristic.evaluate import Interval, read_intervals
from src.temporal_selector.joint_inference import (
    InPlayDecoderConfig,
    decode_inplay_probabilities,
    decoded_intervals,
)
from src.temporal_selector.rally_intervals import read_rally_intervals


def load_holdout(holdout: Path):
    labels_path = holdout / "labels" / "rallies.csv"
    index = read_rally_intervals(labels_path, labels_path.with_suffix(".manifest.json"))
    labels = read_intervals(labels_path)
    return index, labels


def predict(index, holdout: Path, config: InPlayDecoderConfig, check_reference: bool = False):
    predictions = []
    for source, metadata in index.sources.items():
        path = holdout / "artifacts" / source / "rallies"
        rows = [json.loads(line) for line in (path / "probabilities.jsonl").open()]
        frames = [row["frame"] for row in rows]
        assert frames == list(range(metadata.frame_count)), path
        probabilities = [row["inplay_probability"] for row in rows]
        states = decode_inplay_probabilities(probabilities, fps=metadata.fps, config=config)
        intervals = decoded_intervals(frames, states)
        if check_reference:
            reference = read_intervals(path / "rallies.csv")
            assert intervals == tuple((item.start, item.end) for item in reference), path
        predictions.extend(
            Interval(source, f"{source}-{number:04d}", start, end)
            for number, (start, end) in enumerate(intervals, 1)
        )
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-json", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.holdout / "evaluation"))
    from score import score  # noqa: E402

    development = json.loads(args.dev_json.read_text())
    deployment = development["deployment"]
    fixed = development["fixed"]
    configs = {
        "production": InPlayDecoderConfig(
            max_gap_seconds=development["production"]["max_gap_seconds"],
            minimum_duration_seconds=development["production"]["minimum_duration_seconds"],
            **fixed,
        ),
        "deployment": InPlayDecoderConfig(
            max_gap_seconds=deployment["max_gap_seconds"],
            minimum_duration_seconds=deployment["minimum_duration_seconds"],
            **fixed,
        ),
    }
    index, labels = load_holdout(args.holdout)
    results = {}
    matches = {}
    for name, config in configs.items():
        predictions = predict(index, args.holdout, config, check_reference=(name == "production"))
        metrics, details = score(index, predictions, labels)
        results[name] = {
            "config": {
                "threshold": config.threshold,
                "continuation_threshold": config.continuation_threshold,
                "max_gap_seconds": config.max_gap_seconds,
                "minimum_duration_seconds": config.minimum_duration_seconds,
                "preserve_edge_runs": config.preserve_edge_runs,
            },
            "pooled": metrics["pooled"],
            "match_macro_interval_f1": metrics["match_macro_interval_f1"],
        }
        matches[name] = {
            (row["source_id"], row["label_rally_id"]): row
            for row in details if row["match_status"] == "matched"
        }
    shared = set(matches["production"]) & set(matches["deployment"])
    comparison = {
        "shared_matched_rallies": len(shared),
        "production_end_mae_frames": statistics.fmean(abs(matches["production"][key]["end_error"]) for key in shared),
        "deployment_end_mae_frames": statistics.fmean(abs(matches["deployment"][key]["end_error"]) for key in shared),
        "production_start_mae_frames": statistics.fmean(abs(matches["production"][key]["start_error"]) for key in shared),
        "deployment_start_mae_frames": statistics.fmean(abs(matches["deployment"][key]["start_error"]) for key in shared),
    }
    result = {
        "schema": "decoder_factorial_holdout",
        "schema_version": 1,
        "holdout": str(args.holdout),
        "development_deployment": deployment,
        "development_fixed": fixed,
        "results": results,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, value in results.items():
        pooled = value["pooled"]
        print(f"{name:11} F1 {pooled['f1']:.4f} P {pooled['precision']:.4f} R {pooled['recall']:.4f} "
              f"fs/min {pooled['false_starts_per_break_minute']:.2f} endMAE {pooled['end_mae_frames']:.1f} "
              f"matchMacro {value['match_macro_interval_f1']:.4f} splits {pooled['false_split_count']} merges {pooled['false_merge_count']}")
    print("shared", comparison)


if __name__ == "__main__":
    main()
