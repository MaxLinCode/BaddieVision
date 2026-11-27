"""Summarize held-out predictions for the 4, 8, and 16 second InPlay grid."""

import argparse
import hashlib
import json
from itertools import groupby
from pathlib import Path

from src.temporal_selector.rally_dataset import rally_data_config_from_selector_mapping
from src.temporal_selector.rally_intervals import read_rally_intervals
from src.workflow.artifacts import atomic_write_json

SPANS = (4, 8, 16)


def break_predictions(rows, fps):
    """Find decoded intervals that start and finish in reviewed breaks."""
    events = []
    for (is_inplay, _), group in groupby(enumerate(rows), key=lambda item: (
        item[1]["decoded_inplay"], item[1]["frame"] - item[0]
    )):
        if not is_inplay:
            continue
        run = list(group)
        start_index, start = run[0]
        end_index, end = run[-1]
        if start_index == 0 or end_index == len(rows) - 1:
            continue
        before, after = rows[start_index - 1], rows[end_index + 1]
        if (
            before["frame"] != start["frame"] - 1 or after["frame"] != end["frame"] + 1
            or before["decoded_inplay"] or after["decoded_inplay"]
            or before["inplay_target"] != 0 or after["inplay_target"] not in (0, 1)
            or any(row["inplay_target"] != 0 for _, row in run)
        ):
            continue
        events.append({
            "start_frame": start["frame"],
            "end_frame": end["frame"],
            "start_seconds": start["frame"] / fps,
            "duration_seconds": len(run) / fps,
        })
    return events


def summarize(root, fps_by_source):
    runs = {}
    reference = None
    for span in SPANS:
        arm = root / f"{span}s"
        metrics = json.loads((arm / "metrics.json").read_text())
        dataset = metrics["dataset_manifest"]
        if dataset["context_seconds"] * 2 != span:
            raise ValueError(f"{arm} has the wrong context duration")
        split_keys = ("train_window_ids", "validation_window_ids", "test_window_ids")
        identity = {
            "dataset": {key: dataset[key] for key in (
                "sources", "annotation_fingerprint", "feature_view", "feature_names",
                "ownership_seconds", "audio_coverage", "pose_visibility_threshold",
                "pose_coordinate_mode", "pose_representation_version",
            )},
            "pose": {key: value for key, value in dataset["pose_training_metadata"]["preprocessing"].items()
                     if key != "context_seconds"},
            "pose_model": dataset["pose_training_metadata"]["pose_model_sha256"],
            "recipe": metrics["training_recipe"],
            "camera_sources": metrics["camera_sources"],
            "folds": [
                {**{key: fold["split_manifest"][key] for key in split_keys},
                 "ranges": fold["split_manifest"]["train_validation"]["cameras"]}
                for fold in metrics["folds"]
            ],
        }
        by_camera = {}
        labels = hashlib.sha256()
        for fold in metrics["folds"]:
            split = fold["split_manifest"]
            camera = split["held_out_camera"]
            prediction_path = arm / f"fold-{split['fold_id']}-{camera}" / "test-predictions.jsonl"
            sources = {}
            with prediction_path.open() as predictions:
                for source, group in groupby(map(json.loads, predictions), key=lambda row: row["source_id"]):
                    rows = list(group)
                    labels.update(json.dumps([source, [(row["frame"], row["inplay_target"])
                                                       for row in rows]]).encode())
                    events = break_predictions(rows, fps_by_source[source])
                    sources[source] = {
                        "false_starts": len(events),
                        "reviewed_break_seconds": sum(row["inplay_target"] == 0 for row in rows) / fps_by_source[source],
                        "events": events,
                    }
            intervals = fold["test"]["intervals"]
            by_camera[camera] = {
                "false_starts": sum(row["false_starts"] for row in sources.values()),
                "interval_f1": intervals["f1"],
                "interval_precision": intervals["precision"],
                "interval_recall": intervals["recall"],
                "false_splits": intervals["false_split_count"],
                "by_source": sources,
            }
        identity["held_out_labels"] = labels.hexdigest()
        if reference is None:
            reference = identity
        elif reference != identity:
            raise ValueError(f"{arm} differs from the 4-second arm in inputs, recipe, or splits")
        runs[f"{span}s"] = {
            "false_starts": sum(row["false_starts"] for row in by_camera.values()),
            "reviewed_break_seconds": sum(
                source["reviewed_break_seconds"]
                for row in by_camera.values() for source in row["by_source"].values()
            ),
            "macro": metrics["macro"],
            "by_camera": by_camera,
        }
    return {"context_spans_seconds": list(SPANS), "training_recipe": reference["recipe"], "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    config = rally_data_config_from_selector_mapping(
        json.loads(args.config.read_text()), base_dir=args.config.resolve().parent,
    )
    index = read_rally_intervals(config.rally_intervals_path, config.rally_manifest_path)
    result = summarize(args.run_root, {source: manifest.fps for source, manifest in index.sources.items()})
    atomic_write_json(args.run_root / "comparison.json", result)


if __name__ == "__main__":
    main()
