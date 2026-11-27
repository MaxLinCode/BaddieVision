"""Compare the three matched InPlay serve-box feature runs."""

import argparse
import json
from pathlib import Path
from statistics import fmean

from src.temporal_selector.rally_features import CANONICAL_RALLY_FEATURE_VIEW

VIEWS = (CANONICAL_RALLY_FEATURE_VIEW, "serve_boxes", "serve_boxes_diagonal")


def summarize(run_root: Path) -> dict:
    reference = None
    arms = {}
    for view in VIEWS:
        metrics = json.loads((run_root / view / "metrics.json").read_text())
        dataset = metrics["dataset_manifest"]
        if dataset["feature_view"] != view:
            raise ValueError(f"expected {view}, got {dataset['feature_view']}")
        identity = {
            "dataset": {key: dataset[key] for key in (
                "sources", "annotation_fingerprint", "context_seconds", "ownership_seconds",
                "pose_visibility_threshold", "pose_representation_version", "audio_coverage",
            )},
            "recipe": metrics["training_recipe"],
            "folds": [],
        }
        by_camera = {}
        for fold in metrics["folds"]:
            split = fold["split_manifest"]
            identity["folds"].append({
                "camera": split["held_out_camera"],
                "ranges": split["train_validation"]["cameras"],
                **{f"{part}_window_ids": split[f"{part}_window_ids"]
                   for part in ("train", "validation", "test")},
            })
            test = fold["test"]
            starts = [
                match["start_error"] for match in test["interval_matches"]
                if match["match_status"] == "matched"
            ]
            by_camera[split["held_out_camera"]] = {
                "interval_f1": test["intervals"]["f1"],
                "interval_recall": test["intervals"]["recall"],
                "matched_starts": len(starts),
                "start_mae_frames": fmean(map(abs, starts)) if starts else None,
                "signed_start_error_frames": fmean(starts) if starts else None,
            }
        if reference is None:
            reference = identity
        elif reference != identity:
            raise ValueError(f"{view} differs from baseline in data, recipe, or split")
        start_maes = [row["start_mae_frames"] for row in by_camera.values()
                      if row["start_mae_frames"] is not None]
        arms[view] = {
            **metrics["macro"],
            "macro_start_mae_frames": fmean(start_maes) if start_maes else None,
            "by_camera": by_camera,
        }
    return {"training_recipe": reference["recipe"], "arms": arms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    report = summarize(args.run_root)
    (args.run_root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
