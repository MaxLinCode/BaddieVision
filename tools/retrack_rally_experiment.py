"""Regenerate tracks, assignments, and poses for a rally experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from InPlay.heuristic.person_tracks import extract_person_tracks
from InPlay.heuristic.player_interpretation import interpret_players
from src.workflow.source import select_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument(
        "--resume", action="store_true", help="Reuse outputs from the same run"
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    dataset = config["dataset"]
    source_ids = [source["source_id"] for source in dataset["sources"]]
    if (
        not source_ids
        or len(set(source_ids)) != len(source_ids)
        or any(
            not name or Path(name).name != name or name == ".." for name in source_ids
        )
    ):
        parser.error("source IDs must be unique, nonempty path components")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error(
            "output directory already contains data; use --resume for the same run"
        )
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    for key in ("rally_intervals_path", "rally_manifest_path"):
        dataset[key] = str((config_path.parent / dataset[key]).resolve())
    for source in dataset["sources"]:
        source_id = source["source_id"]
        for key in ("video_path", "calibration_path"):
            source[key] = str((config_path.parent / source[key]).resolve(strict=True))
        root = output / "sources" / source_id
        root.mkdir(parents=True, exist_ok=True)
        tracks = root / "person_tracks.jsonl"
        poses = root / "poses.jsonl"
        assignments = root / "assignments.jsonl"
        players_csv = root / "players.csv"
        if not tracks.is_file():
            print(f"{source_id}: tracking", flush=True)
            partial = tracks.with_suffix(".partial.jsonl")
            extract_person_tracks(
                source["video_path"], partial, args.model, device=device
            )
            partial.replace(tracks)
        if not all(path.is_file() for path in (assignments, poses, players_csv)):
            print(f"{source_id}: assigning players and extracting pose", flush=True)
            interpret_players(
                source["video_path"],
                tracks,
                source["calibration_path"],
                poses,
                assignments,
                players_csv,
                pose_model_asset=args.pose_model,
            )
        source["assignments_path"] = str(assignments)
        source["pose_cache_path"] = str(poses)
        print(f"{source_id}: complete", flush=True)
    pending_config = output / "experiment.partial.json"
    pending_config.write_text(json.dumps(config, indent=2) + "\n")
    pending_config.replace(output / "experiment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
