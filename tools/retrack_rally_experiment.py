"""Regenerate tracks, assignments, and poses for a rally experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.workflow.artifacts import atomic_write_json, calibration_fingerprint
from src.workflow.registry import SourceEntry
from src.workflow.source import select_device
from src.workflow.stages import run_person_stage, run_player_stage


def _prepare_source(
    source_id: str,
    video: Path,
    calibration: Path,
    artifact_root: Path,
) -> SourceEntry:
    if calibration.name != "calibration.json" or calibration.parent.name != "court":
        raise ValueError("retracking requires canonical court/calibration.json inputs")
    original = SourceEntry(source_id, video, calibration.parent.parent)
    if artifact_root == original.artifact_root:
        raise ValueError("retracking output must use a separate artifact root")
    calibration_fingerprint(original)

    source = SourceEntry(source_id, video, artifact_root)
    for original_path, snapshot in (
        (original.identity_path, source.identity_path),
        (original.calibration_path, source.calibration_path),
    ):
        value = json.loads(original_path.read_text(encoding="utf-8"))
        if not snapshot.exists():
            atomic_write_json(snapshot, value)
        elif json.loads(snapshot.read_text(encoding="utf-8")) != value:
            raise ValueError(
                f"source inputs changed for {source_id}; use a new output directory"
            )
    return source


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
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
        dataset[key] = str((config_path.parent / Path(dataset[key]).expanduser()).resolve())
    for entry in dataset["sources"]:
        source_id = entry["source_id"]
        video = (
            config_path.parent / Path(entry["video_path"]).expanduser()
        ).resolve(strict=True)
        calibration = (
            config_path.parent / Path(entry["calibration_path"]).expanduser()
        ).resolve(strict=True)
        source = _prepare_source(
            source_id, video, calibration, output / "sources" / source_id
        )
        print(f"{source_id}: checking person tracks", flush=True)
        run_person_stage(source, model=args.model, device=device)
        print(f"{source_id}: checking player assignments and poses", flush=True)
        run_player_stage(source, pose_model=args.pose_model)
        entry.update(
            video_path=str(source.video_path),
            calibration_path=str(source.calibration_path),
            assignments_path=str(source.assignments_path),
            pose_cache_path=str(source.pose_cache_path),
        )
        print(f"{source_id}: complete", flush=True)
    atomic_write_json(output / "experiment.json", config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
