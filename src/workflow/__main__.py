"""Low-friction source registration, annotation, and experiment setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import SourceCatalog


DEFAULT_CATALOG = Path("config/sources.local.json")
DEFAULT_RUNTIME = Path(".annotation-final")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser("source")
    source_commands = source.add_subparsers(dest="source_command", required=True)
    add = source_commands.add_parser("add")
    add.add_argument("--video", type=Path, required=True)
    add.add_argument("--source-id")
    add.add_argument("--artifact-root", type=Path)
    status = source_commands.add_parser("status", aliases=["doctor"])
    status.add_argument("--source", action="append", dest="sources")
    status.add_argument("--json", action="store_true")

    stage = commands.add_parser("stage")
    stage_commands = stage.add_subparsers(dest="stage_command", required=True)
    imported = stage_commands.add_parser("import-calibration")
    imported.add_argument("--source", required=True)
    imported.add_argument("--from", type=Path, required=True, dest="external_path")
    imported.add_argument("--force", action="store_true")
    artifact_import = stage_commands.add_parser("import-artifact")
    artifact_import.add_argument("--source", required=True)
    artifact_import.add_argument(
        "--kind",
        required=True,
        choices=(
            "shuttle-candidates-pilot", "shuttle-candidates-frozen",
            "person-tracks", "player-poses",
        ),
    )
    artifact_import.add_argument("--from", type=Path, required=True, dest="external_path")
    artifact_import.add_argument("--force", action="store_true")
    calibrate = stage_commands.add_parser("calibrate")
    calibrate.add_argument("--source", required=True)
    calibrate.add_argument("--frame", type=int, default=0)
    calibrate.add_argument("--mode", choices=("lines", "points"), default="lines")
    calibrate.add_argument("--ui", choices=("browser", "opencv"), default="browser")
    calibrate.add_argument("--open-browser", action="store_true")
    calibrate.add_argument("--preview", type=Path)
    people = stage_commands.add_parser("person-tracks")
    people.add_argument("--source", required=True)
    people.add_argument("--model", default="yolov8n.pt")
    people.add_argument("--force", action="store_true")
    players = stage_commands.add_parser("players")
    players.add_argument("--source", required=True)
    players.add_argument("--pose-model", type=Path)
    players.add_argument("--force", action="store_true")
    check = stage_commands.add_parser("check")
    check.add_argument("--source", required=True)
    check.add_argument(
        "--artifact",
        required=True,
        choices=("shuttle-candidates", "person-tracks", "calibration", "player-assignments", "player-poses"),
    )

    annotate = commands.add_parser("annotate")
    annotate.add_argument("--annotator", required=True)
    annotate.add_argument("--source", action="append", dest="sources")
    annotate.add_argument("--runtime", type=Path)
    annotate.add_argument("--queue", choices=("adaptive", "audit", "rally-audit", "refill"), default="adaptive")
    annotate.add_argument("--stage", choices=("pilot", "frozen"), default="frozen")
    annotate.add_argument("--host", default="127.0.0.1")
    annotate.add_argument("--port", type=int, default=8050)
    annotate.add_argument("--new-session", action="store_true")

    experiment = commands.add_parser("experiment")
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)
    prepare = experiment_commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, default=Path("config/selector-experiment.generated.json"))
    prepare.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    prepare.add_argument("--source", action="append", dest="sources")
    prepare.add_argument("--seed", type=int, default=1729)
    validate = experiment_commands.add_parser("validate")
    validate.add_argument("--source", action="append", dest="sources")

    rally = commands.add_parser("rally")
    rally_commands = rally.add_subparsers(dest="rally_command", required=True)
    pseudo = rally_commands.add_parser("pseudo-label")
    pseudo.add_argument("--source", required=True)
    pseudo.add_argument("--teacher", type=Path, required=True)
    pseudo.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    pseudo.add_argument("--batch-size", type=int, default=16)
    review = rally_commands.add_parser("review")
    review.add_argument("--source", required=True)
    review.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    review.add_argument("--annotator", required=True)
    return parser


def _load_or_create(path: Path) -> SourceCatalog:
    path = path.expanduser().resolve()
    return SourceCatalog.read(path) if path.exists() else SourceCatalog(path)


def _status_mapping(catalog: SourceCatalog, source_ids: list[str] | None) -> list[dict[str, object]]:
    rows = []
    for source in catalog.select(source_ids):
        status = catalog.status(source)
        rows.append(
            {
                "source_id": status.source_id,
                "enabled": status.enabled,
                "video": status.video,
                "identity": status.identity,
                "candidates": status.candidates,
                "frozen_candidates": status.frozen_candidates,
                "person_tracks": status.person_tracks,
                "assignments": status.assignments,
                "pose_cache": status.pose_cache,
                "calibration": status.calibration,
                "annotation_ready": status.annotation_ready,
                "experiment_ready": status.experiment_ready,
                "rally_ready": status.rally_ready,
                "problems": list(status.problems),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    catalog = _load_or_create(args.catalog)
    if args.command == "source" and args.source_command == "add":
        source = catalog.add(
            args.video,
            source_id=args.source_id,
            output_dir=args.artifact_root,
        )
        print(f"registered {source.source_id}")
        print(f"catalog: {catalog.path}")
        return 0
    if args.command == "source":
        rows = _status_mapping(catalog, args.sources)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                state = "experiment-ready" if row["experiment_ready"] else "annotation-ready" if row["annotation_ready"] else "blocked"
                print(f"{row['source_id']}: {state}")
                for field in ("video", "identity", "candidates", "frozen_candidates", "person_tracks", "assignments", "pose_cache", "calibration"):
                    print(f"  {field.replace('_', ' '):18} {'ready' if row[field] else 'missing'}")
                for problem in row["problems"]:
                    print(f"  problem: {problem}")
        return 0
    if args.command == "stage":
        source = catalog.select([args.source])[0]
        if args.stage_command == "import-calibration":
            from .artifacts import import_calibration

            print(import_calibration(source, args.external_path, force=args.force))
            return 0
        if args.stage_command == "import-artifact":
            from .artifacts import import_artifact

            print(import_artifact(source, args.kind, args.external_path, force=args.force))
            return 0
        if args.stage_command == "calibrate":
            from src.calibrate_court import main as calibrate_main
            from .artifacts import register_native_calibration

            forwarded = [
                str(source.video_path), str(source.calibration_path),
                "--frame", str(args.frame), "--mode", args.mode, "--ui", args.ui,
            ]
            if args.open_browser:
                forwarded.append("--open-browser")
            if args.preview:
                forwarded.extend(("--preview", str(args.preview)))
            selected_frame = calibrate_main(forwarded)
            print(register_native_calibration(source, source_frame=selected_frame))
            return 0
        if args.stage_command == "person-tracks":
            from InPlay.heuristic.person_tracks import extract_person_tracks

            if source.person_tracks_path.exists() and not args.force:
                raise FileExistsError(f"person tracks already exist: {source.person_tracks_path}")
            source.person_tracks_path.parent.mkdir(parents=True, exist_ok=True)
            extract_person_tracks(source.video_path, source.person_tracks_path, args.model)
            print(source.person_tracks_path)
            return 0
        if args.stage_command == "players":
            from InPlay.heuristic.player_interpretation import interpret_players

            if source.assignments_path.exists() and not args.force:
                raise FileExistsError(f"player assignments already exist: {source.assignments_path}")
            if not source.person_tracks_path.is_file():
                raise FileNotFoundError(f"person tracks are missing: {source.person_tracks_path}")
            if not source.calibration_path.is_file():
                raise FileNotFoundError(f"canonical calibration is missing: {source.calibration_path}")
            source.assignments_path.parent.mkdir(parents=True, exist_ok=True)
            assignments, computed = interpret_players(
                source.video_path,
                source.person_tracks_path,
                source.calibration_path,
                source.pose_cache_path,
                source.assignments_path,
                source.artifact_root / "players" / "features.csv",
                pose_model_asset=args.pose_model,
            )
            print(f"{source.source_id}: {len(assignments)} assignment frames; {computed} poses computed")
            return 0
        status = catalog.status(source)
        artifact_state = {
            "shuttle-candidates": status.frozen_candidates or status.candidates,
            "person-tracks": source.person_tracks_path.is_file(),
            "calibration": status.calibration and not any(
                "calibration" in problem for problem in status.problems
            ),
            "player-assignments": status.assignments,
            "player-poses": status.pose_cache,
        }
        if not artifact_state[args.artifact]:
            raise ValueError(f"artifact is missing or stale: {args.artifact}")
        print(f"{source.source_id}: {args.artifact} ready")
        return 0
    if args.command == "annotate":
        # Keep source/status/experiment commands usable without OpenCV and Dash.
        from src.annotation_platform.__main__ import main as annotation_main

        selected = catalog.select(args.sources)
        if not selected:
            raise ValueError("catalog has no enabled sources")
        if args.runtime is not None:
            runtime = args.runtime
        elif len(selected) == 1:
            runtime = Path(f".annotation-{selected[0].source_id}")
        else:
            runtime = DEFAULT_RUNTIME
        generated = runtime / "generated" / "annotation-sources.json"
        catalog.write_annotation_config(generated, source_ids=[s.source_id for s in selected], stage=args.stage)
        forwarded = [
            "--config", str(generated), "--runtime", str(runtime), "serve",
            "--annotator", args.annotator, "--queue", args.queue,
            "--host", args.host, "--port", str(args.port),
        ]
        if args.new_session:
            forwarded.append("--new-session")
        return annotation_main(forwarded)
    if args.command == "rally":
        source = catalog.select([args.source])[0]
        status = catalog.status(source)
        if not status.rally_ready:
            raise ValueError(
                f"source is not rally-ready: {source.source_id}: "
                + "; ".join(status.problems or ("required player artifacts are missing",))
            )
        if args.rally_command == "pseudo-label":
            from src.temporal_selector.rally_dataset import RallySourceConfig
            from src.temporal_selector.rally_pseudo_label import pseudo_label_source

            predictions, proposals = pseudo_label_source(
                RallySourceConfig(source.source_id, source.video_path,
                                  source.assignments_path, source.pose_cache_path,
                                  source.calibration_path),
                args.teacher, device=args.device, batch_size=args.batch_size,
            )
            print(predictions)
            print(proposals)
            return 0
        from InPlay.heuristic.label_intervals import label_video
        from src.temporal_selector.rally_intervals import read_rally_intervals

        runtime = args.runtime.expanduser().resolve()
        intervals_path = runtime / "rallies.csv"
        manifest_path = runtime / "rallies.manifest.json"
        if intervals_path.exists() != manifest_path.exists():
            raise ValueError("canonical rally CSV and manifest must exist together")
        if intervals_path.exists():
            existing = read_rally_intervals(intervals_path, manifest_path)
            if any(span.source_id == source.source_id for span in existing.reviewed_coverage):
                raise ValueError(f"source already has reviewed rally labels: {source.source_id}")
        if not source.rally_proposals_path.is_file():
            raise FileNotFoundError(f"rally proposals are missing: {source.rally_proposals_path}")
        if not source.rally_predictions_path.is_file():
            raise FileNotFoundError(f"rally predictions are missing: {source.rally_predictions_path}")
        # Proposal loading verifies video identity; verify all remaining inference inputs here.
        proposal = json.loads(source.rally_proposals_path.read_text(encoding="utf-8"))
        prediction_lines = [json.loads(line) for line in source.rally_predictions_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if not prediction_lines or prediction_lines[0] != proposal.get("metadata"):
            raise ValueError("rally predictions and proposals have incompatible metadata")
        frame_count = int(prediction_lines[0].get("frame_count", -1))
        if [int(row.get("frame", -1)) for row in prediction_lines[1:]] != list(range(frame_count)):
            raise ValueError("rally predictions do not cover every source frame exactly once")
        from .identity import sha256_file
        expected = proposal.get("metadata", {}).get("input_artifact_sha256", {})
        current = {"calibration": sha256_file(source.calibration_path),
                   "assignments": sha256_file(source.assignments_path),
                   "poses": sha256_file(source.pose_cache_path)}
        if any(expected.get(name) != value for name, value in current.items()):
            raise ValueError("rally proposals are stale for current source artifacts")
        label_video(
            source.video_path, source.source_id, intervals_path, manifest_path,
            proposals=source.rally_proposals_path,
            draft=source.artifact_root / "rallies" / "review.draft.json",
            receipt=source.artifact_root / "rallies" / "review.receipt.json",
            annotator=args.annotator,
        )
        return 0
    if args.experiment_command == "validate":
        rows = _status_mapping(catalog, args.sources)
        blocked = [str(row["source_id"]) for row in rows if not row["experiment_ready"]]
        if blocked:
            raise ValueError("sources are not experiment-ready: " + ", ".join(blocked))
        print(f"experiment inputs ready: {len(rows)} source(s)")
        return 0
    output, manifest = catalog.write_experiment(
        args.output,
        runtime=args.runtime,
        seed=args.seed,
        source_ids=args.sources,
    )
    print(output)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
