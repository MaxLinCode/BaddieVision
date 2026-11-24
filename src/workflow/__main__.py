"""Low-friction source registration, annotation, and experiment setup."""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path

from InPlay.heuristic.person_tracks import BOTSORT_CONTINUOUS, PLAYER_TRACKING_PROFILES

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
    add.add_argument("--rally-30-fps", action="store_true")
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
            "shuttle-candidates-pilot",
            "shuttle-candidates-frozen",
            "person-tracks",
            "player-poses",
        ),
    )
    artifact_import.add_argument(
        "--from", type=Path, required=True, dest="external_path"
    )
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
    people.add_argument("--model", help="person detector weights or TensorRT engine")
    people.add_argument(
        "--tracking-profile", choices=PLAYER_TRACKING_PROFILES, default=BOTSORT_CONTINUOUS
    )
    people.add_argument("--force", action="store_true")
    people.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    players = stage_commands.add_parser("players")
    players.add_argument("--source", required=True)
    players.add_argument("--pose-model", type=Path)
    players.add_argument("--force", action="store_true")
    players.add_argument("--pose-crop-padding", type=float)
    players.add_argument("--pose-backend", choices=("mediapipe", "rtmpose"), default="mediapipe")
    players.add_argument("--pose-device", choices=("cpu", "cuda"))
    players.add_argument("--tracking-profile", choices=PLAYER_TRACKING_PROFILES)
    check = stage_commands.add_parser("check")
    check.add_argument("--source", required=True)
    check.add_argument(
        "--artifact",
        required=True,
        choices=(
            "shuttle-candidates",
            "person-tracks",
            "calibration",
            "player-assignments",
            "player-poses",
        ),
    )

    extract = commands.add_parser(
        "extract", help="shared local and cloud observation extraction"
    )
    extraction_inputs = extract.add_mutually_exclusive_group(required=True)
    extraction_inputs.add_argument("--source")
    extraction_inputs.add_argument("--video", type=Path)
    extract.add_argument("--source-id")
    extract.add_argument("--output-dir", type=Path)
    extract.add_argument("--calibration", type=Path)
    extract.add_argument(
        "--profile", choices=("observations", "rally"), default="observations"
    )
    extract.add_argument("--person-model", help="person detector weights or TensorRT engine")
    extract.add_argument("--pose-model", type=Path)
    extract.add_argument("--model-root", type=Path)
    extract.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    extract.add_argument("--batch-size", type=int, default=16)
    extract.add_argument("--num-workers", type=int, default=0)
    extract.add_argument(
        "--rebuild",
        action="append",
        default=[],
        choices=("person", "pose", "shuttle", "shuttle-links", "preview", "export"),
    )
    extract.add_argument("--preview", action="store_true")
    extract.add_argument("--export-zip", type=Path)
    extract.add_argument("--force", action="store_true")

    annotate = commands.add_parser("annotate")
    annotate.add_argument("--annotator", required=True)
    annotate.add_argument("--source", action="append", dest="sources")
    annotate.add_argument("--runtime", type=Path)
    annotate.add_argument(
        "--queue",
        choices=("adaptive", "audit", "rally-audit", "refill"),
        default="adaptive",
    )
    annotate.add_argument("--stage", choices=("pilot", "frozen"), default="frozen")
    annotate.add_argument("--host", default="127.0.0.1")
    annotate.add_argument("--port", type=int, default=8050)
    annotate.add_argument("--new-session", action="store_true")

    experiment = commands.add_parser("experiment")
    experiment_commands = experiment.add_subparsers(
        dest="experiment_command", required=True
    )
    prepare = experiment_commands.add_parser("prepare")
    prepare.add_argument(
        "--output", type=Path, default=Path("config/selector-experiment.generated.json")
    )
    prepare.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    prepare.add_argument("--source", action="append", dest="sources")
    prepare.add_argument("--seed", type=int, default=1729)
    validate = experiment_commands.add_parser("validate")
    validate.add_argument("--source", action="append", dest="sources")

    rally = commands.add_parser("rally")
    rally_commands = rally.add_subparsers(dest="rally_command", required=True)
    predict = rally_commands.add_parser(
        "predict", help="video to rally CSV and visual overlay"
    )
    inputs = predict.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", type=Path)
    inputs.add_argument("--source")
    predict.add_argument("--source-id")
    predict.add_argument("--output-dir", type=Path)
    predict.add_argument("--calibration", type=Path)
    predict.add_argument("--model", default="rally-six-source-v1")
    predict.add_argument("--checkpoint", type=Path)
    predict.add_argument("--model-manifest", type=Path)
    predict.add_argument("--cache-dir", type=Path)
    predict.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    predict.add_argument("--batch-size", type=int, default=16)
    predict.add_argument("--no-overlay", action="store_true")
    predict.add_argument("--force", action="store_true")
    predict.add_argument(
        "--rebuild",
        action="append",
        default=[],
        choices=("person", "pose", "predict", "decode", "render"),
    )
    pseudo = rally_commands.add_parser("pseudo-label")
    pseudo.add_argument("--source", required=True)
    pseudo.add_argument("--teacher", type=Path, required=True)
    pseudo.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    pseudo.add_argument("--batch-size", type=int, default=16)
    review = rally_commands.add_parser("review")
    review.add_argument("--source", required=True)
    review.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    review.add_argument("--annotator", required=True)

    augment = commands.add_parser(
        "augment", help="create isolated augmented footage sources"
    )
    augment_commands = augment.add_subparsers(
        dest="augment_command", required=True
    )
    augment_list = augment_commands.add_parser("list")
    augment_list.add_argument("--json", action="store_true")
    augment_create = augment_commands.add_parser("create")
    augment_create.add_argument("--parent-source-id", required=True)
    augment_create.add_argument("--parent-video", type=Path, required=True)
    augment_create.add_argument("--parent-calibration", type=Path, required=True)
    augment_create.add_argument("--transform", required=True)
    augment_create.add_argument("--output-root", type=Path, required=True)
    augment_create.add_argument("--source-id")
    return parser


def _load_or_create(path: Path) -> SourceCatalog:
    path = path.expanduser().resolve()
    return SourceCatalog.read(path) if path.exists() else SourceCatalog(path)


def _status_mapping(
    catalog: SourceCatalog, source_ids: list[str] | None
) -> list[dict[str, object]]:
    from .stages import read_run_status

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
                "run": read_run_status(source),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    catalog = _load_or_create(args.catalog)
    if args.command == "extract":
        from .extraction import run_extraction

        if args.source and args.source_id:
            raise ValueError("--source-id applies only to --video")
        try:
            outputs = run_extraction(
                catalog,
                source_id=args.source or args.source_id,
                video=args.video,
                output_dir=args.output_dir,
                calibration=args.calibration,
                profile=args.profile,
                person_model=args.person_model,
                pose_model=args.pose_model,
                model_root=args.model_root,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                rebuild=args.rebuild,
                force=args.force,
                preview=args.preview,
                export_zip=args.export_zip,
            )
        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            subprocess.CalledProcessError,
        ) as exc:
            print(f"extraction failed: {exc}", file=sys.stderr)
            return 2
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0
    if args.command == "rally" and args.rally_command == "predict":
        from .rally_pipeline import run_pipeline

        if args.source and args.source_id:
            raise ValueError("--source-id applies only to --video")
        try:
            outputs = run_pipeline(
                catalog,
                source_id=args.source or args.source_id,
                video=args.video,
                output_dir=args.output_dir,
                calibration=args.calibration,
                model_id=args.model,
                checkpoint=args.checkpoint,
                model_manifest=args.model_manifest,
                cache_dir=args.cache_dir,
                device=args.device,
                batch_size=args.batch_size,
                overlay=not args.no_overlay,
                force=args.force,
                rebuild=args.rebuild,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"rally prediction failed: {exc}", file=sys.stderr)
            return 2
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0
    if args.command == "source" and args.source_command == "add":
        if args.rally_30_fps:
            from .source import resolve_rally_30fps_source

            source = resolve_rally_30fps_source(
                catalog, source_id=args.source_id, video=args.video, output_dir=args.artifact_root
            )
            print(f"{source.source_id}\t{source.artifact_root}")
            return 0
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
                state = (
                    "experiment-ready"
                    if row["experiment_ready"]
                    else "annotation-ready"
                    if row["annotation_ready"]
                    else "blocked"
                )
                print(f"{row['source_id']}: {state}")
                for field in (
                    "video",
                    "identity",
                    "candidates",
                    "frozen_candidates",
                    "person_tracks",
                    "assignments",
                    "pose_cache",
                    "calibration",
                ):
                    print(
                        f"  {field.replace('_', ' '):18} {'ready' if row[field] else 'missing'}"
                    )
                if row.get("run"):
                    for name, stage in row["run"]["stages"].items():
                        print(
                            f"  stage {name:12} {stage['state']} {stage.get('reason', '')}"
                        )
                        if name == "pose" and stage.get("committed_work"):
                            progress = stage["committed_work"]
                            unit = progress.get("last_committed_unit")
                            last = (
                                f"; latest frame {unit['frame']}, track {unit['track_id']}"
                                if unit
                                else ""
                            )
                            print(
                                f"    {progress['retained_unit_count']} retained pose units{last}"
                            )
                for problem in row["problems"]:
                    print(f"  problem: {problem}")
        return 0
    if args.command == "augment":
        from .augmentation import (
            AUGMENTATION_TRANSFORMS,
            augmentation_provenance,
            create_augmented_source,
        )

        if args.augment_command == "list":
            rows = [
                {
                    "name": spec.name,
                    "category": spec.category,
                    "calibration": spec.calibration,
                    "codec": spec.codec,
                    "preset": spec.preset,
                    "crf": spec.crf,
                    "video_filter": spec.video_filter,
                    "description": spec.description,
                }
                for spec in AUGMENTATION_TRANSFORMS.values()
            ]
            if args.json:
                print(json.dumps(rows, indent=2))
            else:
                for row in rows:
                    print(
                        f"{row['name']}: {row['category']}, {row['codec']} "
                        f"crf={row['crf']}, calibration={row['calibration']}"
                    )
            return 0
        try:
            entry = create_augmented_source(
                catalog,
                parent_source_id=args.parent_source_id,
                parent_video=args.parent_video,
                parent_calibration=args.parent_calibration,
                transform=args.transform,
                output_root=args.output_root,
                source_id=args.source_id,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"augmentation failed: {exc}", file=sys.stderr)
            return 2
        provenance = augmentation_provenance(entry)
        print(f"registered {entry.source_id}")
        print(f"transform: {provenance['transform']}")
        print(f"artifact root: {entry.artifact_root}")
        print(f"catalog: {catalog.path}")
        return 0
    if args.command == "stage":
        source = catalog.select([args.source])[0]
        if args.stage_command == "import-calibration":
            from .artifacts import import_calibration

            print(import_calibration(source, args.external_path, force=args.force))
            return 0
        if args.stage_command == "import-artifact":
            from .artifacts import import_artifact

            print(
                import_artifact(source, args.kind, args.external_path, force=args.force)
            )
            return 0
        if args.stage_command == "calibrate":
            from src.calibrate_court import main as calibrate_main
            from .artifacts import register_native_calibration

            forwarded = [
                str(source.video_path),
                str(source.calibration_path),
                "--frame",
                str(args.frame),
                "--mode",
                args.mode,
                "--ui",
                args.ui,
            ]
            if args.open_browser:
                forwarded.append("--open-browser")
            if args.preview:
                forwarded.extend(("--preview", str(args.preview)))
            selected_frame = calibrate_main(forwarded)
            print(register_native_calibration(source, source_frame=selected_frame))
            return 0
        if args.stage_command == "person-tracks":
            from .stages import run_person_stage
            from .rally_pipeline import select_device

            run_person_stage(
                source,
                model=args.model,
                device=select_device(args.device),
                force=args.force,
                tracking_profile=args.tracking_profile,
            )
            print(source.person_tracks_path)
            return 0
        if args.stage_command == "players":
            from .stages import run_player_stage

            run_player_stage(
                source,
                pose_model=args.pose_model,
                pose_crop_padding=args.pose_crop_padding,
                pose_backend=args.pose_backend,
                pose_device=args.pose_device,
                force=args.force,
                tracking_profile=args.tracking_profile,
            )
            print(source.assignments_path)
            return 0
        status = catalog.status(source)
        artifact_state = {
            "shuttle-candidates": status.frozen_candidates or status.candidates,
            "person-tracks": source.person_tracks_path.is_file(),
            "calibration": status.calibration
            and not any("calibration" in problem for problem in status.problems),
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
        catalog.write_annotation_config(
            generated, source_ids=[s.source_id for s in selected], stage=args.stage
        )
        forwarded = [
            "--config",
            str(generated),
            "--runtime",
            str(runtime),
            "serve",
            "--annotator",
            args.annotator,
            "--queue",
            args.queue,
            "--host",
            args.host,
            "--port",
            str(args.port),
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
                + "; ".join(
                    status.problems or ("required player artifacts are missing",)
                )
            )
        if args.rally_command == "pseudo-label":
            from src.temporal_selector.rally_dataset import RallySourceConfig
            from src.temporal_selector.rally_pseudo_label import pseudo_label_source

            predictions, proposals = pseudo_label_source(
                RallySourceConfig(
                    source.source_id,
                    source.video_path,
                    source.assignments_path,
                    source.pose_cache_path,
                    source.calibration_path,
                ),
                args.teacher,
                device=args.device,
                batch_size=args.batch_size,
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
            if any(
                span.source_id == source.source_id
                for span in existing.reviewed_coverage
            ):
                raise ValueError(
                    f"source already has reviewed rally labels: {source.source_id}"
                )
        if not source.rally_proposals_path.is_file():
            raise FileNotFoundError(
                f"rally proposals are missing: {source.rally_proposals_path}"
            )
        if not source.rally_predictions_path.is_file():
            raise FileNotFoundError(
                f"rally predictions are missing: {source.rally_predictions_path}"
            )
        # Proposal loading verifies video identity; verify all remaining inference inputs here.
        proposal = json.loads(source.rally_proposals_path.read_text(encoding="utf-8"))
        prediction_lines = [
            json.loads(line)
            for line in source.rally_predictions_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if not prediction_lines or prediction_lines[0] != proposal.get("metadata"):
            raise ValueError(
                "rally predictions and proposals have incompatible metadata"
            )
        frame_count = int(prediction_lines[0].get("frame_count", -1))
        if [int(row.get("frame", -1)) for row in prediction_lines[1:]] != list(
            range(frame_count)
        ):
            raise ValueError(
                "rally predictions do not cover every source frame exactly once"
            )
        from .identity import sha256_file

        expected = proposal.get("metadata", {}).get("input_artifact_sha256", {})
        current = {
            "calibration": sha256_file(source.calibration_path),
            "assignments": sha256_file(source.assignments_path),
            "poses": sha256_file(source.pose_cache_path),
        }
        if any(expected.get(name) != value for name, value in current.items()):
            raise ValueError("rally proposals are stale for current source artifacts")
        label_video(
            source.video_path,
            source.source_id,
            intervals_path,
            manifest_path,
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
