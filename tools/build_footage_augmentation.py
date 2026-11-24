"""Build isolated footage-augmentation arms for rally robustness experiments.

Run this on the GPU host from a feature-worktree snapshot.  It reads a base
clean-experiment config, creates one isolated source per parent recording per
transform, reruns person tracking and RTMPose player extraction on the
augmented footage, and writes an arm config plus extended rally annotations.

The parent footage, calibrations, and observations are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.temporal_selector.rally_intervals import (
    RallyInterval,
    RallySourceManifest,
    read_rally_intervals,
    write_rally_intervals,
)
from src.workflow.augmentation import (
    AUGMENTATION_TRANSFORMS,
    augmentation_provenance,
    create_augmented_source,
)
from src.workflow.identity import SourceIdentity
from src.workflow.registry import SourceCatalog, SourceEntry
from src.workflow.stages import (
    run_person_stage,
    run_player_stage,
)


def _resolve(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _parent_sources(base_config: Mapping[str, Any], base_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for source in base_config["dataset"]["sources"]:
        row = {
            "source_id": str(source["source_id"]),
            "video_path": str(_resolve(base_dir, source["video_path"])),
            "calibration_path": str(_resolve(base_dir, source["calibration_path"])),
            "assignments_path": str(_resolve(base_dir, source["assignments_path"])),
            "pose_cache_path": str(_resolve(base_dir, source["pose_cache_path"])),
        }
        if source.get("person_tracks_path"):
            row["person_tracks_path"] = str(
                _resolve(base_dir, source["person_tracks_path"])
            )
        for key in ("video_path", "calibration_path", "assignments_path", "pose_cache_path"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(f"{row['source_id']}: missing {key}: {row[key]}")
        rows.append(row)
    return rows


def _annotation_coverage(base_manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = base_manifest.get("reviewed_coverage")
    if not isinstance(raw, dict):
        raise ValueError("base rally manifest has no reviewed_coverage declarations")
    return dict(raw)


def write_arm_annotations(
    *,
    transform: str,
    parent_ids: Sequence[str],
    variant_parents: Mapping[str, str],
    catalog: SourceCatalog,
    base_intervals_path: Path,
    base_manifest_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    base_index = read_rally_intervals(base_intervals_path, base_manifest_path)
    base_manifest = _read(base_manifest_path)
    coverage = _annotation_coverage(base_manifest)
    partial = set(base_manifest.get("partial_start_rally_ids", ()))
    intervals = list(base_index.intervals)
    sources = [
        RallySourceManifest(
            source_id=source_id,
            video_sha256=base_index.sources[source_id].video_sha256,
            fps=base_index.sources[source_id].fps,
            frame_count=base_index.sources[source_id].frame_count,
        )
        for source_id in parent_ids
    ]
    for variant_id, parent_id in variant_parents.items():
        entry = catalog.select([variant_id])[0]
        identity = SourceIdentity.read(entry.identity_path)
        sources.append(
            RallySourceManifest(
                source_id=variant_id,
                video_sha256=identity.video_sha256,
                fps=identity.fps_numerator / identity.fps_denominator,
                frame_count=identity.frame_count,
            )
        )
        coverage[variant_id] = "full_source"
        for interval in base_index.for_source(parent_id):
            suffix = interval.rally_id[len(parent_id):]
            new_id = f"{variant_id}{suffix}"
            intervals.append(
                RallyInterval(
                    source_id=variant_id,
                    rally_id=new_id,
                    start_frame=interval.start_frame,
                    end_frame=interval.end_frame,
                )
            )
            if interval.rally_id in partial:
                partial.add(new_id)
    intervals_path = output_dir / "annotations" / "rallies.csv"
    manifest_path = output_dir / "annotations" / "rallies.manifest.json"
    write_rally_intervals(
        intervals_path,
        manifest_path,
        intervals,
        sources,
        annotation_revision=(
            f"{base_manifest.get('annotation_revision', 'unknown')}"
            f"+footage-augmentation:{transform}"
        ),
        reviewed_coverage=coverage,
        partial_start_rally_ids=partial,
    )
    return intervals_path, manifest_path


def write_arm_config(
    *,
    transform: str,
    base_config: Mapping[str, Any],
    base_dir: Path,
    originals: Sequence[Mapping[str, Any]],
    variants: Mapping[str, SourceEntry],
    variant_parents: Mapping[str, str],
    intervals_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> Path:
    sources = []
    for row in originals:
        sources.append(
            {
                "source_id": row["source_id"],
                "video_path": row["video_path"],
                "assignments_path": row["assignments_path"],
                "pose_cache_path": row["pose_cache_path"],
                "calibration_path": row["calibration_path"],
                **(
                    {"person_tracks_path": row["person_tracks_path"]}
                    if "person_tracks_path" in row
                    else {}
                ),
            }
        )
    for variant_id, entry in variants.items():
        sources.append(
            {
                "source_id": variant_id,
                "video_path": str(entry.video_path),
                "assignments_path": str(entry.assignments_path),
                "pose_cache_path": str(entry.pose_cache_path),
                "calibration_path": str(entry.calibration_path),
            }
        )
    config = {
        "schema": "clean_rally_experiment_config",
        "schema_version": 1,
        "seed": int(base_config.get("seed", 1729)),
        "camera_sources": {
            str(camera): [str(value) for value in values]
            for camera, values in base_config["camera_sources"].items()
        },
        "variant_parents": {
            str(variant): str(parent)
            for variant, parent in variant_parents.items()
        },
        "dataset": {
            "pose_visibility_threshold": float(
                base_config["dataset"].get("pose_visibility_threshold", 0.5)
            ),
            "rally_feature_view": "full_motion",
            "rally_context_seconds": 2.0,
            "rally_intervals_path": str(intervals_path),
            "rally_manifest_path": str(manifest_path),
            "sources": sources,
        },
    }
    path = output_dir / "experiment.json"
    _write(path, config)
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=["horizontal-flip"],
    )
    parser.add_argument("--person-model", default="models/yolo26l.engine")
    parser.add_argument(
        "--pose-model", default="models/rtmpose-l-halpe26-384x288-fp16.onnx"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Only render/register footage; do not rerun tracking and pose.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Limit extraction to these parent source IDs. Defaults to all.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    unknown = [name for name in args.transforms if name not in AUGMENTATION_TRANSFORMS]
    if unknown:
        raise ValueError(f"unknown transforms: {unknown}")
    base_config_path = args.base_config.expanduser().resolve()
    base_dir = base_config_path.parent
    base_config = _read(base_config_path)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    catalog_path = (
        args.catalog.expanduser().resolve()
        if args.catalog
        else output_root / "catalog.json"
    )
    catalog = (
        SourceCatalog.read(catalog_path) if catalog_path.is_file() else SourceCatalog(catalog_path)
    )
    originals = _parent_sources(base_config, base_dir)
    parent_ids = [row["source_id"] for row in originals]
    selected = set(args.source) or set(parent_ids)
    if not selected <= set(parent_ids):
        raise ValueError(f"unknown parent sources: {sorted(selected - set(parent_ids))}")

    base_intervals = _resolve(base_dir, base_config["dataset"]["rally_intervals_path"])
    base_manifest = _resolve(base_dir, base_config["dataset"]["rally_manifest_path"])
    if not base_intervals.is_file() or not base_manifest.is_file():
        raise FileNotFoundError("base rally annotations are missing")

    person_model = str(Path(args.person_model).expanduser())
    pose_model = Path(args.pose_model).expanduser()
    for transform in args.transforms:
        arm_root = output_root / transform
        arm_root.mkdir(parents=True, exist_ok=True)
        variants: dict[str, SourceEntry] = {}
        for row in originals:
            if row["source_id"] not in selected:
                continue
            print(f"[{transform}] {row['source_id']}: render", flush=True)
            entry = create_augmented_source(
                catalog,
                parent_source_id=row["source_id"],
                parent_video=Path(row["video_path"]),
                parent_calibration=Path(row["calibration_path"]),
                transform=transform,
                output_root=arm_root,
            )
            variants[entry.source_id] = entry
            if args.skip_extraction:
                continue
            print(f"[{transform}] {entry.source_id}: person tracks", flush=True)
            run_person_stage(
                entry,
                model=person_model,
                device=args.device,
            )
            print(f"[{transform}] {entry.source_id}: RTMPose players", flush=True)
            run_player_stage(
                entry,
                pose_model=pose_model,
                pose_backend="rtmpose",
                pose_device="cuda" if args.device == "cuda" else None,
            )
            provenance = augmentation_provenance(entry)
            print(
                f"[{transform}] {entry.source_id}: ready "
                f"({provenance['frame_count']} frames)",
                flush=True,
            )
        # Variants from earlier resumed runs still belong to this arm.
        suffix = f"__{AUGMENTATION_TRANSFORMS[transform].slug}"
        all_variants: dict[str, SourceEntry] = {}
        for source_id in catalog.sources:
            if not source_id.endswith(suffix):
                continue
            parent_id = source_id[: -len(suffix)]
            if parent_id in parent_ids:
                all_variants[source_id] = catalog.select([source_id])[0]
        variant_parents = {
            source_id: source_id[: -len(suffix)]
            for source_id in all_variants
        }
        intervals_path, manifest_path = write_arm_annotations(
            transform=transform,
            parent_ids=parent_ids,
            variant_parents=variant_parents,
            catalog=catalog,
            base_intervals_path=base_intervals,
            base_manifest_path=base_manifest,
            output_dir=arm_root,
        )
        config_path = write_arm_config(
            transform=transform,
            base_config=base_config,
            base_dir=base_dir,
            originals=originals,
            variants=all_variants,
            variant_parents=variant_parents,
            intervals_path=intervals_path,
            manifest_path=manifest_path,
            output_dir=arm_root,
        )
        design = {
            "schema": "footage_augmentation_build",
            "schema_version": 1,
            "transform": transform,
            "category": AUGMENTATION_TRANSFORMS[transform].category,
            "spec": asdict(AUGMENTATION_TRANSFORMS[transform]),
            "parents": sorted(variant_parents.values()),
            "variants": variant_parents,
            "base_config": str(base_config_path),
            "base_intervals_sha256": _sha256(base_intervals),
            "built_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write(arm_root / "build.json", design)
        print(f"[{transform}] arm config: {config_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
