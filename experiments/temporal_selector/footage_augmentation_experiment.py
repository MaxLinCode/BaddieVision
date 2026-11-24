"""Paired baseline-versus-footage-augmentation rally experiment.

For each held-out camera, original training recordings are split chronologically
into common train/validation windows exactly once.  The baseline arm trains on
the common windows.  The augmented arm trains on the same common windows plus
their transformed variants, selected with the same source/category draw stream
and the same number of optimizer steps.  Validation, decoder calibration, and
held-out evaluation always use original footage.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.temporal_selector import rally_experiment as training
from src.temporal_selector.rally import (
    RallySegmenter,
    RallySegmenterConfig,
    RallyTrainingRecipe,
)
from src.temporal_selector.rally_dataset import (
    RALLY_TRANSFORMER_CONTEXT_RADIUS_SECONDS,
    RallyDataConfig,
    RallyWindow,
    RallyWindowDataset,
    rally_data_config_from_selector_mapping,
)
from src.temporal_selector.rally_evaluation import (
    chronological_camera_train_validation_split,
    decode_prediction_rows,
    metrics_by_source,
    rally_metrics,
    tune_rally_decoder,
)
from src.temporal_selector.rally_features import rally_feature_view
from src.temporal_selector.training import seed_everything


ARMS = ("baseline", "augmented")
CATEGORY_WEIGHTS = {"ordinary": 1.0, "boundary": 3.0, "short_rally": 5.0}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(_json_safe(dict(row)), sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _camera_sources(value: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    return {
        str(camera): tuple(map(str, sources)) for camera, sources in value.items()
    }


def _fps_by_source(windows: Sequence[RallyWindow]) -> dict[str, float]:
    result: dict[str, float] = {}
    for window in windows:
        source = window.source_id
        fps = float(window.metadata["fps"])
        if source in result and result[source] != fps:
            raise ValueError(f"inconsistent FPS for {source}")
        result[source] = fps
    return result


def source_train_stop(
    split: Mapping[str, Any], training_camera_sources: Mapping[str, Sequence[str]]
) -> dict[str, int]:
    """Return the exclusive source-local train frame bound from the split."""
    stops: dict[str, int] = {}
    cameras = split["cameras"]
    for camera in training_camera_sources:
        metadata = cameras[str(camera)]
        if "source_offsets" not in metadata:
            raise ValueError(
                "footage augmentation requires uniform-fps chronological splits"
            )
        offsets = metadata["source_offsets"]
        counts = metadata["source_frame_counts"]
        train_end = int(metadata["train_global_range"][1])
        for source in metadata["source_order"]:
            offset = int(offsets[source])
            stops[str(source)] = max(
                0, min(int(counts[source]), train_end - offset + 1)
            )
    return stops


def variant_train_windows(
    windows: Sequence[RallyWindow],
    *,
    variant_parents: Mapping[str, str],
    train_sources: set[str],
    stops: Mapping[str, int],
) -> list[RallyWindow]:
    """Keep only variant windows wholly inside their parent's train range."""
    selected = []
    for window in windows:
        parent = variant_parents.get(window.source_id)
        if parent is None or parent not in train_sources:
            continue
        stop = stops.get(parent, 0)
        if stop <= 0:
            continue
        first = min(window.frame_indices)
        last = max(window.frame_indices)
        if first >= 0 and last < stop:
            selected.append(window)
    return selected


class FootageAugmentationSampler(training._RallyBatchSampler):
    """Equal-step source-balanced draws shared across the paired arms.

    A draw first picks one original source.  In the augmented arm it is replaced
    by that source's variant with probability ``variant_share``.  The baseline
    arm never replaces a draw, so the two arms share the original-source draw
    stream while keeping optimizer steps fixed.
    """

    def __init__(
        self,
        windows: Sequence[RallyWindow],
        *,
        steps: int,
        batch_size: int,
        seed: int,
        original_windows: Sequence[RallyWindow],
        variant_parents: Mapping[str, str],
        variant_share: float,
        include_variants: bool,
        boundary_balanced: bool = True,
    ) -> None:
        super().__init__(
            windows,
            batch_size=batch_size,
            seed=seed,
            boundary_balanced=boundary_balanced,
        )
        if steps <= 0:
            raise ValueError("steps must be positive")
        if not 0.0 <= variant_share <= 1.0:
            raise ValueError("variant share must be in [0, 1]")
        self.steps = steps
        self.variant_share = variant_share
        self.include_variants = include_variants
        self.variant_by_parent = {
            str(parent): str(variant)
            for variant, parent in variant_parents.items()
        }
        self._groups: dict[str, np.ndarray] = {}
        self._cdfs: dict[str, np.ndarray] = {}
        categories = [self._category(window) for window in self.windows]
        grouped: dict[str, list[int]] = {}
        for index, window in enumerate(self.windows):
            grouped.setdefault(window.source_id, []).append(index)
        for source, indices in grouped.items():
            weights = np.asarray(
                [CATEGORY_WEIGHTS[categories[index]] for index in indices],
                dtype=np.float64,
            )
            self._groups[source] = np.asarray(indices, dtype=np.int64)
            self._cdfs[source] = np.cumsum(weights) / weights.sum()
        self.original_sources = tuple(
            sorted({window.source_id for window in original_windows})
        )
        if not self.original_sources:
            raise ValueError("augmentation sampler needs original training sources")
        self.available_variant_by_parent = {
            parent: variant
            for variant, parent in variant_parents.items()
            if variant in self._groups
        }
        if include_variants and not self.available_variant_by_parent:
            raise ValueError("augmented arm has no sampled variants")

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1
        draws = self.steps * self.batch_size
        gates, source_fractions, category_fractions = rng.random((3, draws))
        source_draws = np.asarray(self.original_sources, dtype=object)[
            np.floor(source_fractions * len(self.original_sources)).astype(int)
        ]
        if self.include_variants:
            for parent in self.original_sources:
                variant = self.available_variant_by_parent.get(parent)
                if variant is None:
                    continue
                mask = (source_draws == parent) & (gates < self.variant_share)
                source_draws[mask] = variant
        indices = np.empty(draws, dtype=np.int64)
        for source, group in self._groups.items():
            mask = source_draws == source
            if not bool(mask.any()):
                continue
            positions = np.searchsorted(
                self._cdfs[source],
                category_fractions[mask],
                side="right",
            )
            positions = np.minimum(positions, len(group) - 1)
            indices[mask] = group[positions]
        source_counts = Counter(map(str, source_draws.tolist()))
        categories = [self._category(self.windows[index]) for index in indices]
        self.last_sampling_stats = {
            "optimizer_steps": self.steps,
            "requested_window_count": draws,
            "realized_source_counts": dict(sorted(source_counts.items())),
            "realized_category_counts": dict(Counter(categories)),
        }
        for start in range(0, draws, self.batch_size):
            yield indices[start : start + self.batch_size].tolist()


def _compact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_bce": metrics["raw_bce"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "frame_f1": metrics["f1"],
        "decoded_frame_f1": metrics["decoded_frame"]["f1"],
        "intervals": metrics["intervals"],
    }


def _metric_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "interval_f1": lambda value: value["intervals"]["f1"],
        "frame_f1": lambda value: value["frame_f1"],
        "roc_auc": lambda value: value["roc_auc"],
        "pr_auc": lambda value: value["pr_auc"],
        "raw_bce": lambda value: value["raw_bce"],
        "boundary_mae": lambda value: value["intervals"][
            "mean_absolute_boundary_error"
        ],
        "signed_start_error": lambda value: value["intervals"].get(
            "mean_start_boundary_error"
        ),
        "signed_end_error": lambda value: value["intervals"].get(
            "mean_end_boundary_error"
        ),
        "false_splits": lambda value: value["intervals"]["false_split_count"],
        "false_merges": lambda value: value["intervals"]["false_merge_count"],
    }
    result: dict[str, float | None] = {}
    for name, extract in fields.items():
        values = []
        for record in records:
            value = extract(record)
            if value is None:
                continue
            value = float(value)
            if math.isfinite(value):
                values.append(value)
        result[name] = statistics.fmean(values) if values else None
    return result


def _delta(augmented: float | None, baseline: float | None) -> float | None:
    if augmented is None or baseline is None:
        return None
    return augmented - baseline


def _summary(
    fits: Sequence[Mapping[str, Any]], *, seeds: Sequence[int], cameras: Sequence[str]
) -> dict[str, Any]:
    lookup = {
        (str(fit["camera"]), int(fit["seed"]), str(fit["arm"])): fit for fit in fits
    }
    by_camera: dict[str, Any] = {}
    for camera in cameras:
        arm_metrics = {
            arm: _metric_summary(
                [lookup[camera, seed, arm]["test"] for seed in seeds]
            )
            for arm in ARMS
        }
        by_camera[camera] = {
            "arms": arm_metrics,
            "delta_augmented_minus_baseline": {
                name: _delta(arm_metrics["augmented"][name], arm_metrics["baseline"][name])
                for name in arm_metrics["baseline"]
            },
        }
    macro_arms = {
        arm: _metric_summary(
            [lookup[camera, seed, arm]["test"] for camera in cameras for seed in seeds]
        )
        for arm in ARMS
    }
    by_seed: dict[str, Any] = {}
    for seed in seeds:
        metrics = {
            arm: _metric_summary(
                [lookup[camera, seed, arm]["test"] for camera in cameras]
            )
            for arm in ARMS
        }
        by_seed[str(seed)] = {
            "arms": metrics,
            "delta_augmented_minus_baseline": {
                name: _delta(metrics["augmented"][name], metrics["baseline"][name])
                for name in metrics["baseline"]
            },
        }
    return {
        "macro": {
            "arms": macro_arms,
            "delta_augmented_minus_baseline": {
                name: _delta(macro_arms["augmented"][name], macro_arms["baseline"][name])
                for name in macro_arms["baseline"]
            },
        },
        "by_seed": by_seed,
        "by_camera": by_camera,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--boundary-aux-weight", type=float, default=0.25)
    parser.add_argument("--temporal-smoothing-weight", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--variant-share", type=float, default=0.5)
    parser.add_argument("--guard-seconds", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument(
        "--feature-view", default="full_motion", help="Rally feature view for both arms"
    )
    parser.add_argument(
        "--context-seconds",
        type=float,
        default=RALLY_TRANSFORMER_CONTEXT_RADIUS_SECONDS,
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument(
        "--held-out-camera",
        action="append",
        default=[],
        help="Fixed-test camera; defaults to every configured camera",
    )
    parser.add_argument(
        "--max-folds", type=int, help="Pilot limit on held-out cameras"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Build the dataset, print its manifest summary, and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = _read(config_path)
    variant_parents = {
        str(variant): str(parent)
        for variant, parent in raw.get("variant_parents", {}).items()
    }
    data_config: RallyDataConfig = rally_data_config_from_selector_mapping(
        raw,
        base_dir=config_path.parent,
        default_context_seconds=args.context_seconds,
    )
    data_config = replace(
        data_config,
        feature_view=args.feature_view,
        context_seconds=args.context_seconds,
    )
    dataset = RallyWindowDataset(data_config)
    if args.check_only:
        summary = {
            "feature_view": dataset.manifest["feature_view"],
            "feature_dimension": len(dataset.manifest["feature_names"]),
            "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
            "annotation_fingerprint": dataset.manifest["annotation_fingerprint"],
            "window_count": dataset.manifest["window_count"],
            "source_window_counts": dataset.manifest["source_window_counts"],
            "owned_target_counts": dataset.manifest["owned_target_counts"],
            "pose_training_metadata": dataset.manifest.get("pose_training_metadata"),
        }
        _write(output / "dataset-check.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    if not variant_parents:
        raise ValueError("config has no variant_parents mapping")
    groups = _camera_sources(raw["camera_sources"])
    variant_ids = set(variant_parents)
    if variant_ids & {
        source for sources in groups.values() for source in sources
    }:
        raise ValueError("variant IDs must not appear in the original camera groups")
    unknown = variant_ids - {window.source_id for window in dataset.windows}
    if unknown:
        raise ValueError(f"configured variants have no windows: {sorted(unknown)}")
    missing_parents = set(variant_parents.values()) - {
        source for sources in groups.values() for source in sources
    }
    if missing_parents:
        raise ValueError(f"variants reference unknown parents: {sorted(missing_parents)}")
    windows_by_source: dict[str, list[RallyWindow]] = {}
    for window in dataset.windows:
        windows_by_source.setdefault(window.source_id, []).append(window)

    held_out_cameras = list(args.held_out_camera or groups)
    if args.max_folds is not None:
        held_out_cameras = held_out_cameras[: args.max_folds]
    unknown_cameras = set(held_out_cameras) - set(groups)
    if unknown_cameras:
        raise ValueError(f"unknown held-out cameras: {sorted(unknown_cameras)}")

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    feature = rally_feature_view(data_config.feature_view)
    model_config = RallySegmenterConfig(
        frame_feature_dim=feature.dimension,
        audio_attention=feature.uses_audio,
    )
    recipe = RallyTrainingRecipe(
        seed=args.seeds[0],
        epochs=args.epochs,
        batch_size=args.batch_size,
        sampling="boundary_balanced",
        boundary_aux_weight=args.boundary_aux_weight,
        learning_rate=args.learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
        temporal_smoothing_weight=args.temporal_smoothing_weight,
        device=args.device,
    )
    design = {
        "schema": "paired_rally_footage_augmentation",
        "schema_version": 1,
        "arm": args.arm,
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
        "annotation_fingerprint": dataset.annotation_fingerprint,
        "feature_view": feature.name,
        "feature_dimension": feature.dimension,
        "camera_sources": {camera: list(sources) for camera, sources in groups.items()},
        "variant_parents": variant_parents,
        "held_out_cameras": held_out_cameras,
        "seeds": args.seeds,
        "recipe": asdict(recipe),
        "guard_seconds": args.guard_seconds,
        "train_fraction": args.train_fraction,
        "variant_share": args.variant_share,
        "control": (
            "Common train/validation/test windows, class weights, source draw "
            "stream and optimizer steps are fixed across arms. Validation, "
            "decoder calibration and held-out evaluation use originals only."
        ),
    }
    _write(output / "design.json", design)
    design_fingerprint = hashlib.sha256(
        json.dumps(design, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    original_sampler = training._RallyBatchSampler
    original_positive_weight = training._positive_weight
    fits: list[dict[str, Any]] = []
    for camera in held_out_cameras:
        training_groups = {
            name: sources for name, sources in groups.items() if name != camera
        }
        train_sources = {
            source for sources in training_groups.values() for source in sources
        }
        original_windows = [
            window
            for source in train_sources
            for window in windows_by_source.get(source, [])
        ]
        common_train, validation, split = (
            chronological_camera_train_validation_split(
                original_windows,
                training_groups,
                train_fraction=args.train_fraction,
                guard_seconds=args.guard_seconds,
            )
        )
        stops = source_train_stop(split, training_groups)
        variants_train = variant_train_windows(
            dataset.windows,
            variant_parents=variant_parents,
            train_sources=train_sources,
            stops=stops,
        )
        test = [
            window
            for source in groups[camera]
            for window in windows_by_source.get(source, [])
        ]
        if not variants_train:
            raise ValueError(f"fold {camera} has no augmented training windows")
        steps_per_epoch = math.ceil(len(common_train) / args.batch_size)
        optimizer_steps = steps_per_epoch * args.epochs
        fold_dir = output / camera
        _write(
            fold_dir / "split.json",
            {
                "camera": camera,
                "split": split,
                "train_stop": stops,
                "common_train_window_ids": [w.window_id for w in common_train],
                "variant_train_window_ids": [w.window_id for w in variants_train],
                "validation_window_ids": [w.window_id for w in validation],
                "test_window_ids": [w.window_id for w in test],
                "window_counts": {
                    "common_train": len(common_train),
                    "variant_train": len(variants_train),
                    "validation": len(validation),
                    "test": len(test),
                },
                "optimizer_steps": optimizer_steps,
            },
        )
        validation_fps = _fps_by_source(validation)
        test_fps = _fps_by_source(test)
        print(
            f"{camera}: common_train={len(common_train)} "
            f"variant_train={len(variants_train)} validation={len(validation)} "
            f"test={len(test)} steps={optimizer_steps}",
            flush=True,
        )
        for seed in args.seeds:
            for arm in ARMS:
                arm_dir = fold_dir / str(seed) / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = arm_dir / "metrics.json"
                if metrics_path.is_file():
                    fit = _read(metrics_path)
                    if fit.get("design_fingerprint") != design_fingerprint:
                        raise ValueError(
                            f"stale metrics for a changed design: {metrics_path}"
                        )
                    fits.append(fit)
                    print(
                        f"REUSE {camera} seed={seed} arm={arm} "
                        f"interval_f1={fit['test']['intervals']['f1']:.6f}",
                        flush=True,
                    )
                    continue
                include_variants = arm == "augmented"
                fit_windows = [
                    *common_train,
                    *(variants_train if include_variants else ()),
                ]
                seed_everything(seed)
                model = RallySegmenter(model_config).to(device)

                class FoldSampler(FootageAugmentationSampler):
                    def __init__(self, windows, **kwargs):
                        del windows
                        super().__init__(
                            fit_windows,
                            steps=steps_per_epoch,
                            original_windows=common_train,
                            variant_parents=variant_parents,
                            variant_share=args.variant_share,
                            include_variants=include_variants,
                            **kwargs,
                        )

                def fixed_positive_weight(windows, field, **kwargs):
                    del windows
                    return original_positive_weight(common_train, field, **kwargs)

                training._RallyBatchSampler = FoldSampler
                training._positive_weight = fixed_positive_weight
                recovery = arm_dir / "training-recovery.pt"
                try:
                    history = training.train_rally_segmenter_fixed_epochs(
                        model,
                        fit_windows,
                        device=device,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        seed=seed,
                        boundary_aux_weight=args.boundary_aux_weight,
                        temporal_smoothing_weight=args.temporal_smoothing_weight,
                        learning_rate=args.learning_rate,
                        gradient_clip_norm=args.gradient_clip_norm,
                        sampling="boundary_balanced",
                        recovery_path=recovery,
                        resume=recovery.is_file(),
                        recovery_identity={
                            "camera": camera,
                            "arm": arm,
                            "seed": seed,
                            "design": design,
                        },
                    )
                finally:
                    training._RallyBatchSampler = original_sampler
                    training._positive_weight = original_positive_weight
                realized_steps = sum(
                    int(epoch["sampling"]["optimizer_steps"]) for epoch in history
                )
                if realized_steps != optimizer_steps:
                    raise RuntimeError(
                        f"expected {optimizer_steps} optimizer steps, "
                        f"observed {realized_steps}"
                    )
                source_counts: Counter[str] = Counter()
                for epoch in history:
                    source_counts.update(epoch["sampling"]["realized_source_counts"])
                if not include_variants and any(
                    source_counts[variant]
                    for variant in variant_parents
                ):
                    raise RuntimeError("baseline unexpectedly sampled a variant")
                validation_rows = training.predict_rally_windows(
                    model,
                    validation,
                    device=device,
                    batch_size=args.batch_size,
                    partition="validation",
                    include_unlabeled=True,
                )
                decoder, decoder_search = tune_rally_decoder(
                    validation_rows,
                    fps_by_source=validation_fps,
                    hysteresis=True,
                )
                raw_test = training.predict_rally_windows(
                    model,
                    test,
                    device=device,
                    batch_size=args.batch_size,
                    partition="test",
                    include_unlabeled=True,
                )
                decoded = decode_prediction_rows(
                    raw_test,
                    fps_by_source=test_fps,
                    config=decoder,
                )
                test_metrics = rally_metrics(decoded)
                variant_count = sum(
                    source_counts[variant] for variant in variant_parents
                )
                fit = {
                    "camera": camera,
                    "seed": seed,
                    "arm": arm,
                    "design_fingerprint": design_fingerprint,
                    "optimizer_steps": realized_steps,
                    "source_counts": dict(sorted(source_counts.items())),
                    "variant_share": (
                        variant_count / sum(source_counts.values())
                        if source_counts
                        else 0.0
                    ),
                    "decoder": asdict(decoder),
                    "validation_decoder": decoder_search[0],
                    "test": _compact(test_metrics),
                    "test_by_source": {
                        source: _compact(metrics)
                        for source, metrics in metrics_by_source(decoded).items()
                    },
                }
                torch.save(
                    {
                        "model_type": "RallySegmenter",
                        "rally_segmenter_config": asdict(model_config),
                        "feature_view": feature.name,
                        "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
                        "arm": arm,
                        "augmentation": args.arm,
                        "seed": seed,
                        "camera": camera,
                        "chosen_epoch": args.epochs,
                        "decoder_configuration": asdict(decoder),
                        "model_state_dict": model.state_dict(),
                    },
                    arm_dir / "rally-segmenter.pt",
                )
                _write_jsonl(arm_dir / "test-predictions.jsonl", decoded)
                _write(metrics_path, fit)
                fits.append(fit)
                print(
                    f"DONE {camera} seed={seed} arm={arm} "
                    f"interval_f1={test_metrics['intervals']['f1']:.6f} "
                    f"variant_share={fit['variant_share']:.3f}",
                    flush=True,
                )
    summary = _summary(fits, seeds=args.seeds, cameras=held_out_cameras)
    _write(output / "summary.json", {"design": design, **summary})
    print(json.dumps(summary["macro"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
