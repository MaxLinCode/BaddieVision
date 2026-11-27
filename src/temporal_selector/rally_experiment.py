"""Matched camera-held-out experiment for independent rally segmentation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import random
import tempfile
import json
import math
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .config import SelectorConfig
from .crossfit import CrossFitFold
from .dataset import (
    FRAME_DIMS,
    SelectorWindow,
    SelectorWindowDataset,
)
from .training import (
    predict_windows as _predict_windows,
    seed_everything as _seed_everything,
    train_selector_fold as _train_fold,
    aggregate_joint_predictions,
    load_dataset_config,
    prepare_output_directory,
)
from .model import JointRallyShuttleModel
from .rally import (
    RallySegmenter,
    RallySegmenterConfig,
    RallyTrainingRecipe,
    rally_checkpoint,
)
from .rally_dataset import (
    RALLY_TRANSFORMER_CONTEXT_RADIUS_SECONDS,
    RallyDataConfig,
    RallyWindow,
    RallyWindowDataset,
    collate_rally_windows,
    rally_data_config_from_selector_mapping,
)
from .rally_evaluation import (
    DEFAULT_CAMERA_SOURCES,
    CameraHeldOutFold,
    camera_held_out_folds,
    chronological_camera_train_validation_split,
    decode_prediction_rows,
    independent_model_acceptance,
    metrics_by_source,
    paired_bce_bootstrap,
    rally_metrics,
    tune_rally_decoder,
)
from .rally_features import (
    CANONICAL_RALLY_FEATURE_VIEW,
    RALLY_FEATURE_VIEWS,
    rally_feature_view,
)
from .rally_intervals import read_rally_intervals
from .rally_partitions import plan_rally_camera_splits, prepare_rally_partition


class _RallyBatchSampler(Sampler[list[int]]):
    """Frozen deterministic boundary-balanced sampling with length bucketing."""

    def __init__(
        self,
        windows: Sequence[RallyWindow],
        *,
        batch_size: int,
        seed: int,
        boundary_balanced: bool = True,
        bucket_multiplier: int = 8,
    ) -> None:
        if not windows or batch_size <= 0:
            raise ValueError("rally sampler needs windows and a positive batch size")
        self.windows = windows
        self.batch_size = batch_size
        self.seed = seed
        self.boundary_balanced = boundary_balanced
        self.bucket_size = max(batch_size, batch_size * bucket_multiplier)
        self.epoch = 0
        self.last_sampling_stats: dict[str, Any] | None = None

    @staticmethod
    def _category(window: RallyWindow) -> str:
        if window.metadata.get("owns_short_rally"):
            return "short_rally"
        if window.metadata.get("owns_rally_boundary"):
            return "boundary"
        return "ordinary"

    def __len__(self) -> int:
        return math.ceil(len(self.windows) / self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        categories = [self._category(window) for window in self.windows]
        if self.boundary_balanced:
            weights = torch.tensor(
                [
                    {"ordinary": 1.0, "boundary": 3.0, "short_rally": 5.0}[
                        category
                    ]
                    for category in categories
                ],
                dtype=torch.float64,
            )
            indices = torch.multinomial(
                weights,
                len(self.windows),
                replacement=True,
                generator=generator,
            ).tolist()
        else:
            indices = torch.randperm(
                len(self.windows), generator=generator
            ).tolist()
        self.last_sampling_stats = {
            "requested_window_count": len(self.windows),
            "realized_category_counts": {
                category: sum(
                    categories[index] == category for index in indices
                )
                for category in ("ordinary", "boundary", "short_rally")
            },
            "unique_sampled_window_count": len(set(indices)),
        }
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(
                key=lambda index: len(self.windows[index].frame_indices)
            )
            batches.extend(
                bucket[offset : offset + self.batch_size]
                for offset in range(0, len(bucket), self.batch_size)
            )
        for index in torch.randperm(
            len(batches), generator=generator
        ).tolist():
            yield batches[index]


def _owned_target_values(
    windows: Sequence[RallyWindow], field: str
) -> list[float]:
    values: list[float] = []
    for window in windows:
        target = getattr(window, field)
        values.extend(
            float(value)
            for frame, value in zip(window.frame_indices, target)
            if frame in window.owned_frames and float(value) != -100.0
        )
    return values


def _positive_weight(
    windows: Sequence[RallyWindow],
    field: str,
    *,
    device: torch.device,
    cap: float | None = None,
) -> torch.Tensor | None:
    values = _owned_target_values(windows, field)
    positive_mass = sum(values)
    if not values or positive_mass <= 0:
        return None
    weight = (len(values) - positive_mass) / positive_mass
    if cap is not None:
        weight = min(cap, weight)
    return torch.tensor(weight, dtype=torch.float32, device=device)


def predict_rally_windows(
    model: RallySegmenter,
    windows: Sequence[RallyWindow],
    *,
    device: torch.device,
    batch_size: int,
    partition: str,
    include_unlabeled: bool = False,
) -> list[dict[str, Any]]:
    """Predict exactly the owned frames from each candidate-free window."""
    records: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(windows), batch_size):
        window_batch = windows[start : start + batch_size]
        batch = collate_rally_windows(window_batch).to(device)
        with torch.no_grad():
            output = model(batch)
        inplay = output.inplay_logits.cpu()
        start_logits = output.rally_start_logits.cpu()
        end_logits = output.rally_end_logits.cpu()
        for batch_index, window in enumerate(window_batch):
            for local_frame, frame in enumerate(window.frame_indices):
                if frame not in window.owned_frames:
                    continue
                target = int(window.inplay_targets[local_frame])
                if target not in (0, 1) and not include_unlabeled:
                    continue
                logit = float(inplay[batch_index, local_frame])
                partial = next((
                    (has_start, has_end)
                    for first, last, has_start, has_end in window.metadata.get("partial_rally_spans", ())
                    if first <= frame <= last
                ), None)
                records.append(
                    {
                        "source_id": window.source_id,
                        "frame": int(frame),
                        "partition": partition,
                        "window_id": window.window_id,
                        "inplay_target": target,
                        **({"partial_rally": partial} if partial is not None else {}),
                        "inplay_logit": logit,
                        "inplay_probability": float(
                            torch.sigmoid(torch.tensor(logit))
                        ),
                        "rally_start_target": float(
                            window.rally_start_targets[local_frame]
                        ),
                        "rally_start_logit": float(
                            start_logits[batch_index, local_frame]
                        ),
                        "rally_end_target": float(
                            window.rally_end_targets[local_frame]
                        ),
                        "rally_end_logit": float(
                            end_logits[batch_index, local_frame]
                        ),
                    }
                )
    keys = [(row["source_id"], row["frame"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("rally prediction ownership is not unique")
    return sorted(records, key=lambda row: (row["source_id"], row["frame"]))


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _rng_state(device: torch.device) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _window_identity(windows: Sequence[RallyWindow]) -> str:
    """Fingerprint ordered observations, labels and ownership, including synthetic inputs."""
    digest = hashlib.sha256()
    for window in windows:
        digest.update(
            json.dumps(
                [
                    window.source_id,
                    window.window_id,
                    window.frame_indices,
                    sorted(window.owned_frames),
                    _RallyBatchSampler._category(window),
                ],
                separators=(",", ":"),
            ).encode()
        )
        for name in (
            "relative_time_seconds",
            "frame_values",
            "frame_validity",
            "inplay_targets",
            "rally_start_targets",
            "rally_end_targets",
        ):
            value = getattr(window, name).detach().cpu().contiguous()
            digest.update(str((name, tuple(value.shape), value.dtype)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def train_rally_segmenter(
    model: RallySegmenter,
    train_windows: Sequence[RallyWindow],
    validation_windows: Sequence[RallyWindow] | None,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    boundary_aux_weight: float = 0.25,
    temporal_smoothing_weight: float = 0.0,
    inplay_dice_weight: float = 0.0,
    learning_rate: float = 3e-4,
    gradient_clip_norm: float = 1.0,
    sampling: str = "boundary_balanced",
    recovery_path: Path | None = None,
    resume: bool = False,
    recovery_identity: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Commit each epoch durably; validation chooses the best inference state.

    Recovery checkpoints contain the *latest* optimizer/model state and the
    independently retained best state. They are never inference exports.
    With no validation partition this fits production data for exactly epochs.
    """
    recipe = RallyTrainingRecipe(
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        boundary_aux_weight=boundary_aux_weight,
        temporal_smoothing_weight=temporal_smoothing_weight,
        inplay_dice_weight=inplay_dice_weight,
        learning_rate=learning_rate,
        gradient_clip_norm=gradient_clip_norm,
        sampling=sampling,
        device=device.type,
    )
    if not train_windows or (validation_windows is not None and not validation_windows):
        raise ValueError(
            "rally training requires nonempty training and designated validation partitions"
        )
    if resume and recovery_path is None:
        raise ValueError("resume requires a recovery checkpoint path")
    if recovery_path is not None:
        recovery_path = Path(recovery_path)
        if recovery_path.exists() and not resume:
            raise FileExistsError(
                f"recovery checkpoint already exists; use resume: {recovery_path}"
            )
    identity = {
        "recipe": recipe.recovery_settings(),
        "model_config": asdict(model.config),
        "train_windows": _window_identity(train_windows),
        "validation_windows": _window_identity(validation_windows)
        if validation_windows is not None
        else None,
        "provenance": dict(recovery_identity or {}),
    }
    sampler = _RallyBatchSampler(
        train_windows,
        batch_size=batch_size,
        seed=seed,
        boundary_balanced=sampling == "boundary_balanced",
    )
    loader = DataLoader(
        list(train_windows), batch_sampler=sampler, collate_fn=collate_rally_windows
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    weights = {
        field: _positive_weight(
            train_windows,
            field,
            device=device,
            cap=10.0 if field != "inplay_targets" else None,
        )
        for field in ("inplay_targets", "rally_start_targets", "rally_end_targets")
    }
    history: list[dict[str, Any]] = []
    best: tuple[float, int, dict[str, torch.Tensor]] | None = None
    start_epoch = 0
    if resume and recovery_path is not None and recovery_path.exists():
        checkpoint = torch.load(recovery_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("schema") != "rally_training_recovery"
            or checkpoint.get("version") != 1
        ):
            raise ValueError("not a compatible rally recovery checkpoint")
        saved_identity = checkpoint["identity"]
        saved_identity["model_config"].setdefault("audio_attention", False)
        if saved_identity != identity:
            raise ValueError(
                "recovery configuration, data or split identity has changed"
            )
        start_epoch = int(checkpoint["completed_epoch"])
        history = checkpoint["history"]
        if not 0 < start_epoch <= epochs or len(history) != start_epoch:
            raise ValueError("recovery checkpoint has inconsistent epoch progress")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        sampler.epoch = int(checkpoint["sampler_epoch"])
        if sampler.epoch != start_epoch:
            raise ValueError("recovery checkpoint sampler epoch is inconsistent")
        best = checkpoint["best"]
        _restore_rng(checkpoint["rng_state"])
    for epoch in range(start_epoch, epochs):
        model.train()
        sums = [0.0, 0.0, 0.0, 0.0]
        batch_count = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.losses(
                batch,
                inplay_pos_weight=weights["inplay_targets"],
                boundary_start_pos_weight=weights["rally_start_targets"],
                boundary_end_pos_weight=weights["rally_end_targets"],
                boundary_aux_weight=boundary_aux_weight,
                temporal_smoothing_weight=temporal_smoothing_weight,
                inplay_dice_weight=inplay_dice_weight,
                return_counts=False,
            )
            if not bool(torch.isfinite(losses.total)):
                raise RuntimeError("rally training produced a non-finite loss")
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            for index, value in enumerate(
                (losses.total, losses.inplay, losses.rally_start, losses.rally_end)
            ):
                sums[index] += float(value.detach())
            batch_count += 1
        if not batch_count:
            raise ValueError("rally training partition has no batches")
        validation_bce = None
        if validation_windows is not None:
            validation_rows = predict_rally_windows(
                model,
                validation_windows,
                device=device,
                batch_size=batch_size,
                partition="validation",
            )
            validation_bce = float(rally_metrics(validation_rows)["raw_bce"])
            if not math.isfinite(validation_bce):
                raise RuntimeError("rally validation produced a non-finite BCE")
        history.append(
            {
                "epoch": epoch + 1,
                "total": sums[0] / batch_count,
                "inplay": sums[1] / batch_count,
                "rally_start": sums[2] / batch_count,
                "rally_end": sums[3] / batch_count,
                "validation_raw_bce": validation_bce,
                "sampling": sampler.last_sampling_stats,
            }
        )
        candidate = (
            validation_bce if validation_bce is not None else 0.0,
            epoch + 1,
            copy.deepcopy(model.state_dict()),
        )
        if validation_windows is None or best is None or candidate[:2] < best[:2]:
            best = candidate
        if recovery_path is not None:
            _atomic_torch_save(
                {
                    "schema": "rally_training_recovery",
                    "version": 1,
                    "identity": identity,
                    "completed_epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "sampler_epoch": sampler.epoch,
                    "rng_state": _rng_state(device),
                    "history": history,
                    "best": best,
                },
                recovery_path,
            )
        print(
            f"epoch {epoch + 1}/{epochs}: train={sums[0] / batch_count:.6f} "
            f"validation_bce={validation_bce}",
            flush=True,
        )
    assert best is not None
    model.load_state_dict(best[2])
    return history, best[1]


def _window_key(window: RallyWindow | SelectorWindow) -> tuple[Any, ...]:
    return (
        window.source_id,
        tuple(window.frame_indices),
        tuple(sorted(window.owned_frames)),
    )


def _match_legacy_windows(
    rally_windows: Sequence[RallyWindow],
    legacy_windows: Sequence[SelectorWindow],
) -> list[SelectorWindow]:
    legacy = {_window_key(window): window for window in legacy_windows}
    matched = []
    for window in rally_windows:
        key = _window_key(window)
        if key not in legacy:
            raise ValueError(
                f"legacy dataset has no matched rally window {window.window_id}"
            )
        matched.append(legacy[key])
    return matched


def _legacy_prediction_rows(
    model: JointRallyShuttleModel,
    windows: Sequence[SelectorWindow],
    fold: CrossFitFold,
    *,
    device: torch.device,
    batch_size: int,
    partition: str,
    mask_candidates: bool = True,
) -> list[dict[str, Any]]:
    rows = aggregate_joint_predictions(
        _predict_windows(
            model,
            windows,
            fold,
            device=device,
            batch_size=batch_size,
            owned_only=True,
            mask_candidates=mask_candidates,
        )
    )
    return [
        {
            "source_id": str(row["prediction_source_id"]),
            "frame": int(row["frame"]),
            "partition": partition,
            "inplay_target": int(row["inplay_target"]),
            "inplay_logit": float(row["inplay_logit"]),
            "inplay_probability": float(row["in_play_probability"]),
            "rally_start_target": row.get("rally_start_target"),
            "rally_start_logit": row.get("rally_start_logit"),
            "rally_end_target": row.get("rally_end_target"),
            "rally_end_logit": row.get("rally_end_logit"),
        }
        for row in rows
        if row.get("inplay_target") in (0, 1)
    ]


def _macro_metrics(
    metrics: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    scalar_paths = {
        "raw_bce": ("raw_bce",),
        "roc_auc": ("roc_auc",),
        "pr_auc": ("pr_auc",),
        "f1": ("f1",),
        "balanced_accuracy": ("balanced_accuracy",),
        "predicted_positive_rate": ("predicted_positive_rate",),
        "brier_score": ("calibration", "brier_score"),
        "expected_calibration_error": (
            "calibration",
            "expected_calibration_error",
        ),
        "interval_f1": ("intervals", "f1"),
        "boundary_mae": ("intervals", "mean_absolute_boundary_error"),
        "false_splits": ("intervals", "false_split_count"),
        "false_merges": ("intervals", "false_merge_count"),
        "fragmentation": ("intervals", "fragmentation_count"),
    }
    output: dict[str, Any] = {}
    for name, path in scalar_paths.items():
        values = []
        for camera_metrics in metrics.values():
            value: Any = camera_metrics
            for field in path:
                value = value[field]
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        output[name] = statistics.fmean(values) if values else None
    return output


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _training_camera_groups(
    fold: CameraHeldOutFold,
    camera_sources: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    return {
        camera: tuple(map(str, sources))
        for camera, sources in camera_sources.items()
        if camera != fold.held_out_camera
    }


def _prepare_rally_run(
    path: Path,
    dataset: RallyWindowDataset,
    recipe: RallyTrainingRecipe,
    camera_sources: Mapping[str, Sequence[str]],
    *,
    resume: bool,
    partition_fingerprints: Mapping[str, str] | None = None,
) -> tuple[Path, dict[str, list[str]]]:
    path = Path(path).expanduser().resolve()
    # Camera keys are labels. Source lists retain their chronological order.
    ordered_camera_sources = {
        camera: list(camera_sources[camera]) for camera in sorted(camera_sources)
    }
    configuration = {
        "schema": "resolved_rally_training",
        "version": 2,
        "training_recipe": asdict(recipe),
        "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
        "annotation_fingerprint": dataset.annotation_fingerprint,
        "camera_sources": ordered_camera_sources,
        **({"partition_fingerprints": dict(partition_fingerprints)} if partition_fingerprints is not None else {}),
    }
    configuration_path = path / "resolved-config.json"
    if resume:
        if not configuration_path.is_file():
            raise ValueError(
                "resume requires an existing run with resolved-config.json"
            )
        previous = json.loads(configuration_path.read_text())
        if previous.get("version") == 1:
            raise ValueError(
                "resume cannot verify the original camera order from a version 1 run; "
                "start a new run in a different output directory"
            )
        if isinstance(previous.get("training_recipe"), dict):
            previous["training_recipe"].setdefault("temporal_smoothing_weight", 0.0)
        if previous != configuration:
            raise ValueError("resume configuration or dataset identity has changed")
    else:
        path = prepare_output_directory(path)
        _write_json(configuration_path, configuration)
    return path, ordered_camera_sources


def run_independent_rally_experiment(
    rally_dataset: RallyWindowDataset,
    legacy_dataset: SelectorWindowDataset,
    output_dir: Path,
    *,
    camera_sources: Mapping[str, Sequence[str]] = DEFAULT_CAMERA_SOURCES,
    recipe: RallyTrainingRecipe | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train both matched architectures and evaluate each held-out camera once."""
    recipe = recipe or RallyTrainingRecipe()
    device = device or torch.device(recipe.device)
    recipe = replace(recipe, device=device.type)
    output_dir, camera_sources = _prepare_rally_run(
        output_dir, rally_dataset, recipe, camera_sources, resume=False
    )
    _seed_everything(recipe.seed)
    folds = camera_held_out_folds(camera_sources)
    clean_camera_metrics: dict[str, Any] = {}
    legacy_camera_metrics: dict[str, Any] = {}
    clean_test_rows: list[dict[str, Any]] = []
    legacy_test_rows: list[dict[str, Any]] = []
    clean_selected_epochs: list[int] = []
    fold_artifacts: list[dict[str, Any]] = []

    for fold in folds:
        fold_dir = output_dir / f"fold-{fold.fold_id}-{fold.held_out_camera}"
        fold_dir.mkdir()
        training_groups = _training_camera_groups(
            fold, camera_sources
        )
        eligible = [
            window
            for window in rally_dataset.windows
            if window.source_id in fold.training_source_ids
        ]
        train, validation, split = (
            chronological_camera_train_validation_split(
                eligible, training_groups, guard_seconds=1.0
            )
        )
        test = [
            window
            for window in rally_dataset.windows
            if window.source_id in fold.test_source_ids
        ]
        if not test:
            raise ValueError("held-out camera has no test windows")
        split_manifest = {
            "fold_id": fold.fold_id,
            "held_out_camera": fold.held_out_camera,
            "training_source_ids": list(fold.training_source_ids),
            "validation_source_ids": list(fold.validation_source_ids),
            "test_source_ids": list(fold.test_source_ids),
            "train_validation": split,
            "test_window_ids": [window.window_id for window in test],
            "selection_label_partitions": ["train", "validation"],
            "decoder_label_partition": "validation",
            "test_evaluations": 1,
        }
        fps_by_source = {
            window.source_id: float(window.metadata["fps"])
            for window in (*validation, *test)
        }

        clean = RallySegmenter(RallySegmenterConfig()).to(device)
        clean_history, clean_epoch = train_rally_segmenter(
            clean,
            train,
            validation,
            device=device,
            epochs=recipe.epochs,
            batch_size=recipe.batch_size,
            seed=recipe.seed,
            boundary_aux_weight=recipe.boundary_aux_weight,
            learning_rate=recipe.learning_rate,
            gradient_clip_norm=recipe.gradient_clip_norm,
            sampling=recipe.sampling,
        )
        clean_selected_epochs.append(clean_epoch)
        clean_validation = predict_rally_windows(
            clean,
            validation,
            device=device,
            batch_size=recipe.batch_size,
            partition="validation",
        )
        clean_decoder, clean_decoder_search = tune_rally_decoder(
            clean_validation, fps_by_source=fps_by_source
        )

        legacy_train = _match_legacy_windows(
            train, legacy_dataset.windows
        )
        legacy_validation = _match_legacy_windows(
            validation, legacy_dataset.windows
        )
        legacy_test = _match_legacy_windows(test, legacy_dataset.windows)
        selector_config = SelectorConfig(
            context_mode="full_context",
            frame_feature_dim=FRAME_DIMS["full_context"],
        )
        legacy = JointRallyShuttleModel(
            selector_config, boundary_heads=True
        ).to(device)
        legacy_fold = CrossFitFold(
            fold.fold_id,
            fold.training_source_ids,
            fold.test_source_ids,
        )
        legacy_history = _train_fold(
            legacy,
            legacy_train,
            legacy_validation,
            legacy_fold,
            device=device,
            epochs=recipe.epochs,
            batch_size=recipe.batch_size,
            seed=recipe.seed,
            selection_weight=0.0,
            boundary_weight=1.0,
            mask_candidates=True,
            evaluation_owned_only=True,
            sampling_mode=recipe.sampling,
            checkpoint_selection="validation_loss",
            validation_partition_designated=True,
            boundary_aux_weight=recipe.boundary_aux_weight,
            learning_rate=recipe.learning_rate,
            gradient_clip_norm=recipe.gradient_clip_norm,
        )
        legacy_validation_rows = _legacy_prediction_rows(
            legacy,
            legacy_validation,
            legacy_fold,
            device=device,
            batch_size=recipe.batch_size,
            partition="validation",
        )
        legacy_decoder, legacy_decoder_search = tune_rally_decoder(
            legacy_validation_rows, fps_by_source=fps_by_source
        )

        # This is the only point at which held-out labels are materialized.
        clean_raw_test = predict_rally_windows(
            clean,
            test,
            device=device,
            batch_size=recipe.batch_size,
            partition="test",
        )
        legacy_raw_test = _legacy_prediction_rows(
            legacy,
            legacy_test,
            legacy_fold,
            device=device,
            batch_size=recipe.batch_size,
            partition="test",
        )
        clean_decoded = decode_prediction_rows(
            clean_raw_test,
            fps_by_source=fps_by_source,
            config=clean_decoder,
        )
        legacy_decoded = decode_prediction_rows(
            legacy_raw_test,
            fps_by_source=fps_by_source,
            config=legacy_decoder,
        )
        clean_metrics = rally_metrics(clean_decoded)
        legacy_metrics = rally_metrics(legacy_decoded)
        clean_camera_metrics[fold.held_out_camera] = clean_metrics
        legacy_camera_metrics[fold.held_out_camera] = legacy_metrics
        clean_test_rows.extend(clean_decoded)
        legacy_test_rows.extend(legacy_decoded)

        clean_checkpoint = rally_checkpoint(
            clean,
            dataset_fingerprint=rally_dataset.manifest[
                "dataset_fingerprint"
            ],
            annotation_fingerprint=rally_dataset.annotation_fingerprint,
            split_manifest=split_manifest,
            training_recipe=recipe,
            chosen_epoch=clean_epoch,
            decoder_configuration=asdict(clean_decoder),
            pose_training_metadata=rally_dataset.manifest.get("pose_training_metadata"),
            extra={"epoch_history": clean_history},
        )
        torch.save(clean_checkpoint, fold_dir / "rally-segmenter.pt")
        # Preserve the established joint-checkpoint reconstruction contract.
        torch.save(
            {
                "model_state_dict": legacy.state_dict(),
                "model_type": "JointRallyShuttleModel",
                "selector_config": asdict(selector_config),
                "dataset_fingerprint": legacy_dataset.manifest[
                    "dataset_fingerprint"
                ],
                "pose_coordinate_mode": legacy_dataset.manifest[
                    "pose_coordinate_mode"
                ],
                "split": split_manifest,
                "mask_candidates": True,
                "epoch_mean_losses": legacy_history,
                "checkpoint_selection": legacy.checkpoint_selection,
                "sampling_mode": "boundary_balanced",
                "boundary_heads": True,
                "conditioning_mode": legacy.conditioning_mode,
                "boundary_aux_weight": recipe.boundary_aux_weight,
                "decoder_configuration": asdict(legacy_decoder),
            },
            fold_dir / "legacy-joint.pt",
        )
        _write_jsonl(
            fold_dir / "clean-test-predictions.jsonl", clean_decoded
        )
        _write_jsonl(
            fold_dir / "legacy-test-predictions.jsonl", legacy_decoded
        )
        fold_metrics = {
            "split_manifest": split_manifest,
            "clean": {
                "chosen_epoch": clean_epoch,
                "validation_decoder": asdict(clean_decoder),
                "decoder_search": clean_decoder_search,
                "test": clean_metrics,
                "test_by_source": metrics_by_source(clean_decoded),
            },
            "legacy_candidate_masked": {
                "chosen_epoch": legacy.checkpoint_selection[
                    "selected_epoch"
                ],
                "validation_decoder": asdict(legacy_decoder),
                "decoder_search": legacy_decoder_search,
                "test": legacy_metrics,
                "test_by_source": metrics_by_source(legacy_decoded),
            },
        }
        _write_json(fold_dir / "metrics.json", fold_metrics)
        fold_artifacts.append(fold_metrics)

    bootstrap = paired_bce_bootstrap(
        clean_test_rows,
        legacy_test_rows,
        samples=10_000,
        seed=recipe.seed,
    )
    acceptance = independent_model_acceptance(
        clean_camera_metrics, legacy_camera_metrics, bootstrap
    )
    result: dict[str, Any] = {
        "schema": "independent_cross_camera_rally_segmenter",
        "schema_version": 1,
        "dataset_fingerprint": rally_dataset.manifest[
            "dataset_fingerprint"
        ],
        "annotation_fingerprint": rally_dataset.annotation_fingerprint,
        "training_recipe": asdict(recipe),
        "folds": fold_artifacts,
        "clean_by_camera": clean_camera_metrics,
        "legacy_by_camera": legacy_camera_metrics,
        "clean_macro": _macro_metrics(clean_camera_metrics),
        "legacy_macro": _macro_metrics(legacy_camera_metrics),
        "paired_bce_bootstrap": bootstrap,
        "acceptance": acceptance,
        "stable_shuttle_track_follow_up_allowed": acceptance["accepted"],
        "production": None,
    }
    if acceptance["accepted"]:
        production_epoch = round(statistics.median(clean_selected_epochs))
        production = RallySegmenter(RallySegmenterConfig()).to(device)
        # Production has no validation selection; fit the preselected epoch.
        train_rally_segmenter_fixed_epochs(
            production,
            rally_dataset.windows,
            device=device,
            epochs=production_epoch,
            batch_size=recipe.batch_size,
            seed=recipe.seed,
            boundary_aux_weight=recipe.boundary_aux_weight,
            learning_rate=recipe.learning_rate,
            gradient_clip_norm=recipe.gradient_clip_norm,
            sampling=recipe.sampling,
        )
        oof_decoder, oof_search = tune_rally_decoder(
            clean_test_rows,
            fps_by_source={
                window.source_id: float(window.metadata["fps"])
                for window in rally_dataset.windows
            },
        )
        production_split = {
            "strategy": "all-five-sources",
            "source_ids": [
                source.source_id for source in rally_dataset.config.sources
            ],
            "epoch_selection": "median_fold_best_epoch",
            "fold_best_epochs": clean_selected_epochs,
            "decoder_calibration": "out_of_fold_predictions",
            "in_sample_decoder_labels_used": False,
        }
        production_checkpoint = rally_checkpoint(
            production,
            dataset_fingerprint=rally_dataset.manifest[
                "dataset_fingerprint"
            ],
            annotation_fingerprint=rally_dataset.annotation_fingerprint,
            split_manifest=production_split,
            training_recipe=recipe,
            chosen_epoch=production_epoch,
            decoder_configuration=asdict(oof_decoder),
            pose_training_metadata=rally_dataset.manifest.get("pose_training_metadata"),
            extra={
                "out_of_fold_decoder_search": oof_search,
                "acceptance": acceptance,
            },
        )
        production_path = output_dir / "production-rally-segmenter.pt"
        torch.save(production_checkpoint, production_path)
        result["production"] = {
            "checkpoint": production_path.name,
            "chosen_epoch": production_epoch,
            "decoder_configuration": asdict(oof_decoder),
            "decoder_calibration": "out_of_fold_predictions",
        }
    _write_json(output_dir / "metrics.json", result)
    return result


def run_clean_rally_experiment(
    data_config: RallyDataConfig,
    output_dir: Path,
    *,
    camera_sources: Mapping[str, Sequence[str]],
    recipe: RallyTrainingRecipe | None = None,
    device: torch.device | None = None,
    resume: bool = False,
    require_rtmpose: bool = False,
    person_model: str | None = None,
    pose_model: Path | None = None,
) -> dict[str, Any]:
    """Run candidate-free camera-held-out training without loading shuttle artifacts."""
    recipe = recipe or RallyTrainingRecipe()
    device = device or torch.device(recipe.device)
    recipe = replace(recipe, device=device.type)
    if data_config.rally_intervals_path is None or data_config.rally_manifest_path is None:
        raise ValueError("clean rally training requires reviewed rally annotations")
    folds = camera_held_out_folds(camera_sources)
    sources = {source.source_id: source for source in data_config.sources}
    if set(sources) != {source for fold in folds for source in fold.test_source_ids}:
        raise ValueError("camera source groups must cover every configured source exactly once")
    index = read_rally_intervals(data_config.rally_intervals_path, data_config.rally_manifest_path)
    if missing := set(sources) - set(index.sources):
        raise ValueError(f"rally manifest is missing sources: {sorted(missing)}")
    plans = plan_rally_camera_splits(index, camera_sources)
    feature_view = rally_feature_view(data_config.feature_view)

    output_dir = Path(output_dir).expanduser().resolve()
    # Full-source observations are used for held-out scoring and production.
    # A source divided by a fold cut gets independently extracted observations.
    rally_dataset = RallyWindowDataset(data_config)
    if rally_dataset.rally_index != index:
        raise ValueError("rally annotations changed while preparing the train/validation split")
    uses_rtmpose = "pose_training_metadata" in rally_dataset.manifest
    if require_rtmpose and not uses_rtmpose:
        raise ValueError("the RTMPose audio preset requires RTMPose observations for every source")
    full_windows: dict[str, list[RallyWindow]] = {source: [] for source in sources}
    for window in rally_dataset.windows:
        full_windows[window.source_id].append(window)
    partition_windows = {}
    partition_fingerprints = {}
    for plan in plans.values():
        for part in (*plan.train, *plan.validation):
            count = index.sources[part.source_id].frame_count
            if part.start == 0 and part.stop == count:
                partition_windows[part] = full_windows[part.source_id]
                continue
            windows, fingerprint = prepare_rally_partition(
                data_config, sources[part.source_id], part, index,
                output_dir.parent / ".rally-partitions",
                device=device.type, person_model=person_model, pose_model=pose_model,
            )
            partition_windows[part] = windows
            partition_fingerprints[f"{part.source_id}:{part.start}:{part.stop}"] = fingerprint
    output_dir, camera_sources = _prepare_rally_run(
        output_dir, rally_dataset, recipe, camera_sources, resume=resume,
        partition_fingerprints=partition_fingerprints,
    )
    _seed_everything(recipe.seed)
    camera_metrics: dict[str, Any] = {}
    test_rows: list[dict[str, Any]] = []
    selected_epochs: list[int] = []
    fold_artifacts = []
    model_config = RallySegmenterConfig(
        frame_feature_dim=feature_view.dimension, audio_attention=feature_view.uses_audio
    )
    calibration_fingerprints = {
        source: str(value["calibration_sha256"])
        for source, value in rally_dataset.manifest["sources"].items()
    }
    for fold in camera_held_out_folds(camera_sources):
        fold_dir = output_dir / f"fold-{fold.fold_id}-{fold.held_out_camera}"
        fold_dir.mkdir(exist_ok=resume)
        training_cameras = _training_camera_groups(fold, camera_sources)
        train = [
            window for camera in training_cameras for part in plans[camera].train
            for window in partition_windows[part]
        ]
        validation = [
            window for camera in training_cameras for part in plans[camera].validation
            for window in partition_windows[part]
        ]
        split = {
            "strategy": "chronological-source-ranges",
            "frame_range_convention": "source-local, start inclusive, stop exclusive",
            "partition_fingerprints": {
                key: partition_fingerprints[key]
                for camera in training_cameras
                for part in (*plans[camera].train, *plans[camera].validation)
                if (key := f"{part.source_id}:{part.start}:{part.stop}") in partition_fingerprints
            },
            "cameras": {
                camera: {
                    "cut_frame": plans[camera].cut,
                    "total_frames": plans[camera].total_frames,
                    "train": [asdict(part) for part in plans[camera].train],
                    "validation": [asdict(part) for part in plans[camera].validation],
                }
                for camera in training_cameras
            },
        }
        test = [window for source in fold.test_source_ids for window in full_windows[source]]
        if not validation or not test:
            raise ValueError("clean rally fold requires validation and held-out windows")
        split_manifest = {
            "fold_id": fold.fold_id, "held_out_camera": fold.held_out_camera,
            "training_source_ids": list(fold.training_source_ids),
            "validation_source_ids": list(fold.validation_source_ids),
            "test_source_ids": list(fold.test_source_ids), "train_validation": split,
            "train_window_ids": [window.window_id for window in train],
            "validation_window_ids": [window.window_id for window in validation],
            "test_window_ids": [window.window_id for window in test],
            "checkpoint_selection": "validation_raw_bce_only",
            "decoder_selection": "validation_labels_only",
            "test_labels_used_for_selection": False, "test_evaluations": 1,
        }
        model = RallySegmenter(model_config).to(device)
        history, epoch = train_rally_segmenter(
            model,
            train,
            validation,
            device=device,
            epochs=recipe.epochs,
            batch_size=recipe.batch_size,
            seed=recipe.seed,
            boundary_aux_weight=recipe.boundary_aux_weight,
            temporal_smoothing_weight=recipe.temporal_smoothing_weight,
            inplay_dice_weight=recipe.inplay_dice_weight,
            learning_rate=recipe.learning_rate,
            gradient_clip_norm=recipe.gradient_clip_norm,
            sampling=recipe.sampling,
            recovery_path=fold_dir / "training-recovery.pt",
            resume=resume,
            recovery_identity={
                "dataset_fingerprint": rally_dataset.manifest["dataset_fingerprint"],
                "annotation_fingerprint": rally_dataset.annotation_fingerprint,
                "split_manifest": split_manifest,
            },
        )
        selected_epochs.append(epoch)
        validation_rows = predict_rally_windows(
            model, validation, device=device, batch_size=recipe.batch_size,
            partition="validation", include_unlabeled=True,
        )
        fps_by_source = {window.source_id: float(window.metadata["fps"])
                         for window in validation}
        test_fps_by_source = {window.source_id: float(window.metadata["fps"]) for window in test}
        decoder, decoder_search = tune_rally_decoder(
            validation_rows, fps_by_source=fps_by_source, hysteresis=uses_rtmpose
        )
        raw_test = predict_rally_windows(
            model, test, device=device, batch_size=recipe.batch_size, partition="test", include_unlabeled=True,
        )
        decoded = decode_prediction_rows(raw_test, fps_by_source=test_fps_by_source, config=decoder)
        metrics = rally_metrics(decoded)
        camera_metrics[fold.held_out_camera] = metrics
        test_rows.extend(decoded)
        checkpoint = rally_checkpoint(
            model, dataset_fingerprint=rally_dataset.manifest["dataset_fingerprint"],
            annotation_fingerprint=rally_dataset.annotation_fingerprint,
            split_manifest=split_manifest, training_recipe=recipe, chosen_epoch=epoch,
            decoder_configuration=asdict(decoder),
            pose_training_metadata=rally_dataset.manifest.get("pose_training_metadata"),
            feature_view=feature_view.name,
            calibration_fingerprints=calibration_fingerprints,
            extra={"epoch_history": history, "experiment_role": "expanded_clean_only"},
        )
        torch.save(checkpoint, fold_dir / "rally-segmenter.pt")
        _write_jsonl(fold_dir / "test-predictions.jsonl", decoded)
        fold_result = {
            "split_manifest": split_manifest, "chosen_epoch": epoch,
            "validation_decoder": asdict(decoder), "decoder_search": decoder_search,
            "test": metrics, "test_by_source": metrics_by_source(decoded),
        }
        _write_json(fold_dir / "metrics.json", fold_result)
        fold_artifacts.append(fold_result)
    production_epoch = round(statistics.median(selected_epochs))
    production = RallySegmenter(model_config).to(device)
    production_history = train_rally_segmenter_fixed_epochs(
        production,
        rally_dataset.windows,
        device=device,
        epochs=production_epoch,
        batch_size=recipe.batch_size,
        seed=recipe.seed,
        boundary_aux_weight=recipe.boundary_aux_weight,
        temporal_smoothing_weight=recipe.temporal_smoothing_weight,
        inplay_dice_weight=recipe.inplay_dice_weight,
        learning_rate=recipe.learning_rate,
        gradient_clip_norm=recipe.gradient_clip_norm,
        sampling=recipe.sampling,
        recovery_path=output_dir / "production-training-recovery.pt",
        resume=resume,
        recovery_identity={
            "dataset_fingerprint": rally_dataset.manifest["dataset_fingerprint"],
            "annotation_fingerprint": rally_dataset.annotation_fingerprint,
            "epoch_selection": "median_fold_best_epoch",
            "fold_best_epochs": selected_epochs,
        },
    )
    oof_decoder, oof_search = tune_rally_decoder(
        test_rows, fps_by_source={window.source_id: float(window.metadata["fps"])
                                  for window in rally_dataset.windows},
        hysteresis=uses_rtmpose,
    )
    production_split = {
        "strategy": "all-sources", "source_ids": list(sources),
        "camera_groups": {camera: list(sources) for camera, sources in camera_sources.items()},
        "epoch_selection": "median_fold_best_epoch", "fold_best_epochs": selected_epochs,
        "decoder_calibration": "out_of_fold_predictions", "in_sample_decoder_labels_used": False,
    }
    production_checkpoint = rally_checkpoint(
        production, dataset_fingerprint=rally_dataset.manifest["dataset_fingerprint"],
        annotation_fingerprint=rally_dataset.annotation_fingerprint,
        split_manifest=production_split, training_recipe=recipe,
        chosen_epoch=production_epoch, decoder_configuration=asdict(oof_decoder),
        pose_training_metadata=rally_dataset.manifest.get("pose_training_metadata"),
        feature_view=feature_view.name,
        calibration_fingerprints=calibration_fingerprints,
        extra={"epoch_history": production_history, "out_of_fold_decoder_search": oof_search,
               "experiment_role": "expanded_clean_only_production"},
    )
    torch.save(production_checkpoint, output_dir / "production-rally-segmenter.pt")
    result = {
        "schema": "clean_cross_camera_rally_segmenter", "schema_version": 1,
        "dataset_manifest": rally_dataset.manifest,
        "training_recipe": asdict(recipe), "camera_sources": {key: list(value) for key, value in camera_sources.items()},
        "folds": fold_artifacts, "by_camera": camera_metrics,
        "macro": _macro_metrics(camera_metrics),
        "production": {"checkpoint": "production-rally-segmenter.pt",
                       "chosen_epoch": production_epoch,
                       "decoder_configuration": asdict(oof_decoder)},
    }
    _write_json(output_dir / "metrics.json", result)
    return result


def run_candidate_inclusive_rally_experiment(
    rally_dataset: RallyWindowDataset,
    candidate_dataset: SelectorWindowDataset,
    output_dir: Path,
    *, camera_sources: Mapping[str, Sequence[str]],
    recipe: RallyTrainingRecipe | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run the matched candidate-token arm with shuttle-selection loss disabled."""
    recipe = recipe or RallyTrainingRecipe()
    device = device or torch.device(recipe.device)
    recipe = replace(recipe, device=device.type)
    output_dir, camera_sources = _prepare_rally_run(
        output_dir, rally_dataset, recipe, camera_sources, resume=False
    )
    _seed_everything(recipe.seed)
    camera_metrics: dict[str, Any] = {}
    fold_artifacts = []
    for fold in camera_held_out_folds(camera_sources):
        fold_dir = output_dir / f"fold-{fold.fold_id}-{fold.held_out_camera}"
        fold_dir.mkdir()
        eligible = [window for window in rally_dataset.windows
                    if window.source_id in fold.training_source_ids]
        train, validation, split = chronological_camera_train_validation_split(
            eligible, _training_camera_groups(fold, camera_sources), guard_seconds=1.0
        )
        test = [window for window in rally_dataset.windows if window.source_id in fold.test_source_ids]
        candidate_train = _match_legacy_windows(train, candidate_dataset.windows)
        candidate_validation = _match_legacy_windows(validation, candidate_dataset.windows)
        candidate_test = _match_legacy_windows(test, candidate_dataset.windows)
        model = JointRallyShuttleModel(
            SelectorConfig(context_mode="full_context", frame_feature_dim=FRAME_DIMS["full_context"]),
            boundary_heads=True,
        ).to(device)
        crossfit_fold = CrossFitFold(fold.fold_id, fold.training_source_ids, fold.test_source_ids)
        history = _train_fold(
            model,
            candidate_train,
            candidate_validation,
            crossfit_fold,
            device=device,
            epochs=recipe.epochs,
            batch_size=recipe.batch_size,
            seed=recipe.seed,
            selection_weight=0.0,
            boundary_weight=1.0,
            mask_candidates=False,
            evaluation_owned_only=True,
            sampling_mode=recipe.sampling,
            checkpoint_selection="validation_loss",
            validation_partition_designated=True,
            boundary_aux_weight=recipe.boundary_aux_weight,
            learning_rate=recipe.learning_rate,
            gradient_clip_norm=recipe.gradient_clip_norm,
        )
        validation_rows = _legacy_prediction_rows(
            model, candidate_validation, crossfit_fold, device=device,
            batch_size=recipe.batch_size, partition="validation", mask_candidates=False,
        )
        fps_by_source = {window.source_id: float(window.metadata["fps"])
                         for window in (*validation, *test)}
        decoder, decoder_search = tune_rally_decoder(validation_rows, fps_by_source=fps_by_source)
        raw_test = _legacy_prediction_rows(
            model, candidate_test, crossfit_fold, device=device,
            batch_size=recipe.batch_size, partition="test", mask_candidates=False,
        )
        decoded = decode_prediction_rows(raw_test, fps_by_source=fps_by_source, config=decoder)
        metrics = rally_metrics(decoded)
        camera_metrics[fold.held_out_camera] = metrics
        split_manifest = {
            "fold_id": fold.fold_id, "held_out_camera": fold.held_out_camera,
            "training_source_ids": list(fold.training_source_ids),
            "validation_source_ids": list(fold.validation_source_ids),
            "test_source_ids": list(fold.test_source_ids), "train_validation": split,
            "selection_weight": 0.0, "candidate_tokens_masked": False,
            "decoder_selection": "validation_labels_only", "test_labels_used_for_selection": False,
        }
        checkpoint = {
            "model_state_dict": model.state_dict(), "model_type": "JointRallyShuttleModel",
            "selector_config": asdict(model.config),
            "candidate_dataset_fingerprint": candidate_dataset.manifest["dataset_fingerprint"],
            "rally_dataset_fingerprint": rally_dataset.manifest["dataset_fingerprint"],
            "annotation_fingerprint": rally_dataset.annotation_fingerprint,
            "split_manifest": split_manifest, "training_recipe": asdict(recipe),
            "epoch_mean_losses": history, "checkpoint_selection": model.checkpoint_selection,
            "boundary_heads": True, "selection_weight": 0.0, "candidate_tokens_masked": False,
            "decoder_configuration": asdict(decoder),
        }
        torch.save(checkpoint, fold_dir / "candidate-inclusive.pt")
        _write_jsonl(fold_dir / "test-predictions.jsonl", decoded)
        fold_result = {"split_manifest": split_manifest,
                       "chosen_epoch": model.checkpoint_selection["selected_epoch"],
                       "validation_decoder": asdict(decoder), "decoder_search": decoder_search,
                       "test": metrics, "test_by_source": metrics_by_source(decoded)}
        _write_json(fold_dir / "metrics.json", fold_result)
        fold_artifacts.append(fold_result)
    result = {"schema": "candidate_inclusive_cross_camera_rally_segmenter", "schema_version": 1,
              "rally_dataset_manifest": rally_dataset.manifest,
              "candidate_dataset_fingerprint": candidate_dataset.manifest["dataset_fingerprint"],
              "training_recipe": asdict(recipe), "selection_weight": 0.0,
              "candidate_tokens_masked": False,
              "camera_sources": {key: list(value) for key, value in camera_sources.items()},
              "folds": fold_artifacts, "by_camera": camera_metrics,
              "macro": _macro_metrics(camera_metrics)}
    _write_json(output_dir / "metrics.json", result)
    return result


def train_rally_segmenter_fixed_epochs(
    model: RallySegmenter,
    windows: Sequence[RallyWindow],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    boundary_aux_weight: float,
    temporal_smoothing_weight: float = 0.0,
    inplay_dice_weight: float = 0.0,
    learning_rate: float = 3e-4,
    gradient_clip_norm: float = 1.0,
    sampling: str = "boundary_balanced",
    recovery_path: Path | None = None,
    resume: bool = False,
    recovery_identity: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fit production data using the same epoch recovery policy as fold training."""
    history, _ = train_rally_segmenter(
        model,
        windows,
        None,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        boundary_aux_weight=boundary_aux_weight,
        temporal_smoothing_weight=temporal_smoothing_weight,
        inplay_dice_weight=inplay_dice_weight,
        learning_rate=learning_rate,
        gradient_clip_norm=gradient_clip_norm,
        sampling=sampling,
        recovery_path=recovery_path,
        resume=resume,
        recovery_identity=recovery_identity,
    )
    return history


def _load_camera_sources(
    value: Mapping[str, Any] | None,
) -> Mapping[str, tuple[str, ...]]:
    if value is None:
        return DEFAULT_CAMERA_SOURCES
    return {
        str(camera): tuple(map(str, sources))
        for camera, sources in value.items()
    }


def load_experiment_datasets(
    config_path: Path,
) -> tuple[
    RallyWindowDataset,
    SelectorWindowDataset,
    Mapping[str, tuple[str, ...]],
]:
    config_path = config_path.expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    rally_config: RallyDataConfig = rally_data_config_from_selector_mapping(
        raw, base_dir=config_path.parent
    )
    if rally_feature_view(rally_config.feature_view).uses_audio:
        raise ValueError("audio feature views require --clean-only")
    rally_dataset = RallyWindowDataset(rally_config)
    legacy_config, _ = load_dataset_config(
        config_path,
        context_mode="full_context",
        pose_coordinate_mode="player_relative",
    )
    # The matched candidate-masked baseline does not use shuttle-selection
    # labels. Its dataset identity still fingerprints the actual annotation
    # file, but a stale selector-only expected hash must not block rally work.
    legacy_config = replace(
        legacy_config, expected_annotation_sha256=None
    )
    legacy_dataset = SelectorWindowDataset(legacy_config)
    if (
        legacy_dataset.manifest["pose_representation_version"]
        != rally_dataset.manifest["pose_representation_version"]
    ):
        raise ValueError("matched datasets use different pose representations")
    camera_sources = _load_camera_sources(raw.get("camera_sources"))
    configured = {
        source.source_id for source in rally_dataset.config.sources
    }
    grouped = {
        source for sources in camera_sources.values() for source in sources
    }
    if configured != grouped:
        raise ValueError(
            "camera source groups must cover every configured source exactly once"
        )
    return rally_dataset, legacy_dataset, camera_sources


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preset", default="default", choices=("default", "benchmark", "rtmpose-audio")
    )
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--boundary-aux-weight", type=float)
    parser.add_argument("--temporal-smoothing-weight", type=float)
    parser.add_argument("--inplay-dice-weight", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sampling", choices=("boundary_balanced", "uniform"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the clean-only run from committed epochs",
    )
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--person-model", help="Original detector asset for split recordings")
    parser.add_argument("--pose-model", type=Path, help="Original pose asset for split recordings")
    parser.add_argument(
        "--feature-view",
        choices=tuple(RALLY_FEATURE_VIEWS),
        help="Override dataset.rally_feature_view for the clean-only experiment",
    )
    parser.add_argument("--candidate-inclusive-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    raw_training = json.loads(args.config.expanduser().read_text(encoding="utf-8")).get(
        "training", {}
    )
    allowed = set(asdict(RallyTrainingRecipe()))
    unknown = set(raw_training) - allowed
    if unknown:
        raise ValueError(f"unknown training configuration fields: {sorted(unknown)}")
    overrides = {
        **raw_training,
        **{
            key: getattr(args, key)
            for key in allowed
            if hasattr(args, key) and getattr(args, key) is not None
        },
    }
    rtmpose_preset = args.preset == "rtmpose-audio"
    if rtmpose_preset:
        if not args.clean_only:
            raise ValueError("the RTMPose audio preset requires --clean-only")
        if args.feature_view not in (None, "full_audio_zscore", CANONICAL_RALLY_FEATURE_VIEW):
            raise ValueError("the RTMPose audio preset requires a normalized-audio feature view")
        overrides.setdefault("temporal_smoothing_weight", 0.1)
    recipe = RallyTrainingRecipe.from_preset(
        "default" if rtmpose_preset else args.preset, **overrides
    )
    if recipe.temporal_smoothing_weight and not args.clean_only:
        raise ValueError("temporal smoothing requires --clean-only")
    if recipe.inplay_dice_weight and not args.clean_only:
        raise ValueError("in-play Dice loss requires --clean-only")
    if args.resume and not args.clean_only:
        raise ValueError(
            "epoch recovery currently requires --clean-only; comparative study arms use their historical trainer"
        )
    if args.clean_only and args.candidate_inclusive_only:
        raise ValueError("choose only one rally experiment arm")
    if args.feature_view is not None and not args.clean_only:
        raise ValueError("--feature-view requires --clean-only")
    if not args.clean_only and (args.person_model is not None or args.pose_model is not None):
        raise ValueError("partition extraction model options require --clean-only")
    if args.clean_only:
        config_path = args.config.expanduser().resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        data_config = rally_data_config_from_selector_mapping(
            raw, base_dir=config_path.parent,
            default_context_seconds=(
                RALLY_TRANSFORMER_CONTEXT_RADIUS_SECONDS if rtmpose_preset else RallyDataConfig.context_seconds
            ),
        )
        feature_view = args.feature_view
        if rtmpose_preset and feature_view is None:
            feature_view = CANONICAL_RALLY_FEATURE_VIEW
        if feature_view is not None:
            data_config = replace(data_config, feature_view=feature_view)
        camera_sources = _load_camera_sources(raw.get("camera_sources"))
        configured = {source.source_id for source in data_config.sources}
        grouped = [source for sources in camera_sources.values() for source in sources]
        if configured != set(grouped) or len(grouped) != len(set(grouped)):
            raise ValueError(
                "camera source groups must cover every configured source exactly once"
            )
        run_clean_rally_experiment(
            data_config,
            args.output_dir,
            camera_sources=camera_sources,
            recipe=recipe,
            device=torch.device(recipe.device),
            resume=args.resume,
            require_rtmpose=rtmpose_preset,
            person_model=args.person_model,
            pose_model=args.pose_model,
        )
        return 0
    if args.candidate_inclusive_only:
        rally_dataset, candidate_dataset, camera_sources = load_experiment_datasets(
            args.config
        )
        run_candidate_inclusive_rally_experiment(
            rally_dataset,
            candidate_dataset,
            args.output_dir,
            camera_sources=camera_sources,
            recipe=recipe,
            device=torch.device(recipe.device),
        )
        return 0
    rally_dataset, legacy_dataset, camera_sources = load_experiment_datasets(
        args.config
    )
    run_independent_rally_experiment(
        rally_dataset,
        legacy_dataset,
        args.output_dir,
        camera_sources=camera_sources,
        recipe=recipe,
        device=torch.device(recipe.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
