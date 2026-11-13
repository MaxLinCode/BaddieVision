"""Report-only calibrated-camera shortcut diagnostic.

This module intentionally does not choose a production feature schema or train a
production checkpoint. Camera labels and calibration parameters are used only by
the diagnostic orchestration and never enter ``RallyBatch``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from src.court_projection import CourtHomography, HALF_LENGTH, HALF_WIDTH

from .experiment import _seed_everything, prepare_output_directory
from .joint_inference import InPlayDecoderConfig
from .rally import RallySegmenter, RallySegmenterConfig, RallyTrainingRecipe, rally_checkpoint
from .rally_dataset import RallyWindow, RallyWindowDataset
from .rally_evaluation import (
    camera_held_out_folds,
    chronological_camera_train_validation_split,
    decode_prediction_rows,
    metrics_by_source,
    paired_bce_bootstrap,
    rally_metrics,
    tune_rally_decoder,
)
from .rally_experiment import (
    _macro_metrics,
    predict_rally_windows,
    train_rally_segmenter,
)
from .rally_features import (
    FULL_RALLY_FEATURE_NAMES,
    RALLY_FEATURE_VIEWS,
    RALLY_PROBE_GROUPS,
    rally_feature_indices,
)


SEED = 1729
PROBE_BOOTSTRAP_SAMPLES = 10_000
SENSITIVITY_SAMPLES = 100
SENSITIVITY_NOISE_PIXELS = (2.0, 5.0)
SUBSTANTIAL_DOMAIN_LOWER_BOUND = 1.0 / 3.0 + 0.10


@dataclass(frozen=True)
class ProbeSample:
    source_id: str
    camera: str
    state: int
    start_frame: int
    end_frame: int
    values: np.ndarray


def _camera_lookup(camera_sources: Mapping[str, Sequence[str]]) -> dict[str, str]:
    output: dict[str, str] = {}
    for camera, sources in camera_sources.items():
        for source in sources:
            if source in output:
                raise ValueError(f"probe source belongs to multiple cameras: {source}")
            output[str(source)] = str(camera)
    return output


def _owned_local_indices(window: RallyWindow) -> list[int]:
    return [
        index
        for index, frame in enumerate(window.frame_indices)
        if frame in window.owned_frames
    ]


def _aggregate_validity_aware(
    values: torch.Tensor, validity: torch.Tensor
) -> np.ndarray:
    if values.ndim != 2 or validity.shape != values.shape:
        raise ValueError("probe aggregation expects aligned [frames, features] tensors")
    raw = values.detach().cpu().numpy().astype(np.float64, copy=False)
    valid = validity.detach().cpu().numpy().astype(bool, copy=False)
    counts = valid.sum(axis=0)
    safe_counts = np.maximum(counts, 1)
    means = np.where(valid, raw, 0.0).sum(axis=0) / safe_counts
    centered = np.where(valid, raw - means, 0.0)
    standard_deviations = np.sqrt((centered * centered).sum(axis=0) / safe_counts)
    observed_fraction = valid.mean(axis=0)
    means[counts == 0] = 0.0
    standard_deviations[counts == 0] = 0.0
    return np.concatenate((means, standard_deviations, observed_fraction))


def build_probe_samples(
    windows: Sequence[RallyWindow],
    camera_sources: Mapping[str, Sequence[str]],
    feature_group: str,
) -> list[ProbeSample]:
    """Build one complete-state sample from each unique ownership window."""
    if feature_group not in RALLY_PROBE_GROUPS:
        raise ValueError(f"unknown camera probe feature group: {feature_group}")
    source_to_camera = _camera_lookup(camera_sources)
    indices = rally_feature_indices(RALLY_PROBE_GROUPS[feature_group])
    output: list[ProbeSample] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for window in sorted(windows, key=lambda item: (item.source_id, item.anchor_frame)):
        if window.source_id not in source_to_camera:
            raise ValueError(f"probe window has no camera mapping: {window.source_id}")
        if window.frame_values.shape[-1] != len(FULL_RALLY_FEATURE_NAMES):
            raise ValueError("camera probe must be built from the full feature view")
        owned_local = _owned_local_indices(window)
        owned_frames = tuple(window.frame_indices[index] for index in owned_local)
        key = (window.source_id, owned_frames)
        if key in seen:
            raise ValueError("probe ownership windows are not unique")
        seen.add(key)
        targets = [int(window.inplay_targets[index]) for index in owned_local]
        # Mixed, partial, or masked boundary windows are excluded from primary probe.
        if not targets or any(target not in (0, 1) for target in targets) or len(set(targets)) != 1:
            continue
        feature_indices = torch.tensor(indices, dtype=torch.long)
        values = window.frame_values[owned_local].index_select(-1, feature_indices)
        validity = window.frame_validity[owned_local].index_select(-1, feature_indices)
        output.append(
            ProbeSample(
                source_id=window.source_id,
                camera=source_to_camera[window.source_id],
                state=targets[0],
                start_frame=min(owned_frames),
                end_frame=max(owned_frames),
                values=_aggregate_validity_aware(values, validity),
            )
        )
    return output


def chronological_probe_split(
    samples: Sequence[ProbeSample],
    frame_count_by_source: Mapping[str, int],
    fps_by_source: Mapping[str, float],
    *,
    train_fraction: float = 0.8,
    guard_seconds: float = 1.0,
) -> tuple[list[ProbeSample], list[ProbeSample], dict[str, Any]]:
    """Split every source chronologically and remove a one-second guard."""
    if not 0 < train_fraction < 1 or guard_seconds < 0:
        raise ValueError("invalid chronological probe split parameters")
    train: list[ProbeSample] = []
    test: list[ProbeSample] = []
    manifest: dict[str, Any] = {"sources": {}, "train_fraction": train_fraction, "guard_seconds": guard_seconds}
    for source in sorted(frame_count_by_source):
        frame_count = int(frame_count_by_source[source])
        fps = float(fps_by_source[source])
        cut = math.floor(frame_count * train_fraction)
        guard = math.ceil(fps * guard_seconds)
        source_train = [
            sample for sample in samples
            if sample.source_id == source and sample.end_frame < cut - guard
        ]
        source_test = [
            sample for sample in samples
            if sample.source_id == source and sample.start_frame >= cut + guard
        ]
        train.extend(source_train)
        test.extend(source_test)
        manifest["sources"][source] = {
            "frame_count": frame_count,
            "cut_frame": cut,
            "guard_frames": guard,
            "train_window_count_before_balance": len(source_train),
            "test_window_count_before_balance": len(source_test),
        }
    if not train or not test:
        raise ValueError("chronological probe split produced an empty partition")
    return train, test, manifest


def balance_probe_partition(
    samples: Sequence[ProbeSample], *, seed: int = SEED
) -> tuple[list[ProbeSample], dict[str, Any]]:
    """Balance complete InPlay/out-of-play windows equally across cameras."""
    cameras = sorted({sample.camera for sample in samples})
    strata = {
        (camera, state): [sample for sample in samples if sample.camera == camera and sample.state == state]
        for camera in cameras
        for state in (0, 1)
    }
    if any(not values for values in strata.values()):
        missing = [str(key) for key, values in strata.items() if not values]
        raise ValueError(f"probe balancing has empty camera/state strata: {missing}")
    retained = min(map(len, strata.values()))
    selected: list[ProbeSample] = []
    for stratum_index, key in enumerate(sorted(strata)):
        values = sorted(strata[key], key=lambda item: (item.source_id, item.start_frame))
        generator = random.Random(seed + stratum_index)
        chosen = generator.sample(values, retained)
        selected.extend(chosen)
    selected.sort(key=lambda item: (item.source_id, item.start_frame))
    return selected, {
        "seed": seed,
        "windows_per_camera_state": retained,
        "counts_before": {f"{camera}:{state}": len(values) for (camera, state), values in strata.items()},
        "count_after": len(selected),
    }


def _normalization(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def _probe_metrics(labels: Sequence[str], predictions: Sequence[str], cameras: Sequence[str]) -> dict[str, Any]:
    matrix = confusion_matrix(labels, predictions, labels=list(cameras))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(cameras), average="macro", zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "camera_order": list(cameras),
        "per_camera_recall": {
            camera: float(matrix[index, index] / matrix[index].sum()) if matrix[index].sum() else 0.0
            for index, camera in enumerate(cameras)
        },
    }


def temporal_block_probe_bootstrap(
    samples: Sequence[ProbeSample],
    predictions: Sequence[str],
    *,
    samples_count: int = PROBE_BOOTSTRAP_SAMPLES,
    seed: int = SEED,
    block_windows: int = 5,
) -> dict[str, Any]:
    """Camera/state-stratified bootstrap of contiguous source-local blocks."""
    if len(samples) != len(predictions) or samples_count <= 0 or block_windows <= 0:
        raise ValueError("invalid temporal-block bootstrap inputs")
    rows = list(zip(samples, map(str, predictions)))
    cameras = sorted({sample.camera for sample in samples})
    strata: dict[tuple[str, int], list[list[tuple[ProbeSample, str]]]] = {}
    for camera in cameras:
        for state in (0, 1):
            blocks: list[list[tuple[ProbeSample, str]]] = []
            by_source: dict[str, list[tuple[ProbeSample, str]]] = {}
            for row in rows:
                if row[0].camera == camera and row[0].state == state:
                    by_source.setdefault(row[0].source_id, []).append(row)
            for source_rows in by_source.values():
                source_rows.sort(key=lambda row: row[0].start_frame)
                blocks.extend(
                    source_rows[start : start + block_windows]
                    for start in range(0, len(source_rows), block_windows)
                )
            if not blocks:
                raise ValueError("every probe bootstrap camera/state stratum needs a block")
            strata[(camera, state)] = blocks
    generator = random.Random(seed)
    balanced: list[float] = []
    macro_f1: list[float] = []
    for _ in range(samples_count):
        draw: list[tuple[ProbeSample, str]] = []
        for key, blocks in strata.items():
            target = sum(len(block) for block in blocks)
            selected: list[tuple[ProbeSample, str]] = []
            while len(selected) < target:
                selected.extend(blocks[generator.randrange(len(blocks))])
            draw.extend(selected[:target])
        metrics = _probe_metrics(
            [row[0].camera for row in draw], [row[1] for row in draw], cameras
        )
        balanced.append(metrics["balanced_accuracy"])
        macro_f1.append(metrics["macro_f1"])

    def interval(values: list[float]) -> list[float]:
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    return {
        "seed": seed,
        "samples": samples_count,
        "stratification": "camera_state_source_local_temporal_blocks",
        "block_windows": block_windows,
        "balanced_accuracy_confidence_interval_95": interval(balanced),
        "macro_f1_confidence_interval_95": interval(macro_f1),
    }


def run_camera_probe(
    dataset: RallyWindowDataset,
    camera_sources: Mapping[str, Sequence[str]],
    *,
    bootstrap_samples: int = PROBE_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if dataset.feature_view.name != "full":
        raise ValueError("camera probe requires a full-view rally dataset")
    frame_counts = {
        source.source_id: next(
            int(window.metadata["frame_count"])
            for window in dataset.windows
            if window.source_id == source.source_id
        )
        for source in dataset.config.sources
    }
    fps = {
        source.source_id: next(
            float(window.metadata["fps"])
            for window in dataset.windows
            if window.source_id == source.source_id
        )
        for source in dataset.config.sources
    }
    output: dict[str, Any] = {}
    expected_train_keys: set[tuple[str, int, int]] | None = None
    expected_test_keys: set[tuple[str, int, int]] | None = None
    for feature_group in RALLY_PROBE_GROUPS:
        raw = build_probe_samples(dataset.windows, camera_sources, feature_group)
        train, test, split = chronological_probe_split(raw, frame_counts, fps)
        train, train_balance = balance_probe_partition(train)
        test, test_balance = balance_probe_partition(test)
        train_keys = {(row.source_id, row.start_frame, row.end_frame) for row in train}
        test_keys = {(row.source_id, row.start_frame, row.end_frame) for row in test}
        if expected_train_keys is None:
            expected_train_keys, expected_test_keys = train_keys, test_keys
        elif train_keys != expected_train_keys or test_keys != expected_test_keys:
            raise AssertionError("camera probe feature groups do not use identical windows")
        x_train = np.stack([sample.values for sample in train])
        x_test = np.stack([sample.values for sample in test])
        y_train = np.asarray([sample.camera for sample in train])
        y_test = np.asarray([sample.camera for sample in test])
        mean, scale = _normalization(x_train)
        model = LogisticRegression(
            solver="lbfgs", max_iter=2_000, random_state=SEED
        )
        model.fit((x_train - mean) / scale, y_train)
        predicted = model.predict((x_test - mean) / scale)
        cameras = sorted(camera_sources)
        metrics = _probe_metrics(y_test.tolist(), predicted.tolist(), cameras)
        bootstrap = temporal_block_probe_bootstrap(
            test,
            predicted.tolist(),
            samples_count=bootstrap_samples,
            seed=SEED,
        )
        lower = bootstrap["balanced_accuracy_confidence_interval_95"][0]
        output[feature_group] = {
            "feature_names": list(RALLY_PROBE_GROUPS[feature_group]),
            "aggregates_per_feature": ["valid_mean", "valid_standard_deviation", "valid_fraction"],
            "train_window_count": len(train),
            "test_window_count": len(test),
            "split": split,
            "train_balance": train_balance,
            "test_balance": test_balance,
            "normalization_fit_partition": "train_only",
            "normalization_mean": mean.tolist(),
            "normalization_scale": scale.tolist(),
            "classifier": {
                "type": "multinomial_l2_logistic_regression",
                "solver": "lbfgs",
                "seed": SEED,
                "hyperparameter_tuning": False,
            },
            "metrics": metrics,
            "bootstrap": bootstrap,
            "substantial_domain_identity": lower >= SUBSTANTIAL_DOMAIN_LOWER_BOUND,
            "substantial_threshold": SUBSTANTIAL_DOMAIN_LOWER_BOUND,
        }
    return output


def project_rally_windows(
    windows: Sequence[RallyWindow], feature_view: str
) -> list[RallyWindow]:
    """Project full windows and their validity masks through a registered view."""
    view = RALLY_FEATURE_VIEWS.get(feature_view)
    if view is None:
        raise ValueError(f"unknown rally feature view: {feature_view}")
    output: list[RallyWindow] = []
    for window in windows:
        if window.frame_values.shape[-1] != len(FULL_RALLY_FEATURE_NAMES):
            raise ValueError("rally ablations must project from full-view windows")
        values, validity = view.select(window.frame_values, window.frame_validity)
        output.append(
            replace(
                window,
                frame_values=values,
                frame_validity=validity,
                metadata={**window.metadata, "feature_view": feature_view},
            )
        )
    return output


def perturbed_homography(
    calibration_path: Path,
    *,
    sigma_pixels: float,
    noise: Mapping[str, np.ndarray],
) -> CourtHomography:
    """Refit a homography after applying supplied endpoint perturbations."""
    if sigma_pixels <= 0:
        raise ValueError("landmark perturbation sigma must be positive")
    value = json.loads(calibration_path.read_text(encoding="utf-8"))
    image_lines = value.get("image_lines")
    if not isinstance(image_lines, Mapping) or set(image_lines) != set(noise):
        raise ValueError("sensitivity requires matching canonical image-line endpoints")
    perturbed: dict[str, list[list[float]]] = {}
    for name, endpoints in image_lines.items():
        points = np.asarray(endpoints, dtype=float)
        offsets = np.asarray(noise[name], dtype=float)
        if points.shape != (2, 2) or offsets.shape != (2, 2):
            raise ValueError("calibration sensitivity expects two 2D endpoints per line")
        perturbed[str(name)] = (points + sigma_pixels * offsets).tolist()
    homography, _, _ = CourtHomography.from_lines(perturbed)
    return homography


def reproject_full_windows(
    windows: Sequence[RallyWindow],
    homographies: Mapping[str, CourtHomography],
) -> list[RallyWindow]:
    """Change only derived court x/y features while keeping every other field fixed."""
    output: list[RallyWindow] = []
    for window in windows:
        if window.frame_values.shape[-1] != len(FULL_RALLY_FEATURE_NAMES):
            raise ValueError("sensitivity reprojection requires full-view windows")
        homography = homographies[window.source_id]
        values = window.frame_values.clone()
        validity = window.frame_validity.clone()
        for player_offset in (0, 11):
            foot_indices = (player_offset + 4, player_offset + 5)
            court_indices = (player_offset + 6, player_offset + 7)
            observed = validity[:, foot_indices[0]] & validity[:, foot_indices[1]]
            if not bool(observed.any()):
                continue
            feet = values[observed][:, list(foot_indices)].detach().cpu().numpy()
            image_size = window.metadata.get("image_size")
            if image_size is None:
                # Full-view feet are normalized; infer the canonical size from
                # metadata added by the dataset compiler.
                image_size = window.metadata.get("frame_size")
            if image_size is None:
                raise ValueError("sensitivity window metadata has no image size")
            width, height = map(float, image_size)
            feet[:, 0] *= width
            feet[:, 1] *= height
            projected = homography.project_to_court(feet)
            row_indices = torch.nonzero(observed, as_tuple=False).squeeze(-1)
            values[row_indices, court_indices[0]] = torch.tensor(
                projected[:, 0] / HALF_WIDTH, dtype=values.dtype
            )
            values[row_indices, court_indices[1]] = torch.tensor(
                projected[:, 1] / HALF_LENGTH, dtype=values.dtype
            )
        changed = values != window.frame_values
        allowed = torch.zeros_like(changed)
        allowed[:, [6, 7, 17, 18]] = True
        if bool((changed & ~allowed).any()) or not torch.equal(validity, window.frame_validity):
            raise AssertionError("calibration perturbation changed non-court features or validity")
        output.append(replace(window, frame_values=values, frame_validity=validity))
    return output


def _transition_frames(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[int]]:
    output: dict[tuple[str, str], list[int]] = {}
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_id"]), []).append(row)
    for source, source_rows in by_source.items():
        source_rows.sort(key=lambda row: int(row["frame"]))
        previous: bool | None = None
        for row in source_rows:
            current = bool(row["decoded_inplay"])
            if previous is not None and current != previous:
                kind = "start" if current else "end"
                output.setdefault((source, kind), []).append(int(row["frame"]))
            previous = current
    return output


def _boundary_displacements(
    baseline: Sequence[Mapping[str, Any]], perturbed: Sequence[Mapping[str, Any]]
) -> list[int]:
    original = _transition_frames(baseline)
    changed = _transition_frames(perturbed)
    distances: list[int] = []
    for key, frames in original.items():
        candidates = changed.get(key, [])
        if not candidates:
            continue
        distances.extend(min(abs(frame - candidate) for candidate in candidates) for frame in frames)
    return distances


def _sensitivity_seed(camera: str, view: str, sigma: float) -> int:
    digest = hashlib.sha256(f"{camera}:{view}:{sigma}".encode()).digest()
    return SEED + int.from_bytes(digest[:4], "big")


def run_landmark_sensitivity(
    dataset: RallyWindowDataset,
    camera_sources: Mapping[str, Sequence[str]],
    ablation_dir: Path,
    *,
    device: torch.device,
    samples: int = SENSITIVITY_SAMPLES,
) -> dict[str, Any]:
    """Perturb line endpoints without retraining or reassigning player slots."""
    if samples <= 0 or dataset.feature_view.name != "full":
        raise ValueError("invalid landmark sensitivity inputs")
    source_configs = {source.source_id: source for source in dataset.config.sources}
    output: dict[str, Any] = {}
    for fold in camera_held_out_folds(camera_sources):
        camera = fold.held_out_camera
        full_test = [
            window for window in dataset.windows if window.source_id in fold.test_source_ids
        ]
        camera_result: dict[str, Any] = {}
        for view_name in ("full", "calibrated_canonical"):
            checkpoint_path = (
                ablation_dir
                / view_name
                / f"fold-{fold.fold_id}-{camera}"
                / "rally-segmenter.pt"
            )
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = RallySegmenter.from_checkpoint(checkpoint).to(device)
            decoder = InPlayDecoderConfig(**checkpoint["decoder_configuration"])
            baseline_windows = project_rally_windows(full_test, view_name)
            baseline_raw = predict_rally_windows(
                model,
                baseline_windows,
                device=device,
                batch_size=int(checkpoint["training_recipe"]["batch_size"]),
                partition="sensitivity_baseline",
            )
            fps_by_source = {
                window.source_id: float(window.metadata["fps"]) for window in full_test
            }
            baseline = decode_prediction_rows(
                baseline_raw, fps_by_source=fps_by_source, config=decoder
            )
            baseline_probabilities = {
                (str(row["source_id"]), int(row["frame"])): float(row["inplay_probability"])
                for row in baseline
            }
            baseline_states = {
                (str(row["source_id"]), int(row["frame"])): bool(row["decoded_inplay"])
                for row in baseline
            }
            view_result: dict[str, Any] = {}
            for sigma in SENSITIVITY_NOISE_PIXELS:
                generator = np.random.default_rng(_sensitivity_seed(camera, view_name, sigma))
                probability_changes: list[float] = []
                boundary_displacements: list[int] = []
                flips = total_states = 0
                for _ in range(samples):
                    first_source = fold.test_source_ids[0]
                    first_value = json.loads(
                        source_configs[first_source].calibration_path.read_text(encoding="utf-8")
                    )
                    line_names = sorted(first_value.get("image_lines", {}))
                    if not line_names:
                        raise ValueError("sensitivity requires landmark-derived image_lines")
                    noise = {
                        name: generator.normal(0.0, 1.0, size=(2, 2)) for name in line_names
                    }
                    homographies = {
                        source_id: perturbed_homography(
                            source_configs[source_id].calibration_path,
                            sigma_pixels=sigma,
                            noise=noise,
                        )
                        for source_id in fold.test_source_ids
                    }
                    changed_full = reproject_full_windows(full_test, homographies)
                    changed_view = project_rally_windows(changed_full, view_name)
                    raw = predict_rally_windows(
                        model,
                        changed_view,
                        device=device,
                        batch_size=int(checkpoint["training_recipe"]["batch_size"]),
                        partition="sensitivity_perturbed",
                    )
                    decoded = decode_prediction_rows(raw, fps_by_source=fps_by_source, config=decoder)
                    for row in decoded:
                        key = (str(row["source_id"]), int(row["frame"]))
                        probability_changes.append(
                            abs(float(row["inplay_probability"]) - baseline_probabilities[key])
                        )
                        flips += bool(row["decoded_inplay"]) != baseline_states[key]
                        total_states += 1
                    boundary_displacements.extend(_boundary_displacements(baseline, decoded))
                view_result[f"sigma_{sigma:g}px"] = {
                    "seed": _sensitivity_seed(camera, view_name, sigma),
                    "perturbations": samples,
                    "mean_absolute_probability_change": statistics.fmean(probability_changes),
                    "p95_absolute_probability_change": float(np.quantile(probability_changes, 0.95)),
                    "threshold_state_flip_rate": flips / total_states,
                    "decoded_boundary_displacement_frames": {
                        "mean": statistics.fmean(boundary_displacements) if boundary_displacements else None,
                        "p95": float(np.quantile(boundary_displacements, 0.95)) if boundary_displacements else None,
                        "matched_boundary_count": len(boundary_displacements),
                    },
                }
            camera_result[view_name] = view_result
        output[camera] = camera_result
    return {
        "label": "input-projection sensitivity with fixed player-slot assignment",
        "seed": SEED,
        "samples_per_camera_noise_level": samples,
        "noise_pixels": list(SENSITIVITY_NOISE_PIXELS),
        "by_camera": output,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _view_dataset_fingerprint(dataset: RallyWindowDataset, view_name: str) -> str:
    view = RALLY_FEATURE_VIEWS[view_name]
    identity = {
        "base_dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
        "rally_feature_schema": dataset.manifest["rally_feature_schema"],
        "rally_feature_version": dataset.manifest["rally_feature_version"],
        "feature_view": view_name,
        "feature_names": view.names,
        "feature_dimension": view.dimension,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_rally_feature_ablations(
    dataset: RallyWindowDataset,
    camera_sources: Mapping[str, Sequence[str]],
    output_dir: Path,
    *,
    recipe: RallyTrainingRecipe | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train the four registered views on identical camera-held-out folds."""
    if dataset.feature_view.name != "full":
        raise ValueError("rally feature ablations require a full-view dataset")
    recipe = recipe or RallyTrainingRecipe()
    device = device or torch.device(recipe.device)
    _seed_everything(recipe.seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    folds = camera_held_out_folds(camera_sources)
    calibration_fingerprints = {
        source: str(value["calibration_sha256"])
        for source, value in dataset.manifest["sources"].items()
    }
    view_results: dict[str, Any] = {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    for view_name, view in RALLY_FEATURE_VIEWS.items():
        view_dir = output_dir / view_name
        view_dir.mkdir()
        projected = project_rally_windows(dataset.windows, view_name)
        by_camera: dict[str, Any] = {}
        all_test: list[dict[str, Any]] = []
        fold_results: list[dict[str, Any]] = []
        for fold in folds:
            _seed_everything(recipe.seed)
            fold_dir = view_dir / f"fold-{fold.fold_id}-{fold.held_out_camera}"
            fold_dir.mkdir()
            training_groups = {
                camera: tuple(map(str, sources))
                for camera, sources in camera_sources.items()
                if camera != fold.held_out_camera
            }
            eligible = [window for window in projected if window.source_id in fold.training_source_ids]
            train, validation, split = chronological_camera_train_validation_split(
                eligible, training_groups, guard_seconds=1.0
            )
            test = [window for window in projected if window.source_id in fold.test_source_ids]
            split_manifest = {
                "fold_id": fold.fold_id,
                "held_out_camera": fold.held_out_camera,
                "training_source_ids": list(fold.training_source_ids),
                "validation_source_ids": list(fold.validation_source_ids),
                "test_source_ids": list(fold.test_source_ids),
                "train_validation": split,
                "train_window_ids": [window.window_id for window in train],
                "validation_window_ids": [window.window_id for window in validation],
                "test_window_ids": [window.window_id for window in test],
                "checkpoint_selection": "validation_raw_bce_only",
                "decoder_selection": "validation_labels_only",
                "test_labels_used_for_selection": False,
            }
            model = RallySegmenter(
                RallySegmenterConfig(frame_feature_dim=view.dimension)
            ).to(device)
            history, epoch = train_rally_segmenter(
                model,
                train,
                validation,
                device=device,
                epochs=recipe.epochs,
                batch_size=recipe.batch_size,
                seed=recipe.seed,
                boundary_aux_weight=recipe.boundary_aux_weight,
            )
            validation_rows = predict_rally_windows(
                model,
                validation,
                device=device,
                batch_size=recipe.batch_size,
                partition="validation",
            )
            fps_by_source = {
                window.source_id: float(window.metadata["fps"])
                for window in (*validation, *test)
            }
            decoder, decoder_search = tune_rally_decoder(
                validation_rows, fps_by_source=fps_by_source
            )
            raw_test = predict_rally_windows(
                model,
                test,
                device=device,
                batch_size=recipe.batch_size,
                partition="test",
            )
            decoded = decode_prediction_rows(raw_test, fps_by_source=fps_by_source, config=decoder)
            metrics = rally_metrics(decoded)
            by_camera[fold.held_out_camera] = metrics
            all_test.extend(decoded)
            checkpoint = rally_checkpoint(
                model,
                dataset_fingerprint=_view_dataset_fingerprint(dataset, view_name),
                annotation_fingerprint=dataset.annotation_fingerprint,
                split_manifest=split_manifest,
                training_recipe=recipe,
                chosen_epoch=epoch,
                decoder_configuration=asdict(decoder),
                feature_view=view_name,
                calibration_fingerprints=calibration_fingerprints,
                extra={"epoch_history": history, "diagnostic_only": True},
            )
            torch.save(checkpoint, fold_dir / "rally-segmenter.pt")
            _write_jsonl(fold_dir / "test-predictions.jsonl", decoded)
            fold_result = {
                "split_manifest": split_manifest,
                "chosen_epoch": epoch,
                "decoder": asdict(decoder),
                "decoder_search": decoder_search,
                "test": metrics,
                "test_by_source": metrics_by_source(decoded),
            }
            _write_json(fold_dir / "metrics.json", fold_result)
            fold_results.append(fold_result)
        predictions[view_name] = sorted(all_test, key=lambda row: (row["source_id"], row["frame"]))
        view_results[view_name] = {
            "dataset_fingerprint": _view_dataset_fingerprint(dataset, view_name),
            "feature_names": list(view.names),
            "feature_dimension": view.dimension,
            "folds": fold_results,
            "by_camera": by_camera,
            "macro": _macro_metrics(by_camera),
        }
    comparisons = {
        "full_vs_calibrated_canonical": paired_bce_bootstrap(
            predictions["calibrated_canonical"], predictions["full"], samples=10_000, seed=SEED
        ),
        "calibrated_canonical_vs_no_position": paired_bce_bootstrap(
            predictions["no_position"], predictions["calibrated_canonical"], samples=10_000, seed=SEED
        ),
    }
    return {"views": view_results, "paired_bce_bootstrap": comparisons, "predictions": predictions}


def interpret_shortcut_evidence(probe: Mapping[str, Any], ablations: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the predeclared report-only shortcut criteria."""
    comparisons = {
        "image_position": ("image_position_only", "full", "calibrated_canonical", "full_vs_calibrated_canonical"),
        "court_position": ("court_only", "calibrated_canonical", "no_position", "calibrated_canonical_vs_no_position"),
    }
    output: dict[str, Any] = {}
    for name, (probe_group, with_group, without_group, bootstrap_name) in comparisons.items():
        with_macro = ablations["views"][with_group]["macro"]
        without_macro = ablations["views"][without_group]["macro"]
        bootstrap = ablations["paired_bce_bootstrap"][bootstrap_name]
        bce_improvement = float(with_macro["raw_bce"]) - float(without_macro["raw_bce"])
        roc_loss = float(with_macro["roc_auc"]) - float(without_macro["roc_auc"])
        interval_f1_loss = float(with_macro["interval_f1"]) - float(without_macro["interval_f1"])
        checks = {
            "substantial_domain_identity": bool(probe[probe_group]["substantial_domain_identity"]),
            "macro_bce_improves_by_at_least_0_01": bce_improvement >= 0.01,
            "paired_95_interval_excludes_zero": float(bootstrap["confidence_interval_95"][0]) > 0,
            "macro_roc_auc_loss_no_more_than_0_02": roc_loss <= 0.02,
            "macro_interval_f1_loss_no_more_than_0_05": interval_f1_loss <= 0.05,
        }
        harmful = all(checks.values())
        if harmful:
            conclusion = "harmful camera-domain shortcut evidence"
        elif checks["substantial_domain_identity"] and bce_improvement <= 0:
            conclusion = "domain-identifiable but useful geometric/player signal"
        elif checks["substantial_domain_identity"]:
            conclusion = "domain information available, model reliance not established"
        else:
            conclusion = "substantial camera-domain identity not established"
        output[name] = {
            "harmful_shortcut_evidence": harmful,
            "conclusion": conclusion,
            "checks": checks,
            "macro_bce_improvement_when_removed": bce_improvement,
            "macro_roc_auc_loss_when_removed": roc_loss,
            "macro_interval_f1_loss_when_removed": interval_f1_loss,
        }
    return output


def _fingerprint_report(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_diagnostic_report(
    dataset: RallyWindowDataset,
    camera_sources: Mapping[str, Sequence[str]],
    probe: Mapping[str, Any],
    ablations: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "baddievision_calibrated_camera_shortcut_diagnostic",
        "schema_version": 1,
        "seed": SEED,
        "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
        "annotation_fingerprint": dataset.annotation_fingerprint,
        "camera_sources": {camera: list(sources) for camera, sources in camera_sources.items()},
        "calibration_fingerprints": {
            source: value["calibration_sha256"] for source, value in dataset.manifest["sources"].items()
        },
        "camera_domain_probe": probe,
        "rally_feature_ablations": {
            "views": ablations["views"],
            "paired_bce_bootstrap": ablations["paired_bce_bootstrap"],
        },
        "landmark_sensitivity": sensitivity,
        "interpretation": interpret_shortcut_evidence(probe, ablations),
        "limitations": [
            "Camera identity is confounded with match, players, and venue; results are camera-domain shortcut evidence, not definitive calibration leakage.",
            "The evaluated cameras are diagnostic development data because their held-out labels were previously evaluated.",
            "Landmark sensitivity holds player-slot assignment fixed and therefore measures input-projection sensitivity only.",
            "Definitive confirmation requires synchronized independently calibrated viewpoints of the same complete rallies with paired cycles kept in the same split.",
        ],
        "production_action": None,
        "preferred_feature_schema_changed": False,
        "boundary_or_probability_calibration_pipeline_changed": False,
        "shuttle_features_introduced": False,
    }
    report["report_sha256"] = _fingerprint_report(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--probe-bootstrap-samples", type=int, default=PROBE_BOOTSTRAP_SAMPLES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from .rally_experiment import load_experiment_datasets

    args = _parser().parse_args(argv)
    dataset, _, camera_sources = load_experiment_datasets(args.config)
    output = prepare_output_directory(args.output_dir)
    probe = run_camera_probe(dataset, camera_sources, bootstrap_samples=args.probe_bootstrap_samples)
    ablations = run_rally_feature_ablations(
        dataset,
        camera_sources,
        output / "ablations",
        recipe=RallyTrainingRecipe(device=args.device),
        device=torch.device(args.device),
    )
    sensitivity = run_landmark_sensitivity(
        dataset,
        camera_sources,
        output / "ablations",
        device=torch.device(args.device),
    )
    report = build_diagnostic_report(dataset, camera_sources, probe, ablations, sensitivity)
    _write_json(output / "diagnostic-report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
