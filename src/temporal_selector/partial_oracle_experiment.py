"""Matched raw-candidate versus oracle-candidate retraining on dense annotations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.annotation_platform.events import AnnotationEvent, replay_events

from .batch import MASKED_TARGET
from .config import SelectorConfig
from .crossfit import CrossFitFold
from .dataset import FRAME_DIMS, SelectorWindow, SelectorWindowDataset
from .experiment import (
    _binary_roc_auc,
    _predict_windows,
    _seed_everything,
    _train_fold,
    aggregate_joint_predictions,
    compile_metrics,
    load_dataset_config,
    prepare_output_directory,
    select_device,
)
from .model import JointRallyShuttleModel
from .within_camera_decoder import calibrate_validation_decoder, decode_rows
from .within_camera_experiment import DEFAULT_SOURCE_ORDER, _select_threshold

MATCHED_ANNOTATION_SHA256 = (
    "8c19f8d7cebc458fcd1b1138fd503231c9a3f3bea551e807e07e8beb5792c4d5"
)
MATCHED_FRAME_RANGE = (5796, 6968)
MATCHED_SPLIT_FRAME = 6513


def _active_events(path: Path) -> dict[tuple[str, int], AnnotationEvent]:
    events: list[AnnotationEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("type"):
            continue
        events.append(AnnotationEvent.from_mapping(value))
    return {
        (event.source_id, event.frame): event
        for event in replay_events(events).active.values()
    }


def _contiguous_runs(frames: Sequence[int]) -> list[tuple[int, int]]:
    runs: list[list[int]] = []
    for frame in sorted(set(map(int, frames))):
        if not runs or frame != runs[-1][-1] + 1:
            runs.append([])
        runs[-1].append(frame)
    return [(run[0], run[-1]) for run in runs]


def _eligible_windows(
    windows: Sequence[SelectorWindow],
    *,
    source_id: str,
    annotated_frames: set[int],
) -> list[SelectorWindow]:
    return [
        window
        for window in windows
        if window.source_id == source_id
        and set(window.frame_indices) <= annotated_frames
    ]


def _negative_split_frame(
    dataset: SelectorWindowDataset,
    source_id: str,
    start_frame: int,
    end_frame: int,
    *,
    train_fraction: float = 0.6,
) -> int:
    if dataset.rally_index is None:
        raise ValueError("partial oracle experiment requires InPlay intervals")
    desired = start_frame + round((end_frame - start_frame + 1) * train_fraction)
    negatives = [
        frame
        for frame in range(start_frame, end_frame + 1)
        if dataset.rally_index.target(source_id, frame) == 0
    ]
    runs = _contiguous_runs(negatives)
    candidates = [
        ((start + end) // 2, end - start + 1)
        for start, end in runs
        if end - start + 1 >= 60
    ]
    if not candidates:
        raise ValueError("annotated run has no one-second negative split region")
    return min(candidates, key=lambda item: (abs(item[0] - desired), -item[1]))[0]


def oracle_candidate_windows(
    windows: Sequence[SelectorWindow],
    events: Mapping[tuple[str, int], AnnotationEvent],
) -> list[SelectorWindow]:
    """Retain only the human-selected candidate and remove selection supervision."""
    output: list[SelectorWindow] = []
    for window in windows:
        keep_slots: list[int] = []
        for slot, local_frame in enumerate(window.candidate_frame_indices.tolist()):
            frame = window.frame_indices[int(local_frame)]
            event = events.get((window.source_id, frame))
            if (
                event is not None
                and event.label_kind == "selected"
                and window.candidate_ids[slot] == event.candidate_id
            ):
                keep_slots.append(slot)
        selected_frames = {
            window.frame_indices[local]
            for local in range(len(window.frame_indices))
            if (event := events.get((window.source_id, window.frame_indices[local])))
            is not None
            and event.label_kind == "selected"
        }
        retained_frames = {
            window.frame_indices[int(window.candidate_frame_indices[slot])]
            for slot in keep_slots
        }
        missing = sorted(selected_frames - retained_frames)
        if missing:
            raise ValueError(
                f"oracle-selected candidates are absent from frozen window: "
                f"{window.source_id}:{missing[:5]}"
            )
        index = torch.tensor(keep_slots, dtype=torch.long)
        output.append(
            replace(
                window,
                candidate_values=window.candidate_values[index],
                candidate_validity=window.candidate_validity[index],
                candidate_frame_indices=window.candidate_frame_indices[index],
                candidate_ids=tuple(window.candidate_ids[slot] for slot in keep_slots),
                targets=torch.full_like(window.targets, MASKED_TARGET),
                target_status=tuple("oracle_candidate_input" for _ in window.frame_indices),
                metadata={**window.metadata, "candidate_input_mode": "oracle_selected_only"},
            )
        )
    return output


def _variant_metrics(
    rows: list[dict[str, Any]],
    *,
    fps: float,
) -> dict[str, Any]:
    raw = compile_metrics(copy.deepcopy(rows))
    threshold, balanced_accuracy = _select_threshold(rows)
    threshold_rows = copy.deepcopy(rows)
    for row in threshold_rows:
        row["decoded_inplay"] = float(row["in_play_probability"]) >= threshold
    decoder, candidates = calibrate_validation_decoder(rows, fps=fps)
    decoded = decode_rows(copy.deepcopy(rows), fps=fps, config=decoder)
    labels = [
        int(row["inplay_target"]) for row in rows if row.get("inplay_target") in (0, 1)
    ]
    scores = [
        float(row["in_play_probability"])
        for row in rows
        if row.get("inplay_target") in (0, 1)
    ]
    return {
        "raw_0_5": raw,
        "roc_auc": _binary_roc_auc(labels, scores),
        "selector_posterior": _selector_posterior_metrics(rows),
        "selected_threshold": threshold,
        "selected_threshold_balanced_accuracy": balanced_accuracy,
        "selected_threshold_metrics": compile_metrics(threshold_rows),
        "decoder_config": asdict(decoder),
        "decoder_metrics": compile_metrics(decoded),
        "decoder_search_candidate_count": len(candidates),
    }


def _selector_posterior_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize candidate-plus-null confidence and candidate-presence slices."""
    summaries: list[dict[str, Any]] = []
    for row in rows:
        candidate_ids = list(row.get("candidate_ids", ()))
        logits = [
            float(row["candidate_logits"][candidate_id])
            for candidate_id in candidate_ids
        ] + [float(row["null_logit"])]
        maximum = max(logits)
        weights = [math.exp(value - maximum) for value in logits]
        total = sum(weights)
        probabilities = [value / total for value in weights]
        ordered = sorted(probabilities, reverse=True)
        summaries.append(
            {
                "null_probability": probabilities[-1],
                "entropy_nats": -sum(
                    value * math.log(value) for value in probabilities if value > 0
                ),
                "top1_margin": (
                    ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
                ),
                "candidates_present": bool(candidate_ids),
                "routes_null": probabilities[-1] == ordered[0],
                "inplay_target": row.get("inplay_target"),
            }
        )

    def summarize(values: Sequence[Mapping[str, float | bool]]) -> dict[str, Any]:
        return {
            "frame_count": len(values),
            "mean_null_probability": (
                sum(float(value["null_probability"]) for value in values) / len(values)
                if values else None
            ),
            "mean_entropy_nats": (
                sum(float(value["entropy_nats"]) for value in values) / len(values)
                if values else None
            ),
            "mean_top1_margin": (
                sum(float(value["top1_margin"]) for value in values) / len(values)
                if values else None
            ),
            "candidate_route_rate": (
                sum(not bool(value["routes_null"]) for value in values) / len(values)
                if values else None
            ),
            "null_route_rate": (
                sum(bool(value["routes_null"]) for value in values) / len(values)
                if values else None
            ),
        }

    return {
        "all": summarize(summaries),
        "candidates_present": summarize(
            [value for value in summaries if value["candidates_present"]]
        ),
        "zero_candidates": summarize(
            [value for value in summaries if not value["candidates_present"]]
        ),
        "inplay_negative": summarize(
            [value for value in summaries if value["inplay_target"] == 0]
        ),
        "inplay_positive": summarize(
            [value for value in summaries if value["inplay_target"] == 1]
        ),
    }


def _run_variant(
    *,
    name: str,
    train: Sequence[SelectorWindow],
    validation: Sequence[SelectorWindow],
    model_config: SelectorConfig,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    oracle: bool,
    conditioning_mode: str,
    force_candidate_routing: bool = False,
    candidate_input_mode: str | None = None,
) -> dict[str, Any]:
    _seed_everything(seed)
    model = JointRallyShuttleModel(
        model_config,
        boundary_heads=True,
        conditioning_mode=conditioning_mode,
        force_candidate_routing=force_candidate_routing,
    ).to(device)
    fold = CrossFitFold(name, (train[0].source_id,), (validation[0].source_id,))
    history = _train_fold(
        model,
        train,
        validation,
        fold,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        candidate_dropout=0.0 if oracle else 0.2,
        selection_weight=0.0 if oracle else 1.0,
        evaluation_owned_only=True,
        sampling_mode="boundary_balanced",
        checkpoint_selection="validation_loss",
        validation_partition_designated=True,
        boundary_aux_weight=0.25,
    )
    model.eval()
    rows = aggregate_joint_predictions(
        _predict_windows(
            model,
            validation,
            fold,
            device=device,
            batch_size=batch_size,
            owned_only=True,
        )
    )
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = variant_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "JointRallyShuttleModel",
            "selector_config": asdict(model_config),
            "boundary_heads": True,
            "conditioning_mode": conditioning_mode,
            "force_candidate_routing": force_candidate_routing,
            "boundary_aux_weight": 0.25,
            "sampling_mode": "boundary_balanced",
            "checkpoint_selection": model.checkpoint_selection,
            "candidate_input_mode": candidate_input_mode or (
                "oracle_selected_only" if oracle else "all_frozen"
            ),
            "selection_weight": 0.0 if oracle else 1.0,
            "candidate_dropout": 0.0 if oracle else 0.2,
            "selection_supervision": (
                "conditional_on_inplay; out-of-play frames unsupervised; "
                "selector null probability is not an InPlay prediction"
            ),
            "epoch_mean_losses": history,
        },
        checkpoint,
    )
    (variant_dir / "validation-predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metrics = _variant_metrics(
        rows, fps=float(validation[0].metadata["fps"])
    )
    metrics["checkpoint_selection"] = model.checkpoint_selection
    metrics["conditioning_mode"] = conditioning_mode
    metrics["force_candidate_routing"] = force_candidate_routing
    metrics["epoch_mean_losses"] = history
    metrics["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (variant_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def _verified_reference_variant(root: Path, name: str) -> dict[str, Any]:
    variant = root / name
    metrics = json.loads((variant / "metrics.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256((variant / "model.pt").read_bytes()).hexdigest()
    recorded = metrics.get("checkpoint_sha256")
    if recorded != actual:
        raise ValueError(
            f"reference {name} checkpoint hash mismatch: recorded {recorded}, got {actual}"
        )
    return {**metrics, "artifact_dir": str(variant), "checkpoint_sha256": actual}


def _acceptance_summary(
    variants: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    control_bce = float(variants["control"]["raw_0_5"]["mean_inplay_loss"])
    oracle_bce = float(
        variants["oracle_content_isolated"]["raw_0_5"]["mean_inplay_loss"]
    )
    gap = control_bce - oracle_bce
    presence_bce = float(
        variants["presence_isolated"]["raw_0_5"]["mean_inplay_loss"]
    )
    results: dict[str, Any] = {}
    for mode in ("hard_shared", "hard_isolated"):
        raw = variants[mode]["raw_0_5"]
        bce = float(raw["mean_inplay_loss"])
        selector_accuracy = raw["retained_target_accuracy"]["value"]
        boundary_mae = raw["intervals"]["mean_absolute_boundary_error"]
        gap_closed = (control_bce - bce) / gap if gap > 0 else None
        results[mode] = {
            "inplay_bce": bce,
            "control_to_oracle_gap_closed": gap_closed,
            "bce_target_met": bce <= 0.196 and gap_closed is not None and gap_closed >= 0.70,
            "raw_boundary_mae_frames": boundary_mae,
            "boundary_target_met": boundary_mae is not None and boundary_mae <= 17,
            "retained_target_accuracy": selector_accuracy,
            "selector_guardrail_met": (
                selector_accuracy is not None and selector_accuracy >= 0.939
            ),
            "bce_improvement_over_presence_only": presence_bce - bce,
            "materially_outperforms_presence_only": presence_bce - bce >= 0.01,
        }
        results[mode]["accepted"] = all(
            results[mode][key]
            for key in (
                "bce_target_met",
                "boundary_target_met",
                "selector_guardrail_met",
                "materially_outperforms_presence_only",
            )
        )
    recommended = min(results, key=lambda mode: results[mode]["inplay_bce"])
    return {
        "control_inplay_bce": control_bce,
        "oracle_inplay_bce": oracle_bce,
        "control_to_oracle_gap": gap,
        "presence_isolated_inplay_bce": presence_bce,
        "oracle_presence_isolated_inplay_bce": float(
            variants["oracle_presence_isolated"]["raw_0_5"]["mean_inplay_loss"]
        ),
        "modes": results,
        "recommended_mode": recommended,
        "production_direction_accepted": bool(results[recommended]["accepted"]),
    }


def run_partial_oracle_experiment(
    *,
    config_path: Path,
    baseline_validation_path: Path,
    output_dir: Path,
    epochs: int = 25,
    batch_size: int = 16,
    seed: int = 1729,
    device: torch.device | None = None,
    require_matched_snapshot: bool = True,
    reference_run_dir: Path | None = None,
) -> dict[str, Any]:
    if require_matched_snapshot and (epochs, batch_size, seed) != (25, 16, 1729):
        raise ValueError(
            "matched diagnostic requires epochs=25, batch_size=16, and seed=1729"
        )
    device = device or select_device()
    if require_matched_snapshot and device.type != "mps":
        raise ValueError(
            "matched diagnostic requires host MPS; use "
            "--allow-unmatched-snapshot for CPU smoke tests"
        )
    if require_matched_snapshot and reference_run_dir is None:
        raise ValueError("matched diagnostic requires --reference-run-dir")
    output_dir = prepare_output_directory(output_dir)
    annotation_source = Path(
        json.loads(config_path.read_text(encoding="utf-8"))["dataset"]["annotations_path"]
    )
    if not annotation_source.is_absolute():
        annotation_source = (config_path.parent / annotation_source).resolve()
    snapshot = output_dir / "annotations.snapshot.jsonl"
    snapshot.write_bytes(annotation_source.read_bytes())
    annotation_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if require_matched_snapshot and annotation_sha != MATCHED_ANNOTATION_SHA256:
        raise ValueError(
            "selection-conditioning diagnostic requires annotation snapshot "
            f"{MATCHED_ANNOTATION_SHA256}, got {annotation_sha}"
        )
    events = _active_events(snapshot)

    config, _ = load_dataset_config(
        config_path,
        context_mode="full_context",
        pose_coordinate_mode="player_relative",
    )
    allowed = set(DEFAULT_SOURCE_ORDER)
    config = replace(
        config,
        sources=tuple(source for source in config.sources if source.source_id in allowed),
        annotations_path=snapshot,
        expected_annotation_sha256=annotation_sha,
    )
    dataset = SelectorWindowDataset(config)
    baseline_rows = [
        json.loads(line)
        for line in baseline_validation_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_counts = Counter(str(row["prediction_source_id"]) for row in baseline_rows)
    source_id, _ = source_counts.most_common(1)[0]
    baseline_frames = {
        int(row["frame"])
        for row in baseline_rows
        if row["prediction_source_id"] == source_id
    }
    compatible_frames = {
        frame
        for event_source, frame in events
        if event_source == source_id and frame in baseline_frames
    }
    runs = _contiguous_runs(sorted(compatible_frames))
    if not runs:
        raise ValueError("baseline validation partition has no compatible annotations")
    run_start, run_end = max(runs, key=lambda item: item[1] - item[0])
    annotated_frames = set(range(run_start, run_end + 1))
    eligible = _eligible_windows(
        dataset.windows, source_id=source_id, annotated_frames=annotated_frames
    )
    split_frame = _negative_split_frame(
        dataset, source_id, run_start, run_end
    )
    if require_matched_snapshot and (
        (run_start, run_end) != MATCHED_FRAME_RANGE
        or split_frame != MATCHED_SPLIT_FRAME
    ):
        raise ValueError(
            "selection-conditioning diagnostic partition drifted from "
            f"frames {MATCHED_FRAME_RANGE[0]}-{MATCHED_FRAME_RANGE[1]} "
            f"with split {MATCHED_SPLIT_FRAME}; got frames "
            f"{run_start}-{run_end} with split {split_frame}"
        )
    train = [window for window in eligible if max(window.frame_indices) < split_frame]
    validation = [
        window for window in eligible if min(window.frame_indices) > split_frame
    ]
    if not train or not validation:
        raise ValueError("partial oracle split produced an empty partition")
    oracle_train = oracle_candidate_windows(train, events)
    oracle_validation = oracle_candidate_windows(validation, events)

    model_config = SelectorConfig(
        context_mode="full_context",
        frame_feature_dim=FRAME_DIMS["full_context"],
    )
    references = (
        {
            name: _verified_reference_variant(reference_run_dir, name)
            for name in ("control", "soft_joint", "oracle")
        }
        if reference_run_dir is not None
        else {}
    )
    control = references.get("control")
    if control is None:
        control = _run_variant(
            name="control",
            train=train,
            validation=validation,
            model_config=model_config,
            output_dir=output_dir,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            oracle=False,
            conditioning_mode="none",
        )
    conditioned = {}
    for conditioning_mode in (
        "hard_shared",
        "hard_isolated",
        "presence_isolated",
    ):
        conditioned[conditioning_mode] = _run_variant(
            name=conditioning_mode,
            train=train,
            validation=validation,
            model_config=model_config,
            output_dir=output_dir,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            oracle=False,
            conditioning_mode=conditioning_mode,
        )
    oracle_content = _run_variant(
        name="oracle_content_isolated",
        train=oracle_train,
        validation=oracle_validation,
        model_config=model_config,
        output_dir=output_dir,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        oracle=True,
        conditioning_mode="hard_isolated",
        force_candidate_routing=True,
    )
    oracle_presence = _run_variant(
        name="oracle_presence_isolated",
        train=oracle_train,
        validation=oracle_validation,
        model_config=model_config,
        output_dir=output_dir,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        oracle=True,
        conditioning_mode="presence_isolated",
    )
    variants = {
        "control": control,
        "verified_reference_artifacts": references,
        **conditioned,
        "oracle_content_isolated": oracle_content,
        "oracle_presence_isolated": oracle_presence,
    }
    acceptance = _acceptance_summary(variants)
    summary = {
        "schema": "selection_conditioned_inplay_diagnostic",
        "schema_version": 3,
        "annotation_snapshot_sha256": annotation_sha,
        "dataset_fingerprint": dataset.manifest["dataset_fingerprint"],
        "source_id": source_id,
        "longest_annotated_run": [run_start, run_end],
        "compatible_annotated_frame_count": len(compatible_frames),
        "eligible_window_count": len(eligible),
        "split_frame": split_frame,
        "train_window_count": len(train),
        "validation_window_count": len(validation),
        "train_owned_frame_count": sum(len(window.owned_frames) for window in train),
        "validation_owned_frame_count": sum(
            len(window.owned_frames) for window in validation
        ),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "boundary_heads": True,
        "boundary_aux_weight": 0.25,
        "sampling_mode": "boundary_balanced",
        "legacy_presence_only_interpretation": "post_encoder_presence_ablation",
        "selection_supervision": (
            "conditional_on_inplay; out-of-play frames unsupervised; "
            "selector null probability is not an InPlay prediction"
        ),
        "control": control,
        **conditioned,
        "oracle_content_isolated": oracle_content,
        "oracle_presence_isolated": oracle_presence,
        "acceptance": acceptance,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-run-dir",
        type=Path,
        help="existing control/soft_joint/oracle run whose checkpoint hashes are verified",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--allow-unmatched-snapshot",
        action="store_true",
        help="disable the frozen snapshot/frame/split guard for smoke tests only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = None if args.device == "auto" else torch.device(args.device)
    metrics = run_partial_oracle_experiment(
        config_path=args.config,
        baseline_validation_path=args.baseline_validation,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        require_matched_snapshot=not args.allow_unmatched_snapshot,
        reference_run_dir=args.reference_run_dir,
    )
    print(json.dumps({
        "control": metrics["control"]["decoder_metrics"]["inplay"],
        "hard_shared": metrics["hard_shared"]["raw_0_5"]["inplay"],
        "hard_isolated": metrics["hard_isolated"]["raw_0_5"]["inplay"],
        "presence_isolated": metrics["presence_isolated"]["raw_0_5"]["inplay"],
        "oracle_content_isolated": metrics["oracle_content_isolated"]["raw_0_5"]["inplay"],
        "oracle_presence_isolated": metrics["oracle_presence_isolated"]["raw_0_5"]["inplay"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
