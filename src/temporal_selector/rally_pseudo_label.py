"""Boundary-aware, source-local rally pseudo labels for human review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .joint_inference import InPlayDecoderConfig, decode_inplay_probabilities, decoded_intervals
from .rally import RallySegmenter
from .rally_dataset import RallyDataConfig, RallySourceConfig, RallyWindow, RallyWindowDataset, collate_rally_windows
from .rally_features import FULL_RALLY_FEATURE_NAMES
from src.workflow.identity import SourceIdentity, sha256_file

SNAP_RADII_SECONDS = (0.0, 0.25, 0.5, 1.0)
SNAP_CONFIDENCE_FLOORS = (0.0, 0.25, 0.5, 0.75)
DEFAULT_MAJORITY_MAX_GAP_SECONDS = 0.75


def fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def bridge_short_majority_gaps(states: Sequence[bool], *, fps: float,
                               max_gap_seconds: float) -> tuple[bool, ...]:
    """Apply an asymmetric exit hold without delaying entry or extending true ends."""
    if fps <= 0 or max_gap_seconds < 0:
        raise ValueError("majority gap FPS must be positive and duration nonnegative")
    output = list(map(bool, states))
    maximum = round(fps * max_gap_seconds)
    if maximum == 0:
        return tuple(output)
    start = None
    for index, state in enumerate((*output, True)):
        if index < len(output) and not state and start is None:
            start = index
        elif state and start is not None:
            # Only close an interior gap: source edges and sustained exits stay put.
            if start > 0 and index < len(output) and index - start <= maximum:
                output[start:index] = [True] * (index - start)
            start = None
    return tuple(output)


def _peak(probabilities: Sequence[float], edge: int, radius: int, floor: float) -> int:
    left, right = max(0, edge - radius), min(len(probabilities) - 1, edge + radius)
    candidates = [frame for frame in range(left, right + 1) if probabilities[frame] >= floor]
    # Highest confidence, then nearest to the decoder edge, then earlier frame.
    return min(candidates, key=lambda frame: (-probabilities[frame], abs(frame - edge), frame)) if candidates else edge


def snap_intervals(
    intervals: Sequence[tuple[int, int]],
    start_probabilities: Sequence[float],
    end_probabilities: Sequence[float],
    *,
    fps: float,
    radius_seconds: float,
    confidence_floor: float,
    minimum_duration_seconds: float,
) -> tuple[tuple[int, int], ...]:
    """Snap decoder edges while deterministically retaining invalid original edges."""
    if len(start_probabilities) != len(end_probabilities):
        raise ValueError("start and end boundary probabilities must align")
    if fps <= 0 or radius_seconds < 0 or not 0 <= confidence_floor <= 1:
        raise ValueError("invalid boundary snap configuration")
    radius = round(radius_seconds * fps)
    minimum = max(1, round(minimum_duration_seconds * fps))
    originals = list(intervals)
    proposed: list[tuple[int, int]] = []
    for original_start, original_end in intervals:
        start = _peak(start_probabilities, original_start, radius, confidence_floor)
        end = _peak(end_probabilities, original_end, radius, confidence_floor)
        if end < start or end - start + 1 < minimum:
            start, end = original_start, original_end
        proposed.append((start, end))
    invalid: set[int] = set()
    for index in range(len(proposed) - 1):
        if originals[index][1] >= originals[index + 1][0]:
            raise ValueError("base decoder intervals overlap")
        if proposed[index][1] >= proposed[index + 1][0]:
            invalid.update((index, index + 1))
    return tuple(originals[index] if index in invalid else item
                 for index, item in enumerate(proposed))


def select_snap_configuration(
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    decoder: InPlayDecoderConfig,
    score: Any,
) -> dict[str, float]:
    """Freeze snapping on validation rows only; ``score`` receives snapped intervals."""
    probabilities = [float(row["inplay_probability"]) for row in validation_rows]
    starts = [float(row["rally_start_probability"]) for row in validation_rows]
    ends = [float(row["rally_end_probability"]) for row in validation_rows]
    base = decoded_intervals(
        list(range(len(probabilities))),
        decode_inplay_probabilities(probabilities, fps=fps, config=decoder),
    )
    candidates = []
    for radius in SNAP_RADII_SECONDS:
        for floor in SNAP_CONFIDENCE_FLOORS:
            snapped = snap_intervals(base, starts, ends, fps=fps, radius_seconds=radius,
                                     confidence_floor=floor,
                                     minimum_duration_seconds=decoder.minimum_duration_seconds)
            candidates.append((tuple(score(snapped)), -radius, -floor, radius, floor))
    best = max(candidates)
    return {"radius_seconds": best[-2], "confidence_floor": best[-1]}


def boundary_snap_diagnostic(
    validation_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    *,
    fps: float,
    decoder: InPlayDecoderConfig,
) -> dict[str, Any]:
    """Select on validation, then compare base and snapped decoding once on held-out rows."""
    from .rally_evaluation import rally_metrics

    def probability(row: Mapping[str, Any], name: str) -> float:
        key = f"rally_{name}_probability"
        if key in row:
            return float(row[key])
        return float(torch.sigmoid(torch.tensor(float(row[f"rally_{name}_logit"]))))

    def decoded(rows: Sequence[Mapping[str, Any]], snap: Mapping[str, float] | None) -> list[dict[str, Any]]:
        by_source: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            by_source.setdefault(str(row["source_id"]), []).append(row)
        output = []
        for source_rows in by_source.values():
            source_rows.sort(key=lambda row: int(row["frame"]))
            frames = [int(row["frame"]) for row in source_rows]
            if any(right != left + 1 for left, right in zip(frames, frames[1:])):
                raise ValueError("diagnostic rows must be contiguous within each source")
            probabilities = [float(row["inplay_probability"]) for row in source_rows]
            states = decode_inplay_probabilities(probabilities, fps=fps, config=decoder)
            intervals = decoded_intervals(list(range(len(source_rows))), states)
            if snap is not None:
                intervals = snap_intervals(
                    intervals,
                    [probability(row, "start") for row in source_rows],
                    [probability(row, "end") for row in source_rows],
                    fps=fps, radius_seconds=float(snap["radius_seconds"]),
                    confidence_floor=float(snap["confidence_floor"]),
                    minimum_duration_seconds=decoder.minimum_duration_seconds,
                )
            snapped_states = [False] * len(source_rows)
            for left, right in intervals:
                snapped_states[left:right + 1] = [True] * (right - left + 1)
            output.extend({**row, "decoded_inplay": state,
                           "decoder_configuration": as_decoder(decoder)}
                          for row, state in zip(source_rows, snapped_states))
        return sorted(output, key=lambda row: (str(row["source_id"]), int(row["frame"])))

    def summary(rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> dict[str, Any]:
        intervals = metrics["intervals"]
        splits, merges = _split_merge_counts(rows)
        return {
            "interval_f1": intervals["f1"],
            "mean_absolute_boundary_error": intervals["mean_absolute_boundary_error"],
            "frame_f1": metrics["decoded_frame"]["f1"],
            "splits": splits,
            "merges": merges,
        }

    def validation_score(snap: Mapping[str, float]) -> tuple[float, ...]:
        metrics = rally_metrics(decoded(validation_rows, snap))
        interval = metrics["intervals"]
        boundary = interval["mean_absolute_boundary_error"]
        return (float(interval["f1"]), -float(boundary if boundary is not None else math.inf),
                float(metrics["decoded_frame"]["f1"]))

    candidates = []
    for radius in SNAP_RADII_SECONDS:
        for floor in SNAP_CONFIDENCE_FLOORS:
            candidate = {"radius_seconds": radius, "confidence_floor": floor}
            candidates.append((validation_score(candidate), -radius, -floor, candidate))
    snap = max(candidates, key=lambda item: item[:3])[-1]
    baseline_rows = decoded(heldout_rows, None)
    snapped_rows = decoded(heldout_rows, snap)
    baseline = summary(baseline_rows, rally_metrics(baseline_rows))
    snapped = summary(snapped_rows, rally_metrics(snapped_rows))
    delta = {key: (None if baseline[key] is None or snapped[key] is None
                   else float(snapped[key]) - float(baseline[key])) for key in baseline}
    return {"decoder": as_decoder(decoder), "selected_snap": snap,
            "selection_partition": "validation_only", "evaluation_partition": "heldout_once",
            "heldout": {"unchanged_decoder": baseline, "boundary_snapped": snapped, "delta": delta}}


def as_decoder(decoder: InPlayDecoderConfig) -> dict[str, Any]:
    return {"threshold": decoder.threshold, "max_gap_seconds": decoder.max_gap_seconds,
            "minimum_duration_seconds": decoder.minimum_duration_seconds,
            "preserve_edge_runs": decoder.preserve_edge_runs}


def _split_merge_counts(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    def intervals(field: str) -> list[tuple[str, int, int]]:
        output = []
        for source in sorted({str(row["source_id"]) for row in rows}):
            source_rows = sorted((row for row in rows if str(row["source_id"]) == source),
                                 key=lambda row: int(row["frame"]))
            start = None
            for row in source_rows:
                state = bool(row["decoded_inplay"]) if field == "prediction" else int(row["inplay_target"]) == 1
                frame = int(row["frame"])
                if state and start is None:
                    start = frame
                elif not state and start is not None:
                    output.append((source, start, frame - 1))
                    start = None
            if start is not None:
                output.append((source, start, int(source_rows[-1]["frame"])))
        return output
    truth, predictions = intervals("truth"), intervals("prediction")
    def overlaps(item: tuple[str, int, int], other: Sequence[tuple[str, int, int]]) -> int:
        return sum(item[0] == value[0] and min(item[2], value[2]) >= max(item[1], value[1])
                   for value in other)
    return (sum(max(0, overlaps(item, predictions) - 1) for item in truth),
            sum(max(0, overlaps(item, truth) - 1) for item in predictions))


def combine_boundary_diagnostics(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Combine the three frozen held-out evaluations into per-camera and macro deltas."""
    if len(reports) != 3:
        raise ValueError("boundary diagnostic requires exactly three held-out cameras")
    deltas = {source: dict(report["heldout"]["delta"]) for source, report in reports.items()}
    keys = next(iter(deltas.values())).keys()
    macro = {
        key: (sum(float(values[key]) for values in deltas.values() if values[key] is not None)
              / sum(values[key] is not None for values in deltas.values()))
        if any(values[key] is not None for values in deltas.values()) else None
        for key in keys
    }
    return {"schema": "rally_boundary_snap_diagnostic", "schema_version": 1,
            "per_camera": dict(reports), "per_camera_deltas": deltas, "macro_delta": macro}


def build_teacher_manifest(entries: Sequence[Mapping[str, Any]], output: Path, *,
                           majority_max_gap_seconds: float = DEFAULT_MAJORITY_MAX_GAP_SECONDS) -> Path:
    """Write a portable, fingerprinted three-model annotation teacher."""
    if len(entries) != 3:
        raise ValueError("an annotation teacher requires exactly three checkpoints")
    normalized = []
    contracts = set()
    for item in entries:
        checkpoint_path = Path(item["checkpoint_path"]).expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        RallySegmenter.from_checkpoint(checkpoint)
        contract = (checkpoint["rally_feature_schema"], int(checkpoint["rally_feature_version"]),
                    checkpoint["rally_feature_view"], tuple(checkpoint["rally_feature_names"]))
        contracts.add(contract)
        if checkpoint["rally_feature_view"] != "full" or tuple(checkpoint["rally_feature_names"]) != FULL_RALLY_FEATURE_NAMES:
            raise ValueError("annotation teacher checkpoints must use the full rally feature view")
        if checkpoint.get("diagnostic_only") is not True:
            raise ValueError("annotation teacher checkpoints must remain diagnostic_only")
        normalized.append({
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "held_out_source_id": str(item["held_out_source_id"]),
            "decoder": dict(item.get("decoder", checkpoint["decoder_configuration"])),
            "snap": dict(item["snap"]),
        })
    if len(contracts) != 1:
        raise ValueError("annotation teacher checkpoints have different feature contracts")
    if majority_max_gap_seconds < 0:
        raise ValueError("teacher majority gap duration cannot be negative")
    payload = {"schema": "rally_annotation_teacher", "schema_version": 1, "models": normalized,
               "ensemble_decoder": {"entry_votes": 2, "exit_votes": 2,
                                    "maximum_interior_gap_seconds": majority_max_gap_seconds,
                                    "preserve_entry_frame": True, "preserve_sustained_exit_edge": True}}
    payload["fingerprint"] = fingerprint(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def generate_teacher_diagnostic(
    config_path: Path,
    checkpoint_root: Path,
    output_dir: Path,
    *,
    device: str = "cpu",
    batch_size: int = 16,
) -> tuple[Path, Path]:
    """Reproduce frozen validation predictions and build the real three-fold teacher."""
    from .rally_dataset import rally_data_config_from_selector_mapping
    from .rally_experiment import predict_rally_windows
    from .shortcut_diagnostic import _view_dataset_fingerprint

    config_path = config_path.expanduser().resolve()
    value = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = RallyWindowDataset(rally_data_config_from_selector_mapping(value, base_dir=config_path.parent))
    expected_dataset = _view_dataset_fingerprint(dataset, "full")
    expected_calibrations = {source: str(item["calibration_sha256"])
                             for source, item in dataset.manifest["sources"].items()}
    fold_dirs = sorted(path for path in checkpoint_root.expanduser().resolve().iterdir()
                       if path.is_dir() and (path / "rally-segmenter.pt").is_file())
    if len(fold_dirs) != 3:
        raise ValueError("teacher diagnostic requires exactly three checkpoint fold directories")
    if output_dir.exists():
        raise FileExistsError(f"teacher diagnostic output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    reports: dict[str, Any] = {}
    entries = []
    torch_device = torch.device(device)
    for fold_dir in fold_dirs:
        checkpoint_path = fold_dir / "rally-segmenter.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (checkpoint.get("dataset_fingerprint") != expected_dataset
                or checkpoint.get("annotation_fingerprint") != dataset.annotation_fingerprint
                or checkpoint.get("calibration_fingerprints") != expected_calibrations):
            raise ValueError(f"checkpoint dataset or calibration lineage is stale: {checkpoint_path}")
        validation_ids = set(map(str, checkpoint["split_manifest"]["validation_window_ids"]))
        validation = [window for window in dataset.windows if window.window_id in validation_ids]
        if {window.window_id for window in validation} != validation_ids:
            raise ValueError(f"checkpoint validation windows cannot be reproduced: {checkpoint_path}")
        model = RallySegmenter.from_checkpoint(checkpoint).to(torch_device)
        validation_rows = predict_rally_windows(model, validation, device=torch_device,
                                                 batch_size=batch_size, partition="validation")
        heldout_rows = [json.loads(line) for line in (fold_dir / "test-predictions.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        fps_values = {float(window.metadata["fps"]) for window in validation}
        if len(fps_values) != 1:
            raise ValueError("teacher diagnostic currently requires one common validation FPS")
        report = boundary_snap_diagnostic(
            validation_rows, heldout_rows, fps=fps_values.pop(),
            decoder=InPlayDecoderConfig(**checkpoint["decoder_configuration"]),
        )
        camera = str(checkpoint["split_manifest"]["held_out_camera"])
        report["checkpoint_sha256"] = sha256_file(checkpoint_path)
        report["validation_row_count"] = len(validation_rows)
        report["heldout_row_count"] = len(heldout_rows)
        reports[camera] = report
        fold_output = output_dir / fold_dir.name
        fold_output.mkdir()
        (fold_output / "validation-predictions.jsonl").write_text("".join(
            json.dumps(row, sort_keys=True) + "\n" for row in validation_rows), encoding="utf-8")
        (fold_output / "boundary-snap-diagnostic.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        entries.append({"checkpoint_path": checkpoint_path, "held_out_source_id": camera,
                        "decoder": checkpoint["decoder_configuration"], "snap": report["selected_snap"]})
    combined = combine_boundary_diagnostics(reports)
    diagnostic_path = output_dir / "boundary-snap-diagnostic.json"
    diagnostic_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    teacher_path = build_teacher_manifest(entries, output_dir / "teacher.manifest.json")
    return diagnostic_path, teacher_path


def _load_teacher(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop("fingerprint", None)
    if value.get("schema") != "rally_annotation_teacher" or len(value.get("models", ())) != 3:
        raise ValueError("invalid rally annotation teacher manifest")
    if claimed != fingerprint(value):
        raise ValueError("annotation teacher manifest fingerprint differs")
    value["fingerprint"] = claimed
    return value


def _predict(model: RallySegmenter, windows: Sequence[RallyWindow], device: torch.device,
             batch_size: int = 16) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    model.eval()
    for offset in range(0, len(windows), batch_size):
        group = windows[offset:offset + batch_size]
        batch = collate_rally_windows(group).to(device)
        with torch.no_grad():
            output = model(batch)
        for batch_index, window in enumerate(group):
            for local, frame in enumerate(window.frame_indices):
                if frame in window.owned_frames:
                    rows.append({
                        "frame": int(frame),
                        "inplay_probability": float(torch.sigmoid(output.inplay_logits[batch_index, local]).cpu()),
                        "rally_start_probability": float(torch.sigmoid(output.rally_start_logits[batch_index, local]).cpu()),
                        "rally_end_probability": float(torch.sigmoid(output.rally_end_logits[batch_index, local]).cpu()),
                    })
    rows.sort(key=lambda row: int(row["frame"]))
    return rows


def pseudo_label_source(source: RallySourceConfig, teacher_path: Path, *, device: str = "cpu",
                        output_dir: Path | None = None, batch_size: int = 16) -> tuple[Path, Path]:
    """Run three independent decoders and emit majority-vote review proposals."""
    teacher_path = teacher_path.expanduser().resolve()
    teacher = _load_teacher(teacher_path)
    dataset = RallyWindowDataset(RallyDataConfig(
        sources=(source,), rally_intervals_path=None, rally_manifest_path=None, feature_view="full"
    ))
    identity = SourceIdentity.read(source.calibration_path.parent.parent / "source.json")
    torch_device = torch.device(device)
    fps = identity.fps_numerator / identity.fps_denominator
    per_model = []
    checkpoint_hashes = []
    for model_entry in teacher["models"]:
        checkpoint_path = Path(model_entry["checkpoint_path"])
        if sha256_file(checkpoint_path) != model_entry["checkpoint_sha256"]:
            raise ValueError("teacher checkpoint fingerprint differs")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("rally_feature_view") != "full":
            raise ValueError("teacher checkpoint is not full-view")
        model = RallySegmenter.from_checkpoint(checkpoint).to(torch_device)
        raw = _predict(model, dataset.windows, torch_device, batch_size)
        if [int(row["frame"]) for row in raw] != list(range(identity.frame_count)):
            raise ValueError("rally inference predictions do not cover every source frame exactly once")
        decoder = InPlayDecoderConfig(**model_entry["decoder"])
        base = decoded_intervals(list(range(identity.frame_count)), decode_inplay_probabilities(
            [row["inplay_probability"] for row in raw], fps=fps, config=decoder))
        snap = model_entry["snap"]
        intervals = snap_intervals(base, [row["rally_start_probability"] for row in raw],
                                  [row["rally_end_probability"] for row in raw], fps=fps,
                                  radius_seconds=float(snap["radius_seconds"]),
                                  confidence_floor=float(snap["confidence_floor"]),
                                  minimum_duration_seconds=decoder.minimum_duration_seconds)
        states = [False] * identity.frame_count
        for left, right in intervals:
            states[left:right + 1] = [True] * (right - left + 1)
        per_model.append((raw, states))
        checkpoint_hashes.append(model_entry["checkpoint_sha256"])
    input_hashes = {"video": sha256_file(source.video_path), "calibration": sha256_file(source.calibration_path),
                    "assignments": sha256_file(source.assignments_path), "poses": sha256_file(source.pose_cache_path)}
    metadata = {"type": "metadata", "schema": "rally_pseudo_predictions", "schema_version": 1,
                "source_id": source.source_id, "frame_count": identity.frame_count,
                "frame_indexing": "source_local_zero_based", "teacher_fingerprint": teacher["fingerprint"],
                "checkpoint_sha256": checkpoint_hashes, "input_artifact_sha256": input_hashes,
                "ensemble_decoder": dict(teacher.get("ensemble_decoder", {}))}
    rows = []
    majority = []
    for frame in range(identity.frame_count):
        votes = sum(states[frame] for _, states in per_model)
        majority.append(votes >= 2)
        rows.append({"frame": frame, "models": [{**raw[frame], "decoded_inplay": states[frame]}
                                                 for raw, states in per_model],
                     "vote_count": votes, "pseudo_inplay": votes >= 2, "disagreement": votes not in (0, 3)})
    proposals = []
    ensemble = teacher.get("ensemble_decoder", {})
    if int(ensemble.get("entry_votes", 2)) != 2 or int(ensemble.get("exit_votes", 2)) != 2:
        raise ValueError("teacher ensemble vote contract is incompatible")
    majority = list(bridge_short_majority_gaps(
        majority, fps=fps,
        max_gap_seconds=float(ensemble.get("maximum_interior_gap_seconds", 0.0)),
    ))
    for frame, state in enumerate(majority):
        rows[frame]["pseudo_inplay"] = state
        rows[frame]["bridged_majority_gap"] = state and rows[frame]["vote_count"] < 2
    for number, (left, right) in enumerate(decoded_intervals(list(range(identity.frame_count)), majority), 1):
        votes = [rows[frame]["vote_count"] for frame in range(left, right + 1)]
        probabilities = [sum(model["inplay_probability"] for model in rows[frame]["models"]) / 3
                         for frame in range(left, right + 1)]
        proposals.append({"proposal_id": f"{source.source_id}-{number:04d}", "start_frame": left,
                          "end_frame": right, "agreement": "3/3" if min(votes) == 3 else "2/3",
                          "confidence": sum(probabilities) / len(probabilities),
                          "bridged_gap_frames": sum(rows[frame]["bridged_majority_gap"]
                                                    for frame in range(left, right + 1)),
                          "requires_partial_start_review": left == 0})
    proposal_payload = {"schema": "rally_pseudo_proposals", "schema_version": 1,
                        "source_id": source.source_id, "metadata": metadata, "proposals": proposals}
    proposal_payload["fingerprint"] = fingerprint(proposal_payload)
    output_dir = output_dir or source.calibration_path.parent.parent / "rallies"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path, proposals_path = output_dir / "predictions.jsonl", output_dir / "proposals.json"
    predictions_path.write_text(json.dumps(metadata, sort_keys=True) + "\n" + "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    proposals_path.write_text(json.dumps(proposal_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return predictions_path, proposals_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    diagnostic, teacher = generate_teacher_diagnostic(
        args.config, args.checkpoint_root, args.output_dir,
        device=args.device, batch_size=args.batch_size,
    )
    print(diagnostic)
    print(teacher)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
