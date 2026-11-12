"""Label-independent rally inference and prediction-only video artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import torch

from src.workflow.identity import SourceIdentity, sha256_file

from .joint_inference import (
    InPlayDecoderConfig,
    decode_inplay_probabilities,
    decoded_intervals,
)
from .rally import RallySegmenter
from .rally_dataset import (
    RallyDataConfig,
    RallySourceConfig,
    RallyWindow,
    RallyWindowDataset,
    collate_rally_windows,
)

PROBABILITY_FIELDS = (
    "inplay_probability", "rally_start_probability", "rally_end_probability",
)


def predict_windows(
    model: RallySegmenter,
    windows: Sequence[RallyWindow],
    device: torch.device | str = "cpu",
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Predict owned frames, including frames with no annotation targets.

    Ownership is the only selection mask. Training targets and shuttle data are
    deliberately never inspected. This helper also serves annotation teachers.
    """
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("inference batch size must be positive")
    if len({window.source_id for window in windows}) > 1:
        raise ValueError("owned-frame inference requires one source at a time")
    model = model.to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for offset in range(0, len(windows), batch_size):
            group = windows[offset:offset + batch_size]
            batch = collate_rally_windows(group).to(device)
            output = model(batch)
            logits = (
                output.inplay_logits,
                output.rally_start_logits,
                output.rally_end_logits,
            )
            if any(not bool(torch.isfinite(item[batch.frame_mask]).all()) for item in logits):
                raise ValueError("rally model produced nonfinite real-frame logits")
            probabilities = [torch.sigmoid(item).cpu().tolist() for item in logits]
            for batch_index, window in enumerate(group):
                if not window.owned_frames.issubset(window.frame_indices):
                    raise ValueError("rally ownership extends beyond temporal context")
                for local, frame in enumerate(window.frame_indices):
                    if frame in window.owned_frames:
                        rows.append({
                            "frame": int(frame),
                            **{
                                field: float(values[batch_index][local])
                                for field, values in zip(PROBABILITY_FIELDS, probabilities)
                            },
                        })
    rows.sort(key=lambda row: row["frame"])
    frames = [row["frame"] for row in rows]
    if len(frames) != len(set(frames)):
        raise ValueError("rally inference has duplicate owned frames")
    return rows


def _decoder(value: Mapping[str, Any] | InPlayDecoderConfig) -> InPlayDecoderConfig:
    config = value if isinstance(value, InPlayDecoderConfig) else InPlayDecoderConfig(**value)
    if any(not math.isfinite(float(number)) for number in (
        config.threshold, config.max_gap_seconds, config.minimum_duration_seconds,
    )):
        raise ValueError("rally decoder settings must be finite")
    if not isinstance(config.preserve_edge_runs, bool):
        raise TypeError("preserve_edge_runs must be boolean")
    return config


def _identity(source: RallySourceConfig) -> SourceIdentity:
    identity = SourceIdentity.read(source.calibration_path.parent.parent / "source.json")
    if identity.source_id != source.source_id or identity.video_sha256 != sha256_file(source.video_path):
        raise ValueError("source identity does not match inference video")
    return identity


def _validate_rows(rows: Sequence[Mapping[str, Any]], frame_count: int) -> None:
    if [row.get("frame") for row in rows] != list(range(frame_count)):
        raise ValueError("rally inference predictions must cover every source frame exactly once")
    for row in rows:
        for field in PROBABILITY_FIELDS:
            value = float(row[field])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field} must be finite and in [0, 1]")


def predict_source(
    source: RallySourceConfig,
    model: RallySegmenter,
    *,
    preprocessing: Mapping[str, Any],
    decoder: Mapping[str, Any] | InPlayDecoderConfig,
    device: torch.device | str = "cpu",
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Compile unlabeled observations, infer every frame, and apply a frozen decoder."""
    config = _decoder(decoder)
    identity = _identity(source)
    keys = (
        "pose_visibility_threshold", "pose_coordinate_mode", "context_seconds",
        "ownership_seconds", "feature_view", "cached_court_tolerance",
    )
    settings = {key: preprocessing[key] for key in keys if key in preprocessing}
    # Release manifests may use the existing checkpoint's view key.
    if "rally_feature_view" in preprocessing and "feature_view" not in settings:
        settings["feature_view"] = preprocessing["rally_feature_view"]
    for key in ("pose_visibility_threshold", "context_seconds", "ownership_seconds", "cached_court_tolerance"):
        if key in settings and not math.isfinite(float(settings[key])):
            raise ValueError(f"rally preprocessing {key} must be finite")
    dataset = RallyWindowDataset(RallyDataConfig(
        sources=(source,), rally_intervals_path=None, rally_manifest_path=None, **settings,
    ))
    if model.config.frame_feature_dim != dataset.feature_view.dimension:
        raise ValueError("rally model dimension differs from preprocessing feature view")
    rows = predict_windows(model, dataset.windows, device=device, batch_size=batch_size)
    _validate_rows(rows, identity.frame_count)
    fps = identity.fps_numerator / identity.fps_denominator
    states = decode_inplay_probabilities(
        [row["inplay_probability"] for row in rows], fps=fps, config=config,
    )
    for row, state in zip(rows, states):
        row["decoded_inplay"] = state
        row["rally_id"] = None
    for number, (start, end) in enumerate(decoded_intervals(list(range(len(rows))), states), 1):
        for frame in range(start, end + 1):
            rows[frame]["rally_id"] = f"{source.source_id}-{number:04d}"
    return rows


def _render_overlay(
    source: RallySourceConfig,
    rows: Sequence[Mapping[str, Any]],
    identity: SourceIdentity,
    path: Path,
) -> None:
    """Render every video frame using predictions and observed player boxes only."""
    with source.assignments_path.open(encoding="utf-8") as handle:
        assignments = {}
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if value.get("type") != "metadata" and "frame" in value:
                    assignments[int(value["frame"])] = value
    fps = identity.fps_numerator / identity.fps_denominator
    capture = cv2.VideoCapture(str(source.video_path))
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (identity.width, identity.height),
    )
    try:
        if not capture.isOpened() or not writer.isOpened():
            raise RuntimeError("cannot open video input or MP4 overlay encoder")
        for index, row in enumerate(rows):
            ok, image = capture.read()
            if not ok or image.shape[:2] != (identity.height, identity.width):
                raise ValueError(f"video ended early or changed geometry at frame {index}")
            for role, slot in assignments.get(index, {}).get("slots", {}).items():
                assignment = slot.get("assignment")
                if not isinstance(assignment, dict) or not assignment.get("bbox"):
                    continue
                x1, y1, x2, y2 = [round(value) for value in assignment["bbox"]]
                color = (255, 190, 0) if role == "P1" else (0, 190, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, str(role), (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1)
            state = "IN PLAY" if row["decoded_inplay"] else "OUT OF PLAY"
            lines = (
                f"Frame {index} | {index / fps:.2f}s | {state}",
                f"P(InPlay) {float(row['inplay_probability']):.3f} | Rally {row.get('rally_id') or '-'}",
            )
            cv2.rectangle(image, (0, 0), (identity.width - 1, min(59, identity.height - 1)), (20, 20, 20), -1)
            for number, text in enumerate(lines):
                cv2.putText(image, text, (8, 22 + 24 * number), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            writer.write(image)
        if capture.read()[0]:
            raise ValueError("video contains more frames than source identity")
    finally:
        writer.release()
        capture.release()
    verification = cv2.VideoCapture(str(path))
    try:
        if (
            not verification.isOpened()
            or int(verification.get(cv2.CAP_PROP_FRAME_COUNT)) != identity.frame_count
            or not math.isclose(verification.get(cv2.CAP_PROP_FPS), fps, rel_tol=0, abs_tol=1e-3)
        ):
            raise RuntimeError("overlay frame count or FPS verification failed")
    finally:
        verification.release()


def write_prediction_artifacts(
    source: RallySourceConfig,
    rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    overlay: bool = True,
) -> dict[str, Path]:
    """Stage a complete prediction bundle before atomically replacing each file."""
    identity = _identity(source)
    _validate_rows(rows, identity.frame_count)
    if any(not isinstance(row.get("decoded_inplay"), bool) for row in rows):
        raise ValueError("prediction artifacts require boolean decoded states")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    intervals = decoded_intervals(
        list(range(identity.frame_count)), [row["decoded_inplay"] for row in rows],
    )
    header = {
        **dict(metadata or {}), "type": "metadata", "schema": "rally_predictions",
        "schema_version": 1, "source_id": source.source_id,
        "frame_count": identity.frame_count, "frame_indexing": "source_local_zero_based",
        "source_identity_sha256": identity.fingerprint,
        "input_artifact_sha256": {
            "video": identity.video_sha256, "calibration": sha256_file(source.calibration_path),
            "assignments": sha256_file(source.assignments_path), "poses": sha256_file(source.pose_cache_path),
        },
    }
    files = {"predictions": "predictions.jsonl", "rallies": "rallies.csv"}
    if overlay:
        files["overlay"] = "overlay.mp4"
    with TemporaryDirectory(prefix=".rally-output-", dir=directory) as temporary:
        staging = Path(temporary)
        with (staging / files["predictions"]).open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(header, allow_nan=False) + "\n")
            for row in rows:
                handle.write(json.dumps(dict(row), allow_nan=False) + "\n")
        with (staging / files["rallies"]).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("source_id", "rally_id", "start_frame", "end_frame"))
            writer.writeheader()
            for number, (start, end) in enumerate(intervals, 1):
                writer.writerow({
                    "source_id": source.source_id, "rally_id": f"{source.source_id}-{number:04d}",
                    "start_frame": start, "end_frame": end,
                })
        if overlay:
            _render_overlay(source, rows, identity, staging / files["overlay"])
        for name in files.values():
            (staging / name).replace(directory / name)
    return {key: directory / name for key, name in files.items()}


def decoder_mapping(decoder: Mapping[str, Any] | InPlayDecoderConfig) -> dict[str, Any]:
    """Normalize and validate a frozen decoder for provenance output."""
    return asdict(_decoder(decoder))
