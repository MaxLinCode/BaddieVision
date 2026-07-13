"""Lossless-above-threshold TrackNet shuttle evidence and conservative links."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


CANDIDATE_SCHEMA = "shuttle_candidates"
TRACKLET_SCHEMA = "shuttle_tracklets"
HYPOTHESES_SCHEMA = "shuttle_hypotheses"
CANDIDATE_SCHEMA_VERSION = 2
CANDIDATE_EXTRACTION_VERSION = "tracknet-components-v2.0"
CANDIDATE_THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
CANDIDATE_RETENTION_KS: tuple[int | None, ...] = (1, 2, 3, 5, 8, 12, None)
CANDIDATE_RETENTION_POLICY = (
    "peak_activation_desc",
    "mean_activation_desc",
    "area_normalized_desc",
    "candidate_id_asc",
)


def legacy_tracknet_bbox(heatmap: np.ndarray) -> tuple[int, int, int, int] | None:
    """Reproduce untouched TrackNet's threshold/contour/tie decision exactly."""
    mask = (np.asarray(heatmap) > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    rects = [cv2.boundingRect(contour) for contour in contours]
    largest = 0
    largest_area = rects[0][2] * rects[0][3]
    for index in range(1, len(rects)):
        area = rects[index][2] * rects[index][3]
        if area > largest_area:
            largest = index
            largest_area = area
    return tuple(int(value) for value in rects[largest])


def _is_sha256(value: str | None) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def tracknet_candidate_frame_range(
    frame_count: int,
    sequence_length: int,
    ensemble_mode: str,
) -> tuple[int, int]:
    """Validate a working video and return its expected inclusive frame range."""
    frame_count, sequence_length = int(frame_count), int(sequence_length)
    if frame_count <= 0:
        raise ValueError("cannot extract shuttle candidates from an empty or unreadable working video")
    if sequence_length <= 0:
        raise ValueError("TrackNet sequence_length must be positive")
    if ensemble_mode != "nonoverlap" and frame_count < sequence_length:
        raise ValueError(
            "TrackNet overlap candidate extraction requires at least one complete "
            f"sequence: frame_count={frame_count}, sequence_length={sequence_length}"
        )
    return 0, frame_count - 1


class ShuttleCandidateCollector:
    """Collect every deterministic component from each TrackNet output frame."""

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        heatmap_size: tuple[int, int],
        fps: float,
        thresholds: Sequence[float] = CANDIDATE_THRESHOLDS,
        threshold: float | None = None,
        checkpoint_sha256: str | None = None,
        inference_model_sha256: str | None = None,
        inference_model_artifact: str | None = None,
        tracknet_config: Mapping[str, Any] | None = None,
        overlap_ensemble_mode: str | None = None,
        source_frame_range: tuple[int, int] | None = None,
        allow_unverified_provenance_for_testing: bool = False,
    ):
        """Initialize a schema-v2 proposal collector.

        ``threshold`` is a compatibility alias for callers that intentionally
        want a single threshold. Production extraction uses ``thresholds`` and
        never truncates candidates during collection. ``source_frame_range``
        is the expected inclusive zero-based working-video range; writing fails
        unless every frame in that range was added, including empty frames.
        """
        if threshold is not None:
            thresholds = (threshold,)
        normalized_thresholds = tuple(sorted({float(value) for value in thresholds}))
        if not normalized_thresholds or any(not math.isfinite(value) or not 0 <= value <= 1 for value in normalized_thresholds):
            raise ValueError("candidate thresholds must be finite values between zero and one")
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError("image_size must contain positive width and height")
        if len(heatmap_size) != 2 or min(heatmap_size) <= 0:
            raise ValueError("heatmap_size must contain positive height and width")
        if not math.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError("fps must be positive")
        if source_frame_range is not None:
            source_frame_range = tuple(int(value) for value in source_frame_range)
            if source_frame_range[0] < 0 or source_frame_range[1] < source_frame_range[0]:
                raise ValueError("source_frame_range must be an inclusive, non-negative range")
        if not allow_unverified_provenance_for_testing:
            missing_hashes = []
            if not _is_sha256(checkpoint_sha256):
                missing_hashes.append("checkpoint_sha256")
            if not _is_sha256(inference_model_sha256):
                missing_hashes.append("inference_model_sha256")
            if missing_hashes:
                raise ValueError(
                    "schema-v2 candidate extraction requires verified SHA-256 provenance: "
                    + ", ".join(missing_hashes)
                )
        self.image_size = tuple(int(value) for value in image_size)
        self.heatmap_size = tuple(int(value) for value in heatmap_size)
        self.fps = float(fps)
        self.thresholds = normalized_thresholds
        self.checkpoint_sha256 = checkpoint_sha256
        self.inference_model_sha256 = inference_model_sha256
        self.inference_model_artifact = inference_model_artifact
        self.tracknet_config = dict(tracknet_config or {})
        self.overlap_ensemble_mode = overlap_ensemble_mode
        self.source_frame_range = source_frame_range
        self.allow_unverified_provenance_for_testing = bool(allow_unverified_provenance_for_testing)
        self.provenance_verified = _is_sha256(checkpoint_sha256) and _is_sha256(inference_model_sha256)
        self._frames: dict[int, list[dict]] = {}

    def add(self, frame: int, heatmap: np.ndarray) -> None:
        """Add an unthresholded, single-frame TrackNet heatmap exactly once."""
        frame = int(frame)
        if frame in self._frames:
            return
        heatmap = np.asarray(heatmap, dtype=np.float32)
        if heatmap.shape != self.heatmap_size:
            raise ValueError(f"expected heatmap {self.heatmap_size}, got {heatmap.shape}")
        heatmap_h, heatmap_w = self.heatmap_size
        image_w, image_h = self.image_size
        sx, sy = image_w / heatmap_w, image_h / heatmap_h
        heatmap_area = heatmap_h * heatmap_w
        components: list[dict[str, Any]] = []
        for threshold_value in self.thresholds:
            mask = (heatmap > threshold_value).astype(np.uint8)
            labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            threshold_components = []
            for label in range(1, labels_count):
                x, y, width, height, area = [int(value) for value in stats[label]]
                ys, xs = np.nonzero(labels == label)
                activations = heatmap[ys, xs].astype(np.float64)
                peak_offset = int(np.argmax(activations))
                peak_x, peak_y = int(xs[peak_offset]), int(ys[peak_offset])
                total_activation = float(activations.sum())
                weighted_x = float(np.dot(xs.astype(np.float64) + 0.5, activations) / total_activation)
                weighted_y = float(np.dot(ys.astype(np.float64) + 0.5, activations) / total_activation)
                threshold_components.append({
                    "threshold": threshold_value,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "area": area,
                    "peak_activation": float(activations[peak_offset]),
                    "mean_activation": float(activations.mean()),
                    "total_activation": total_activation,
                    "peak_heatmap": (peak_x + 0.5, peak_y + 0.5),
                    "weighted_heatmap": (weighted_x, weighted_y),
                    "legacy_largest_component": False,
                })
            # This ordering is explicit rather than relying on OpenCV label internals.
            threshold_components.sort(key=lambda item: (
                item["y"], item["x"], item["height"], item["width"],
                -item["area"], -item["peak_activation"],
            ))
            for component_index, component in enumerate(threshold_components):
                component["component_index"] = component_index
            if threshold_value == 0.5 and threshold_components:
                # Identify the component chosen by untouched TrackNet, including
                # OpenCV contour ordering for equal-area ties.
                legacy_bbox = legacy_tracknet_bbox(heatmap)
                legacy_index = next(
                    index for index, item in enumerate(threshold_components)
                    if (item["x"], item["y"], item["width"], item["height"]) == legacy_bbox
                )
                threshold_components[legacy_index]["legacy_largest_component"] = True
            components.extend(threshold_components)
        records = []
        for component in components:
            x, y = component["x"], component["y"]
            width, height, area = component["width"], component["height"], component["area"]
            peak_x, peak_y = component["peak_heatmap"]
            weighted_x, weighted_y = component["weighted_heatmap"]
            center = [float((x + width / 2) * sx), float((y + height / 2) * sy)]
            peak_position = [float(peak_x * sx), float(peak_y * sy)]
            weighted_centroid = [float(weighted_x * sx), float(weighted_y * sy)]
            bbox = [float(x * sx), float(y * sy), float((x + width) * sx), float((y + height) * sy)]
            records.append({
                "candidate_id": (
                    f"f{frame:06d}-t{int(round(component['threshold'] * 100)):03d}"
                    f"-c{component['component_index']:03d}"
                ),
                "threshold": component["threshold"],
                "center": center,
                "center_normalized": [center[0] / image_w, center[1] / image_h],
                "peak_position": peak_position,
                "peak_position_normalized": [peak_position[0] / image_w, peak_position[1] / image_h],
                "weighted_centroid": weighted_centroid,
                "weighted_centroid_normalized": [weighted_centroid[0] / image_w, weighted_centroid[1] / image_h],
                "bbox": bbox,
                "bbox_normalized": [bbox[0] / image_w, bbox[1] / image_h, bbox[2] / image_w, bbox[3] / image_h],
                "area": area,
                "area_normalized": area / heatmap_area,
                "peak_value": component["peak_activation"],
                "peak_activation": component["peak_activation"],
                "mean_activation": component["mean_activation"],
                "total_activation": component["total_activation"],
                "peak_activation_normalized": component["peak_activation"],
                "mean_activation_normalized": component["mean_activation"],
                "total_activation_normalized": component["total_activation"] / heatmap_area,
                "legacy_largest_component": component["legacy_largest_component"],
            })
        self._frames[frame] = records

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        image_w, image_h = self.image_size
        heatmap_h, heatmap_w = self.heatmap_size
        observed_range = [min(self._frames), max(self._frames)] if self._frames else None
        if self.source_frame_range is not None:
            expected_start, expected_end = self.source_frame_range
            missing = [
                frame for frame in range(expected_start, expected_end + 1)
                if frame not in self._frames
            ]
            unexpected = sorted(
                frame for frame in self._frames
                if frame < expected_start or frame > expected_end
            )
            if missing or unexpected:
                def summarize(values: list[int]) -> str:
                    preview = values[:10]
                    suffix = f" (+{len(values) - len(preview)} more)" if len(values) > len(preview) else ""
                    return f"{preview}{suffix}"

                raise ValueError(
                    "candidate frames do not completely cover expected inclusive working-video "
                    f"range [{expected_start}, {expected_end}]; "
                    f"missing={summarize(missing)}, unexpected={summarize(unexpected)}"
                )
        source_range = list(self.source_frame_range) if self.source_frame_range is not None else observed_range
        metadata = {
            "type": "metadata", "schema": CANDIDATE_SCHEMA, "schema_version": CANDIDATE_SCHEMA_VERSION,
            "model_stage": "tracknet_pre_inpaint", "thresholds": list(self.thresholds), "fps": self.fps,
            "image_size": [image_w, image_h], "heatmap_size": [heatmap_w, heatmap_h],
            "coordinate_scaling": {"x": image_w / heatmap_w, "y": image_h / heatmap_h},
            "connectivity": 8,
            "threshold_comparator": ">",
            "extraction_version": CANDIDATE_EXTRACTION_VERSION,
            "checkpoint_sha256": self.checkpoint_sha256,
            "inference_model_sha256": self.inference_model_sha256,
            "inference_model_artifact": self.inference_model_artifact,
            "provenance_verified": self.provenance_verified,
            "nonproduction_unverified_provenance": (
                self.allow_unverified_provenance_for_testing and not self.provenance_verified
            ),
            "tracknet_config": self.tracknet_config,
            "overlap_ensemble_mode": self.overlap_ensemble_mode,
            "source_frame_range": source_range,
            "source_frame_count": (
                source_range[1] - source_range[0] + 1 if source_range is not None else 0
            ),
            "source_frame_range_inclusive": True,
            "source_frame_index_space": "zero_based_working_video",
            "retention_policy": list(CANDIDATE_RETENTION_POLICY),
            "legacy_compatibility_threshold": 0.5 if 0.5 in self.thresholds else None,
            "pixel_position_convention": "heatmap_pixel_centers_scaled_to_image_space",
            "normalization": {
                "image_coordinates": "x/image_width,y/image_height",
                "area": "component_pixels/heatmap_pixels",
                "peak_activation": "identity",
                "mean_activation": "identity",
                "total_activation": "sum/heatmap_pixels",
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output:
            output.write(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n")
            for frame in sorted(self._frames):
                output.write(json.dumps({"type": "frame", "frame": frame, "candidates": self._frames[frame]}, sort_keys=True, separators=(",", ":")) + "\n")
        return path


def candidate_retention_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, str]:
    """Return the frozen, deterministic K-retention ordering key."""
    peak = float(candidate.get("peak_activation", candidate.get("peak_value", 0.0)))
    mean = float(candidate.get("mean_activation", peak))
    normalized_area = float(candidate.get("area_normalized", candidate.get("area", 0.0)))
    return -peak, -mean, -normalized_area, str(candidate["candidate_id"])


def rank_shuttle_candidates(candidates: Iterable[Mapping[str, Any]], k: int | None = None) -> list[dict[str, Any]]:
    """Rank a frame's proposals using the exact production retention policy."""
    if k is not None and k < 0:
        raise ValueError("k must be non-negative or None")
    ranked = sorted((dict(candidate) for candidate in candidates), key=candidate_retention_key)
    return ranked if k is None else ranked[:k]


@dataclass(frozen=True)
class ShuttleLinkConfig:
    max_missing_frames: int = 1
    max_speed_image_diagonals_per_second: float = 6.0
    ambiguity_ratio: float = 0.7
    candidate_threshold: float | None = 0.5


@dataclass(frozen=True)
class ShuttleHypothesisConfig:
    """Configuration for the diagnostic, tracklet-level association decoder."""

    max_gap_seconds: float = 0.5
    max_speed_image_diagonals_per_second: float = 6.0
    max_hypotheses_per_region: int = 5
    # The legal-edge graph is a DAG, but the number of paths in it can still
    # be exponential.  Keep decoding bounded; this is a diagnostic decoder,
    # not an exhaustive graph enumerator.
    max_path_candidates_per_region: int = 4096
    min_symmetric_node_difference: float = 0.25
    score_version: str = "tracknet_motion_v1"


def _read_jsonl(path: Path, schema: str, schema_versions: Sequence[int] = (1,)) -> tuple[dict, list[dict]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty artifact: {path}")
    metadata = json.loads(lines[0])
    if metadata.get("schema") != schema or metadata.get("schema_version") not in schema_versions:
        versions = "/".join(f"v{version}" for version in schema_versions)
        raise ValueError(f"expected {schema} schema {versions} artifact: {path}")
    return metadata, [json.loads(line) for line in lines[1:] if line.strip()]


def read_shuttle_candidates(path: str | Path) -> tuple[dict, list[dict]]:
    """Read either the legacy v1 or current v2 candidate artifact."""
    return _read_jsonl(Path(path), CANDIDATE_SCHEMA, (1, CANDIDATE_SCHEMA_VERSION))


def evaluate_candidate_retention_recall(
    candidate_path: str | Path,
    labels: Iterable[Mapping[str, Any]],
    *,
    ks: Sequence[int | None] = CANDIDATE_RETENTION_KS,
) -> dict[str, Any]:
    """Evaluate exact selected-candidate retention at each K.

    Labels must be resolved, one-per-frame records with ``frame`` and a
    ``label_kind``/``outcome`` of ``selected``, ``missing_proposal``,
    ``no_shuttle``, or ``unsure``. Selected records also carry ``candidate_id``.

    This deliberately measures whether the exact human-selected proposal
    survives the production K ranking. It is *not* a threshold-comparison
    metric: nested components for the same shuttle have different durable IDs.
    A threshold pilot must use human adjudication or spatial ground truth rather
    than treating IDs from different thresholds as equivalent.
    """
    metadata, frame_records = read_shuttle_candidates(candidate_path)
    by_frame = {
        int(record["frame"]): list(record.get("candidates", []))
        for record in frame_records
        if record.get("type") == "frame"
    }
    normalized_ks: list[int | None] = []
    for k in ks:
        if k is not None:
            k = int(k)
            if k < 1:
                raise ValueError("recall K values must be positive or None for all")
        if k not in normalized_ks:
            normalized_ks.append(k)

    hits = {k: 0 for k in normalized_ks}
    present_frames = missing_frames = no_shuttle_frames = unsure_frames = 0
    seen_frames: set[int] = set()
    for label in labels:
        frame = int(label["frame"])
        if frame in seen_frames:
            raise ValueError(f"multiple resolved recall labels for frame {frame}")
        seen_frames.add(frame)
        kind = str(label.get("label_kind", label.get("outcome", ""))).lower()
        if kind in {"no_shuttle", "unsure"}:
            if kind == "no_shuttle":
                no_shuttle_frames += 1
            else:
                unsure_frames += 1
            continue
        if kind in {"missing_proposal", "missing"}:
            present_frames += 1
            missing_frames += 1
            continue
        if kind not in {"selected", "candidate"}:
            raise ValueError(f"unsupported recall label kind for frame {frame}: {kind!r}")
        candidates = by_frame.get(frame)
        if candidates is None:
            raise ValueError(f"label references frame absent from candidate artifact: {frame}")
        selected_id = str(label.get("candidate_id", ""))
        candidate_ids = {str(candidate["candidate_id"]) for candidate in candidates}
        if selected_id not in candidate_ids:
            raise ValueError(f"selected candidate {selected_id!r} is not present at frame {frame}")
        present_frames += 1
        ranked_ids = [str(candidate["candidate_id"]) for candidate in rank_shuttle_candidates(candidates)]
        selected_rank = ranked_ids.index(selected_id) + 1
        for k in normalized_ks:
            if k is None or selected_rank <= k:
                hits[k] += 1

    recall_at_k = {
        "all" if k is None else str(k): (hits[k] / present_frames if present_frames else None)
        for k in normalized_ks
    }
    candidate_counts = [len(candidates) for candidates in by_frame.values()]
    return {
        "candidate_schema_version": metadata["schema_version"],
        "retention_policy": list(CANDIDATE_RETENTION_POLICY),
        "label_equivalence": "exact_candidate_id",
        "threshold_comparison_valid": False,
        "present_shuttle_frames": present_frames,
        "missing_proposal_frames": missing_frames,
        "no_shuttle_frames": no_shuttle_frames,
        "unsure_frames": unsure_frames,
        "recall_at_k": recall_at_k,
        "candidates_per_frame": (
            sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
        ),
        "maximum_candidates_per_frame": max(candidate_counts, default=0),
    }


def _unambiguous_best(distance: float, alternatives: Iterable[float], ratio: float) -> bool:
    next_best = min(alternatives, default=None)
    return next_best is None or (distance < next_best and distance <= ratio * next_best)


def link_shuttle_candidates(candidate_path: str | Path, tracklet_path: str | Path, config: ShuttleLinkConfig | None = None) -> Path:
    """Link components conservatively using the legacy 0.5 view by default.

    Schema v2 deliberately contains nested proposals from four thresholds.
    Feeding all of those near-duplicates into the legacy diagnostic linker can
    multiply tracklets and hypothesis paths, so the default compatibility view
    links only threshold 0.5. Set ``candidate_threshold=None`` explicitly to
    experiment with linking the complete proposal set.
    """
    config = config or ShuttleLinkConfig()
    candidate_path, tracklet_path = Path(candidate_path), Path(tracklet_path)
    metadata, frames = read_shuttle_candidates(candidate_path)
    fps = float(metadata["fps"])
    width, height = (float(value) for value in metadata["image_size"])
    motion_gate_per_frame = config.max_speed_image_diagonals_per_second * math.hypot(width, height) / fps
    candidates_by_frame: dict[int, list[dict]] = {}
    for frame_record in frames:
        if frame_record.get("type") != "frame":
            raise ValueError("candidate artifact contains a non-frame record")
        frame = int(frame_record["frame"])
        for candidate in frame_record.get("candidates", []):
            if (
                metadata.get("schema_version") == CANDIDATE_SCHEMA_VERSION
                and config.candidate_threshold is not None
                and not math.isclose(
                    float(candidate.get("threshold", math.nan)),
                    config.candidate_threshold,
                    rel_tol=0,
                    abs_tol=1e-9,
                )
            ):
                continue
            candidates_by_frame.setdefault(frame, []).append(candidate)

    tracklets: list[dict] = []
    active: dict[str, dict] = {}
    next_tracklet = 0

    def start(frame: int, candidate: dict) -> None:
        nonlocal next_tracklet
        tracklet_id = f"t{next_tracklet:06d}"
        next_tracklet += 1
        item = {"type": "tracklet", "tracklet_id": tracklet_id, "candidate_ids": [candidate["candidate_id"]], "frames": [frame]}
        tracklets.append(item)
        active[tracklet_id] = {"item": item, "points": [(frame, candidate["center"])]}

    for frame in sorted(candidates_by_frame):
        current = candidates_by_frame[frame]
        active = {key: state for key, state in active.items() if frame - state["points"][-1][0] <= config.max_missing_frames + 1}
        distances: dict[tuple[str, int], float] = {}
        for tracklet_id, state in active.items():
            points = state["points"]
            last_frame, last = points[-1]
            delta = frame - last_frame
            predicted = list(last)
            if len(points) >= 2:
                previous_frame, previous = points[-2]
                observed_delta = last_frame - previous_frame
                if observed_delta:
                    predicted = [last[axis] + (last[axis] - previous[axis]) * delta / observed_delta for axis in (0, 1)]
            for index, candidate in enumerate(current):
                distance = math.dist(predicted, candidate["center"])
                if distance <= motion_gate_per_frame * delta:
                    distances[(tracklet_id, index)] = distance

        accepted: list[tuple[str, int]] = []
        for (tracklet_id, index), distance in distances.items():
            track_alternatives = [value for (other_tracklet, other_index), value in distances.items() if other_tracklet == tracklet_id and other_index != index]
            candidate_alternatives = [value for (other_tracklet, other_index), value in distances.items() if other_index == index and other_tracklet != tracklet_id]
            best_for_track = min((value for (other_tracklet, _), value in distances.items() if other_tracklet == tracklet_id), default=math.inf)
            best_for_candidate = min((value for (_, other_index), value in distances.items() if other_index == index), default=math.inf)
            if distance == best_for_track == best_for_candidate and _unambiguous_best(distance, track_alternatives, config.ambiguity_ratio) and _unambiguous_best(distance, candidate_alternatives, config.ambiguity_ratio):
                accepted.append((tracklet_id, index))

        accepted_tracks = {tracklet_id for tracklet_id, _ in accepted}
        # A tracklet with gated candidates but no uniquely safe continuation ends here.
        ambiguous_tracks = {tracklet_id for tracklet_id, _ in distances} - accepted_tracks
        for tracklet_id in ambiguous_tracks:
            active.pop(tracklet_id, None)
        used = set()
        for tracklet_id, index in accepted:
            candidate = current[index]
            state = active[tracklet_id]
            state["item"]["candidate_ids"].append(candidate["candidate_id"])
            state["item"]["frames"].append(frame)
            state["points"].append((frame, candidate["center"]))
            used.add(index)
        for index, candidate in enumerate(current):
            if index not in used:
                start(frame, candidate)

    fingerprint = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    output_metadata = {
        "type": "metadata", "schema": TRACKLET_SCHEMA, "schema_version": 1,
        "candidate_artifact": candidate_path.name, "candidate_sha256": fingerprint,
        "link_config": asdict(config),
        "candidate_view": {
            "schema_version": metadata.get("schema_version"),
            "threshold": config.candidate_threshold if metadata.get("schema_version") == CANDIDATE_SCHEMA_VERSION else None,
            "purpose": "v1_compatibility" if metadata.get("schema_version") == CANDIDATE_SCHEMA_VERSION else "legacy_v1",
        },
    }
    tracklet_path.parent.mkdir(parents=True, exist_ok=True)
    with tracklet_path.open("w", encoding="utf-8") as output:
        output.write(json.dumps(output_metadata, sort_keys=True, separators=(",", ":")) + "\n")
        for item in tracklets:
            output.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
    return tracklet_path


def link_shuttle_hypotheses(
    candidate_path: str | Path,
    tracklet_path: str | Path,
    hypotheses_path: str | Path,
    config: ShuttleHypothesisConfig | None = None,
) -> Path:
    """Decode diversified, replayable shuttle association hypotheses.

    This intentionally consumes only immutable candidate/tracklet evidence.  It
    does not alter the legacy TrackNet CSV path or attempt camera-specific
    ballistic modelling.
    """
    config = config or ShuttleHypothesisConfig()
    if config.max_gap_seconds < 0 or config.max_speed_image_diagonals_per_second <= 0:
        raise ValueError("hypothesis gap and speed limits must be positive")
    if config.max_hypotheses_per_region < 1:
        raise ValueError("max_hypotheses_per_region must be at least one")
    if config.max_path_candidates_per_region < config.max_hypotheses_per_region:
        raise ValueError("max_path_candidates_per_region must cover the requested hypotheses")
    if not 0 <= config.min_symmetric_node_difference <= 1:
        raise ValueError("min_symmetric_node_difference must be between zero and one")

    candidate_path, tracklet_path, hypotheses_path = map(Path, (candidate_path, tracklet_path, hypotheses_path))
    candidate_metadata, candidate_frames = read_shuttle_candidates(candidate_path)
    tracklet_metadata, tracklet_records = _read_jsonl(tracklet_path, TRACKLET_SCHEMA)
    candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if tracklet_metadata.get("candidate_artifact") != candidate_path.name or tracklet_metadata.get("candidate_sha256") != candidate_hash:
        raise ValueError("tracklet artifact does not reference this exact candidate artifact")

    fps = float(candidate_metadata["fps"])
    width, height = (float(value) for value in candidate_metadata["image_size"])
    speed_gate = config.max_speed_image_diagonals_per_second * math.hypot(width, height) / fps
    candidates = {
        item["candidate_id"]: (int(frame_record["frame"]), item)
        for frame_record in candidate_frames
        if frame_record.get("type") == "frame"
        for item in frame_record.get("candidates", [])
    }

    nodes: list[dict] = []
    for record in tracklet_records:
        if record.get("type") != "tracklet":
            raise ValueError("tracklet artifact contains a non-tracklet record")
        candidate_ids = list(record.get("candidate_ids", []))
        frames = [int(frame) for frame in record.get("frames", [])]
        if not candidate_ids or len(candidate_ids) != len(frames) or frames != sorted(frames) or len(set(frames)) != len(frames):
            raise ValueError(f"invalid tracklet geometry: {record.get('tracklet_id')}")
        points = []
        for candidate_id, frame in zip(candidate_ids, frames):
            candidate_frame, candidate = candidates.get(candidate_id, (None, None))
            if candidate is None or candidate_frame != frame:
                raise ValueError(f"tracklet references missing or mismatched candidate: {candidate_id}")
            points.append((frame, tuple(float(value) for value in candidate["center"])))
        nodes.append({
            "tracklet_id": record["tracklet_id"], "candidate_ids": candidate_ids, "frames": frames,
            "points": points,
            "tracknet_confidence": sum(float(candidates[item][1]["peak_value"]) for item in candidate_ids) / len(candidate_ids),
        })
    nodes.sort(key=lambda node: (node["frames"][0], node["frames"][-1], node["tracklet_id"]))
    if len({node["tracklet_id"] for node in nodes}) != len(nodes):
        raise ValueError("tracklet IDs must be unique")

    edges: dict[int, list[tuple[int, dict]]] = {index: [] for index in range(len(nodes))}
    for source_index, source in enumerate(nodes):
        tail_frame, tail_point = source["points"][-1]
        if len(source["points"]) > 1:
            previous_frame, previous_point = source["points"][-2]
            elapsed = tail_frame - previous_frame
            velocity = tuple((tail_point[axis] - previous_point[axis]) / elapsed for axis in (0, 1))
        else:
            velocity = (0.0, 0.0)
        for target_index, target in enumerate(nodes):
            head_frame, head_point = target["points"][0]
            if head_frame <= tail_frame:
                continue
            delta_frames = head_frame - tail_frame
            gap_seconds = (delta_frames - 1) / fps
            if gap_seconds > config.max_gap_seconds:
                continue
            predicted = tuple(tail_point[axis] + velocity[axis] * delta_frames for axis in (0, 1))
            distance = math.dist(predicted, head_point)
            gate = speed_gate * delta_frames
            if distance > gate:
                continue
            motion_compatibility = max(0.0, 1.0 - distance / gate)
            gap_penalty = gap_seconds / config.max_gap_seconds if config.max_gap_seconds else 0.0
            edge_score = motion_compatibility * (1.0 - gap_penalty)
            edges[source_index].append((target_index, {
                "motion_compatibility": motion_compatibility,
                "gap_penalty": gap_penalty,
                "edge_score": edge_score,
            }))
    for values in edges.values():
        values.sort(key=lambda item: (nodes[item[0]]["frames"][0], nodes[item[0]]["tracklet_id"]))

    # Connected components of the legal-edge graph are independent association regions.
    adjacent = {index: set() for index in range(len(nodes))}
    for source_index, values in edges.items():
        for target_index, _ in values:
            adjacent[source_index].add(target_index)
            adjacent[target_index].add(source_index)
    regions: list[list[int]] = []
    unseen = set(range(len(nodes)))
    while unseen:
        pending = [min(unseen)]
        region = []
        while pending:
            index = pending.pop()
            if index not in unseen:
                continue
            unseen.remove(index)
            region.append(index)
            pending.extend(sorted(adjacent[index] & unseen, reverse=True))
        regions.append(sorted(region, key=lambda item: (nodes[item]["frames"][0], nodes[item]["tracklet_id"])))

    def score_path(path: list[int], path_edges: list[dict]) -> tuple[float, dict]:
        confidence = sum(nodes[index]["tracknet_confidence"] for index in path) / len(path)
        motion = sum(edge["motion_compatibility"] for edge in path_edges) / len(path_edges) if path_edges else 0.0
        gap = sum(edge["gap_penalty"] for edge in path_edges) / len(path_edges) if path_edges else 0.0
        edge_score = sum(edge["edge_score"] for edge in path_edges) / len(path_edges) if path_edges else 0.0
        total = confidence if not path_edges else (confidence + edge_score) / 2
        return total, {
            "tracknet_confidence": confidence,
            "motion_compatibility": motion,
            "gap_penalty": gap,
            "edge_score": edge_score,
            "player_contact": None,
        }

    hypothesis_records = []
    region_metadata = []
    for region_number, region in enumerate(regions):
        region_id = f"r{region_number:04d}"
        region_set = set(region)
        candidates_for_region: list[tuple[float, list[int], list[dict], dict]] = []
        # Exhaustively enumerating paths is unsafe here: a region with only a
        # few plausible alternatives can contain tens of millions of paths.
        # Seed every singleton so isolated/weak evidence remains observable,
        # then use a deterministic beam per starting node for longer paths.
        for node in region:
            total, breakdown = score_path([node], [])
            candidates_for_region.append((total, [node], [], breakdown))

        beam_width = max(32, config.max_hypotheses_per_region * 8)
        path_budget = config.max_path_candidates_per_region - len(region)
        for node in region:
            if path_budget <= 0:
                break
            frontier = [([node], [])]
            while frontier and path_budget > 0:
                expanded: list[tuple[float, list[int], list[dict], dict]] = []
                for path, path_edges in frontier:
                    for target, edge in edges[path[-1]]:
                        if target not in region_set:
                            continue
                        extended_path = path + [target]
                        extended_edges = path_edges + [edge]
                        total, breakdown = score_path(extended_path, extended_edges)
                        expanded.append((total, extended_path, extended_edges, breakdown))
                if not expanded:
                    break
                expanded.sort(key=lambda item: (
                    -item[0], -len(item[1]),
                    tuple(nodes[index]["tracklet_id"] for index in item[1]),
                ))
                expanded = expanded[:min(beam_width, path_budget)]
                candidates_for_region.extend(expanded)
                path_budget -= len(expanded)
                frontier = [(item[1], item[2]) for item in expanded]
        candidates_for_region.sort(key=lambda item: (-item[0], -len(item[1]), tuple(nodes[index]["tracklet_id"] for index in item[1])))
        retained: list[tuple[float, list[int], list[dict], dict]] = []
        for candidate in candidates_for_region:
            candidate_nodes = set(candidate[1])
            if all(
                len(candidate_nodes.symmetric_difference(set(existing[1]))) / max(len(candidate_nodes), len(existing[1])) >= config.min_symmetric_node_difference
                for existing in retained
            ):
                retained.append(candidate)
                if len(retained) == config.max_hypotheses_per_region:
                    break
        region_metadata.append({
            "region_id": region_id,
            "tracklet_ids": [nodes[index]["tracklet_id"] for index in region],
            "frame_start": min(nodes[index]["frames"][0] for index in region),
            "frame_end": max(nodes[index]["frames"][-1] for index in region),
        })
        for rank, (total, path, _, breakdown) in enumerate(retained, start=1):
            hypothesis_records.append({
                "type": "hypothesis", "region_id": region_id, "rank": rank,
                "tracklet_ids": [nodes[index]["tracklet_id"] for index in path],
                "candidate_ids": [candidate_id for index in path for candidate_id in nodes[index]["candidate_ids"]],
                "frame_start": nodes[path[0]]["frames"][0], "frame_end": nodes[path[-1]]["frames"][-1],
                "total_score": total, "score_breakdown": breakdown,
            })

    hypotheses_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "metadata", "schema": HYPOTHESES_SCHEMA, "schema_version": 1,
        "candidate_artifact": candidate_path.name, "candidate_sha256": candidate_hash,
        "tracklet_artifact": tracklet_path.name, "tracklet_sha256": hashlib.sha256(tracklet_path.read_bytes()).hexdigest(),
        "decoder_config": asdict(config), "score_version": config.score_version,
        "association_regions": region_metadata,
    }
    with hypotheses_path.open("w", encoding="utf-8") as output:
        output.write(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n")
        for record in hypothesis_records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return hypotheses_path
