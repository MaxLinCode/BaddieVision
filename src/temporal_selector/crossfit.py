"""Source-disjoint crossfit manifests and out-of-source prediction checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CrossFitFold:
    fold_id: str
    training_source_ids: tuple[str, ...]
    evaluation_source_ids: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.fold_id

    @property
    def train_sources(self) -> tuple[str, ...]:
        return self.training_source_ids

    @property
    def predict_sources(self) -> tuple[str, ...]:
        return self.evaluation_source_ids


@dataclass(frozen=True)
class CrossFitManifest:
    seed: int
    folds: tuple[CrossFitFold, ...]
    fingerprint: str


def build_two_source_crossfit(
    source_ids: Sequence[str], *, seed: int = 1729
) -> CrossFitManifest:
    sources = tuple(dict.fromkeys(map(str, source_ids)))
    if len(sources) != 2:
        raise ValueError("two-source cross-fitting requires exactly two unique sources")
    folds = (
        CrossFitFold("A", (sources[0],), (sources[1],)),
        CrossFitFold("B", (sources[1],), (sources[0],)),
    )
    payload = {"seed": int(seed), "folds": [vars(fold) for fold in folds]}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CrossFitManifest(int(seed), folds, fingerprint)


def build_leave_one_source_out(
    source_ids: Sequence[str], *, seed: int = 1729
) -> CrossFitManifest:
    """Build deterministic folds holding out exactly one complete source each."""
    sources = tuple(dict.fromkeys(map(str, source_ids)))
    if len(sources) < 2:
        raise ValueError("leave-one-source-out requires at least two unique sources")
    folds = tuple(
        CrossFitFold(
            chr(ord("A") + index),
            tuple(source for source in sources if source != held_out),
            (held_out,),
        )
        for index, held_out in enumerate(sources)
    )
    payload = {"seed": int(seed), "folds": [vars(fold) for fold in folds]}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CrossFitManifest(int(seed), folds, fingerprint)


def build_leave_one_group_out(
    source_groups: Sequence[Sequence[str]], *, seed: int = 1729
) -> CrossFitManifest:
    """Build folds that hold out complete related-source groups together."""
    groups = tuple(tuple(dict.fromkeys(map(str, group))) for group in source_groups)
    if len(groups) < 2 or any(not group for group in groups):
        raise ValueError("leave-one-group-out requires at least two non-empty groups")
    flattened = tuple(source for group in groups for source in group)
    if len(flattened) != len(set(flattened)):
        raise ValueError("crossfit source groups must be disjoint")
    folds = tuple(
        CrossFitFold(
            chr(ord("A") + index),
            tuple(source for other in groups if other != held_out for source in other),
            held_out,
        )
        for index, held_out in enumerate(groups)
    )
    payload = {"seed": int(seed), "folds": [vars(fold) for fold in folds]}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CrossFitManifest(int(seed), folds, fingerprint)


def validate_out_of_source_predictions(
    manifest: CrossFitManifest, predictions: Iterable[Mapping[str, Any]]
) -> None:
    folds = {fold.fold_id: fold for fold in manifest.folds}
    for prediction in predictions:
        fold = folds.get(str(prediction.get("fold_id")))
        training_sources = tuple(map(str, prediction.get("training_source_ids", ())))
        evaluation_sources = tuple(
            map(str, prediction.get("evaluation_source_ids", ()))
        )
        source = str(prediction.get("prediction_source_id", ""))
        label_queue = str(prediction.get("label_queue", ""))
        if (
            fold is None
            or training_sources != fold.training_source_ids
            or evaluation_sources != fold.evaluation_source_ids
            or source not in fold.evaluation_source_ids
            or source in fold.training_source_ids
            or label_queue not in {"audit", "adaptive", "refill", "dense"}
        ):
            raise ValueError("prediction is not out-of-source for its cross-fit fold")


def partition_metrics_by_queue(
    manifest: CrossFitManifest,
    predictions: Iterable[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    predictions = list(predictions)
    validate_out_of_source_predictions(manifest, predictions)
    result = {"audit": [], "adaptive": []}
    for prediction in predictions:
        kind = str(prediction.get("label_queue"))
        if kind not in result:
            raise ValueError(f"unknown queue kind: {kind!r}")
        result[kind].append(prediction)
    return result
