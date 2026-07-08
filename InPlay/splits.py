"""Source/video-level splits for InPlay sequence datasets."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Mapping, Sequence


def source_group_split(
    metadata: Sequence[Mapping[str, object]],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split sequence indices by ``source_id`` to avoid adjacent-frame leakage."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        source = row.get("source_id")
        if source in ("", None):
            raise ValueError(f"metadata row {index} has no source_id")
        groups[str(source)].append(index)
    if len(groups) < 2:
        raise ValueError("at least two source_id groups are required")

    rng = random.Random(seed)
    source_ids = list(groups)
    rng.shuffle(source_ids)
    target = max(1, round(len(metadata) * validation_fraction))
    validation_sources: set[str] = set()
    validation_count = 0
    for source in source_ids:
        validation_sources.add(source)
        validation_count += len(groups[source])
        if validation_count >= target and len(validation_sources) < len(source_ids):
            break
    if len(validation_sources) == len(source_ids):
        validation_sources.remove(source_ids[-1])

    train, validation = [], []
    for source, indices in groups.items():
        (validation if source in validation_sources else train).extend(indices)
    if not train or not validation:
        raise ValueError("split produced an empty train or validation set")
    return sorted(train), sorted(validation)
