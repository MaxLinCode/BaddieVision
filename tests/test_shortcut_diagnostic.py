from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
import torch

from src.court_projection import CourtHomography
from src.temporal_selector import RallyBatch
from src.temporal_selector.rally_dataset import RallyWindow
from src.temporal_selector.rally_features import (
    FULL_RALLY_FEATURE_NAMES,
    RALLY_FEATURE_VIEWS,
)
from src.temporal_selector.shortcut_diagnostic import (
    ProbeSample,
    balance_probe_partition,
    chronological_probe_split,
    project_rally_windows,
    reproject_full_windows,
    temporal_block_probe_bootstrap,
)


def _window(source: str = "source", frames: int = 4) -> RallyWindow:
    values = torch.arange(frames * 154, dtype=torch.float32).reshape(frames, 154)
    validity = torch.ones_like(values, dtype=torch.bool)
    validity[1, ::3] = False
    return RallyWindow(
        source_id=source,
        window_id=f"{source}-window",
        anchor_frame=1,
        frame_indices=tuple(range(frames)),
        owned_frames=frozenset(range(frames)),
        relative_time_seconds=torch.arange(frames, dtype=torch.float32) - 1,
        frame_values=values,
        frame_validity=validity,
        inplay_targets=torch.tensor([0] * frames),
        rally_start_targets=torch.zeros(frames),
        rally_end_targets=torch.zeros(frames),
        metadata={"image_size": [1920, 1080], "fps": 30.0, "frame_count": frames},
    )


def test_registered_feature_views_have_exact_names_dimensions_and_masks() -> None:
    assert len(FULL_RALLY_FEATURE_NAMES) == 154
    assert {name: view.dimension for name, view in RALLY_FEATURE_VIEWS.items()} == {
        "full": 154,
        "calibrated_canonical": 142,
        "no_position": 138,
        "image_position": 150,
    }
    full = _window()
    for name, view in RALLY_FEATURE_VIEWS.items():
        projected = project_rally_windows([full], name)[0]
        expected_indices = torch.tensor(view.full_indices)
        assert torch.equal(projected.frame_values, full.frame_values.index_select(-1, expected_indices))
        assert torch.equal(projected.frame_validity, full.frame_validity.index_select(-1, expected_indices))
        assert tuple(FULL_RALLY_FEATURE_NAMES[index] for index in view.full_indices) == view.names


def test_homography_mutation_changes_only_derived_court_features() -> None:
    window = _window(frames=3)
    # Put valid normalized image feet in-range before applying a different projection.
    window.frame_values[:, [4, 15]] = 0.5
    window.frame_values[:, [5, 16]] = 0.75
    changed = reproject_full_windows([window], {"source": CourtHomography(np.eye(3))})[0]
    difference = changed.frame_values != window.frame_values
    changed_indices = set(torch.nonzero(difference, as_tuple=False)[:, 1].tolist())
    assert changed_indices <= {6, 7, 17, 18}
    assert changed_indices
    assert torch.equal(changed.frame_validity, window.frame_validity)
    assert changed.inplay_targets is window.inplay_targets
    assert changed.rally_start_targets is window.rally_start_targets
    assert changed.rally_end_targets is window.rally_end_targets


def _probe_rows() -> list[ProbeSample]:
    rows = []
    for camera_index, camera in enumerate(("a", "b", "c")):
        for start in range(0, 100, 2):
            rows.append(
                ProbeSample(
                    source_id=camera,
                    camera=camera,
                    state=(start // 2) % 2,
                    start_frame=start,
                    end_frame=start + 1,
                    values=np.asarray([camera_index, start], dtype=float),
                )
            )
    return rows


def test_probe_split_balance_and_temporal_bootstrap_are_deterministic() -> None:
    rows = _probe_rows()
    train, test, manifest = chronological_probe_split(
        rows,
        {camera: 100 for camera in ("a", "b", "c")},
        {camera: 2.0 for camera in ("a", "b", "c")},
        guard_seconds=1.0,
    )
    assert all(row.end_frame < manifest["sources"][row.source_id]["cut_frame"] - 2 for row in train)
    assert all(row.start_frame >= manifest["sources"][row.source_id]["cut_frame"] + 2 for row in test)
    first, first_manifest = balance_probe_partition(train)
    second, second_manifest = balance_probe_partition(train)
    assert [(row.source_id, row.start_frame) for row in first] == [
        (row.source_id, row.start_frame) for row in second
    ]
    assert first_manifest == second_manifest
    counts = {(camera, state): 0 for camera in ("a", "b", "c") for state in (0, 1)}
    for row in first:
        counts[(row.camera, row.state)] += 1
    assert len(set(counts.values())) == 1
    predictions = [row.camera for row in test]
    boot1 = temporal_block_probe_bootstrap(test, predictions, samples_count=40)
    boot2 = temporal_block_probe_bootstrap(test, predictions, samples_count=40)
    assert boot1 == boot2
    assert boot1["balanced_accuracy_confidence_interval_95"] == pytest.approx([1.0, 1.0])


def test_model_tensor_contract_contains_no_camera_or_calibration_authority() -> None:
    forbidden = ("camera", "source", "calibration", "homography", "landmark")
    assert not {
        field.name
        for field in fields(RallyBatch)
        if any(token in field.name for token in forbidden)
    }
