import numpy as np
import pytest
import torch

from src.court_projection import CourtHomography
from src.temporal_selector.rally_features import (
    CANONICAL_RALLY_FEATURE_VIEW,
    FULL_RALLY_FEATURE_NAMES,
    SERVE_BOX_NAMES,
    rally_feature_view,
    rally_serve_box_features,
)


def test_service_box_scores_use_both_ankles_and_mask_missing_players():
    # Pixel court origin (10, 10), one metre per pixel.
    court = CourtHomography(np.array([[1, 0, -10], [0, 1, -10], [0, 0, 1]]))
    values = torch.zeros((4, len(FULL_RALLY_FEATURE_NAMES)))
    validity = torch.zeros_like(values, dtype=torch.bool)
    poses = values[:, 22:].reshape(-1, 2, 33, 2)
    visible = validity[:, 22:].reshape_as(poses)
    poses[:3, :, 27:29] = torch.tensor([
        [[[-1.4, -4], [-0.6, -4]], [[0.6, 4], [1.4, 4]]],
        [[[0.6, -4], [1.4, -4]], [[0.6, 4], [1.4, 4]]],
        [[[0.6, -4], [1.4, -4]], [[-1.4, 4], [-0.6, 4]]],
    ]).add(10).div(20)
    visible[:3, :, 27:29] = True
    visible[2, 1, 28] = False

    scores, valid = rally_serve_box_features(values, validity, court, (20, 20))

    decay = np.exp(-0.5 * (1 / 0.75) ** 2)
    assert scores[0].tolist() == pytest.approx([1, decay, decay, 1, 1], abs=2e-6)
    assert scores[1].tolist() == pytest.approx([decay, 1, decay, 1, decay], abs=2e-6)
    assert scores[2].tolist() == pytest.approx([decay, 1, 0, 0, 0], abs=2e-6)
    assert valid.tolist() == [[True] * 5, [True] * 5,
                              [True, True, False, False, False], [False] * 5]
    assert not scores[3].any()


def test_serve_views_keep_baseline_order_and_audio_tail():
    baseline = rally_feature_view(CANONICAL_RALLY_FEATURE_VIEW)
    four = rally_feature_view("serve_boxes")
    five = rally_feature_view("serve_boxes_diagonal")
    for view, count in ((four, 4), (five, 5)):
        assert view.dimension == baseline.dimension + count
        assert view.names[-7 - count:-7] == SERVE_BOX_NAMES[:count]
        assert view.names[:-7 - count] + view.names[-7:] == baseline.names
        values = torch.arange(view.dimension, dtype=torch.float32)[None, :]
        valid = torch.ones_like(values, dtype=torch.bool)
        selected, selected_valid = view.select(values, valid)
        assert torch.equal(selected, values)
        assert torch.equal(selected_valid, valid)
