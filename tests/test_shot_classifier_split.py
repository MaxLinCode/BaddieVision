import numpy as np
import pytest

from src.train_shot_classifier import grouped_train_val_split


class DummyDataset:
    def __init__(self, groups):
        self.source_groups = groups
        self.labels = [index % 2 for index in range(len(groups))]

    def __len__(self):
        return len(self.source_groups)


def test_grouped_split_has_no_source_overlap():
    dataset = DummyDataset(["a"] * 5 + ["b"] * 5 + ["c"] * 5)
    train, validation = grouped_train_val_split(
        dataset, val_size=0.34, random_state=42
    )
    groups = np.asarray(dataset.source_groups)
    train_groups = set(groups[train.indices])
    validation_groups = set(groups[validation.indices])
    assert train_groups.isdisjoint(validation_groups)


def test_grouped_split_requires_two_sources():
    with pytest.raises(ValueError, match="at least two"):
        grouped_train_val_split(DummyDataset(["a"] * 5))
