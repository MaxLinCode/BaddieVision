"""Configurable temporal shuttle proposal selector."""

from .batch import MASKED_TARGET, NULL_TARGET, SelectorBatch
from .config import ContextMode, SelectorConfig
from .model import (
    CandidateSelectionHead,
    EncodedSelectorBatch,
    NullSelectionHead,
    SelectorOutput,
    TemporalShuttleEncoder,
    TemporalShuttleSelector,
)
from .dataset import (
    FRAME_DIMS,
    SelectorDataConfig,
    SelectorSourceConfig,
    SelectorWindow,
    SelectorWindowDataset,
    collate_selector_windows,
)
from .crossfit import (
    CrossFitFold,
    CrossFitManifest,
    build_two_source_crossfit,
    partition_metrics_by_queue,
    validate_out_of_source_predictions,
)

__all__ = [
    "MASKED_TARGET",
    "NULL_TARGET",
    "CandidateSelectionHead",
    "ContextMode",
    "EncodedSelectorBatch",
    "NullSelectionHead",
    "SelectorBatch",
    "SelectorConfig",
    "SelectorOutput",
    "TemporalShuttleEncoder",
    "TemporalShuttleSelector",
    "FRAME_DIMS",
    "SelectorDataConfig",
    "SelectorSourceConfig",
    "SelectorWindow",
    "SelectorWindowDataset",
    "collate_selector_windows",
    "CrossFitFold",
    "CrossFitManifest",
    "build_two_source_crossfit",
    "partition_metrics_by_queue",
    "validate_out_of_source_predictions",
]
