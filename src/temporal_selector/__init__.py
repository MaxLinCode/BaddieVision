"""Configurable temporal shuttle proposal selector."""

from .batch import MASKED_TARGET, NULL_TARGET, SelectorBatch
from .config import ContextMode, SelectorConfig
from .model import (
    CandidateSelectionHead,
    EncodedSelectorBatch,
    InPlayHead,
    JointLosses,
    JointRallyShuttleModel,
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
    build_leave_one_source_out,
    build_two_source_crossfit,
    partition_metrics_by_queue,
    validate_out_of_source_predictions,
)
from .rally_intervals import (
    BOUNDARY_DEFINITION,
    RallyInterval,
    RallyIntervalIndex,
    RallySourceManifest,
    read_rally_intervals,
    refill_frames,
    write_rally_intervals,
)
from .joint_inference import (
    InPlayDecoderConfig,
    calibrate_inplay_decoder,
    decode_inplay_probabilities,
    decoded_intervals,
    write_joint_artifacts,
)

__all__ = [
    "MASKED_TARGET",
    "NULL_TARGET",
    "CandidateSelectionHead",
    "ContextMode",
    "EncodedSelectorBatch",
    "InPlayHead",
    "JointLosses",
    "JointRallyShuttleModel",
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
    "build_leave_one_source_out",
    "build_two_source_crossfit",
    "partition_metrics_by_queue",
    "validate_out_of_source_predictions",
    "BOUNDARY_DEFINITION",
    "RallyInterval",
    "RallyIntervalIndex",
    "RallySourceManifest",
    "read_rally_intervals",
    "refill_frames",
    "write_rally_intervals",
    "InPlayDecoderConfig",
    "calibrate_inplay_decoder",
    "decode_inplay_probabilities",
    "decoded_intervals",
    "write_joint_artifacts",
]
