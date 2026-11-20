# Research and historical workflow index

Start with [workflow.md](workflow.md) for maintained source registration,
calibration, extraction, and recovery. The following material explains research
history rather than defining the current observation workflow.

- [Named reproducible studies](../experiments/README.md): oracle, camera shortcut,
  counterfactual, boundary, and within-camera orchestration and canonical entrypoints.
- [Archived prototypes](../archive/README.md): older fixed-input extraction,
  dense/LSTM training, and prediction scripts.
- [Within-camera study](within-camera-inplay.md)
- [Joint selector handoff](joint-inplay-handoff.md)
- [Conditional shuttle study](plans/joint-inplay-conditional-shuttle.md)
- [Selector candidate study](plans/rally-shuttle-candidate-selection-v1.md)
- [Independent rally segmenter design](independent-rally-segmenter.md)
- [Repository simplicity assessment](repository-simplicity-review.md)

Run studies from `experiments.temporal_selector` and prototypes from
`archive.prototypes`. Their original study-specific hashes, partitions, and
asset expectations remain attached to the study; they are not default runtime
configuration.
