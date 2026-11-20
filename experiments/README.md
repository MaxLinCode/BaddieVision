# Research studies

Named study orchestration lives here so importing maintained runtime APIs does
not load a study or make its frozen annotation hashes and frame partitions part
of the runtime contract. Import and run studies from
`experiments.temporal_selector.<study>`; obsolete aliases in `src` were removed.

| Module under `experiments.temporal_selector` | Reproduction scope |
| --- | --- |
| `partial_oracle_experiment` | Matched raw/oracle candidate retraining on the frozen dense annotation snapshot |
| `oracle_track_experiment` | Leakage-safe oracle track and corruption audit using that same study split |
| `shortcut_diagnostic` | Report-only camera shortcut diagnostic |
| `candidate_counterfactual` | Frozen checkpoint candidate masking |
| `boundary_snap_experiment` | Prediction-only operational boundary evaluation |
| `within_camera_experiment` | Chronological one-camera learning diagnostic |
| `within_camera_decoder` | Offline validation-only decoder tuning |

Run `python -m experiments.temporal_selector.<module> --help` from the repository
root for each study's configuration and output arguments. The original fixed
annotation identity and partition rules remain alongside the study that owns
them. Supply the original dataset configuration, annotation artifacts, and model
checkpoints; these studies do not download or regenerate private assets.

Maintained training entrypoints remain `src.temporal_selector.experiment` and
`src.temporal_selector.rally_experiment`. Reusable selector training/prediction
helpers live in `src.temporal_selector.training`; shared player/court feature
helpers live in `src.temporal_selector.observations`.
