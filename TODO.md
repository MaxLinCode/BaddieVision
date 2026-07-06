# TODO

## Court-space shot-classifier rollout

- [ ] Copy `config/court_calibrations.example.json` to
  `features/court/calibrations.json`.
- [ ] Calibrate every static source video and save each calibration under
  `features/court/` with its recorded `image_size`.
- [ ] Complete `clip_overrides` for legacy clip names that do not contain a
  source-video ID.
- [ ] Regenerate all shot-classifier arrays with `src/extract_clip_features.py`
  and confirm they have shape `(36, 76)`.
- [ ] Retrain the shot classifier because existing 73-input checkpoints are
  incompatible.
- [ ] Review source-grouped validation metrics and compare them with the
  previous classifier baseline.
