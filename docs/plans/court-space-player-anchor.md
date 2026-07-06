# Court-space player-anchor integration

## Goal

Add a camera-invariant anchor describing where the player stands on the court
to the shot-classifier input. This change intentionally excludes `InPlay` and
TrackNetV3.

## Design

- Reuse one image-to-court homography for every clip from the same static source
  video. Resolve modern clip names by source ID and legacy names through explicit
  overrides in a local calibration registry.
- Estimate each foot's ground contact from visibility-weighted MediaPipe heel
  and foot-index landmarks, falling back to its ankle. Project each available
  foot separately and average the results in court coordinates.
- Normalize court coordinates by the badminton half-width and half-length, then
  append `[court_x, court_y, observed]` after the existing 73 features.
- Interpolate missing runs up to five frames while keeping `observed = 0`.
  Reject longer gaps and clips with no usable anchor.
- Split training and validation by source-video group to prevent clips from the
  same recording leaking across the evaluation boundary.

## Compatibility and acceptance

Generated arrays change from `(36, 73)` to `(36, 76)`, so feature regeneration
and classifier retraining are required. Existing shot-classifier checkpoints
are not compatible. Tests must cover projection, fallback, imputation, registry
resolution, feature ordering, and source-group isolation.
