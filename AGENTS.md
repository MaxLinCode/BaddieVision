# Repository guidance

## Shot-classifier features

The per-frame shot-classifier schema is ordered as:

1. 66 MediaPipe pose coordinates
2. 7 shuttle position/motion features
3. normalized court anchor `[x, y, observed]`

Do not change this order without regenerating every file in `clip_features/` and
retraining the classifier. Existing 73-input checkpoints predate court anchors
and are incompatible.

Court anchors apply only to the shot classifier. `InPlay` is a separate rally
classifier, and TrackNetV3/InpaintNet are shuttle-tracking components.

## Court calibration

`features/court/calibrations.json` is local generated-data configuration and is
ignored by Git. Copy `config/court_calibrations.example.json` as a starting
point. One source ID represents one static camera segment. If the camera moves,
create a new source ID and calibration and map affected clips explicitly.

Calibration JSON files must include `image_size`. Feature extraction must fail
on unresolved or ambiguous sources instead of silently emitting missing court
features.

See `docs/plans/court-space-player-anchor.md` for the implemented design.
