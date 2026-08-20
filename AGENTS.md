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

The maintained source workflow stores one canonical calibration at
`<artifact_root>/court/calibration.json`. `config/sources.local.json` resolves
only the local video and artifact root; it must not contain a competing
calibration path. Legacy files under `features/court/` are supported only as
explicit import sources. Shot-classifier clip mapping may continue to read the
legacy calibration registry until its separate migration is implemented.

One source ID represents one static camera segment and its own zero-based frame
sequence `0 .. frame_count - 1`. Parent-video trim bounds are provenance only
and never affect artifact or annotation frame indices. If the camera moves,
create a new source ID and calibration and map affected clips explicitly.

Calibration JSON files must include `image_size`. Feature extraction must fail
on unresolved or ambiguous sources instead of silently emitting missing court
features.

See `docs/plans/court-space-player-anchor.md` for the implemented design.

## Local accelerator

Apple MPS is available on the host but may appear unavailable from inside the
sandbox. Do not treat a sandboxed `torch.backends.mps.is_available()` result as
authoritative for experiment runs. Run MPS training outside the sandbox when
host acceleration is required.
