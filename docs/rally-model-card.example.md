# RallySegmenter — rally-six-source-v1

Release preparation template. Complete the asset notices and verification
record before placing this document in a public bundle as MODEL_CARD.md.

## Intended use

Offline rally segmentation for static-camera singles badminton video. The
model consumes player/court context and normalized player-relative pose;
it does not consume shuttle candidates. It uses centered temporal context
and is not a streaming or causal model.

## Evaluation

The maintained private research evaluation used six reviewed sources across
four camera groups, 126 strict rallies, and 52,012 owned labeled frames.
Reported camera-held-out macro results:

| Metric | Value |
|---|---:|
| Frame F1 | 0.718 |
| ROC AUC | 0.88974 |
| PR AUC | 0.82313 |
| Interval F1 | 0.41915 |
| Boundary MAE | 26.11 frames |

These are small private-dataset research results, not a public benchmark.
Fragmented intervals, boundary errors, and camera-domain calibration remain
limitations. A model-predicted interval is not a reviewed annotation.

## Inference contract

The accompanying manifest records code revision, schema/version/ordered
feature names, model architecture, offline context, preprocessing, decoder,
runtime versions, and the URL/size/SHA-256 of each inference asset. Decoder
settings are frozen; this packaging release does not retrain the model.

No training annotations, recording paths, detailed split manifests, full source
videos, or raw research checkpoints are included. Only the specifically
approved short video sample is distributed with predictions and calibration.

## Asset notices — complete before publication

Record the exact upstream source/version and license notices for the supplied
person detector and MediaPipe pose-landmarker assets. Verify these are the
intended pretrained assets and contain no custom private training metadata.
Project source-code license: see the repository LICENSE. Do not infer asset
license terms from the source-code license.

## Verification record — complete before publication

Record the release implementation commit, full-suite CI result, actual
production-checkpoint export equivalence result, fresh-checkout real inference
result, approved sample review, and offline rerun result. Publish the artifact
checksums alongside this completed model card.
