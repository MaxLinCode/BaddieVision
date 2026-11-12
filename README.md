# BaddieVision

BaddieVision turns static-camera singles badminton video into structured rally
intervals. It also contains research tooling for player pose, court projection,
shuttle tracking, annotation, and shot classification.

The maintained RallySegmenter uses player/court observations and player-relative
pose. It does not need shuttle candidates, training labels, or an annotation
teacher at inference time. It is an offline model with centered temporal context.

The six-source, four-camera research baseline reports macro frame F1 **0.718**,
ROC AUC **0.890**, interval F1 **0.41915**, and boundary MAE **26.11 frames**
on 126 rallies / 52,012 labeled frames. These results come from a small private
dataset; they are not a public benchmark or a production accuracy guarantee.
See [the evaluation protocol](docs/independent-rally-segmenter.md).

## Install

Use Python 3.11 and install FFmpeg on your system.

```bash
git clone --recurse-submodules https://github.com/MaxLinCode/BaddieVision.git
cd BaddieVision
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
```

For Linux CPU-only installations, first install the matching CPU wheels:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-inference.txt
```

For development, install `requirements-dev.txt` with the inference pins as
constraints, then run tests:

```bash
python -m pip install -r requirements-dev.txt -c requirements-inference.txt
python -m pytest -q
python scripts/check_private_assets.py
```

The TrackNet submodule is pinned by its Git commit. Do not replace that commit
with a moving branch during setup or CI.

## Video to rally CSV and overlay

```bash
python -m src.workflow rally predict \
  --video match.mp4 \
  --output-dir outputs/match \
  --model rally-six-source-v1 \
  --device auto
```

If court calibration is missing, the command reports it and opens the browser
calibration tool. Save the calibration to continue. For a noninteractive run,
provide `--calibration court.json`.

The named model is downloaded only from its versioned release and every asset is
checked against its size and SHA-256. **The public model release is pending
verification of the existing local weights and approved sample.** Until that
release is available, use a sanitized local export:

```bash
python -m src.workflow rally predict \
  --video match.mp4 --calibration court.json --output-dir outputs/match \
  --checkpoint release/rally-segmenter.pt --model-manifest release/manifest.json
```

The pipeline registers an immutable source, extracts or verifies person tracks
and pose, predicts one state per frame, then writes:

```text
outputs/match/
  source.json
  court/calibration.json
  players/                       Observations and stage receipts
  rallies/predictions.jsonl       One probability/state record per frame
  rallies/rallies.csv             Inclusive zero-based rally intervals
  rallies/overlay.mp4             Prediction-only video overlay
  rallies/run.manifest.json       Inputs, versions, and output fingerprints
```

A verified cached rerun reuses the same outputs. Stale or incomplete artifacts
fail with a `--force` recovery instruction. `--no-overlay` skips video rendering.
CUDA, MPS, and CPU are supported; `auto` selects the first available device.

See [inference and release instructions](docs/rally-inference.md) for the complete
command, registered-source reuse, checkpoint export, and sample preparation.

## Private data

Videos, annotations, extracted features, model weights, local recording configs,
and detector training images/labels stay outside tracked Git files. The checkout
contains synthetic regression fixtures and generic example configs. Restore your
own data locally; none is necessary for the default test suite.

Previously committed data remains in Git history. This change preserves history
and prevents the known private asset classes from being reintroduced in future
commits. Release packages use an explicit file allowlist. Only approved sample
media and inference assets belong in a verified release.

## Other workflows

- [Source registration, calibration, and annotation](docs/workflow.md)
- [Annotation platform](docs/annotation-platform.md)
- [Court-space shot features](docs/plans/court-space-player-anchor.md)
- [Research evaluation](docs/independent-rally-segmenter.md)

The shot classifier remains a separate workflow. Its `(36, 76)` feature arrays
contain 66 pose coordinates, seven shuttle features, and a three-value court
anchor. Older 73-input checkpoints require feature regeneration and retraining.

Court calibration represents one static camera segment, uses `image_size`, and
maps floor points to metres. Camera movement requires a new source/calibration.
Airborne-shuttle projection does not recover true 3D location.

## License

The project is [MIT licensed](LICENSE). TrackNet retains its own license.
Detector/pose assets retain their upstream license terms; redistribution must
include the applicable notices in the model card.
