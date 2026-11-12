# Reproducible rally inference

## Run a video

Install Python 3.11, FFmpeg, and `requirements-inference.txt` from the repository
root. The public `rally-six-source-v1` weights are pending real-asset validation;
the CLI explains how to use a local sanitized export if the release is absent.

```bash
python -m src.workflow rally predict \
  --video match.mp4 --output-dir outputs/match \
  --calibration court.json --model rally-six-source-v1 --device auto
```

Omit `--calibration` to launch the existing browser court calibration UI. A
missing, canceled, invalid, or mismatched calibration stops the pipeline before
player extraction. A browser URL is printed for environments without browser
launch support. Supply a calibration file for unattended execution.

Use `--source-id match-a` to name a new source. For a registered source:

```bash
python -m src.workflow --catalog config/sources.local.json \
  rally predict --source match-a --model rally-six-source-v1
```

Model and performance options:

- `--checkpoint PATH --model-manifest PATH`: use a sanitized local model package.
- `--cache-dir DIR`: override the asset cache; `BADDIEVISION_CACHE_DIR` also works.
- `--device auto|cuda|mps|cpu`: auto prefers CUDA, then MPS, then CPU. An explicitly
  requested unavailable device is an error.
- `--batch-size 16`: inference window batch size; must be positive.
- `--no-overlay`: produce predictions and CSV without an MP4.
- `--force`: regenerate stages whose inputs or outputs no longer match receipts.

The full video retains its source-local frame numbering. CSV endpoints are
inclusive and start at zero. No rallies is a successful header-only CSV. Inference
never reads canonical training labels or writes reviewed annotations.

Stage receipts include video identity, asset hashes, preprocessing, runtime/code
identity, and output hashes. Completed stages are reused only on an exact match.
Interrupted/stale stages require `--force`. Video identity changes require a new
source rather than silently reusing old observations.

## Export the existing production checkpoint

Run this only on a trusted locally trained checkpoint, using the exact person
and pose assets used by the maintained preprocessing recipe. No retraining is
part of this release. The output directory must be empty.

```bash
python tools/export_rally_release.py \
  --checkpoint outputs/research/production-rally-segmenter.pt \
  --person-detector models/yolov8n.pt \
  --pose-landmarker models/pose_landmarker_full.task \
  --output-dir release/rally-six-source-v1 \
  --code-revision COMMIT_SHA \
  --base-url https://github.com/MaxLinCode/BaddieVision/releases/download/rally-six-source-v1
```

Replace `COMMIT_SHA` with the verified implementation commit. If the training
preprocessing differs from the maintained defaults, provide its exact JSON with
`--preprocessing FILE`; never infer it from current defaults alone.

The exporter copies only model architecture/state, feature contract, frozen
decoder/preprocessing, code revision, and runtime versions. It removes annotation
fingerprints, detailed split/window records, private source paths and training
extras. Export equivalence is checked independently in tests; verify the actual
production model before publication. Public inference uses `weights_only=True`.

Every asset has an immutable filename, URL, size, and SHA-256 in `manifest.json`.
The runtime verifies downloads and cached files before use, writes downloads
atomically, and supports offline use after all assets are cached. Do not replace
an existing model version with different weights.

## Sample and publication acceptance

Use only the specifically approved recording excerpt. Read the original local
source identity and demo provenance to confirm its origin, then create a silent,
frame-exact, approximately ten-second crop with preserved dimensions/FPS. Rebase
sample frames to zero and remove private provenance paths and ground-truth labels.
The sample helper and package helper expose their arguments through `--help`.

Prepare the sample into a separate empty directory:

```bash
python tools/prepare_rally_sample.py \
  --video APPROVED_SOURCE.mp4 --calibration APPROVED_COURT.json \
  --center-frame 2605 --duration-seconds 10 --output-dir release/sample
```

Private origin details are written beside that sample directory, never inside
it. Copy only `sample.mp4`, `sample-calibration.json`, and `sample-source.json`
into the model bundle. Run `rally predict` on the cropped video with
`--source-id rally-demo`, its sample calibration, and the local exported
checkpoint/manifest. Copy the resulting overlay as `sample-overlay.mp4` and its
prediction CSV as `sample-rallies.csv`; extract `sample-poster.jpg` from that
prediction-only overlay. Keep all observations, receipts, and runtime catalogs
outside the bundle.

Complete [the model-card template](rally-model-card.example.md) as `MODEL_CARD.md`
and copy `requirements-inference.txt` into the bundle. Validate and package:

```bash
python tools/package_rally_release.py \
  --bundle-dir release/rally-six-source-v1 \
  --output-zip release/rally-six-source-v1.zip
```

The packager rejects unknown files and private metadata, checks model/sample
identity and geometry, and creates a deterministic ZIP with `SHA256SUMS`.

Before publishing:

1. Run the full pytest suite, changed-file Ruff, and tracked-data privacy check.
2. Verify the sanitized real checkpoint matches original model predictions.
3. Run real detector/pose/model inference from a clean checkout and empty cache
   against the approved sample; inspect its CSV, full-frame overlay, and poster.
4. Repeat offline from the populated cache and verify identical output receipts.
5. Build the explicit allowlisted package, including manifest, checksums, model
   card, dependency pins, approved sample/calibration, prediction overlay/poster,
   and applicable upstream asset license notices.
6. Publish the versioned GitHub Release only after these checks pass. Training
   annotations, private full videos, raw research checkpoints, features, and local
   recording catalogs must never be attached.

The model card must identify the small private evaluation dataset only through
aggregate counts/results and describe the known fragmentation and camera-transfer
limitations. This release changes packaging and inference, not model accuracy.
