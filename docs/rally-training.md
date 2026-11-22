# Rally training and recovery

The maintained candidate-free experiment accepts ordinary training settings:

```bash
python -m src.temporal_selector.rally_experiment \
  --config config/rally-experiment.example.json \
  --output-dir outputs/rally-cpu-smoke --clean-only \
  --device cpu --epochs 1 --batch-size 2 --learning-rate 0.001
```

A config's optional `training` object accepts `seed`, `epochs`, `batch_size`,
`sampling` (`boundary_balanced` or `uniform`), `boundary_aux_weight`, `optimizer`
(`AdamW`), `learning_rate`, `gradient_clip_norm`, and `device`. CLI values take
precedence. Every run writes `resolved-config.json` before training, and final
metrics and inference exports record the resolved recipe. The historical
25-epoch MPS configuration remains available as `--preset benchmark`; overrides
that would change that recipe are rejected.

For a clean-only run, each completed training epoch atomically commits a
`training-recovery.pt` file inside its fold directory. Production training uses
`production-training-recovery.pt`. Re-run the same command and output directory
with `--resume` to restore the latest committed epoch. Completed training loops
load their state without repeating optimization; fold evaluation and export
can run again. An interrupted epoch repeats from its beginning.

Fold IDs use sorted camera names. Reordering the keys in `camera_sources`
leaves the fold IDs unchanged. Changing which sources belong to a camera or
reordering its source list changes the split and requires a new run. Resolved
configs from the earlier draft (version 1) lack the original fold order and
cannot be resumed with this version.

Recovery files contain the latest model and optimizer states, epoch, sampler
position, Python/NumPy/Torch/device RNG states, epoch metrics, and a separate
copy of the best validation model. They validate configuration, ordered window
observations/labels/ownership, dataset and annotation fingerprints, and split
identity before restoring. Changing those inputs requires a new output
directory. Final `rally-segmenter.pt` and `production-rally-segmenter.pt` files
remain inference exports; they are not accepted as recovery checkpoints.

Epoch recovery applies to the maintained clean-only fold and production
trainers. The comparative historical study arms accept recipe overrides but
do not support `--resume`. Recovery does not claim identical results across
different accelerator/runtime versions or nondeterministic device kernels.

## Canonical motion, vision, and audio

The canonical `full_audio_zscore_motion` view adds signed player and court
velocity to the baseline `full_audio_zscore` view. The audio portion appends
seven normalized audio features to the visual representation. It reads the
first audio stream with FFmpeg and FFprobe and uses
video presentation timestamps to align the audio, including when frame spacing
varies. The features capture RMS, peak level, power in four frequency bands,
and positive spectral flux. A 48 ms window centered on each video frame's
display interval captures short transients such as shuttle hits and shoe
squeaks. The model's feature-validity mask marks windows as invalid when they
overlap gaps or fall outside the recorded audio. Recorded silence remains
valid. Spectral flux requires coverage in both analysis windows. The
transformer masks missing audio tokens in attention while retaining the
corresponding video tokens.
The run records audio coverage per source.

Run the canonical view and its ablations with the same source config, device,
seed, and training settings.
Supply a local config that points to source recordings with their original
audio and to the matching annotations and player observations.

```bash
for view in full full_audio_zscore full_audio_zscore_motion; do
  python -m src.temporal_selector.rally_experiment \
    --config config/rally-experiment.local.json --clean-only \
    --feature-view "$view" --output-dir "outputs/rally-${view}"
done
```

Compare `macro.interval_f1`, `macro.boundary_mae`, false splits, and false merges
in each run's `metrics.json`. Check `by_camera` for performance on each held-out
camera and `dataset_manifest.audio_coverage` for the number of observed frames
per source. The checkpoint and release manifest identify the feature view
automatically. Export the trained checkpoint with
[the existing inference exporter](rally-inference.md#export-the-existing-production-checkpoint)
and use its manifest for prediction.

The audio z-score uses the trailing ten seconds of observed values for each
channel and emits zero when there is insufficient history or variance.
`full_audio_zscore` remains available as the audio-only baseline. Use
`full_audio` to compare against raw audio with the same attention architecture
and camera splits.

## Clean train and validation partitions

Each fold holds out one complete camera. For each remaining camera, the clean
trainer plans the split from source frame counts and annotations before building
feature windows. It prefers a whole-recording boundary that leaves 70–90% of the
frames in training. Otherwise it splits near 80%, moving the cut up to two
seconds to reach a nearby reviewed break when available.

For a recording cut in two, the trainer extracts each frame range as its own
clip and reruns person tracking and player pose processing on both clips.
Assignment, activity, pose repair, and transformer windows stay inside their
clip. Each clip stores PCM audio in Matroska to preserve the timing of audio
gaps. Audio features and their trailing z-score history start fresh for each
clip. Edge windows retain their supervised frames. Prediction
rows use the original source and frame IDs, and the resolved run configuration
records the partition artifact fingerprints for resume.

The trainer looks up the original detector and pose assets from the source
artifacts. Use `--person-model` or `--pose-model` to provide relocated assets.
The derived clips and observations are cached in `.rally-partitions` next to
the run directory. Whole recordings use their existing features.

A rally spanning a cut keeps its in-play labels and any actual start or end on
each side. Complete-interval metrics exclude the cut rally. Decoder calibration
uses continuous validation predictions, including frames without labels, and
scores only reviewed frames. The matched comparative experiment still uses
its historical split.

## RTMPose and normalized audio recipe

Start with a constant 30 fps source. Inputs below 30 fps are rejected.
Downsampling a higher-rate input creates a converted MP4 and a distinct source identity before any tracks,
pose caches, or frame-indexed annotations are produced. Use an unused source
ID and artifact root when migrating an existing recording. Calibrate and
review labels on the resulting source-local frame sequence.

```bash
python -m src.workflow source add --video match.mp4 --rally-30-fps \
  --source-id match-30fps --artifact-root outputs/match-30fps
python -m src.workflow stage players --source match-30fps \
  --pose-backend rtmpose --pose-model models/rtmpose-l-halpe26-384x288-fp16.onnx \
  --pose-device cpu
```

Run court calibration and the person-tracks stage before the player stage.
RTMPose uses Halpe26 FP16 ONNX, 15% crop padding, and no retry. Every training
source must use the same pose model and crop recipe. The dataset and checkpoints
record the SHA-256 of the pose asset and the preprocessing settings.

```bash
python -m src.temporal_selector.rally_experiment \
  --config config/rally-experiment.local.json --output-dir outputs/rally-rtmpose-audio \
  --clean-only --preset rtmpose-audio --device cuda
```

The preset selects the canonical `full_audio_zscore_motion` view, a four-second
centered transformer window (two seconds on either side), and a smoothing
weight of 0.1. At 30 fps, a
full window contains 121 frames including the center and owns 30 frames.
To compare another context radius, set `dataset.rally_context_seconds`
in the local experiment config. The loss penalizes probability changes between adjacent
supervised frames with the same label. It excludes rally boundaries and
masked frames. A four-camera replay of earlier experiments favored 0.1 over
0.3 for macro interval F1 (0.501 versus 0.463). The combined recipe requires
its own camera-held-out evaluation.

Each fold tunes a higher rally entry threshold and a lower continuation
threshold on its validation frames.
The production checkpoint freezes both thresholds using out-of-fold
predictions, along with the gap and minimum-duration settings.

## Joint and court velocity representation

The canonical view adds signed x/y velocity for each pose landmark and each
player's court position. Use `full_motion` for the visual-only experiment, or
pass `--feature-view full_audio_zscore` explicitly to train the audio-only
baseline.
Both views use the existing court-centered coordinate convention, with the
court center at zero and the outer lines at ±1 along each axis. Coordinates
can extend beyond the lines. A centered three-frame median on each court
axis suppresses isolated spikes in the position channels. It uses one frame
of lookahead within the offline source or split clip. All three samples must
be valid, belong to the same track, and have an accepted, unrepaired pose with
visible landmarks on both feet. An available foot diagnostic must not
disagree with the assigned side and court bounds. Other positions retain
their original values and masks.

The motion views map crop landmarks back to image pixels using the exact
stored crop box. They center pose on the midpoint of the hips and normalize
by the median visible shoulder-midpoint to hip-midpoint distance for that
player track within the source or split clip. Frames with either hip missing
or no usable track torso scale have masked pose features.

Joint velocity uses a trailing three-frame slope and court velocity uses
four raw projected positions before the median. Slopes are per second, in torso-length units for joints and
court half-width or half-length units for court position. A velocity requires
all samples in its window to be valid and on the same track. Motion is computed
before slicing overlapping model windows.

Compare motion views on the same camera folds and source artifacts, with the
training recipe fixed. Report interval F1 and signed start/end errors by
camera. The motion views change both pose normalization and velocity
channels, so comparisons against the original views measure their combined
effect. The filter windows span less time at higher FPS.
