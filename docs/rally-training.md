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

## Compare vision and audio

The opt-in `full_audio` view appends seven features to the default `full`
visual view. It reads the first audio stream with FFmpeg and FFprobe and uses
video presentation timestamps to align the audio, including when frame spacing
varies. The features capture RMS, peak level, power in four frequency bands,
and positive spectral flux. A 48 ms window centered on each video frame's
display interval captures short transients such as shuttle hits and shoe
squeaks. Clips with no audio and windows outside audio coverage use the model's
existing feature-validity mask. The run records audio coverage per source.

Run both views with the same source config, device, seed, and training settings.
Supply a local config that points to source recordings with their original
audio and to the matching annotations and player observations.

```bash
for view in full full_audio; do
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

### Rolling normalization and audio dropout

Use `full_audio_zscore` to standardize all seven audio channels over the trailing
10 seconds of the source recording. Each channel uses only its observed samples,
with windows shortened at the start of a recording. Invalid entries stay masked.
Constant or single-observation histories emit zero. Validation and inference use
the same timestamp-based normalization.

```bash
python -m src.temporal_selector.rally_experiment \
  --config config/rally-experiment.local.json --clean-only \
  --feature-view full_audio_zscore --audio-dropout 0.4 \
  --output-dir outputs/rally-audio-zscore
```

The dropout probability applies independently each time a training window is
sampled. A dropped window has zero values and false validity for all audio
channels across all its frames. Validation, held-out scoring, and inference use
the available audio. The resolved recipe and recovery checkpoints record the
dropout probability.

### Centered spectral-flux baseline

The `full_audio_centered_flux` view keeps log RMS, log peak, and the four
spectral power bands at their original scales. It replaces spectral flux with
its positive excess above the median of observed flux in a centered 1.5-second
window. The scale is 1.4826 times the median absolute deviation, with a floor
of 0.05 in the original log-flux units. The result is capped at 8. The same
normalization applies during training and offline inference.

Run this view with `--clean-only --feature-view full_audio_centered_flux
--audio-dropout 0` to compare the centered baseline directly with the raw
`full_audio` view under the same training recipe.

## RTMPose and raw audio recipe

Start with a constant 30 fps source. Preparing a source at a different rate
creates a converted MP4 and a distinct source identity before any tracks,
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

The preset selects the raw `full_audio` view, a four-second centered transformer
window (two seconds on either side), and a smoothing weight of 0.1. At 30 fps, a
full window contains 121 frames including the center and owns 30 frames.
To compare another context radius, set `dataset.rally_context_seconds`
in the local experiment config. The loss penalizes probability changes between adjacent
supervised frames with the same label. It excludes rally boundaries and
masked frames. A four-camera replay of earlier experiments favored 0.1 over
0.3 for macro interval F1 (0.501 versus 0.463). The combined recipe requires
its own camera-held-out evaluation.

The split for this recipe keeps every transformer input frame at least three seconds from the
train/validation cut for centered player-assignment evidence. It adds the
configured context radius to that guard for window centers, giving the
standard recipe a five-second guard on each side of the cut. Each fold tunes
a higher rally entry threshold and a lower continuation threshold on its
validation frames.
The production checkpoint freezes both thresholds using out-of-fold
predictions, along with the gap and minimum-duration settings.
