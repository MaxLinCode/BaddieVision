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
