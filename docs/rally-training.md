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
