# Within-camera InPlay learning diagnostic

This experiment asks whether the temporal model can learn rally state within one
clean camera before testing cross-camera transfer. It uses the two maintained
Max-vs-Nik source sequences in parent chronology without merging their
source-local frame indices.

The runner assigns the first 60% of chronological frames to training, the next
20% to validation, and the final 20% to test. One-second guard bands plus each
window's own context containment prevent frames from crossing split boundaries.
The validation block selects a probability threshold by balanced accuracy; the
test block is evaluated once with that threshold.

The current split contains 8,370 owned training frames, 2,700 validation frames,
and 2,740 test frames. Its exact ranges, class counts, representation version,
and dataset fingerprint are written to `metrics.json` and the checkpoint.

Start with the two candidate-free pose representations:

```bash
.venv/bin/python -m src.temporal_selector.within_camera_experiment \
  --config config/selector-experiment.local.json \
  --output-dir outputs/within-maxnik-image-pose-v1 \
  --context-mode full_context --pose-coordinate-mode image \
  --epochs 25 --batch-size 8 --seed 1729 --device mps

.venv/bin/python -m src.temporal_selector.within_camera_experiment \
  --config config/selector-experiment.local.json \
  --output-dir outputs/within-maxnik-player-relative-pose-v1 \
  --context-mode full_context --pose-coordinate-mode player_relative \
  --epochs 25 --batch-size 8 --seed 1729 --device mps
```

Then localize which inputs carry learnable signal:

| Ablation | Additional arguments |
|---|---|
| Temporal prior only | `--context-mode candidates_only` |
| Player/court only | `--context-mode players_court` |
| Image-space pose | `--context-mode full_context --pose-coordinate-mode image` |
| Player-relative pose | `--context-mode full_context --pose-coordinate-mode player_relative` |
| Candidate tokens only | `--context-mode candidates_only --with-candidates` |
| Candidates + relative pose | `--context-mode full_context --pose-coordinate-mode player_relative --with-candidates` |

Every run requires a different empty output directory. Do not add boundary
weighting during this first matrix; the purpose is to test representation
learnability without changing the supervision distribution.

## Validation-tuned interval decoding

After training, tune the deterministic interval decoder on validation without
retraining or consulting test labels:

```bash
.venv/bin/python -m src.temporal_selector.within_camera_decoder \
  --config config/selector-experiment.local.json \
  --checkpoint outputs/within-maxnik-relative-pose-with-candidates-v2/model.pt \
  --output-dir outputs/within-maxnik-relative-pose-with-candidates-decoded-v1 \
  --batch-size 16 --device mps
```

The decoder searches a fixed grid over threshold, short-gap filling, and
minimum interval duration. It selects by validation interval F1, freezes that
configuration, and evaluates test once. `decoder.json` records the complete
search; `decoded-rallies.csv` marks intervals that touch an observed boundary
as open at that edge. The test-only `rally-segmentation-timeline.png` stacks one
probability timeline per labeled rally, sorted with unmatched rallies first;
`rally-diagnostics.csv` records the same boundary, IoU, and low-confidence-gap
analysis as a compact table.
