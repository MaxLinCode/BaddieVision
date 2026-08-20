# M3 Transformer Training Speed

## Status

Investigation started on 2026-07-20 on the 16 GB, 8-core M3 MacBook Air used
for this repository. The maintained temporal transformer is
`src.temporal_selector`; `src/train_shot_classifier.py` is an LSTM and is not
part of this work.

The saved selector runs under `outputs/joint-selector-*` all recorded
`device: cpu` and `batch_size: 1`. The project virtual environment contains
PyTorch 2.12.1 with working MPS support. Managed sandbox execution hides the
Metal device: inside the sandbox MPS reports unavailable and emits a misleading
minimum-macOS error, while the identical interpreter outside the sandbox sees
one MPS device and successfully allocates and synchronizes an MPS tensor.

This distinction is essential for future diagnostics. Sandbox results must not
be used to judge whether the host PyTorch/MPS installation works. GPU training
and MPS benchmarks need execution with Metal device access.

## Likely bottlenecks

1. Training can fall back silently to CPU when MPS is unavailable, including
   when launched in an execution sandbox that hides Metal.
2. The default batch size of one pays model, optimizer, collation, and Python
   overhead once per two-second window.
3. Token packing in `TemporalShuttleEncoder.forward` uses nested Python loops,
   device-to-Python conversions, and individual tensor assignments. On MPS,
   `.tolist()`, `int(tensor)`, and similar operations synchronize the GPU.
4. Sparse selection loss is calculated in another Python loop over supervised
   frames.
5. Training copies three scalar losses from the device to the CPU every batch.
6. Full `SelectorBatch.validate` runs during collation, candidate dropout,
   model forward, and loss calculation. Its data-dependent Python checks are
   useful at the input boundary but expensive in the model hot path.
7. The M3 MacBook Air is fanless, so sustained workloads may throttle after
   short benchmarks.

## Work plan

- [x] Record the initial environment and saved-run findings.
- [x] Warn when PyTorch contains MPS support but cannot make MPS available.
- [x] Accumulate training metrics on the training device and transfer them to
  the CPU once per epoch.
- [x] Establish preliminary CPU and MPS baselines for batch sizes 1, 4, 8,
  and 16 using a controlled synthetic full-context workload.
- [x] Verify MPS outside the managed sandbox. The maintained PyTorch 2.12.1
  environment works; no wheel change is needed.
- [x] Replace Python token packing with batched count/rank/scatter operations
  in the model. Keep the maps on-device and preserve exact packed ordering.
- [x] Vectorize frame-local selection loss while preserving its exact sparse
  supervision and null-target semantics.
- [x] Track validation provenance so immutable CPU-collated batches do not
  repeat data-dependent validation in every forward pass.
- [x] Benchmark CPU against MPS after removing synchronization points.
- [x] Precompute packed frame/candidate positions and frame-local candidate
  indices during vectorized CPU collation, and transfer the maps with the batch.
- [x] Size attention inputs to the exact largest real packed sequence in each
  batch instead of the frame-plus-padded-candidate upper bound.
- [x] Deterministically bucket shuffled training windows by token count to
  reduce attention padding without dropping or duplicating samples.
- [x] Batch calibration and evaluation inference and transfer logits to CPU
  once per batch while preserving prediction order and ownership semantics.
- [x] Avoid per-batch device transfers for joint supervised-frame counts while
  retaining immediate non-finite-loss checks and sparse-only loss evaluation.
- [ ] Only then evaluate MPS mixed precision or `torch.compile`; retain them
  only if measured faster and numerically stable.

## Benchmark protocol

Use the same dataset fingerprint, cross-fit manifest, context mode, seed, and
model configuration for every comparison. Record:

- PyTorch, Python, and macOS versions;
- device and whether an actual MPS tensor can be allocated;
- batch size and peak memory;
- dataset construction time;
- per-fold training seconds and median seconds per epoch after the first epoch;
- evaluation/calibration time separately from training;
- final loss and validation metrics to detect behavior changes.

Short benchmarks should use at least three epochs because the first MPS epoch
can include setup costs. Full accuracy comparisons should keep optimizer
semantics explicit: increasing batch size changes the number of optimizer
steps, so a faster run is not automatically an equivalent training run.

## Immediate commands

Confirm the active runtime before each benchmark:

```bash
.venv/bin/python -c 'import platform, torch; print(platform.platform()); print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'
```

Once MPS works, compare explicit devices and batches rather than relying on
automatic selection:

```bash
.venv/bin/python -m src.temporal_selector.experiment \
  --config config/selector-experiment.local.json \
  --output-dir outputs/selector-benchmark-b8 \
  --context-mode full_context --epochs 3 --batch-size 8 --device mps
```

Every output directory must be new because the experiment runner deliberately
refuses to overwrite prior runs.

## Preliminary CPU microbenchmark

On 2026-07-20, a one-epoch synthetic benchmark used 16 full-context windows,
61 frames per window, three candidates per frame, the production model size,
and no candidate dropout. It measured the existing end-to-end training helper,
including collation, validation, packing, forward/backward, clipping, and the
optimizer step:

| Batch | Seconds | Windows/second |
| ---: | ---: | ---: |
| 1 | 0.965 | 16.58 |
| 4 | 0.613 | 26.10 |
| 8 | 0.653 | 24.51 |
| 16 | 0.810 | 19.76 |

Batch four delivered about 57% more throughput than batch one in this short
CPU test. This is directional rather than conclusive: it did not include real
selection supervision, its duration was susceptible to setup noise, and larger
batches perform fewer optimizer steps. The next real-data smoke run should
therefore begin with batch four and record wall-clock time and metrics.

## MPS wheel experiments

Temporary environments under `/private/tmp` tested the maintained Python
3.11.15 interpreter with PyTorch 2.9.1 and 2.5.1. Both were native arm64 wheels,
both reported MPS as built but unavailable, and both rejected creation of a
one-element MPS tensor with an incorrect minimum-macOS error—but those probes
were run inside the managed sandbox. An elevated probe outside the sandbox then
proved that the unchanged project PyTorch 2.12.1 environment works correctly:
MPS was available, device count was one, and tensor allocation succeeded. The
earlier wheel experiments were therefore sandbox false negatives, not evidence
of a broken host installation. The project venv was not modified.

## Preliminary MPS microbenchmark

The same synthetic workload was run outside the sandbox with actual MPS access
for two epochs:

| Batch | Seconds (2 epochs) | Windows/second |
| ---: | ---: | ---: |
| 1 | 8.570 | 3.73 |
| 4 | 7.511 | 4.26 |
| 8 | 8.750 | 3.66 |
| 16 | 9.264 | 3.45 |

This is substantially slower than the preliminary CPU results. It supports the
hot-path diagnosis: this small model currently performs many Python-driven
device reads and indexed writes during validation and token packing, making MPS
synchronization overhead dominate useful transformer computation. Until token
packing and sparse loss are vectorized, CPU with batch four is the practical
default. MPS should be benchmarked again after that refactor rather than assumed
to be faster merely because it is available.

## Vectorized-packing result

After replacing the nested Python packing loops with batched tensor operations,
the controlled two-epoch benchmark measured:

| Device | Batch | Before (windows/s) | After (windows/s) |
| --- | ---: | ---: | ---: |
| CPU | 1 | 16.58 | 31.37 |
| CPU | 4 | 26.10 | 93.61 |
| CPU | 8 | 24.51 | 96.11 |
| MPS | 1 | 3.73 | 5.56 |
| MPS | 4 | 4.26 | 6.59 |
| MPS | 8 | 3.66 | 10.01 |

The before/after runs differed in epoch count and were intentionally short, so
the ratios are directional. Still, the removal of Python packing overhead is a
clear improvement on both devices, and MPS batch eight improved by roughly
2.7x. CPU remains much faster for this small 128-wide, four-layer transformer.
The GPU has insufficient sustained matrix work to amortize its kernel launches
and remaining synchronization points. The next GPU-oriented work is vectorized
sparse loss and boundary-only validation; after that, a real-data benchmark
should decide the production device rather than assuming MPS is preferable.

## Vectorized-loss and boundary-validation result

A follow-up benchmark included representative sparse selection targets rather
than measuring only the dense InPlay head. It used 32 windows, 61 frames and
three candidates per frame, the production full-context model size, and three
epochs:

| Device | Batch | Windows/second |
| --- | ---: | ---: |
| CPU | 1 | 44.45 |
| CPU | 4 | 78.33 |
| CPU | 8 | 85.55 |
| CPU | 16 | 83.51 |
| MPS | 1 | 22.99 |
| MPS | 4 | 81.40 |
| MPS | 8 | 119.34 |
| MPS | 16 | 102.24 |

Batch eight is the measured sweet spot on both devices. For the current CPU
focus it is about 92% faster than batch one. MPS batch eight is now about 40%
faster than CPU batch eight, showing that the earlier GPU deficit came largely
from synchronization-heavy repository code rather than transformer size alone.
This is still a synthetic throughput result: a real experiment must verify
wall-clock improvement, memory, and model quality because larger batches take
fewer optimizer steps per epoch.

## Final safe-optimization acceptance

The final implementation keeps all structural preprocessing vectorized. An
initial Python-loop version of CPU packing metadata was rejected after it
regressed throughput, and an attempted branch-free loss was also rejected
because it evaluated masked frames unnecessarily. The accepted version uses
vectorized CPU collation and retains a single sparse-supervision guard.

The final three-epoch acceptance rerun at batch eight measured 63.59 windows/s
on CPU and 69.13 windows/s on MPS. These short M3 Air measurements vary with
warm-up and thermal state, so the durable conclusions are that batch eight
remains the preferred setting, CPU and MPS are now competitive, and real-run
wall time should select the device. Twenty-seven focused selector/joint tests
cover exact packing metadata, deterministic bucketing, batched-versus-single
prediction equivalence, sparse loss equivalence, and backpropagation.

Per the optimization scope decision, work stops here. Mixed precision and
`torch.compile` remain deliberately unimplemented.

## Batch-size verification sweep

A final sweep warmed each backend, randomized batch-size order, and measured
three repetitions of two training epochs over 32 representative full-context
windows with sparse selection and dense InPlay supervision:

| Batch | CPU median windows/s | MPS median windows/s |
| ---: | ---: | ---: |
| 1 | 70.50 | 33.21 |
| 2 | 83.46 | 65.33 |
| 4 | 100.68 | 75.08 |
| 8 | **104.72** | 123.83 |
| 12 | 103.76 | 131.97 |
| 16 | 99.74 | 133.89 |
| 24 | 100.63 | **136.10** |
| 32 | 92.35 | 105.85 |

CPU batch eight is the verified throughput winner, though batches 4–12 form a
narrow plateau. MPS batch 24 produced the highest median, but MPS measurements
were considerably noisier on the fanless Air. Batch 16 was within 2% of batch
24 and had the most consistent high-throughput results, making it the safer MPS
default. Batch 32 regressed on both devices. These are throughput settings, not
optimizer-equivalence guarantees; final model comparisons must account for the
different number of optimizer steps.
