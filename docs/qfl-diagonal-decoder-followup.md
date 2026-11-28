# QFL and diagonal serve-box factorial, with decoder follow-up

This note records the matched five-seed comparison of two proposed InPlay model
changes and the decoder retune that followed it. Development data only; the
November 27 holdout is used once for the final decoder confirmation.

## Summary

- Across 5 seeds × 4 held-out-camera folds, neither Gaussian QFL nor the diagonal
  serve-box features produce a statistically resolved interval-F1 gain with the
  production decoder frozen.
- QFL's only resolved effect is undesirable: false starts during reviewed breaks
  rise by 0.37/minute (95% CI +0.04 to +0.70).
- The diagonal serve-box arm is directionally the best model change (+0.011 F1,
  best end MAE and false-start rate, 14/20 fold wins), but its CI includes zero.
- The decoder is the largest resolved lever. A leave-one-camera-out gap sweep
  selects 2.0 s / 0.5 s and gives +0.045 F1 on development and +0.027 F1 on the
  holdout, with false starts roughly halved and rally ends ~2.7 frames later.
- Recommendation: keep the model recipe; adopt the decoder configuration as a
  post-processing-only change.

## Model factorial

Design: seven-source development data, four held-out-camera folds, seeds
1729–1733, 25 epochs, batch size 16. Every fold evaluation uses the fixed
production decoder (threshold 0.40, continuation threshold 0.35, maximum gap
0.5 s, minimum duration 0.3 s, edge preservation on). The all-source production
fit is skipped so each arm is compared on identical folds.

| Arm | Feature view | Boundary supervision |
| --- | --- | --- |
| baseline | `full_audio_zscore_motion` | triangle BCE |
| qfl | `full_audio_zscore_motion` | Gaussian quality focal loss |
| diagonal | `serve_boxes_diagonal` | triangle BCE |
| qfl-diagonal | `serve_boxes_diagonal` | Gaussian quality focal loss |

Five-seed means ± SE across seeds:

| Arm | Interval F1 | End MAE (frames) | False starts/min |
| --- | ---: | ---: | ---: |
| baseline | 0.8098 ± 0.0124 | 19.31 ± 0.61 | 0.77 |
| qfl | 0.8189 ± 0.0083 | 18.83 ± 0.53 | 1.15 |
| diagonal | **0.8207 ± 0.0094** | **18.48 ± 0.45** | **0.74** |
| qfl-diagonal | 0.8013 ± 0.0118 | 19.33 ± 0.98 | 0.95 |

Paired fold-seed deltas against baseline (n = 20, 95% CI):

| Arm | ΔF1 | Δ recall | Δ false starts/min | Δ end MAE |
| --- | ---: | ---: | ---: | ---: |
| qfl | +0.0091 [−0.0264, +0.0446] | +0.0244 [−0.0029, +0.0516] | **+0.3715 [+0.0416, +0.7015]** | −0.47 ns |
| diagonal | +0.0109 [−0.0203, +0.0420] | +0.0067 ns | −0.0288 ns | −0.83 ns |
| qfl-diagonal | −0.0085 [−0.0311, +0.0142] | +0.0049 ns | +0.1728 ns | +0.03 ns |

The diagonal arm improves F1 in 4/5 seeds; the pooled effect remains unresolved
(t₄ = 2.37, p ≈ 0.077). QFL's F1 direction is also 4/5 positive but one seed
reverses by −0.035, and its false-start increase is resolved.

## Decoder factorial

The zero-shot review of the probabilities showed that the 0.5 s production gap
was leaving interval F1 on the table. A preregistered grid varied only the gap
and minimum duration, holding threshold, continuation threshold, and edge
preservation fixed:

- `max_gap_seconds`: 0.75, 1.0, 1.5, 2.0
- `minimum_duration_seconds`: 0.3, 0.5
- control: production 0.5 / 0.3

Selection is leave-one-camera-out within each seed: choose the config on the
other three camera folds, evaluate it on the held-out camera. The deployment
config is the modal choice across the 20 seed × camera cells.

Development, production model arm (20 cells):

| gap / min | Interval F1 | Recall | False starts/min | End MAE |
| --- | ---: | ---: | ---: | ---: |
| 2.0 / 0.5 | **0.860** | 0.845 | **0.33** | 21.3 |
| 1.5 / 0.5 | 0.856 | 0.850 | 0.42 | 20.3 |
| 2.0 / 0.3 | 0.853 | 0.845 | 0.42 | 21.3 |
| 1.0 / 0.5 | 0.848 | 0.849 | 0.47 | 19.7 |
| 1.5 / 0.3 | 0.847 | 0.850 | 0.54 | 20.3 |
| 1.0 / 0.3 | 0.839 | 0.849 | 0.60 | 19.7 |
| 0.75 / 0.5 | 0.839 | 0.844 | 0.49 | 19.1 |
| 0.75 / 0.3 | 0.827 | 0.844 | 0.64 | 19.1 |
| 0.5 / 0.3 (production) | 0.810 | 0.841 | 0.77 | 19.1 |

LOCO-selected decoder: F1 0.854, false starts/min 0.38, end MAE 20.9.

- ΔF1 vs production: +0.0446 [+0.0284, +0.0608]
- Δ false starts/min: −0.3916 [−0.5197, −0.2635]
- Δ end MAE: +1.86 frames [+0.40, +3.31]

Modal choice: 2.0 / 0.5 (11/20), then 1.5 / 0.5 (5/20). The frozen deployment
config is 2.0 s / 0.5 s.

Holdout, one shot with the frozen config (saved production probabilities):

| Metric | Production 0.5 / 0.3 | Frozen 2.0 / 0.5 |
| --- | ---: | ---: |
| Pooled interval F1 | 0.855 | **0.882** |
| Match-macro interval F1 | 0.855 | **0.884** |
| Precision | 0.764 | **0.831** |
| Recall | **0.970** | 0.940 |
| False starts/min | 1.23 | **0.61** |
| False splits | 3 | 1 |
| False merges | 0 | 0 |
| End MAE (157 shared matches) | **25.45** | 28.11 |

The holdout reproduces the development direction. The tradeoff is later rally
ends (~2.7 frames, ~90 ms).

## Decision

Adopt the decoder change: maximum gap 2.0 s, minimum duration 0.5 s, with the
existing threshold, continuation threshold, and edge preservation. It requires no
retraining or checkpoint change. If end latency is a hard constraint, 1.5 / 0.5
is the development compromise (+0.046 F1, +1.2 frames predicted) but was not
holdout-confirmed.

Do not change the model recipe on this evidence. The diagonal serve-box features
remain the only model change worth a larger follow-up; QFL should not be adopted
because of the false-start regression.

## Reproduction

- Five-seed factorial: `tools/run_qfl_diagonal_followup.py`, then
  `tools/score_qfl_diagonal_followup.py`.
- Decoder development selection: `tools/run_decoder_factorial.py` over the saved
  fold predictions.
- Holdout confirmation: `tools/confirm_decoder_holdout.py` over the saved
  production probabilities and the frozen development JSON.

Caveats: the holdout had already been used for model-arm comparisons, so this is
a single frozen-decoder confirmation rather than a pristine second holdout. The
decoder grid was informed by an earlier exploratory sweep, but the selection
rule, deployment rule, and confirmation procedure were frozen before the
factorial run.
