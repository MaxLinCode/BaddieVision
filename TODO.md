# TODO

## Court-space shot-classifier rollout

- [ ] Copy `config/court_calibrations.example.json` to
  `features/court/calibrations.json`.
- [ ] Calibrate every static source video and save each calibration under
  `features/court/` with its recorded `image_size`.
- [ ] Complete `clip_overrides` for legacy clip names that do not contain a
  source-video ID.
- [ ] Regenerate all shot-classifier arrays with `src/extract_clip_features.py`
  and confirm they have shape `(36, 76)`.
- [ ] Retrain the shot classifier because existing 73-input checkpoints are
  incompatible.
- [ ] Review source-grouped validation metrics and compare them with the
  previous classifier baseline.

## InPlay player tracking

- [ ] After merging player-slot stability, rerun the player tracking / pose
  extraction step for representative videos and inspect `players.csv` plus
  `player_poses.jsonl` to confirm stable P1/P2 slots and calibration fallback
  diagnostics. Shot-classifier `clip_features/` regeneration is not required
  for this change.

## Joint InPlay validation follow-up

Current candidate-free cross-camera baseline: six reviewed sources across four
camera groups, 126 strict rally intervals, and 52,012 owned dense targets
(19,427 positive and 32,585 negative). The clean model reaches macro ROC AUC
0.88974, PR AUC 0.82313, and interval F1 0.41915; interval fragmentation and
camera-transfer calibration remain the main limitations.

- [x] Run the predeclared within-camera learning sequence for the Max-vs-Nik
  chronological split (seed 1729, 25 epochs, batch size 16, relative pose plus
  candidates, and no InPlay boundary weighting):
  1. checkpoint control: uniform sampling, no auxiliary heads, restore the
     lowest-validation-InPlay-loss epoch;
  2. sampling ablation: add boundary-balanced sampling;
  3. boundary-head ablation: retain balanced sampling and add auxiliary
     rally-start/rally-end heads at weight 0.25.
  Compare validation InPlay BCE/AUC and interval diagnostics, choose and freeze
  one model on validation, calibrate the unchanged decoder on validation, and
  evaluate the fixed test segment once without further tuning.
  The boundary-head variant won validation (BCE 0.16296, ROC AUC 0.98521);
  the frozen decoder configuration was threshold 0.60, maximum gap 0.50s, and
  minimum duration 0.30s. Final test interval F1 was 0.70 with 7/8 rallies
  matched, no false splits or merges, and 13.93-frame mean absolute boundary
  error.

- [ ] Run the leakage-safe Max-vs-Nik within-camera ablation matrix documented
  in `docs/within-camera-inplay.md`; compare temporal-only, player/court,
  image-pose, player-relative-pose, candidates-only, and candidates-plus-pose
  using the fixed chronological test block.

- [x] Mark all six reviewed videos as full-source rally coverage; the combined
  review produced 126 strict rally intervals, including one partial start.
- [x] Recompile the dense dataset and assert 52,012 owned binary state targets:
  19,427 positive, 32,585 negative, and no unexpected `-100` targets. These
  counts match the 126 authoritative strict rally intervals.
- [x] Run the expanded four-camera-group experiment with the frozen seed,
  batch size, device policy, and 25-epoch recipe. The clean model improved to
  macro raw BCE 0.59745, ROC AUC 0.88974, PR AUC 0.82313, interval F1 0.41915,
  and 26.11-frame boundary MAE.
- [x] Add per-epoch held-out binary loss, precision, recall, F1, negative-frame
  false-positive rate, ROC AUC, and boundary-window loss curves.
- [x] Run candidate-free and candidate-inclusive `InPlay`-only arms with
  shuttle-selection loss disabled. Candidate tokens did not improve interval
  F1 and reduced macro ROC/PR AUC, so the candidate-free model remains the
  maintained rally segmenter.
- [ ] Improve held-out false-positive diagnostics before further loss weighting
  or transition-frame oversampling:
  - report true positive prevalence and predicted-positive rate per epoch;
  - report specificity / true-negative rate and balanced accuracy;
  - report false-positive rate across fixed probability thresholds, not only
    the current `0.5` operating point;
  - report per-source metrics within each held-out camera group;
  - distinguish ranking failure from calibration failure using ROC AUC plus
    precision-recall and threshold-sweep summaries;
  - keep raw-threshold diagnostics separate from calibrated temporal-decoder
    metrics.
- [ ] After the false-positive diagnostics are available, run the true
  no-shuttle-input baseline without boundary weighting, then compare configurable
  positive-class weights (`1.0`, capped, and natural imbalance weighting) before
  testing more boundary weighting or transition oversampling.
- [ ] If held-out ranking remains poor, localize camera-domain leakage with
  full-context, player/court-only, temporal-prior-only, and player-relative-pose
  InPlay ablations; inspect calibration, assignment-confidence, missing-pose,
  and geometry distributions by camera group.
  - Add an explicit, fingerprinted `player_relative` pose representation: center
    each player on the pelvis (bbox-center fallback), normalize by torso/shoulder
    scale (bbox-height fallback), retain visibility, and keep absolute court
    position, bbox scale, activity, and assignment confidence as separate
    features. Do not silently change the existing 154-feature semantics.
  - Compare image-space and player-relative pose both with and without shuttle
    candidate inputs; consider left/right-safe reflection augmentation and
    normalized joint velocities only after the static representation ablation.
- [ ] Plot reliability and precision-recall curves and evaluate whether decoder
  calibration transfers across held-out cameras.
- [ ] If dense negative supervision and calibration do not sharpen transitions,
  test boundary weighting, an auxiliary start/end head, ambiguous-boundary
  masking, and causal versus centered temporal context.
