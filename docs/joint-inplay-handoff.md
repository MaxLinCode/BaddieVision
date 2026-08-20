# Joint InPlay and Shuttle Selection Handoff

## Summary

The shuttle task was redefined so continuous localization is no longer required
between rallies. The temporal transformer now supports two related outputs:

1. Strict binary `InPlay` state for every frame.
2. Shuttle selection only for frames decoded as in play.

A held, stationary, or stray shuttle outside a rally is now useful negative
context for the `InPlay` head rather than a TrackNet proposal-recall failure.
TrackNet proposal recall remains meaningful inside rallies when the target
shuttle is visibly identifiable.

## What Was Implemented

### Joint model contract

- Added `SelectorBatch.inplay_targets` with values `0`, `1`, or `-100` for
  context/padding.
- Added `JointRallyShuttleModel` with a binary `InPlay` head alongside the
  existing candidate-selection and null-selection heads.
- Added independently normalized selection and class-weighted binary losses.
- Added deterministic whole-window candidate dropout on 20% of joint-training
  windows. Selection supervision is masked on dropped windows while `InPlay`
  supervision remains active.
- Kept the legacy `TemporalShuttleSelector` parameter contract unchanged so
  existing selector checkpoints do not acquire a new required head.

### Rally annotations and label reuse

- Added strict inclusive rally intervals:
  - Start: serve racket–shuttle contact.
  - End: terminal landing, net contact, fault, or other event making a return
    impossible.
- Added a fingerprinted interval manifest containing the CSV hash, video hashes,
  FPS, frame counts, boundary definition, and annotation revision.
- Interval loading fails on overlaps, invalid ranges, unknown sources, or
  fingerprint/alignment mismatches.
- Existing shuttle labels are reused when they fall inside a rally.
- All shuttle-selection targets outside rallies are masked rather than converted
  to null selections.
- Added a refill queue covering the first, last, and midpoint one-second regions
  of each rally while excluding already labeled and reserved audit frames.
- Added an immutable rally-audit extension for rallies without existing uniform
  audit coverage.

### Dataset, inference, and evaluation

- Added dense two-second context windows with non-overlapping one-second loss
  ownership, giving every source frame exactly one state target owner.
- Kept all candidate tokens available to the transformer; predicted `InPlay` is
  never used to remove evidence before encoding.
- Added center-weighted aggregation of overlapping state, null, and candidate
  logits. Candidate observations are joined by stable candidate ID rather than
  tensor slot.
- Added training-source decoder calibration over probability threshold,
  short-gap merging, and minimum rally duration.
- Joint inference writes:
  - `predictions.jsonl`
  - `rally_shuttle_predictions.jsonl`
  - `rallies.csv`
  - `metrics.json`
  - one checkpoint per crossfit fold
- Outside decoded rallies, the published shuttle outcome is `not_required` and
  no coordinate is emitted.
- Added frame and interval `InPlay` metrics, boundary error, split/merge counts,
  conditional proposal recall, selector accuracy, and shortcut-detection slices.

### Compatibility and documentation

- TrackNetV3 and InpaintNet were not retrained or modified.
- The ordered 76-input shot-classifier schema and existing `clip_features/`
  artifacts were not changed.
- The previous candidate-selector plan is retained as provenance and marked as
  superseded for production task semantics.
- Added usage documentation in `docs/annotation-platform.md` and
  `docs/plans/joint-inplay-conditional-shuttle.md`.

## Verification Completed

- Ruff passes for the changed Python packages and tests.
- All 199 repository tests pass.
- Diff whitespace and selector configuration JSON validation pass.
- The real frozen sources were compiled successfully into 180 dense windows with
  aligned state targets.

## What Needs to Be Done Next

### 1. Annotate strict rally intervals

Run the interval labeler once for each current selector source. Reuse the same
CSV and manifest paths; writing the second source preserves the first source.

```bash
python3 -m InPlay.heuristic.label_intervals \
  --video outputs/SOURCE/SOURCE_input.mp4 \
  --source-id SOURCE \
  --output .annotation-final/rallies.csv \
  --manifest .annotation-final/rallies.manifest.json
```

Review boundaries at native frame rate. Do not include pre-serve preparation or
post-rally padding in the canonical intervals.

### 2. Reserve and complete the rally audit

Build this before the refill queue so audit frames remain uncontaminated.

```bash
python3 -m src.annotation_platform \
  --config config/annotation-sources.local.json \
  --runtime .annotation-final \
  build-rally-audit \
  --intervals .annotation-final/rallies.csv \
  --interval-manifest .annotation-final/rallies.manifest.json

python3 -m src.annotation_platform \
  --config config/annotation-sources.local.json \
  --runtime .annotation-final \
  serve --annotator NAME --queue rally-audit
```

### 3. Build and complete the in-rally refill queue

```bash
python3 -m src.annotation_platform \
  --config config/annotation-sources.local.json \
  --runtime .annotation-final \
  build-rally-refill \
  --intervals .annotation-final/rallies.csv \
  --interval-manifest .annotation-final/rallies.manifest.json

python3 -m src.annotation_platform \
  --config config/annotation-sources.local.json \
  --runtime .annotation-final \
  serve --annotator NAME --queue refill
```

### 4. Update the experiment configuration

Set these currently null fields in `config/selector-experiment.local.json`:

```json
"rally_intervals_path": "../.annotation-final/rallies.csv",
"rally_manifest_path": "../.annotation-final/rallies.manifest.json"
```

Also add these completed queues to `dataset.queue_paths`:

```json
"../.annotation-final/queues/shuttle-rally-audit.json",
"../.annotation-final/queues/shuttle-refill.json"
```

### 5. Run the two-source pilot

```bash
python3 -m src.temporal_selector.experiment \
  --config config/selector-experiment.local.json \
  --output-dir outputs/joint-selector-run \
  --context-mode full_context
```

Check the following before accepting the pilot:

- Source-disjoint frame and interval `InPlay` F1.
- Start/end boundary errors and false split/merge counts.
- `InPlay` recall on active frames with zero TrackNet candidates.
- False-positive rate on out-of-play frames that contain candidates.
- In-play proposal recall and retained-target selector accuracy.
- Selector accuracy should remain within two percentage points of the
  selector-only baseline.

### 6. Add more source diversity before production use

The current two-source crossfit validates the pipeline but is not sufficient for
a production generalization claim. Label at least one additional independent
source/static-camera segment, regenerate the source-disjoint split, and rerun the
joint experiment before promoting a checkpoint.

## Operational Notes

- Keep clip padding in the exporter; do not expand canonical `InPlay` labels.
- One source ID must continue to represent one static camera segment.
- Do not regenerate shot-classifier features or checkpoints for this change.
- Do not interpret poor held/stationary shuttle recall outside rallies as a
system error under the new task definition.
