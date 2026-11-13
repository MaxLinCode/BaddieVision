# Joint InPlay and Conditional Shuttle Selection

## Task contract

The temporal transformer predicts strict binary `InPlay` state on every frame
and selects a shuttle only on decoded in-play frames. Canonical rally labels are
inclusive from serve contact through the terminal event. Export padding is not
part of the label.

Outside rallies, candidate tokens remain model evidence but the shuttle target
is masked and the published outcome is `not_required`. Proposal recall is
defined only for in-play frames where the target shuttle is visibly
identifiable. A `missing_proposal` inside that denominator remains a TrackNet
failure.

## Annotation migration

Rally intervals use `source_id,rally_id,start_frame,end_frame` CSV plus a
`rally_interval_manifest` v1 JSON file. The manifest fixes the strict boundary
definition and records the CSV SHA-256, source video SHA-256, FPS, frame count,
and annotation revision. Overlapping intervals and mismatched sources fail.

Existing shuttle events are joined by source/frame. Labels inside intervals are
reused; labels outside intervals are retained for audit but masked from the
selection loss. The refill queue owns only still-unlabeled frames in the first,
last, and midpoint one-second regions of every rally.

## Model and data flow

- Dense two-second context windows have non-overlapping one-second ownership
  regions, so every source frame has exactly one state loss/metric owner.
- `SelectorBatch.inplay_targets` contains `0`, `1`, or `-100` for context and
  padding. Shuttle `targets` retain their existing frame-local semantics.
- `JointRallyShuttleModel` shares `TemporalShuttleEncoder` and exposes candidate,
  null, and InPlay heads. Selection and class-weighted binary losses are
  normalized independently and summed equally.
- Twenty percent whole-window candidate dropout masks selection supervision but
  retains state supervision, preventing candidate visibility from becoming the
  only rally cue.
- Overlapping state logits are decoded with a training-source calibrated
  threshold, short-gap merge, and minimum-duration policy. Candidates are never
  removed before the encoder.

Crossfit inference writes `predictions.jsonl`, the aligned
`rally_shuttle_predictions.jsonl`, strict `rallies.csv`, metrics, and one
checkpoint per fold. The per-frame artifact emits `selected`, `no_shuttle`, or
`not_required` and includes hashes plus selected proposal coordinates.

## Evaluation

Report frame and interval InPlay F1, IoU matching, boundary errors, false
splits/merges, conditional proposal recall, and retained-target accuracy. Always
report active positive frames with zero candidates and negative frames with one
or more candidates as shortcut-detection slices. The existing two sources are
a pipeline crossfit pilot; add another independent camera segment before making
production generalization claims.

This change does not modify TrackNetV3, InpaintNet, `clip_features/`, or the
ordered 76-input shot-classifier schema.
