# Candidate Selection Transformer V1

## Summary

Build a high-recall TrackNet proposal pipeline, reusable Dash annotation platform, proposal-recall evaluation, and small temporal selector.

The model uses separate `FRAME` and `CANDIDATE` tokens. Candidate attribution remains independent of `InPlay`: select the target-match shuttle whenever visible, including between rallies.

Defer the InPlay head, TrackNet internal features, full-heatmap storage, tracklet/hypothesis inputs, and structured decoding.

## Key Changes

### 1. High-recall proposal extraction

- Introduce `shuttle_candidates` schema v2 while retaining v1 reader compatibility.
- Selector candidates are the direct, pre-InpaintNet TrackNet proposals. InpaintNet output is not an input to candidate extraction, annotation, or selector training.
- Existing candidate artifacts already satisfy the pre-InpaintNet contract and are not regenerated for this change.
- From each TrackNet heatmap, simultaneously extract 8-connected components at thresholds `0.2`, `0.3`, `0.4`, and `0.5`.
- Preserve every component and every empty frame in durable artifacts; never apply extraction-time `Kmax`.
- Preserve existing `center` semantics and add peak position, activation-weighted centroid, normalized bounding-box geometry, component area, peak/mean/total activation, and normalized forms.
- Normalize area by heatmap area. Normalize image-space coordinates and boxes by image dimensions.
- Record checkpoint hash, TrackNet sequence/background/preprocessing configuration, overlap ensemble mode, threshold, extraction version, FPS, image/heatmap sizes, and source frame range.
- Rank candidates for `Kmax` retention by peak activation descending, mean activation descending, normalized area descending, then stable candidate ID.
- Evaluate `K ∈ {1,2,3,5,8,12,all}` and report proposal recall for this exact retention policy.
- Freeze threshold, ranking policy, and `Kmax` after the recall pilot and before large-scale training annotation.

### 2. General annotation platform

- Build a local Dash application with shared source registration, task registration, queues, sessions, keyboard controls, revision replay, audit views, and export APIs.
- Define task plugins that provide payload validation, overlays, label options, and queue scoring. Implement shuttle selection as the first plugin.
- Show a clip covering one second before and after the center frame, with candidate overlays throughout and an exact center-frame selection view.
- Support candidate selection, `NO_IN_FRAME_TARGET`, `MISSING_PROPOSAL`, `OCCLUDED_INFERABLE`, `UNSURE`, skip, undo, and correction without pixel annotation.
- Store immutable append-only JSONL events containing revision ID, task, source/frame, label kind, candidate ID when applicable, candidate-artifact SHA-256, annotator/session, timestamp, and superseded revision.
- For every new schema-v2 selected event, resolve the selected candidate from the fingerprint-verified candidate artifact on the server and snapshot:
  - `candidate_position.coordinate_space = "normalized_image_xy"`;
  - `candidate_position.canonical_field = "peak_position_normalized"`;
  - finite `[0,1]` values for `peak_position_normalized`, `weighted_centroid_normalized`, and `center_normalized`.
- Never accept client-supplied candidate coordinates. Reject selected labels whose artifact fingerprint, candidate ID, or required normalized position cannot be resolved and validated. Non-selected labels carry no `candidate_position`.
- Keep historical event logs append-only. Migration and export may enrich an old selected label only when its exact candidate artifact is available; otherwise preserve the label and mark its coordinates unavailable.
- Queue one-second native-FPS labeling bursts around adaptive anchors. Bootstrap prioritization from candidate count and weak TrackNet evidence; incorporate model uncertainty later.
- Maintain a separately seeded, immutable uniform audit queue so evaluation is not biased by adaptive sampling.

### 3. Recall and dataset contracts

- Treat `MISSING_PROPOSAL` as an extraction failure used for recall metrics, never as a model class.
- `NO_IN_FRAME_TARGET` supplies supervised `NULL_SELECTION`. Mask `MISSING_PROPOSAL`, `OCCLUDED_INFERABLE`, `UNSURE`, and legacy `no_shuttle` labels from selector loss.
- Exclude `NO_IN_FRAME_TARGET` frames from the present-shuttle proposal-recall denominator.
- Report recall, missing-proposal rate, recall@K, candidates/frame, maximum count, component merging, and per-source results.
- Compile two-second windows containing the full native-frame candidate and context streams. Resolve durable candidate IDs to temporary tensor indices only inside the loader.
- Retain at most the frozen `Kmax` candidates using the evaluated ranking policy. Record selected labels outside K as `dropped_by_k` and mask them from selection loss.
- Join assigned P1/P2 geometry, court coordinates, confidence, MediaPipe pose, and calibrated court geometry by source and frame.
- Represent every optional feature group with explicit validity/missingness masks; zero-filled values never imply validity.
- Fail on missing or ambiguous calibration, invalid `image_size`, incompatible fingerprints, or frame-alignment errors. Missing players and poses remain valid through masks.
- Keep selector features independent of the fixed shot-classifier feature schema.
- Split strictly by source video, never by windows. Use identical split manifests and random seeds for all ablations.

### 4. Temporal selector

- Provide a standalone selector package with configurable `SelectorConfig`, validated `SelectorBatch`, `TemporalShuttleEncoder`, separate selection/null heads, and frame-local sparse cross-entropy. Define this provisional batch contract independently of the real dataset loader. The core accepts only numeric tensors and masks; durable candidate IDs remain loader/bookkeeping metadata.
- Each candidate supplies exactly 12 numeric features: normalized weighted-centroid `(x, y)`, normalized peak `(x, y)`, normalized box `(x_min, y_min, x_max, y_max)`, normalized area, and normalized peak/mean/total activation. Concatenate explicit feature-validity bits inside the candidate encoder.
- Do not include candidate IDs, tracklets, hypotheses, or pre-associated velocity/acceleration as model features.
- Encode candidate and frame inputs with separate encoders. Add a learned token-type embedding and a two-layer continuous-time embedding. Supply `relative_time_seconds`, centered at zero, for every token; do not add ordinal positional embeddings, so candidates within a frame remain permutation-equivariant.
- Use a configurable `SelectorConfig`. Baseline defaults are 128-dimensional tokens, four transformer layers, four attention heads, 256-dimensional feed-forward blocks, GELU, `0.1` dropout, pre-norm, and batch-first operation. Token size, layer/head counts, feed-forward size, activation, dropout, and normalization behavior remain configurable.
- The `TemporalShuttleEncoder` is non-causal: tokens may attend to context on both sides of time zero.
- A temporal window spans two seconds: approximately one second before and one second after its center.
- Pack one frame token followed by that frame's real retained candidate tokens. Do not materialize `Kmax` padding tokens inside each frame.
- Pad only the total packed token sequences needed for batching and mask that batch-level padding from attention and loss.
- Keep the encoder separate from the selection and null heads so a future trajectory head can consume encoded tokens without changing selection loss. Score all candidates with one shared selection head and derive each frame's null logit from its frame token.
- Represent each frame target as a frame-local candidate index for `SELECTED_PROPOSAL`, `-1` for supervised `NULL_SELECTION`, or `-100` for masked/unsupervised frames. Supervision may be sparse on any frame; a dataset adapter may initially supervise only the center frame without constraining the model core.
- Resolve candidate targets to packed token positions inside the loss and remap `-1` to that frame's null-logit position. Reject targets that name a nonexistent candidate, batch padding, or a candidate belonging to another frame.
- Apply sparse cross-entropy independently over each supervised frame's real candidate logits plus its null logit, then average over supervised frames only. A completely unsupervised batch has no supervised-frame contribution.
- Keep a frame token in every ablation:
  - `candidates_only`: learned base frame representation plus time/type embeddings, with no player, pose, or court input.
  - `players_court`: configured player and court values with their corresponding validity masks.
  - `full_context`: configured player, court, and pose values with their corresponding validity masks.
- Milestone 3 owns the concrete frame-feature views and dimensions. The model consumes the configured frame values and validity masks without embedding those loader decisions into its contract.
- Train all ablations under the identical source split, seed, optimizer policy, and evaluation protocol.
- Do not repair occluded coordinates upstream. A future trajectory head may learn from visible peak-coordinate labels to infer occluded positions, but trajectory prediction and trajectory loss are deferred.

### 5. Window aggregation and output

- Run two-second inference windows with one-second stride.
- Aggregate repeated predictions by stable bookkeeping keys, never tensor slot:
  - `(frame_index, candidate_id) → candidate logit observations`
  - `frame_index → null-logit observations`
- IDs remain metadata and are never supplied to the model.
- Center-weight each logit observation according to its temporal position within the window. Use a linear weight of `1.0` at the center and `0.5` at either boundary so edge frames always retain nonzero coverage.
- Compute the weighted mean logit for each bookkeeping key, then construct the final frame-wise softmax from that frame’s aggregated candidate logits and aggregated null logit.
- Milestone 5 writes fingerprinted `shuttle_selections.jsonl` selected outputs with:
  - outcome enum: `selected`, `no_shuttle`, or `abstained`;
  - selected candidate ID for provenance and frame-local bookkeeping only;
  - all three normalized selected positions (`peak_position_normalized`, `weighted_centroid_normalized`, and `center_normalized`), with peak position identified as canonical;
  - aggregated candidate and null logits;
  - derived frame-local softmax probabilities, explicitly marked as comparable only within that frame’s candidate set;
  - calibration status/method;
  - source artifact hashes, checkpoint hash, and inference configuration.
- Initial V1 decoding emits only `selected` or `no_shuttle`. Reserve `abstained` for a future confidence/margin policy without requiring a schema migration.
- Keep this JSONL artifact as the only new selector output; do not replace or export over `tracks.csv`.

## Test and Acceptance Plan

- Unit-test multi-threshold extraction, normalized statistics, deterministic ranking, empty frames, provenance hashes, and v1/v2 compatibility.
- Test recall@K using the exact production retention policy.
- Test annotation revision replay, interrupted writes, artifact mismatch rejection, burst generation, and separation of adaptive and audit queues.
- Test dataset joins, calibration failures, validity masks, K truncation, sparse labels, packed tokens, labels outside K, and source-disjoint splits.
- Test configurable baseline capacity, output shapes and CPU backpropagation, batch-padding invariance, candidate permutation equivariance, non-causal influence from both sides of zero, centered relative times, frame-level null scoring, and all three context/validity-mask modes.
- Test center and non-center sparse supervision, zero-candidate frames with valid null targets, frame-local target-to-packed-token resolution, completely unsupervised batches, and invalid candidate-index rejection.
- Test aggregation with deliberately reordered candidate slots across windows and verify grouping by `(frame, candidate_id)`.
- Test center weighting, boundary coverage, frame-local softmax construction, and outcome-schema validation.
- Add a synthetic integration test that overfits a small temporal association problem and writes a valid selection artifact.
- Manually verify the Dash workflow on both existing videos, including synchronization, overlays, hotkeys, undo, restart recovery, and export.
- The recall milestone passes when the pilot identifies an acceptable threshold/K tradeoff; no fixed recall percentage is assumed beforehand.
- Report per-video macro candidate accuracy conditional on proposal availability, overall selection accuracy, `NULL_SELECTION` precision/recall/F1, proposal recall@K, and ablation deltas.

## Assumptions

- `Kmax` is exclusively a dataset/model view over an untruncated durable proposal artifact.
- `NULL_SELECTION` is the selector model/loss term. Only `NO_IN_FRAME_TARGET` annotation semantics supply that supervised target.
- `MISSING_PROPOSAL` means the shuttle is visibly identifiable to the annotator but absent from the proposal set.
- Occluded coordinates are neither repaired upstream nor used as selector coordinate targets.
- A visible target-match shuttle is selected regardless of rally state.
- Candidate probabilities are frame-local because candidate-set cardinality varies.
- Optional InPlay multi-task learning begins only after the selector and its ablations establish a useful baseline.
