# Candidate Selection Transformer V1

## Summary

Build a high-recall TrackNet proposal pipeline, reusable Dash annotation platform, proposal-recall evaluation, and small temporal selector.

The model uses separate `FRAME` and `CANDIDATE` tokens. Candidate attribution remains independent of `InPlay`: select the target-match shuttle whenever visible, including between rallies.

Defer the InPlay head, TrackNet internal features, full-heatmap storage, tracklet/hypothesis inputs, and structured decoding.

## Key Changes

### 1. High-recall proposal extraction

- Introduce `shuttle_candidates` schema v2 while retaining v1 reader compatibility.
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
- Support candidate selection, `NO_SHUTTLE`, `MISSING_PROPOSAL`, skip, undo, and correction without pixel annotation.
- Store immutable append-only JSONL events containing revision ID, task, source/frame, label kind, candidate ID when applicable, candidate-artifact SHA-256, annotator/session, timestamp, and superseded revision.
- Reject labels whose artifact fingerprint or candidate ID cannot be resolved.
- Queue one-second native-FPS labeling bursts around adaptive anchors. Bootstrap prioritization from candidate count and weak TrackNet evidence; incorporate model uncertainty later.
- Maintain a separately seeded, immutable uniform audit queue so evaluation is not biased by adaptive sampling.

### 3. Recall and dataset contracts

- Treat `MISSING_PROPOSAL` as an extraction failure used for recall metrics, never as a model class.
- Exclude `NO_SHUTTLE` frames from the present-shuttle proposal-recall denominator.
- Report recall, missing-proposal rate, recall@K, candidates/frame, maximum count, component merging, and per-source results.
- Compile two-second windows containing the full native-frame candidate and context streams. Resolve durable candidate IDs to temporary tensor indices only inside the loader.
- Retain at most the frozen `Kmax` candidates using the evaluated ranking policy. Record selected labels outside K as `dropped_by_k` and mask them from selection loss.
- Join assigned P1/P2 geometry, court coordinates, confidence, MediaPipe pose, and calibrated court geometry by source and frame.
- Represent every optional feature group with explicit validity/missingness masks; zero-filled values never imply validity.
- Fail on missing or ambiguous calibration, invalid `image_size`, incompatible fingerprints, or frame-alignment errors. Missing players and poses remain valid through masks.
- Keep selector features independent of the fixed shot-classifier feature schema.
- Split strictly by source video, never by windows. Use identical split manifests and random seeds for all ablations.

### 4. Temporal selector

- Candidate features include normalized weighted/peak positions, normalized box geometry, normalized area, activation statistics, and validity masks.
- Do not include candidate IDs, tracklets, hypotheses, or pre-associated velocity/acceleration as model features.
- Encode candidate and frame inputs with separate MLPs into 128-dimensional tokens. Add token-type and continuous relative-time embeddings.
- Use a four-layer, four-head bidirectional transformer with 256-dimensional feed-forward blocks and `0.1` dropout.
- A temporal window spans two seconds: approximately one second before and one second after its center.
- Emit one frame token per frame and only the real retained candidate tokens. Do not materialize `Kmax` padding tokens inside each frame.
- Pad only the total packed token sequences needed for batching and mask that batch-level padding from attention and loss.
- Score candidates with a shared candidate head. Produce the frame-level `NO_SHUTTLE` logit from the corresponding frame token.
- Apply frame-wise masked cross-entropy over that frame’s real candidates plus `NO_SHUTTLE`, only where usable labels exist.
- Keep a frame token in every ablation:
  - Candidates-only: learned base frame embedding plus time/type embeddings, with no player, pose, or court input.
  - Players/court: player and court inputs with their validity masks.
  - Full context: players, court, and pose with their validity masks.
- Train all ablations under the identical source split, seed, optimizer policy, and evaluation protocol.

### 5. Window aggregation and output

- Run two-second inference windows with one-second stride.
- Aggregate repeated predictions by stable bookkeeping keys, never tensor slot:
  - `(frame_index, candidate_id) → candidate logit observations`
  - `frame_index → NO_SHUTTLE logit observations`
- IDs remain metadata and are never supplied to the model.
- Center-weight each logit observation according to its temporal position within the window. Use a linear weight of `1.0` at the center and `0.5` at either boundary so edge frames always retain nonzero coverage.
- Compute the weighted mean logit for each bookkeeping key, then construct the final frame-wise softmax from that frame’s aggregated candidate logits and aggregated `NO_SHUTTLE` logit.
- Write fingerprinted `shuttle_selections.jsonl` schema v1 with:
  - outcome enum: `selected`, `no_shuttle`, or `abstained`;
  - selected candidate ID and position when applicable;
  - aggregated candidate and `NO_SHUTTLE` logits;
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
- Test transformer padding, candidate permutation equivariance, frame-level null scoring, sparse loss, and identical ablation split/seed usage.
- Test aggregation with deliberately reordered candidate slots across windows and verify grouping by `(frame, candidate_id)`.
- Test center weighting, boundary coverage, frame-local softmax construction, and outcome-schema validation.
- Add a synthetic integration test that overfits a small temporal association problem and writes a valid selection artifact.
- Manually verify the Dash workflow on both existing videos, including synchronization, overlays, hotkeys, undo, restart recovery, and export.
- The recall milestone passes when the pilot identifies an acceptable threshold/K tradeoff; no fixed recall percentage is assumed beforehand.
- Report per-video macro candidate accuracy conditional on proposal availability, overall selection accuracy, `NO_SHUTTLE` precision/recall/F1, proposal recall@K, and ablation deltas.

## Assumptions

- `Kmax` is exclusively a dataset/model view over an untruncated durable proposal artifact.
- `NO_SHUTTLE` means the target-match shuttle is not visibly represented because it is absent or not visible.
- `MISSING_PROPOSAL` means the shuttle is visibly identifiable to the annotator but absent from the proposal set.
- A visible target-match shuttle is selected regardless of rally state.
- Candidate probabilities are frame-local because candidate-set cardinality varies.
- Optional InPlay multi-task learning begins only after the selector and its ablations establish a useful baseline.
