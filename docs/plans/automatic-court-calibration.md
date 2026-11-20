# Automatic court calibration

Status: first implementation available; real-source acceptance calibration remains
pending. See `docs/automatic-court-calibration.md` for commands and limits.

Implemented: bounded median sampling and alignment diagnostics, classical
floor-line/template proposals, bounded robust fitting, optional-line browser
correction and orientation confirmation, canonical multi-frame publication, and
offline evaluation tooling. Automatic candidates require review until a labeled,
source-disjoint benchmark establishes acceptance thresholds. A learned detector
remains conditional on measured classical-detector failures.

## Goal and scope

Estimate one image-to-court homography per static source using distributed video
frames, with missing corners and partially occluded court lines supported. Start
with classical line detection and court-template fitting; evaluate before adding
a learned landmark or segmentation model.

Keep the existing manual point/line workflow as the correction path. An
underdetermined or ambiguous result must remain an unresolved calibration rather
than becoming a silently accepted projection.

## Existing foundation

- `src/court_projection.py` defines named landmarks, floor-plane calibration
  lines, finite rendered court segments, homography fitting, and overlays.
- `src/calibrate_court.py` provides browser point/line selection. Line mode
  currently infers intersections from at least two lines in each court direction.
- The current robust point fitter enumerates four-point subsets. Automatic
  observations will need bounded hypothesis generation rather than enumerating
  every subset of every frame's detections.
- The maintained workflow writes one canonical
  `<artifact_root>/court/calibration.json` and attaches source identity metadata.
  Its current native-registration API records one selected frame; automatic
  calibration needs explicit multi-frame provenance.

## 1. Define the landmark and observation contract

Reuse existing landmark names and the existing origin and axis convention.
Support these floor markings on both near/far and left/right sides:

- Doubles corners and singles-sideline/baseline intersections.
- Doubles long-service intersections with singles and doubles sidelines.
- Short-service intersections with singles and doubles sidelines.
- Center-service intersections with short-service lines, doubles long-service
  lines, and baselines. Add missing baseline-center names if needed.

Exclude the elevated net from automatic floor-plane evidence, including existing
`net_*` landmark names. Do not invent observed intersections where finite painted
segments do not meet. A mathematically inferred intersection of extended lines
must be marked as inferred rather than directly detected.

Represent point/line observations with identity (or candidate identities), pixel
geometry, confidence, source frame/background origin, and visible support.
Missing observations are absent; a predicted coordinate is not evidence of
visibility. Retain visible line segments even when their intersections are hidden
or outside the frame.

Use painted-line centers consistently for detected geometry. Audit their mapping
to the existing metric template before implementation; do not silently reinterpret
existing calibrations or change court coordinates.

## 2. Build a reproducible background estimator

Add a small background module, separate from the UI and geometric fitter.

- Sample a configurable number of frames uniformly across the source-local frame
  domain, with deterministic indices. Start experiments with 31 frames.
- Decode at bounded working resolution and compute a per-pixel temporal median.
  Keep the transform back to original image pixels explicit.
- Retain selected original frames for verification and recovery of markings lost
  in the median. Use per-window backgrounds as a fallback if persistent player
  occupancy affects the global median.
- Check sample geometry and cross-frame alignment. Reject camera cuts or material
  movement; the user must split moving-camera footage into separate source IDs.
- Record actual decoded indices, sample settings, working resolution, and source
  fingerprint. Parent-video trim offsets never enter these indices.

Cache the background as a reproducible diagnostic artifact; avoid loading the
full-resolution video into memory.

## 3. Detect lines and propose court correspondences

Add `src/court_detection.py` for the classical baseline:

1. Produce a line-evidence mask from local contrast/color and edges. Avoid a
   hard requirement for a particular court surface color.
2. Extract segments with a line-segment detector or Hough transform; merge
   fragments and estimate line centers, avoiding duplicate paint edges.
3. Group segments into the two projective court directions. Parallel world lines
   generally converge in the image; do not require image-space parallelism.
4. Generate bounded matches to the known court template using line ordering,
   intersections, and projective consistency. Test singles/doubles and
   baseline/long-service alternatives explicitly.
5. Use individual frames to confirm or supplement background observations.

Retain multiple candidate matches when evidence cannot distinguish them. Adjacent
courts and unrelated floor markings are explicit evaluation cases.

## 4. Fit, refine, and validate

Initialize from nondegenerate point correspondences or sufficient identified
lines. Four point correspondences in general position are the algebraic minimum,
not an automatic acceptance rule.

- Use bounded robust sampling with a deterministic seed and reject rank-deficient
  or poorly conditioned samples, including three-collinear minimal sets.
- Refine against visible finite court segments using robust point-to-line or
  distance-transform residuals. Balance support across distinct markings so one
  long line cannot dominate the fit.
- Do not count all intersections generated by one detected line as independent
  confidence votes. Likewise, repeated frames of one landmark improve its
  localization but do not add spatial coverage.
- Score spatial coverage, residuals, line support, held-out-frame consistency,
  plausible projected geometry, and the margin over competing templates.
- Distinguish unknown/unobserved markings from contradictory visible evidence;
  off-screen or occluded lines should not automatically invalidate a fit.
- Detect reflection/near-far ambiguity. Court symmetry cannot always resolve
  orientation visually; expose the convention and request correction when needed.
- Validate against evidence not used for initialization. Never accept based only
  on reprojection error from the fitted points.

Return a candidate plus diagnostics and a decision: accepted, needs review, or
failed. Choose numerical thresholds from the evaluation set and record them in
the algorithm configuration. Strong lens distortion remains outside the first
version and should produce poor-fit diagnostics rather than false confidence.

## 5. Integrate review and canonical publication

Extend the existing calibration command with an explicit automatic mode; preserve
manual modes. Proposed usage:

```sh
python3 -m src.workflow stage calibrate --source SOURCE --mode auto
```

Show the candidate overlay on the median background and distributed raw frames.
Allow correction using the existing draggable lines, optional landmarks, and
orientation controls. Review must permit missing observations instead of requiring
every configured guide. A headless uncertain run exits with diagnostics and leaves
calibration unresolved.

Keep candidates and diagnostic images separate from the authoritative calibration.
Publish accepted automatic or manually corrected results atomically to the sole
canonical path. Preserve an existing calibration unless replacement is explicitly
requested through the workflow's replacement mechanism.

Extend metadata registration for method/version, sampled source-local indices,
background parameters, fitting configuration, quality metrics, and review status.
Retain `image_size`, source identity/fingerprint, and compatibility with existing
calibration loaders and legacy manual calibrations. Retain a representative
`source_frame` if needed for old consumers, without disguising multi-frame origin.
Use existing calibration fingerprints to invalidate dependent artifacts when the
accepted calibration changes. Do not add a calibration path to the source catalog.

## 6. Verify and decide whether learning is necessary

Build a small representative evaluation set, grouped by static camera source:
clear courts, cropped corners, player occlusion, cluttered/multiple floor markings,
different viewpoints, adjacent courts, and negative examples/camera movement.

Manually annotate visible held-out intersections and line segments. Include
interior landmarks, not only the four points used to create a reference fit.
Separate threshold development sources from final evaluation sources.

Measure:

- Automatic acceptance rate and false acceptance rate, reported separately.
- Held-out image-space alignment error normalized by image size.
- Court-space projection error and spatial coverage, including near/far regions.
- Orientation/line-identity mistakes, review frequency, and runtime/memory.
- Improvement from single-frame to median-background to combined-frame evidence.

Add focused synthetic tests for known homographies, missing landmarks, occlusion,
outliers, degeneracy, inferred-intersection dependence, ambiguous templates, and
resolution scaling. Add integration checks for deterministic sampling, provenance,
canonical publication, uncertain-run behavior, and compatibility with existing
manual calibration files.

Use evaluation failures to decide whether a learned keypoint/line-segmentation
front end is justified. It should replace detection while keeping observation,
fitting, validation, review, and publication contracts intact.

## Delivery order

1. Observation/template contract, median background, and diagnostic previews.
2. Classical detector and candidate fitter, evaluated offline without publication.
3. Quality gates and benchmark report; establish acceptance thresholds.
4. Browser correction and workflow publication integration.
5. Learned detector only if measured failures justify it.

This work changes calibration production, not the shot-classifier feature schema.
Rebuilding an accepted calibration still requires rebuilding its dependent outputs
through the existing workflow.
