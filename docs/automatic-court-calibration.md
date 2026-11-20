# Automatic court calibration

The automatic path estimates a temporal-median background from distributed
source-local frames, detects court-line fragments, and proposes a homography by
matching known badminton geometry. Missing corners are supported: visible line
fragments can constrain intersections outside the image.

Automatic proposals now require a matched short-service line with a supported
center-service T junction to disambiguate its identity. If those distinctive
markings are hidden or cannot be matched, detection fails with
`insufficient_distinctive_court_markings`; manual calibration remains available.
The detector also checks finite segment overlap and unexplained paint inside the
projected court so neighboring-court extensions cannot count as supporting lines.

## Run

For a registered source, generate a proposal and review it in the browser:

```sh
python3 -m src.workflow stage calibrate --source SOURCE --mode auto --review --open-browser
```

The browser starts on the median background. Switch between Background and raw
frames to check alignment. Existing handles remain in place when changing frames.
Drag handles to correct detected lines, or select a named line and use Add line /
Remove line. At least two identified sidelines and two cross-court lines are
required; the four outer corners do not need to be visible. Line endpoints can be
any two points along the same visible painted line.

Check the complete projected court, particularly the singles/doubles sidelines
and baseline/long-service alternatives. Use Swap left/right or Swap near/far if
needed, then check Orientation checked before saving. A good point fit alone is
not evidence that line identities are correct.

Save publishes the reviewed result to the sole canonical
`<artifact_root>/court/calibration.json`. The workflow attaches source identity,
multi-frame provenance, settings, diagnostics, and review status atomically. It
does not change `config/sources.local.json` or the shot-classifier feature schema.
Use `--force` to replace an existing calibration. Calibration changes invalidate
dependent artifacts through the existing fingerprints; rebuild those outputs.

For an image/video that is not registered:

```sh
python3 -m src.calibrate_court VIDEO OUTPUT.json --mode auto --review --open-browser
```

The standalone command records the input hash and sample indices. Use
`stage import-calibration` to register a reviewed result for the source.
Automatic imports require completed review and a matching video fingerprint.

## Diagnostics and unattended runs

Omit `--review` to generate diagnostics without opening the browser. This initial
version always exits with status 2 without publishing an automatic calibration.
Its acceptance thresholds have not been validated on a representative,
source-disjoint real-video dataset, and court symmetry requires an orientation
convention. This is an explicit review requirement, not an unattended acceptance
mode.

Diagnostics default to `<artifact_root>/court/automatic/`:

- `background.jpg`: median background at working resolution.
- `candidate.json`: unreviewed candidate in original-image pixel coordinates,
  when detection succeeds. This is not a canonical calibration.
- `overlay.jpg` and `validation-*.jpg`: projected court on the background and
  distributed raw samples, at original image dimensions.
- `line-assignments.jpg`: observed segments labeled with their proposed identities,
  to expose baseline/service-line and singles/doubles mistakes during review.
- `court-selection.jpg`: ranked physical court footprints with the selected
  footprint in green; scores are ranking qualities, not probabilities.
- `diagnostics.json`: sampled indices, scale/geometry, alignment checks, detector
  scores, alternative-template ambiguity, and failure/review reasons.

Use `--diagnostics DIRECTORY` to choose a diagnostic directory. Its known output
files are replaced on each run so failed reruns cannot expose stale candidates.

Defaults are `--samples 31 --max-dimension 960 --max-hypotheses 65536`. Template
search uses a bounded, deterministic total hypothesis budget across global and
automatic local proposal searches. A 65,536-hypothesis run took about 30 seconds
on the initial local video trial. `--max-hypotheses 8192` is a faster option but
may leave difficult views unresolved. Sampling is deterministic and
includes source-local endpoints; short videos use each available frame at most
once. Images use one sample. Samples are resized before stacking; the stack has a
256 MiB limit. The retained raw samples also contribute to the median and are
verification views, not an independent labeled evaluation set.

## Main-court selection

Global hypotheses generate a bounded set of automatic local proposal regions.
These regions only help re-detect line fragments; they never hide original image
evidence during final validation. When no explicit target region is supplied,
half the hypothesis budget is reserved for these local proposals. Equivalent
line-identity interpretations are grouped by their clipped physical footprint
so they cannot consume the entire shortlist.

Each hypothesis must pass geometric checks before ranking: independent support,
service-line structure, finite segment overlap, line endpoint consistency, and
unexplained interior markings. Fully observable short-service markings need
paint support. A candidate occupying less than 10% of the visible image is
currently left unresolved. These conservative, provisional checks can reject
heavily occluded or very distant courts; manual calibration remains available.

Valid hypotheses are ranked with the following initial weights:

| Component | Weight |
| --- | ---: |
| Geometry: template support, explained paint, service-line topology | 70% |
| Independent named marking coverage across both court directions | 20% |
| Court polygon area clipped to the visible image | 7% |
| Clipped court centroid proximity to the image center | 3% |

Size and centrality cannot rescue rejected geometry. Refinement is checked by
the same gates before it can replace an initial fit. Diagnostics distinguish
competition between physical courts from alternative line identities on the
selected court. A missing runner-up produces a null margin, not evidence of
certainty. Review remains required while weights and acceptance thresholds lack
a labeled source-disjoint evaluation.

## Limits

One source must contain one static camera segment. Detected material camera
movement rejects the automatic run. Feature-poor alignment can be inconclusive;
review the raw frames rather than interpreting that as proof of a static camera.
Cuts, changing court layouts, persistent player occlusion, adjacent courts, low
line contrast, and strong lens distortion may need manual calibration or source
splitting. Camera movement requires a new source ID and calibration.

For multi-court scenes, restrict detection with `--court-region region.json`.
This file selects the visible target-court area; it is not a second calibration
or a change to the source catalog. Use original-image coordinates and a convex
polygon in boundary order:

```json
{
  "image_size": [1920, 1080],
  "polygon": [[20, 650], [900, 640], [1900, 1050], [150, 1050]]
}
```

These example coordinates are illustrative, not a calibration for a particular
video. Include the visible target court and exclude neighboring courts as far as
possible. The region is scaled for detection and embedded in calibration
provenance. It does not establish baseline/service-line identities by itself;
the resulting correspondence still needs review.

The metric template excludes the elevated net. Bright posts and reflections can
still enter the raw line mask and affect fitting or rejection; the template does
not make those pixels valid floor observations. Existing manual calibration files
and point/line commands remain supported.

## Evaluate on labeled sources

Create a manifest with separate development/evaluation sources. Landmark pixels
are measured in original image coordinates; reference calibration agreement is
reported separately from manually annotated landmark errors.

```json
{
  "sources": [
    {
      "source_id": "camera-a",
      "group": "evaluation",
      "video": "/path/to/source.mp4",
      "expected": "court",
      "heldout_landmarks": {
        "near_left_singles": [120, 690],
        "far_short_service_right": [800, 240]
      }
    }
  ]
}
```

```sh
python3 -m src.evaluate_court_calibration --manifest evaluation.json --output report.json
```

Paths can also be relative to the manifest. Include `expected: "no_court"` negative
examples and optionally `reference_calibration`. The report separates candidate
coverage, negative-source false candidates, runtimes, held-out pixel/metre errors,
and reference agreement. Unique source IDs and resolved video paths are enforced
across groups; verify shared-camera and duplicate-file provenance when assembling
the dataset. The evaluator never publishes calibrations or selects thresholds.

## Initial validation

The initial calibration and downstream workflow regression run passed 139 tests.
Synthetic full, cropped, corner-occluded, and strongly oblique courts recover
held-out interior projections within 7 pixels at the tested working dimensions.
Browser interaction checks covered optional-line correction, background switching,
orientation changes, and successful reviewed submission.

A local smoke test on `IMG_1048.mov` (1920×1080, 36,827 frames) produced a candidate
from seven distributed samples at 960-pixel working width. This is one unlabeled
development source, not an accuracy benchmark. A separate 31-sample CLI run
matched eight markings, flagged template ambiguity and weak temporal support,
returned status 2, and left the calibration unpublished. The labeled,
source-disjoint acceptance benchmark remains outstanding; review is required.

Visual review of that initial candidate exposed a baseline/long-service-line
assignment error and contamination from the adjacent right-hand court. A large
number of matched markings or a small fitted residual did not establish a correct
calibration. Finite segment overlap, evidence left unexplained inside the court,
and an optional explicit target region address those failure modes; the original
candidate must not be treated as a reference calibration.

After the consistency checks were tightened, the same unrestricted 31-frame
`IMG_1048.mov` run fails with `contradictory_interior_court_markings`, leaves no
candidate file, and publishes no calibration. This is a rejection of an invalid
fit, not a claim that this clip's calibration has been solved automatically.

## Geometry-first trial

The geometry-first implementation passes 88 focused calibration, selection,
artifact, evaluation, and workflow tests. Its synthetic two-court scene selects
the main court without a manual region and recovers held-out interior points
within 0.6 pixels. Tests also cover clipped visible area, physical-court shortlist
diversity, and retaining a valid runner-up after rejecting a stronger invalid fit.

An unrestricted 31-frame run on `ff_bothell-seg5.mp4` with the new 65,536-hypothesis
default selects the central foreground court with nine matched markings. Its
ranking components are approximately geometry 0.927, coverage 0.917, visible image
area 0.235, and centrality 0.635, for an overall quality of 0.868. These are internal
ranking values, not measured probabilities or labeled projection accuracy. No
manual region or reference calibration was read by this run. It remains an
unreviewed proposal and publishes no canonical calibration.

`IMG_1048.mov` remains unresolved under the new search, rather than reviving its
known incorrect candidate. A broader ML-dependent regression run was interrupted
during native dependency import before tests executed; it is not included in the
88 passing focused checks.

## Perspective-aware rejection diagnostics (v4)

Debugging `max_vs_lucas_6-19` showed that its cropped near baseline was already
unobserved. The false veto came primarily from fixed 9--26 pixel endpoint probes
at the compressed far end: those probes crossed other court markings and were
mistaken for paint continuing beyond a baseline. Some promising four-line seeds
were also rejected before their independently matched lines could refine them.

Endpoint probes now cover a fixed physical neighborhood, 0.15--0.70 metres, with
local projected paint-width tolerances. A branch whose perpendicular extent is
too small to resolve is reported as `unresolved_perspective`; a cropped branch is
`outside_observed_region`. Neither is contradictory paint evidence. Fits with
sufficient independent lines may refine before rejection, but the refined result
must pass the same final gates on original image evidence.

Clipped-footprint overlap also normalizes subpixel shared-edge noise and uses a
float64 clipping fallback if OpenCV reports asymmetric intersections. This keeps
line-identity alternatives on the same court from becoming separate physical
court candidates merely because both footprints touch the image border.

Inspect `detection.selection.rejection_gates` for specific first-failing checks,
and `rejected_candidates` for bounded seed examples with their transforms and
gate details. `junction_probe_diagnostics` in candidate consistency explains
unobserved and unresolved endpoint checks. The v4 run on `max_vs_lucas_6-19`
produces a reviewable nine-marking proposal, with line-identity ambiguity still
flagged; the existing canonical calibration remains unchanged. The existing
`ff_bothell-seg5` trial also retains its nine-marking proposal.

The v4 focused regression run passes 91 tests, including cropped near baselines,
unresolvable far-end probes, symmetric shared-edge footprint overlap, refinement
recovery, and the earlier baseline/service-line and neighboring-court failures.
