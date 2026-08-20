# Independent cross-camera rally segmenter

`RallySegmenter` is the offline, centered InPlay model. It consumes only the
154-value player/pose frame representation and its validity mask. Its batch and
model contracts contain no shuttle candidates, candidate presence, candidate
coordinates, selector logits, or selector modules.

The fixed frame schema is:

1. P1 and P2 bounding boxes, feet, normalized court position, detector
   confidence, assignment confidence, and activity (22 values);
2. P1 and P2 MediaPipe poses, pelvis centered and normalized by
   shoulder/torso scale with bounding-box fallbacks (132 values).

Missing players and landmarks remain zero-valued with explicit false validity.
The dataset reads player assignments, pose caches, court calibration, video
provenance, and strict rally annotations directly. It does not load a shuttle
candidate artifact. This schema is separate from the shot-classifier schema.

## Matched camera-held-out experiment

Run the experiment on the host so Apple MPS is visible:

```bash
PYTHONPATH=. .venv/bin/python \
  -m src.temporal_selector.rally_experiment \
  --config config/selector-experiment.local.json \
  --output-dir outputs/independent-rally-camera-held-out-v1 \
  --device mps
```

The recipe is frozen in code: seed 1729, 25 epochs, batch size 16,
boundary-balanced sampling, start/end auxiliary weight 0.25, AdamW at `3e-4`,
and gradient clipping at 1.0.

Each of Malaysia, Max-vs-Nik, and Bothell is held out once. Within a fold, every
training camera is split independently at 80% chronology. One-second guard
bands and whole-window containment keep temporal context out of the other
partition. Validation selects the lowest raw-BCE epoch and searches the fixed
threshold/gap/minimum-duration decoder grid. Test labels are first materialized
after both choices are frozen.

Every fold reruns:

- the existing `JointRallyShuttleModel` with all candidate inputs and selection
  targets masked;
- the candidate-independent `RallySegmenter`.

`metrics.json` includes per-source, per-camera, and macro frame/calibration/
interval metrics, the seed-1729 10,000-sample stratified rally-cycle bootstrap,
and each predeclared acceptance check. The held-out prediction JSONL files make
the pairing auditable.

The clean checkpoint records its model type, feature schema/version, dataset
and annotation fingerprints, `offline_centered` context direction, split
manifest, frozen training recipe, chosen epoch, and decoder configuration.
Legacy joint checkpoint reconstruction remains unchanged.

If all clean-model guardrails pass, the runner trains the production model on
all five sources for the median fold-selected epoch. Its decoder is calibrated
from the concatenated out-of-fold predictions. If they do not pass,
`stable_shuttle_track_follow_up_allowed` is false and no shuttle fusion should
be trained.

## Expanded maintained baseline

The subsequent maintained run uses six reviewed sources grouped into Malaysia,
Max-vs-Nik, Bothell, and Max-vs-Lucas cameras:

```bash
PYTHONPATH=. .venv/bin/python \
  -m src.temporal_selector.rally_experiment \
  --config config/rally-expanded-six-source.local.json \
  --output-dir outputs/clean-rally-expanded-six-source-v1 \
  --device mps
```

It contains 126 strict rally intervals and 52,012 owned frame targets. Its
camera-held-out macro results are raw BCE 0.59745, ROC AUC 0.88974, PR AUC
0.82313, interval F1 0.41915, and 26.11-frame boundary MAE. The production
checkpoint uses the median selected epoch and a decoder calibrated from the
concatenated out-of-fold predictions.

The matched candidate-inclusive arm keeps shuttle-selection loss at zero. It
does not improve interval F1 and reduces macro ROC/PR AUC, so raw shuttle
candidate tokens are not part of the maintained rally segmenter.
