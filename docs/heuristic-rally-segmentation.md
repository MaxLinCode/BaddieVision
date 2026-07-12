# Heuristic rally segmentation

This pipeline is separate from the learned `InPlay` model and from the shot
classifier. It does not change the shot-classifier feature schema.

Run segmentation:

```bash
python -m InPlay.heuristic.segment \
  --tracks TRACK.csv --fps 30 --image-size WIDTH HEIGHT \
  --source-id CAMERA_SEGMENT --output rallies.csv \
  --debug-frames rally_frames.csv
```

Track CSVs require `Frame,X,Y,Visibility`; `PeakValue` is optional for legacy
TrackNet output. Frame records must be contiguous, unique, and ordered. All
thresholds are frame counts tuned for 30 FPS. Other frame rates continue
without rescaling and receive a warning plus `non_30_fps` output flag.

Optional inputs:

```bash
python -m InPlay.heuristic.players \
  --video INPUT.mp4 --output players.csv

python -m InPlay.heuristic.segment ... \
  --players players.csv --court-calibration calibration.json
```

Player activity can delay an end decision but cannot start a rally. The court
signal is deliberately a tolerant image-region check for an airborne shuttle,
not a floor-plane shuttle projection. Invalid optional calibration produces a
review flag and disables that signal.

Evaluation uses inclusive intervals and maximum-IoU one-to-one matching:

```bash
python -m InPlay.heuristic.evaluate \
  --predictions rallies.csv --labels labels.csv \
  --metrics metrics.json --matches matches.csv
```

Labels use `source_id,rally_id,start_frame,end_frame`.

Manual correction is a CSV round trip:

```bash
python -m InPlay.heuristic.validate_corrections --input reviewed.csv
python -m InPlay.heuristic.finalize \
  --input reviewed.csv --output final.csv --fps 30
```

Allowed manual decisions are blank, `accept`, and `reject`. Flags are
semicolon-separated. Finalization applies manual boundaries, excludes rejected
rows, and retains the automatic confidence, flags, failure reason, and manual
audit fields.

# Heuristic rally decoder v2

The default layered workflow is an offline, precision-first singles decoder in
`InPlay.heuristic.rally_v2`. It consumes immutable shuttle candidates and
tracklets, replayable shuttle hypotheses, player assignments, the pose cache,
source metadata, and one matching static-camera calibration. Every artifact is
fingerprinted and its FPS, frame count, image size, and upstream references are
validated before decoding.

Run it through the existing segmentation entry point:

```bash
python3 -m InPlay.heuristic.segment \
  --shuttle-candidates shuttle_candidates.jsonl \
  --shuttle-tracklets shuttle_tracklets.jsonl \
  --shuttle-hypotheses shuttle_hypotheses.jsonl \
  --player-assignments player_assignments.jsonl \
  --pose-cache pose_cache.jsonl --metadata video_metadata.json \
  --court-calibration court.json --output rallies.csv \
  --state-events rally_state_events.jsonl
```

`rallies.csv` retains the correction/finalization schema. Canonical boundaries
are not padded; use `clip_export.padded_bounds` when exporting clips. The JSONL
event stream contains the versioned profile, all input SHA-256 fingerprints,
and one deterministic evidence/transition record per source frame. Degraded
mode (`--degraded-mode`) can only lower results to `review`.

The older TrackNet-visibility state machine remains available through the
legacy CLI arguments and is explicitly imported by the learned InPlay decoder
from `InPlay.heuristic.legacy_visibility`.
