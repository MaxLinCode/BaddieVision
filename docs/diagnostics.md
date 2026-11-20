# Rally diagnostics

Run `python -m src.diagnostics` from the repository root with the existing
project environment. Pass individual input files and a new output directory
under `outputs/diagnostics/`. The command prints the path to `report.md`.
Give that report to Codex and use its artifact links for visual review.

## Compare predicted and labeled intervals

```bash
python -m src.diagnostics intervals \
  --predictions outputs/my-run/decoded-rallies.csv \
  --labels path/to/labels.csv \
  --output-dir outputs/diagnostics/my-interval-review
```

Inputs use `source_id,rally_id,start_frame,end_frame` and inclusive,
source-local frame indices. Use labels from the same evaluated split and
coverage as the predictions. The report gives interval metrics and up to five
examples by default, with unmatched intervals first and then the lowest-IoU
matches. Each row has a corresponding timeline in `intervals.png`.

## Review saved InPlay failures on video

```bash
python -m src.diagnostics rally-video \
  --config path/to/selector-config.json \
  --predictions outputs/my-run/predictions.jsonl \
  --table outputs/my-run/rally-diagnostics.csv \
  --source camera-a-segment-1 --rally EXACT_RALLY_ID \
  --output-dir outputs/diagnostics/my-video-review
```

Use the saved within-camera decoder artifacts from one run and the selector
config that points to its source-local video. Repeat `--rally` up to five
times. The linked MP4s overlay the selection state, InPlay probability, and
labeled and decoded intervals. If you request extra context using
`--context-frames`, those frames must also have saved predictions.

## Compare person tracks and player assignments

```bash
python -m src.diagnostics player-assignment \
  --person-tracks ARTIFACT_ROOT/players/person_tracks.jsonl \
  --assignments ARTIFACT_ROOT/players/assignments.jsonl \
  --court-calibration ARTIFACT_ROOT/court/calibration.json \
  --output-dir outputs/diagnostics/player-assignment-review
```

The report counts missing near and far slots with an unused tracked box on the
expected court half, missing slots with tracks elsewhere or already assigned,
and missing slots on frames with zero tracks. It also summarizes the saved
assignment reasons when a same-half track was available. All source frames are
included. A same-half box can belong to another person. Boxes enter the saved
person tracks only after ByteTrack assigns an ID, so the report cannot isolate
YOLO detection misses from ByteTrack losses.

## Compare ByteTrack and BoT-SORT

```bash
python -m src.diagnostics trackers \
  --video path/to/source.mp4 \
  --model path/to/yolov8n.pt \
  --court-calibration ARTIFACT_ROOT/court/calibration.json \
  --device mps \
  --output-dir outputs/diagnostics/tracker-comparison
```

Both trackers receive the person detections from one model pass. ByteTrack
uses its bundled settings. BoT-SORT uses native detector features for
re-identification and disables camera-motion compensation for static footage.
The same court-aware assignment logic runs on both outputs. The report counts
slot coverage, absent selected tracks, and track ID changes between successive
assignments, including across gaps. Coverage counts populated slots and does
not measure correctness against player labels. Use `--max-frames` for a quick
check on the beginning of the source.

## Codex CLI prompts

From the repository root, pass the saved Markdown prompt as the initial
message and append the diagnostic arguments or consolidation request:

```bash
codex "$(cat ~/.codex/prompts/baddie-diagnose.md)

<subcommand and explicit arguments>"

codex "$(cat ~/.codex/prompts/baddie-consolidate.md)

<requested diagnostic change>"
```

OpenAI deprecated the custom slash-prompt loader. The commands above use
ordinary prompt text and require no skill.
