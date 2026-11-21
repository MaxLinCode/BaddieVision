# Source workflow

For end-to-end single-model video inference, use `python -m src.workflow rally predict`
and follow [the inference guide](rally-inference.md). The pseudo-label commands
below are a separate three-teacher annotation-review workflow.

The workflow CLI keeps source paths in one local catalog and generates the
annotation and selector-experiment configurations from it. It does not weaken
the annotation platform's video, candidate, queue, or label fingerprint checks.

## Register and inspect a source

One source ID represents one immutable video and static-camera segment. Register
a new segment with:

```bash
python3 -m src.workflow source add \
  --video outputs/camera_c_segment_2/camera_c_segment_2_input.mp4 \
  --source-id camera_c_segment_2

python3 -m src.workflow source status
```

The default catalog is the Git-ignored `config/sources.local.json`. It resolves
only the video and artifact root, with relative paths when possible. Portable
identity is stored at `ARTIFACT_ROOT/source.json`; calibration has one authority
at `ARTIFACT_ROOT/court/calibration.json`. Copy `config/sources.example.json` to
seed a catalog manually.

Every source is decoded as its own frame sequence numbered from zero through
`frame_count - 1`. Parent-video trim times and frame estimates are provenance
only and never become annotation or artifact frame indices.

Import an existing calibration explicitly:

```bash
python3 -m src.workflow stage import-calibration \
  --source SOURCE --from features/court/legacy.json
```

The import validates `image_size` and source identity, then writes the canonical
artifact. Editing it invalidates dependent player assignments by fingerprint.

During the validation cycle, copy existing observations without moving their
legacy files, then regenerate calibration-dependent player outputs:

```bash
python3 -m src.workflow stage import-artifact --source SOURCE \
  --kind shuttle-candidates-frozen --from outputs/frozen/SOURCE.jsonl
python3 -m src.workflow stage import-artifact --source SOURCE \
  --kind person-tracks --from outputs/SOURCE/person_tracks.jsonl
python3 -m src.workflow stage import-artifact --source SOURCE \
  --kind player-poses --from outputs/SOURCE/pose_cache.jsonl
python3 -m src.workflow stage players --source SOURCE
```

Canonical observations are grouped beneath `ARTIFACT_ROOT/shuttle/`,
`ARTIFACT_ROOT/players/`, and `ARTIFACT_ROOT/court/`. Imports are atomic and do
not modify their source files.

Status distinguishes `annotation-ready` from `experiment-ready`. Annotation
needs the video and either pilot or frozen candidates. Experiment preparation
also requires frozen candidates, player assignments, a pose cache, and a valid
calibration JSON containing `image_size`.

## Pseudo-label rallies for review

Rally collection has its own readiness contract: immutable video identity,
canonical court calibration, person tracks, player assignments, and pose cache.
It never requires or reads shuttle candidates or shuttle annotations. With a
fingerprinted three-checkpoint annotation-teacher manifest:

```bash
python3 -m src.workflow rally pseudo-label \
  --source SOURCE --teacher outputs/rally-teacher/manifest.json --device mps

python3 -m src.workflow rally review \
  --source SOURCE --runtime .annotation-final --annotator NAME
```

Inference writes source-local `rallies/predictions.jsonl` and
`rallies/proposals.json`. Every model keeps its own validated decoder and
boundary-snap settings; a frame is proposed as in play only when at least two
of the three models agree. The labeler colors 2/3 and 3/3 proposals separately.
Edits autosave to a fingerprinted source-local draft. `Q` leaves canonical
annotations unchanged; press `F`, then `Y`, to finalize the full source.
Finalization replaces only that source in the canonical rally CSV/manifest and
writes a receipt tied to both proposal and teacher fingerprints. A frame-zero
proposal is only a review prompt: use `P` explicitly when play truly began
before the source.

## Start annotation

For one new source, no config or runtime path is required:

```bash
python3 -m src.workflow annotate --annotator max --source camera_c_segment_2
```

It writes a generated annotation catalog and uses the source-scoped runtime
`.annotation-camera_c_segment_2`. This avoids modifying older immutable combined
queues. Frozen candidates are preferred; if they do not exist, pilot candidates
are used. Select them explicitly with `--stage pilot` or `--stage frozen`.

For an established multi-source runtime:

```bash
python3 -m src.workflow annotate --annotator max --runtime .annotation-final
```

Queue, host, port, and new-session options remain available. Existing queues
are validated by the annotation platform before the server starts.

## Prepare an experiment snapshot

After the enabled sources and `.annotation-final` runtime are complete:

```bash
python3 -m src.workflow experiment prepare

python3 -m src.temporal_selector.experiment \
  --config config/selector-experiment.generated.json \
  --output-dir outputs/joint-selector-run \
  --context-mode full_context
```

Preparation fails if any selected source or required runtime input is missing.
It captures the annotation SHA-256, includes the completed standard queues, and
generates a deterministic leave-one-source-out manifest. The generated config
and split manifest are local snapshots and ignored by Git.

## Shared extraction and recovery

Use one callable or CLI on local, Colab, and Kaggle:

```bash
python -m src.workflow extract --source SOURCE --profile observations \
  --pose-model models/pose_landmarker_full.task \
  --model-root . --device auto
python -m src.workflow extract --source SOURCE --profile observations \
  --model-root . --rebuild pose
```

For an unregistered video, replace `--source SOURCE` with `--video segment.mp4
--source-id SOURCE --output-dir /durable/artifacts/SOURCE`. Pass `--calibration
court.json` to import a calibration explicitly. The canonical copy is always
`ARTIFACT_ROOT/court/calibration.json`; `image_size` is required. A source is one
static camera segment with its own zero-based frames. Use a new source ID when
the camera moves. Trimming/resizing must happen before registration; see
`src.single_video.video.prepare_working_video` for that optional preparation.

`--profile rally` completes person/court/pose observations without TrackNet.
`observations` adds shuttle tracks, candidates, tracklets, and diagnostic
hypotheses using the TrackNet bundle. `--rebuild shuttle-links` recomputes the
derived tracklets/hypotheses while retaining tracking.
The maintained notebook is
[`Single_Video_Feature_Extraction.ipynb`](../notebooks/Single_Video_Feature_Extraction.ipynb).
Its bootstrap selects storage paths; extraction logic lives in
`src.workflow.extraction.run_extraction`. The old platform-specific notebooks have been removed.

Manual `stage person-tracks` and `stage players` use the same functions,
validation, backend/model identities, and receipts as extraction and rally
prediction. Use the same model, device, and crop settings when combining manual
and automatic stages. Explicit person-track imports receive a validated import
receipt and act as a supplied source layer until explicitly rebuilt. Imported
pose records are reused only when their observation/model/preprocessing keys
match. Before using person tracks, the players stage verifies their hash and
source identity against the person-track receipt. Import or regenerate older
tracks that lack a receipt.

Each stage checks its own inputs, relevant implementation, model bytes, runtime
packages, and output hashes. Stale or corrupt artifacts require an explicit
`--rebuild STAGE`; `--force` rebuilds all requested stages. Rebuilding person
tracks also rebuilds dependent player outputs; rebuilding pose leaves person
and shuttle extraction reusable. Calibration changes require `--rebuild pose`
(or `--force`) and reuse matching committed pose crops. Custom backends must
have inspectable implementation code or an explicit stable backend identity;
change that identity when backend behavior changes.

`run-status.json` records pending, running, completed, failed, and stale stages.
Inspect it directly or use `source status --source SOURCE --json`. Status refreshes
the retained pose-unit count and last committed frame/track even after an abrupt
exit. Counts include retained historical backend/source partitions. Pose crops
are committed incrementally under `players/.pose-work/`, including source,
frame, track, box, model/backend, and preprocessing identity. Restarting a failed
player stage reuses matching crops. Pose snapshots publish atomically. Final
outputs and their receipts use a recoverable publication journal; interrupted
publication restores the previous bundle on the next run when it stopped
before commit. Recovery reports the outcome and receipt path on stderr.
Human annotation files are never extraction outputs.

Optional `--preview` and `--export-zip /durable/result.zip` run after observations
complete. Preview failure leaves completed observations usable. Exports contain
canonical observations, calibration and source identity, excluding work caches.
Store the artifact root on durable storage in cloud runtimes or sync committed
units periodically. A session-local final ZIP cannot recover a lost runtime.

Rally prediction also accepts `--rebuild person|pose|predict|decode|render`.
`rallies/probabilities.jsonl` preserves the model output independently of the
decoder. Decoding produces `predictions.jsonl` and `rallies.csv`; rendering reads
those saved predictions. Decoder or overlay changes can reuse observations and
probabilities. Rendering failure preserves the prediction receipt.

Person tracking and TrackNet currently recover at **stage boundaries**. An
interrupted tracking stage restarts from frame zero. Arbitrary frame seeking is
not supported: BoT-SORT state/track identities and TrackNet temporal overlap
and background preprocessing must be restored or deterministically replayed
before a future implementation can claim equivalent in-stage tracking resume.
For epoch recovery and configurable rally experiments, see
[training and recovery](rally-training.md).

## YOLO26L person detector

Person extraction defaults to YOLO26L with fixed-camera BoT-SORT. CUDA uses
`models/yolo26l.engine` when that FP16 TensorRT engine is present. CPU, Apple
MPS, and CUDA without the engine use `yolo26l.pt`. Build the engine on the target
NVIDIA GPU and TensorRT installation because engines are hardware and version
specific.

```bash
mkdir -p models
python -c 'from ultralytics import YOLO; YOLO("yolo26l.pt").export(format="engine", half=True, device=0, imgsz=960)'
mv yolo26l.engine models/yolo26l.engine
python -m src.workflow extract --help
```

Tracking uses a 960 × 960 input to retain more detail for the far player.
It costs more GPU time and memory than the previous 640 × 640 input.

Install compatible CUDA, TensorRT, and an Ultralytics release supporting YOLO26
before export. Use `--device cuda` for extraction. The engine stores its FP16
precision. CUDA PyTorch inference also requests FP16. Use `--person-model`
with `extract` or `--model` with `stage person-tracks` to select a detector
explicitly. Rebuild existing generated observations with `--rebuild person`
or `stage person-tracks --force` to apply the new tracker and assignment rules.

### Player assignment diagnostics

Person extraction passes detections to BoT-SORT at a confidence floor of 0.10.
BoT-SORT uses a 0.10 low match threshold, 0.25 high and new-track thresholds, and
a 30-frame lost-track buffer. Camera-motion compensation and ReID are disabled.

After tracking, the player stage assigns one track per court half using the
projected box bottom center and a 1 m margin beyond the calibrated court. A current
incumbent keeps its slot and can return for up to 0.5 s after its last eligible
observation. For an open slot, tracks with at least 0.25 s of eligible
observations in the next 0.5 s are ranked by unpadded court occupancy, then
detection confidence. Missing observations remain empty.

Assignment JSONL includes the role decision and whether visible heels or toes
agree with the box's court half and padded bounds. The preview highlights the
projected pose foot in red on disagreements. These diagnostics use the original
MediaPipe result before temporal interpolation.

Use separate artifact roots for diagnostic runs. Existing tracking outputs can
be read or imported with their original provenance.
