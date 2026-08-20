# Source workflow

The workflow CLI keeps source paths in one local catalog and generates the
annotation and selector-experiment configurations from it. It does not weaken
the annotation platform's video, candidate, queue, or label fingerprint checks.

## Register and inspect a source

One source ID represents one immutable video and static-camera segment. Register
a new segment with:

```bash
python3 -m src.workflow source add \
  --video outputs/ff-bothell-seg4/ff-bothell-seg4_input.mp4 \
  --source-id ff-bothell-seg4

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
python3 -m src.workflow annotate --annotator max --source ff-bothell-seg4
```

It writes a generated annotation catalog and uses the source-scoped runtime
`.annotation-ff-bothell-seg4`. This avoids modifying older immutable combined
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
