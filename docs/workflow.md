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

The default catalog is the Git-ignored `config/sources.local.json`. Paths are
stored relative to the catalog when possible. Unless overridden, artifacts are
expected under `outputs/SOURCE_ID/` and the court calibration at
`features/court/SOURCE_ID.json`. Copy `config/sources.example.json` to seed a
catalog manually.

Status distinguishes `annotation-ready` from `experiment-ready`. Annotation
needs the video and either pilot or frozen candidates. Experiment preparation
also requires frozen candidates, player assignments, a pose cache, and a valid
calibration JSON containing `image_size`.

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
