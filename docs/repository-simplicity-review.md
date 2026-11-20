# Repository simplicity review

Reviewed 2026-11-14 against these goals: understandable layout, fast experiments,
replaceable components, one extraction workflow across local/Colab/Kaggle, and
preserved progress after interruption. This is an analysis and proposed scope;
implementation and historical-code moves were subsequently authorized. See the
implementation status below; the original findings remain a historical snapshot.

## Implementation status

Implemented shared extraction functions/CLI and one maintained notebook, stage
receipts and targeted rebuilds, scoped identities, optional preview/export,
incremental durable pose work, configurable training and epoch recovery, lazy
imports and public shared helpers, explicit annotation payload contracts, and
archived fixed-input prototypes. Redundant compatibility entrypoints and
platform-specific notebooks were subsequently removed; use the canonical
modules and single maintained notebook. See
[the workflow guide](workflow.md), [training guide](rally-training.md), and
[research index](research.md) for the supported paths.

Tracking recovery remains stage-level: TrackNet and ByteTrack do not yet expose
a tested deterministic state restoration contract. No shot-feature schema
changes or retraining are part of this implementation. Verification for implementation is
reported in the change's completion message; the original review's verification
section below describes the earlier read-only review.

Implementation verification: the full local suite passed with **403 passed, one
skipped**. CI-selected Ruff checks passed for all 61 changed/new Python files;
tracked-file privacy and whitespace checks passed. CLI help paths were exercised
and all maintained notebook code cells compiled. Recovery checks include abrupt
publication exits and resumed CPU training matching uninterrupted training. No
private-video inference, hosted notebook session, benchmark, or retraining was
performed for this implementation.

After removing the redundant wrappers and platform notebooks, the full suite
passed with **402 passed, one skipped**; changed-file lint and reference checks
also passed.

## Assessment

The biggest problem is accumulated workflow debt: old scripts, research studies,
notebooks, and the maintained pipeline all remain plausible entrypoints. There
is some premature generalization, but the more damaging issue is that existing
pieces do not compose reliably. The repo is simultaneously overbuilt around
some interfaces and underbuilt around routine execution and recovery.

I would simplify by consolidating execution and narrowing responsibilities,
not by replacing everything with a new framework.

Measured scope, excluding the TrackNet submodule, generated data, and notebooks:

| Area | Python files | Lines, including comments/blanks | Interpretation |
| --- | ---: | ---: | --- |
| `src/temporal_selector` | 24 | 12,160 | Maintained rally code, selector code, and studies mixed together |
| `src/annotation_platform` | 10 | 4,810 | Substantial tooling, with some task-general abstraction leakage |
| `InPlay/heuristic` | 17 | 3,816 | Contains maintained extraction despite the historical package name |
| Top-level `src` modules | 10 | 2,781 | Mixed extraction, calibration, visualization, training, release tooling |
| `src/workflow` | 8 | 1,861 | Useful orchestration with duplicated stage execution policies |
| `src/single_video` | 5 | 1,570 | Useful helpers; notebook orchestration remains outside them |
| Top-level `InPlay` modules | 12 | 1,271 | Includes older fixed-input training/inference prototypes |
| **Total** | **86** | **28,269** | First-party Python in `src` and `InPlay` |

There are also 32 test modules totaling 8,001 lines. Ten experiment/diagnostic
runner modules account for 6,605 lines, about 23% of the first-party Python
above. That is research surface to organize, not a justified deletion estimate:
it includes active training. The three single-video notebooks have approximately
60–73% pairwise ordered-line similarity in their code cells. The two 107-line
pose extraction files are byte-identical.

There is no defensible automatic percentage of “slop.” The concrete waste is
duplicated execution, overly broad invalidation, misleading entrypoints, and
historical study details leaking into reusable code. A large file or a
correctness check is not sufficient evidence of waste or AI-generated code.

## Findings, ordered by impact

### 1. Expensive caches depend on unrelated downstream code

`src/workflow/rally_pipeline.py:56` hashes a list covering extraction, datasets,
inference, and release code. That complete code identity and the complete runtime
package identity enter person extraction inputs at line 312 and player inputs
at line 346. Changing inference/decoder code can invalidate detector and pose
artifacts. `_reusable` at line 79 then rejects stale artifacts, with global
`--force` as recovery; that forces all stages to regenerate.

**Change:** give each stage only its actual inputs, model/preprocessing identity,
and relevant implementation identity. Add stage-specific rebuilding. Preserve
hash checks; reduce their dependency scope. Rendering should depend on saved
predictions so a styling change does not require model inference.

### 2. Manual stages and the end-to-end runner disagree about completion

`src/workflow/__main__.py:280` runs person extraction and player interpretation
directly. It writes the same canonical outputs used by the rally pipeline but
does not publish that pipeline's receipts. The pipeline treats existing outputs
without receipts as incomplete, so doing the advertised manual stages first
does not make the end-to-end runner ready to reuse them.

**Change:** share ordinary `run_person_stage` and `run_player_stage` functions
between both entrypoints, including validation and completion receipts. Keep
individual stages available for debugging; normal use should call one runner.

### 3. Notebook duplication is already causing behavioral drift

The three `Single_Video_Feature_Extraction_*` notebooks independently perform
TrackNet, person extraction, interpretation/pose, previews, metadata, and ZIP
creation. Colab and Kaggle contain an identical main execution cell; the local
version divides essentially the same workflow across cells. Batch sizes,
worker counts, setup paths, and packaging details differ.

Colab's `COURT_CALIBRATION` comment at JSON line 72 still promises a screen-space
fallback. Current `interpret_players` at
`InPlay/heuristic/player_interpretation.py:757` instead requires canonical
`court/calibration.json`, a source identity, and matching lineage. The notebook
interface has drifted from the implementation.

Reruns also unconditionally execute tracking. TrackNet writes the CSV after
inference (`src/TrackNetV3/predictArgs.py:573`) and then writes candidates. The
candidate writer rejects an existing file (`src/single_video/shuttle.py:236`).
Thus a rerun can spend the inference time before discovering an output conflict.

**Change:** one `run_extraction(...)` function and CLI, called by one thin
notebook. Platform-specific installation, storage mounting, and path selection
belong in a small bootstrap section. Extraction logic must have one owner.
Keep preview and ZIP export optional and separate from completing observations.

### 4. Current caching does not preserve progress within interrupted stages

The maintained rally pipeline safely stages and validates outputs, then writes
receipts. This protects published artifacts and skips completed stages, but
temporary extraction directories are removed on ordinary failure. It does not
provide durable in-stage recovery.

Pose extraction already has useful cache keys: frame, track, bbox, model, and
preprocessing (`player_interpretation.py:495`). Newly computed poses accumulate
in `new_records` at line 571 and are written only after the extraction loop at
line 585. Existing caches are rewritten with `open("w")`; an interruption during
that rewrite can truncate them. TrackNet likewise accumulates predictions and
candidate records until completion.

The maintained rally training loop keeps its best state and history in memory
(`src/temporal_selector/rally_experiment.py:271`). Model exports after training
are not optimizer/RNG checkpoints from which interrupted training can resume.

**Change:** persist validated pose work units incrementally, retain persistent
work directories, and atomically publish final outputs. Add epoch recovery
checkpoints containing model, optimizer, epoch, RNG/sampler state, configuration,
data/split identity, metrics, and best-state selection. Keep final inference
exports separate from recovery checkpoints.

TrackNet resume must retain/recompute temporal overlap and consistent background
preprocessing. ByteTrack needs tracker-state restoration or deterministic
replay; naive frame seeking changes track identities. Do not promise arbitrary
frame resume without establishing those contracts. Stage skipping is the first
increment; durable pose progress is the next; stateful tracker recovery is a
separate increment.

### 5. The frozen benchmark recipe blocks ordinary experiments

`RallyTrainingRecipe` at `src/temporal_selector/rally.py:445` rejects every value
combination except the fixed 25-epoch MPS recipe. The CLI at
`rally_experiment.py:1129` offers CPU/CUDA, but constructs that recipe using the
requested device. Both choices fail recipe validation, independently of device
availability. Short runs and changed batch size/LR/seed also require source edits.

**Change:** retain the exact historical benchmark as a named preset. Make normal
training configuration adjustable and record the resolved configuration in each
run. Changing configuration must actually reach the trainer: AdamW's learning
rate and gradient clipping are also hardcoded in the training loop.

### 6. Runtime code depends on studies and private implementation details

The maintained independent rally model sits under `temporal_selector`, beside
joint selector models and historical diagnostics. `rally_dataset.py:18` imports
private helpers from the selector dataset. `rally_experiment.py:24` imports
private training helpers from another experiment. `partial_oracle_experiment.py:38`
contains a fixed annotation hash and frame boundaries; `oracle_track_experiment.py:29`
imports that study's private functions.

`src/temporal_selector/__init__.py` eagerly imports runtime, crossfit, model,
dataset, and pseudo-label APIs. Workflow even duplicates fold construction
explicitly to avoid importing torch (`src/workflow/registry.py:38`).

**Change:** extract only genuinely shared helpers into lightweight modules with
public names. Move named study orchestration under `experiments/` with its
parameters, documentation, and tests. Keep runtime imports small. Historical
studies remain reproducible without defining today's supported public API.

### 7. Some entrypoints are prototypes presented as maintained workflows

`src/extract_features.py` loads models and processes files at import time, uses
fixed directories, and has its entire shuttle execution block commented out.
`InPlay/extract_features.py` is another legacy extraction entrypoint.
`src/pose_extraction.py` and `InPlay/pose_extraction.py` are identical.

Older `InPlay/train_model.py`, `train_lstm_model.py`, and `predict.py` load fixed
data or execute work at import. The older LSTM trainer also stores an uncloned
`model.state_dict()` as its best state at line 124, allowing later updates to
mutate it. These are historical prototypes, not good interchangeable components.

**Change:** mark/archive prototypes explicitly and route supported legacy
commands through thin wrappers where useful. Consolidate the duplicated pose
implementation. Do not delete research solely because it has few callers.

### 8. Annotation generalization is incomplete

`TaskPlugin` at `src/annotation_platform/core.py:110` describes a generic task,
but `events.py:338` handles shuttle-specific label kinds and discovers an
undeclared `resolve_candidate_position` hook. `dash_app.py:649` calls another
undeclared method. A replacement plugin therefore needs hidden knowledge of
surrounding code.

**Change:** when this area is next needed, either make it explicitly a shuttle
annotator or make payload construction/validation an explicit task responsibility.
Preserve event history, corrections, and stale-browser-action checks. Do not
expand the plugin system speculatively.

## Proposed execution and seam design

Extend the existing workflow CLI rather than introducing a second runner. For
example, these are proposed commands, not commands available today:

```bash
python -m src.workflow extract --source MATCH --profile observations
python -m src.workflow extract --source MATCH --profile observations --rebuild pose
```

`observations` would select the relevant extraction branches; rally prediction
would continue to use only person/court/pose observations and would not require
TrackNet. Shot feature generation remains a separate consumer. Small explicit
profiles and function calls are enough; no general plugin loader, dependency
injection container, workflow service, or custom scheduling framework is needed.

Seams should follow actual replaceable responsibilities:

| Boundary | Stable output | Example change contained behind it |
| --- | --- | --- |
| Person extractor | Frame-indexed boxes, confidence, track IDs, provenance | Replace detector/tracker adapter |
| Shuttle extractor | Frame-indexed candidates/tracks and provenance | Replace TrackNet adapter |
| Player assignment | Selected player observations and assignment evidence | Change assignment strategy |
| Pose extractor | Pose records keyed by observation/model/preprocessing | Replace pose backend |
| Task feature builder | Versioned ordered feature schema | Add a new model input view |
| Predictor | Frame probabilities or task predictions | Replace trained model |
| Renderer/exporter | Preview, CSV, portable package | Change presentation without extraction |

Start with ordinary typed records/dataclasses and functions. A backend adapter
owns model loading and conversion into the existing artifact contract. The
runner owns paths, stage dependencies, progress, and publication. Notebook/CLI
code only resolves configuration and invokes the runner. Backend identity must
participate in cache keys; current callable injection alone does not identify
which implementation produced an artifact.

Calibration changes should rebuild dependent assignments/features while reusing
raw person detections and any pose records whose observation keys still match.
Switching a rally decoder should reuse observations and raw predictions.
Switching a pose backend should invalidate pose and its consumers, not shuttle
or raw person extraction.

Store committed work under the artifact root, alongside existing canonical
artifacts. A small run record should show completed, pending, failed, and stale
stages with the reason and last committed unit. A completed manifest is written
only after coverage and lineage validation. Recovering an uncommitted attempt
must not delete the last valid output or human annotations. Explicitly imported
artifacts need a validated import path into the same completion policy.

On cloud notebooks, configure durable storage or periodically sync committed
units; a final ZIP in a session-local directory cannot protect against runtime
loss. Do not rely on notebook kernel variables as the progress database.

## Layout direction, after execution is consolidated

An eventual layout could expose `src/observations/`, `src/rally/`, `src/shot/`,
`src/workflow/`, and the existing annotation package; keep TrackNet vendored and
adapt it at the boundary. Place named studies in `experiments/`, clearly obsolete
prototypes in `archive/`, and the single maintained extraction notebook in
`notebooks/`. Avoid moving every file first: fix behavior, then move bounded
groups with compatibility wrappers where needed.

One short workflow guide should identify maintained commands, artifact paths,
and recovery behavior. Preserve old study documents in a clearly labeled
research index, rather than leaving historical instructions indistinguishable
from the current path. Dependency profiles may remain different for inference,
research, and notebook environments; that does not justify duplicating pipelines.

## Recommended scope for confirmation

1. **First implementation:** shared extraction stages, one callable/CLI, one
   notebook, stage-specific cache keys/rebuilds, optional preview/export, durable
   incremental pose saves, and a visible run status. Reuse the current canonical
   source/calibration conventions and receipt checks.
2. **Experiment usability:** configurable training with a named benchmark preset,
   epoch recovery checkpoints, and saved resolved configs/metrics.
3. **Separate cleanup approval:** move historical studies/prototypes, consolidate
   imports and pose duplication, simplify annotation seams, and update the layout
   and documentation index. These can be separate small changes.
4. **Stateful resume increment:** define and test TrackNet/ByteTrack restoration
   semantics before claiming full in-stage tracking recovery.

Acceptance checks should exercise behavior, not mirror implementation: a second
identical run invokes no expensive extractor; an inference-only edit reuses
extraction; a pose backend change reruns only pose and dependent work; interruption
reuses committed pose units; a preview failure leaves observations usable;
manual and automatic stages interoperate; a short CPU training configuration is
accepted; recovery keeps the same data/split identity. Stateful resumed results
must match uninterrupted results under the chosen deterministic contract.

Keep source-local zero-based frame indexing, one canonical calibration with
`image_size`, rejection of unresolved/ambiguous sources, annotation provenance,
split/leakage checks, and model/schema compatibility. Preserve shot features in
their existing order: 66 pose values, seven shuttle values, and three court-anchor
values. None of this review authorizes silent schema changes or retraining.

## Verification and limits

Three delegated read-only reviews used GPT-6.1 Sol at low reasoning effort, with
root review of the findings. Counts came from source inspection and Python AST
analysis. Pose-file equality was checked byte-for-byte. The recipe rejection was
reproduced by extracting and executing only its dataclass, without importing ML
dependencies: MPS accepted, CPU and CUDA rejected.

No training, private-video inference, hosted notebook session, or full regression
suite was run. Performance impact is inferred from actual stage dependencies and
write locations, not measured wall-clock benchmarks. No executable source files
were changed; this report is the only addition.
