# Repository guidance

## Local worktrees

Create new local Git worktrees under this repository's `.worktrees/<name>/`
directory, on `codex/` branches, unless the user requests another location.
Keep large/private data in the main checkout's ignored `.local-data/` directory
and reference it explicitly from worktrees; do not duplicate or delete it when
creating or removing a worktree. Worktree removal is for disposable code only.

## Preserve local data

Repository scrubbing means removing private assets from Git tracking, not
deleting their local copies. Preserve videos, annotations, features, calibration,
checkpoints, and experiment outputs on disk. Use `.gitignore` and
`git rm --cached` when untracking data. Never delete or trash those assets as
part of a repository cleanup. `.local-data/` is an ignored recovery/data area;
do not remove or clean it, including with `git clean -x`.

Start each substantial feature in its own worktree and draft pull request.
Keep the main checkout for integration and local data. From that checkout:

```sh
git fetch origin main
git worktree add -b codex/<name> .worktrees/<name> origin/main
```

Run the installer below from a checkout containing it to guard local commits
to `main`. It shares the hook across linked worktrees and checks for an
existing hook before installing.

```sh
./scripts/install-main-guard
```

From the feature worktree, push the feature branch and open a draft PR against
`main`. For dependent changes, open the draft against its prerequisite branch,
then retarget to `main` after that PR lands. Run focused tests and keep the PR
scoped to one feature. GitHub requires a PR and the **Python 3.11 / CPU** check
for changes to `main`.

## Shot-classifier features

The per-frame shot-classifier schema is ordered as:

1. 66 MediaPipe pose coordinates
2. 7 shuttle position/motion features
3. normalized court anchor `[x, y, observed]`

Do not change this order without regenerating every file in `clip_features/` and
retraining the classifier. Existing 73-input checkpoints predate court anchors
and are incompatible.

Court anchors apply only to the shot classifier. `InPlay` is a separate rally
classifier, and TrackNetV3/InpaintNet are shuttle-tracking components.

## Court calibration

The maintained source workflow stores one canonical calibration at
`<artifact_root>/court/calibration.json`. `config/sources.local.json` resolves
only the local video and artifact root; it must not contain a competing
calibration path. Legacy files under `features/court/` are supported only as
explicit import sources. Shot-classifier clip mapping may continue to read the
legacy calibration registry until its separate migration is implemented.

One source ID represents one static camera segment and its own zero-based frame
sequence `0 .. frame_count - 1`. Parent-video trim bounds are provenance only
and never affect artifact or annotation frame indices. If the camera moves,
create a new source ID and calibration and map affected clips explicitly.

Calibration JSON files must include `image_size`. Feature extraction must fail
on unresolved or ambiguous sources instead of silently emitting missing court
features.

See `docs/plans/court-space-player-anchor.md` for the implemented design.

## Local accelerator

Apple MPS is available on the host but may appear unavailable from inside the
sandbox. Do not treat a sandboxed `torch.backends.mps.is_available()` result as
authoritative for experiment runs. Run MPS training outside the sandbox when
host acceleration is required.
