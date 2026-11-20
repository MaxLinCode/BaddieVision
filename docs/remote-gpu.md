# Home GPU jobs

Run `scripts/gpu` from your feature worktree to submit that worktree's source
and local edits. The launcher sends a snapshot to WSL and runs the command
through Windows Task Scheduler. Native Linux SSH hosts use a detached tmux
session. Jobs survive a Mac disconnect. Windows sleep, shutdown, or WSL
shutdown can interrupt them.

Create a `.env` file at the root of the worktree. Git already ignores this file.
Replace the example values with your SSH alias and remote checkout path.

```dotenv
BV_GPU_HOST=gpu-host
BV_GPU_DIR='~/path/to/BaddieVision'
BV_GPU_TRANSPORT=native
```

Set `BV_GPU_TRANSPORT=wsl` for Windows OpenSSH with WSL. Keep device names,
addresses, and usernames in your local SSH configuration.

```bash
git submodule update --init --recursive
./scripts/gpu doctor
./scripts/gpu submit -- python -m src.workflow rally predict \
  --video videos/match.mp4 \
  --calibration features/match/court/calibration.json \
  --output-dir outputs/match --model rally-six-source-v1 --device cuda
./scripts/gpu status
./scripts/gpu logs JOB_ID
./scripts/gpu logs JOB_ID --follow
./scripts/gpu fetch JOB_ID --path outputs/match
```

Replace the example video and calibration with existing remote inputs. Calibration
must include `image_size` and belong to the source's static camera segment. A
registered source uses its canonical `<artifact_root>/court/calibration.json`;
`config/sources.local.json` contains only the remote video/artifact-root mapping.

Exported `BV_GPU_*` settings override `.env`. Pass `--host` or `--remote-dir`
before the subcommand to override either setting for a single command.
The existing remote `.venv` supplies Python and installed dependencies. The
launcher does not install or update dependencies; run `doctor` to check CUDA.
A command is an argument list, not a shell expression; explicitly use
`bash -lc '...'` when pipes or shell expansion are needed.

Each submission uploads tracked and nonignored untracked files from the
worktree containing the launcher, including local edits and initialized
submodule source. Initialize TrackNet in each worktree where you use it. Its
checked-out revision is included in the snapshot. Keep credentials, environments,
and model files in ignored locations and pass remote data paths to the command.
Check for private files and symlink targets in the worktree before submitting.
Job metadata records command arguments, so keep credentials out of the command line.

Jobs live under `<remote-dir>/.remote-gpu/jobs/JOB_ID/`:

- `workspace/`: fresh source snapshot and job outputs.
- `job.json`: submitted argument list and workspace path.
- `job.log`: stdout and stderr.
- `started`, `exit-code`: execution receipts.

Large existing input directories (`videos`, `clips`, `features`, `clip_features`,
`models`, the corresponding InPlay data directories, and TrackNet checkpoints)
are symlinked from the home checkout. Its `config/*.local*.json` files are also
reused. These inputs are shared: commands that write to them affect the original
remote data. Write run results under the job's `outputs/` directory. Local
source paths are never substituted for remote paths. No input data transfer is
automatic; transfer new videos/calibrations separately before submitting.

The original home checkout and its uncommitted source edits are preserved.
Separate submissions have separate source and output directories, but share GPU
resources; this launcher does not queue or serialize jobs. Avoid overlapping
large training runs. `status` reports interrupted sessions if tmux disappears
without an exit receipt. Training recovery remains the trainer's responsibility.

`fetch` downloads the requested output directory into
`outputs/remote-gpu/JOB_ID/`, or `--destination PATH`. It rejects symlinks and
special files in downloaded archives. Logs can be viewed even after completion;
Ctrl-C exits log following without stopping the remote job.

## SSH transport

The launcher defaults to native Linux transport. For WSL transport, configure
Windows OpenSSH to run noninteractive commands in cmd.exe. The launcher enters
WSL using `wsl.exe --cd ~ --exec python3` and encoded payloads.

WSL submissions create a `BaddieVision-JOB_ID` scheduled task under the existing
Windows account, without a password or elevated run level. The task keeps a
foreground `wsl.exe` process attached to the job, preventing WSL idle shutdown.
Tasks are retained for diagnostics; this is a job runner, not a recurring task.

If WSL stops responding, Windows SSH can still inspect its service. Restarting
WSL interrupts its running Linux processes. Replace `gpu-host` with your configured
SSH alias.

```bash
ssh gpu-host 'powershell.exe -NoProfile -Command Get-Service WslService'
ssh gpu-host 'powershell.exe -NoProfile -Command Restart-Service WslService -Force'
ssh gpu-host 'wsl.exe --exec uname -a'
```
