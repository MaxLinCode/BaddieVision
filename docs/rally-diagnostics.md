# Local rally diagnostics

The viewer worktree keeps both prototypes available while their diagnostics are
consolidated into `src.rally_review`. Use rally review for playback and run
comparison. The Dash explorer exposes additional shuttle evidence and accepts
explicit prediction and artifact paths via
`python3 -m src.rally_error_explorer --help`.

From the checkout root, run:

```sh
python3 -m src.rally_review --open
```

The viewer opens at http://127.0.0.1:8768. It needs Python 3.11 or later and `ffprobe` on your PATH. The command serves only on loopback. The video and experiment data stay in the checkout. Stop the server with Ctrl+C.

Choose an experiment, recording, and run. Choose a second run to compare them against the same recording. The player and timeline seek together. P1 near is blue and P2 far is pink on both the video and the timeline. Use **Boxes** and **Poses** under the timeline to toggle the saved tracking overlays. The review menu highlights false starts, boundary offsets, missed rallies, splits, merges, and low overlap using the saved rally labels. Click **Refresh experiments** to pick up new runs while the viewer is open.

## What the viewer discovers

- It scans `.local-data/experiments/`, `experiments/`, `outputs/`, and `.local-data/rally-inference/` for frame-level rally prediction JSONL. Recognized names are `predictions.jsonl`, `test-predictions.jsonl`, `val-predictions.jsonl`, and `scored-predictions.jsonl`. Each source needs a source ID and source-local frame indices. Rows need InPlay scores. Saved decoder states enable the failure filters.
- It resolves footage using the run's experiment config or `config/sources.local.json`. Use a canonical source catalog with a `source_id`, `video_path`, and `artifact_root`. Both paths should point within the checkout. For standalone single-source runs a sibling `video.mp4` also works.
- Strict rally labels come from the run's frozen labels, its experiment config, or the source's reviewed annotations. The viewer verifies label fingerprints against the source video and uses the manifest's reviewed coverage. If strict labels are unavailable, it uses labeled frames from the saved evaluation output when present.
- Optional player lanes come from the primary run's source artifacts. For comparison, choose runs for the same source-local video.

The viewer reads the saved artifacts directly each time their file versions change. No separate metrics export is needed. Use `--root` to point a worktree at the main checkout's data, or `--port` to choose a different loopback port.
