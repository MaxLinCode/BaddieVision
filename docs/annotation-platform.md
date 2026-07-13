# Local annotation platform

The annotation package is a reusable event/queue/session core with shuttle
candidate selection as its first task plugin. Runtime state defaults to the
Git-ignored `.annotation/` directory. Candidate artifacts may use
`shuttle_candidates` schema v1 or v2; every label and queue is bound to the
exact candidate-artifact and source-video fingerprints registered at startup.

Create a local JSON source catalog. Paths are resolved relative to the catalog:

```json
{
  "sources": [
    {
      "source_id": "match-a-static-camera",
      "video_path": "../videos/match-a.mp4",
      "candidates_path": "../outputs/match-a/shuttle_candidates.jsonl"
    }
  ]
}
```

`fps` and `frame_count` are optional and otherwise probed from the video. One
source ID should still represent one immutable video/camera segment.

Build the separately seeded immutable audit queue and adaptive queue, then run
the app:

```bash
python3 -m src.annotation_platform --config config/annotation-sources.local.json build-queues \
  --audit-seed 1729 --audit-count 100 --adaptive-count 200
python3 -m src.annotation_platform --config config/annotation-sources.local.json serve \
  --annotator alice --queue adaptive
```

Use `Space` to confirm an adaptive suggestion, `N` for `NO_SHUTTLE`, `M` for
`MISSING_PROPOSAL`, `U` for `UNSURE`, Backspace to undo,
the arrow keys to navigate, and digits 1-9 as shortcuts. All center-frame
candidates remain clickable even when a frame contains more than nine.
Playback includes one second before and after the exact rendered center frame,
with overlays updated throughout.

Other commands:

```bash
# Decode and render one center-frame smoke image for every registered source.
python3 -m src.annotation_platform --config config/annotation-sources.local.json smoke

# Export the current replayed labels without replacing the canonical event log.
python3 -m src.annotation_platform --config config/annotation-sources.local.json export \
  --output .annotation/exports/shuttle-labels.jsonl
```

Labels are immutable JSONL revisions. Corrections supersede the current
frame revision, while undo appends another revision and restores the preceding
label. If a process leaves a partial final JSON object, replay ignores that
tail but blocks new writes. Preserve it and resume in a new append-only segment
with `EventStore.recover_interrupted_tail()`; the method writes a fingerprinted
recovery audit record and never truncates the damaged segment.
