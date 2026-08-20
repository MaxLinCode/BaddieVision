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

With no session option, `serve` resumes the compatible session with the
furthest event-reconciled progress. Use the printed `--session-id ID` command
to resume that exact session later, or pass `--new-session` to create a new
audit identity at the first currently unlabeled queue frame. The two options
are mutually exclusive. Startup and normal shutdown both print the durable
session ID, queue position, current frame, and executable resume command.

Active labels in the canonical event log count as complete regardless of which
annotator or session created them. Refresh and crash recovery therefore skip a
label whose event reached disk before its cursor update. Previous-target
navigation remains available for intentional corrections during a live
session; the correction appends a superseding revision.

The annotator shows one exact native frame at a time. Use `Space` to confirm an
adaptive suggestion, `I` for `OCCLUDED_INFERABLE`, `N` for
`NO_IN_FRAME_TARGET`, `M` for `MISSING_PROPOSAL`, `U` for `UNSURE`, Backspace
to undo, the arrow keys to move between annotation targets,
`,` / `.` (`<` / `>`) to inspect native frames before and after the fixed center
frame, `/` to return to the center, and digits 1-9 as shortcuts. Motion-preview frames are shown without
candidate marks; labels and candidate controls always apply to the center frame.
Hold `O` to temporarily show the clean
center image and release it to restore the numbered candidate overlay; `O` is
reserved for this image-only view and checked against task label hotkeys at startup.
Candidates use small, exact-position, color-coded markers and
leader-connected labels. Labels are placed inside the image with collision
avoidance so they do not cover a candidate marker; the highest-confidence
candidate has a slightly larger marker. Compact labels show only the candidate
number. Enable **Verbose candidate labels** to also show available TrackNet peak
activation and component area values. The legend and buttons use the same
number and color and show the durable candidate ID. All
center-frame candidates remain clickable even when a frame contains more than
nine. Adaptive recommendations appear directly over the center image with a
high-contrast **PRESS SPACE TO ACCEPT** banner; candidate recommendations also
receive a numbered gold ring at their exact location. Cross-threshold proposals
are grouped from high to low. A lower-threshold component joins at most one
group when that group's fixed representative peak lies inside its bounding box;
weighted-centroid distance and stable candidate IDs break ties. This prevents
transitive bounding-box bridges and duplicate-threshold group membership. The
grouping version and every raw member ID are retained in event metadata. The UI
can show grouped or raw candidates and apply a minimum-threshold filter. The raw
candidate artifact is never modified.

`no_shuttle` remains readable in historical logs but is not writable. Only
`NO_IN_FRAME_TARGET` supplies the selector's supervised `NULL_SELECTION` target.
`MISSING_PROPOSAL`, `OCCLUDED_INFERABLE`, `UNSURE`, and legacy `no_shuttle` are
masked from selector loss. Occluded coordinates are not repaired upstream.
Observed proposal recall is `selected / (selected + missing_proposal)`;
inferable occlusions and frames with no in-image target are reported separately.

New schema-v2 selected events snapshot finite normalized image coordinates for
the candidate's peak, weighted centroid, and bounding-box center. Peak position
is canonical. The server resolves these values from the fingerprint-verified
candidate artifact; the client never supplies coordinates. Non-selected labels
carry no candidate position. Historical logs remain append-only: migration or
export enriches an old selected label only when the exact candidate artifact is
available and otherwise marks its coordinates unavailable.

## Threshold pilot and migration

Production candidate extraction uses the non-overwriting threshold ladder
`0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50`. Generate it to new candidate paths;
never replace the v1 artifacts. Build and freeze the audit report with:

```bash
python3 -m src.annotation_platform --config config/pilot-sources.local.json \
  pilot-report --labels .annotation-pilot/exports/audit.jsonl \
  --output .annotation-pilot/reports/thresholds.json
python3 -m src.annotation_platform --config config/pilot-sources.local.json \
  freeze-pilot --report .annotation-pilot/reports/thresholds.json \
  --output-dir outputs/frozen --target-recall 0.99
```

The report includes the observed denominator, Wilson 95% interval, raw/grouped
candidate-count percentiles and maximum, and recall at
`K={1,2,3,5,8,12,all}`. Freezing chooses the highest cutoff reaching the 99%
point estimate. If none does, it chooses maximum recall and then fewer
candidates. It freezes the smallest K with the same audit hits as `all`.
Filtered artifacts retain stable raw IDs and record their parent SHA-256.

Create a pilot runtime without modifying the v1 runtime:

```bash
python3 -m src.annotation_platform --config config/pilot-sources.local.json \
  --runtime .annotation-pilot migrate-v1 \
  --v1-runtime .annotation-two-folders \
  --v1-config config/annotation-sources.local.json
```

Migration validates source/video/model hashes, geometry, extraction semantics,
frame ranges, and exact shared-threshold records before writing. It migrates the
408 unambiguous selected labels and leaves 102 labels for review. The original
600-frame adaptive and 300-frame audit orders are preserved exactly. Use
`finalize-runtime` after freezing; selected groups map to retained members or
become lineage-preserving `missing_proposal` labels when no member survives.

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
