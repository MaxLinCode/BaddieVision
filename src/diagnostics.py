"""Diagnose saved rally predictions and player observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from contextlib import closing
from itertools import islice, zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory


def _cell(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _intervals(args: argparse.Namespace, output: Path) -> list[str]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from InPlay.heuristic.evaluate import evaluate, read_intervals

    predictions = read_intervals(args.predictions)
    labels = read_intervals(args.labels)
    predicted_by_id = {(item.source_id, item.rally_id): item for item in predictions}
    labeled_by_id = {(item.source_id, item.rally_id): item for item in labels}
    if len(predicted_by_id) != len(predictions) or len(labeled_by_id) != len(labels):
        raise ValueError("rally IDs must be unique within each source and input file")
    metrics, details = evaluate(predictions, labels, args.iou_threshold)
    examples = sorted(
        details,
        key=lambda row: (row["match_status"] == "matched", row["iou"]),
    )[:args.limit]
    report = [
        f"Interval matching at IoU ≥ {args.iou_threshold:g}.",
        "",
        f"- Precision {metrics['precision']:.3f}, recall {metrics['recall']:.3f}, "
        f"F1 {metrics['f1']:.3f}",
        f"- {metrics['matched_count']} matched, "
        f"{metrics['unmatched_prediction_count']} extra predictions, "
        f"{metrics['unmatched_label_count']} missed labels",
        f"- {metrics['false_split_count']} split labels, "
        f"{metrics['false_merge_count']} merged predictions",
        "",
    ]
    if not examples:
        return report + ["Both inputs contain zero accepted intervals."]

    report.extend([
        "[Interval timelines](intervals.png)",
        "",
        "| Source | Prediction | Label | Status | IoU | Start/end error (frames) |",
        "| --- | --- | --- | --- | ---: | --- |",
    ])
    figure = Figure(figsize=(9, 1.7 * len(examples) + 0.6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(examples), squeeze=False)
    for index, row in enumerate(examples):
        source = row["source_id"]
        pred_id = row["prediction_rally_id"]
        label_id = row["label_rally_id"]
        status = str(row["match_status"]).replace("_", " ")
        errors = f"{row['start_error']} / {row['end_error']}" if status == "matched" else "—"
        report.append(
            f"| {_cell(source)} | {_cell(pred_id or '—')} | {_cell(label_id or '—')} | "
            f"{status} | {row['iou']:.3f} | {errors} |"
        )
        axis = axes[index, 0]
        for level, item, color in (
            (1, labeled_by_id.get((source, label_id)), "#357846"),
            (0, predicted_by_id.get((source, pred_id)), "#356ab0"),
        ):
            if item is not None:
                axis.barh(level, item.end - item.start + 1, left=item.start, color=color)
        axis.set_yticks([0, 1], ["Prediction", "Label"])
        axis.set_title(f"{source} | {pred_id or '—'} / {label_id or '—'} | {status}", fontsize=10)
        axis.set_xlabel("Source-local frame")
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    figure.savefig(output / "intervals.png", dpi=120)
    return report


def _rally_video(args: argparse.Namespace, output: Path) -> list[str]:
    from .temporal_selector.rally_video_diagnostics import (
        load_rally_diagnostics,
        render_rally_videos,
    )

    rows = load_rally_diagnostics(args.table, (), args.source, args.rally)
    clips = render_rally_videos(
        config_path=args.config,
        predictions_path=args.predictions,
        diagnostics_path=args.table,
        output_dir=output,
        source_id=args.source,
        rally_ids=args.rally,
        context_frames=args.context_frames,
        counterfactual_predictions_path=args.counterfactual_predictions,
    )
    return [
        f"Source {_cell(args.source)}. Frame indices are inclusive and source-local.",
        "",
        "| Rally | Frames | Status | Best IoU | Longest low-confidence gap (frames) | Video |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ] + [
        f"| {_cell(row['rally_id'])} | {row['start_frame']}–{row['end_frame']} | "
        f"{_cell(row['status'])} | {float(row['best_iou']):.3f} | "
        f"{row['longest_internal_low_gap_frames']} | [MP4](<{clip.name}>) |"
        for row, clip in zip(rows, clips, strict=True)
    ]


def _player_assignment(args: argparse.Namespace, _output: Path) -> list[str]:
    from InPlay.heuristic.person_tracks import load_person_tracks, raw_artifact_fingerprint
    from InPlay.heuristic.player_interpretation import _project_observation, load_calibration

    raw, observations = load_person_tracks(args.person_tracks)
    frame_count = int(raw["frame_count"])
    if frame_count <= 0:
        raise ValueError("person tracks must describe a nonempty source")
    frame_size = tuple(map(int, raw["frame_size"]))
    court = load_calibration(args.court_calibration, frame_size)
    roles = {"P1": "near", "P2": "far"}
    tracks_by_frame: list[set[int]] = [set() for _ in range(frame_count)]
    same_half = {slot: [set() for _ in range(frame_count)] for slot in roles}
    for observation in observations:
        frame = int(observation["frame"])
        if not 0 <= frame < frame_count:
            raise ValueError("person observation is outside the source frame range")
        track_id = int(observation["track_id"])
        if track_id in tracks_by_frame[frame]:
            raise ValueError("duplicate person track on a source frame")
        tracks_by_frame[frame].add(track_id)
        projected = _project_observation(observation, court)
        if projected["in_court"]:
            if projected["court_y"] <= 0:
                same_half["P1"][frame].add(track_id)
            if projected["court_y"] >= 0:
                same_half["P2"][frame].add(track_id)

    counts = {slot: Counter() for slot in roles}
    reasons = {slot: Counter() for slot in roles}
    calibration_hash = hashlib.sha256(args.court_calibration.read_bytes()).hexdigest()
    with args.assignments.open(encoding="utf-8") as handle:
        metadata = json.loads(next(handle, "null"))
        if not metadata or metadata.get("schema") != "player_assignments":
            raise ValueError("invalid player-assignment metadata")
        if (metadata.get("raw_artifact_fingerprint") != raw_artifact_fingerprint(args.person_tracks)
                or metadata.get("calibration_sha256") != calibration_hash
                or metadata.get("frame_count") != frame_count
                or metadata.get("frame_size") != raw["frame_size"]):
            raise ValueError("assignments do not match the supplied person tracks and court calibration")
        records = (json.loads(line) for line in handle if line.strip())
        for frame, record in zip_longest(range(frame_count), records):
            if frame is None or record is None or record["frame"] != frame:
                raise ValueError("assignments must cover each source frame exactly once in order")
            slots = record["slots"]
            assigned_ids = {
                int(slots[slot]["assignment"]["track_id"])
                for slot in roles if slots[slot]["assignment"] is not None
            }
            tracks = tracks_by_frame[frame]
            if assigned_ids - tracks:
                raise ValueError(f"assignment references an absent person track on frame {frame}")
            for slot in roles:
                if slots[slot]["assignment"] is not None:
                    counts[slot]["assigned"] += 1
                elif same_half[slot][frame] - assigned_ids:
                    counts[slot]["available"] += 1
                    reasons[slot][slots[slot]["ambiguity_reason"] or "unspecified"] += 1
                elif tracks:
                    counts[slot]["no_free_track"] += 1
                else:
                    counts[slot]["no_tracks"] += 1

    return [
        f"Source {_cell(metadata.get('source_id') or args.assignments.parent.name)}. "
        f"{frame_count:,} source frames per slot.",
        "",
        "| Player slot | Assigned | Missing with unused same-half track | "
        "Missing with tracks but none available on that half | Missing with zero tracks |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| {slot} ({role}) | " + " | ".join(
                f"{counts[slot][key]:,} ({counts[slot][key] / frame_count:.1%})"
                for key in ("assigned", "available", "no_free_track", "no_tracks")
            ) + " |"
            for slot, role in roles.items()
        ],
        "",
        "### Reasons for missing slots with an unused same-half track",
        "",
        *[
            f"- {slot} " + (", ".join(f"{_cell(reason)} {count:,}"
                                       for reason, count in reasons[slot].most_common()) or "none")
            for slot in roles
        ],
        "",
        "Court half is based on the bounding-box bottom center and the assignment court margin. "
        "An available track can belong to a different person. With no available track, the "
        "saved artifacts leave detection, tracking, and court projection as possible causes. "
        "The person-track artifact contains only boxes with ByteTrack IDs.",
    ]


def _tracker_comparison(args: argparse.Namespace, output: Path) -> list[str]:
    from ultralytics import YOLO, __version__
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml

    from InPlay.heuristic.person_tracks import PERSON_CLASS_ID, raw_artifact_fingerprint
    from InPlay.heuristic.player_interpretation import assign_singles_slots, load_calibration
    from InPlay.heuristic.players import ensure_person_detector
    from src.single_video.video import read_video_info

    video = read_video_info(args.video)
    frame_size = (int(video["width"]), int(video["height"]))
    fps = float(video["fps"])
    expected = int(video["frame_count"])
    if args.max_frames is not None:
        expected = min(expected, args.max_frames)
    court = load_calibration(args.court_calibration, frame_size)
    detector = YOLO(str(args.model))
    ensure_person_detector(detector, str(args.model))
    byte_config = YAML.load(check_yaml("bytetrack.yaml"))
    bot_config = YAML.load(check_yaml("botsort.yaml"))
    bot_config.update(gmc_method="none", with_reid=True, model="auto")
    tracker_config = output / "botsort.yaml"
    YAML.save(tracker_config, bot_config)
    byte_tracker = BYTETracker(IterableSimpleNamespace(**byte_config))
    byte_observations = []
    bot_observations = []
    processed = 0
    detection_count = 0

    def record_boxes(observations, boxes, frame):
        for x1, y1, x2, y2, track_id, confidence, *_ in boxes:
            observations.append({
                "frame": frame, "track_id": int(track_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(confidence),
            })

    def update_bytetrack(predictor):
        # Registered before Ultralytics replaces detections with BoT-SORT results.
        nonlocal processed, detection_count
        result, = predictor.results
        boxes = result.boxes.cpu().numpy()
        detection_count += len(boxes)
        record_boxes(byte_observations, byte_tracker.update(boxes, result.orig_img), processed)
        processed += 1

    detector.add_callback("on_predict_postprocess_end", update_bytetrack)
    results = detector.track(
        source=str(args.video), classes=[PERSON_CLASS_ID], tracker=str(tracker_config),
        stream=True, persist=True, verbose=False, device=args.device,
    )
    frames_read = 0
    with closing(results):
        for frame, result in enumerate(islice(results, args.max_frames)):
            if result.boxes.id is not None:
                record_boxes(bot_observations, result.boxes.cpu().numpy().data, frame)
            frames_read += 1
            if frames_read % 1000 == 0:
                print(f"{args.video.name}: {frames_read:,}/{expected:,} frames", flush=True)
    if not processed == frames_read == expected or not frames_read:
        raise ValueError(f"expected {expected:,} frames, decoded {frames_read:,}")
    if detector.predictor.trackers[0].args.model != "auto":
        raise ValueError("BoT-SORT requires native detector features for this comparison")

    lines = [
        f"**{_cell(args.video.name)}** · {frames_read:,} frames at {fps:g} FPS.",
        "",
        f"Both trackers received the same {detection_count:,} YOLO person detections "
        f"(Ultralytics {__version__}, device {detector.predictor.device}). "
        "BoT-SORT used native appearance features with camera-motion compensation disabled. "
        "Both used the bundled confidence thresholds and track buffer.",
        "",
        "| Tracker | Tracked boxes | Frames with both slots | P1 covered | P2 covered | "
        "P1/P2 selected track absent | P1/P2 assigned ID changes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, observations in (("ByteTrack", byte_observations), ("BoT-SORT + ReID", bot_observations)):
        assignments = assign_singles_slots(observations, court, frames_read, fps, frame_size=frame_size)
        covered = Counter()
        absent = Counter()
        changes = Counter()
        last_id = {}
        both = 0
        for row in assignments:
            slots = row["slots"]
            both += all(slot["assignment"] is not None for slot in slots.values())
            for role, slot in slots.items():
                assignment = slot["assignment"]
                if assignment is None:
                    absent[role] += slot["ambiguity_reason"] == "selected_track_absent"
                    continue
                track_id = assignment["track_id"]
                covered[role] += 1
                changes[role] += role in last_id and last_id[role] != track_id
                last_id[role] = track_id
        lines.append(
            f"| {name} | {len(observations):,} | {both:,} ({both / frames_read:.1%}) | "
            f"{covered['P1']:,} ({covered['P1'] / frames_read:.1%}) | "
            f"{covered['P2']:,} ({covered['P2'] / frames_read:.1%}) | "
            f"{absent['P1']:,} / {absent['P2']:,} | {changes['P1']:,} / {changes['P2']:,} |"
        )
    return lines + [
        "",
        "Coverage measures populated slots, including any incorrect assignments. "
        "Assigned ID changes include changes across missing frames. Pose extraction was skipped.",
        "",
        f"Detector `{_cell(args.model.name)}` ({raw_artifact_fingerprint(args.model)}).",
        "[BoT-SORT settings](botsort.yaml)",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    intervals = commands.add_parser("intervals", help="Compare two saved interval CSVs.")
    intervals.add_argument("--labels", type=Path, required=True)
    intervals.add_argument("--iou-threshold", type=float, default=0.5)
    intervals.add_argument("--limit", type=int, choices=range(1, 11), default=5)
    video = commands.add_parser("rally-video", help="Render selected rallies from saved InPlay output.")
    video.add_argument("--config", type=Path, required=True)
    video.add_argument("--table", type=Path, required=True, help="Saved rally-diagnostics.csv.")
    video.add_argument("--source", required=True, help="Exact source ID.")
    video.add_argument("--rally", action="append", required=True, help="Exact rally ID. Repeat up to five times.")
    video.add_argument("--context-frames", type=int, default=0)
    video.add_argument("--counterfactual-predictions", type=Path)
    for command in (intervals, video):
        command.add_argument("--predictions", type=Path, required=True)
    player = commands.add_parser("player-assignment", help="Compare tracked boxes with singles assignments.")
    player.add_argument("--person-tracks", type=Path, required=True)
    player.add_argument("--assignments", type=Path, required=True)
    player.add_argument("--court-calibration", type=Path, required=True)
    trackers = commands.add_parser("trackers", help="Compare ByteTrack and BoT-SORT on shared detections.")
    trackers.add_argument("--video", type=Path, required=True)
    trackers.add_argument("--model", type=Path, required=True, help="Local person-detector weights.")
    trackers.add_argument("--court-calibration", type=Path, required=True)
    trackers.add_argument("--device", required=True)
    trackers.add_argument("--max-frames", type=int, help="Measure only this many frames from the start.")
    for command in (intervals, video, player, trackers):
        command.add_argument("--output-dir", type=Path, required=True, help="New artifact directory.")

    args = parser.parse_args()
    if args.command == "intervals":
        if not 0 < args.iou_threshold <= 1:
            parser.error("--iou-threshold must be greater than 0 and at most 1")
        run = _intervals
    elif args.command == "rally-video":
        if len(args.rally) > 5 or len(set(args.rally)) != len(args.rally):
            parser.error("supply up to five distinct --rally IDs")
        if args.context_frames < 0:
            parser.error("--context-frames must be non-negative")
        run = _rally_video
    elif args.command == "player-assignment":
        run = _player_assignment
    else:
        if args.max_frames is not None and args.max_frames <= 0:
            parser.error("--max-frames must be positive")
        if not args.model.is_file():
            parser.error("--model must be an existing local weights file")
        run = _tracker_comparison

    output = args.output_dir
    try:
        if output.exists():
            raise FileExistsError(f"output directory already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".diagnostics-", dir=output.parent) as temporary:
            staging = Path(temporary)
            report = run(args, staging)
            (staging / "report.md").write_text(
                "\n".join(["# Rally diagnostics", "", *report, ""]),
                encoding="utf-8",
            )
            staging.rename(output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"diagnostics: {error}\n")
    print(output / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
