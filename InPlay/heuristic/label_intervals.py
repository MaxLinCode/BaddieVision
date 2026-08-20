"""Keyboard-driven, source-aware rally interval labeler.

Keys: left/right or a/d step one frame, j/l step ten, s marks start, e marks end,
x removes the rally under the cursor, p marks a frame-0 rally as already in play
when the clip began, w writes, and q quits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import uuid
from pathlib import Path

import cv2

from .evaluate import LABEL_FIELDS
from src.temporal_selector.rally_intervals import (
    RallyInterval,
    RallySourceManifest,
    file_sha256,
    read_rally_intervals,
    write_rally_intervals,
)


# OpenCV reports navigation keys differently on Windows, macOS, and Linux.
LEFT_KEYS = {81, 63234, 65361, 2424832}
RIGHT_KEYS = {83, 63235, 65363, 2555904}
TIMELINE_MARGIN = 30
TIMELINE_HEIGHT = 36
TIMELINE_BOTTOM_MARGIN = 62
LOCAL_TIMELINE_HEIGHT = 28
LOCAL_TIMELINE_BOTTOM_MARGIN = 18


def timeline_frame(x: int, width: int, frame_count: int) -> int:
    """Map an image-space timeline position to a source frame."""
    left, right = TIMELINE_MARGIN, max(TIMELINE_MARGIN + 1, width - TIMELINE_MARGIN)
    ratio = (min(max(x, left), right) - left) / (right - left)
    return round(ratio * max(0, frame_count - 1))


def local_timeline_frame(x: int, width: int, frame: int, frame_count: int,
                         fps: float) -> int:
    left, right = TIMELINE_MARGIN, max(TIMELINE_MARGIN + 1, width - TIMELINE_MARGIN)
    radius = max(1, round(3 * fps))
    local_left = max(0, frame - radius)
    local_right = min(frame_count - 1, frame + radius)
    ratio = (min(max(x, left), right) - left) / (right - left)
    return round(local_left + ratio * max(1, local_right - local_left))


def selected_interval(frame: int, intervals: list[tuple[int, int]]) -> tuple[int, int] | None:
    return next((item for item in intervals if item[0] <= frame <= item[1]), None)


def rally_boundaries(intervals: list[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(sorted(frame for interval in intervals for frame in interval))


def nearest_boundary(frame: int, intervals: list[tuple[int, int]]) -> int | None:
    boundaries = rally_boundaries(intervals)
    return min(boundaries, key=lambda value: (abs(value - frame), value)) if boundaries else None


def adjacent_boundary(frame: int, intervals: list[tuple[int, int]], direction: int) -> int | None:
    boundaries = rally_boundaries(intervals)
    candidates = [value for value in boundaries if value > frame] if direction > 0 else [value for value in boundaries if value < frame]
    return (min(candidates) if direction > 0 else max(candidates)) if candidates else None


def adjusted_boundary(intervals: list[tuple[int, int]], frame: int,
                      edge: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the target interval and its safe replacement at the cursor."""
    selected = selected_interval(frame, intervals)
    if edge == "start":
        target = selected or next((item for item in intervals if item[0] > frame), None)
        if target is None or frame > target[1]:
            raise ValueError("no rally start can be moved to this frame")
        replacement = (frame, target[1])
    elif edge == "end":
        target = selected or next((item for item in reversed(intervals) if item[1] < frame), None)
        if target is None or frame < target[0]:
            raise ValueError("no rally end can be moved to this frame")
        replacement = (target[0], frame)
    else:
        raise ValueError("rally edge must be start or end")
    others = [item for item in intervals if item != target]
    if any(not (replacement[1] < left or replacement[0] > right) for left, right in others):
        raise ValueError("adjusted boundary would overlap another rally")
    return target, replacement


def merge_across_cursor(intervals: list[tuple[int, int]], frame: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return the two rallies surrounding a gap and their merged replacement."""
    if selected_interval(frame, intervals) is not None:
        raise ValueError("place the cursor in the gap between rallies to merge them")
    left = next((item for item in reversed(intervals) if item[1] < frame), None)
    right = next((item for item in intervals if item[0] > frame), None)
    if left is None or right is None:
        raise ValueError("cursor is not between two rallies")
    return left, right, (left[0], right[1])


def draw_timeline(
    image,
    frame: int,
    frame_count: int,
    intervals: list[tuple[int, int]],
    start: int | None,
    partial_starts: set[tuple[int, int]] | None = None,
    agreements: dict[tuple[int, int], str] | None = None,
    fps: float = 30.0,
) -> None:
    """Draw a large overview plus a frame-precise local boundary strip."""
    height, width = image.shape[:2]
    left, right = TIMELINE_MARGIN, width - TIMELINE_MARGIN
    bottom = height - TIMELINE_BOTTOM_MARGIN
    top = bottom - TIMELINE_HEIGHT
    span = max(1, frame_count - 1)

    cv2.rectangle(image, (left - 2, top - 2), (right + 2, bottom + 2), (0, 0, 0), -1)
    cv2.rectangle(image, (left, top), (right, bottom), (75, 75, 75), -1)
    partial_starts = partial_starts or set()
    agreements = agreements or {}
    selected = selected_interval(frame, intervals)
    for interval_start, interval_end in intervals:
        x1 = left + round(interval_start / span * (right - left))
        x2 = left + round(interval_end / span * (right - left))
        color = ((70, 190, 255) if agreements.get((interval_start, interval_end)) == "2/3"
                 else (170, 70, 220) if agreements.get((interval_start, interval_end)) == "3/3"
                 else (190, 120, 35) if (interval_start, interval_end) in partial_starts
                 else (40, 190, 40))
        cv2.rectangle(image, (x1, top), (max(x1 + 1, x2), bottom), color, -1)
        cv2.line(image, (x1, top - 3), (x1, bottom + 3), (255, 255, 255), 1)
        cv2.line(image, (x2, top - 3), (x2, bottom + 3), (255, 255, 255), 1)
        if selected == (interval_start, interval_end):
            cv2.rectangle(image, (x1 - 2, top - 3), (max(x1 + 3, x2 + 2), bottom + 3),
                          (0, 255, 255), 3)
    if start is not None:
        x1 = left + round(start / span * (right - left))
        x2 = left + round(frame / span * (right - left))
        cv2.rectangle(image, (min(x1, x2), top), (max(x1 + 1, x2), bottom),
                      (0, 180, 255), -1)
    cursor = left + round(frame / span * (right - left))
    cv2.line(image, (cursor, top - 5), (cursor, bottom + 5), (255, 255, 255), 2)
    cv2.putText(image, "rallies", (left, top - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (40, 255, 40), 1)

    local_bottom = height - LOCAL_TIMELINE_BOTTOM_MARGIN
    local_top = local_bottom - LOCAL_TIMELINE_HEIGHT
    radius = max(1, round(3 * fps))
    local_left_frame = max(0, frame - radius)
    local_right_frame = min(frame_count - 1, frame + radius)
    local_span = max(1, local_right_frame - local_left_frame)
    cv2.rectangle(image, (left - 2, local_top - 2), (right + 2, local_bottom + 2), (0, 0, 0), -1)
    cv2.rectangle(image, (left, local_top), (right, local_bottom), (45, 45, 45), -1)
    for interval_start, interval_end in intervals:
        if interval_end < local_left_frame or interval_start > local_right_frame:
            continue
        x1 = left + round((max(interval_start, local_left_frame) - local_left_frame) / local_span * (right - left))
        x2 = left + round((min(interval_end, local_right_frame) - local_left_frame) / local_span * (right - left))
        color = (0, 255, 255) if selected == (interval_start, interval_end) else (70, 190, 255)
        cv2.rectangle(image, (x1, local_top), (max(x1 + 1, x2), local_bottom), color, -1)
        if local_left_frame <= interval_start <= local_right_frame:
            cv2.line(image, (x1, local_top - 5), (x1, local_bottom + 4), (255, 255, 255), 2)
            cv2.putText(image, f"S {interval_start}", (max(left, x1 - 35), local_top - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if local_left_frame <= interval_end <= local_right_frame:
            cv2.line(image, (x2, local_top - 5), (x2, local_bottom + 4), (255, 255, 255), 2)
            cv2.putText(image, f"E {interval_end}", (max(left, x2 - 35), local_top - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    local_cursor = left + round((frame - local_left_frame) / local_span * (right - left))
    cv2.line(image, (local_cursor, local_top - 7), (local_cursor, local_bottom + 7), (255, 0, 0), 2)
    cv2.putText(image, "DETAIL +/-3s", (right - 125, local_top + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


def label_video(
    video: str | Path,
    source_id: str,
    output: str | Path,
    manifest: str | Path | None = None,
    *,
    proposals: str | Path | None = None,
    draft: str | Path | None = None,
    receipt: str | Path | None = None,
    annotator: str | None = None,
) -> None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_manifest = RallySourceManifest(
        source_id,
        file_sha256(video),
        fps,
        frame_count,
    )
    pseudo = proposals is not None
    agreements: dict[tuple[int, int], str] = {}
    proposal_fingerprint = None
    teacher_fingerprint = None
    if pseudo:
        proposal_value = load_proposals(proposals, source_id, source_manifest)
        proposal_fingerprint = str(proposal_value["fingerprint"])
        teacher_fingerprint = str(proposal_value["metadata"]["teacher_fingerprint"])
        intervals = [(int(item["start_frame"]), int(item["end_frame"]))
                     for item in proposal_value["proposals"]]
        agreements = {(int(item["start_frame"]), int(item["end_frame"])): str(item["agreement"])
                      for item in proposal_value["proposals"]}
        partial_starts: set[tuple[int, int]] = set()
        if draft is None or annotator is None:
            raise ValueError("pseudo review requires a draft path and annotator")
        resumed = load_review_draft(draft, source_id, proposal_fingerprint)
        if resumed is not None:
            intervals, partial_starts = resumed
    else:
        intervals, partial_starts = load_existing_annotations(
            output, manifest, source_id, source_manifest,
        )
    frame = intervals[0][0] if pseudo and intervals else (
        min(frame_count - 1, intervals[-1][1] + 1) if intervals else 0
    )
    navigation = {"frame": frame, "width": 0, "dragging": False}

    def seek(event: int, x: int, y: int, flags: int, _parameter: object) -> None:
        width = navigation["width"]
        if not width:
            return
        timeline_bottom = navigation.get("height", 0) - TIMELINE_BOTTOM_MARGIN
        timeline_top = timeline_bottom - TIMELINE_HEIGHT
        local_bottom = navigation.get("height", 0) - LOCAL_TIMELINE_BOTTOM_MARGIN
        local_top = local_bottom - LOCAL_TIMELINE_HEIGHT
        if event == cv2.EVENT_LBUTTONDOWN and local_top - 8 <= y <= local_bottom + 8:
            navigation["frame"] = local_timeline_frame(
                x, width, navigation["frame"], frame_count, fps
            )
            return
        if event == cv2.EVENT_LBUTTONDOWN and timeline_top - 8 <= y <= timeline_bottom + 8:
            navigation["dragging"] = True
        if event == cv2.EVENT_LBUTTONUP:
            navigation["dragging"] = False
        if navigation["dragging"] and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            navigation["frame"] = timeline_frame(x, width, frame_count)

    start = None
    undo_stack: list[tuple[list[tuple[int, int]], set[tuple[int, int]],
                           dict[tuple[int, int], str], int | None]] = []

    def remember() -> None:
        undo_stack.append((list(intervals), set(partial_starts), dict(agreements), start))

    status = f"resumed {len(intervals)} saved rallies" if intervals else ""
    window = f"Rally labeler: {source_id}"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, seek)
    while True:
        frame = navigation["frame"]
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            break
        selected = selected_interval(frame, intervals)
        position = "OUTSIDE"
        detail = ""
        if selected is not None:
            left, right = selected
            position = "START" if frame == left else "END" if frame == right else "INSIDE"
            detail = f" rally={left}-{right} from_start=+{frame-left} to_end=-{right-frame}"
        cv2.rectangle(image, (8, 7), (min(image.shape[1] - 8, 1160), 112), (0, 0, 0), -1)
        color = (0, 255, 255) if selected is not None else (150, 150, 150)
        cv2.putText(image, f"{source_id} frame={frame}  {position}{detail}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
        cv2.putText(image, f"pending_start={start} rallies={len(intervals)}  {status}", (20, 67),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(image, "B snap  [/] boundaries  Shift+S/E move edge  M merge gap  Z undo",
                    (20, 99), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1)
        navigation["width"] = image.shape[1]
        navigation["height"] = image.shape[0]
        draw_timeline(image, frame, frame_count, intervals, start, partial_starts, agreements, fps)
        cv2.imshow(window, image)
        key = cv2.waitKeyEx(20)
        if key == -1:
            continue
        if key in (ord("q"), 27):
            break
        if key in (ord("z"), ord("Z")):
            if undo_stack:
                saved_intervals, saved_partial, saved_agreements, saved_start = undo_stack.pop()
                intervals[:] = saved_intervals
                partial_starts.clear()
                partial_starts.update(saved_partial)
                agreements.clear()
                agreements.update(saved_agreements)
                start = saved_start
                status = f"undid last edit ({len(undo_stack)} undo step(s) remain)"
            else:
                status = "nothing to undo in this session"
        elif key == ord("S"):
            try:
                old, replacement = adjusted_boundary(intervals, frame, "start")
            except ValueError as error:
                status = str(error)
            else:
                remember()
                intervals[intervals.index(old)] = replacement
                agreements.pop(old, None)
                was_partial = old in partial_starts
                partial_starts.discard(old)
                if was_partial and replacement[0] == 0:
                    partial_starts.add(replacement)
                intervals.sort()
                status = f"moved rally start {old[0]} -> {frame}"
        elif key == ord("E"):
            try:
                old, replacement = adjusted_boundary(intervals, frame, "end")
            except ValueError as error:
                status = str(error)
            else:
                remember()
                intervals[intervals.index(old)] = replacement
                agreements.pop(old, None)
                if old in partial_starts:
                    partial_starts.remove(old)
                    partial_starts.add(replacement)
                intervals.sort()
                status = f"moved rally end {old[1]} -> {frame}"
        elif key in (ord("m"), ord("M")):
            try:
                left, right, merged = merge_across_cursor(intervals, frame)
            except ValueError as error:
                status = str(error)
            else:
                remember()
                intervals.remove(left)
                intervals.remove(right)
                intervals.append(merged)
                intervals.sort()
                agreements.pop(left, None)
                agreements.pop(right, None)
                was_partial = left in partial_starts
                partial_starts.discard(left)
                partial_starts.discard(right)
                if was_partial:
                    partial_starts.add(merged)
                status = f"merged {left[0]}-{left[1]} and {right[0]}-{right[1]}"
        elif key == ord("s"):
            if any(left <= frame <= right for left, right in intervals):
                status = "start overlaps an existing rally; press X there to replace it"
                start = None
            else:
                start = frame
                status = "start marked"
        elif key == ord("e") and start is not None and frame >= start:
            proposed = (start, frame)
            if any(not (proposed[1] < left or proposed[0] > right) for left, right in intervals):
                status = "rally overlaps an existing interval"
            else:
                remember()
                intervals.append(proposed)
                intervals.sort()
                agreements.pop(proposed, None)
                status = f"added rally {start}-{frame}"
                start = None
        elif key in (ord("x"), ord("u")):
            selected = next((item for item in intervals if item[0] <= frame <= item[1]), None)
            if selected is not None:
                remember()
                intervals.remove(selected)
                agreements.pop(selected, None)
                partial_starts.discard(selected)
                removed = selected
                navigation["frame"] = removed[0]
                start = None
                status = f"removed rally {removed[0]}-{removed[1]}"
            else:
                status = "no rally under cursor"
        elif key == ord("p"):
            selected = next((item for item in intervals if item[0] <= frame <= item[1]), None)
            if selected is None:
                status = "no rally under cursor"
            elif selected[0] != 0:
                status = "only a rally beginning at source frame 0 can be partial-start"
            elif selected in partial_starts:
                remember()
                partial_starts.remove(selected)
                status = "frame-0 rally now has an observed start"
            else:
                remember()
                partial_starts.add(selected)
                status = "frame-0 rally marked: already in play when source began"
        elif key == ord("w") and not pseudo:
            write_labels(
                output,
                source_id,
                intervals,
                manifest_path=manifest,
                source_manifest=source_manifest,
                partial_starts=partial_starts,
            )
            status = f"saved {len(intervals)} rallies"
        elif key in (ord("f"), ord("F")) and pseudo:
            status = "finalize armed; press Y to replace this source's canonical labels"
        elif key in (ord("y"), ord("Y")) and pseudo and status.startswith("finalize armed"):
            write_labels(output, source_id, intervals, manifest_path=manifest,
                         source_manifest=source_manifest, partial_starts=partial_starts)
            write_review_receipt(receipt, source_id=source_id, annotator=annotator,
                                 proposal_fingerprint=proposal_fingerprint,
                                 teacher_fingerprint=teacher_fingerprint,
                                 intervals_path=Path(output), manifest_path=Path(manifest))
            status = "review finalized"
            break
        elif key in (ord("b"), ord("B")):
            boundary = nearest_boundary(frame, intervals)
            if boundary is not None:
                navigation["frame"] = boundary
                status = f"snapped to boundary {boundary}"
        elif key == ord("["):
            boundary = adjacent_boundary(frame, intervals, -1)
            if boundary is not None:
                navigation["frame"] = boundary
                status = f"previous boundary {boundary}"
        elif key == ord("]"):
            boundary = adjacent_boundary(frame, intervals, 1)
            if boundary is not None:
                navigation["frame"] = boundary
                status = f"next boundary {boundary}"
        elif key in RIGHT_KEYS or key in (ord("d"), ord("l")):
            navigation["frame"] = min(
                frame_count - 1, frame + (10 if key == ord("l") else 1)
            )
        elif key in LEFT_KEYS or key in (ord("a"), ord("j")):
            navigation["frame"] = max(0, frame - (10 if key == ord("j") else 1))
        if pseudo and key not in (-1, ord("q"), 27):
            write_review_draft(draft, source_id=source_id, annotator=annotator,
                               proposal_fingerprint=proposal_fingerprint,
                               intervals=intervals, partial_starts=partial_starts)
    capture.release()
    cv2.destroyWindow(window)
    if not pseudo:
        write_labels(output, source_id, intervals, manifest_path=manifest,
                     source_manifest=source_manifest, partial_starts=partial_starts)


def _json_fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_proposals(path: str | Path, source_id: str,
                   source_manifest: RallySourceManifest) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = value.pop("fingerprint", None)
    if value.get("schema") != "rally_pseudo_proposals" or value.get("source_id") != source_id:
        raise ValueError("incompatible rally proposals")
    if claimed != _json_fingerprint(value):
        raise ValueError("rally proposal fingerprint differs")
    metadata = value.get("metadata", {})
    hashes = metadata.get("input_artifact_sha256", {})
    if hashes.get("video") != source_manifest.video_sha256 or int(metadata.get("frame_count", -1)) != source_manifest.frame_count:
        raise ValueError("rally proposals are stale for this video")
    value["fingerprint"] = claimed
    return value


def write_review_draft(path: str | Path, *, source_id: str, annotator: str,
                       proposal_fingerprint: str, intervals: list[tuple[int, int]],
                       partial_starts: set[tuple[int, int]]) -> Path:
    path = Path(path)
    value = {"schema": "rally_review_draft", "schema_version": 1, "source_id": source_id,
             "annotator": annotator, "proposal_fingerprint": proposal_fingerprint,
             "intervals": [list(item) for item in sorted(intervals)],
             "partial_starts": [list(item) for item in sorted(partial_starts)]}
    value["fingerprint"] = _json_fingerprint(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_review_draft(path: str | Path, source_id: str,
                      proposal_fingerprint: str) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    path = Path(path)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop("fingerprint", None)
    if claimed != _json_fingerprint(value) or value.get("schema") != "rally_review_draft":
        raise ValueError("rally review draft fingerprint differs")
    if value.get("source_id") != source_id or value.get("proposal_fingerprint") != proposal_fingerprint:
        raise ValueError("rally review draft is incompatible with these proposals")
    return ([tuple(map(int, item)) for item in value["intervals"]],
            {tuple(map(int, item)) for item in value["partial_starts"]})


def write_review_receipt(path: str | Path | None, *, source_id: str, annotator: str,
                         proposal_fingerprint: str, teacher_fingerprint: str,
                         intervals_path: Path,
                         manifest_path: Path) -> Path:
    if path is None:
        raise ValueError("pseudo review finalization requires a receipt path")
    path = Path(path)
    value = {"schema": "rally_review_receipt", "schema_version": 1, "source_id": source_id,
             "annotator": annotator, "proposal_fingerprint": proposal_fingerprint,
             "teacher_fingerprint": teacher_fingerprint,
             "rallies_sha256": file_sha256(intervals_path),
             "rally_manifest_sha256": file_sha256(manifest_path)}
    value["fingerprint"] = _json_fingerprint(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_existing_labels(
    path: str | Path,
    manifest_path: str | Path | None,
    source_id: str,
    source_manifest: RallySourceManifest,
) -> list[tuple[int, int]]:
    """Load this source's saved intervals so an interrupted session can resume."""
    return load_existing_annotations(path, manifest_path, source_id, source_manifest)[0]


def load_existing_annotations(
    path: str | Path,
    manifest_path: str | Path | None,
    source_id: str,
    source_manifest: RallySourceManifest,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """Load intervals and source-boundary state for an interrupted session."""
    path = Path(path)
    if manifest_path is None:
        return [], set()
    manifest_path = Path(manifest_path)
    if not path.exists() and not manifest_path.exists():
        return [], set()
    if not path.exists() or not manifest_path.exists():
        raise ValueError("rally interval CSV and manifest must exist together")
    existing = read_rally_intervals(path, manifest_path)
    saved_source = existing.sources.get(source_id)
    if saved_source is not None and saved_source != source_manifest:
        raise ValueError(f"saved rally source metadata does not match video: {source_id}")
    intervals = [
        (item.start_frame, item.end_frame)
        for item in existing.intervals
        if item.source_id == source_id
    ]
    partial = {
        (item.start_frame, item.end_frame)
        for item in existing.intervals
        if item.source_id == source_id and existing.starts_before_source(item.rally_id)
    }
    return intervals, partial


def write_labels(
    path: str | Path,
    source_id: str,
    intervals: list[tuple[int, int]],
    *,
    manifest_path: str | Path | None = None,
    source_manifest: RallySourceManifest | None = None,
    partial_starts: set[tuple[int, int]] | None = None,
) -> None:
    path = Path(path)
    if manifest_path is None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(LABEL_FIELDS,
                key=("source_id", "rally_id", "start_frame", "end_frame").index))
            writer.writeheader()
            for number, (start, end) in enumerate(intervals, 1):
                writer.writerow({"source_id": source_id, "rally_id": f"{source_id}-{number:04d}",
                                 "start_frame": start, "end_frame": end})
        return
    if source_manifest is None or source_manifest.source_id != source_id:
        raise ValueError("fingerprinted interval writes require matching source metadata")
    manifest_path = Path(manifest_path)
    existing_intervals: list[RallyInterval] = []
    existing_sources: dict[str, RallySourceManifest] = {}
    reviewed_coverage: dict[str, str | list[list[int]]] = {}
    partial_start_rally_ids: set[str] = set()
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            raise ValueError("rally interval CSV and manifest must exist together")
        existing = read_rally_intervals(path, manifest_path)
        existing_intervals.extend(
            item for item in existing.intervals if item.source_id != source_id
        )
        existing_sources.update(existing.sources)
        partial_start_rally_ids.update(
            rally_id
            for rally_id in existing.partial_start_rally_ids
            if any(item.rally_id == rally_id and item.source_id != source_id for item in existing.intervals)
        )
        for span in existing.reviewed_coverage:
            reviewed_coverage.setdefault(span.source_id, []).append(
                [span.start_frame, span.end_frame]
            )
    existing_sources[source_id] = source_manifest
    # The interactive labeler traverses this source as a complete sequence.
    # Declare only this source fully reviewed; preserve the explicit safe
    # coverage loaded for every other source.
    reviewed_coverage[source_id] = "full_source"
    partial_starts = partial_starts or set()
    for number, (start, end) in enumerate(sorted(intervals), 1):
        rally_id = f"{source_id}-{number:04d}"
        existing_intervals.append(RallyInterval(source_id, rally_id, start, end))
        if (start, end) in partial_starts:
            partial_start_rally_ids.add(rally_id)
    write_rally_intervals(
        path,
        manifest_path,
        existing_intervals,
        existing_sources.values(),
        annotation_revision=str(uuid.uuid4()),
        reviewed_coverage=reviewed_coverage,
        partial_start_rally_ids=partial_start_rally_ids,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    label_video(args.video, args.source_id, args.output, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
