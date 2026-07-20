"""Keyboard-driven, source-aware rally interval labeler.

Keys: left/right or a/d step one frame, j/l step ten, s marks start, e marks end,
w writes the existing evaluation CSV, and q quits.
"""

from __future__ import annotations

import argparse
import csv
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


def label_video(
    video: str | Path,
    source_id: str,
    output: str | Path,
    manifest: str | Path | None = None,
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
    frame, start, intervals = 0, None, []
    window = f"Rally labeler: {source_id}"
    while True:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            break
        cv2.putText(image, f"{source_id} frame={frame} start={start}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(window, image)
        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            start = frame
        elif key == ord("e") and start is not None and frame >= start:
            intervals.append((start, frame))
            start = None
        elif key == ord("w"):
            write_labels(
                output,
                source_id,
                intervals,
                manifest_path=manifest,
                source_manifest=source_manifest,
            )
        elif key in RIGHT_KEYS or key in (ord("d"), ord("l")):
            frame = min(frame_count - 1, frame + (10 if key == ord("l") else 1))
        elif key in LEFT_KEYS or key in (ord("a"), ord("j")):
            frame = max(0, frame - (10 if key == ord("j") else 1))
    capture.release()
    cv2.destroyWindow(window)
    write_labels(
        output,
        source_id,
        intervals,
        manifest_path=manifest,
        source_manifest=source_manifest,
    )


def write_labels(
    path: str | Path,
    source_id: str,
    intervals: list[tuple[int, int]],
    *,
    manifest_path: str | Path | None = None,
    source_manifest: RallySourceManifest | None = None,
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
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            raise ValueError("rally interval CSV and manifest must exist together")
        existing = read_rally_intervals(path, manifest_path)
        existing_intervals.extend(
            item for item in existing.intervals if item.source_id != source_id
        )
        existing_sources.update(existing.sources)
    existing_sources[source_id] = source_manifest
    existing_intervals.extend(
        RallyInterval(source_id, f"{source_id}-{number:04d}", start, end)
        for number, (start, end) in enumerate(intervals, 1)
    )
    write_rally_intervals(
        path,
        manifest_path,
        existing_intervals,
        existing_sources.values(),
        annotation_revision=str(uuid.uuid4()),
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
