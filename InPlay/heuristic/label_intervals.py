"""Keyboard-driven, source-aware rally interval labeler.

Keys: left/right step one frame, j/l step ten, s marks start, e marks end,
w writes the existing evaluation CSV, and q quits.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from .evaluate import LABEL_FIELDS


def label_video(video: str | Path, source_id: str, output: str | Path) -> None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
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
            intervals.append((start, frame)); start = None
        elif key == ord("w"):
            write_labels(output, source_id, intervals)
        elif key in (2555904, ord("l")):
            frame = min(frame_count - 1, frame + (10 if key == ord("l") else 1))
        elif key in (2424832, ord("j")):
            frame = max(0, frame - (10 if key == ord("j") else 1))
    capture.release()
    cv2.destroyWindow(window)
    write_labels(output, source_id, intervals)


def write_labels(path: str | Path, source_id: str, intervals: list[tuple[int, int]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(LABEL_FIELDS,
            key=("source_id", "rally_id", "start_frame", "end_frame").index))
        writer.writeheader()
        for number, (start, end) in enumerate(intervals, 1):
            writer.writerow({"source_id": source_id, "rally_id": f"{source_id}-{number:04d}",
                             "start_frame": start, "end_frame": end})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True); parser.add_argument("--source-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    label_video(args.video, args.source_id, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
