"""Verify built footage-augmentation arms against their parent recordings.

Produces side-by-side frame/pose contact sheets, numeric court-coordinate
invariants, and media-property checks.  Read-only with respect to the sources.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.pose_estimator import pose_connections

POSE_CONNECTIONS = pose_connections()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _load_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        metadata = json.loads(next(handle))
        rows = [json.loads(line) for line in handle if line.strip()]
    return metadata, rows


def _frame_sheet(
    videos: Mapping[str, Path],
    *,
    frame: int,
    fps: float,
    output: Path,
) -> None:
    images = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, video in videos.items():
            image_path = Path(directory) / f"{name}.png"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    f"{frame / fps:.6f}",
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    str(image_path),
                ],
                check=True,
            )
            images[name] = cv2.imread(str(image_path))
    if any(image is None for image in images.values()):
        raise RuntimeError("could not decode a verification frame")
    width = 480
    height = int(round(width * 9 / 16))
    columns = list(images)
    canvas = np.full((height + 30, width * len(columns), 3), 30, dtype=np.uint8)
    for index, name in enumerate(columns):
        tile = cv2.resize(images[name], (width, height), interpolation=cv2.INTER_AREA)
        canvas[30:, index * width : (index + 1) * width] = tile
        cv2.putText(
            canvas,
            name,
            (index * width + 8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def _draw_pose(image: np.ndarray, pose: Mapping[str, Any], color: tuple[int, int, int]) -> None:
    bbox = pose.get("pose_bbox")
    landmarks = pose.get("pose_landmarks")
    if not isinstance(bbox, list) or len(bbox) != 4 or not isinstance(landmarks, list):
        return
    points = []
    for landmark in landmarks:
        x = float(bbox[0]) + float(landmark["x"]) * (float(bbox[2]) - float(bbox[0]))
        y = float(bbox[1]) + float(landmark["y"]) * (float(bbox[3]) - float(bbox[1]))
        visible = float(landmark.get("visibility", 0)) >= 0.5
        points.append((int(round(x)), int(round(y))) if visible else None)
    for first, second in POSE_CONNECTIONS:
        if first < len(points) and second < len(points):
            if points[first] is not None and points[second] is not None:
                cv2.line(image, points[first], points[second], color, 2, cv2.LINE_AA)
    for point in points:
        if point is not None:
            cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)


def _pose_sheet(
    recordings: Mapping[str, dict[str, Path]],
    *,
    frame: int,
    fps: float,
    output: Path,
) -> None:
    """Parent and flipped variant with their own re-extracted poses."""
    columns: list[tuple[str, np.ndarray]] = []
    with tempfile.TemporaryDirectory() as directory:
        for name, paths in recordings.items():
            image_path = Path(directory) / f"{name}.png"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    f"{frame / fps:.6f}",
                    "-i",
                    str(paths["video"]),
                    "-frames:v",
                    "1",
                    str(image_path),
                ],
                check=True,
            )
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"could not decode pose frame for {name}")
            _, poses = _load_jsonl(paths["poses"])
            for pose in poses:
                if int(pose["frame"]) == frame and pose.get("status") == "detected":
                    _draw_pose(image, pose, (0, 255, 255))
            columns.append((name, image))
    width = 640
    height = int(round(width * 9 / 16))
    canvas = np.full((height + 30, width * len(columns), 3), 30, dtype=np.uint8)
    for index, (name, image) in enumerate(columns):
        canvas[30:, index * width : (index + 1) * width] = cv2.resize(
            image, (width, height), interpolation=cv2.INTER_AREA
        )
        cv2.putText(
            canvas,
            name,
            (index * width + 8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def _assignments(path: Path) -> dict[int, dict[str, Any]]:
    _, rows = _load_jsonl(path)
    return {int(row["frame"]): row for row in rows}


def _invariants(
    parent: Mapping[int, dict[str, Any]],
    variant: Mapping[int, dict[str, Any]],
    *,
    image_width: int,
    mirrored: bool,
) -> dict[str, Any]:
    court_x: list[float] = []
    court_y: list[float] = []
    foot_x: list[float] = []
    foot_y: list[float] = []
    for frame in sorted(set(parent) & set(variant)):
        for role in ("P1", "P2"):
            first = parent[frame].get("slots", {}).get(role, {}).get("assignment")
            second = variant[frame].get("slots", {}).get(role, {}).get("assignment")
            if not isinstance(first, dict) or not isinstance(second, dict):
                continue
            court_x.append(second["court_x"] - (-first["court_x"] if mirrored else first["court_x"]))
            court_y.append(second["court_y"] - first["court_y"])
            expected_foot_x = (image_width - 1 - first["foot"][0]) if mirrored else first["foot"][0]
            foot_x.append(second["foot"][0] - expected_foot_x)
            foot_y.append(second["foot"][1] - first["foot"][1])

    def summary(values: Sequence[float]) -> dict[str, float] | None:
        finite = [value for value in values if math.isfinite(value)]
        if not finite:
            return None
        ordered = sorted(abs(value) for value in finite)
        return {
            "count": len(finite),
            "mean": statistics.fmean(finite),
            "median": statistics.median(finite),
            "p95_absolute": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        }

    return {
        "court_x_residual": summary(court_x),
        "court_y_residual": summary(court_y),
        "foot_x_residual": summary(foot_x),
        "foot_y_residual": summary(foot_y),
    }


def _media(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(result.stdout)
    video = next(
        stream for stream in value["streams"] if stream["codec_type"] == "video"
    )
    audio = next(
        (stream for stream in value["streams"] if stream["codec_type"] == "audio"),
        None,
    )
    return {
        "frame_count": int(video.get("nb_frames") or 0),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "avg_frame_rate": video.get("avg_frame_rate"),
        "duration": value["format"].get("duration"),
        "codec": video.get("codec_name"),
        "audio": (
            {
                "codec": audio.get("codec_name"),
                "sample_rate": audio.get("sample_rate"),
                "channels": audio.get("channels"),
                "duration": audio.get("duration"),
                "nb_frames": audio.get("nb_frames"),
            }
            if audio
            else None
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--rallies", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    config_path = args.parent_config.expanduser().resolve()
    config = _read(config_path)
    source = next(
        row for row in config["dataset"]["sources"]
        if row["source_id"] == args.source_id
    )
    base_dir = config_path.parent
    parent = {
        "video": _resolve(base_dir, source["video_path"]),
        "assignments": _resolve(base_dir, source["assignments_path"]),
        "poses": _resolve(base_dir, source["pose_cache_path"]),
    }
    arms = ("horizontal-flip", "lighting-color", "h264-low", "h264-control")
    variants: dict[str, dict[str, Path]] = {}
    for arm in arms:
        arm_config_path = run_root / arm / "experiment.json"
        if not arm_config_path.is_file():
            continue
        arm_config = _read(arm_config_path)
        entry = next(
            (
                row
                for row in arm_config["dataset"]["sources"]
                if row["source_id"].startswith(f"{args.source_id}__")
            ),
            None,
        )
        if entry is None:
            continue
        variants[arm] = {
            "video": Path(entry["video_path"]),
            "assignments": Path(entry["assignments_path"]),
            "poses": Path(entry["pose_cache_path"]),
        }
    if not variants:
        raise ValueError(f"no built variants found for {args.source_id}")
    fps = 30.0
    frames = list(args.frames or [])
    if not frames and args.rallies and args.rallies.is_file():
        import csv

        with args.rallies.open(newline="", encoding="utf-8") as handle:
            rows = [
                row for row in csv.DictReader(handle)
                if row["source_id"] == args.source_id
            ]
        for row in rows[:2]:
            frames.extend([int(row["start_frame"]) + 5, int(row["end_frame"]) - 5])
    frames = frames or [300, 900]
    output_dir = args.output_dir or (run_root / "verification" / args.source_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_assignments = _assignments(parent["assignments"])
    report: dict[str, Any] = {
        "source_id": args.source_id,
        "parent": {name: str(path) for name, path in parent.items()},
        "frames": frames,
        "media": {"parent": _media(parent["video"])},
        "invariants": {},
        "sheets": [],
    }
    for arm, paths in variants.items():
        report["media"][arm] = _media(paths["video"])
        report["invariants"][arm] = _invariants(
            parent_assignments,
            _assignments(paths["assignments"]),
            image_width=int(report["media"]["parent"]["width"]),
            mirrored=arm == "horizontal-flip",
        )
    for frame in frames:
        videos = {args.source_id: parent["video"]}
        videos.update({arm: paths["video"] for arm, paths in variants.items()})
        sheet = output_dir / f"frames-{frame:09d}.png"
        _frame_sheet(videos, frame=frame, fps=fps, output=sheet)
        report["sheets"].append(str(sheet))
        if "horizontal-flip" in variants:
            pose_sheet = output_dir / f"pose-{frame:09d}.png"
            _pose_sheet(
                {
                    args.source_id: parent,
                    "horizontal-flip": variants["horizontal-flip"],
                },
                frame=frame,
                fps=fps,
                output=pose_sheet,
            )
            report["sheets"].append(str(pose_sheet))
    (output_dir / "verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
