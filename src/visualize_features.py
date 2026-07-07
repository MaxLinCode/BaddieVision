"""Render extracted badminton features back onto a source video."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from .court_features import Calibration, CalibrationRegistry, build_court_anchor_features
    from .court_projection import (
        HALF_LENGTH,
        HALF_WIDTH,
        CourtHomography,
        court_line_segments,
        draw_court_overlay,
    )
    from .extract_clip_features import load_shuttle_positions
    from .pose_estimator import pose_connections
except ImportError:
    from court_features import Calibration, CalibrationRegistry, build_court_anchor_features
    from court_projection import (
        HALF_LENGTH,
        HALF_WIDTH,
        CourtHomography,
        court_line_segments,
        draw_court_overlay,
    )
    from extract_clip_features import load_shuttle_positions
    from pose_estimator import pose_connections


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEOS_DIR = PROJECT_ROOT / "videos"
DEFAULT_POSE_DIR = PROJECT_ROOT / "features" / "pose"
DEFAULT_SHUTTLE_DIR = PROJECT_ROOT / "features" / "shuttle"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CALIBRATION_REGISTRY = PROJECT_ROOT / "features" / "court" / "calibrations.json"
POSE_NUM_KEYPOINTS = 33


@dataclass(frozen=True)
class FeatureStream:
    pose_frames: list[dict]
    shuttle: np.ndarray
    court: np.ndarray | None


def _normalize_base_name(value: str) -> str:
    return Path(value).stem.lower()


def _resolve_existing_path(path: Path, description: str) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(f"{description} not found: {path}")


def _find_case_insensitive(directory: Path, target_name: str) -> Path | None:
    target_name = target_name.lower()
    for candidate in directory.iterdir():
        if candidate.name.lower() == target_name:
            return candidate
    return None


def resolve_video_path(video: str | Path) -> Path:
    candidate = Path(video)
    if candidate.exists():
        return candidate
    fallback = DEFAULT_VIDEOS_DIR / str(video)
    return _resolve_existing_path(fallback, "video")


def resolve_artifact_path(explicit: str | Path | None, directory: Path, filename: str, description: str) -> Path:
    if explicit is not None:
        return _resolve_existing_path(Path(explicit), description)
    if not directory.exists():
        raise FileNotFoundError(f"{description} directory not found: {directory}")
    exact = next((candidate for candidate in directory.iterdir() if candidate.name == filename), None)
    if exact is not None:
        return exact
    direct = directory / filename
    matched = _find_case_insensitive(directory, filename)
    if matched is not None:
        return matched
    raise FileNotFoundError(f"{description} not found: expected {direct}")


def load_pose_frames(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"pose JSON must contain a list of frames: {path}")
    return data


def resolve_calibration(
    *,
    base_name: str,
    calibration_path: str | Path | None,
    calibration_registry: str | Path | None,
) -> Calibration | None:
    if calibration_path is not None:
        path = Path(calibration_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        image_size = data.get("image_size")
        if not isinstance(image_size, list) or len(image_size) != 2:
            raise ValueError(f"calibration {path} must contain image_size")
        return Calibration(CourtHomography.load(path), (int(image_size[0]), int(image_size[1])))
    if calibration_registry is None or str(calibration_registry).strip() == "":
        return None
    registry_path = Path(calibration_registry)
    if not registry_path.exists():
        return None
    registry = CalibrationRegistry(registry_path)
    return registry.calibration_for_clip(base_name)


def build_feature_stream(
    *,
    pose_path: str | Path,
    shuttle_path: str | Path,
    frame_width: int,
    frame_height: int,
    calibration: Calibration | None,
) -> FeatureStream:
    pose_frames = load_pose_frames(pose_path)
    shuttle = load_shuttle_positions(
        str(Path(shuttle_path).parent),
        Path(shuttle_path).name.replace("_ball.csv", ""),
        num_frames=len(pose_frames),
        frame_width=frame_width,
        frame_height=frame_height,
    )
    court = None
    if calibration is not None:
        court = build_court_anchor_features(
            pose_frames,
            calibration,
            num_frames=len(pose_frames),
        )
    return FeatureStream(pose_frames=pose_frames, shuttle=shuttle, court=court)


def _point_from_normalized(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    return (int(round(x * width)), int(round(y * height)))


def draw_pose(frame: np.ndarray, pose_frame: dict, *, visibility_threshold: float = 0.3) -> int:
    keypoints = pose_frame.get("keypoints", {})
    height, width = frame.shape[:2]
    visible_points: dict[int, tuple[int, int]] = {}
    for index in range(POSE_NUM_KEYPOINTS):
        landmark = keypoints.get(str(index), {})
        x = landmark.get("x")
        y = landmark.get("y")
        visibility = float(landmark.get("visibility", 0.0))
        if x is None or y is None or visibility < visibility_threshold:
            continue
        point = _point_from_normalized(float(x), float(y), width, height)
        visible_points[index] = point
    for start, end in pose_connections():
        if start in visible_points and end in visible_points:
            cv2.line(frame, visible_points[start], visible_points[end], (0, 255, 0), 2, cv2.LINE_AA)
    for index, point in visible_points.items():
        color = (0, 220, 255) if index >= 27 else (0, 255, 0)
        cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)
    return len(visible_points)


def draw_shuttle(frame: np.ndarray, shuttle_features: np.ndarray, trail: Iterable[tuple[int, int]]) -> None:
    for point in trail:
        cv2.circle(frame, point, 2, (0, 140, 255), -1, cv2.LINE_AA)
    if float(shuttle_features[2]) != 1.0:
        return
    height, width = frame.shape[:2]
    center = _point_from_normalized(float(shuttle_features[0]), float(shuttle_features[1]), width, height)
    cv2.circle(frame, center, 6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, center, 2, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_text_block(frame: np.ndarray, lines: list[str], *, origin: tuple[int, int]) -> None:
    x, y = origin
    line_height = 20
    width = max((len(line) for line in lines), default=0) * 8 + 16
    height = line_height * len(lines) + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - 16), (x + width, y - 16 + height), (24, 24, 24), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0.0, dst=frame)
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        y += line_height


def draw_court_minimap(
    frame: np.ndarray,
    *,
    anchor: np.ndarray | None,
    observed: bool,
    origin: tuple[int, int] | None = None,
    size: tuple[int, int] = (220, 140),
) -> None:
    width = frame.shape[1]
    height = frame.shape[0]
    panel_w, panel_h = size
    x0 = width - panel_w - 20 if origin is None else origin[0]
    y0 = height - panel_h - 20 if origin is None else origin[1]
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0.0, dst=frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (180, 180, 180), 1, cv2.LINE_AA)

    def map_court_point(point: tuple[float, float]) -> tuple[int, int]:
        px = (point[0] + HALF_WIDTH) / (2 * HALF_WIDTH)
        py = (point[1] + HALF_LENGTH) / (2 * HALF_LENGTH)
        return (
            int(round(x0 + px * panel_w)),
            int(round(y0 + py * panel_h)),
        )

    for start, end in court_line_segments():
        cv2.line(frame, map_court_point(start), map_court_point(end), (80, 220, 220), 1, cv2.LINE_AA)

    cv2.putText(
        frame,
        "Court Anchor",
        (x0 + 8, y0 + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    if anchor is None:
        return
    point = (float(anchor[0]) * HALF_WIDTH, float(anchor[1]) * HALF_LENGTH)
    color = (0, 200, 0) if observed else (0, 165, 255)
    cv2.circle(frame, map_court_point(point), 4, color, -1, cv2.LINE_AA)


def annotate_frame(
    frame: np.ndarray,
    *,
    frame_index: int,
    pose_frame: dict,
    shuttle_features: np.ndarray,
    court_features: np.ndarray | None,
    draw_projected_court: bool,
    calibration: Calibration | None,
    trail: Iterable[tuple[int, int]],
) -> np.ndarray:
    output = frame.copy()
    if draw_projected_court and calibration is not None:
        output = draw_court_overlay(output, calibration.homography, color=(255, 200, 0), thickness=2)

    visible_pose_points = draw_pose(output, pose_frame)
    draw_shuttle(output, shuttle_features, trail)

    court_anchor = None
    observed = False
    lines = [
        f"frame {frame_index}",
        f"pose visible {visible_pose_points}/{POSE_NUM_KEYPOINTS}",
        (
            "shuttle "
            f"x={shuttle_features[0]:.3f} y={shuttle_features[1]:.3f} "
            f"vis={int(shuttle_features[2])}"
        ),
        (
            "shuttle "
            f"vx={shuttle_features[3]:.2f} vy={shuttle_features[4]:.2f} "
            f"ax={shuttle_features[5]:.2f} ay={shuttle_features[6]:.2f}"
        ),
    ]
    if court_features is not None:
        court_anchor = court_features[:2]
        observed = bool(round(float(court_features[2])))
        lines.append(
            f"court x={court_features[0]:.3f} y={court_features[1]:.3f} observed={int(observed)}"
        )
    else:
        lines.append("court unavailable")

    _draw_text_block(output, lines, origin=(16, 28))
    draw_court_minimap(output, anchor=court_anchor, observed=observed)
    return output


def render_video(
    *,
    video_path: str | Path,
    pose_path: str | Path,
    shuttle_path: str | Path,
    output_path: str | Path,
    calibration: Calibration | None,
    max_frames: int | None = None,
    draw_projected_court: bool = True,
) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stream = build_feature_stream(
        pose_path=pose_path,
        shuttle_path=shuttle_path,
        frame_width=width,
        frame_height=height,
        calibration=calibration,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter: {output_path}")

    frame_limit = len(stream.pose_frames)
    if max_frames is not None:
        frame_limit = min(frame_limit, max_frames)
    trail: deque[tuple[int, int]] = deque(maxlen=10)
    try:
        for frame_index in range(frame_limit):
            ok, frame = capture.read()
            if not ok:
                break
            shuttle_features = stream.shuttle[frame_index]
            if float(shuttle_features[2]) == 1.0:
                trail.append(
                    _point_from_normalized(
                        float(shuttle_features[0]),
                        float(shuttle_features[1]),
                        width,
                        height,
                    )
                )
            court_features = None if stream.court is None else stream.court[frame_index]
            annotated = annotate_frame(
                frame,
                frame_index=frame_index,
                pose_frame=stream.pose_frames[frame_index],
                shuttle_features=shuttle_features,
                court_features=court_features,
                draw_projected_court=draw_projected_court,
                calibration=calibration,
                trail=trail,
            )
            writer.write(annotated)
    finally:
        capture.release()
        writer.release()
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Source video path or filename under videos/")
    parser.add_argument("--base-name", help="Override the artifact stem used for pose/shuttle lookup")
    parser.add_argument("--pose-json", help="Explicit pose JSON path")
    parser.add_argument("--shuttle-csv", help="Explicit shuttle CSV path")
    parser.add_argument("--calibration", help="Explicit court calibration JSON path")
    parser.add_argument(
        "--calibration-registry",
        default=str(DEFAULT_CALIBRATION_REGISTRY),
        help="Calibration registry used to resolve clip names",
    )
    parser.add_argument(
        "--output",
        help="Output MP4 path. Defaults to outputs/<base>_feature_overlay.mp4",
    )
    parser.add_argument("--max-frames", type=int, help="Stop rendering after N frames")
    parser.add_argument(
        "--skip-projected-court",
        action="store_true",
        help="Do not draw the full projected court on the video frame",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    video_path = resolve_video_path(args.video)
    base_name = _normalize_base_name(args.base_name or video_path.stem)
    pose_path = resolve_artifact_path(args.pose_json, DEFAULT_POSE_DIR, f"{base_name}_pose.json", "pose JSON")
    shuttle_path = resolve_artifact_path(args.shuttle_csv, DEFAULT_SHUTTLE_DIR, f"{base_name}_ball.csv", "shuttle CSV")
    calibration = resolve_calibration(
        base_name=base_name,
        calibration_path=args.calibration,
        calibration_registry=args.calibration_registry,
    )
    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"{base_name}_feature_overlay.mp4"
    )
    rendered = render_video(
        video_path=video_path,
        pose_path=pose_path,
        shuttle_path=shuttle_path,
        output_path=output_path,
        calibration=calibration,
        max_frames=args.max_frames,
        draw_projected_court=not args.skip_projected_court,
    )
    print(f"Rendered feature overlay to {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
