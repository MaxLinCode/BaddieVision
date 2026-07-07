import json
from pathlib import Path

import cv2
import numpy as np

from src.court_features import Calibration
from src.court_projection import CourtHomography
from src.visualize_features import (
    annotate_frame,
    build_feature_stream,
    draw_court_minimap,
    resolve_artifact_path,
)


def _landmark(x, y, visibility=1.0):
    return {"x": x, "y": y, "visibility": visibility}


def _pose_frames(count: int) -> list[dict]:
    frames = []
    for index in range(count):
        keypoints = {
            "0": _landmark(0.5, 0.1),
            "11": _landmark(0.4, 0.4),
            "12": _landmark(0.6, 0.4),
            "23": _landmark(0.45, 0.7),
            "24": _landmark(0.55, 0.7),
            "29": _landmark(0.45 + index * 0.01, 0.9),
            "30": _landmark(0.55 + index * 0.01, 0.9),
            "31": _landmark(0.45 + index * 0.01, 0.92),
            "32": _landmark(0.55 + index * 0.01, 0.92),
        }
        frames.append({"frame": index, "keypoints": keypoints})
    return frames


def _write_pose_json(path: Path, frames: list[dict]) -> None:
    path.write_text(json.dumps(frames), encoding="utf-8")


def _write_shuttle_csv(path: Path, rows: list[tuple[int, int, int, int]]) -> None:
    path.write_text(
        "Frame,X,Y,Visibility\n"
        + "\n".join(f"{frame},{x},{y},{visibility}" for frame, x, y, visibility in rows)
        + "\n",
        encoding="utf-8",
    )


def identity_calibration() -> Calibration:
    return Calibration(CourtHomography(np.eye(3)), (100, 100))


def test_resolve_artifact_path_matches_case_insensitive(tmp_path):
    target = tmp_path / "Smash_IMG_3214_ball.csv"
    target.write_text("Frame,X,Y,Visibility\n", encoding="utf-8")
    resolved = resolve_artifact_path(None, tmp_path, "smash_img_3214_ball.csv", "shuttle CSV")
    assert resolved == target


def test_build_feature_stream_computes_shuttle_and_court_features(tmp_path):
    frames = _pose_frames(4)
    pose_path = tmp_path / "clip_pose.json"
    shuttle_path = tmp_path / "clip_ball.csv"
    _write_pose_json(pose_path, frames)
    _write_shuttle_csv(
        shuttle_path,
        [
            (0, 50, 20, 1),
            (1, 60, 30, 1),
            (2, 70, 40, 1),
            (3, 0, 0, 0),
        ],
    )

    stream = build_feature_stream(
        pose_path=pose_path,
        shuttle_path=shuttle_path,
        frame_width=100,
        frame_height=100,
        calibration=identity_calibration(),
    )

    assert len(stream.pose_frames) == 4
    assert stream.shuttle.shape == (4, 7)
    assert stream.court is not None
    assert stream.court.shape == (4, 3)
    np.testing.assert_allclose(stream.shuttle[1, :5], [0.6, 0.3, 1.0, 10.0, 10.0], atol=1e-5)
    assert stream.court[0, 2] == 1.0


def test_annotate_frame_draws_overlay_content():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    pose_frame = _pose_frames(1)[0]
    shuttle_features = np.asarray([0.5, 0.25, 1.0, 1.5, -2.0, 0.1, 0.2], dtype=np.float32)
    court_features = np.asarray([0.1, -0.2, 1.0], dtype=np.float32)

    annotated = annotate_frame(
        frame,
        frame_index=0,
        pose_frame=pose_frame,
        shuttle_features=shuttle_features,
        court_features=court_features,
        draw_projected_court=False,
        calibration=None,
        trail=[(100, 80), (110, 70)],
    )

    assert annotated.shape == frame.shape
    assert np.count_nonzero(cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)) > 0


def test_draw_court_minimap_marks_anchor():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    draw_court_minimap(frame, anchor=np.asarray([0.0, 0.0]), observed=True)
    assert np.count_nonzero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 0
