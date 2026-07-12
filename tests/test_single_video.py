import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.single_video import (
    copy_model_tree,
    load_track_rows,
    prepare_working_video,
    read_video_info,
    render_preview,
    validate_model_root,
    validate_prepared_video,
    write_metadata,
    write_result_manifest,
    zip_result_dir,
)
from src.single_video import video as video_helpers


def _write_video(path: Path, *, fps: float = 60.0, frames: int = 60, size: tuple[int, int] = (64, 48)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((size[1], size[0], 3), index % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_prepare_working_video_untrimmed_is_byte_identical(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _write_video(source, fps=30, frames=10)

    output = tmp_path / "working.mp4"
    result = prepare_working_video(source, output, None, None, 30)

    assert result == output
    assert output.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("suffix", [".mov", ".avi", ".mkv"])
def test_prepare_working_video_rejects_non_mp4_before_reading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suffix: str
) -> None:
    source = tmp_path / f"source{suffix}"
    source.write_bytes(b"not a video")
    monkeypatch.setattr(video_helpers, "read_video_info", lambda _: pytest.fail("input was opened"))
    with pytest.raises(ValueError, match="must be an MP4"):
        prepare_working_video(source, tmp_path / "output.mp4", None, None)


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [(-1, None, "start_time_sec"), (None, -1, "end_time_sec"), (2, 1, "Invalid or empty"), (2, 2, "Invalid or empty"), (2, None, "Invalid or empty")],
)
def test_prepare_working_video_validates_options(
    tmp_path: Path, start: int | None, end: int | None, message: str
) -> None:
    source = tmp_path / "source.mp4"
    _write_video(source, fps=30, frames=30)
    with pytest.raises(ValueError, match=message):
        prepare_working_video(source, tmp_path / "output.mp4", start, end)


@pytest.mark.parametrize(("start", "end"), [(1, None), (None, 2), (1, 2)])
def test_trim_uses_ffmpeg_stream_copy_without_video_conversion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, start: int | None, end: int | None
) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _write_video(source, fps=30, frames=90)
    commands: list[list[str]] = []
    monkeypatch.setattr(video_helpers.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def fake_run(command, **kwargs):
        commands.append(list(command))
        _write_video(Path(command[-1]), fps=30, frames=30)
        return video_helpers.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(video_helpers.subprocess, "run", fake_run)
    assert prepare_working_video(source, output, start, end) == output
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("-c") + 1] == "copy"
    forbidden = {"-c:v", "-vf", "-filter:v", "-r", "-pix_fmt", "libx264", "h264_nvenc"}
    assert forbidden.isdisjoint(command)


def test_missing_ffmpeg_fails_only_for_trim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _write_video(source, fps=30, frames=10)
    monkeypatch.setattr(video_helpers.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="FFmpeg is required when trimming"):
        prepare_working_video(source, output, None, 0.2)

    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp.mp4"))

    assert prepare_working_video(source, output, None, None) == output


def test_fps_conversion_prefers_fast_nvenc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _write_video(source, fps=60, frames=60)
    commands = []
    monkeypatch.setattr(video_helpers.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def fake_run(command, **kwargs):
        commands.append(list(command))
        _write_video(Path(command[-1]), fps=30, frames=30)
        return video_helpers.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(video_helpers, "_run_ffmpeg_with_progress", fake_run)
    assert prepare_working_video(source, output, None, None, 30) == output
    assert len(commands) == 1
    assert commands[0][commands[0].index("-vf") + 1] == "fps=30.000000000"
    assert "h264_nvenc" in commands[0]
    assert "-preset" in commands[0] and "p1" in commands[0]
    assert commands[0][commands[0].index("-g") + 1] == "60"


def test_fps_conversion_falls_back_to_ultrafast_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _write_video(source, fps=60, frames=60)
    commands = []
    monkeypatch.setattr(video_helpers.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def fake_run(command, **kwargs):
        commands.append(list(command))
        if "h264_nvenc" in command:
            return video_helpers.subprocess.CompletedProcess(command, 1, "", "NVENC unavailable")
        _write_video(Path(command[-1]), fps=30, frames=30)
        return video_helpers.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(video_helpers, "_run_ffmpeg_with_progress", fake_run)
    assert prepare_working_video(source, output, None, None, 30) == output
    assert len(commands) == 2
    assert "libx264" in commands[1]
    assert commands[1][commands[1].index("-preset") + 1] == "ultrafast"


def test_validate_prepared_video_reports_mismatch(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_video(video, fps=30, frames=5, size=(64, 48))
    with pytest.raises(RuntimeError, match="dimensions.*frame count"):
        validate_prepared_video(video, width=32, height=24, fps=30, frame_count=4)


def test_validate_prepared_video_accepts_bounded_container_frame_drift(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_video(video, fps=30, frames=98, size=(64, 48))

    info = validate_prepared_video(
        video, width=64, height=48, fps=30, frame_count=100, frame_count_tolerance=2
    )

    assert info["frame_count"] == 98


def test_model_tree_copy_and_validation(tmp_path: Path) -> None:
    source, target = tmp_path / "bundle", tmp_path / "repo"
    for relative in (
        "models/TrackNet_torchscript.pt",
        "models/InpaintNet_torchscript.pt",
        "InPlay/models/player.pt",
        "src/TrackNetV3/ckpts/TrackNet_best.pt",
        "src/TrackNetV3/ckpts/InpaintNet_best.pt",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")

    assert validate_model_root(source) == source
    copy_model_tree(source, target)
    assert (target / "InPlay/models/player.pt").read_bytes() == b"model"
    assert validate_model_root(target) == target


def test_manifest_and_zip_keep_existing_layout(tmp_path: Path) -> None:
    result_dir = tmp_path / "sample"
    result_dir.mkdir()
    working_video = result_dir / "sample_input.mp4"
    working_video.write_bytes(b"portable source video")
    tracks = result_dir / "tracks.csv"
    tracks.write_text("Frame,X,Y,Visibility\n", encoding="utf-8")
    metadata = result_dir / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    manifest_path = result_dir / "artifact_manifest.json"

    manifest = write_result_manifest(manifest_path, result_dir, "sample", [working_video, tracks, metadata])
    assert [item["relative_path"] for item in manifest["files"]] == ["sample_input.mp4", "tracks.csv", "metadata.json"]
    assert json.loads(manifest_path.read_text())["source_id"] == "sample"

    zip_path = result_dir / "sample.zip"
    zip_result_dir(result_dir, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == [
            "sample/artifact_manifest.json",
            "sample/metadata.json",
            "sample/sample_input.mp4",
            "sample/tracks.csv",
        ]


def test_preview_track_loading_and_metadata_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import torch

    video = tmp_path / "input.mp4"
    preview = tmp_path / "preview.mp4"
    tracks = tmp_path / "tracks.csv"
    poses = tmp_path / "player_poses.jsonl"
    metadata_path = tmp_path / "metadata.json"
    _write_video(video, fps=30, frames=3)
    tracks.write_text(
        "Frame,X,Y,Visibility\n0,10,10,1\n1,12,12,0\n2,14,14,1\n",
        encoding="utf-8",
    )
    poses.write_text(
        "\n".join(
            [json.dumps({"type": "metadata"})]
            + [json.dumps({"type": "frame", "frame": index, "detections": []}) for index in range(3)]
        )
        + "\n",
        encoding="utf-8",
    )

    assert sorted(load_track_rows(tracks)) == [0, 1, 2]
    render_preview(video, tracks, poses, preview, ())
    assert read_video_info(preview)["frame_count"] == 3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = read_video_info(video)
    models = {"tracknet_checkpoint": "/models/track.pt", "inpaintnet_checkpoint": "/models/inpaint.pt"}
    pose_backend = {"name": "automatic-test-backend"}
    write_metadata(
        metadata_path,
        "sample",
        video,
        video,
        info,
        info,
        models,
        None,
        None,
        None,
        player_detector="yolov8n.pt",
        pose_backend=pose_backend,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_info"] == {
        "tracknet_checkpoint": "/models/track.pt",
        "inpaintnet_checkpoint": "/models/inpaint.pt",
        "player_detector": "yolov8n.pt",
        "player_detector_class_id": 0,
        "player_detector_class_name": "person",
        "pose_backend": pose_backend,
    }
    assert metadata["segment"] == {"start_time_sec": None, "end_time_sec": None, "is_clipped": False}
    assert metadata["working_video"]["input_handling_mode"] == "byte-copy"
    assert metadata["working_video"]["stream_copy"] is False
    assert "color_management" not in metadata
