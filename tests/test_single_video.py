import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.single_video import (
    build_ffmpeg_command,
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


def test_prepare_working_video_noop_returns_original(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _write_video(source, fps=30, frames=10)

    result = prepare_working_video(source, tmp_path / "unused.mp4", None, None, None, "sdr")

    assert result != source
    assert (tmp_path / "unused.mp4").exists()


@pytest.mark.parametrize(
    ("start", "end", "fps", "message"),
    [(-1, None, None, "start_time_sec"), (None, -1, None, "end_time_sec"), (None, None, 0, "target_fps"), (2, 1, 30, "Invalid segment")],
)
def test_prepare_working_video_validates_options(
    tmp_path: Path, start: int | None, end: int | None, fps: int | None, message: str
) -> None:
    source = tmp_path / "source.mp4"
    _write_video(source, fps=30, frames=30)
    with pytest.raises(ValueError, match=message):
        prepare_working_video(source, tmp_path / "output.mp4", start, end, fps, "sdr")


def test_gpu_ffmpeg_command_uses_nvdec_nvenc_and_cfr(tmp_path: Path) -> None:
    command = build_ffmpeg_command(
        "/usr/bin/ffmpeg",
        tmp_path / "input.mp4",
        tmp_path / "output.mp4",
        source_fps=60,
        effective_fps=30,
        start_frame=60,
        end_frame=180,
        expected_frames=60,
        gpu=True,
    )
    joined = " ".join(command)
    assert "-hwaccel cuda -hwaccel_output_format cuda" in joined
    assert "-vf scale_cuda=format=nv12" in joined
    assert "-c:v h264_nvenc -preset p1 -rc vbr -cq 20" in joined
    assert "-an" in command
    assert command[command.index("-r") + 1] == "30.000000000"
    assert command[command.index("-frames:v") + 1] == "60"


def test_color_managed_ingest_uses_cpu_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _write_video(source, fps=60, frames=60)
    commands: list[list[str]] = []

    real_which = video_helpers.shutil.which
    monkeypatch.setattr(video_helpers.shutil, "which", lambda name: real_which(name))

    def fake_run(command):
        commands.append(list(command))
        _write_video(Path(command[-1]), fps=30, frames=30)

    monkeypatch.setattr(video_helpers, "_run_ffmpeg", fake_run)
    result = prepare_working_video(source, output, None, None, 30, "sdr")

    assert result == output
    assert len(commands) == 1
    assert "libx264" in commands[0]
    assert "zscale=" in " ".join(commands[0])


def test_color_managed_ffmpeg_failure_does_not_fall_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _write_video(source, fps=60, frames=60)
    real_which = video_helpers.shutil.which
    monkeypatch.setattr(video_helpers.shutil, "which", lambda name: real_which(name))
    monkeypatch.setattr(video_helpers, "_run_ffmpeg", lambda _: (_ for _ in ()).throw(RuntimeError("failed")))

    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError, match="Every video preparation backend failed"):
        prepare_working_video(source, output, None, None, 30, "sdr")


@pytest.mark.parametrize(("source_fps", "target_fps", "frames", "expected"), [(60.0, 30, 60, 30), (59.94, 24, 60, 24)])
def test_missing_color_tools_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_fps: float,
    target_fps: int,
    frames: int,
    expected: int,
) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _write_video(source, fps=source_fps, frames=frames)
    monkeypatch.setattr(video_helpers.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="ffprobe is required"):
        prepare_working_video(source, output, None, None, target_fps, "sdr")
    assert not output.exists()


def test_failed_backends_clean_temporary_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _write_video(source, fps=30, frames=10)
    monkeypatch.setattr(video_helpers.shutil, "which", lambda _: None)
    monkeypatch.setattr(video_helpers, "_prepare_with_opencv", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("decode failed")))

    with pytest.raises(RuntimeError, match="ffprobe is required"):
        prepare_working_video(source, output, None, 0.2, 15, "sdr")

    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp.mp4"))


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
    tracks = result_dir / "tracks.csv"
    tracks.write_text("Frame,X,Y,Visibility\n", encoding="utf-8")
    metadata = result_dir / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    manifest_path = result_dir / "artifact_manifest.json"

    manifest = write_result_manifest(manifest_path, result_dir, "sample", [tracks, metadata])
    assert [item["relative_path"] for item in manifest["files"]] == ["tracks.csv", "metadata.json"]
    assert json.loads(manifest_path.read_text())["source_id"] == "sample"

    zip_path = result_dir / "sample.zip"
    zip_result_dir(result_dir, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == [
            "sample/artifact_manifest.json",
            "sample/metadata.json",
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
    assert metadata["working_video"]["target_fps"] is None
