import json
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from src.temporal_selector import rally_partitions as partitions
from src.temporal_selector.rally_audio import RALLY_AUDIO_FEATURE_NAMES, read_rally_audio
from src.temporal_selector.rally_dataset import RallyDataConfig, RallySourceConfig
from src.temporal_selector.rally_evaluation import rally_metrics, tune_rally_decoder
from src.temporal_selector.rally_intervals import (
    RallyCoverageSpan,
    RallyInterval,
    RallyIntervalIndex,
    RallySourceManifest,
)
from src.workflow.identity import inspect_source, sha256_file
from src.workflow.registry import SourceCatalog
from src.workflow.source import resolve_rally_30fps_source


def _index(counts, intervals=(), fps=30):
    return RallyIntervalIndex(
        intervals,
        {source: RallySourceManifest(source, "a" * 64, fps, count) for source, count in counts.items()},
        "test", "test",
        tuple(RallyCoverageSpan(source, 0, count - 1) for source, count in counts.items()),
    )


def test_plan_prefers_whole_recordings_then_nearby_reviewed_break():
    groups = {"whole": ["one", "two"], "cut": ["three"], "partial": ["four"]}
    index = _index(
        {"one": 80, "two": 20, "three": 300, "four": 300},
        (RallyInterval("three", "short", 225, 249), RallyInterval("four", "long", 169, 299)),
    )
    plan = partitions.plan_rally_camera_splits(index, groups)
    assert plan["whole"].train == (partitions.RallySourceRange("one", 0, 80),)
    assert plan["whole"].validation == (partitions.RallySourceRange("two", 0, 20),)
    assert plan["cut"].cut == 251
    assert plan["partial"].train == (partitions.RallySourceRange("four", 0, 240),)
    assert plan["partial"].validation == (partitions.RallySourceRange("four", 240, 300),)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="media tools required")
def test_partition_stages_receive_disjoint_clips_and_keep_original_frame_keys(tmp_path, monkeypatch):
    raw = tmp_path / "raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    assert writer.isOpened()
    for frame in range(60):
        writer.write(np.full((48, 64, 3), 32 if frame < 30 else 192, np.uint8))
    writer.release()
    video = tmp_path / "parent.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(raw), "-itsoffset", "1.2",
        "-f", "lavfi", "-i", "sine=duration=0.4", "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "alac", str(video),
    ], check=True)
    parent = inspect_source("parent", video)
    parent.write(tmp_path / "source.json")
    calibration = tmp_path / "court" / "calibration.json"
    calibration.parent.mkdir()
    calibration.write_text(json.dumps({
        "version": 1, "image_size": [64, 48],
        "image_to_court": [[.05, 0, -1.6], [0, .1, -2.4], [0, 0, 1]],
    }))
    source = RallySourceConfig(
        "parent", video, tmp_path / "assignments.jsonl", tmp_path / "poses.jsonl", calibration,
    )
    source.pose_cache_path.write_text('{"type":"metadata"}\n')
    index = _index({"parent": 60}, (RallyInterval("parent", "cut", 27, 32),))
    detector_inputs = []
    pose_inputs = []

    def person_stage(entry, **kwargs):
        cap = cv2.VideoCapture(str(entry.video_path))
        pixels = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                pixels.append(float(frame.mean()))
        finally:
            cap.release()
        detector_inputs.append((entry.video_path, pixels))

    def pose_stage(entry, **kwargs):
        identity = inspect_source(entry.source_id, entry.video_path)
        pose_inputs.append(entry.video_path)
        lineage = "sha256:" + "a" * 64
        from src.workflow.identity import SourceIdentity

        stored = SourceIdentity.read(entry.identity_path)
        assignments = [{
            "type": "metadata", "fps": 30, "frame_count": identity.frame_count, "frame_size": [64, 48],
            "source_id": entry.source_id, "source_identity_sha256": stored.fingerprint,
            "calibration_sha256": sha256_file(entry.calibration_path), "raw_artifact_fingerprint": lineage,
        }] + [
            {"type": "frame", "frame": frame, "slots": {}} for frame in range(identity.frame_count)
        ]
        entry.assignments_path.parent.mkdir(parents=True, exist_ok=True)
        entry.assignments_path.write_text("".join(json.dumps(row) + "\n" for row in assignments))
        entry.pose_cache_path.write_text(json.dumps({
            "type": "metadata", "raw_artifact_fingerprint": lineage, "pose_model_fingerprint": lineage,
            "preprocessing_fingerprint": lineage,
        }) + "\n")

    monkeypatch.setattr(partitions, "run_person_stage", person_stage)
    monkeypatch.setattr(partitions, "run_player_stage", pose_stage)
    config = RallyDataConfig((source,), None, None, feature_view="full_audio_zscore")
    detector = tmp_path / "fixture.pt"
    detector.write_bytes(b"test detector")
    source.assignments_path.with_name("person-tracks.manifest.json").write_text(
        json.dumps({"inputs": {"detector_sha256": sha256_file(detector)}}, indent=2)
    )
    train, validation = [
        partitions.prepare_rally_partition(
            config, source, partitions.RallySourceRange("parent", start, stop), index,
            tmp_path / "cache", device="cpu", person_model=str(detector),
        )[0][0]
        for start, stop in ((0, 30), (30, 60))
    ]
    assert all(len(pixels) == 30 for _, pixels in detector_inputs)
    assert max(detector_inputs[0][1]) < 40
    assert min(detector_inputs[1][1]) > 180
    assert pose_inputs == [path for path, _ in detector_inputs]
    assert train.frame_indices == tuple(range(30))
    assert validation.frame_indices == tuple(range(30, 60))
    assert train.inplay_targets[-3:].tolist() == [1, 1, 1]
    assert validation.inplay_targets[:3].tolist() == [1, 1, 1]
    assert train.rally_start_targets[27] == 1
    assert train.rally_end_targets.max() == 0
    assert validation.rally_start_targets.max() == 0
    assert validation.rally_end_targets[2] == 1

    train_audio = read_rally_audio(pose_inputs[0], frame_count=30, fps=30)
    validation_audio = read_rally_audio(pose_inputs[1], frame_count=30, fps=30)
    assert not train_audio.validity.any()
    assert np.flatnonzero(validation_audio.validity[:, 0]).tolist() == list(range(7, 17))
    audio_dim = validation_audio.values.shape[1]
    np.testing.assert_array_equal(validation.frame_validity[:, -audio_dim:], validation_audio.validity)
    np.testing.assert_array_equal(validation.frame_values[:, -audio_dim:], validation_audio.zscore())

    catalog = SourceCatalog(tmp_path / "catalog.json")
    with pytest.raises(ValueError, match="below 30 fps"):
        low_rate = tmp_path / "slow.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(raw), "-vf", "fps=24", str(low_rate),
        ], check=True)
        resolve_rally_30fps_source(catalog, source_id=None, video=low_rate, output_dir=tmp_path / "slow")
    assert not catalog.sources


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="media tools required")
def test_validation_audio_normalization_excludes_training_recording(tmp_path):
    full_audio = []
    validation_audio = []
    for training_gain in (0.1, 3):
        parent_video = tmp_path / f"parent-{training_gain}.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error",
            "-f", "lavfi", "-i", "color=size=64x48:rate=30:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=2",
            "-af", f"asetnsamples=n=480:p=0,volume='if(lt(t,1),{training_gain},if(lt(t,1.5),0.5,1))':eval=frame",
            "-c:v", "mpeg4", "-c:a", "alac", str(parent_video),
        ], check=True, capture_output=True)
        source = RallySourceConfig(
            "parent", parent_video, tmp_path / "assignments", tmp_path / "poses", tmp_path / "calibration",
        )
        parent = inspect_source(source.source_id, parent_video)
        output = tmp_path / f"validation-{training_gain}" / "clip.mkv"
        partitions._write_partition_video(
            source, partitions.RallySourceRange(source.source_id, 30, 60), parent, output,
        )
        full = read_rally_audio(parent_video, frame_count=60, fps=30)
        clip = read_rally_audio(output, frame_count=30, fps=30)
        full_audio.append(full.zscore())
        validation_audio.append((clip.zscore(), clip.validity))

    # Normalizing the full recording would carry the earlier gain into validation.
    assert abs(full_audio[0][35, 0] - full_audio[1][35, 0]) > 1
    first, second = validation_audio
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1][1, 0] and first[0][1, 0] == 0
    assert first[0][20, 0] > 0


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="media tools required")
def test_partition_preserves_audio_gap_frame_alignment(tmp_path):
    video = tmp_path / "gap.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error",
        "-f", "lavfi", "-i", "color=size=64x48:rate=30:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=3",
        "-af", "asetnsamples=n=1024:p=0,aselect='not(between(t,1.02,1.53))'",
        "-c:v", "mpeg4", "-c:a", "aac", str(video),
    ], check=True, capture_output=True)
    source = RallySourceConfig(
        "parent", video, tmp_path / "assignments", tmp_path / "poses", tmp_path / "calibration",
    )
    first, stop = 16, 80
    clip = tmp_path / "partition" / "clip.mkv"
    partitions._write_partition_video(
        source, partitions.RallySourceRange(source.source_id, first, stop),
        inspect_source(source.source_id, video), clip,
    )
    parent = read_rally_audio(video, frame_count=90, fps=30)
    partition = read_rally_audio(clip, frame_count=stop - first, fps=30)
    # Edge windows can lose coverage when cut. The interior must keep its frame alignment.
    expected = parent.validity[first + 5:stop - 5]
    assert not expected.all()
    assert expected.any()
    np.testing.assert_array_equal(partition.validity[5:-5], expected)


def test_short_dip_policy_fixes_bridge_duration_during_calibration():
    rows = [
        {
            "source_id": "source",
            "frame": frame,
            "inplay_target": int(frame in (1, 2, 6, 7)),
            "inplay_probability": 0.99 if frame in (1, 2, 6, 7) else 0.01,
        }
        for frame in range(9)
    ]
    fps = {"source": 10}
    ordinary, _ = tune_rally_decoder(rows, fps_by_source=fps, hysteresis=True)
    bridged, search = tune_rally_decoder(
        rows, fps_by_source=fps, hysteresis=True, bridge_short_dips=True,
    )

    assert ordinary.max_gap_seconds == 0.0
    assert bridged.max_gap_seconds == 0.5
    assert {candidate["configuration"]["max_gap_seconds"] for candidate in search} == {0.5}


def test_complete_interval_metrics_skip_cut_rally_but_keep_its_frames():
    rows = [
        {
            "source_id": "camera", "frame": frame, "inplay_target": int(state),
            "inplay_probability": float(state), "decoded_inplay": state,
            **({"partial_rally": (False, True)} if frame < 2 else {}),
        }
        for frame, state in enumerate((True, True, False, False, True, True, False))
    ]
    metrics = rally_metrics(rows)
    assert metrics["frame_count"] == 7
    assert metrics["decoded_frame"]["true_positive"] == 4
    assert len(metrics["interval_matches"]) == 1
    assert metrics["intervals"]["f1"] == 1
    # A one-frame error on the visible side of the cut rally is counted,
    # while its artificial start is omitted from boundary scoring.
    rows[2]["decoded_inplay"] = True
    metrics = rally_metrics(rows)
    assert metrics["intervals"]["label_count"] == 1
    assert metrics["visible_boundary_mae"] == pytest.approx(1 / 3)


def test_clean_training_uses_partition_windows_and_full_held_out_recordings(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from src.temporal_selector import rally_experiment as training
    from src.temporal_selector.rally import RallyTrainingRecipe
    from src.temporal_selector.rally_dataset import RallyWindow, collate_rally_windows
    from src.temporal_selector.rally_features import RALLY_FEATURE_VIEWS

    counts = {"a": 100, "b": 100}
    index = _index(counts, fps=5)
    paths = [tmp_path / name for name in ("video", "assignments", "poses", "calibration")]
    config = RallyDataConfig(
        tuple(RallySourceConfig(source, *paths) for source in counts),
        tmp_path / "rallies.csv", tmp_path / "rallies.json",
        feature_view="full_audio_zscore",
    )
    feature_view = RALLY_FEATURE_VIEWS[config.feature_view]

    def window(source, start, stop):
        frames = tuple(range(start, stop))
        targets = torch.zeros(len(frames), dtype=torch.long)
        values = torch.zeros(len(frames), feature_view.dimension)
        validity = torch.ones_like(values, dtype=torch.bool)
        validity[0, -len(RALLY_AUDIO_FEATURE_NAMES):] = False
        return RallyWindow(
            source, f"{source}-{start}-{stop}", start, frames, frozenset(frames),
            torch.zeros(len(frames)), values, validity,
            targets, targets.float(), targets.float(), {"fps": 5, "frame_count": counts[source]},
        )

    dataset = SimpleNamespace(
        rally_index=index,
        windows=[window(source, 0, count) for source, count in counts.items()],
        manifest={"dataset_fingerprint": "full", "sources": {
            source: {"calibration_sha256": source} for source in counts
        }},
        annotation_fingerprint="annotations",
    )
    monkeypatch.setattr(training, "read_rally_intervals", lambda *args: index)
    monkeypatch.setattr(training, "RallyWindowDataset", lambda *_: dataset)

    def prepare(_, source, block, *args, **kwargs):
        return (window(source.source_id, block.start, block.stop),), f"artifact-{source.source_id}-{block.start}"

    validations = []

    def fit(model, train, validation, **kwargs):
        assert len(train) == len(validation) == 1
        before, after = train[0], validation[0]
        assert before.source_id == after.source_id
        assert before.frame_indices == tuple(range(80))
        assert after.frame_indices == tuple(range(80, 100))
        assert model.config.audio_attention
        batch = collate_rally_windows([before, after])
        loss = model.losses(batch).total
        assert torch.isfinite(loss)
        loss.backward()
        validations.append(after.source_id)
        return [], 1

    held_out = []

    def predict(model, windows, *, partition, include_unlabeled, **kwargs):
        assert include_unlabeled
        item, = windows
        if partition == "test":
            assert item.frame_indices == tuple(range(100))
            held_out.append(item.source_id)
        return [
            {"source_id": item.source_id, "frame": frame, "inplay_target": 0, "inplay_probability": 0.01}
            for frame in item.frame_indices
        ]

    monkeypatch.setattr(training, "prepare_rally_partition", prepare)
    monkeypatch.setattr(training, "train_rally_segmenter", fit)
    monkeypatch.setattr(training, "train_rally_segmenter_fixed_epochs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(training, "predict_rally_windows", predict)
    output = tmp_path / "run"
    result = training.run_clean_rally_experiment(
        config, output, camera_sources={"A": ["a"], "B": ["b"]},
        recipe=RallyTrainingRecipe(epochs=1, device="cpu"),
    )
    assert validations == ["b", "a"]
    assert held_out == ["a", "b"]
    split = result["folds"][0]["split_manifest"]["train_validation"]
    assert split["strategy"] == "chronological-source-ranges"
    assert split["cameras"]["B"]["validation"] == [{"source_id": "b", "start": 80, "stop": 100}]
    configuration = json.loads((output / "resolved-config.json").read_text())
    assert configuration["partition_fingerprints"]["b:80:100"] == "artifact-b-80"
    checkpoint = torch.load(output / "production-rally-segmenter.pt", weights_only=True)
    assert checkpoint["rally_feature_view"] == "full_audio_zscore"
    assert checkpoint["rally_segmenter_config"]["audio_attention"]
