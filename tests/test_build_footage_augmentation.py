import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.court_projection import COURT_LANDMARKS, CourtHomography
from src.temporal_selector.rally_intervals import (
    RallyInterval,
    RallySourceManifest,
    read_rally_intervals,
    write_rally_intervals,
)
from src.workflow.augmentation import create_augmented_source
from src.workflow.registry import SourceCatalog
from tools.build_footage_augmentation import write_arm_annotations, write_arm_config


def _write_video(path: Path, label: str, *, frames: int = 24) -> None:
    size = (160, 120)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((size[1], size[0], 3), index * 8, dtype=np.uint8)
        cv2.putText(
            frame, label, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        writer.write(frame)
    writer.release()


def _calibration() -> dict:
    image = np.array([[10.0, 110.0], [150.0, 110.0], [120.0, 20.0], [40.0, 20.0]])
    court = np.array(
        [
            COURT_LANDMARKS["near_left_doubles"],
            COURT_LANDMARKS["near_right_doubles"],
            COURT_LANDMARKS["far_right_doubles"],
            COURT_LANDMARKS["far_left_doubles"],
        ]
    )
    calibration, _ = CourtHomography.from_points(image, court)
    return calibration.to_dict(image_size=(160, 120))


def _fixture(tmp_path: Path):
    catalog = SourceCatalog(tmp_path / "catalog.json")
    originals = []
    intervals = []
    manifests = []
    for source_id in ("alpha", "beta"):
        video = tmp_path / f"{source_id}.mp4"
        _write_video(video, source_id)
        calibration = tmp_path / f"{source_id}-calibration.json"
        calibration.write_text(json.dumps(_calibration()))
        originals.append(
            {
                "source_id": source_id,
                "video_path": str(video),
                "calibration_path": str(calibration),
                "assignments_path": str(tmp_path / f"{source_id}-assignments.jsonl"),
                "pose_cache_path": str(tmp_path / f"{source_id}-poses.jsonl"),
            }
        )
        intervals.append(
            RallyInterval(source_id, f"{source_id}-0001", 0 if source_id == "alpha" else 5, 15)
        )
        manifests.append(
            RallySourceManifest(
                source_id,
                hashlib.sha256(video.read_bytes()).hexdigest(),
                30.0,
                24,
            )
        )
    intervals_path = tmp_path / "rallies.csv"
    manifest_path = tmp_path / "rallies.manifest.json"
    write_rally_intervals(
        intervals_path,
        manifest_path,
        intervals,
        manifests,
        annotation_revision="test",
        partial_start_rally_ids=["alpha-0001"],
    )
    variants = {}
    for row in originals:
        entry = create_augmented_source(
            catalog,
            parent_source_id=row["source_id"],
            parent_video=Path(row["video_path"]),
            parent_calibration=Path(row["calibration_path"]),
            transform="horizontal-flip",
            output_root=tmp_path / "variants",
        )
        variants[entry.source_id] = entry
    return catalog, originals, variants, intervals_path, manifest_path


def test_write_arm_annotations_copies_and_renames_parent_rallies(tmp_path: Path) -> None:
    catalog, originals, variants, intervals_path, manifest_path = _fixture(tmp_path)
    variant_parents = {
        source_id: row["source_id"]
        for source_id, row in zip(variants, originals)
    }
    output = tmp_path / "arm"
    arm_intervals, arm_manifest = write_arm_annotations(
        transform="horizontal-flip",
        parent_ids=[row["source_id"] for row in originals],
        variant_parents=variant_parents,
        catalog=catalog,
        base_intervals_path=intervals_path,
        base_manifest_path=manifest_path,
        output_dir=output,
    )

    index = read_rally_intervals(arm_intervals, arm_manifest)
    alpha_flip = "alpha__hflip"
    assert set(index.sources) == {"alpha", "beta", alpha_flip, "beta__hflip"}
    variant_intervals = index.for_source(alpha_flip)
    assert len(variant_intervals) == 1
    assert variant_intervals[0].rally_id == f"{alpha_flip}-0001"
    assert (variant_intervals[0].start_frame, variant_intervals[0].end_frame) == (0, 15)
    assert variant_intervals[0].rally_id in index.partial_start_rally_ids
    assert index.is_reviewed(alpha_flip, 0)
    assert index.is_reviewed(alpha_flip, 23)


def test_write_arm_config_lists_originals_and_variants(tmp_path: Path) -> None:
    catalog, originals, variants, intervals_path, manifest_path = _fixture(tmp_path)
    variant_parents = {
        source_id: row["source_id"] for source_id, row in zip(variants, originals)
    }
    base_config = {
        "seed": 1729,
        "camera_sources": {"Camera A": ["alpha"], "Camera B": ["beta"]},
        "dataset": {"pose_visibility_threshold": 0.5},
    }
    config_path = write_arm_config(
        transform="horizontal-flip",
        base_config=base_config,
        base_dir=tmp_path,
        originals=originals,
        variants=variants,
        variant_parents=variant_parents,
        intervals_path=intervals_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "arm",
    )
    config = json.loads(config_path.read_text())
    source_ids = [row["source_id"] for row in config["dataset"]["sources"]]
    assert source_ids[:2] == ["alpha", "beta"]
    assert set(source_ids[2:]) == {alpha_flip for alpha_flip in variants}
    assert config["variant_parents"] == variant_parents
    assert config["camera_sources"] == {
        "Camera A": ["alpha"],
        "Camera B": ["beta"],
    }
    assert config["dataset"]["rally_feature_view"] == "full_motion"
