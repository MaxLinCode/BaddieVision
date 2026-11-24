import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.court_projection import COURT_LANDMARKS, CourtHomography
from src.single_video.video import read_video_info
from src.workflow.artifacts import calibration_fingerprint
from src.workflow.augmentation import (
    AUGMENTATION_TRANSFORMS,
    augmentation_provenance,
    create_augmented_source,
    identity_calibration,
    mirror_calibration,
    render_augmented_video,
    swap_left_right,
)
from src.workflow.identity import SourceIdentity, sha256_file
from src.workflow.registry import SourceCatalog


def _write_video(
    path: Path, *, fps: float = 30.0, frames: int = 12, size: tuple[int, int] = (160, 120)
) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((size[1], size[0], 3), index * 15, dtype=np.uint8)
        cv2.putText(
            frame, str(index), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        writer.write(frame)
    writer.release()


def _calibration_value(size: tuple[int, int] = (160, 120)) -> dict:
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
    return calibration.to_dict(
        image_landmarks={
            "near_left_doubles": (10.0, 110.0),
            "near_right_doubles": (150.0, 110.0),
            "far_right_doubles": (120.0, 20.0),
            "far_left_doubles": (40.0, 20.0),
        },
        image_size=size,
        image_lines={
            "near_baseline": [(10.0, 110.0), (150.0, 110.0)],
            "left_doubles_sideline": [(10.0, 110.0), (40.0, 20.0)],
            "right_doubles_sideline": [(150.0, 110.0), (120.0, 20.0)],
            "far_baseline": [(40.0, 20.0), (120.0, 20.0)],
        },
    )


def test_swap_left_right_handles_compounds() -> None:
    assert (
        swap_left_right("left_doubles_sideline__near_short_service")
        == "right_doubles_sideline__near_short_service"
    )
    assert swap_left_right("near_left_doubles") == "near_right_doubles"
    assert (
        swap_left_right("near_doubles_long_service_center")
        == "near_doubles_long_service_center"
    )


def test_mirror_negates_court_x_and_preserves_y() -> None:
    size = (160, 120)
    value = _calibration_value(size)
    mirrored = mirror_calibration(value, size[0])
    original_h = CourtHomography(np.asarray(value["image_to_court"]))
    mirrored_h = CourtHomography(np.asarray(mirrored["image_to_court"]))
    points = np.array([[30.0, 100.0], [130.0, 90.0], [80.0, 50.0]])
    court = original_h.project_to_court(points)
    mirrored_points = np.column_stack([size[0] - 1 - points[:, 0], points[:, 1]])
    mirrored_court = mirrored_h.project_to_court(mirrored_points)
    np.testing.assert_allclose(
        mirrored_court, np.column_stack([-court[:, 0], court[:, 1]]), atol=1e-9
    )


def test_mirror_swaps_labels_and_round_trips() -> None:
    size = (160, 120)
    value = _calibration_value(size)
    mirrored = mirror_calibration(value, size[0])

    assert set(mirrored["image_landmarks"]) == {
        "near_right_doubles",
        "near_left_doubles",
        "far_left_doubles",
        "far_right_doubles",
    }
    assert mirrored["image_landmarks"]["near_right_doubles"] == pytest.approx(
        [size[0] - 1 - 10.0, 110.0]
    )
    assert set(mirrored["image_lines"]) == {
        "near_baseline",
        "right_doubles_sideline",
        "left_doubles_sideline",
        "far_baseline",
    }
    assert mirrored["image_size"] == list(size)
    assert "artifact_metadata" not in mirrored

    round_trip = mirror_calibration(mirrored, size[0])
    np.testing.assert_allclose(
        np.asarray(round_trip["image_to_court"]),
        np.asarray(value["image_to_court"]),
        atol=1e-12,
    )
    assert round_trip["image_landmarks"] == value["image_landmarks"]


def test_mirror_rejects_unknown_labels() -> None:
    value = _calibration_value()
    value["image_landmarks"] = {"mystery_line": [1.0, 2.0]}
    with pytest.raises(ValueError, match="not a known court name"):
        mirror_calibration(value, 160)


def test_identity_calibration_drops_source_metadata() -> None:
    value = _calibration_value()
    value["artifact_metadata"] = {"source_id": "old"}
    copied = identity_calibration(value)
    assert "artifact_metadata" not in copied
    assert copied["image_to_court"] == value["image_to_court"]
    assert value["artifact_metadata"] == {"source_id": "old"}


@pytest.mark.parametrize("name", sorted(AUGMENTATION_TRANSFORMS))
def test_render_preserves_geometry(tmp_path: Path, name: str) -> None:
    source = tmp_path / "source.mp4"
    _write_video(source, fps=30.0, frames=12)
    output = tmp_path / f"{name}.mp4"
    render_augmented_video(source, output, AUGMENTATION_TRANSFORMS[name])
    expected = read_video_info(source)
    actual = read_video_info(output)
    assert actual == expected


def test_create_augmented_source_is_isolated_and_registers(tmp_path: Path) -> None:
    parent_video = tmp_path / "parent.mp4"
    _write_video(parent_video)
    parent_calibration = tmp_path / "parent-calibration.json"
    parent_calibration.write_text(json.dumps(_calibration_value()))
    catalog = SourceCatalog(tmp_path / "catalog.json")
    parent_video_sha = sha256_file(parent_video)
    parent_calibration_sha = sha256_file(parent_calibration)

    entry = create_augmented_source(
        catalog,
        parent_source_id="parent",
        parent_video=parent_video,
        parent_calibration=parent_calibration,
        transform="horizontal-flip",
        output_root=tmp_path / "variants",
    )

    assert entry.source_id == "parent__hflip"
    assert sha256_file(parent_video) == parent_video_sha
    assert sha256_file(parent_calibration) == parent_calibration_sha
    identity = SourceIdentity.read(entry.identity_path)
    assert identity.frame_count == read_video_info(parent_video)["frame_count"]
    assert identity.width == 160 and identity.height == 120
    calibration = json.loads(entry.calibration_path.read_text())
    assert calibration["artifact_metadata"]["source_id"] == entry.source_id
    assert calibration["artifact_metadata"]["source_identity_sha256"] == identity.fingerprint
    assert calibration["artifact_metadata"]["augmentation"]["parent_source_id"] == "parent"
    assert calibration_fingerprint(entry)
    provenance = augmentation_provenance(entry)
    assert provenance["transform"] == "horizontal-flip"
    assert provenance["parent_video_sha256"] == parent_video_sha
    assert SourceCatalog.read(catalog.path).select(["parent__hflip"])[0] == entry

    again = create_augmented_source(
        catalog,
        parent_source_id="parent",
        parent_video=parent_video,
        parent_calibration=parent_calibration,
        transform="horizontal-flip",
        output_root=tmp_path / "variants",
    )
    assert again == entry


def test_create_augmented_source_rejects_conflicting_transform(tmp_path: Path) -> None:
    parent_video = tmp_path / "parent.mp4"
    _write_video(parent_video)
    parent_calibration = tmp_path / "parent-calibration.json"
    parent_calibration.write_text(json.dumps(_calibration_value()))
    catalog = SourceCatalog(tmp_path / "catalog.json")
    create_augmented_source(
        catalog,
        parent_source_id="parent",
        parent_video=parent_video,
        parent_calibration=parent_calibration,
        transform="horizontal-flip",
        output_root=tmp_path / "variants",
        source_id="shared",
    )
    with pytest.raises(ValueError, match="different augmentation"):
        create_augmented_source(
            catalog,
            parent_source_id="parent",
            parent_video=parent_video,
            parent_calibration=parent_calibration,
            transform="h264-low",
            output_root=tmp_path / "variants",
            source_id="shared",
        )
