"""Reproducible single-video inference using versioned, candidate-free assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from .artifacts import (
    calibration_fingerprint,
)
from .identity import SourceIdentity, sha256_file
from .stages import (
    implementation_identity,
    run_person_stage,
    run_player_stage,
    run_stage,
    stage_status,
)
from .registry import SourceCatalog
from .source import (
    calibrate_source as _calibrate,
    select_device as select_device,
    resolve_rally_30fps_source,
    resolve_source,
)


def run_pipeline(
    catalog: SourceCatalog,
    *,
    source_id: str | None = None,
    video: Path | None = None,
    output_dir: Path | None = None,
    calibration: Path | None = None,
    model_id: str = "rally-six-source-v1",
    checkpoint: Path | None = None,
    model_manifest: Path | None = None,
    cache_dir: Path | None = None,
    device: str = "auto",
    batch_size: int = 16,
    overlay: bool = True,
    bridge_short_dips: bool = False,
    force: bool = False,
    rebuild: Sequence[str] = (),
    person_backend_identity: str | None = None,
    pose_backend_identity: str | None = None,
    person_extractor: Callable[..., Any] | None = None,
    player_interpreter: Callable[..., Any] | None = None,
) -> dict[str, Path]:
    """Generate or verify observations, then infer using the frozen release contract."""
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch size must be positive")
    if (checkpoint is None) != (model_manifest is None):
        raise ValueError("--checkpoint and --model-manifest must be supplied together")
    resolved_device = select_device(device)

    from src.rally_release import (
        LEGACY_RELEASE_VERSION,
        release_tracking_profile,
        resolve_release,
    )
    from src.temporal_selector.rally_audio import rally_audio_runtime
    from src.temporal_selector.rally_dataset import RallySourceConfig
    from src.temporal_selector.rally_features import rally_feature_view
    from src.temporal_selector.rally_inference import (
        predict_source,
        write_prediction_artifacts,
    )

    model, manifest, assets = resolve_release(
        model_id, checkpoint=checkpoint, manifest=model_manifest, cache_dir=cache_dir
    )
    tracking_profile = release_tracking_profile(manifest)
    rtmpose = manifest["schema_version"] != LEGACY_RELEASE_VERSION
    source_resolver = resolve_rally_30fps_source if rtmpose else resolve_source
    source = source_resolver(
        catalog, source_id=source_id, video=video, output_dir=output_dir
    )
    identity = SourceIdentity.read(source.identity_path)
    _calibrate(source, calibration, force=force or "pose" in rebuild)
    print(
        f"{source.source_id}: verified model assets; using {resolved_device}",
        flush=True,
    )
    audio_runtime = (
        rally_audio_runtime()
        if rally_feature_view(manifest["rally_feature_view"]).uses_audio
        else None
    )
    preprocessing = {
        **manifest["preprocessing"],
        "feature_view": manifest["rally_feature_view"],
    }
    status_path = source.artifact_root / "run-status.json"
    previous = (
        json.loads(status_path.read_text()) if status_path.exists() else {"stages": {}}
    )
    for stage in ["person", "pose", "predict", "decode"] + (
        ["render"] if overlay else []
    ):
        if stage not in previous["stages"]:
            stage_status(source, stage, "pending")
    unknown = set(rebuild) - {"person", "pose", "predict", "decode", "render"}
    if unknown:
        raise ValueError(f"unknown rebuild stages: {sorted(unknown)}")
    person_changed = run_person_stage(
        source,
        model=assets["person_detector"],
        device=resolved_device,
        force=force or "person" in rebuild,
        extractor=person_extractor,
        extractor_identity=person_backend_identity,
        tracking_profile=tracking_profile,
    )
    pose_asset = assets["pose_model" if rtmpose else "pose_landmarker"]
    players_changed = run_player_stage(
        source,
        pose_model=pose_asset,
        pose_backend="rtmpose" if rtmpose else "mediapipe",
        pose_device="cuda" if rtmpose and resolved_device == "cuda" else None,
        pose_crop_padding=preprocessing["pose_crop_padding"],
        force=force or person_changed or "pose" in rebuild,
        interpreter=player_interpreter,
        interpreter_identity=pose_backend_identity,
        tracking_profile=tracking_profile,
    )
    prediction_inputs = {
        "identity": identity.fingerprint,
        "assignments_sha256": sha256_file(source.assignments_path),
        "poses_sha256": sha256_file(source.pose_cache_path),
        "calibration_sha256": calibration_fingerprint(source),
        "preprocessing": preprocessing,
        "implementation": implementation_identity("predict"),
        "checkpoint_sha256": sha256_file(assets["checkpoint"]),
        **({"audio_runtime": audio_runtime} if audio_runtime is not None else {}),
        "device": resolved_device,
        "batch_size": batch_size,
    }
    rally_root = source.artifact_root / "rallies"
    rally_root.mkdir(parents=True, exist_ok=True)
    rally_source = RallySourceConfig(
        source.source_id,
        source.video_path,
        source.assignments_path,
        source.pose_cache_path,
        source.calibration_path,
    )
    raw_path = rally_root / "probabilities.jsonl"

    def infer(paths):
        rows = predict_source(
            rally_source,
            model,
            preprocessing=preprocessing,
            decoder=manifest["decoder_configuration"],
            device=resolved_device,
            batch_size=batch_size,
        )
        with paths[0].open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        {
                            key: value
                            for key, value in row.items()
                            if key not in ("decoded_inplay", "rally_id")
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )

    def validate_probabilities(paths):
        from src.temporal_selector.rally_inference import validate_prediction_rows

        rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
        validate_prediction_rows(rows, identity.frame_count)

    predicted = run_stage(
        source,
        "predict",
        prediction_inputs,
        [raw_path],
        rally_root / "probabilities.manifest.json",
        infer,
        validate_probabilities,
        force=force or players_changed or "predict" in rebuild,
    )
    outputs = {
        "predictions": rally_root / "predictions.jsonl",
        "rallies": rally_root / "rallies.csv",
    }
    receipt = rally_root / "run.manifest.json"
    decoder = manifest["decoder_configuration"]
    if bridge_short_dips:
        from src.temporal_selector.rally_evaluation import RALLY_SHORT_GAP_SECONDS

        decoder = {
            **decoder,
            "max_gap_seconds": max(
                decoder.get("max_gap_seconds", RALLY_SHORT_GAP_SECONDS),
                RALLY_SHORT_GAP_SECONDS,
            ),
        }
    decode_inputs = {
        "probabilities_sha256": sha256_file(raw_path),
        "decoder": decoder,
        "implementation": implementation_identity("decode"),
    }

    def decode(paths):
        from src.temporal_selector.rally_inference import decode_prediction_rows

        rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
        decode_prediction_rows(rows, source.source_id, identity, decoder)
        write_prediction_artifacts(
            rally_source,
            rows,
            paths[0].parent,
            metadata={
                "model_id": manifest["model_id"],
                "input_fingerprints": prediction_inputs,
            },
            overlay=False,
        )

    def validate_decoded(paths):
        with paths[0].open() as handle:
            header = json.loads(next(handle))
            rows = [json.loads(line) for line in handle]
        from src.temporal_selector.rally_inference import validate_prediction_rows

        validate_prediction_rows(rows, identity.frame_count)
        if header.get("source_identity_sha256") != identity.fingerprint:
            raise ValueError("prediction source lineage differs")

    decoded = run_stage(
        source,
        "decode",
        decode_inputs,
        list(outputs.values()),
        receipt,
        decode,
        validate_decoded,
        force=force or predicted or "decode" in rebuild,
    )
    if overlay:
        from src.temporal_selector.rally_inference import render_overlay

        output = rally_root / "overlay.mp4"
        render_inputs = {
            "predictions_sha256": sha256_file(outputs["predictions"]),
            "assignments_sha256": sha256_file(source.assignments_path),
            "renderer": render_implementation_identity(),
        }

        def render(paths):
            with outputs["predictions"].open() as handle:
                next(handle)
                rows = [json.loads(line) for line in handle]
            render_overlay(rally_source, rows, identity, paths[0])

        run_stage(
            source,
            "render",
            render_inputs,
            [output],
            rally_root / "overlay.manifest.json",
            render,
            lambda paths: None,
            force=force or decoded or "render" in rebuild,
        )
        outputs["overlay"] = output
    return {**outputs, "manifest": receipt}


def render_implementation_identity() -> str:
    import inspect
    from src.temporal_selector.rally_inference import render_overlay
    from .artifacts import canonical_json_sha256

    return canonical_json_sha256({"renderer": inspect.getsource(render_overlay)})
