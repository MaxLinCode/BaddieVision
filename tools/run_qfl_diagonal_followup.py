"""Train the frozen QFL and diagonal service-box factorial on development folds."""

import gc
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.temporal_selector.joint_inference import InPlayDecoderConfig
from src.temporal_selector.rally import RallyTrainingRecipe
from src.temporal_selector.rally_dataset import (
    RALLY_TRANSFORMER_CONTEXT_RADIUS_SECONDS,
    rally_data_config_from_selector_mapping,
)
from src.temporal_selector.rally_experiment import run_clean_rally_experiment
from src.workflow.identity import sha256_file

SEEDS = (1729, 1730, 1731, 1732, 1733)
ARMS = {
    "baseline": ("full_audio_zscore_motion", "triangle_bce"),
    "qfl": ("full_audio_zscore_motion", "gaussian_qfl"),
    "diagonal": ("serve_boxes_diagonal", "triangle_bce"),
    "qfl-diagonal": ("serve_boxes_diagonal", "gaussian_qfl"),
}
FIXED_DECODER = InPlayDecoderConfig(
    threshold=0.40, continuation_threshold=0.35, max_gap_seconds=0.5,
    minimum_duration_seconds=0.3, preserve_edge_runs=True,
)
REFERENCE_JOBS = {
    "baseline": (
        "20261127T070834-40a77acc", "serve-box-ablation/full_audio_zscore_motion",
    ),
    "qfl": ("20261127T010615-4123a4ac", "boundary/gaussian-qfl"),
    "diagonal": (
        "20261127T070834-40a77acc", "serve-box-ablation/serve_boxes_diagonal",
    ),
}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def output_dir(jobs, job):
    return jobs / job / "workspace" / "outputs"


def fold_identity(metrics):
    entries = []
    for fold in metrics["folds"]:
        split = fold["split_manifest"]
        entries.append({
            "fold_id": split["fold_id"],
            "held_out_camera": split["held_out_camera"],
            "training_source_ids": split["training_source_ids"],
            "validation_source_ids": split["validation_source_ids"],
            "test_source_ids": split["test_source_ids"],
            "train_window_ids": split["train_window_ids"],
            "validation_window_ids": split["validation_window_ids"],
            "test_window_ids": split["test_window_ids"],
            "ranges": split["train_validation"]["cameras"],
        })
    return entries


def validate_design(metrics, reference):
    actual_data = metrics["dataset_manifest"]
    expected_data = reference["dataset_manifest"]
    for key in ("annotation_fingerprint", "sources", "context_seconds", "ownership_seconds"):
        if actual_data[key] != expected_data[key]:
            raise ValueError(f"Development dataset mismatch in {key}")
    if metrics["camera_sources"] != reference["camera_sources"] or fold_identity(metrics) != fold_identity(reference):
        raise ValueError("Historical camera-fold or window partition changed")


def run():
    jobs = Path(".venv").resolve().parent / ".remote-gpu" / "jobs"
    config_path = output_dir(jobs, "20261121T175749-2a6a1067") / "paired-rally-pose" / "rtmpose.json"
    raw_config = read_json(config_path)
    data_config = rally_data_config_from_selector_mapping(
        raw_config, base_dir=config_path.parent,
        default_context_seconds=RALLY_TRANSFORMER_CONTEXT_RADIUS_SECONDS,
    )
    references = {
        arm: read_json(output_dir(jobs, job) / subdir / "metrics.json")
        for arm, (job, subdir) in REFERENCE_JOBS.items()
    }
    reference = references["baseline"]
    recipe = RallyTrainingRecipe(**reference["training_recipe"])
    if (recipe.seed, recipe.epochs, recipe.batch_size, recipe.device) != (1729, 25, 16, "cuda"):
        raise ValueError("The archived production recipe is unexpected")
    if len(data_config.sources) != 7 or len(reference["folds"]) != 4:
        raise ValueError("The preregistered design requires seven development sources and four folds")
    if raw_config["camera_sources"] != reference["camera_sources"]:
        raise ValueError("The development configuration does not match the historical camera groups")
    for arm, historical in references.items():
        validate_design(historical, reference)
        expected_recipe = replace(recipe, boundary_supervision=ARMS[arm][1])
        if RallyTrainingRecipe(**historical["training_recipe"]) != expected_recipe:
            raise ValueError(f"The archived {arm} training recipe differs outside its planned change")
        if historical["dataset_manifest"]["feature_view"] != ARMS[arm][0]:
            raise ValueError(f"The archived {arm} feature channel does not match")
    release_manifest = Path(".venv").resolve().parent / ".remote-gpu" / "models" / "seven-source-full-motion-rtmpose-audio-20261122" / "manifest.json"
    if read_json(release_manifest)["decoder_configuration"] != asdict(FIXED_DECODER):
        raise ValueError("The requested decoder does not match the frozen production release")

    output = Path("outputs/qfl-diagonal-followup")
    results = Path("outputs/qfl-diagonal-results")
    results.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "qfl_diagonal_factorial_development",
        "schema_version": 1,
        "dataset": "existing_seven_source_development",
        "held_out_camera_folds": 4,
        "seed_order": SEEDS,
        "arms": {
            name: {"feature_view": view, "boundary_supervision": supervision}
            for name, (view, supervision) in ARMS.items()
        },
        "reference_training_recipe": asdict(recipe),
        "production_decoder": asdict(FIXED_DECODER),
        "development_config_sha256": sha256_file(config_path),
        "development_annotation_fingerprint": reference["dataset_manifest"]["annotation_fingerprint"],
        "reference_jobs": REFERENCE_JOBS,
        "source_revision": "e19ffa5",
        "experiment_trainer_sha256": sha256_file(Path("src/temporal_selector/rally_experiment.py")),
        "holdout_used": False,
    }
    write_json(results / "protocol.json", protocol)

    archive_cache = output_dir(jobs, REFERENCE_JOBS["diagonal"][0]) / "serve-box-ablation" / ".rally-partitions"
    cache = output / ".rally-partitions"
    if not cache.exists():
        shutil.copytree(archive_cache, cache)
    print("Validated matched seven-source data, four-fold splits and fixed decoder", flush=True)

    for seed in SEEDS:
        seed_output = output / f"seed-{seed}"
        seed_output.mkdir(exist_ok=True)
        link = seed_output / ".rally-partitions"
        if not link.is_symlink():
            link.symlink_to(Path("..") / cache.name, target_is_directory=True)
        for arm, (view, supervision) in ARMS.items():
            arm_output = seed_output / arm
            arm_results = results / f"seed-{seed}" / arm
            started = time.monotonic()
            print(f"START seed {seed} {arm}", flush=True)
            if (arm_output / "metrics.json").exists():
                scores = read_json(arm_output / "metrics.json")
            else:
                scores = run_clean_rally_experiment(
                    replace(data_config, feature_view=view), arm_output,
                    camera_sources=raw_config["camera_sources"],
                    recipe=replace(recipe, seed=seed, boundary_supervision=supervision),
                    device=torch.device("cuda"), resume=(arm_output / "resolved-config.json").exists(),
                    require_rtmpose=True, person_model="models/yolo26l.engine",
                    pose_model=Path("models/rtmpose-l-halpe26-384x288-fp16.onnx"),
                    evaluation_decoder=FIXED_DECODER,
                )
            validate_design(scores, reference)
            if scores["production"] is not None:
                raise ValueError("Development-only training unexpectedly produced an all-source model")
            if arm in references:
                actual = scores["dataset_manifest"]["dataset_fingerprint"]
                expected = references[arm]["dataset_manifest"]["dataset_fingerprint"]
                if actual != expected:
                    raise ValueError(f"The {arm} dataset fingerprint differs from its historical control")
            if any(fold["test_decoder"] != asdict(FIXED_DECODER) for fold in scores["folds"]):
                raise ValueError("A development fold did not use the fixed decoder")
            arm_results.mkdir(parents=True, exist_ok=True)
            for name in ("resolved-config.json", "metrics.json"):
                shutil.copy2(arm_output / name, arm_results / name)
            for fold in scores["folds"]:
                split = fold["split_manifest"]
                name = f"fold-{split['fold_id']}-{split['held_out_camera']}"
                dst = arm_results / name
                dst.mkdir(exist_ok=True)
                for filename in ("metrics.json", "test-predictions.jsonl"):
                    shutil.copy2(arm_output / name / filename, dst / filename)
            print(f"DONE seed {seed} {arm} in {time.monotonic() - started:.1f}s", flush=True)
            del scores
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
