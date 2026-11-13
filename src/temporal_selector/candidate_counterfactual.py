"""Export candidate-masked predictions from a frozen within-camera checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from .config import SelectorConfig
from .crossfit import CrossFitFold
from .dataset import SelectorWindowDataset
from .experiment import load_dataset_config, prepare_output_directory, select_device
from .joint_inference import InPlayDecoderConfig
from .model import JointRallyShuttleModel
from .within_camera_decoder import _predict_partition, decode_rows
from .within_camera_experiment import chronological_window_split


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def export_candidate_masked_predictions(
    *,
    config_path: Path,
    checkpoint_path: Path,
    decoder_metrics_path: Path,
    output_dir: Path,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> None:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selector_config = SelectorConfig(**checkpoint["selector_config"])
    dataset_config, _ = load_dataset_config(
        config_path,
        context_mode=selector_config.context_mode,
        pose_coordinate_mode=str(checkpoint["pose_coordinate_mode"]),
    )
    source_order = tuple(checkpoint["split"]["source_order"])
    allowed = set(source_order)
    dataset_config = replace(
        dataset_config,
        sources=tuple(
            source for source in dataset_config.sources if source.source_id in allowed
        ),
    )
    dataset = SelectorWindowDataset(dataset_config)
    if dataset.manifest["dataset_fingerprint"] != checkpoint["dataset_fingerprint"]:
        raise ValueError("checkpoint and counterfactual dataset fingerprints differ")
    _, validation, test, split = chronological_window_split(
        dataset.windows,
        source_order,
        guard_seconds=float(checkpoint["split"]["guard_seconds"]),
    )
    if split != checkpoint["split"]:
        raise ValueError("checkpoint and counterfactual chronological splits differ")
    fps_values = {float(window.metadata["fps"]) for window in validation + test}
    if len(fps_values) != 1:
        raise ValueError("counterfactual partitions must share one FPS")
    fps = next(iter(fps_values))
    decoder_metrics = json.loads(decoder_metrics_path.read_text(encoding="utf-8"))
    decoder = InPlayDecoderConfig(**decoder_metrics["decoder_config"])
    expected_sha = str(decoder_metrics["checkpoint_sha256"])
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_sha != expected_sha:
        raise ValueError("decoder metrics and counterfactual checkpoint differ")

    device = device or select_device()
    model = JointRallyShuttleModel.from_checkpoint(checkpoint).to(device)
    model.eval()
    fold = CrossFitFold("within-camera", source_order, source_order)
    partitions = {
        "validation": validation,
        "test": test,
    }
    output_dir = prepare_output_directory(output_dir)
    for partition, windows in partitions.items():
        rows = _predict_partition(
            model,
            windows,
            fold,
            device=device,
            batch_size=batch_size,
            mask_candidates=True,
        )
        decoded = decode_rows(rows, fps=fps, config=decoder)
        for row in decoded:
            row["checkpoint_sha256"] = checkpoint_sha
            row["decoder_config"] = asdict(decoder)
            row["counterfactual"] = "all_candidate_tokens_masked"
            row["partition"] = partition
        _write_jsonl(output_dir / f"{partition}-predictions.jsonl", decoded)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = None if args.device == "auto" else torch.device(args.device)
    export_candidate_masked_predictions(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        decoder_metrics_path=args.decoder_metrics,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
