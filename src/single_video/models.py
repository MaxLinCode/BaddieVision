"""Model bundle and loading helpers for single-video extraction."""

from __future__ import annotations

import shutil
from pathlib import Path


MODEL_TREE_PATHS = ("models", "InPlay/models", "src/TrackNetV3/ckpts")
REQUIRED_MODEL_FILES = (
    "models/TrackNet_torchscript.pt",
    "src/TrackNetV3/ckpts/TrackNet_best.pt",
)
INPAINTNET_MODEL_FILES = (
    "models/InpaintNet_torchscript.pt",
    "src/TrackNetV3/ckpts/InpaintNet_best.pt",
)


def validate_model_root(path: Path, *, use_inpaintnet: bool = False) -> Path:
    path = Path(path)
    missing = [relative for relative in MODEL_TREE_PATHS if not (path / relative).is_dir()]
    required_files = REQUIRED_MODEL_FILES + (INPAINTNET_MODEL_FILES if use_inpaintnet else ())
    missing += [relative for relative in required_files if not (path / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Model bundle at {path} is missing: {missing}")
    return path


def looks_like_model_root(path: Path, *, use_inpaintnet: bool = False) -> bool:
    try:
        validate_model_root(path, use_inpaintnet=use_inpaintnet)
    except FileNotFoundError:
        return False
    return True


def copy_model_tree(source_root: Path, target_root: Path, *, use_inpaintnet: bool = False) -> None:
    source_root = validate_model_root(source_root, use_inpaintnet=use_inpaintnet)
    target_root = Path(target_root)
    for relative in MODEL_TREE_PATHS:
        source = source_root / relative
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)
    print(f"Copied model files from {source_root} into {target_root}")


def load_track_models(repo_dir: Path, *, use_inpaintnet: bool = False) -> dict[str, object]:
    import torch
    from TrackNetV3.predictArgs import load_torchscript_model

    repo_dir = validate_model_root(repo_dir, use_inpaintnet=use_inpaintnet)
    tracknet_script = repo_dir / "models" / "TrackNet_torchscript.pt"
    tracknet_path = repo_dir / "src" / "TrackNetV3" / "ckpts" / "TrackNet_best.pt"
    tracknet_ckpt = torch.load(tracknet_path, map_location=torch.device("cpu"))
    models: dict[str, object] = {
        "tracknet": load_torchscript_model(str(tracknet_script)),
        "tracknet_seq_len": int(tracknet_ckpt["param_dict"]["seq_len"]),
        "bg_mode": tracknet_ckpt["param_dict"]["bg_mode"],
        "tracknet_model": str(tracknet_script),
        "tracknet_checkpoint": str(tracknet_path),
        "inpaintnet_enabled": use_inpaintnet,
        "tracking_stage": "tracknet_inpaintnet" if use_inpaintnet else "tracknet",
    }
    if use_inpaintnet:
        inpaintnet_script = repo_dir / "models" / "InpaintNet_torchscript.pt"
        inpaintnet_path = repo_dir / "src" / "TrackNetV3" / "ckpts" / "InpaintNet_best.pt"
        inpaintnet_ckpt = torch.load(inpaintnet_path, map_location=torch.device("cpu"))
        models.update({
            "inpaintnet": load_torchscript_model(str(inpaintnet_script)),
            "inpaintnet_seq_len": int(inpaintnet_ckpt["param_dict"]["seq_len"]),
            "inpaintnet_model": str(inpaintnet_script),
            "inpaintnet_checkpoint": str(inpaintnet_path),
        })
    return models
