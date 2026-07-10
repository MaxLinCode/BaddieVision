"""Model bundle and loading helpers for single-video extraction."""

from __future__ import annotations

import shutil
from pathlib import Path


MODEL_TREE_PATHS = ("models", "InPlay/models", "src/TrackNetV3/ckpts")
REQUIRED_MODEL_FILES = (
    "models/TrackNet_torchscript.pt",
    "models/InpaintNet_torchscript.pt",
    "src/TrackNetV3/ckpts/TrackNet_best.pt",
    "src/TrackNetV3/ckpts/InpaintNet_best.pt",
)


def validate_model_root(path: Path) -> Path:
    path = Path(path)
    missing = [relative for relative in MODEL_TREE_PATHS if not (path / relative).is_dir()]
    missing += [relative for relative in REQUIRED_MODEL_FILES if not (path / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Model bundle at {path} is missing: {missing}")
    return path


def looks_like_model_root(path: Path) -> bool:
    try:
        validate_model_root(path)
    except FileNotFoundError:
        return False
    return True


def copy_model_tree(source_root: Path, target_root: Path) -> None:
    source_root = validate_model_root(source_root)
    target_root = Path(target_root)
    for relative in MODEL_TREE_PATHS:
        source = source_root / relative
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)
    print(f"Copied model files from {source_root} into {target_root}")


def load_track_models(repo_dir: Path) -> dict[str, object]:
    import torch
    from TrackNetV3.predictArgs import load_torchscript_model

    repo_dir = validate_model_root(repo_dir)
    tracknet_script = repo_dir / "models" / "TrackNet_torchscript.pt"
    inpaintnet_script = repo_dir / "models" / "InpaintNet_torchscript.pt"
    tracknet_path = repo_dir / "src" / "TrackNetV3" / "ckpts" / "TrackNet_best.pt"
    inpaintnet_path = repo_dir / "src" / "TrackNetV3" / "ckpts" / "InpaintNet_best.pt"
    tracknet_ckpt = torch.load(tracknet_path, map_location=torch.device("cpu"))
    inpaintnet_ckpt = torch.load(inpaintnet_path, map_location=torch.device("cpu"))
    return {
        "tracknet": load_torchscript_model(str(tracknet_script)),
        "inpaintnet": load_torchscript_model(str(inpaintnet_script)),
        "tracknet_seq_len": int(tracknet_ckpt["param_dict"]["seq_len"]),
        "inpaintnet_seq_len": int(inpaintnet_ckpt["param_dict"]["seq_len"]),
        "bg_mode": tracknet_ckpt["param_dict"]["bg_mode"],
        "tracknet_checkpoint": str(tracknet_path),
        "inpaintnet_checkpoint": str(inpaintnet_path),
    }
