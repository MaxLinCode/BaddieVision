#!/usr/bin/env python3
"""Reject private/generated data in the current Git index; never inspect history."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Iterable


BINARY_ASSET_SUFFIXES = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png", ".webp",
    ".pt", ".pth", ".onnx", ".torchscript", ".tflite", ".task", ".zip",
    ".npy", ".npz",
})
PRIVATE_DIRECTORIES = frozenset({
    "videos", "clips", "scratch", "features", "clip_features", "outputs",
})
PERSONAL_HOME = re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/")


def private_path_reason(name: str) -> str | None:
    """Use paths, rather than ignore rules, so force-added assets also fail."""
    path = PurePosixPath(name)
    if path.name.endswith(".private-provenance.json"):
        return "private sample provenance"
    if any(part == ".annotation" or part.startswith(".annotation-") for part in path.parts):
        return "private annotation runtime"
    if any(part in PRIVATE_DIRECTORIES for part in path.parts):
        return "private or generated source artifacts"
    if path.suffix.lower() in BINARY_ASSET_SUFFIXES:
        return "media, model weights, or generated binary data belong in verified release assets"
    if path.parts[:2] == ("yolov8", "shuttle-yolo-dataset") and any(
        part in {"images", "labels"} for part in path.parts[2:]
    ):
        return "private detector dataset"
    if ".local" in path.name and path.suffix == ".json":
        return "private local configuration"
    if path.parts and path.parts[0] == "config" and (
        ".local" in path.name or path.name.endswith("-manifest.json")
        or ".generated" in path.name
    ):
        return "local recording configuration or generated split manifest"
    if path.parts[:3] == ("tests", "fixtures", "golden"):
        return "real-recording golden data; use synthetic fixtures"
    if path.suffix.lower() in {".csv", ".jsonl"} and path.parts[:2] != ("tests", "fixtures"):
        return "recording annotations or observations outside synthetic test fixtures"
    return None


def check_paths(root: Path, names: Iterable[str]) -> list[str]:
    problems: list[str] = []
    for name in names:
        reason = private_path_reason(name)
        if reason:
            problems.append(f"{name}: {reason}")
            continue
        path = root / name
        if not path.is_file():
            continue  # Submodules and intentionally deleted working files.
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if PERSONAL_HOME.search(text):
            problems.append(f"{name}: personal absolute home path")
        if path.suffix == ".py" and PurePosixPath(name).parts[0] != "tests":
            try:
                tree = ast.parse(text)
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign):
                        continue
                    if not any(isinstance(target, ast.Name) and target.id in {"rally_intervals", "rally_segments", "label_string"} for target in node.targets):
                        continue
                    if isinstance(node.value, (ast.List, ast.Tuple)) and node.value.elts:
                        problems.append(f"{name}: embedded recording-specific rally annotations")
                        break
            except SyntaxError:
                pass  # Python syntax is checked separately by lint and tests.
        if path.suffix == ".ipynb":
            try:
                notebook = json.loads(text)
                cells = notebook["cells"]
                if any(cell.get("outputs") or cell.get("execution_count") is not None for cell in cells):
                    problems.append(f"{name}: saved notebook output or execution state")
                # Local hit lists should never be embedded in notebook source.
                source = "\n".join("".join(cell.get("source", ())) for cell in cells)
                if re.search(r"(?m)^\s*(?:serve|clear|lift|smash|drop|net|drive|backhand)\s*:\s*\d", source):
                    problems.append(f"{name}: embedded recording-specific hit annotations")
            except (ValueError, KeyError, TypeError):
                problems.append(f"{name}: invalid notebook")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    result = subprocess.run(
        ["git", "ls-files", "--cached", "-z"], cwd=root, check=True, capture_output=True,
    )
    names = result.stdout.decode("utf-8").split("\0")
    problems = check_paths(root, (name for name in names if name))
    if problems:
        print("Private/generated data must remain local or in an approved release:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("Tracked-file privacy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
