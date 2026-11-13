#!/usr/bin/env python3
"""Lint Python files changed from a CI base revision, without vendored code."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    output = subprocess.check_output([
        "git", "diff", "--name-only", "--diff-filter=ACMR", "-z", args.base, "HEAD",
    ])
    paths = [
        name for name in output.decode().split("\0")
        if name.endswith(".py") and not name.startswith("src/TrackNetV3/")
        and Path(name).is_file()
    ]
    return subprocess.call([sys.executable, "-m", "ruff", "check", "--isolated", "--select", "E4,E7,E9,F", "--", *paths]) if paths else 0


if __name__ == "__main__":
    raise SystemExit(main())
