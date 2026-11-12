"""Export a trusted local research checkpoint as a sanitized public model bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rally_release import export_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--person-detector", type=Path, required=True)
    parser.add_argument("--pose-landmarker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="rally-six-source-v1")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--preprocessing",
        type=Path,
        help="JSON containing the exact preprocessing contract used for this model",
    )
    args = parser.parse_args()
    preprocessing = (
        json.loads(args.preprocessing.read_text()) if args.preprocessing else None
    )
    result = export_release(
        args.checkpoint,
        args.output_dir,
        person_detector=args.person_detector,
        pose_landmarker=args.pose_landmarker,
        code_revision=args.code_revision,
        base_url=args.base_url,
        model_id=args.model_id,
        preprocessing=preprocessing,
    )
    print(result)


if __name__ == "__main__":
    main()
