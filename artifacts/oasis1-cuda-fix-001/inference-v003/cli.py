"""Command-line entry point for one subject's nine original-view slices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference import EXPECTED_CHECKPOINT_SHA256, load_model, predict_subject_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="OASIS-1 CNN v003 exploratory subject inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--slice", dest="slices", action="append", required=True,
                        help="Repeat exactly nine times; original-view PNGs only")
    args = parser.parse_args()
    model = load_model(args.checkpoint, expected_sha256=EXPECTED_CHECKPOINT_SHA256)
    print(json.dumps(predict_subject_paths(model, args.slices, subject_id=args.subject_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
