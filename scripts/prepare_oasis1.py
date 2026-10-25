"""Prepare official OASIS-1 age-60+ slices/manifests without training or bulk extraction."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neuro_path_ai.data.oasis1_prepare import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--reliability", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        prepare(args.data_dir, args.clinical, args.output_dir, args.reliability, args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Preparation stopped: {exc}\n")


if __name__ == "__main__":
    main()
