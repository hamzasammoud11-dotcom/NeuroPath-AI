from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


CHECKPOINT_DIR = Path("artifacts/checkpoints")
CHECKPOINTS = ("best_model.pt", "latest_model.pt")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(command: list[str], repo_root: Path) -> None:
    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "Checkpoint command failed.").strip()
        raise SystemExit(message)


def _git_lfs_available(repo_root: Path) -> None:
    _run(["git", "lfs", "version"], repo_root)


def pull(repo_root: Path) -> None:
    _git_lfs_available(repo_root)
    _run(["git", "lfs", "install"], repo_root)
    _run(["git", "lfs", "pull"], repo_root)
    print("Checkpoints pulled successfully.")


def push(repo_root: Path, message: str) -> None:
    _git_lfs_available(repo_root)
    _run(["git", "lfs", "install"], repo_root)
    checkpoint_args = [str(CHECKPOINT_DIR / name) for name in CHECKPOINTS]
    _run(["git", "add", *checkpoint_args], repo_root)
    _run(["git", "commit", "-m", message], repo_root)
    _run(["git", "push"], repo_root)
    print("Checkpoints pushed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync NeuroPath-AI checkpoints between machines.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pull_parser = subparsers.add_parser("pull", help="Pull checkpoint files with Git LFS.")

    push_parser = subparsers.add_parser("push", help="Commit and push checkpoint files.")
    push_parser.add_argument(
        "--message",
        default="Update checkpoints",
        help="Commit message used when pushing checkpoints.",
    )

    args = parser.parse_args()
    repo_root = _repo_root()

    if args.command == "pull":
        pull(repo_root)
    elif args.command == "push":
        push(repo_root, args.message)


if __name__ == "__main__":
    main()