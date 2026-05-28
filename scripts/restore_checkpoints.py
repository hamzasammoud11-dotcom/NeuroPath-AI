from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            header = handle.read(128)
    except OSError:
        return False
    return header.startswith(b"version https://git-lfs.github.com/spec/v1")


def _checkpoint_status(checkpoint_dir: Path) -> list[str]:
    status: list[str] = []
    for name in ("best_model.pt", "latest_model.pt"):
        path = checkpoint_dir / name
        if not path.exists():
            status.append(f"missing: {path}")
        elif _is_lfs_pointer(path):
            status.append(f"lfs-pointer: {path}")
        else:
            status.append(f"ok: {path}")
    return status


def _run_git_lfs_pull(repo_root: Path) -> None:
    git_check = subprocess.run(["git", "lfs", "version"], cwd=repo_root, capture_output=True, text=True)
    if git_check.returncode != 0:
        raise SystemExit(
            "Git LFS n'est pas disponible. Installe Git LFS, puis relance: git lfs install && git lfs pull"
        )

    install = subprocess.run(["git", "lfs", "install"], cwd=repo_root, capture_output=True, text=True)
    if install.returncode != 0:
        raise SystemExit((install.stderr or install.stdout or "Impossible d'initialiser Git LFS.").strip())

    pull = subprocess.run(["git", "lfs", "pull"], cwd=repo_root, capture_output=True, text=True)
    if pull.returncode != 0:
        raise SystemExit((pull.stderr or pull.stdout or "Impossible de récupérer les objets Git LFS.").strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore NeuroPath-AI checkpoints stored in Git LFS.")
    parser.add_argument("--check-only", action="store_true", help="Only report checkpoint status.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = repo_root / "artifacts" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    status = _checkpoint_status(checkpoint_dir)
    needs_restore = any(item.startswith(("missing:", "lfs-pointer:")) for item in status)

    for item in status:
        print(item)

    if args.check_only:
        raise SystemExit(0 if not needs_restore else 1)

    if not needs_restore:
        print("Checkpoints already restored.")
        return

    _run_git_lfs_pull(repo_root)

    refreshed = _checkpoint_status(checkpoint_dir)
    for item in refreshed:
        print(item)

    if any(item.startswith(("missing:", "lfs-pointer:")) for item in refreshed):
        raise SystemExit("Les checkpoints ne sont pas encore récupérés. Vérifie la connexion Git LFS et le remote.")

    print("Checkpoints restored successfully.")


if __name__ == "__main__":
    main()