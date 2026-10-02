"""Git helpers: provenance, clean-tree checks and per-stage commits of research files.

Commits only ever include the paths they are given (``git commit --only``), so unrelated work in
the checkout is never swept in. Nothing here pushes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


class GitError(RuntimeError):
    pass


def git(root: Path, *args: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", *args],
        check=False,
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if check and process.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {process.stderr.strip()}")
    return process.stdout


def is_repo(root: Path) -> bool:
    try:
        return git(root, "rev-parse", "--is-inside-work-tree").strip() == "true"
    except (GitError, FileNotFoundError):
        return False


def head(root: Path) -> str | None:
    try:
        return git(root, "rev-parse", "HEAD").strip()
    except GitError:
        return None


def dirty(root: Path, paths: list[Path]) -> list[str]:
    """Paths among ``paths`` with uncommitted changes, including untracked files."""

    existing = [str(p) for p in paths if p.exists()]
    if not existing:
        return []
    out = git(root, "status", "--porcelain", "--untracked-files=all", "--", *existing)
    return [line[3:].strip().strip('"') for line in out.splitlines() if line.strip()]


def require_clean(root: Path, paths: list[Path], what: str) -> None:
    changed = dirty(root, paths)
    if changed:
        raise GitError(
            f"{what} has uncommitted changes, so a result could not be tied to a commit:\n  "
            + "\n  ".join(changed)
        )


def commit(root: Path, paths: list[Path], subject: str, body: str = "") -> str | None:
    """Commit exactly ``paths`` (new files included); returns the new SHA or None if unchanged."""

    existing = [str(p) for p in paths if p.exists()]
    if not existing:
        return None
    git(root, "add", "--", *existing)
    staged = git(root, "diff", "--cached", "--name-only", "--", *existing).strip()
    if not staged:
        return None
    parts = [subject, body.strip(), TRAILER]
    message = "\n\n".join(part for part in parts if part) + "\n"
    process = subprocess.run(
        ["git", "commit", "--quiet", "--only", "--file=-", "--", *existing],
        check=False,
        cwd=root,
        input=message,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if process.returncode != 0:
        raise GitError(f"git commit failed: {process.stderr.strip() or process.stdout.strip()}")
    return head(root)


def tag(root: Path, name: str, message: str) -> None:
    git(root, "tag", "--annotate", name, "--message", message)
