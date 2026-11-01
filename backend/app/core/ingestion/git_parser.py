"""Git history parsing: log / diff / name-status / blame (§4.1.2, §4.1.5)."""
from __future__ import annotations

import subprocess
from datetime import datetime

from app.utils.time import utc_now
from pathlib import Path

from app.models.repository import CommitRecord, DeltaSnapshot, FileChange
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _run_git(repo_path: Path | str, args: list[str], timeout: int = 120) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True, text=True, timeout=timeout, check=True,
    )
    return result.stdout


def get_head_commit(repo_path: Path | str) -> str:
    return _run_git(repo_path, ["rev-parse", "HEAD"]).strip()


def get_branches(repo_path: Path | str) -> list[str]:
    out = _run_git(repo_path, ["branch", "-a", "--format=%(refname:short)"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def get_tags(repo_path: Path | str) -> list[str]:
    out = _run_git(repo_path, ["tag", "--sort=-creatordate"])
    return [line.strip() for line in out.splitlines() if line.strip()][:50]


def parse_git_log(repo_path: Path | str, max_commits: int = 500) -> list[CommitRecord]:
    """Parse `git log` into CommitRecords (§4.1.2 'Git History').

    Format uses \x1f (unit separator) to survive arbitrary commit messages.
    """
    sep = "\x1f"
    fmt = sep.join(["%H", "%an", "%ae", "%aI", "%B", "--"])
    try:
        out = _run_git(repo_path, ["log", f"--max-count={max_commits}", f"--pretty=format:{fmt}", "--numstat"])
    except subprocess.CalledProcessError as exc:
        logger.warning(f"git log failed: {exc.stderr}")
        return []

    commits: list[CommitRecord] = []
    current: dict | None = None
    for line in out.splitlines():
        if sep in line and line.count(sep) >= 4:
            if current:
                commits.append(_mk_commit(current))
            parts = line.split(sep)
            current = {
                "hash": parts[0], "author": parts[1], "author_email": parts[2],
                "timestamp": parts[3], "message": parts[4],
                "files_changed": [], "additions": 0, "deletions": 0,
            }
        elif current and line.strip() and "\t" in line:
            add, delete, path = line.split("\t", 2)
            current["files_changed"].append(path.strip())
            current["additions"] += int(add) if add.isdigit() else 0
            current["deletions"] += int(delete) if delete.isdigit() else 0
    if current:
        commits.append(_mk_commit(current))
    return commits


def _mk_commit(c: dict) -> CommitRecord:
    try:
        ts = datetime.fromisoformat(c["timestamp"])
    except (ValueError, KeyError):
        ts = utc_now()
    return CommitRecord(
        hash=c["hash"], author=c["author"], author_email=c.get("author_email", ""),
        timestamp=ts, message=(c.get("message") or "").strip(),
        files_changed=c.get("files_changed", []),
        additions=c.get("additions", 0), deletions=c.get("deletions", 0),
    )


def get_changed_files(repo_path: Path | str, since_commit: str, to_commit: str = "HEAD") -> list[FileChange]:
    """`git diff --name-status` per §4.1.5 Step 2."""
    out = _run_git(repo_path, ["diff", "--name-status", "-M", since_commit, to_commit])
    changes: list[FileChange] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0][0]  # R100 -> R
        if status == "R" and len(parts) >= 3:
            changes.append(FileChange(path=parts[2], status="R", old_path=parts[1]))
        elif len(parts) >= 2:
            changes.append(FileChange(path=parts[-1], status=status))
    return changes


def incremental_update(repo_id: str, repo_path: Path | str, last_indexed_commit: str) -> DeltaSnapshot:
    """§4.1.5 Incremental Indexing Algorithm — verbatim logic."""
    current_commit = get_head_commit(repo_path)
    if current_commit == last_indexed_commit:
        return DeltaSnapshot(repo_id=repo_id, from_commit=last_indexed_commit, to_commit=current_commit)

    changed_files = get_changed_files(repo_path, last_indexed_commit, current_commit)
    added = [f for f in changed_files if f.status == "A"]
    modified = [f for f in changed_files if f.status == "M"]
    deleted = [f for f in changed_files if f.status == "D"]
    renamed = [f for f in changed_files if f.status == "R"]
    return DeltaSnapshot(
        repo_id=repo_id, added=added, modified=modified,
        deleted=deleted, renamed=renamed,
        from_commit=last_indexed_commit, to_commit=current_commit,
    )


def get_last_commit_for_file(repo_path: Path | str, file_path: str) -> CommitRecord | None:
    """Latest commit touching a file — used by provenance (§9.3.2)."""
    sep = "\x1f"
    try:
        out = _run_git(repo_path, ["log", "-1", f"--pretty=format:{sep.join(['%H', '%an', '%aI'])}", "--", file_path])
    except subprocess.CalledProcessError:
        return None
    if not out.strip():
        return None
    parts = out.strip().split(sep)
    if len(parts) < 3:
        return None
    try:
        ts = datetime.fromisoformat(parts[2])
    except ValueError:
        ts = utc_now()
    return CommitRecord(hash=parts[0], author=parts[1], author_email="", timestamp=ts, message="")
