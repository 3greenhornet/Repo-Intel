"""Repository ingestion data models (§4.1.6)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.utils.time import utc_now
from enum import Enum
from typing import Any, Optional


class FileCategory(str, Enum):
    """File classification categories (§4.1.3)."""

    CODE = "code"
    CONFIG = "config"
    DOCS = "docs"
    TESTS = "tests"
    GENERATED = "generated"
    BINARY = "binary"
    VENDORED = "vendored"
    DATA = "data"
    OTHER = "other"


@dataclass
class FileRecord:
    """A single walked file (§4.1.4 algorithm output)."""

    path: str
    category: str  # code, config, docs, tests, generated, binary, vendored, data
    language: Optional[str]
    size_bytes: int
    content_hash: str
    last_modified: datetime


@dataclass
class FileChange:
    """A file change from a git diff (§4.1.5)."""

    path: str
    status: str  # A, M, D, R
    old_path: Optional[str] = None
    diff: Optional[str] = None


@dataclass
class DeltaSnapshot:
    """Incremental update delta (§4.1.5)."""

    repo_id: str
    added: list[FileChange] = field(default_factory=list)
    modified: list[FileChange] = field(default_factory=list)
    deleted: list[FileChange] = field(default_factory=list)
    renamed: list[FileChange] = field(default_factory=list)
    from_commit: str = ""
    to_commit: str = ""

    @property
    def total_changes(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted) + len(self.renamed)


@dataclass
class CommitRecord:
    """A git commit (§4.1.6)."""

    hash: str
    author: str
    author_email: str
    timestamp: datetime
    message: str
    files_changed: list[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0


@dataclass
class IssueRecord:
    """A GitHub issue (§4.1.6)."""

    number: int
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    state: str = "open"  # open, closed
    created_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    comments: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PRRecord:
    """A GitHub pull request (§4.1.6)."""

    number: int
    title: str
    body: str
    reviewers: list[str] = field(default_factory=list)
    state: str = "open"  # open, merged, closed
    merged_at: Optional[datetime] = None
    files_changed: list[str] = field(default_factory=list)
    comments: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DocRecord:
    """A documentation file (§4.1.2 'Documentation' source)."""

    file_path: str
    doc_type: str  # readme, api_doc, architecture, manifest
    content: str


@dataclass
class RepositorySnapshot:
    """Full ingestion snapshot (§4.1.6)."""

    repo_id: str
    repo_url: str
    branch: str
    commit_hash: str
    files: list[FileRecord] = field(default_factory=list)
    commits: list[CommitRecord] = field(default_factory=list)
    issues: list[IssueRecord] = field(default_factory=list)
    pull_requests: list[PRRecord] = field(default_factory=list)
    documentation: list[DocRecord] = field(default_factory=list)
    ingested_at: datetime = field(default_factory=utc_now)

    def stats(self) -> dict[str, Any]:
        """Summary stats used by the dashboard and logs."""
        return {
            "repo_id": self.repo_id,
            "files": len(self.files),
            "commits": len(self.commits),
            "issues": len(self.issues),
            "pull_requests": len(self.pull_requests),
            "docs": len(self.documentation),
            "commit_hash": self.commit_hash,
            "by_category": {
                c: sum(1 for f in self.files if f.category == c)
                for c in {f.category for f in self.files}
            },
        }
