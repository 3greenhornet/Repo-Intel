"""IngestionComponent — top-level ingestion orchestrator (§3.2.1 interface).

```python
class IngestionComponent:
    def ingest(repo_url, branch="main") -> RepositorySnapshot
    def incremental_update(repo_id, since_commit) -> DeltaSnapshot
    def get_changed_files(since_commit) -> list[FileChange]
```
"""
from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.core.ingestion.file_walker import read_file_content, walk_repository
from app.core.ingestion.github_client import GitHubClient, resolve_repo_id
from app.core.ingestion.git_parser import (
    get_branches,
    get_head_commit,
    get_tags,
    incremental_update as git_incremental_update,
    parse_git_log,
)
from app.core.ingestion.issue_fetcher import fetch_issues_and_prs
from app.models.repository import (DeltaSnapshot, DocRecord,
                                    FileChange, RepositorySnapshot)
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DOC_TYPES = {
    "readme": {"readme", "readme.md"},
    "architecture": {"architecture", "design"},
    "api_doc": {"api"},
}


class IngestionComponent:
    """Concrete IngestionComponent (§3.2.1). Supports github URLs and local paths."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._local_repos: dict[str, Path] = {}  # repo_id → on-disk path

    # ------------------------------------------------------------------ ingest
    def ingest(
        self,
        repo_url: str,
        branch: str = "main",
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        fetch_github_data: bool = True,
    ) -> RepositorySnapshot:
        """Full ingestion. Returns snapshot with commit hash (§3.2.1)."""
        repo_id = resolve_repo_id(repo_url)

        if GitHubClient.validate_repo_url(repo_url):
            repo_path = self.settings.repos_dir / repo_id
            if not (repo_path / ".git").exists():
                GitHubClient().clone(repo_url, repo_path, branch)
            else:
                logger.info(f"Reusing existing clone at {repo_path}")
            online = fetch_github_data
        else:
            repo_path = Path(repo_url).resolve()
            if not repo_path.exists():
                raise FileNotFoundError(f"Local repository path does not exist: {repo_url}")
            online = False

        self._local_repos[repo_id] = repo_path

        # STEP 1a: file walk (§4.1.4)
        files = walk_repository(repo_path, include_patterns, exclude_patterns)

        # STEP 1b: git history (§4.1.2)
        commit_hash = get_head_commit(repo_path)
        commits = parse_git_log(repo_path)
        branches = get_branches(repo_path)
        tags = get_tags(repo_path)

        # STEP 1c: issues + PRs (§4.1.2)
        issues, pull_requests = fetch_issues_and_prs(repo_url, repo_path, online=online)

        # STEP 1d: documentation extraction (§4.1.2)
        documentation = self._extract_documentation(repo_path, files)

        snapshot = RepositorySnapshot(
            repo_id=repo_id,
            repo_url=repo_url,
            branch=branch,
            commit_hash=commit_hash,
            files=files,
            commits=commits,
            issues=issues,
            pull_requests=pull_requests,
            documentation=documentation,
        )
        logger.info("ingestion_complete", extra={"extra_fields": {
            **snapshot.stats(), "branches": len(branches), "tags": len(tags),
        }})
        return snapshot

    # ------------------------------------------------------ incremental_update
    def incremental_update(self, repo_id: str, since_commit: str, repo_url: str = "") -> "DeltaSnapshot":  # type: ignore[name-defined]
        """Only process changed files (§3.2.1 / §4.1.5)."""
        repo_path = self._local_repos.get(repo_id)
        if repo_path is None:
            if repo_url:
                self._local_repos[repo_id] = (
                    self.settings.repos_dir / repo_id
                    if GitHubClient.validate_repo_url(repo_url)
                    else Path(repo_url).resolve()
                )
                repo_path = self._local_repos[repo_id]
        if repo_path is None or not repo_path.exists():
            raise FileNotFoundError(f"Repository path for {repo_id} unknown — run full ingest first")
        delta = git_incremental_update(repo_id, repo_path, since_commit)
        logger.info("incremental_delta", extra={"extra_fields": {
            "repo_id": repo_id, "changes": delta.total_changes, "to_commit": delta.to_commit[:12],
        }})
        return delta

    # -------------------------------------------------------------- get_changed
    def get_changed_files(self, repo_path: str | Path, since_commit: str) -> list["FileChange"]:  # type: ignore[name-defined]
        """Git-diff-based file list (§3.2.1)."""
        from app.core.ingestion.git_parser import get_changed_files as gcf

        return gcf(Path(repo_path), since_commit)

    # ---------------------------------------------------------------- internals
    def _extract_documentation(self, repo_path: Path, files) -> list[DocRecord]:
        docs: list[DocRecord] = []
        for file in files:
            if file.category != "docs":
                continue
            name = file.path.rsplit("/", 1)[-1].lower().removesuffix(".md").removesuffix(".rst").removesuffix(".txt")
            doc_type = "readme" if name in _DOC_TYPES["readme"] else (
                "architecture" if any(k in name for k in _DOC_TYPES["architecture"]) else (
                    "api_doc" if any(k in name for k in _DOC_TYPES["api_doc"]) else "doc"
                )
            )
            content, ok = read_file_content(repo_path, file.path)
            if ok:
                docs.append(DocRecord(file_path=file.path, doc_type=doc_type, content=content))
        return docs

    def repo_path_for(self, repo_id: str) -> Path | None:
        return self._local_repos.get(repo_id)
