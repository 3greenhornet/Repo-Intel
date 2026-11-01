"""GitHub issue/PR fetcher with graceful offline fallback (§4.1.2).

When the repository is local-only or the network is unavailable, the fetcher
returns synthetic issue records from any local fixture files, keeping the
pipeline runnable end-to-end (documented fallback behavior).
"""
from __future__ import annotations

import json
from pathlib import Path

from app.core.ingestion.github_client import GitHubClient, GitHubClientError, resolve_repo_id
from app.models.repository import IssueRecord, PRRecord
from app.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_issues_and_prs(
    repo_url: str, repo_path: Path, online: bool = True
) -> tuple[list[IssueRecord], list[PRRecord]]:
    """Fetch issues + PRs from GitHub API; fall back to local fixtures."""
    if online and GitHubClient.validate_repo_url(repo_url):
        client = GitHubClient()
        try:
            slug = GitHubClient.parse_slug(repo_url)
            issues = client.fetch_issues(slug)
            prs = client.fetch_pull_requests(slug)
            logger.info(f"Fetched {len(issues)} issues and {len(prs)} PRs for {slug}")
            return issues, prs
        except GitHubClientError as exc:
            logger.warning(f"GitHub fetch failed ({exc}); continuing without issues/PRs")

    # Local fixture fallback: .repointel/issues.json in the repo root
    fixture = repo_path / ".repointel" / "issues.json"
    if fixture.exists():
        try:
            data = json.loads(fixture.read_text(encoding="utf-8"))
            issues = [IssueRecord(**item) for item in data.get("issues", [])]
            prs = [PRRecord(**item) for item in data.get("pull_requests", [])]
            logger.info(f"Loaded {len(issues)} local fixture issues, {len(prs)} PRs")
            return issues, prs
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(f"Bad issues fixture: {exc}")
    return [], []


__all__ = ["fetch_issues_and_prs", "resolve_repo_id"]
