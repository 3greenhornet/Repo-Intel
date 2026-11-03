"""GitHub REST client + repository acquisition (§4.1.1–§4.1.2).

Two acquisition modes:
- ``github``  — clone via git (token-aware) + fetch issues/PRs via REST API
- ``local``   — analyze a repository already on disk (offline/dev mode)

Error handling follows the §4.1.7 table (401/403 rate-limit backoff, clone
retry, IO skip-and-continue).
"""
from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.config import get_settings
from app.models.repository import IssueRecord, PRRecord
from app.utils.logger import get_logger
from app.utils.retry import retry

logger = get_logger(__name__)

_REPO_URL_RE = re.compile(r"^(https?://|git@)(github\.com|gitlab\.com|bitbucket\.org)[:/]", re.I)


class GitHubClientError(Exception):
    pass


class GitHubClient:
    """Thin, dependency-light GitHub REST client (§4.1.1)."""

    def __init__(self, token: str | None = None) -> None:
        settings = get_settings()
        self.token = token or settings.github_token
        self.api_url = settings.github_api_url.rstrip("/")

    # ------------------------------------------------------------------ utils
    @staticmethod
    def validate_repo_url(url: str) -> bool:
        return bool(_REPO_URL_RE.match(url.strip()))

    @staticmethod
    def parse_slug(url: str) -> str:
        """owner/repo from a GitHub URL (https or ssh form)."""
        parsed = urlparse(url)
        if parsed.scheme:
            path = parsed.path
        else:  # git@github.com:owner/repo.git
            path = url.split(":", 1)[1] if ":" in url else url
        parts = [p for p in path.strip("/").split("/") if p]
        if len(parts) >= 2:
            repo = parts[1]
            if repo.endswith(".git"):
                repo = repo[:-4]
            return f"{parts[0]}/{repo}"
        raise GitHubClientError(f"Cannot parse repository slug from URL: {url}")

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    # ------------------------------------------------------------- repository
    def clone(self, repo_url: str, target_dir: Path, branch: str = "main") -> Path:
        """Clone with token-aware URL + 3 retries (§4.1.7 clone failure)."""
        target_dir.mkdir(parents=True, exist_ok=True)
        clone_url = repo_url
        if self.token and repo_url.startswith("https://") and "x-access-token" not in repo_url:
            clone_url = repo_url.replace("https://", f"https://x-access-token:{self.token}@")
        cmd = [
            "git", "clone", "--depth", "500",
            "--single-branch", "--branch", branch, clone_url, str(target_dir),
        ]
        # Fall back to default branch if requested branch does not exist.
        try:
            retry(
                lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True),
                attempts=3, base_delay=2.0, label=f"clone {repo_url}", logger=logger,
            )
        except subprocess.CalledProcessError:
            logger.warning(f"Branch {branch} not found, retrying with default branch")
            cmd = ["git", "clone", "--depth", "500", clone_url, str(target_dir)]
            retry(
                lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True),
                attempts=3, base_delay=2.0, label=f"clone default {repo_url}", logger=logger,
            )
        return target_dir

    # ------------------------------------------------------------------ issues
    def fetch_issues(self, owner_repo: str, max_items: int = 200) -> list[IssueRecord]:
        """GET /repos/{owner}/{repo}/issues (§4.1.2) with rate-limit backoff."""
        issues: list[IssueRecord] = []
        page = 1
        try:
            while len(issues) < max_items:
                resp = httpx.get(
                    f"{self.api_url}/repos/{owner_repo}/issues",
                    headers=self._headers(),
                    params={"state": "all", "per_page": 100, "page": page},
                    timeout=30,
                )
                if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
                    logger.warning("GitHub rate limit hit — backing off")
                    break
                if resp.status_code == 401:
                    raise GitHubClientError("GitHub API 401: invalid token")
                if resp.status_code != 200:
                    logger.warning(f"Issues API returned {resp.status_code}")
                    break
                batch = resp.json()
                if not batch:
                    break
                for item in batch:
                    if "pull_request" in item:  # issues endpoint also returns PRs
                        continue

                    issues.append(
                        IssueRecord(
                            number=item["number"],
                            title=item.get("title", ""),
                            body=item.get("body") or "",
                            labels=[lb["name"] for lb in item.get("labels", [])],
                            state=item.get("state", "open"),
                            created_at=_parse_dt(item.get("created_at")),
                            closed_at=_parse_dt(item.get("closed_at")),
                        )
                    )
                page += 1
        except (httpx.HTTPError, GitHubClientError) as exc:
            logger.warning(f"Issue fetch failed for {owner_repo}: {exc}")
        return issues[:max_items]

    # ---------------------------------------------------------------------- PRs
    def fetch_pull_requests(self, owner_repo: str, max_items: int = 200) -> list[PRRecord]:
        """GET /repos/{owner}/{repo}/pulls (§4.1.2)."""
        prs: list[PRRecord] = []
        page = 1
        try:
            while len(prs) < max_items:
                resp = httpx.get(
                    f"{self.api_url}/repos/{owner_repo}/pulls",
                    headers=self._headers(),
                    params={"state": "all", "per_page": 100, "page": page},
                    timeout=30,
                )
                if resp.status_code != 200:
                    break
                batch = resp.json()
                if not batch:
                    break

                for item in batch:
                    prs.append(
                        PRRecord(
                            number=item["number"],
                            title=item.get("title", ""),
                            body=item.get("body") or "",
                            reviewers=[r["login"] for r in item.get("requested_reviewers", [])],
                            state=item.get("state", "open"),
                            merged_at=_parse_dt(item.get("merged_at")),
                        )
                    )
                page += 1
        except httpx.HTTPError as exc:
            logger.warning(f"PR fetch failed for {owner_repo}: {exc}")
        return prs[:max_items]


def _parse_dt(value: str | None):
    if not value:
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def resolve_repo_id(repo_url: str) -> str:
    """Stable repo id (uuid5 of the normalized URL). Case-sensitive for paths."""
    clean = repo_url.strip().removesuffix(".git")
    if clean.startswith(("http://", "https://", "git@")):
        normalized = clean.lower()
    else:
        normalized = clean
    return str(uuid.uuid5(uuid.NAMESPACE_URL, normalized))
