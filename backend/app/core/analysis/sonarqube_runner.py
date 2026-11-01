"""SonarQube runner (§4.2.3 Component C) + paginated findings fetch (§23.3 G3 fix).

Runs the sonar-scanner CLI when enabled and a SonarQube server is reachable;
otherwise returns [] so the built-in analyzer carries the pipeline. The
documented pagination algorithm is implemented verbatim.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import httpx

from app.config import get_settings
from app.core.analysis.findings_parser import parse_sonar_json
from app.models.finding import Finding
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SonarQubeError(Exception):
    pass


def _server_available(url: str) -> bool:
    try:
        resp = httpx.get(f"{url.rstrip('/')}/api/system/status", timeout=3)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def run_sonarqube(repo_path: str | Path, repo_id: str) -> list[Finding]:
    """§4.2.3: generate config → run scanner → fetch findings (paginated §23.3).

    FIX-007: the sonar-project.properties file is written to a temp directory
    and passed via -Dproject.settings, never into the analyzed working tree,
    so incremental diffs and the vector index never see it.
    """
    settings = get_settings()
    if not settings.enable_sonar:
        logger.info("SonarQube disabled (ENABLE_SONAR=0) — skipping")
        return []
    if not _server_available(settings.sonar_url):
        logger.warning(f"SonarQube not reachable at {settings.sonar_url} — skipping")
        return []

    repo_path = Path(repo_path)

    with tempfile.TemporaryDirectory(prefix="repointel-sonar-") as tmpdir:
        # Step 1: sonar-project.properties (in the temp dir — FIX-007)
        props = Path(tmpdir) / "sonar-project.properties"
        props.write_text(
            f"sonar.projectKey={repo_id}\n"
            f"sonar.projectBaseDir={repo_path}\n"
            f"sonar.sources={repo_path}\n"
            "sonar.sourceEncoding=UTF-8\n",
            encoding="utf-8",
        )
        # Step 2: run scanner against the temp config
        cmd = [
            "sonar-scanner",
            f"-Dproject.settings={props}",
            f"-Dsonar.host.url={settings.sonar_url}",
        ]
        if settings.sonar_token:
            cmd.append(f"-Dsonar.login={settings.sonar_token}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise SonarQubeError(result.stderr[-500:])

    # Step 3: fetch findings
    issues = fetch_all_sonar_issues(settings.sonar_url, settings.sonar_token, repo_id)
    return parse_sonar_json(issues)


def fetch_all_sonar_issues(sonar_url: str, token: str, project_key: str) -> list[dict]:
    """§23.3 pagination algorithm — max 500/page, stop at total."""
    all_issues: list[dict] = []
    page = 1
    page_size = 500
    base = sonar_url.rstrip("/")
    auth = (token, "") if token else None
    while True:
        resp = httpx.get(
            f"{base}/api/issues/search",
            params={"componentKeys": project_key, "resolved": "false", "ps": page_size, "p": page},
            auth=auth, timeout=30,
        )
        if resp.status_code == 401:
            raise SonarQubeError("SonarQube 401: check token")
        if resp.status_code != 200:
            raise SonarQubeError(f"SonarQube API returned {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        issues = data.get("issues", [])
        if not issues:
            break
        all_issues.extend(issues)
        total = data.get("total", 0)
        if len(all_issues) >= total:
            break
        page += 1
        time.sleep(0.1)  # be polite between pages (§23.3)
    return all_issues
