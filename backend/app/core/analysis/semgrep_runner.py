"""Semgrep SAST runner with offline execution capability (§4.2.3).

Follows 2026 offline execution standards:
- Respects `SEMGREP_OFFLINE=1` to prevent network calls to registry during demos.
- Uses local ruleset cache or passes `--disable-version-check`.
- Emits SARIF/JSON parsed directly into standard `Finding` objects.
- Gracefully degrades to `builtin_analyzer` if Semgrep CLI is not installed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from app.models.finding import Finding
from app.utils.logger import get_logger

logger = get_logger(__name__)


def is_semgrep_available() -> bool:
    """Check if semgrep binary is on PATH."""
    return shutil.which("semgrep") is not None


def run_semgrep_scan(
    repo_path: str | Path,
    config: str | None = None,
    offline: bool = True,
    timeout: int = 120,
) -> list[Finding]:
    """Execute Semgrep scan in offline-safe mode and return standard Findings."""
    if not is_semgrep_available():
        logger.debug("Semgrep CLI not found on PATH — relying on built-in SAST analyzer")
        return []

    repo_path = Path(repo_path)
    env = os.environ.copy()
    if offline or os.getenv("SEMGREP_OFFLINE", "1") == "1":
        env["SEMGREP_OFFLINE"] = "1"

    # Default to cached local ruleset to ensure air-gapped demo runs
    if config is None:
        candidates = [
            Path(__file__).resolve().parents[4] / "config" / "semgrep" / "rules.yaml",
            Path(__file__).resolve().parents[4] / "config" / "semgrep",
            Path.cwd() / "config" / "semgrep" / "rules.yaml",
            Path.cwd() / "config" / "semgrep",
        ]
        found_config = None
        for cand in candidates:
            if cand.exists():
                found_config = str(cand)
                break
        rule_config = found_config or os.getenv("SEMGREP_CONFIG", "auto")
    else:
        rule_config = config

    cmd = [
        "semgrep",
        "scan",
        f"--config={rule_config}",
        "--json",
        "--quiet",
        "--disable-version-check",
        "--no-git-ignore",
        str(repo_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        if proc.returncode not in (0, 1):  # Semgrep returns 1 when findings are present
            logger.warning(f"Semgrep returned non-zero code {proc.returncode}: {proc.stderr[:200]}")
            return []

        data = json.loads(proc.stdout)
        results = data.get("results", [])

        findings: list[Finding] = []
        for r in results:
            path = r.get("path", "")
            try:
                rel_path = Path(path).relative_to(repo_path).as_posix()
            except ValueError:
                rel_path = path

            extra = r.get("extra", {})
            severity_raw = extra.get("severity", "WARNING").upper()
            sev_map = {"ERROR": "critical", "WARNING": "major", "INFO": "minor"}
            severity = sev_map.get(severity_raw, "major")

            rule_id = r.get("check_id", "semgrep")
            message = extra.get("message", "Semgrep security finding")
            line_start = r.get("start", {}).get("line", 1)
            line_end = r.get("end", {}).get("line", line_start)

            findings.append(Finding(
                id="",
                tool="semgrep",
                rule_id=rule_id,
                message=message,
                severity=severity,
                finding_type="vulnerability" if severity in ("critical", "major") else "code_smell",
                file_path=rel_path,
                line_start=line_start,
                line_end=line_end,
            ))

        logger.info(f"Semgrep scan completed: {len(findings)} findings in {repo_path}")
        return findings

    except subprocess.TimeoutExpired:
        logger.warning(f"Semgrep scan timed out after {timeout}s")
        return []
    except Exception as exc:
        logger.warning(f"Semgrep scan failed ({exc}) — falling back to built-in analyzer")
        return []

