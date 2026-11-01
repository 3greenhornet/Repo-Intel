"""CodeQL runner (§4.2.3 Component C).

Creates a CodeQL database, runs the documented security query suite, emits
SARIF, and parses it via the safe parser. Skips gracefully when the CodeQL
CLI is absent (L7: license/availability).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from app.config import get_settings
from app.core.analysis.findings_parser import parse_sarif_safe
from app.models.finding import Finding
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CodeQLError(Exception):
    pass


QUERY_SUITE_BY_LANG = {  # FIX-035 Change C
    "python": "codeql/python-queries:codeql-suites/python-security-extended.qls",
    "java": "codeql/java-queries:codeql-suites/java-security-extended.qls",
    "javascript": "codeql/javascript-queries:codeql-suites/javascript-security-extended.qls",
    "typescript": "codeql/javascript-queries:codeql-suites/javascript-security-extended.qls",
    "c": "codeql/cpp-queries:codeql-suites/cpp-security-extended.qls",
    "cpp": "codeql/cpp-queries:codeql-suites/cpp-security-extended.qls",
}


def _detect_dominant_language(repo_path: Path) -> str | None:
    """FIX-035 Change B: infer the dominant language from file extensions."""
    counts: dict[str, int] = {}
    for ext in repo_path.rglob("*"):
        if not ext.is_file():
            continue
        suffix = ext.suffix.lower()
        if suffix in (".py",):
            counts["python"] = counts.get("python", 0) + 1
        elif suffix in (".java",):
            counts["java"] = counts.get("java", 0) + 1
        elif suffix in (".js", ".jsx"):
            counts["javascript"] = counts.get("javascript", 0) + 1
        elif suffix in (".ts", ".tsx"):
            counts["typescript"] = counts.get("typescript", 0) + 1
        elif suffix in (".c", ".h"):
            counts["c"] = counts.get("c", 0) + 1
        elif suffix in (".cpp", ".cc", ".hpp"):
            counts["cpp"] = counts.get("cpp", 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def run_codeql(repo_path: str | Path, repo_id: str,
               language: str | None = None) -> list[Finding]:
    """§4.2.3: database create → analyze (SARIF) → parse.

    FIX-035: ENABLE_CODEQL=1 with a missing CLI is a configuration error and
    now raises instead of silently returning []; the dominant language is
    detected from the ingested files; the query suite is per-language.
    """
    settings = get_settings()
    if not settings.enable_codeql:
        logger.info("CodeQL disabled (ENABLE_CODEQL=0) — built-in analyzer provides coverage")
        return []

    codeql = shutil.which(settings.codeql_path) or (
        Path(settings.codeql_path) if Path(settings.codeql_path).exists() else None
    )
    if codeql is None:
        raise CodeQLError(
            f"ENABLE_CODEQL=1 but CodeQL CLI not found at {settings.codeql_path!r}. "
            "Install from https://github.com/github/codeql-cli-binaries/releases "
            "or set CODEQL_PATH to a valid path.")

    repo_path = Path(repo_path)
    if language is None:
        language = _detect_dominant_language(repo_path)
        if language is None:
            logger.warning("No supported language files found — skipping CodeQL")
            return []

    suite = QUERY_SUITE_BY_LANG.get(language)
    if suite is None:
        logger.warning(f"No CodeQL query suite configured for {language}")
        return []

    workdir = settings.data_dir / "codeql" / repo_id
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = workdir / f"{repo_id}-db"
    sarif_out = workdir / "results.sarif"

    # Step 1: database create
    r = subprocess.run(
        [str(codeql), "database", "create", str(db_path),
         f"--language={language}", f"--source-root={repo_path}"],
        capture_output=True, text=True, timeout=1800,
    )
    if r.returncode != 0:
        raise CodeQLError(r.stderr[-500:])

    # Step 2: analyze
    r = subprocess.run(
        [str(codeql), "database", "analyze", str(db_path),
         suite,
         "--format=sarif-latest", f"--output={sarif_out}"],
        capture_output=True, text=True, timeout=3600,
    )
    if r.returncode != 0:
        raise CodeQLError(r.stderr[-500:])

    # Step 3: parse SARIF
    return parse_sarif_safe(sarif_out)
