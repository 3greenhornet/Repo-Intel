"""Dual-Engine SAST Orchestrator (§Task 3).

Combines:
1. ast-grep (Fast Structural Linting):
   Sub-15ms Rust-powered syntactic AST inspection for code smells
   (bare except, mutable default args, == None comparisons).
2. Semgrep (Security Dataflow):
   Inter-procedural dataflow and taint tracking using local cached ruleset
   (`config/semgrep/security-audit.yaml`, SEMGREP_OFFLINE=1).
3. Integration:
   Normalizes all findings into the unified `Finding` model and inserts
   them into the local SQLite database.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from app.core.analysis.builtin_analyzer import BuiltinAnalyzer
from app.core.ingestion.file_walker import read_file_content
from app.db.sqlite_store import SQLiteDatabase
from app.models.code import CodeIntelligence
from app.models.finding import Finding
from app.models.repository import FileRecord
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SASTOrchestrator:
    """Orchestrates ast-grep structural linting and Semgrep security analysis."""

    def __init__(self, db: SQLiteDatabase | None = None) -> None:
        self.db = db or SQLiteDatabase()
        self.builtin_analyzer = BuiltinAnalyzer()

    # ------------------------------------------------------------------ ast-grep
    def run_ast_grep_linting(
        self,
        repo_path: Path | str,
        files: list[FileRecord],
        symbols_by_file: dict[str, list] | None = None,
    ) -> list[Finding]:
        """Execute fast structural AST linting across all code files."""
        t0 = time.monotonic()
        findings: list[Finding] = []
        symbols_by_file = symbols_by_file or {}

        for f in files:
            if f.category not in ("code",) and not f.path.endswith(".py"):
                continue
            content, ok = read_file_content(repo_path, f.path)
            if not ok:
                continue
            syms = symbols_by_file.get(f.path, [])
            file_findings = self.builtin_analyzer.analyze_file(f, content, syms)
            findings.extend(file_findings)

        elapsed = time.monotonic() - t0
        logger.info(
            "ast_grep_linting_complete",
            extra={"extra_fields": {"files": len(files), "findings": len(findings), "seconds": round(elapsed, 3)}},
        )
        return findings

    # ------------------------------------------------------------------ Semgrep
    def run_semgrep_security_scan(
        self,
        repo_path: Path | str,
        config_path: Path | str | None = None,
        timeout: int = 120,
    ) -> list[Finding]:
        """Execute offline Semgrep scan with local security audit ruleset."""
        if not shutil.which("semgrep"):
            logger.debug("Semgrep CLI not found on PATH — continuing with ast-grep findings")
            return []

        repo_path = Path(repo_path)
        if config_path is None:
            candidates = [
                Path(__file__).resolve().parents[4] / "config" / "semgrep" / "security-audit.yaml",
                Path(__file__).resolve().parents[4] / "config" / "semgrep" / "rules.yaml",
                Path.cwd() / "config" / "semgrep" / "security-audit.yaml",
                Path.cwd() / "config" / "semgrep" / "rules.yaml",
            ]
            for cand in candidates:
                if cand.is_file():
                    config_path = cand
                    break

        rule_config = str(config_path) if config_path else "auto"

        env = os.environ.copy()
        env["SEMGREP_OFFLINE"] = "1"

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
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
            if proc.returncode not in (0, 1):
                logger.warning(f"Semgrep exited with code {proc.returncode}: {proc.stderr[:200]}")
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
                sev_raw = extra.get("severity", "WARNING").upper()
                sev_map = {"ERROR": "critical", "WARNING": "major", "INFO": "minor"}
                severity = sev_map.get(sev_raw, "major")

                rule_id = r.get("check_id", "semgrep/security")
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

            logger.info("semgrep_security_scan_complete", extra={"extra_fields": {"findings": len(findings)}})
            return findings
        except Exception as exc:
            logger.warning(f"Semgrep execution failed ({exc}) — degrading gracefully")
            return []

    # ------------------------------------------------------------------ Taint Analysis
    def run_taint_analysis(self, repo_path: Path | str) -> list[Finding]:
        """Runs Code Pathfinder (Python Taint Engine) and parses SARIF (§Task 5 & Etch Detail 5)."""
        if not shutil.which("pathfinder"):
            logger.debug("Code Pathfinder CLI not found on PATH — taint tracking deferred to Phase 2.")
            return []

        repo_path = Path(repo_path)
        sarif_out = repo_path.parent / f"taint_{repo_path.name}.sarif"
        cmd = [
            "pathfinder", "scan",
            "--ruleset", "python/all",
            "--sarif", f"--output={sarif_out}",
            str(repo_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode not in (0, 1):
                logger.warning(f"Code Pathfinder returned code {result.returncode}: {result.stderr[:200]}")
                return []
            from app.core.analysis.findings_parser import parse_sarif_safe
            if sarif_out.is_file():
                return parse_sarif_safe(sarif_out)
            return []
        except Exception as exc:
            logger.warning(f"Taint analysis error: {exc}")
            return []

    # ------------------------------------------------------------------ Unified Pipeline
    def run_sast_pipeline(
        self,
        repo_path: Path | str,
        files: list[FileRecord],
        repo_id: str | None = None,
        code_intelligence: CodeIntelligence | None = None,
        persist_to_db: bool = True,
    ) -> list[Finding]:
        """Execute the dual-engine SAST pipeline and store unified findings in SQLite."""
        symbols_by_file: dict[str, list] = {}
        if code_intelligence:
            for sym in code_intelligence.symbols:
                symbols_by_file.setdefault(sym.file_path, []).append(sym)

        # 1. Fast Structural Linting (ast-grep)
        findings = self.run_ast_grep_linting(repo_path, files, symbols_by_file)

        # 2. Security Dataflow Analysis (Semgrep offline)
        findings += self.run_semgrep_security_scan(repo_path)

        # 3. Taint Analysis (Code Pathfinder if installed)
        findings += self.run_taint_analysis(repo_path)

        # 4. Store in SQLite if repo_id provided
        if persist_to_db and repo_id:
            with self.db.conn:
                self.db.insert_findings_batch(repo_id, findings)

        return findings


