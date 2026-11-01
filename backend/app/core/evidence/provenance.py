"""Component 6C: Provenance attachment (§9.3).

Every piece of evidence gets source_file, symbol, lines, commit hash/author/
timestamp, and analyzer rule — the "every recommendation traceable" principle
(§1.3 core design table).
"""
from __future__ import annotations

from pathlib import Path

from app.models.evidence import Evidence, Provenance
from app.models.finding import Finding
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ProvenanceAttacher:
    """§9.3.2 attach_provenance — with git commit + rule lookup."""

    def __init__(self, repo_path: Path | None, findings: list[Finding] | None = None) -> None:
        self.repo_path = repo_path
        self._findings_by_file: dict[str, list[Finding]] = {}
        for f in (findings or []):
            self._findings_by_file.setdefault(f.file_path, []).append(f)
        self._commit_cache: dict[str, dict | None] = {}

    def attach(self, evidence: list[Evidence]) -> list[Evidence]:
        """Attach provenance to every evidence item (in place)."""
        for e in evidence:
            chunk = e.chunk
            commit = self._last_commit_for_file(chunk.file_path)
            rule = self._rule_for_chunk(chunk)
            e.provenance = Provenance(
                source_file=chunk.file_path,
                symbol_id=chunk.symbol_id,
                line_start=chunk.start_line,
                line_end=chunk.end_line,
                commit_hash=commit.get("hash") if commit else None,
                commit_author=commit.get("author") if commit else None,
                commit_timestamp=commit.get("timestamp") if commit else None,
                analyzer_rule=rule[0] if rule else None,
                analyzer_tool=rule[1] if rule else None,
                confidence=e.score if e.score <= 1 else min(e.score / 20.0, 1.0),
            )
        return evidence

    def _last_commit_for_file(self, file_path: str) -> dict | None:
        if file_path.startswith("issue:"):
            return None
        if file_path in self._commit_cache:
            return self._commit_cache[file_path]
        commit = None
        if self.repo_path and (self.repo_path / ".git").exists():
            try:
                from app.core.ingestion.git_parser import get_last_commit_for_file

                record = get_last_commit_for_file(self.repo_path, file_path)
                if record:
                    commit = {"hash": record.hash, "author": record.author,
                              "timestamp": record.timestamp.isoformat()}
            except Exception as exc:
                logger.debug(f"Provenance git lookup failed for {file_path}: {exc}")
        self._commit_cache[file_path] = commit
        return commit

    def _rule_for_chunk(self, chunk) -> tuple[str, str] | None:
        if chunk.chunk_type != "finding":
            return None
        candidates = self._findings_by_file.get(chunk.file_path, [])
        for f in candidates:
            if f.line_start == chunk.start_line and f.rule_id in chunk.content:
                return f.rule_id, f.tool
        if candidates:
            return candidates[0].rule_id, candidates[0].tool
        return None
