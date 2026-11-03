"""Components 6A + 6D + 6E: selection, budgeting, assembly (§9.1, §9.4, §9.5).

The EvidenceComponent (§3.2.7): select / score_nli / attach_provenance /
build_context.
"""
from __future__ import annotations

from typing import Any
from pathlib import Path

from app.models.evidence import ContextBudget, Evidence, NLIRelation, Provenance
from app.utils.logger import get_logger
from app.utils.token_counter import count_tokens

logger = get_logger(__name__)

PROMPT_OVERHEAD = 400  # §9.4.2 constant for prompt scaffolding


def _fetch_snippet_for_chunk(chunk: Any) -> str:
    """Fetch ±3 surrounding source lines for finding chunks lacking inline code context."""
    try:
        repo_id = getattr(chunk, "repo_id", "")
        file_path = getattr(chunk, "file_path", "")
        if not file_path:
            return ""

        candidates = [
            Path("data/repos") / repo_id / file_path,
            Path("data/repos") / file_path,
            Path(file_path),
        ]
        base_repos = Path("data/repos")
        if base_repos.exists():
            for sub in base_repos.iterdir():
                if sub.is_dir():
                    cand = sub / file_path
                    if cand.exists():
                        candidates.append(cand)
                        break

        for path in candidates:
            if path.exists() and path.is_file():
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
                start_line = getattr(chunk, "start_line", 1) or 1
                end_line = getattr(chunk, "end_line", start_line) or start_line
                start = max(0, start_line - 4)
                end = min(len(lines), end_line + 3)
                snippet_lines = [f"{i+1:4d} | {lines[i]}" for i in range(start, end)]
                if snippet_lines:
                    return f"\n\nContext source code ({file_path}:{start+1}-{end}):\n" + "\n".join(snippet_lines)
    except Exception:
        pass
    return ""


def select_evidence(evidence: list[Evidence], token_budget: int = 8000) -> list[Evidence]:
    """§9.1.2 select_evidence — top-scoring evidence within budget."""
    ranked = sorted(evidence, key=lambda e: e.score, reverse=True)
    selected: list[Evidence] = []
    total = 0
    for e in ranked:
        chunk_tokens = count_tokens(e.chunk.content)
        if total + chunk_tokens > token_budget:
            break
        selected.append(e)
        total += chunk_tokens
    if len(selected) < len(ranked):
        logger.info("evidence_selection_truncated", extra={"extra_fields": {
            "selected": len(selected), "available": len(ranked)}})
    return selected


def budget_context(
    evidence: list[Evidence], llm_context_window: int = 32768,
    reserved_for_response: int = 4096,
) -> ContextBudget:
    """§9.4.2 budget_context — fit evidence + provenance into the window."""
    available = llm_context_window - reserved_for_response - PROMPT_OVERHEAD
    selected: list[Evidence] = []
    used = 0
    for e in evidence:
        tokens = count_tokens(e.chunk.content) + _provenance_tokens(e.provenance)
        if used + tokens > available:
            break
        selected.append(e)
        used += tokens
    return ContextBudget(
        evidence=selected, tokens_used=used, tokens_available=available,
        truncated=len(selected) < len(evidence))


def _provenance_tokens(p: Provenance | None) -> int:
    if p is None:
        return 0
    return count_tokens(f"{p.source_file}{p.symbol_id or ''}{p.commit_hash or ''}{p.analyzer_rule or ''}")


def assemble_context(evidence: list[Evidence], nli_relations: list[NLIRelation] | None = None) -> str:
    """§9.5.2 assemble_context — the documented LLM context format."""
    parts: list[str] = ["=== EVIDENCE ===\n"]
    id_to_index = {e.chunk.id: i for i, e in enumerate(evidence, start=1)}

    for i, e in enumerate(evidence, start=1):
        parts.append(f"[Evidence {i}]")
        p = e.provenance
        if p:
            parts.append(f"File: {p.source_file}")
            if p.symbol_id:
                parts.append(f"Symbol: {p.symbol_id}")
            parts.append(f"Lines: {p.line_start}-{p.line_end}")
            if p.commit_hash:
                parts.append(f"Commit: {p.commit_hash[:12]} by {p.commit_author} "
                             f"at {p.commit_timestamp}")
            if p.analyzer_rule:
                parts.append(f"Rule: {p.analyzer_tool}:{p.analyzer_rule}")
        else:
            parts.append(f"File: {e.chunk.file_path}")
            parts.append(f"Lines: {e.chunk.start_line}-{e.chunk.end_line}")
            if e.chunk.symbol_name:
                parts.append(f"Symbol: {e.chunk.symbol_name}")
        # FIX-033: use the cached, stably-truncated prompt fragment so
        # repeated queries produce identical prefixes (provider prompt cache).
        fragment = e.chunk.prompt_fragment()
        if getattr(e.chunk, "chunk_type", "") == "finding" and "Context source code" not in fragment:
            code_snippet = _fetch_snippet_for_chunk(e.chunk)
            if code_snippet:
                fragment = f"{fragment}{code_snippet}"
        parts.append(f"Content:\n```\n{fragment}\n```")
        parts.append("")

    if nli_relations:
        parts.append("=== EVIDENCE RELATIONS ===\n")
        for r in nli_relations:
            a = id_to_index.get(r.evidence_a, r.evidence_a[:8])
            b = id_to_index.get(r.evidence_b, r.evidence_b[:8])
            parts.append(f"Evidence {a} {r.relation.upper()} Evidence {b} "
                         f"(confidence: {r.confidence:.2f})")
        parts.append("")
    return "\n".join(parts)
