"""Component 8: Developer output (§11) — Q&A, recommendations, explanations.

Formats the LLM response into the three documented output types, parses
[ Evidence N ] citations (§11.3), computes recommendation priority
(§11.2 compute_priority), and generates follow-ups.
"""
from __future__ import annotations

import re

from app.models.evidence import Evidence
from app.models.finding import Finding
from app.models.query import Citation
from app.models.response import DeveloperOutput, ExplanationOutput, Recommendation
from app.utils.logger import get_logger

logger = get_logger(__name__)

SEVERITY_WEIGHT = {"blocker": 1.0, "critical": 0.8, "major": 0.5, "minor": 0.2}  # §11.2
# FIX-004: block-level citation matcher. Captures the full body of a
# [Evidence ...] block so every number inside it can be extracted, including
# comma lists ([Evidence 1, 2, 3]), "and" separators ([Evidence 1 and 2])
# and ranges ([Evidence 1-3, 5]).
CITATION_BLOCK_RE = re.compile(
    r"\[\s*Evidence\s+([\d,\-–\s]+?(?:and\s+\d+)?)\s*\]",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\d+")


def _citation_numbers(answer: str) -> list[int]:
    """Extract ALL evidence indices cited, incl. ranges & comma lists (FIX-004).

    Handles: [Evidence 1], [Evidence 1, 2], [Evidence 1, 2, 3],
    [Evidence 1 and 2], [Evidence 1-3], [Evidence 1-3, 5].
    """
    nums: list[int] = []
    for block in CITATION_BLOCK_RE.finditer(answer):
        body = block.group(1)
        for token in re.split(r",\s+and\s+|\s+and\s+|\s*,\s*", body):
            token = token.strip()
            if not token:
                continue
            rng = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", token)
            if rng:
                start, end = int(rng.group(1)), int(rng.group(2))
                if end < start:
                    start, end = end, start
                for n in range(start, end + 1):
                    if n not in nums:
                        nums.append(n)
            else:
                for n in NUMBER_RE.findall(token):
                    n_int = int(n)
                    if n_int not in nums:
                        nums.append(n_int)
    return nums


def compute_priority(finding: Finding, impact: list) -> float:
    """§11.2 compute_priority — severity weight × impact amplification."""
    severity_weight = SEVERITY_WEIGHT.get(finding.severity or "", 0.1)
    impact_weight = len(impact) / 10.0
    return severity_weight * (1 + min(impact_weight, 1.5))


def extract_citations(answer: str, evidence: list[Evidence]) -> list[Citation]:
    """§11.3 parse [Evidence N] references into citation objects."""
    citations: list[Citation] = []
    for n in _citation_numbers(answer):
        if n <= len(evidence):
            e = evidence[n - 1]
            p = e.provenance
            snippet = e.chunk.content.strip().splitlines()
            citations.append(Citation(
                evidence_id=e.chunk.id,
                source_file=p.source_file if p else e.chunk.file_path,
                line_start=(p.line_start if p else e.chunk.start_line) or 0,
                line_end=(p.line_end if p else e.chunk.end_line) or 0,
                symbol_id=(p.symbol_id if p else e.chunk.symbol_id),
                commit_hash=(p.commit_hash if p else None),
                snippet=snippet[0][:160] if snippet else "",
            ))
    return citations


def format_qa(question: str, answer: str, evidence: list[Evidence],
              confidence: float, follow_ups: list[str] | None = None) -> DeveloperOutput:
    """§11.1 Repository Q&A output."""
    return DeveloperOutput(
        output_type="qa",
        question=question,
        answer=answer,
        citations=extract_citations(answer, evidence),
        confidence=confidence,
        follow_up_questions=follow_ups or [],
        metadata={"evidence_count": len(evidence)},
    )


def format_recommendation(
    finding: Finding, impact: list[str], explanation: str, evidence: list[Evidence],
) -> Recommendation:
    """§11.2 explainable recommendation."""
    return Recommendation(
        finding_id=finding.id,
        severity=finding.severity or "info",
        impact=impact[:10],
        explanation=explanation,
        fix_suggestion=_extract_fix(explanation),
        citations=extract_citations(explanation, evidence),
        priority=compute_priority(finding, impact),
    )


def format_evidence_linked(answer: str, evidence: list[Evidence]) -> ExplanationOutput:
    """§11.3 evidence-linked explanation with provenance list."""
    citations = extract_citations(answer, evidence)
    provenance = []
    for e in evidence:
        if e.provenance:
            provenance.append({
                "source_file": e.provenance.source_file,
                "lines": f"{e.provenance.line_start}-{e.provenance.line_end}",
                "commit": (e.provenance.commit_hash or "")[:12],
                "rule": e.provenance.analyzer_rule,
            })
    return ExplanationOutput(
        text=answer, citations=citations, provenance=provenance,
        traceable=bool(citations))


def _extract_fix(explanation: str) -> str:
    """Pull the fix suggestion from an LLM recommendation."""
    for marker in ("Fix:", "Recommendation:", "Suggested fix:"):
        if marker in explanation:
            return explanation.split(marker, 1)[1].strip().splitlines()[0][:300]
    lines = [l for l in explanation.splitlines() if l.strip()]
    return lines[-1][:300] if lines else ""


def generate_followups(llm_client, answer: str, question: str) -> list[str]:
    """§11.1 follow-up question generation via LLM."""
    from app.core.reasoning.prompt_templates import build_followup_prompt

    try:
        raw = llm_client.invoke(build_followup_prompt(answer, question), max_tokens=150)
        lines = [l.strip().lstrip("-•0123456789. ").strip() for l in raw.splitlines()]
        return [l for l in lines if 10 < len(l) < 120][:3]
    except Exception as exc:
        logger.debug(f"Follow-up generation skipped: {exc}")
        return []
