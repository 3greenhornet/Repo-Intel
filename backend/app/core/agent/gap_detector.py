"""Component 5B: Gap detection (§8.2) + 5A sufficiency checks (§8.1)."""
from __future__ import annotations

import re

from app.models.evidence import Evidence
from app.utils.logger import get_logger

logger = get_logger(__name__)

MULTI_HOP_KEYWORDS = ["why", "impact", "break", "affect", "depend", "caller", "called by"]
STOPWORDS = {
    "the", "is", "are", "what", "who", "does", "this", "that", "how", "why",
    "and", "or", "in", "on", "of", "to", "a", "an", "with", "for", "when",
    "where", "which", "do", "can", "should", "would", "my", "our", "it",
}


def is_multi_hop_query(query: str) -> bool:
    """§8.1.2 is_multi_hop_query."""
    return any(kw in query.lower() for kw in MULTI_HOP_KEYWORDS)


def classify_query_complexity(query: str, llm_client=None) -> tuple[str, str]:
    """Adaptive RAG: classify query complexity into 'simple' vs 'complex'.

    Routes direct questions to single-pass retrieval, and multi-module/architectural
    questions to the iterative agentic exploration loop.
    Returns: (complexity: 'simple' | 'complex', reason: str)
    """
    if llm_client is not None:
        prompt = (
            "You are a query router in a code repository intelligence system.\n"
            "Classify the following query into exactly one category:\n"
            "- SIMPLE: Direct question about a single function, file, definition, or variable.\n"
            "- COMPLEX: Architectural flow, cross-module trace, dependency analysis, or multi-step logic.\n\n"
            f"QUERY: \"{query}\"\n\n"
            "Respond in format: CATEGORY; brief reason\n"
            "Example: SIMPLE; asks about a single function signature\n"
            "RESPONSE:"
        )
        try:
            resp = llm_client.invoke(prompt, max_tokens=30).strip()
            parts = resp.split(";", 1)
            kind = parts[0].strip().upper()
            reason = parts[1].strip() if len(parts) > 1 else resp
            if "COMPLEX" in kind:
                return "complex", reason
            elif "SIMPLE" in kind:
                return "simple", reason
        except Exception:
            pass

    # High-precision heuristic fallback
    complex_triggers = ["architecture", "flow", "trace", "lifecycle", "pipeline", "interact", "sequence"]
    if is_multi_hop_query(query) or any(t in query.lower() for t in complex_triggers) or len(query.split()) > 12:
        return "complex", "Multi-hop query or architectural keywords detected"
    return "simple", "Direct single-entity query"


def check_sufficiency(query: str, evidence: list[Evidence], threshold: float = 0.7) -> tuple[bool, float, str]:
    """§8.1.2 check_sufficiency — (is_sufficient, confidence, reason)."""
    if not evidence:
        return False, 0.0, "No evidence retrieved"

    top_score = max(e.score for e in evidence)
    unique_sources = len({e.chunk.file_path for e in evidence[:5]})
    unique_symbols = len({e.chunk.symbol_id for e in evidence if e.chunk.symbol_id})
    is_multihop = is_multi_hop_query(query)

    if is_multihop:
        sufficient = unique_symbols >= 3
        reason = f"Multi-hop query. Found {unique_symbols} unique symbols across {unique_sources} files."
    else:
        sufficient = (top_score >= threshold or unique_sources >= 2) and len(evidence) >= 3
        reason = f"Top score {top_score:.2f}, {len(evidence)} pieces of evidence, {unique_sources} unique files."
    confidence = min(1.0, top_score if top_score <= 1 else top_score / 30)
    return sufficient, confidence, reason


def check_sufficiency_llm(query: str, evidence: list[Evidence],
                          llm_client) -> tuple[bool, float, str]:
    """§8.1.3 / FIX-037: LLM-based sufficiency check.

    More accurate than the keyword heuristic for subtle insufficiency
    (named symbols whose cross-module dependencies were not retrieved).
    Falls back to the heuristic when the LLM call fails.
    """
    if not evidence:
        return False, 0.0, "No evidence retrieved"

    from app.core.reasoning.prompt_templates import build_sufficiency_prompt

    summary = "\n".join(
        f"- [{i + 1}] {e.chunk.file_path}:{e.chunk.start_line} "
        f"({e.chunk.chunk_type}) — {e.chunk.content[:120]}"
        for i, e in enumerate(evidence[:5])
    )
    prompt = build_sufficiency_prompt(query, summary)
    try:
        response = llm_client.invoke(prompt, max_tokens=120)
        upper = response.strip().upper()
        sufficient = upper.startswith("YES")
        confidence = 0.8 if sufficient else 0.3
        return sufficient, confidence, response[:200]
    except Exception as exc:
        logger.warning(f"LLM sufficiency failed ({exc}) — falling back to heuristic")
        return check_sufficiency(query, evidence)


def extract_terms(query: str) -> list[str]:
    """Key query terms (stopword-filtered, code-aware)."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", query)
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def identify_gaps(query: str, evidence: list[Evidence]) -> list[str]:
    """§8.2.2 identify_gaps — term coverage + relationship + context gaps."""
    if not evidence:
        return ["No evidence retrieved at all"]

    gaps: list[str] = []
    query_terms = extract_terms(query)
    for term in query_terms:
        if not any(term in e.chunk.content.lower() for e in evidence):
            gaps.append(f"Missing information about: {term}")

    if is_multi_hop_query(query):
        symbols_found = {e.chunk.symbol_name for e in evidence if e.chunk.symbol_id}
        for symbol in list(symbols_found)[:5]:
            caller_info = any(e.chunk.metadata.get("relation") in ("calls", "impacted_by")
                              for e in evidence)
            callee_info = any(e.chunk.metadata.get("relation") in ("called_by", "calls")
                              for e in evidence)
            if not caller_info:
                gaps.append(f"Missing callers of: {symbol}")
            if not callee_info:
                gaps.append(f"Missing callees of: {symbol}")

    top_file = evidence[0].chunk.file_path
    file_covered = sum(1 for e in evidence if e.chunk.file_path == top_file)
    if file_covered == 1:
        gaps.append(f"Missing surrounding context for: {top_file}")
    return gaps
