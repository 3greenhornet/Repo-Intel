"""Component 5C: Query formulation (§8.3) — gaps → follow-up retrieval query.

Simple mode concatenates gaps (deterministic, no LLM required);
LLM mode asks the reasoning LLM to synthesize a sharper query.
"""
from __future__ import annotations

from app.utils.logger import get_logger

logger = get_logger(__name__)


def formulate_query(gaps: list[str], original_query: str, llm_client=None) -> str:
    """§8.3.2 formulate_query."""
    if not gaps:
        return original_query
    if llm_client is None:
        # simple approach (§8.3.2 first branch)
        if len(gaps) == 1:
            return gaps[0]
        return f"{original_query} — focusing on: {'; '.join(g[:80] for g in gaps[:3])}"

    prompt = (
        f"Original question: {original_query}\n"
        f"Missing information:\n" + "\n".join(gaps[:8]) +
        "\n\nWrite a single focused retrieval query (max 20 words) that would find "
        "the missing information in a code repository. Reply with ONLY the query."
    )
    try:
        result = llm_client.invoke(prompt, max_tokens=60)
        return result.strip().strip('"').splitlines()[0][:200]
    except Exception as exc:
        logger.warning(f"LLM query formulation failed ({exc}) — using simple mode")
        return f"{original_query} — focusing on: {'; '.join(g[:60] for g in gaps[:3])}"
