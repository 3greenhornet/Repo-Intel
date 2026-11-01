"""Token counting utilities (used by budgeting everywhere)."""
from __future__ import annotations


def count_tokens(text: str) -> int:
    """Approximate token count (~4 chars/token). Deterministic and fast.

    A tiktoken-based implementation can be swapped in when the exact
    tokenizer matters (component-swap principle, §1.3).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def count_tokens_batch(texts: list[str]) -> int:
    return sum(count_tokens(t) for t in texts)
