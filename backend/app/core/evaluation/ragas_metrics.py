"""RAGAS-based answer-quality metrics (§14.3.2) + LLM-as-a-Judge (2026 correction).

FIX-034: activates when RAGAS is installed and an LLM judge is available
(OpenAI-compatible endpoint). Falls back to the heuristic in metrics.py
otherwise.

2026 correction: LLM-as-a-Judge is now standard practice (studies show ~96%
alignment with human annotators for Faithfulness). `llm_judge_faithfulness`
uses the configured provider as the judge and complements RAGAS with a
direct claim-vs-evidence check that also works with non-OpenAI providers
(the z-ai bridge included).
"""
from __future__ import annotations

import re

from app.utils.logger import get_logger

logger = get_logger(__name__)


def ragas_available() -> bool:
    try:
        import ragas  # noqa: F401

        return True
    except ImportError:
        return False


def evaluate_rag_quality(questions, answers, contexts, ground_truths) -> dict:
    """Run RAGAS over a batch. Returns a dict of metric → mean score."""
    if not ragas_available():
        logger.warning("RAGAS not installed — returning empty metrics")
        return {}
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy, context_precision, context_recall, faithfulness,
    )

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return {k: float(v) for k, v in result.items()}


# ----------------------------------------------------------- LLM-as-a-Judge
_JUDGE_PROMPT = """You are an evaluation judge for a retrieval-augmented code
intelligence system. Given a QUESTION, the ANSWER produced by the system, and
the EVIDENCE chunks it retrieved, score the answer on two dimensions:

1. faithfulness: are all claims in the answer supported by the evidence?
   (1 = fully supported, 0 = contradicted or fabricated)
2. answer_relevancy: does the answer address the question? (1 = directly, 0 = not at all)

Reply with EXACTLY one line in the format:
FAITHFULNESS=<0.0-1.0> RELEVANCY=<0.0-1.0>

QUESTION: {question}

ANSWER: {answer}

EVIDENCE:
{evidence}
"""


def _parse_judge_response(response: str) -> tuple[float | None, float | None]:
    match = re.search(
        r"FAITHFULNESS\s*=\s*([0-9.]+)\s+RELEVANCY\s*=\s*([0-9.]+)",
        response or "", re.IGNORECASE)
    if not match:
        return None, None
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return None, None


def llm_judge_scores(llm_client, question: str, answer: str,
                     evidence_texts: list[str]) -> dict:
    """2026 correction: LLM-as-a-Judge faithfulness + relevancy.

    Works with any provider behind LLMClient (incl. the z-ai bridge).
    Returns {} when the judge call fails or returns an unparseable verdict.
    """
    if not answer or not evidence_texts:
        return {}
    evidence = "\n".join(f"[{i + 1}] {t[:1500]}" for i, t in enumerate(evidence_texts[:8]))
    prompt = _JUDGE_PROMPT.format(question=question[:500], answer=answer[:3000],
                                  evidence=evidence[:12000])
    try:
        response = llm_client.invoke(prompt, max_tokens=60)
        faith, relev = _parse_judge_response(response)
        if faith is None:
            logger.warning("LLM judge returned an unparseable verdict")
            return {}
        return {"llm_judge_faithfulness": faith, "llm_judge_relevancy": relev}
    except Exception as exc:
        logger.warning(f"LLM judge failed: {exc}")
        return {}
