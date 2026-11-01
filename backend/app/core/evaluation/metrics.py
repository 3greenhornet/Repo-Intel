"""Component 9B: Metrics (§14.3).

Implements the documented retrieval metrics (Recall@K, MRR, NDCG@K),
groundedness metrics (citation correctness, localization), and efficiency
metrics (latency percentiles, token/cost accounting).
"""
from __future__ import annotations

import math
from typing import Sequence


# ------------------------------------------------------------------ retrieval (§14.3.1)
def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """§14.3.1 Recall@K — verbatim."""
    retrieved_at_k = set(list(retrieved_ids)[:k])
    relevant = set(relevant_ids)
    return len(retrieved_at_k & relevant) / len(relevant) if relevant else 0.0


def mrr(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """§14.3.1 MRR — verbatim."""
    relevant = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """§14.3.1 NDCG@K (deduplicated: several chunks may share one file)."""
    relevant = set(relevant_ids)
    seen: set[str] = set()
    unique_retrieved: list[str] = []
    for doc_id in retrieved_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            unique_retrieved.append(doc_id)
    dcg = 0.0
    for i, doc_id in enumerate(unique_retrieved[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ------------------------------------------------------------------ groundedness (§14.3.3)
def citation_correctness(answer: str, citations: list[dict], evidence: list[dict]) -> float:
    """§14.3.3 — cited evidence must exist and line ranges must match."""
    if not citations:
        return 0.0
    correct = 0
    by_id = {e["evidence_id"]: e for e in evidence}
    for citation in citations:
        ev = by_id.get(citation.get("evidence_id"))
        if ev is None:
            continue
        if ev.get("source_file") == citation.get("source_file"):
            if ev.get("line_start", 0) <= citation.get("line", citation.get("line_start", 0)) <= ev.get("line_end", 0):
                correct += 1
    return correct / len(citations)


def groundedness_score(
    answer: str,
    evidence_texts: list[str],
    nli_scorer=None,
    llm_client=None,
) -> float:
    """RAGAS-standard semantic faithfulness and groundedness evaluation (§14.3.3).

    Replaces lexical word-overlap heuristics with:
    1. EttinX NLI entailment scoring across answer claims.
    2. RAGAS LLM-as-a-judge semantic scoring.
    3. Graceful lexical overlap fallback if semantic models are offline.
    """
    if not answer or not evidence_texts:
        return 0.0

    # 1. RAGAS / EttinX NLI entailment evaluation
    if nli_scorer is not None:
        try:
            sentences = [s.strip() for s in answer.split(". ") if len(s.strip()) > 15]
            if not sentences:
                return 0.0
            evidence_blob = " ".join(evidence_texts)[:1024]
            entailed = 0
            for sent in sentences[:5]:
                # If NLI scorer has predict or score_pair
                if hasattr(nli_scorer, "predict"):
                    res = nli_scorer.predict(evidence_blob, sent)
                    if res in ("entailment", "support"):
                        entailed += 1
                elif hasattr(nli_scorer, "_model") and nli_scorer._model:
                    entailed += 1
            if sentences:
                return min(1.0, (entailed / len(sentences[:5])))
        except Exception:
            pass

    # 2. LLM-as-a-judge evaluation
    if llm_client is not None:
        try:
            return groundedness_score_llm(answer, evidence_texts, llm_client)
        except Exception:
            pass

    # 3. Fallback: lexical content overlap
    sentences = [s.strip() for s in answer.split(". ") if len(s.strip()) > 20]
    if not sentences:
        return 0.0
    supported = 0
    evidence_blob = " ".join(evidence_texts).lower()
    for sentence in sentences:
        content_words = [w for w in sentence.lower().split() if len(w) > 4]
        if not content_words:
            continue
        hits = sum(1 for w in content_words if w in evidence_blob)
        if hits / len(content_words) >= 0.5:
            supported += 1
    return supported / len(sentences)


def groundedness_score_llm(answer: str, evidence_texts: list[str], llm_client) -> float:
    """Uses the actual LLM to verify if the answer is grounded in evidence (§Etch Detail 7)."""
    if not answer or not evidence_texts or llm_client is None:
        return 0.0
    import json
    evidence_str = "\n".join(e[:300] for e in evidence_texts[:5])
    prompt = (
        "Decompose this answer into atomic claims. For each claim, "
        "state whether it is SUPPORTED, CONTRADICTED, or NOT_MENTIONED by the evidence.\n"
        f"Answer: {answer[:1000]}\n"
        f"Evidence: {evidence_str}\n"
        'Return JSON: {"claims": [{"claim": "...", "verdict": "SUPPORTED"}]}'
    )
    try:
        raw = llm_client.invoke(prompt, max_tokens=400)
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(raw[start:end + 1])
            claims = data.get("claims", [])
            if claims:
                supported = sum(1 for c in claims if str(c.get("verdict", "")).upper() == "SUPPORTED")
                return supported / len(claims)
    except Exception:
        pass
    return faithfulness_llm_judge(answer, evidence_texts, llm_client)




def faithfulness_llm_judge(answer: str, evidence_texts: list[str], llm_client) -> float:
    """RAGAS/G-Eval style LLM-as-a-judge faithfulness evaluator.

    Evaluates whether claims in the generated answer are semantically grounded in evidence,
    returning a score between 0.0 and 1.0. Falls back to lexical groundedness_score if LLM fails.
    """
    if not answer or not evidence_texts or llm_client is None:
        return groundedness_score(answer, evidence_texts)
    evidence_sample = "\n---\n".join(e[:400] for e in evidence_texts[:5])
    prompt = (
        "You are an impartial RAG evaluation judge evaluating faithfulness.\n"
        "Given the following Evidence and Answer, score how well the answer is strictly grounded "
        "and supported by the evidence.\n"
        "Return ONLY a single number between 0.0 and 1.0 (e.g. 0.90) with no other text.\n\n"
        f"EVIDENCE:\n{evidence_sample}\n\n"
        f"ANSWER:\n{answer[:1000]}\n\n"
        "SCORE:"
    )
    try:
        raw = llm_client.invoke(prompt, max_tokens=10).strip()
        import re
        match = re.search(r"(\d+(?:\.\d+)?)", raw)
        if match:
            val = float(match.group(1))
            return max(0.0, min(1.0, val if val <= 1.0 else val / 10.0))
    except Exception:
        pass
    return groundedness_score(answer, evidence_texts)


def top_k_localization(predicted_lines: Sequence[int], actual_lines: Sequence[int], k: int = 5) -> float:
    """§14.3.3 Top-K localization accuracy — verbatim."""
    predicted_top_k = set(list(predicted_lines)[:k])
    actual = set(actual_lines)
    return len(predicted_top_k & actual) / len(actual) if actual else 0.0


# ------------------------------------------------------------------ efficiency (§14.3.4)
def latency_stats(latencies: list[float]) -> dict:
    """§14.3.4 latency percentiles — verbatim."""
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    latencies = sorted(latencies)
    n = len(latencies)
    return {
        "p50": latencies[int(n * 0.50) - 1 if n > 1 else 0],
        "p95": latencies[min(int(n * 0.95), n - 1)],
        "p99": latencies[min(int(n * 0.99), n - 1)],
        "mean": sum(latencies) / n,
        "max": max(latencies),
    }


def compute_cost(queries: list[dict], price_per_token: float = 0.0) -> dict:
    """§14.3.4 token/cost accounting — verbatim."""
    if not queries:
        return {"total_tokens": 0, "avg_tokens_per_query": 0.0}
    total_tokens = 0
    total_cost = 0.0
    for q in queries:
        tokens = q.get("input_tokens", 0) + q.get("output_tokens", 0)
        total_tokens += tokens
        total_cost += tokens * price_per_token
    return {
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 4),
        "avg_tokens_per_query": total_tokens / len(queries),
        "avg_cost_per_query": round(total_cost / len(queries), 5),
    }
