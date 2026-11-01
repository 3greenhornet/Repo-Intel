"""Component 4D: Retrieval Fusion — RRF (§7.4 / §23.6 corrected algorithm).

Handles ranked lists of UNEQUAL lengths (G6 fix): rank starts at 1 per list,
a document found by multiple retrievers accumulates score, no normalization.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from app.models.evidence import Evidence
from app.utils.logger import get_logger

logger = get_logger(__name__)

RRF_K = 60  # §7.4.2 standard constant


def reciprocal_rank_fusion(ranked_lists: list[list[Evidence]], k: int = RRF_K) -> list[Evidence]:
    """§23.6 RRF with unequal list lengths — verbatim corrected algorithm."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, "Evidence.chunk.__class__"] = {}  # type: ignore[name-defined]

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        for rank, evidence in enumerate(ranked_list, start=1):
            chunk_id = evidence.chunk.id
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = evidence.chunk
                scores[chunk_id] = 0.0
            scores[chunk_id] += 1.0 / (k + rank)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused: list[Evidence] = []
    for i, chunk_id in enumerate(sorted_ids, start=1):
        fused.append(Evidence(
            chunk=chunk_map[chunk_id], score=scores[chunk_id], source="fused", rank=i))
    return fused


def hybrid_retrieve(
    lexical_search_fn, dense_search_fn, graph_search_fn, query: str, top_k: int = 20,
    parallel: bool = True, max_workers: int = 3,
) -> tuple[list[Evidence], dict[str, int]]:
    """§7.4.3 hybrid_retrieve — parallel retrieval + RRF fusion.

    Returns (fused_evidence, per-retriever counts). Per §7.4.4 error handling,
    failed retrievers are dropped and the remaining lists still fuse.
    """
    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            lexical_future = executor.submit(_safe, lexical_search_fn, query, 50)
            dense_future = executor.submit(_safe, dense_search_fn, query, 50)
            graph_future = executor.submit(_safe, graph_search_fn, query, 50)
            results = [f.result() for f in (lexical_future, dense_future, graph_future)]
    else:
        results = [
            _safe(lexical_search_fn, query, 50),
            _safe(dense_search_fn, query, 50),
            _safe(graph_search_fn, query, 50),
        ]

    counts = {"lexical": len(results[0]), "dense": len(results[1]), "graph": len(results[2])}
    active = [r for r in results if r]
    if not active:
        logger.warning("All retrievers returned empty (§7.4.4)")
        return [], counts

    fused = reciprocal_rank_fusion(active)
    logger.info("hybrid_retrieve_complete", extra={"extra_fields": {
        "query": query[:80], **counts, "fused": len(fused),
    }})
    return fused[:top_k * 2], counts  # keep extra head-room for reranking


def _safe(fn, query, k):
    """§7.4.4: one retriever failing must not kill the others."""
    try:
        return fn(query, k)
    except Exception as exc:
        logger.warning(f"Retriever {getattr(fn, '__name__', fn)} failed: {exc}")
        return []
