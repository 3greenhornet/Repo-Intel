"""Component 4B: Dense retrieval (§7.2) — embedding similarity via VectorIndexer."""
from __future__ import annotations

from app.models.evidence import Evidence


def dense_search(vector_indexer, repo_id: str, query: str, top_k: int = 20) -> list[Evidence]:
    """§7.2.3 dense search operation (FAISS/numpy via the vector indexer)."""
    results = vector_indexer.dense_search(repo_id, query, top_k)
    evidence: list[Evidence] = []
    for rank, (chunk, score) in enumerate(results, start=1):
        evidence.append(Evidence(chunk=chunk, score=float(score), source="dense", rank=rank))
    return evidence


def embed_query(engine, query: str):
    """§7.2.3 embed_query — code-like queries use the code model."""
    return engine.embed_query(query)
