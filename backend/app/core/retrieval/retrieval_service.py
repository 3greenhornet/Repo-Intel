"""Step 4 complete pipeline (§7.6) — the RetrievalComponent (§3.2.5).

Wires the three retrievers + RRF fusion + reranking, bound to one repository's
knowledge base.

Concurrency model (FIX-003): the component is a singleton shared by all HTTP
requests, so it holds NO per-query state. Lexical (BM25) indexes are cached
per repository under a lock; every retriever takes an explicit ``repo_id``
argument so concurrent queries against different repositories can never
cross-contaminate.
"""
from __future__ import annotations

import threading

from app.core.knowledge.graph_builder import BaseGraphStore
from app.core.knowledge.vector_indexer import VectorIndexer
from app.core.retrieval.fusion import hybrid_retrieve
from app.core.retrieval.graph_retriever import graph_search
from app.core.retrieval.lexical import LexicalIndex
from app.core.retrieval.reranker import Reranker
from app.models.evidence import Evidence, RerankedEvidence
from app.utils.logger import get_logger
from app.utils.time import utc_now

logger = get_logger(__name__)


class RetrievalComponent:
    """§3.2.5 interface: lexical_search / dense_search / graph_search / fuse / rerank.

    Safe for concurrent use across repositories (FIX-003).
    """

    def __init__(self, vector_indexer: VectorIndexer, graph_store: BaseGraphStore) -> None:
        self.vector_indexer = vector_indexer
        self.graph_store = graph_store
        self.reranker = Reranker()
        # Per-repo lexical index cache, guarded by a reentrant lock
        self._lexical_by_repo: dict[str, LexicalIndex] = {}
        self._lexical_lock = threading.RLock()

    # -------------------------------------------------------------- binding
    def bind_repo(self, repo_id: str, chunks: list) -> None:
        """Force (re)build the lexical index for a repository.

        FIX-011: an empty chunk list is NOT cached — a cached empty index
        would mask a later `restore()` that populates the chunks.
        """
        with self._lexical_lock:
            index = LexicalIndex()
            if index.build(chunks):
                self._lexical_by_repo[repo_id] = index
            else:
                self._lexical_by_repo.pop(repo_id, None)

    def _ensure_bound(self, repo_id: str) -> LexicalIndex:
        """Return the repo's lexical index, building it if needed.

        FIX-011: raises a RuntimeError when no chunks are loaded, making
        query-before-restore an explicit failure instead of a silently
        empty lexical channel.
        """
        with self._lexical_lock:
            index = self._lexical_by_repo.get(repo_id)
            if index is not None and index.bm25 is not None:
                return index
            chunks = self.vector_indexer.chunks_by_repo.get(repo_id, [])
            if not chunks:
                raise RuntimeError(
                    f"Lexical index requested for {repo_id}, but no chunks are "
                    f"loaded. Call phase1.restore_knowledge({repo_id!r}) before "
                    f"querying.")
            new_index = LexicalIndex()
            new_index.build(chunks)
            self._lexical_by_repo[repo_id] = new_index
            return new_index

    def invalidate_repo(self, repo_id: str) -> None:
        """FIX-014: drop the per-repo lexical cache after Phase 1 mutations."""
        with self._lexical_lock:
            self._lexical_by_repo.pop(repo_id, None)
        logger.info("retrieval_cache_invalidated",
                    extra={"extra_fields": {"repo_id": repo_id}})

    # ------------------------------------------------------------ retrievers
    def lexical_search(self, repo_id: str, query: str, top_k: int = 20) -> list[Evidence]:
        return self._ensure_bound(repo_id).search(query, top_k)

    def dense_search(self, repo_id: str, query: str, top_k: int = 20) -> list[Evidence]:
        from app.core.retrieval.dense import dense_search

        return dense_search(self.vector_indexer, repo_id, query, top_k)

    def graph_search(self, repo_id: str, query: str, top_k: int = 20) -> list[Evidence]:
        return graph_search(self.graph_store, repo_id, query, top_k=top_k)

    # --------------------------------------------------------------- step 4
    def execute_step_4(self, repo_id: str, query: str, top_k: int = 10) -> RerankedEvidence:
        """§7.6 execute_step_4: hybrid retrieval → rerank → top-k evidence."""
        self._ensure_bound(repo_id)
        fused, source_counts = hybrid_retrieve(
            lambda q, k: self.lexical_search(repo_id, q, k),
            lambda q, k: self.dense_search(repo_id, q, k),
            lambda q, k: self.graph_search(repo_id, q, k),
            query, top_k=top_k,
        )
        reranked = self.reranker.rerank(query, fused, top_k=top_k)

        logger.info("step4_complete", extra={"extra_fields": {
            "repo_id": repo_id, "query": query[:80], "fused": len(fused),
            "reranked": len(reranked), "sources": source_counts,
        }})
        return RerankedEvidence(
            query=query, evidence=reranked, total_retrieved=len(fused),
            top_k=top_k, timestamp=utc_now(),
        )
