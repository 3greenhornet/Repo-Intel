"""Component 4E: Reranking — cross-encoder (§7.5).

Loads `cross-encoder/ms-marco-MiniLM-L-6-v2` (default, §2.4). Error handling
per §7.5.4: model not loadable → skip reranking and keep fused order;
long query/chunk → truncate to 512 tokens; OOM → batch.
"""
from __future__ import annotations

import threading

from app.models.evidence import Evidence
from app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_PAIR_TOKENS = 512  # §7.5.4 truncation

# FIX-049: module-level lazy singleton — the cross-encoder (~80MB) is loaded
# once per process and shared by every Reranker instance, so a future
# per-request construction cannot regress into model-per-request.
_reranker_model = None
_reranker_model_name: str | None = None
_reranker_lock = threading.Lock()
_flashrank_model = None
_flashrank_lock = threading.Lock()


def _get_shared_reranker(model_name: str):
    global _reranker_model, _reranker_model_name
    if _reranker_model is None or _reranker_model_name != model_name:
        with _reranker_lock:
            if _reranker_model is None or _reranker_model_name != model_name:
                from sentence_transformers import CrossEncoder

                _reranker_model = CrossEncoder(model_name)
                _reranker_model_name = model_name
    return _reranker_model


def _get_shared_flashrank(model_name: str = "ms-marco-MiniLM-L-12-v2"):
    global _flashrank_model
    if _flashrank_model is None:
        with _flashrank_lock:
            if _flashrank_model is None:
                from flashrank import Ranker
                _flashrank_model = Ranker(model_name=model_name)
    return _flashrank_model


def _truncate_tokens(text: str, max_tokens: int) -> str:
    """Approximate token truncation (~4 chars/token)."""
    if not text:
        return text
    char_limit = max_tokens * 4
    if len(text) <= char_limit:
        return text
    return text[:char_limit]


class Reranker:
    """Dual-tier reranker: FlashRank (ONNX CPU, ~30ms) with CrossEncoder & Fused fallback."""

    def __init__(self, model_name: str | None = None) -> None:
        from app.config import get_settings

        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self._model = None
        self._backend = "pending"

    def _load(self):
        if self._model is None:
            try:
                self._model = _get_shared_reranker(self.model_name)  # FIX-049
                self._backend = "cross-encoder"
            except Exception as exc:
                logger.warning(f"CrossEncoder unavailable ({exc}) — keeping fused order (§7.5.4)")
                self._model = False  # sentinel: tried and failed
                self._backend = "fused"
        return self._model or None

    def rerank(self, query: str, candidates: list[Evidence], top_k: int = 10) -> list[Evidence]:
        """§7.5.3 reranking with FlashRank-first dual-tier strategy."""
        if not candidates:
            return []

        query_trunc = _truncate_tokens(query, MAX_PAIR_TOKENS)

        # 1. Try ultra-lightweight FlashRank (CPU ONNX, ~30ms, ~4-34MB)
        try:
            ranker = _get_shared_flashrank()
            from flashrank import RerankRequest

            passages = [
                {"id": str(i), "text": _truncate_tokens(e.chunk.content, MAX_PAIR_TOKENS)}
                for i, e in enumerate(candidates)
            ]
            req = RerankRequest(query=query_trunc, passages=passages)
            ranked_results = ranker.rerank(req)

            results: list[Evidence] = []
            for rank_idx, res in enumerate(ranked_results[:top_k], start=1):
                orig_idx = int(res["id"])
                evidence = candidates[orig_idx]
                origin = evidence.chunk.metadata.setdefault("retrieval_origin", evidence.source)
                evidence.chunk.metadata["retrieval_origin"] = origin
                evidence.score = float(res["score"])
                evidence.rank = rank_idx
                evidence.source = "flashrank"
                results.append(evidence)
            self._backend = "flashrank"
            return results
        except (ImportError, Exception) as f_exc:
            logger.debug(f"FlashRank skipped ({f_exc}), using CrossEncoder fallback...")

        # 2. PyTorch CrossEncoder fallback
        model = self._load()
        if model is None:
            # skip reranking, use fused order (§7.5.4 first row)
            for i, e in enumerate(candidates[:top_k], start=1):
                e.rank = i
                e.source = "fused"
            self._backend = "fused"
            return candidates[:top_k]

        pairs = [(query_trunc, _truncate_tokens(e.chunk.content, MAX_PAIR_TOKENS))
                 for e in candidates]
        try:
            scores = model.predict(pairs, batch_size=8)  # §7.5.4 batch on OOM
        except Exception as exc:
            logger.warning(f"Rerank scoring failed ({exc}) — keeping fused order")
            self._backend = "fused"
            return candidates[:top_k]

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        results = []
        for i, (evidence, score) in enumerate(ranked[:top_k], start=1):
            origin = evidence.chunk.metadata.setdefault("retrieval_origin", evidence.source)
            evidence.chunk.metadata["retrieval_origin"] = origin
            evidence.score = float(score)
            evidence.rank = i
            evidence.source = "reranked"
            results.append(evidence)
        self._backend = "cross-encoder"
        return results
