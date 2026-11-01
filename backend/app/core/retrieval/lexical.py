"""Component 4A: Lexical retrieval — BM25 (§7.1).

Code-aware tokenizer (camelCase / snake_case / kebab-case splitting) per
§7.1.3, rank_bm25 backend, index built over all repository chunks.
"""
from __future__ import annotations

import pickle
import re

from app.config import get_settings
from app.models.evidence import Chunk, Evidence
from app.utils.logger import get_logger

logger = get_logger(__name__)


def tokenize_code(text: str) -> list[str]:
    """§7.1.3 tokenize_code — camelCase/snake/kebab splitting."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ")
    return re.findall(r"\w+", text.lower())


class LexicalIndex:
    """BM25 index over repository chunks."""

    def __init__(self) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25_cls = BM25Okapi
        self.bm25 = None
        self.chunk_ids: list[str] = []
        self.chunks: dict[str, Chunk] = {}

    def build(self, chunks: list[Chunk]) -> bool:
        """Build the BM25 index. Returns True if the index was populated.

        FIX-011: an empty build resets the index state and returns False so
        callers do not cache a silently-empty index.
        """
        if not chunks:
            self.bm25 = None
            self.chunk_ids = []
            self.chunks = {}
            return False
        tokenized = [tokenize_code(c.content) for c in chunks]
        self.bm25 = self._bm25_cls(tokenized)
        self.chunk_ids = [c.id for c in chunks]
        self.chunks = {c.id: c for c in chunks}
        logger.info(f"BM25 index built over {len(chunks)} chunks")
        return True

    def search(self, query: str, top_k: int = 20) -> list[Evidence]:
        """§7.1.4 lexical_search operation."""
        if self.bm25 is None or not query.strip():
            return []
        import numpy as np

        query_tokens = tokenize_code(query)
        if not query_tokens:
            return []
        try:
            scores = self.bm25.get_scores(query_tokens)
        except Exception as exc:  # §7.1.6 tokenization failure fallback
            logger.warning(f"BM25 scoring failed: {exc}")
            return []
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(-scores)[:top_k]
        results: list[Evidence] = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] <= 0:
                continue
            chunk = self.chunks.get(self.chunk_ids[idx])
            if chunk is None:
                continue
            results.append(Evidence(chunk=chunk, score=float(scores[idx]), source="lexical", rank=rank))
        return results

    def persist(self, repo_id: str) -> None:
        settings = get_settings()
        base = settings.faiss_dir
        base.mkdir(parents=True, exist_ok=True)
        with open(base / f"{repo_id}_bm25.pkl", "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, f)

    def load(self, repo_id: str) -> bool:
        settings = get_settings()
        path = settings.faiss_dir / f"{repo_id}_bm25.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]
        return True
