"""Embedding engine (§2.4 / §4.3.2 + FIX-027 + 2026 correction).

Dual-model design per the documentation: one model for code chunks, one for
docs. The 2026-corrected default is **UniXcoder** (`microsoft/unixcoder-base`,
768-dim) for both channels — it unifies code + natural-language
representations (AST + data-flow aware pretraining), beating CodeBERT on
code search while keeping a single dimension for the vector index.

Model resolution chain: UniXcoder/CodeBERT-style raw transformers →
mean-pooling wrapper; sentence-transformers models → direct load;
anything failing → deterministic hashed-TF fallback so the system still
functions with no model weights (embedding model swap = config).
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel(Protocol):
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray: ...


class TransformerCodeEmbedder:
    """Raw transformers model (UniXcoder/CodeBERT) with mean pooling (FIX-027).

    These checkpoints are not sentence-transformers models, so they are
    wrapped with attention-mask-weighted mean pooling. The wrapper is
    thread-safe for read-only inference.
    """

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self._torch = torch
        self.dimension = self.model.config.hidden_size  # 768 for unixcoder/codebert

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        with self._torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                tokens = self.tokenizer(batch, padding=True, truncation=True,
                                        max_length=512, return_tensors="pt")
                outputs = self.model(**tokens)
                # Mean-pool over non-padding tokens
                mask = tokens["attention_mask"].unsqueeze(-1).float()
                summed = (outputs.last_hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
                out.append(pooled.cpu().numpy())
        if not out:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)

class SentenceTransformerModel:
    """Real HF sentence-transformers backend (default)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        try:
            self.dimension = self.model.get_embedding_dimension()
        except AttributeError:  # older sentence-transformers
            self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)


class HashingTFEmbedder:
    """Fallback embedder: hashed bag-of-words with sublinear TF weighting.

    Keeps the dense-retrieval path functional offline; dimensionality matches
    the configured embedding_dimension so downstream code is unchanged.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _tokens(self, text: str) -> list[str]:
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # camelCase split
        text = text.replace("_", " ").replace("-", " ")
        return re.findall(r"\w+", text.lower())

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self._tokens(text)
            for tok in tokens:
                bucket = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % self.dimension
                out[i, bucket] += 1.0
            if tokens:
                nonzero = out[i] > 0
                out[i, nonzero] = 1.0 + np.log(out[i, nonzero])  # sublinear TF
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)


class EmbeddingEngine:
    """§4.3.2 generate_embeddings: code model for code chunks, doc model otherwise."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._code_model: EmbeddingModel | None = None
        self._doc_model: EmbeddingModel | None = None
        self.backend = "sentence-transformers"

    def _load(self, model_name: str) -> EmbeddingModel:
        # FIX-027 + 2026 correction: raw transformer code models get the
        # mean-pooling wrapper; sentence-transformers models load directly.
        lowered = (model_name or "").lower()
        if "unixcoder" in lowered or "codebert" in lowered:
            try:
                return TransformerCodeEmbedder(model_name)
            except Exception as exc:
                logger.warning(
                    f"{model_name} load failed ({exc}) — trying sentence-transformers path")
        try:
            return SentenceTransformerModel(model_name)
        except Exception as exc:
            logger.warning(f"Cannot load '{model_name}' ({exc}) — using hashed TF fallback")
            self.backend = "hashed-tf"
            return HashingTFEmbedder(self.settings.embedding_dimension)

    @property
    def code_model(self) -> EmbeddingModel:
        if self._code_model is None:
            self._code_model = self._load(self.settings.embedding_model_code)
        return self._code_model

    @property
    def doc_model(self) -> EmbeddingModel:
        if self._doc_model is None:
            self._doc_model = self._load(self.settings.embedding_model_doc)
        return self._doc_model

    def embed(self, texts: list[str], chunk_type: str = "doc", batch_size: int = 64) -> np.ndarray:
        if chunk_type == "code":
            return np.asarray(self.code_model.encode(texts, batch_size), dtype=np.float32)
        return np.asarray(self.doc_model.encode(texts, batch_size), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """§7.2.3 embed_query: code-like queries use the code model."""
        looks_like_code = any(ch in query for ch in ("(", ")", "{", "}", "::", "=", ".py"))
        model = self.code_model if looks_like_code else self.doc_model
        return np.asarray(model.encode([query])[0], dtype=np.float32)

    @property
    def model_name(self) -> str:
        if self.backend == "hashed-tf":
            return "hashed-tf"
        return str(self.settings.embedding_model_code)

    @property
    def dimension(self) -> int:
        return len(self.embed_query("dimension probe"))
