"""Vector index — Store B (§4.3.2) with chunking + embedding + search.

Backends:
- ``NumpyFlatIndex`` (default): exact cosine search — the documented
  `IndexFlatL2` behaviour (§23.5: exact search is optimal for < 10k vectors).
- ``FaissIndex``: used automatically when faiss-cpu is importable; index
  type is selected adaptively (FIX-026, §23.5 decision table + 2026
  correction: memory budget-aware, IVF-HNSW-PQ for > 1M vectors).
- ``PineconeIndex``: NOT implemented — VECTOR_STORE=pinecone fails fast
  (FIX-006b: a silent no-op is worse than a clear error).

Chunking follows §4.3.2 exactly: code by symbol, docs by section,
fallback by size with 500-token cap; split blocks overlap ~20% (FIX-032).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.core.knowledge.embedding_engine import EmbeddingEngine
from app.models.evidence import Chunk
from app.models.repository import FileRecord
from app.utils.logger import get_logger
from app.utils.token_counter import count_tokens

logger = get_logger(__name__)

MAX_CHUNK_TOKENS = 500
CODE_CATEGORIES = {"code", "tests"}
# FIX-032: ~20% overlap between adjacent split blocks (50-line blocks)
SPLIT_OVERLAP_LINES = 10


# --------------------------------------------------------------------------- chunking
def chunk_file(repo_id: str, file: FileRecord, source: str, symbols: list) -> list[Chunk]:
    """§4.3.2 chunk_file: code → by symbol, docs → by section, else by size."""
    if file.category in CODE_CATEGORIES and symbols:
        return _chunk_by_symbol(repo_id, file, source, symbols)
    if file.category == "docs":
        return _chunk_by_section(repo_id, file, source)
    return _chunk_by_size(repo_id, file, source)


def _chunk_by_symbol(repo_id: str, file: FileRecord, source: str, symbols: list) -> list[Chunk]:
    chunks: list[Chunk] = []
    lines = source.splitlines()
    file_symbols = sorted(
        [s for s in symbols if s.file_path == file.path and s.type in ("function", "class")],
        key=lambda s: s.location.start_line,
    )
    covered: list[tuple[int, int]] = []
    for sym in file_symbols:
        start, end = sym.location.start_line, sym.location.end_line
        if any(start <= c[1] and end >= c[0] for c in covered):
            continue  # nested symbol already covered by parent
        content = "\n".join(lines[start - 1: end])
        if not content.strip():
            continue
        if count_tokens(content) > MAX_CHUNK_TOKENS:
            sub_chunks = _split_by_logical_blocks(content, block_lines=50,
                                                  overlap_lines=SPLIT_OVERLAP_LINES)
            for i, (rel_start, rel_end, sub) in enumerate(sub_chunks):
                absolute_start = start + rel_start - 1  # start is the symbol's 1-based first line
                absolute_end = start + rel_end - 1
                chunks.append(Chunk(
                    id="", repo_id=repo_id, file_path=file.path, content=sub,
                    chunk_type="code",
                    start_line=absolute_start,
                    end_line=absolute_end,
                    symbol_id=sym.id, symbol_name=sym.name, language=file.language,
                    metadata={"split": i, "symbol_signature": sym.signature},
                ))
        else:
            chunks.append(Chunk(
                id="", repo_id=repo_id, file_path=file.path, content=content,
                chunk_type="code", start_line=start, end_line=end,
                symbol_id=sym.id, symbol_name=sym.name, language=file.language,
                metadata={"symbol_signature": sym.signature},
            ))
        covered.append((start, end))

    # module-level head (imports/constants) not covered by any symbol
    first_symbol_line = min((c[0] for c in covered), default=len(lines) + 1)
    if first_symbol_line > 1:
        head = "\n".join(lines[: first_symbol_line - 1])
        if head.strip() and count_tokens(head) <= MAX_CHUNK_TOKENS:
            chunks.append(Chunk(
                id="", repo_id=repo_id, file_path=file.path, content=head,
                chunk_type="code", start_line=1, end_line=first_symbol_line - 1,
                language=file.language, metadata={"part": "module_head"},
            ))
    return chunks


def _split_by_logical_blocks(
    content: str, block_lines: int = 50, overlap_lines: int = 0,
) -> list[tuple[int, int, str]]:
    """§4.3.2 split large symbols into logical blocks (FIX-010, FIX-032).

    Returns (relative_start_line, relative_end_line, text) tuples — 1-based,
    relative to the body's first line — so callers can compute absolute line
    ranges instead of assuming every block is exactly ``block_lines`` long.
    ``overlap_lines`` (FIX-032) overlaps adjacent blocks by ~20% when set.
    """
    lines = content.splitlines()
    if not lines:
        return []
    blocks: list[tuple[int, int, str]] = []
    step = max(1, block_lines - overlap_lines)
    i = 0
    while i < len(lines):
        start_rel = i + 1
        end_rel = min(i + block_lines, len(lines))
        text = "\n".join(lines[i:end_rel])
        if text.strip():
            blocks.append((start_rel, end_rel, text))
        if end_rel == len(lines):
            break
        i += step
    return blocks


def _chunk_by_section(repo_id: str, file: FileRecord, source: str) -> list[Chunk]:
    """Docs chunked by markdown heading; split oversized sections."""
    lines = source.splitlines()
    sections: list[tuple[int, int, str]] = []
    current_start = 1
    heading = ""
    for lineno, line in enumerate(lines, start=1):
        if re.match(r"^#{1,3}\s+\S", line):
            if current_start < lineno:
                sections.append((current_start, lineno - 1, heading))
            current_start = lineno
            heading = line.lstrip("# ").strip()[:80]
    sections.append((current_start, len(lines), heading))

    chunks: list[Chunk] = []
    for start, end, title in sections:
        content = "\n".join(lines[start - 1: end])
        if not content.strip():
            continue
        if count_tokens(content) > MAX_CHUNK_TOKENS:
            for i, (rel_start, rel_end, sub) in enumerate(_split_by_logical_blocks(
                    content, block_lines=50, overlap_lines=SPLIT_OVERLAP_LINES)):
                chunks.append(Chunk(
                    id="", repo_id=repo_id, file_path=file.path, content=sub,
                    chunk_type="doc",
                    start_line=start + rel_start - 1,
                    end_line=start + rel_end - 1,
                    metadata={"section": title, "split": i},
                ))
        else:
            chunks.append(Chunk(
                id="", repo_id=repo_id, file_path=file.path, content=content,
                chunk_type="doc", start_line=start, end_line=end,
                metadata={"section": title},
            ))
    return chunks or _chunk_by_size(repo_id, file, source)


def _chunk_by_size(repo_id: str, file: FileRecord, source: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[Chunk]:
    lines = source.splitlines()
    chunks: list[Chunk] = []
    start = 0
    current: list[str] = []
    for lineno, line in enumerate(lines, start=1):
        current.append(line)
        if count_tokens("\n".join(current)) >= max_tokens:
            chunks.append(Chunk(
                id="", repo_id=repo_id, file_path=file.path,
                content="\n".join(current), chunk_type="config" if file.category == "config" else "doc",
                start_line=start + 1, end_line=lineno,
            ))
            current, start = [], lineno
    if current:
        chunks.append(Chunk(
            id="", repo_id=repo_id, file_path=file.path, content="\n".join(current),
            chunk_type="config" if file.category == "config" else "doc",
            start_line=start + 1, end_line=len(lines),
        ))
    return chunks


def chunk_issue(repo_id: str, issue) -> Chunk:
    """Issues/PRs become chunks (§4.3.2 chunking table)."""
    content = f"#{issue.number} [{issue.state}] {issue.title}\n\n{(issue.body or '')[:4000]}"
    labels = getattr(issue, "labels", None) or []
    return Chunk(
        id="", repo_id=repo_id, file_path=f"issue:{issue.number}", content=content,
        chunk_type="issue", start_line=1, end_line=1,
        metadata={"number": issue.number, "state": issue.state, "labels": labels},
    )


def chunk_finding(repo_id: str, finding) -> Chunk:
    """Findings become retrievable evidence chunks (Store C ↔ Store B link)."""
    content = (f"[{finding.severity.upper()}] {finding.rule_id} in {finding.file_path}:"
               f"{finding.line_start}\n{finding.message}")
    return Chunk(
        id="", repo_id=repo_id, file_path=finding.file_path, content=content,
        chunk_type="finding", start_line=finding.line_start, end_line=finding.line_end,
        metadata={"tool": finding.tool, "rule_id": finding.rule_id,
                  "severity": finding.severity, "finding_type": finding.finding_type},
    )


# --------------------------------------------------------------------------- index
class BaseVectorIndex:
    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...
    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...


class NumpyFlatIndex(BaseVectorIndex):
    """Exact cosine-similarity search (documented IndexFlat behaviour)."""

    def __init__(self) -> None:
        self.embeddings: np.ndarray | None = None

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (embeddings / norms).astype(np.float32)

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        q = query_embedding / (np.linalg.norm(query_embedding) or 1.0)
        scores = self.embeddings @ q.astype(np.float32)
        top_k = min(top_k, len(scores))
        indices = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in indices]

    def save(self, path: Path) -> None:
        np.save(path, self.embeddings)

    def load(self, path: Path) -> None:
        self.embeddings = np.load(path)


class FaissFlatIndex(NumpyFlatIndex):
    """FAISS index with automatic type selection (§23.5 + FIX-026).

    Normalized inner-product (cosine) search. IVF indexes require training
    before ``add``; HNSW/Flat do not.
    """

    def __init__(self) -> None:
        import faiss

        self.faiss = faiss
        self.index = None

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        # Normalize for inner-product search to behave like cosine
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = (embeddings / norms).astype(np.float32)

        dim = normalized.shape[1]
        index = select_faiss_index(len(normalized), dim)
        if hasattr(index, "train"):
            # IVF indexes require training; HNSW/Flat do not
            try:
                index.train(normalized)
            except AttributeError:
                pass
        index.add(normalized)
        self.index = index
        self.embeddings = normalized

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = query_embedding / (np.linalg.norm(query_embedding) or 1.0)
        q = np.asarray([q], dtype=np.float32)
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, top_k)
        # Inner product on normalized vectors = cosine similarity → higher is better
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def save(self, path: Path) -> None:
        self.faiss.write_index(self.index, str(path))

    def load(self, path: Path) -> None:
        self.index = self.faiss.read_index(str(path))


def select_faiss_index(num_vectors: int, dimension: int,
                       memory_budget_mb: int | None = None):
    """§23.5 FAISS index selection by dataset size + memory budget (FIX-026).

    Decision table (with the 2026 correction: memory budget aware, and
    IVF-PQ preferred when memory-constrained at any size):

    | Size       | Memory OK       | Memory tight |
    |------------|-----------------|--------------|
    | < 10k      | Flat (exact)    | Flat         |
    | 10k – 1M   | IVF-Flat        | IVF-PQ       |
    | > 1M       | HNSW            | IVF-PQ       |
    """
    import faiss

    if memory_budget_mb is None:
        from app.config import get_settings
        memory_budget_mb = get_settings().vector_memory_budget_mb

    approx_mb = (num_vectors * dimension * 4) / (1024 * 1024)
    # Empirical: IVF-Flat ≈ 1x memory, HNSW ≈ 2x memory, IVF-PQ ≈ 0.25x memory
    memory_ok = approx_mb * 2 < memory_budget_mb

    if num_vectors < 10_000:
        logger.info("FAISS index selection: Flat (small dataset, exact search optimal)")
        return faiss.IndexFlatIP(dimension)

    if num_vectors >= 1_000_000 and memory_ok:
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        logger.info("FAISS index selection: HNSW (M=32, efConstruction=200, efSearch=64)")
        return index

    if memory_ok:
        nlist = min(4096, max(2, int(np.sqrt(num_vectors))))
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = 32
        logger.info(f"FAISS index selection: IVF-Flat (nlist={nlist}, nprobe=32)")
        return index

    # Memory-constrained fallback — product quantization
    nlist = min(4096, max(2, int(np.sqrt(num_vectors))))
    m = 8        # number of subquantizers
    nbits = 8    # bits per subquantizer
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits,
                             faiss.METRIC_INNER_PRODUCT)
    index.nprobe = 32
    logger.info(f"FAISS index selection: IVF-PQ (nlist={nlist}, m={m}, nbits={nbits})")
    return index


class VectorIndexer:
    """§3.2.4 build_vector_index — orchestrates chunking + embedding + index."""

    def __init__(self) -> None:
        self.settings = get_settings()
        if self.settings.vector_store == "pinecone":
            # FIX-006b: a silent no-op is worse than a clear error.
            raise NotImplementedError(
                "Pinecone backend is not implemented in this release. "
                "Set VECTOR_STORE=faiss or VECTOR_STORE=numpy.")
        self.engine = EmbeddingEngine()
        self.chunks: dict[str, Chunk] = {}
        self.chunks_by_repo: dict[str, list[Chunk]] = {}
        self.indexes: dict[str, BaseVectorIndex] = {}
        try:
            import faiss  # noqa: F401

            self._index_factory = FaissFlatIndex
            self.index_backend = "faiss"
        except ImportError:
            self._index_factory = NumpyFlatIndex
            self.index_backend = "numpy"

    def build_vector_index(self, repo_id: str, chunks: list[Chunk]) -> dict:
        """Embed + index all chunks for a repository (§4.3.2)."""
        # FIX-012: prune chunks that no longer belong to this repo from the
        # global map so re-indexing does not grow it monotonically.
        previous_ids = {c.id for c in self.chunks_by_repo.get(repo_id, [])}
        new_ids = {c.id for c in chunks}
        for old_id in previous_ids - new_ids:
            self.chunks.pop(old_id, None)

        if not chunks:
            logger.warning(f"No chunks to index for {repo_id}")
            self.indexes.pop(repo_id, None)
            self.chunks_by_repo[repo_id] = []
            return {"chunks": 0}

        code_texts = [c.content for c in chunks if c.chunk_type == "code"]
        doc_texts = [c.content for c in chunks if c.chunk_type != "code"]
        embeddings = np.zeros((len(chunks), self.engine.dimension), dtype=np.float32)

        code_idx = [i for i, c in enumerate(chunks) if c.chunk_type == "code"]
        doc_idx = [i for i, c in enumerate(chunks) if c.chunk_type != "code"]

        if code_idx:
            vecs = self.engine.embed(code_texts, chunk_type="code")
            embeddings[code_idx] = vecs
        if doc_idx:
            vecs = self.engine.embed(doc_texts, chunk_type="doc")
            embeddings[doc_idx] = vecs

        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
            self.chunks[chunk.id] = chunk

        index = self._index_factory()
        index.build(chunks, embeddings)
        self.indexes[repo_id] = index
        self.chunks_by_repo[repo_id] = chunks
        logger.info("vector_index_built", extra={"extra_fields": {
            "repo_id": repo_id, "chunks": len(chunks),
            "backend": self.index_backend, "dimension": embeddings.shape[1],
            "code_chunks": len(code_idx), "doc_chunks": len(doc_idx),
        }})
        return {"chunks": len(chunks), "backend": self.index_backend}

    def dense_search(self, repo_id: str, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """§7.2.3 dense search operation."""
        index = self.indexes.get(repo_id)
        if index is None:
            return []
        query_embedding = self.engine.embed_query(query)
        results = index.search(query_embedding, top_k)
        return [(self.chunks_by_repo[repo_id][i], score) for i, score in results if i < len(self.chunks_by_repo[repo_id])]

    def persist(self, repo_id: str) -> None:
        """Save index + full chunk data (content included) for cross-process restore."""
        base = self.settings.faiss_dir
        base.mkdir(parents=True, exist_ok=True)
        index = self.indexes.get(repo_id)
        if index:
            np.save(base / f"{repo_id}.index.npy", index.embeddings if hasattr(index, 'embeddings') else None)
        # full chunk payload incl. content (needed by BM25 + context assembly)
        import json

        payload = []
        for c in self.chunks_by_repo.get(repo_id, []):
            d = c.to_dict()
            d["content"] = c.content
            payload.append(d)
        (base / f"{repo_id}_chunks.json").write_text(json.dumps(payload), encoding="utf-8")

        # Save index manifest for cross-session integrity and dimension validation
        manifest = {
            "repo_id": repo_id,
            "embedding_model": getattr(self.engine, "model_name", "unknown"),
            "dimension": self.engine.dimension,
            "chunk_count": len(payload),
            "backend": self.index_backend,
        }
        (base / f"{repo_id}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def restore(self, repo_id: str) -> bool:
        """Reload a previously persisted index (embeddings + chunks)."""
        import json

        base = self.settings.faiss_dir
        chunks_file = base / f"{repo_id}_chunks.json"
        if not chunks_file.exists():
            return False
        try:
            payload = json.loads(chunks_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        chunks = [Chunk(**d) for d in payload]
        if not chunks:
            return False
        for c in chunks:
            self.chunks[c.id] = c
        self.chunks_by_repo[repo_id] = chunks

        # Check index manifest to detect model or dimension changes
        manifest_file = base / f"{repo_id}_manifest.json"
        force_reembed = False
        if manifest_file.exists():
            try:
                m = json.loads(manifest_file.read_text(encoding="utf-8"))
                if m.get("dimension") != self.engine.dimension:
                    logger.warning(
                        f"Dimension mismatch in manifest for {repo_id} "
                        f"({m.get('dimension')} vs {self.engine.dimension}); triggering auto-reembed."
                    )
                    force_reembed = True
            except json.JSONDecodeError:
                pass

        emb_path = base / f"{repo_id}.index.npy"
        embeddings = None
        if emb_path.exists() and not force_reembed:
            embeddings = np.load(emb_path)
        if embeddings is None or len(embeddings) != len(chunks):
            # re-embed (persisted embeddings missing/stale)
            embeddings = np.zeros((len(chunks), self.engine.dimension), dtype=np.float32)
            code_idx = [i for i, c in enumerate(chunks) if c.chunk_type == "code"]
            doc_idx = [i for i, c in enumerate(chunks) if c.chunk_type != "code"]
            if code_idx:
                embeddings[code_idx] = self.engine.embed(
                    [chunks[i].content for i in code_idx], chunk_type="code")
            if doc_idx:
                embeddings[doc_idx] = self.engine.embed(
                    [chunks[i].content for i in doc_idx], chunk_type="doc")
        index = self._index_factory()
        index.build(chunks, embeddings)
        self.indexes[repo_id] = index
        logger.info("vector_index_restored", extra={"extra_fields": {
            "repo_id": repo_id, "chunks": len(chunks), "backend": self.index_backend}})
        return True

    def stats(self) -> dict:
        return {
            "repos": len(self.indexes),
            "total_chunks": sum(len(cs) for cs in self.chunks_by_repo.values()),
            "backend": self.index_backend,
        }
