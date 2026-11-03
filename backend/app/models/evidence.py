"""Evidence / retrieval data models (§4.3.2, §7.1.5, §8.4.4, §9)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime

from app.utils.time import utc_now
from typing import Any, Optional

import numpy as np


@dataclass
class Chunk:
    """A retrievable chunk (§4.3.2 Chunk Data Model)."""

    id: str  # hash of (file_path, start_line, end_line)
    repo_id: str
    file_path: str
    content: str
    chunk_type: str  # code, doc, config, issue, pr, finding
    start_line: int
    end_line: int
    symbol_id: Optional[str] = None  # if code
    symbol_name: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    language: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = hashlib.sha1(
                f"{self.repo_id}:{self.file_path}:{self.start_line}:{self.end_line}".encode()
            ).hexdigest()[:16]

    def prompt_fragment(self) -> str:
        """FIX-033: cached, stable serialization for LLM context assembly.

        Prompt-prefix stability (same content, same truncation, same order)
        lets the provider's native prompt cache (Anthropic cache_control /
        OpenAI cached input tokens) hit across repeated queries. Transformer-
        level KV caching would need a custom inference server — this is the
        pragmatic MVP approximation documented in the fix worklist.
        """
        cached = self.metadata.get("_prompt_fragment")
        if cached is None:
            content = self.content
            if len(content) > 6000:
                head = content[:3000]
                tail = content[-3000:]
                content = f"{head}\n... [truncated {len(self.content) - 6000} chars] ...\n{tail}"
            cached = content
            self.metadata["_prompt_fragment"] = cached
        return cached

    def token_estimate(self) -> int:
        """Rough token count (≈4 chars/token) used for budgeting."""
        return max(1, len(self.content) // 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "file_path": self.file_path,
            "content": self.content,
            "chunk_type": self.chunk_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol_id": self.symbol_id,
            "symbol_name": self.symbol_name,
            "language": self.language,
            "metadata": self.metadata,
        }


@dataclass
class Evidence:
    """A scored retrieval result (§7.1.5)."""

    chunk: Chunk
    score: float
    source: str  # lexical, dense, graph, fused, reranked
    rank: int = 0
    provenance: Optional["Provenance"] = None
    origin: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.origin:
            self.origin = self.source


@dataclass
class GraphIntent:
    """Parsed graph query intent (§7.3.4)."""

    type: str  # callers, callees, dependencies, impact, symbol_lookup
    target: Optional[str]
    raw_query: str = ""


@dataclass
class Provenance:
    """Evidence source metadata (§9.3.3)."""

    source_file: str
    symbol_id: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    commit_hash: Optional[str] = None
    commit_author: Optional[str] = None
    commit_timestamp: Optional[datetime] = None
    analyzer_rule: Optional[str] = None
    analyzer_tool: Optional[str] = None
    confidence: float = 0.0


@dataclass
class NLIRelation:
    """Pairwise evidence relation (§9.2.4)."""

    evidence_a: str  # chunk id
    evidence_b: str  # chunk id
    relation: str  # entailment, contradiction, neutral
    confidence: float = 0.0


@dataclass
class ContextBudget:
    """Context budgeting result (§9.4.3)."""

    evidence: list[Evidence] = field(default_factory=list)
    tokens_used: int = 0
    tokens_available: int = 0
    truncated: bool = False


@dataclass
class ExplorationResult:
    """Agentic exploration result (§8.4.4)."""

    evidence: list[Evidence] = field(default_factory=list)
    iterations: int = 0
    sufficient: bool = False
    reason: str = ""
    elapsed_seconds: float = 0.0
    gap_history: list[str] = field(default_factory=list)
    follow_up_queries: list[str] = field(default_factory=list)


@dataclass
class RerankedEvidence:
    """Step 4 complete output (§7.6)."""

    query: str
    evidence: list[Evidence] = field(default_factory=list)
    total_retrieved: int = 0
    top_k: int = 10
    timestamp: datetime = field(default_factory=utc_now)
