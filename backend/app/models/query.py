"""API query/response models (§10.3.2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from app.utils.time import utc_now


@dataclass
class QueryOptions:
    """Per-query options (§10.3.2 POST /api/v1/query)."""

    max_iterations: int = 5
    top_k: int = 10
    include_provenance: bool = True
    use_agentic: bool = True
    use_nli: bool = True
    stream: bool = False
    generate_followups: bool = True


@dataclass
class QueryRequest:
    repo_id: str
    question: str
    options: QueryOptions = field(default_factory=QueryOptions)


@dataclass
class Citation:
    """A single answer citation (§10.3.2 response)."""

    evidence_id: str
    source_file: str
    line_start: int
    line_end: int
    symbol_id: Optional[str] = None
    commit_hash: Optional[str] = None
    snippet: str = ""


@dataclass(frozen=True)  # FIX-051: post-construction mutation is forbidden
class QueryResponse:
    """Full pipeline response (§10.3.2).

    Frozen so `to_dict()` always serializes a consistent snapshot — any
    future streaming feature must build a new response object.
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    repo_id: str = ""
    question: str = ""
    exploration_iterations: int = 0
    elapsed_seconds: float = 0.0
    confidence: float = 0.0
    grounded: bool = False
    abstained: bool = False  # FIX-001: distinguish abstention from hallucination
    evidence_count: int = 0
    follow_up_questions: list[str] = field(default_factory=list)
    retrieval_sources: dict[str, int] = field(default_factory=dict)
    evidence_files: list[str] = field(default_factory=list)  # deduped selected files
    contradictions: list[dict] = field(default_factory=list)  # FIX-029
    prior_investigations: list[dict] = field(default_factory=list)  # FIX-031
    timestamp: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "question": self.question,
            "answer": self.answer,
            "citations": [
                {
                    "evidence_id": c.evidence_id,
                    "source_file": c.source_file,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                    "symbol_id": c.symbol_id,
                    "commit_hash": c.commit_hash,
                    "snippet": c.snippet[:200],
                }
                for c in self.citations
            ],
            "exploration_iterations": self.exploration_iterations,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "confidence": round(self.confidence, 3),
            "grounded": self.grounded,
            "abstained": self.abstained,
            "evidence_count": self.evidence_count,
            "follow_up_questions": self.follow_up_questions,
            "retrieval_sources": self.retrieval_sources,
            "evidence_files": self.evidence_files,
            "contradictions": self.contradictions,
            "prior_investigations": self.prior_investigations,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HealthStatus:
    """Health check response (§24.5 — dependency validating)."""

    status: str  # healthy, degraded
    components: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)
