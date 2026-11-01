"""Developer output models (§11: Q&A, recommendations, evidence-linked explanations)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.query import Citation


@dataclass
class Recommendation:
    """Explainable recommendation (§11.2)."""

    finding_id: str
    severity: str
    impact: list[str]  # impacted symbols/files
    explanation: str
    fix_suggestion: str
    citations: list[Citation] = field(default_factory=list)
    priority: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "recommendation",
            "finding_id": self.finding_id,
            "severity": self.severity,
            "impact": self.impact,
            "explanation": self.explanation,
            "fix_suggestion": self.fix_suggestion,
            "citations": [c.__dict__ for c in self.citations],
            "priority": round(self.priority, 3),
        }


@dataclass
class ExplanationOutput:
    """Evidence-linked explanation (§11.3)."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    traceable: bool = True


@dataclass
class DeveloperOutput:
    """Step 8 unified output (§11)."""

    output_type: str  # qa, recommendation, explanation
    question: str = ""
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    recommendations: list[Recommendation] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.output_type,
            "question": self.question,
            "answer": self.answer,
            "citations": [
                {
                    "source_file": c.source_file,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                    "evidence_id": c.evidence_id,
                }
                for c in self.citations
            ],
            "confidence": round(self.confidence, 3),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "follow_up_questions": self.follow_up_questions,
            "metadata": self.metadata,
        }
