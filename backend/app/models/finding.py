"""Findings / quality prediction / history data models (§4.2.3–§4.2.5)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from app.utils.time import utc_now


def _compute_finding_id(tool: str, rule_id: str, file_path: str, line_start: int,
                        line_end: int, message: str) -> str:
    """FIX-005: content-derived finding id.

    Two runs of build_knowledge over identical content produce identical
    ids so INSERT OR REPLACE upserts instead of duplicating rows.
    """
    key = f"{tool}|{rule_id}|{file_path}|{line_start}|{line_end}|{message[:200]}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:32]


@dataclass
class Finding:
    """A static-analysis finding from any tool (§4.2.3)."""

    tool: str  # repointel (built-in), sonarqube, codeql
    rule_id: str
    message: str
    severity: str  # blocker, critical, major, minor, info
    file_path: str
    line_start: int = 0
    line_end: int = 0
    finding_type: str = "code_smell"  # bug, vulnerability, code_smell, defect_prediction, td_prediction
    symbol_id: Optional[str] = None
    confidence: float = 1.0
    status: str = "open"
    repo_id: Optional[str] = None
    id: str = field(default="")
    metadata: dict[str, Any] = field(default_factory=dict)
    commit_hash: Optional[str] = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _compute_finding_id(
                self.tool, self.rule_id, self.file_path,
                self.line_start, self.line_end, self.message)


@dataclass
class QualityPrediction:
    """ML-based quality prediction (§4.2.4)."""

    file_path: str
    symbol_id: Optional[str]
    defect_probability: float  # 0-1
    smell_type: Optional[str]
    smell_confidence: float
    td_score: float
    satd_flag: bool
    features: dict[str, float] = field(default_factory=dict)
    model_version: str = "v1.0"
    predicted_at: datetime = field(default_factory=utc_now)

    @property
    def id(self) -> str:
        """FIX-005 Part C: deterministic id derived from (file, symbol)."""
        return hashlib.sha1(
            f"{self.file_path}|{self.symbol_id or ''}".encode()).hexdigest()[:32]


@dataclass
class FindingCorrelation:
    """Finding ↔ commit correlation (§4.2.5)."""

    finding_id: str
    commit_hash: str
    introduced_at: Optional[datetime]
    fixed_at: Optional[datetime]
    status: str = "open"


@dataclass
class HistoryIntelligence:
    """Git history analytics (§4.2.5)."""

    change_frequency: dict[str, int] = field(default_factory=dict)
    hotspots: list[str] = field(default_factory=list)
    authors: dict[str, int] = field(default_factory=dict)
    coupling: dict[tuple[str, str], int] = field(default_factory=dict)
    finding_correlations: list[FindingCorrelation] = field(default_factory=list)
    churn_threshold: int = 3

    def stats(self) -> dict:
        return {
            "hotspot_count": len(self.hotspots),
            "hotspots": self.hotspots[:10],
            "top_authors": sorted(self.authors.items(), key=lambda x: -x[1])[:5],
            "coupled_pairs": len(self.coupling),
            "top_couplings": [
                {"files": list(pair), "count": count}
                for pair, count in sorted(self.coupling.items(), key=lambda x: -x[1])[:5]
            ],
        }
