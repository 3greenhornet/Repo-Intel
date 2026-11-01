"""Data models — faithful to the documented dataclasses (§4.1.6, §4.2, §4.3.2, §7–§9)."""
from app.models.repository import (
    FileRecord, FileChange, DeltaSnapshot, RepositorySnapshot,
    CommitRecord, IssueRecord, PRRecord, DocRecord, FileCategory,
)
from app.models.finding import Finding, QualityPrediction, HistoryIntelligence, FindingCorrelation
from app.models.code import Symbol, Reference, Edge, CodeIntelligence, SourceLocation
from app.models.evidence import (
    Chunk, Evidence, Provenance, NLIRelation, ContextBudget,
    ExplorationResult, GraphIntent, RerankedEvidence,
)
from app.models.query import QueryRequest, QueryOptions, QueryResponse, Citation, HealthStatus
from app.models.response import DeveloperOutput, Recommendation, ExplanationOutput

__all__ = [
    "FileRecord", "FileChange", "DeltaSnapshot", "RepositorySnapshot",
    "CommitRecord", "IssueRecord", "PRRecord", "DocRecord", "FileCategory",
    "Finding", "QualityPrediction", "HistoryIntelligence", "FindingCorrelation",
    "Symbol", "Reference", "Edge", "CodeIntelligence", "SourceLocation",
    "Chunk", "Evidence", "Provenance", "NLIRelation", "ContextBudget",
    "ExplorationResult", "GraphIntent", "RerankedEvidence",
    "QueryRequest", "QueryOptions", "QueryResponse", "Citation", "HealthStatus",
    "DeveloperOutput", "Recommendation", "ExplanationOutput",
]
