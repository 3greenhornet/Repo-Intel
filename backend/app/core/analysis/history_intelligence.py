"""History intelligence (§4.2.5 Component E) — verbatim algorithm."""
from __future__ import annotations

from app.models.finding import Finding, FindingCorrelation, HistoryIntelligence
from app.models.repository import CommitRecord
from app.utils.logger import get_logger

logger = get_logger(__name__)


def analyze_history(
    commits: list[CommitRecord], files: list[str], findings: list[Finding] | None = None,
    churn_threshold: int = 3,
) -> HistoryIntelligence:
    """§4.2.5 analyze_history: churn / hotspots / authors / coupling / correlations."""
    findings = findings or []

    change_freq: dict[str, int] = {}
    for commit in commits:
        for file in commit.files_changed:
            change_freq[file] = change_freq.get(file, 0) + 1

    hotspots = [f for f, freq in change_freq.items() if freq > churn_threshold]

    authors: dict[str, int] = {}
    for commit in commits:
        authors[commit.author] = authors.get(commit.author, 0) + 1

    coupling: dict[tuple[str, str], int] = {}
    for commit in commits:
        changed = [f for f in commit.files_changed if not f.startswith(".")]
        for i, f1 in enumerate(changed):
            for f2 in changed[i + 1:]:
                pair = tuple(sorted([f1, f2]))
                coupling[pair] = coupling.get(pair, 0) + 1

    finding_correlations = correlate_findings_with_commits(findings, commits)

    intel = HistoryIntelligence(
        change_frequency=change_freq, hotspots=hotspots, authors=authors,
        coupling=coupling, finding_correlations=finding_correlations,
        churn_threshold=churn_threshold,
    )
    logger.info("history_analysis_complete", extra={"extra_fields": intel.stats()})
    return intel


def correlate_findings_with_commits(
    findings: list[Finding], commits: list[CommitRecord]
) -> list[FindingCorrelation]:
    """§4.2.5: attach the most recent commit touching each finding's file."""
    by_file: dict[str, CommitRecord] = {}
    for commit in commits:
        for file in commit.files_changed:
            if file not in by_file or commit.timestamp > by_file[file].timestamp:
                by_file[file] = commit
    correlations = []
    for finding in findings:
        commit = by_file.get(finding.file_path)
        if commit:
            correlations.append(FindingCorrelation(
                finding_id=finding.id, commit_hash=commit.hash,
                introduced_at=commit.timestamp, fixed_at=None, status="open",
            ))
    return correlations
