"""One-directional adapter: RepoIntel -> ICE (§Task 9).

RepoIntel is NOT modified. ICE consumes these as Evidence payloads.
Enables downstream consumption of repository intelligence findings by ICE agents.
"""
from __future__ import annotations

from typing import Any
from app.models.finding import Finding


def finding_to_ice_evidence(finding: Finding) -> dict[str, Any]:
    """Maps a RepoIntel Finding to the payload ICE's ingest_evidence expects."""
    return {
        "content": (
            f"[{finding.severity.upper()}] {finding.message}\n"
            f"File: {finding.file_path}:{finding.line_start}"
        ),
        "evidence_type": "TOOL_RESULT",
        "metadata": {
            "tool": "repointel",
            "rule_id": finding.rule_id,
            "severity": finding.severity,
            "file_path": finding.file_path,
            "line_start": finding.line_start,
            "line_end": finding.line_end,
            "finding_type": finding.finding_type,
        },
    }


def export_findings_for_ice(findings: list[Finding]) -> list[dict[str, Any]]:
    """Batch converts RepoIntel findings into ICE-compatible evidence list."""
    return [finding_to_ice_evidence(f) for f in findings]

