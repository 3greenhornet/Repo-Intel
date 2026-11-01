"""FindingsParser — SARIF + SonarQube JSON → Finding objects (§4.2.3, §23.4).

`parse_sarif_safe` implements the hardened G4 parser (empty SARIF, multi-run,
missing fields, invalid JSON). `parse_sonar_json` maps the documented SonarQube
issue fields.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.models.finding import Finding
from app.utils.logger import get_logger

logger = get_logger(__name__)

SARIF_SEVERITY_MAP = {
    "error": "critical",
    "warning": "major",
    "note": "minor",
    "none": "info",
}


def map_sarif_level(level: str) -> str:
    """§23.4 severity mapping."""
    return SARIF_SEVERITY_MAP.get((level or "").lower(), "info")


def parse_sarif_safe(sarif_path: str | Path) -> list[Finding]:
    """Parse SARIF with all §23.4 edge cases handled."""
    try:
        data = json.loads(Path(sarif_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error(f"Invalid SARIF JSON: {exc}")
        return []
    except FileNotFoundError:
        logger.error(f"SARIF file not found: {sarif_path}")
        return []

    findings: list[Finding] = []
    runs = data.get("runs", [])
    if not runs:
        logger.warning("SARIF has no runs")
        return []

    for run_idx, run in enumerate(runs):
        results = run.get("results", [])
        tool_name = run.get("tool", {}).get("driver", {}).get("name", "codeql")
        for result in results:
            try:
                rule_id = result.get("ruleId", "unknown")
                message = result.get("message", {}).get("text", "")
                level = result.get("level", "warning")
                locations = result.get("locations", [])
                if not locations:
                    locations = result.get("relatedLocations", [])  # §23.4 fallback
                for loc in locations:
                    physical = loc.get("physicalLocation", {})
                    artifact = physical.get("artifactLocation", {})
                    region = physical.get("region", {})
                    findings.append(Finding(
                        tool=tool_name, rule_id=rule_id, message=message,
                        severity=map_sarif_level(level),
                        file_path=artifact.get("uri", "unknown"),
                        line_start=region.get("startLine", 0),
                        line_end=region.get("endLine", region.get("startLine", 0)),
                        metadata={"run_index": run_idx},
                    ))
            except Exception as exc:  # per-result isolation (§23.4)
                logger.warning(f"Failed to parse SARIF result: {exc}")
                continue

    logger.info(f"Parsed {len(findings)} findings from {len(runs)} SARIF runs")
    return findings


def parse_sonar_json(issues: list[dict]) -> list[Finding]:
    """Map SonarQube issue fields (§4.2.3 'Finding Fields Extracted')."""
    findings: list[Finding] = []
    for issue in issues:
        component = issue.get("component", "")
        file_path = component.split(":", 1)[1] if ":" in component else component
        findings.append(Finding(
            tool="sonarqube",
            rule_id=issue.get("rule", "unknown"),
            message=issue.get("message", ""),
            severity=(issue.get("severity") or "info").lower(),
            finding_type=_sonar_type(issue),
            file_path=file_path,
            line_start=issue.get("line", 0),
            line_end=issue.get("line", 0),
            status=issue.get("status", "OPEN").lower(),
            metadata={"debt": issue.get("debt"), "tags": issue.get("tags", [])},
        ))
    return findings


def _sonar_type(issue: dict) -> str:
    t = issue.get("type", "CODE_SMELL").upper()
    return {"BUG": "bug", "VULNERABILITY": "vulnerability"}.get(t, "code_smell")
