"""Intent layer — check codebase structure against declared architecture (Reflexion Model)."""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import yaml

from app.models.finding import Finding
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_intent(repo_path: Path) -> dict[str, Any] | None:
    """Read .repointel/architecture.yml if present."""
    candidates = [
        repo_path / ".repointel" / "architecture.yml",
        repo_path / ".repointel" / "architecture.yaml",
    ]
    for intent_path in candidates:
        if intent_path.is_file():
            try:
                return yaml.safe_load(intent_path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                logger.warning(f"Invalid architecture.yml: {exc}")
                return None
    return None


def _classify(file_path: str, components: dict[str, Any]) -> str | None:
    """Map a relative file path to a defined architectural component."""
    norm_path = file_path.replace("\\", "/").lstrip("/")
    for name, spec in components.items():
        if isinstance(spec, dict):
            patterns = spec.get("paths", [])
        elif isinstance(spec, list):
            patterns = spec
        else:
            patterns = [str(spec)]

        for pattern in patterns:
            pattern = pattern.replace("\\", "/")
            if fnmatch.fnmatch(norm_path, pattern):
                return name
            # Handle directory wildcard match e.g. "app/api/**" matching "app/api/auth.py"
            if pattern.endswith("/**") and norm_path.startswith(pattern[:-3]):
                return name
    return None


def check_intent(
    repo_path: Path,
    graph_store: Any,
    intent: dict[str, Any],
    repo_id: str = "",
) -> list[Finding]:
    """Compare declared architecture rules against observed graph edges.

    Includes FIX-BUG-2: Scopes query to repo_id to prevent cross-repo edge mixing in SQLite.
    """
    findings: list[Finding] = []
    if not intent:
        return findings

    components = intent.get("components", {})
    rules = intent.get("dependencies", {})
    forbidden = {(r.get("from"), r.get("to")) for r in rules.get("forbidden", []) if isinstance(r, dict)}
    allowed = {(r.get("from"), r.get("to")) for r in rules.get("allowed", []) if isinstance(r, dict)}

    if not hasattr(graph_store, "conn") or graph_store.conn is None:
        return findings

    cursor = graph_store.conn.cursor()

    if repo_id:
        file_prefix = f"file:{repo_id}:%"
        cursor.execute("""
            SELECT n1.file_path AS src_file, n2.file_path AS dst_file, e.edge_type
            FROM edges e
            JOIN nodes n1 ON e.source_id = n1.id
            JOIN nodes n2 ON e.target_id = n2.id
            WHERE e.edge_type IN ('IMPORTS', 'CALLS')
              AND (
                  n1.id LIKE ? OR n2.id LIKE ?
                  OR n1.id IN (SELECT target_id FROM edges WHERE source_id LIKE ? AND edge_type = 'CONTAINS')
                  OR n2.id IN (SELECT target_id FROM edges WHERE source_id LIKE ? AND edge_type = 'CONTAINS')
              )
        """, (file_prefix, file_prefix, file_prefix, file_prefix))
    else:
        cursor.execute("""
            SELECT n1.file_path AS src_file, n2.file_path AS dst_file, e.edge_type
            FROM edges e
            JOIN nodes n1 ON e.source_id = n1.id
            JOIN nodes n2 ON e.target_id = n2.id
            WHERE e.edge_type IN ('IMPORTS', 'CALLS')
        """)

    edges = cursor.fetchall()
    observed_pairs: set[tuple[str, str]] = set()

    for row in edges:
        src_file = row["src_file"] if isinstance(row, dict) else row[0]
        dst_file = row["dst_file"] if isinstance(row, dict) else row[1]
        edge_type = row["edge_type"] if isinstance(row, dict) else row[2]

        if not src_file or not dst_file:
            continue

        src_comp = _classify(src_file, components)
        dst_comp = _classify(dst_file, components)

        if not src_comp or not dst_comp or src_comp == dst_comp:
            continue

        observed_pairs.add((src_comp, dst_comp))

        if (src_comp, dst_comp) in forbidden:
            findings.append(Finding(
                tool="intent_checker",
                rule_id=f"intent:forbidden:{src_comp}->{dst_comp}",
                message=(
                    f"Architecture violation: '{src_comp}' depends on '{dst_comp}', which is strictly "
                    f"forbidden by declared architecture. Offending file: {src_file}"
                ),
                severity="major",
                finding_type="architecture_violation",
                file_path=src_file,
                line_start=0,
                line_end=0,
                metadata={"from": src_comp, "to": dst_comp, "edge": edge_type},
            ))

    # Mature intent status check for missing declared dependencies
    if intent.get("intent", {}).get("status") == "mature":
        for pair in allowed:
            if pair not in observed_pairs and pair[0] != pair[1]:
                findings.append(Finding(
                    tool="intent_checker",
                    rule_id=f"intent:missing:{pair[0]}->{pair[1]}",
                    message=f"Architecture gap: declared expected flow {pair[0]} → {pair[1]}, but no dependency was observed in code graph.",
                    severity="minor",
                    finding_type="architecture_violation",
                    file_path="(architecture.yml)",
                    line_start=0,
                    line_end=0,
                    metadata={"declared_from": pair[0], "declared_to": pair[1]},
                ))

    logger.info("intent_check_complete", extra={"extra_fields": {"findings": len(findings)}})
    return findings
