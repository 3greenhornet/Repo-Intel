"""Tool registry + executor for the RepoIntel Investigator agent.
Each tool wraps an existing RepoIntel function. The registry exports:
TOOL_SCHEMAS — list of {name, description, params} for the planner prompt
ToolExecutor — class with .execute(name, args, repo_id) -> dict
"""
from __future__ import annotations

import logging
from typing import Any

from app.models.evidence import Evidence, GraphIntent

logger = logging.getLogger(__name__)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_code",
        "description": "Search the repository for code chunks matching a query. "
        "Returns chunks with file path, line range, content, and score.",
        "params": {"query": "string", "top_k": "integer (default 5)"},
    },
    {
        "name": "get_findings",
        "description": "List static-analysis findings (bugs, vulnerabilities, smells). "
        "Optionally filter by severity: blocker, critical, major, minor, info.",
        "params": {"severity": "string | null"},
    },
    {
        "name": "get_callers",
        "description": "Find functions that call the given symbol.",
        "params": {"symbol": "string"},
    },
    {
        "name": "get_impact",
        "description": "Find everything that breaks if the given symbol changes "
        "(transitive callers).",
        "params": {"symbol": "string"},
    },
    {
        "name": "get_architecture",
        "description": "Get a repository structure summary: file count, function count, "
        "class count, call-edge count, top hub symbols.",
        "params": {},
    },
    {
        "name": "check_relations",
        "description": "Check entailment/contradiction between evidence chunks from "
        "previous search_code calls. Pass the evidence IDs.",
        "params": {"evidence_ids": "array of string"},
    },
    {
        "name": "finish",
        "description": "Stop investigating and emit the final report. "
        "The report must cite file:line for every claim.",
        "params": {"report": "string"},
    },
]


def render_tool_list() -> str:
    lines = []
    for t in TOOL_SCHEMAS:
        params = ", ".join(f"{k}: {v}" for k, v in t["params"].items()) or "none"
        lines.append(f"- {t['name']}({params}): {t['description']}")
    return "\n".join(lines)


class ToolExecutor:
    """Executes one tool call against the loaded RepoIntel pipelines."""

    def __init__(self, phase1, phase2) -> None:
        self.phase1 = phase1
        self.phase2 = phase2
        self.evidence_cache: dict[str, Evidence] = {}

    def execute(self, name: str, args: dict[str, Any], repo_id: str) -> dict[str, Any]:
        if name == "search_code":
            return self._search_code(repo_id, str(args.get("query", "")), int(args.get("top_k", 5)))
        if name == "get_findings":
            return self._get_findings(repo_id, args.get("severity"))
        if name == "get_callers":
            return self._get_callers(repo_id, str(args.get("symbol", "")))
        if name == "get_impact":
            return self._get_impact(repo_id, str(args.get("symbol", "")))
        if name == "get_architecture":
            return self._get_architecture(repo_id)
        if name == "check_relations":
            return self._check_relations(args.get("evidence_ids", []))
        if name == "finish":
            return {"report": str(args.get("report", ""))}
        raise ValueError(f"Unknown tool: {name}")

    def _search_code(self, repo_id: str, query: str, top_k: int) -> dict[str, Any]:
        try:
            step4 = self.phase2.retrieval.execute_step_4(repo_id, query, top_k=top_k)
        except Exception as exc:
            logger.warning("search_code failed: %s", exc)
            return {"evidence": [], "error": str(exc)[:200]}
        out = []
        for ev in step4.evidence:
            self.evidence_cache[ev.chunk.id] = ev
            out.append({
                "id": ev.chunk.id,
                "file": ev.chunk.file_path,
                "lines": f"{ev.chunk.start_line}-{ev.chunk.end_line}",
                "content": ev.chunk.content[:600],
                "score": round(float(ev.score), 3),
            })
        return {"evidence": out}

    def _get_findings(self, repo_id: str, severity: str | None) -> dict[str, Any]:
        try:
            rows = self.phase1.db.get_findings(repo_id, severity=severity, limit=20)
        except Exception as exc:
            logger.warning("get_findings failed: %s", exc)
            return {"findings": [], "error": str(exc)[:200]}
        return {
            "findings": [
                {
                    "id": r["id"],
                    "severity": r["severity"],
                    "rule": r["rule_id"],
                    "file": r["file_path"],
                    "line": r["line_start"],
                    "message": (r["message"] or "")[:200],
                    "type": r["finding_type"],
                }
                for r in rows
            ]
        }

    def _get_callers(self, repo_id: str, symbol: str) -> dict[str, Any]:
        try:
            intent = GraphIntent(type="callers", target=symbol)
            records = self.phase1.graph_store.graph_search(repo_id, intent, depth=3, top_k=10)
        except Exception as exc:
            logger.warning("get_callers failed: %s", exc)
            return {"callers": [], "error": str(exc)[:200]}
        return {"callers": records}

    def _get_impact(self, repo_id: str, symbol: str) -> dict[str, Any]:
        try:
            records = self.phase1.graph_store.get_impact_analysis(repo_id, symbol)
        except Exception as exc:
            logger.warning("get_impact failed: %s", exc)
            return {"impacted": [], "error": str(exc)[:200]}
        return {"impacted": records[:20]}

    def _get_architecture(self, repo_id: str) -> dict[str, Any]:
        try:
            return self.phase1.graph_store.get_architecture_summary(repo_id)
        except Exception as exc:
            logger.warning("get_architecture failed: %s", exc)
            return {"error": str(exc)[:200]}

    def _check_relations(self, evidence_ids: list[str]) -> dict[str, Any]:
        evidence = [self.evidence_cache[eid] for eid in evidence_ids if eid in self.evidence_cache]
        if len(evidence) < 2:
            return {
                "relations": [],
                "note": "need at least 2 valid evidence IDs from search_code",
            }
        try:
            relations = self.phase2.nli.score_nli(evidence)
        except Exception as exc:
            logger.warning("check_relations failed: %s", exc)
            return {"relations": [], "error": str(exc)[:200]}
        return {
            "relations": [
                {
                    "a": r.evidence_a,
                    "b": r.evidence_b,
                    "relation": r.relation,
                    "confidence": round(float(r.confidence), 3),
                }
                for r in relations
            ]
        }
