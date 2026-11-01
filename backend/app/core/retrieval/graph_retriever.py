"""Component 4C: Graph retrieval (§7.3) — Neo4j Cypher patterns via graph store.

Includes the §7.3.4 query intent parser (callers / callees / dependencies /
impact / symbol_lookup) and the §7.3.5 graph_search operation. Graph matches
are exact by structure, so score = 1.0 per the documentation.
"""
from __future__ import annotations

import re

from app.models.evidence import Chunk, Evidence, GraphIntent
from app.utils.logger import get_logger

logger = get_logger(__name__)

MULTI_HOP_HINTS = ("why", "impact", "break", "affect", "depend", "caller", "called by")


def parse_graph_intent(query: str) -> GraphIntent:
    """§7.3.4 parse_graph_intent."""
    q = query.lower()
    if "calls" in q or "caller" in q or "who calls" in q:
        return GraphIntent(type="callers", target=extract_symbol(query), raw_query=query)
    if "depend" in q or "import" in q:
        return GraphIntent(type="dependencies", target=extract_symbol(query), raw_query=query)
    if "impact" in q or "break" in q or "affects" in q:
        return GraphIntent(type="impact", target=extract_symbol(query), raw_query=query)
    if "what does" in q and "call" in q:
        return GraphIntent(type="callees", target=extract_symbol(query), raw_query=query)
    return GraphIntent(type="symbol_lookup", target=extract_symbol(query), raw_query=query)


def extract_symbol(query: str) -> str | None:
    """§7.3.4 extract_symbol: backtick-quoted, CamelCase, then snake_case."""
    patterns = [
        r"`([^`]+)`",
        r"\b([A-Z][a-zA-Z0-9]{2,})\b",       # CamelCase / PascalCase
        r"\b([a-z_][a-z0-9_]{2,})\b",        # snake_case (len>=3)
    ]
    stopwords = {
        "the", "who", "what", "does", "this", "that", "function", "method", "file",
        "code", "repo", "repository", "why", "how", "when", "where", "which", "and",
        "for", "with", "from", "call", "calls", "called", "caller", "change", "changing",
        "impact", "breaks", "break", "slow", "bug", "issue", "class",
        "about", "there", "any", "all", "can", "will", "would", "should", "are", "was",
    }
    for pattern in patterns:
        matches = re.findall(pattern, query)
        for m in matches:
            if m.lower() not in stopwords:
                return m
    return None


def graph_search(graph_store, repo_id: str, query: str, max_depth: int = 3, top_k: int = 20) -> list[Evidence]:
    """§7.3.5 graph_search — intent parse → graph traversal → Evidence."""
    intent = parse_graph_intent(query)
    if intent.target is None:
        logger.debug(f"No symbol extracted from query: {query!r}")
        return []
    records = graph_store.graph_search(repo_id, intent, depth=max_depth, top_k=top_k)
    evidence: list[Evidence] = []
    for rank, rec in enumerate(records, start=1):
        node_id = rec.get("node_id") or rec.get("id") or ""
        name = rec.get("name") or rec.get("path") or node_id
        file_path = rec.get("file_path") or (rec.get("path") or "graph")
        line = rec.get("line") or rec.get("start_line") or 0
        chunk = Chunk(
            id=f"graph::{node_id}"[:64] or f"graph::{name}",
            repo_id=repo_id,
            file_path=str(file_path),
            content=_record_to_content(rec, name),
            chunk_type="graph",
            start_line=int(line or 0), end_line=int(line or 0),
            symbol_id=rec.get("id") or node_id,
            symbol_name=str(name),
            metadata={"relation": rec.get("relation"), "graph_intent": intent.type},
        )
        evidence.append(Evidence(chunk=chunk, score=1.0, source="graph", rank=rank))
    return evidence


def _record_to_content(rec: dict, name) -> str:
    parts = [f"Graph node: {name}"]
    if rec.get("relation"):
        parts.append(f"relation: {rec['relation']}")
    if rec.get("signature"):
        parts.append(f"signature: {rec['signature']}")
    if rec.get("file_path"):
        parts.append(f"defined in: {rec['file_path']}:{rec.get('line', 0)}")
    if rec.get("related_to"):
        parts.append(f"related to: {rec['related_to']}")
    if rec.get("rule"):
        parts.append(f"rule: {rec['rule']} severity={rec.get('severity')}")
    if rec.get("message"):
        parts.append(str(rec["message"]))
    if rec.get("title"):
        parts.append(f"title: {rec['title']} [{rec.get('state', '')}]")
    return "\n".join(parts)
