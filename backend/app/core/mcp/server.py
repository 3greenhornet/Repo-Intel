"""MCP server (§10.3.1) — Model Context Protocol tools over stdio.

2026-07-28 protocol revision (FIX-045 + 2026 correction): the MCP core is
now STATELESS. The protocol-level session and the `Mcp-Session-Id` header
have been removed; the `initialize`/`initialized` handshake is no longer the
correct flow. Every JSON-RPC request is self-contained, and discovery uses
the now-mandatory `server/discover` method.

Back-compat: `initialize` is still answered (as a no-op carrying server
info) so older clients do not break, but no session state is created.

Implements the five documented tools:
- query_repository(repo_id, question)
- find_findings(repo_id, severity?)
- explain_finding(finding_id)
- get_architecture(repo_id)
- get_impact(repo_id, symbol_id)

SECURITY MODEL (FIX-045):
- This server communicates over stdio (stdin/stdout) with the process that
  spawned it. There is no network listener.
- The trust boundary is the operating system's process model. If a client
  process can spawn this server, that client already has equivalent access
  to the filesystem and the knowledge base.
- Do NOT expose this server over a network socket without adding TLS and
  authentication. For remote access, wrap the stdio server in an
  authenticated reverse proxy or use an MCP gateway that provides auth.
- The server inherits environment variables (including LLM API keys) from
  the parent process. Do not spawn it from untrusted processes.

Speaks the MCP JSON-RPC framing over stdio without external SDK
dependencies; compatible with Claude Code / Pi style registration:
    { "mcpServers": { "repointel": { "command": "python",
      "args": ["-m", "app.core.mcp.server"] } } }
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.knowledge.knowledge_service import Phase1Pipeline  # noqa: E402
from app.core.reasoning.query_pipeline import Phase2Pipeline  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("repointel.mcp")

PROTOCOL_VERSION = "2026-07-28"  # stateless core revision

TOOLS = [
    {"name": "query_repository", "description": "Ask an evidence-grounded question about a repository.",
     "inputSchema": {"type": "object", "properties": {
         "repo_id": {"type": "string"}, "question": {"type": "string"}},
         "required": ["repo_id", "question"]}},
    {"name": "find_findings", "description": "Get security/bug/code smell findings for a repository.",
     "inputSchema": {"type": "object", "properties": {
         "repo_id": {"type": "string"}, "severity": {"type": "string"}}}},
    {"name": "explain_finding", "description": "Explain a specific finding with rule detail.",
     "inputSchema": {"type": "object", "properties": {
         "repo_id": {"type": "string"}, "finding_id": {"type": "string"}}}},
    {"name": "get_architecture", "description": "Get repository architecture overview (files, hubs, call edges).",
     "inputSchema": {"type": "object", "properties": {"repo_id": {"type": "string"}}}},
    {"name": "get_impact", "description": "Calculate blast radius and downstream callers of changing a symbol.",
     "inputSchema": {"type": "object", "properties": {
         "repo_id": {"type": "string"}, "symbol": {"type": "string"}},
         "required": ["repo_id", "symbol"]}},
    {"name": "run_sast_scan", "description": "Execute or retrieve SAST static analysis findings by category.",
     "inputSchema": {"type": "object", "properties": {
         "repo_id": {"type": "string"},
         "category": {"type": "string", "enum": ["vulnerability", "bug", "code_smell", "all"]}},
         "required": ["repo_id"]}},
    {"name": "find_symbol_references", "description": "Look up callers, callees, and definition of a symbol.",
     "inputSchema": {"type": "object", "properties": {
         "repo_id": {"type": "string"}, "symbol": {"type": "string"}},
         "required": ["repo_id", "symbol"]}},
]

DISCOVERY = {
    "protocolVersion": PROTOCOL_VERSION,
    "serverInfo": {"name": "repointel-ai", "version": "1.1.0"},
    "stateless": True,
    "capabilities": {"tools": {"list": True, "call": True}},
    "methods": ["server/discover", "tools/list", "tools/call", "ping"],
    # deprecated-but-answered for pre-2026 clients:
    "deprecated": ["initialize"],
}

_phase1: Phase1Pipeline | None = None
_phase2: Phase2Pipeline | None = None


def _pipelines() -> tuple[Phase1Pipeline, Phase2Pipeline]:
    global _phase1, _phase2
    if _phase1 is None:
        _phase1 = Phase1Pipeline()
        _phase2 = Phase2Pipeline(_phase1)
    return _phase1, _phase2


def handle_tool(name: str, arguments: dict) -> dict:
    phase1, phase2 = _pipelines()
    repo_id = arguments.get("repo_id", "")

    if name == "query_repository":
        phase1.restore_knowledge(repo_id)
        response = phase2.execute_full_pipeline(repo_id, arguments["question"])
        return {"content": [{"type": "text", "text": response.answer}],
                "citations": [c.__dict__ for c in response.citations][:10]}

    if name == "find_findings":
        findings = phase1.db.get_findings(repo_id, severity=arguments.get("severity"))
        return {"content": [{"type": "text", "text": json.dumps(findings[:50], default=str)}]}

    if name == "explain_finding":
        rows = phase1.db.get_findings(repo_id)
        finding = next((f for f in rows if f["id"] == arguments["finding_id"]), None)
        if finding is None:
            return {"content": [{"type": "text", "text": "Finding not found."}]}
        text = (f"[{finding['severity']}] {finding['rule_id']} in "
                f"{finding['file_path']}:{finding['line_start']}\n{finding['message']}")
        return {"content": [{"type": "text", "text": text}]}

    if name == "get_architecture":
        phase1.restore_knowledge(repo_id)
        return {"content": [{"type": "text", "text": json.dumps(
            phase1.graph_store.get_architecture_summary(repo_id))}]}

    if name == "get_impact":
        phase1.restore_knowledge(repo_id)
        results = phase1.graph_store.get_impact_analysis(repo_id, arguments["symbol"])
        return {"content": [{"type": "text", "text": json.dumps(results[:20], default=str)}]}

    if name == "run_sast_scan":
        cat = arguments.get("category", "all")
        finding_type = None if cat == "all" else cat
        findings = phase1.db.get_findings(repo_id, finding_type=finding_type)
        return {
            "content": [{"type": "text", "text": json.dumps({
                "repo_id": repo_id,
                "category": cat,
                "total_findings": len(findings),
                "findings": findings[:100],
            }, default=str)}]
        }

    if name == "find_symbol_references":
        phase1.restore_knowledge(repo_id)
        symbol = arguments["symbol"]
        impact = phase1.graph_store.get_impact_analysis(repo_id, symbol)
        arch = phase1.graph_store.get_architecture_summary(repo_id)
        return {
            "content": [{"type": "text", "text": json.dumps({
                "symbol": symbol,
                "impacted_callers": impact,
                "hub_degree": next((h[1] for h in arch.get("top_hub_symbols", []) if h[0] == symbol), 0),
            }, default=str)}]
        }

    return {"content": [{"type": "text", "text": f"Unknown tool {name}"}]}


def _preflight() -> None:
    """FIX-045: refuse dangerous network exposure flags."""
    if os.environ.get("MCP_ALLOW_NETWORK", "0") == "1":
        raise RuntimeError(
            "MCP_ALLOW_NETWORK is not supported in this release. "
            "See the server module docstring for the security model.")


def dispatch(request: dict) -> dict | None:
    """Stateless request dispatcher (2026-07-28 protocol).

    Returns the JSON-RPC response dict, or None for notifications.
    No session state is created or referenced anywhere.
    """
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "server/discover":
        # Mandatory in the 2026-07-28 revision — replaces initialize.
        return {"jsonrpc": "2.0", "id": req_id, "result": DISCOVERY}

    if method == "initialize":
        # Back-compat no-op: answer with server info, no session created.
        return {"jsonrpc": "2.0", "id": req_id, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "repointel-ai", "version": "1.1.0"},
            "deprecated": True,
            "note": "stateless core — use server/discover (2026-07-28 spec)"}}

    if method == "notifications/initialized":
        return None  # notification — no response, no state

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        try:
            result = handle_tool(request["params"]["name"],
                                 request["params"].get("arguments", {}))
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": req_id, "error": {
                "code": -32000, "message": str(exc)[:300]}}

    return {"jsonrpc": "2.0", "id": req_id, "result": {}}


def main() -> None:
    """Stateless JSON-RPC loop over stdio."""
    _preflight()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = dispatch(request)
        if response is None:
            continue
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
