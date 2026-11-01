"""SQLite Recursive CTE Graph Store — Local-First Code Knowledge Graph.

Replaces volatile in-memory NetworkX with a single, persistent SQLite database.
Stores repository graph nodes (Files, Functions, Classes, Variables, Findings)
and edges (CALLS, IMPORTS, CONTAINS, REFERENCES, HAS_FINDING).

Uses SQLite WITH RECURSIVE CTEs for:
1. Upstream Blast Radius (reverse call tree traversal with cycle guards).
2. Downstream Call Hierarchy (forward call tree).
3. Transitive Dependency & Import Resolution.
4. Sub-millisecond architecture summaries and central hub metrics.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from app.models.code import CodeIntelligence
from app.models.evidence import GraphIntent
from app.models.finding import Finding
from app.models.repository import RepositorySnapshot
from app.utils.logger import get_logger

logger = get_logger(__name__)

GRAPH_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    label TEXT NOT NULL,
    name TEXT,
    file_path TEXT,
    line INTEGER,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_repo_name ON graph_nodes(repo_id, name);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_repo_label ON graph_nodes(repo_id, label);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_repo_file ON graph_nodes(repo_id, file_path);

CREATE TABLE IF NOT EXISTS graph_edges (
    id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    type TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_callee_caller ON graph_edges(repo_id, type, target, source);
CREATE INDEX IF NOT EXISTS idx_graph_edges_caller_callee ON graph_edges(repo_id, type, source, target);
"""


class SQLiteGraphStore:
    """Persistent, embedded graph store using SQLite + Recursive CTEs."""

    _SYMBOL_LABELS = {
        "function": "Function",
        "class": "Class",
        "variable": "Variable",
        "constant": "Variable",
        "interface": "Class",
        "enum": "Class",
    }

    def __init__(self, db_path: Path | str | None = None) -> None:
        from app.config import get_settings

        if db_path is None:
            db_path = get_settings().data_dir / "repointel.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            c = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode = WAL")
            c.execute("PRAGMA synchronous = NORMAL")
            c.execute("PRAGMA foreign_keys = ON")
            self._local.conn = c
        return self._local.conn

    def _init_db(self) -> None:
        with self.conn:
            self.conn.executescript(GRAPH_SCHEMA)

    # ------------------------------------------------------------------ build
    def build_graph(
        self,
        snapshot: RepositorySnapshot,
        code_intelligence: CodeIntelligence,
        findings: list[Finding],
    ) -> dict[str, Any]:
        """Index nodes and edges atomically into SQLite."""
        self.clear_repo(snapshot.repo_id)

        node_rows = []
        edge_rows = []

        # 1. Repository node
        node_rows.append((
            f"repo:{snapshot.repo_id}",
            snapshot.repo_id,
            "Repository",
            snapshot.repo_id,
            "",
            0,
            json.dumps({"url": snapshot.repo_url, "branch": snapshot.branch, "commit": snapshot.commit_hash}),
        ))

        # 2. File nodes + HAS_FILE edges
        for file in snapshot.files:
            file_id = f"file:{snapshot.repo_id}:{file.path}"
            node_rows.append((
                file_id,
                snapshot.repo_id,
                "File",
                file.path,
                file.path,
                0,
                json.dumps({"language": file.language, "category": file.category, "size": file.size_bytes}),
            ))
            edge_rows.append((
                f"edge:{f'repo:{snapshot.repo_id}'}->{file_id}:HAS_FILE",
                snapshot.repo_id,
                f"repo:{snapshot.repo_id}",
                file_id,
                "HAS_FILE",
                "{}",
            ))

        # 3. Symbol nodes + CONTAINS edges
        for sym in code_intelligence.symbols:
            label = self._SYMBOL_LABELS.get(sym.type, "Variable")
            file_id = f"file:{snapshot.repo_id}:{sym.file_path}"
            node_rows.append((
                sym.id,
                snapshot.repo_id,
                label,
                sym.name,
                sym.file_path,
                sym.location.start_line,
                json.dumps({"signature": sym.signature, "type": sym.type}),
            ))
            edge_rows.append((
                f"edge:{file_id}->{sym.id}:CONTAINS",
                snapshot.repo_id,
                file_id,
                sym.id,
                "CONTAINS",
                "{}",
            ))

        # 4. Code relation edges (CALLS, IMPORTS, REFERENCES, INHERITS)
        for idx, edge in enumerate(code_intelligence.edges):
            edge_type = edge.type.upper()
            edge_rows.append((
                f"edge:{edge.source}->{edge.target}:{edge_type}:{idx}",
                snapshot.repo_id,
                edge.source,
                edge.target,
                edge_type,
                json.dumps(edge.metadata or {}),
            ))

        # 5. Finding nodes + HAS_FINDING edges
        for f in findings:
            finding_id = f.id or f"finding:{snapshot.repo_id}:{f.file_path}:{f.line_start}:{f.rule_id}"
            file_id = f"file:{snapshot.repo_id}:{f.file_path}"
            node_rows.append((
                finding_id,
                snapshot.repo_id,
                "Finding",
                f.rule_id,
                f.file_path,
                f.line_start,
                json.dumps({"severity": f.severity, "type": f.finding_type, "message": f.message, "tool": f.tool}),
            ))
            edge_rows.append((
                f"edge:{file_id}->{finding_id}:HAS_FINDING",
                snapshot.repo_id,
                file_id,
                finding_id,
                "HAS_FINDING",
                "{}",
            ))

        # Atomic batch insertion
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO graph_nodes (id, repo_id, label, name, file_path, line, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                node_rows,
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO graph_edges (id, repo_id, source, target, type, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                edge_rows,
            )

        logger.info(
            "sqlite_graph_indexed",
            extra={"extra_fields": {"repo_id": snapshot.repo_id, "nodes": len(node_rows), "edges": len(edge_rows)}},
        )
        return {
            "backend": "sqlite",
            "nodes": len(node_rows),
            "edges": len(edge_rows),
            "symbols": len(code_intelligence.symbols),
            "findings": len(findings),
        }

    # ------------------------------------------------------------------ queries & CTEs
    def trace_impact_radius(
        self,
        repo_id: str,
        symbol_name: str,
        max_depth: int = 5,
        limit: int = 50,
    ) -> list[dict]:
        """Upstream Blast Radius calculation using Recursive CTE.

        Traces all callers of callers (transitive dependents) that will break
        if the specified symbol is modified.
        Uses comma-padded path tracking for cycle prevention.
        """
        query = """
        WITH RECURSIVE blast_radius(node_id, depth, path) AS (
            -- Anchor: Immediate callers of target symbol
            SELECT
                e.source AS node_id,
                1 AS depth,
                ',' || n.id || ',' || e.source || ',' AS path
            FROM graph_edges e
            JOIN graph_nodes n ON e.target = n.id
            WHERE e.repo_id = ?
              AND n.repo_id = ?
              AND (n.name = ? OR n.id = ?)
              AND e.type = 'CALLS'

            UNION ALL

            -- Recursive step: Callers of callers
            SELECT
                e.source AS node_id,
                br.depth + 1 AS depth,
                br.path || e.source || ',' AS path
            FROM graph_edges e
            JOIN blast_radius br ON e.target = br.node_id
            WHERE e.repo_id = ?
              AND e.type = 'CALLS'
              -- Guard against cycles (A calls B calls A)
              AND instr(br.path, ',' || e.source || ',') = 0
              -- Horizon limiter
              AND br.depth < ?
        )
        SELECT
            gn.id AS node_id,
            gn.label,
            gn.name,
            gn.file_path,
            gn.line,
            gn.metadata,
            MIN(br.depth) AS path_length,
            COUNT(DISTINCT br.path) AS num_call_paths,
            CASE
                WHEN MIN(br.depth) = 1 THEN 'DIRECT'
                ELSE 'TRANSITIVE'
            END AS impact_type
        FROM blast_radius br
        JOIN graph_nodes gn ON br.node_id = gn.id
        WHERE gn.repo_id = ?
          AND gn.label IN ('Function', 'Class')
        GROUP BY gn.id
        ORDER BY path_length ASC, num_call_paths DESC
        LIMIT ?;
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (repo_id, repo_id, symbol_name, symbol_name, repo_id, max_depth, repo_id, limit))
        results = []
        for row in cursor.fetchall():
            rec = dict(row)
            rec["relation"] = "impacted_by"
            rec["other"] = symbol_name
            try:
                rec["metadata"] = json.loads(rec.get("metadata") or "{}")
            except Exception:
                pass
            results.append(rec)
        return results

    def trace_callees(
        self,
        repo_id: str,
        symbol_name: str,
        max_depth: int = 5,
        limit: int = 50,
    ) -> list[dict]:
        """Downstream call chain using Recursive CTE."""
        query = """
        WITH RECURSIVE callee_tree(node_id, depth, path) AS (
            SELECT
                e.target AS node_id,
                1 AS depth,
                ',' || n.id || ',' || e.target || ',' AS path
            FROM graph_edges e
            JOIN graph_nodes n ON e.source = n.id
            WHERE e.repo_id = ?
              AND n.repo_id = ?
              AND (n.name = ? OR n.id = ?)
              AND e.type = 'CALLS'

            UNION ALL

            SELECT
                e.target AS node_id,
                ct.depth + 1 AS depth,
                ct.path || e.target || ',' AS path
            FROM graph_edges e
            JOIN callee_tree ct ON e.source = ct.node_id
            WHERE e.repo_id = ?
              AND e.type = 'CALLS'
              AND instr(ct.path, ',' || e.target || ',') = 0
              AND ct.depth < ?
        )
        SELECT
            gn.id AS node_id,
            gn.label,
            gn.name,
            gn.file_path,
            gn.line,
            gn.metadata,
            MIN(ct.depth) AS path_length
        FROM callee_tree ct
        JOIN graph_nodes gn ON ct.node_id = gn.id
        WHERE gn.repo_id = ?
        GROUP BY gn.id
        ORDER BY path_length ASC
        LIMIT ?;
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (repo_id, repo_id, symbol_name, symbol_name, repo_id, max_depth, repo_id, limit))
        results = []
        for row in cursor.fetchall():
            rec = dict(row)
            rec["relation"] = "called_by"
            rec["other"] = symbol_name
            results.append(rec)
        return results

    def trace_dependencies(
        self,
        repo_id: str,
        target: str,
        max_depth: int = 3,
        limit: int = 50,
    ) -> list[dict]:
        """Transitive dependencies and imports resolution."""
        query = """
        WITH RECURSIVE dep_tree(node_id, depth, path) AS (
            SELECT
                e.target AS node_id,
                1 AS depth,
                ',' || e.source || ',' || e.target || ',' AS path
            FROM graph_edges e
            WHERE e.repo_id = ?
              AND (e.source LIKE '%' || ? || '%' OR e.source IN (
                  SELECT id FROM graph_nodes WHERE repo_id = ? AND (name = ? OR file_path LIKE '%' || ? || '%')
              ))
              AND e.type IN ('IMPORTS', 'CALLS', 'CONTAINS')

            UNION ALL

            SELECT
                e.target AS node_id,
                dt.depth + 1 AS depth,
                dt.path || e.target || ',' AS path
            FROM graph_edges e
            JOIN dep_tree dt ON e.source = dt.node_id
            WHERE e.repo_id = ?
              AND e.type IN ('IMPORTS', 'CALLS')
              AND instr(dt.path, ',' || e.target || ',') = 0
              AND dt.depth < ?
        )
        SELECT
            gn.id AS node_id,
            gn.label,
            gn.name,
            gn.file_path,
            gn.line,
            MIN(dt.depth) AS path_length
        FROM dep_tree dt
        JOIN graph_nodes gn ON dt.node_id = gn.id
        WHERE gn.repo_id = ?
        GROUP BY gn.id
        ORDER BY path_length ASC
        LIMIT ?;
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (repo_id, target, repo_id, target, target, repo_id, max_depth, repo_id, limit))
        results = []
        for row in cursor.fetchall():
            rec = dict(row)
            rec["relation"] = "depends_on"
            results.append(rec)
        return results

    # ------------------------------------------------------------------ BaseGraphStore api
    def graph_search(
        self,
        repo_id: str,
        intent: GraphIntent,
        depth: int = 3,
        top_k: int = 20,
    ) -> list[dict]:
        """Fulfill §7.3.5 graph search queries using fast SQL + CTEs."""
        target = intent.target
        if not target:
            return []

        if intent.type == "impact":
            return self.trace_impact_radius(repo_id, target, max_depth=depth, limit=top_k)

        if intent.type == "callers":
            query = """
            SELECT DISTINCT gn.id AS node_id, gn.label, gn.name, gn.file_path, gn.line
            FROM graph_edges e
            JOIN graph_nodes target_n ON e.target = target_n.id
            JOIN graph_nodes gn ON e.source = gn.id
            WHERE e.repo_id = ? AND e.type = 'CALLS'
              AND (target_n.name = ? OR target_n.id = ?)
            LIMIT ?;
            """
            cursor = self.conn.cursor()
            cursor.execute(query, (repo_id, target, target, top_k))
            return [dict(r, relation="calls", other=target) for r in cursor.fetchall()]

        if intent.type == "callees":
            return self.trace_callees(repo_id, target, max_depth=depth, limit=top_k)

        if intent.type == "dependencies":
            return self.trace_dependencies(repo_id, target, max_depth=depth, limit=top_k)

        if intent.type == "symbol_lookup":
            query = """
            SELECT id AS node_id, label, name, file_path, line, metadata
            FROM graph_nodes
            WHERE repo_id = ? AND (name = ? OR id = ?)
            LIMIT ?;
            """
            cursor = self.conn.cursor()
            cursor.execute(query, (repo_id, target, target, top_k))
            return [dict(r, relation="definition") for r in cursor.fetchall()]

        return []

    def get_architecture_summary(self, repo_id: str) -> dict:
        """High-speed aggregated architecture summary from SQLite tables."""
        cursor = self.conn.cursor()

        # Counts
        cursor.execute(
            "SELECT label, count(*) FROM graph_nodes WHERE repo_id = ? GROUP BY label",
            (repo_id,),
        )
        counts = {r[0]: r[1] for r in cursor.fetchall()}

        cursor.execute(
            "SELECT count(*) FROM graph_edges WHERE repo_id = ? AND type = 'CALLS'",
            (repo_id,),
        )
        call_edges = cursor.fetchone()[0]

        # Top Hub Symbols (Highest incoming call degrees)
        cursor.execute(
            """
            SELECT gn.name, count(e.id) as in_degree
            FROM graph_edges e
            JOIN graph_nodes gn ON e.target = gn.id
            WHERE e.repo_id = ? AND e.type = 'CALLS' AND gn.name IS NOT NULL AND gn.name != ''
            GROUP BY gn.name
            ORDER BY in_degree DESC
            LIMIT 10;
            """,
            (repo_id,),
        )
        hub_symbols = cursor.fetchall()

        return {
            "backend": "sqlite_cte",
            "code_files": counts.get("File", 0),
            "functions": counts.get("Function", 0),
            "classes": counts.get("Class", 0),
            "call_edges": call_edges,
            "top_hub_symbols": [[h[0], h[1]] for h in hub_symbols],
        }

    def get_impact_analysis(self, repo_id: str, symbol_name: str, max_depth: int = 5) -> list[dict]:
        return self.trace_impact_radius(repo_id, symbol_name, max_depth=max_depth)

    def stats(self) -> dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT count(*) FROM graph_nodes")
        total_nodes = cursor.fetchone()[0]
        cursor.execute("SELECT count(*) FROM graph_edges")
        total_edges = cursor.fetchone()[0]
        return {"backend": "sqlite", "total_nodes": total_nodes, "total_edges": total_edges}

    def clear_repo(self, repo_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM graph_nodes WHERE repo_id = ?", (repo_id,))
            self.conn.execute("DELETE FROM graph_edges WHERE repo_id = ?", (repo_id,))

    def persist(self, path: Path | str | None = None) -> None:
        """SQLite is already persistent on disk."""
        pass

    @classmethod
    def load(cls, path: Path | str) -> SQLiteGraphStore:
        return cls(db_path=path)


# Alias as CodeGraphStore for 2026 local-first code graph standards
CodeGraphStore = SQLiteGraphStore

