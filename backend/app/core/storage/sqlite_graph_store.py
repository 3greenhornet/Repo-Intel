"""SQLite Recursive CTE Graph Store — 2026 In-Process Graph Engine (§Task 1).

Eliminates in-memory NetworkX and external graph servers (Neo4j).
Uses a unified local SQLite database with tables:
- `nodes` (id, qualified_name, entity_type, file_path, source_code, structural_hash)
- `edges` (source_id, target_id, edge_type)

Executes sub-millisecond graph traversals (Upstream Blast Radius, Downstream Call Chains)
using SQLite WITH RECURSIVE Common Table Expressions (CTEs) with cycle prevention.
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

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    qualified_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    source_code TEXT DEFAULT '',
    structural_hash TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_nodes_qname ON nodes(qualified_name);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(entity_type);
CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file_path);

CREATE TABLE IF NOT EXISTS edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id, edge_type);
"""


class SQLiteGraphStore:
    """Persistent, zero-infrastructure graph store using SQLite + Recursive CTEs."""

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
            self._local.conn = c
        return self._local.conn

    def _init_db(self) -> None:
        with self.conn:
            self.conn.executescript(SCHEMA_SQL)

    # ------------------------------------------------------------------ basic api
    def add_node(
        self,
        node_id: str,
        qualified_name: str,
        entity_type: str,
        file_path: str,
        source_code: str = "",
        structural_hash: str = "",
    ) -> None:
        """Insert or replace a node in the graph."""
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO nodes (id, qualified_name, entity_type, file_path, source_code, structural_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (node_id, qualified_name, entity_type, file_path, source_code, structural_hash),
            )

    def add_edge(self, source_id: str, target_id: str, edge_type: str) -> None:
        """Insert or ignore an edge between two nodes."""
        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO edges (source_id, target_id, edge_type)
                VALUES (?, ?, ?)
                """,
                (source_id, target_id, edge_type.upper()),
            )

    def remove_file_graph(self, repo_id: str, file_paths: list[str]) -> None:
        """Delete all nodes and edges associated with specific files in a repo."""
        if not file_paths:
            return
        with self.conn:
            placeholders = ",".join("?" for _ in file_paths)
            file_node_ids = [f"file:{repo_id}:{p}" for p in file_paths]
            file_id_placeholders = ",".join("?" for _ in file_node_ids)

            # 1. Get IDs of nodes to delete (to clean up attached edges)
            cursor = self.conn.execute(
                f"SELECT id FROM nodes WHERE file_path IN ({placeholders}) OR id IN ({file_id_placeholders})",
                [*file_paths, *file_node_ids],
            )
            node_ids = [r[0] for r in cursor.fetchall()]

            # 2. Delete nodes for these files
            self.conn.execute(
                f"DELETE FROM nodes WHERE file_path IN ({placeholders}) OR id IN ({file_id_placeholders})",
                [*file_paths, *file_node_ids],
            )

            # 3. Delete edges where source or target is one of the deleted nodes
            if node_ids:
                id_placeholders = ",".join("?" for _ in node_ids)
                self.conn.execute(
                    f"DELETE FROM edges WHERE source_id IN ({id_placeholders}) OR target_id IN ({id_placeholders})",
                    [*node_ids, *node_ids],
                )

    def insert_partial_graph(self, repo_id: str, touched_files: list, code_intelligence: Any, findings: list) -> None:
        """Insert nodes and edges for only the touched files during incremental updates."""
        node_rows = []
        edge_rows = []
        touched_paths = {getattr(f, "path", str(f)) for f in touched_files}

        # 1. Add File nodes
        for f in touched_files:
            file_path = getattr(f, "path", str(f))
            content_hash = getattr(f, "content_hash", "") or ""
            file_id = f"file:{repo_id}:{file_path}"
            node_rows.append((file_id, file_path, "File", file_path, "", content_hash))
            edge_rows.append((f"repo:{repo_id}", file_id, "HAS_FILE"))

        # 2. Add Symbol nodes + CONTAINS edges
        for sym in getattr(code_intelligence, "symbols", []):
            if sym.file_path not in touched_paths:
                continue
            node_rows.append((
                sym.id,
                sym.name,
                sym.type.capitalize(),
                sym.file_path,
                sym.signature or "",
                "",
            ))
            file_id = f"file:{repo_id}:{sym.file_path}"
            edge_rows.append((file_id, sym.id, "CONTAINS"))

        # 3. Add Relation edges (CALLS, IMPORTS)
        for edge in getattr(code_intelligence, "edges", []):
            edge_rows.append((edge.source, edge.target, edge.type.upper()))

        # 4. Add Findings
        for f in findings:
            if f.file_path not in touched_paths:
                continue
            finding_id = f.id or f"finding:{repo_id}:{f.file_path}:{f.line_start}:{f.rule_id}"
            file_id = f"file:{repo_id}:{f.file_path}"
            node_rows.append((
                finding_id,
                f.rule_id,
                "Finding",
                f.file_path,
                f.message,
                f.severity,
            ))
            edge_rows.append((file_id, finding_id, "HAS_FINDING"))

        with self.conn:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO nodes (id, qualified_name, entity_type, file_path, source_code, structural_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                node_rows,
            )
            self.conn.executemany(
                """
                INSERT OR IGNORE INTO edges (source_id, target_id, edge_type)
                VALUES (?, ?, ?)
                """,
                edge_rows,
            )

    # ------------------------------------------------------------------ recursive CTE
    def trace_impact_radius(self, start_node: str, max_depth: int = 5) -> list[dict]:
        """Upstream Blast Radius: traces callers of callers using SQLite WITH RECURSIVE CTE.

        Uses comma-padded path tracking for cycle detection:
        instr(br.path, ',' || e.source_id || ',') = 0
        """
        query = """
        WITH RECURSIVE blast_radius(node_id, depth, path) AS (
            -- Anchor: nodes directly calling the start node (or symbol qualified_name)
            SELECT
                e.source_id AS node_id,
                1 AS depth,
                ',' || n.id || ',' || e.source_id || ',' AS path
            FROM edges e
            JOIN nodes n ON e.target_id = n.id
            WHERE (n.id = ? OR n.qualified_name = ?)
              AND e.edge_type = 'CALLS'

            UNION ALL

            -- Recursive step: nodes calling the nodes in blast_radius
            SELECT
                e.source_id AS node_id,
                br.depth + 1 AS depth,
                br.path || e.source_id || ',' AS path
            FROM edges e
            JOIN blast_radius br ON e.target_id = br.node_id
            WHERE e.edge_type = 'CALLS'
              AND instr(br.path, ',' || e.source_id || ',') = 0
              AND br.depth < ?
        )
        SELECT
            n.id AS node_id,
            n.qualified_name AS name,
            n.entity_type AS label,
            n.file_path,
            MIN(br.depth) AS path_length,
            COUNT(DISTINCT br.path) AS num_call_paths,
            CASE
                WHEN MIN(br.depth) = 1 THEN 'DIRECT'
                ELSE 'TRANSITIVE'
            END AS impact_type
        FROM blast_radius br
        JOIN nodes n ON br.node_id = n.id
        WHERE n.entity_type IN ('Function', 'Class', 'function', 'class')
        GROUP BY n.id
        ORDER BY path_length ASC, num_call_paths DESC;
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (start_node, start_node, max_depth))
        results = []
        for row in cursor.fetchall():
            rec = dict(row)
            rec["relation"] = "impacted_by"
            rec["other"] = start_node
            results.append(rec)
        return results

    # ------------------------------------------------------------------ BaseGraphStore interface
    def build_graph(
        self,
        snapshot: RepositorySnapshot,
        code_intelligence: CodeIntelligence,
        findings: list[Finding],
    ) -> dict[str, Any]:
        """Atomically populate nodes and edges for a repository."""
        self.clear_repo(snapshot.repo_id)

        node_rows = []
        edge_rows = []

        # Repository Node
        node_rows.append((
            f"repo:{snapshot.repo_id}",
            snapshot.repo_id,
            "Repository",
            "",
            "",
            snapshot.commit_hash or "",
        ))

        # File Nodes
        for f in snapshot.files:
            file_id = f"file:{snapshot.repo_id}:{f.path}"
            node_rows.append((
                file_id,
                f.path,
                "File",
                f.path,
                "",
                f.content_hash or "",
            ))
            edge_rows.append((f"repo:{snapshot.repo_id}", file_id, "HAS_FILE"))

        # Symbol Nodes
        for sym in code_intelligence.symbols:
            node_rows.append((
                sym.id,
                sym.name,
                sym.type.capitalize(),
                sym.file_path,
                sym.signature or "",
                "",
            ))
            edge_rows.append((f"file:{snapshot.repo_id}:{sym.file_path}", sym.id, "CONTAINS"))

        # Relation Edges
        for edge in code_intelligence.edges:
            edge_rows.append((edge.source, edge.target, edge.type.upper()))

        # Findings
        for f in findings:
            finding_id = f.id or f"finding:{snapshot.repo_id}:{f.file_path}:{f.line_start}:{f.rule_id}"
            file_id = f"file:{snapshot.repo_id}:{f.file_path}"
            node_rows.append((
                finding_id,
                f.rule_id,
                "Finding",
                f.file_path,
                f.message,
                f.severity,
            ))
            edge_rows.append((file_id, finding_id, "HAS_FINDING"))

        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO nodes (id, qualified_name, entity_type, file_path, source_code, structural_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                node_rows,
            )
            self.conn.executemany(
                "INSERT OR IGNORE INTO edges (source_id, target_id, edge_type) "
                "VALUES (?, ?, ?)",
                edge_rows,
            )

        logger.info("sqlite_graph_store_built", extra={"extra_fields": {
            "nodes": len(node_rows), "edges": len(edge_rows),
        }})
        return {"backend": "sqlite_cte", "nodes": len(node_rows), "edges": len(edge_rows)}

    def get_impact_analysis(self, repo_id: str, symbol_name: str, max_depth: int = 5) -> list[dict]:
        return self.trace_impact_radius(symbol_name, max_depth=max_depth)

    def get_architecture_summary(self, repo_id: str) -> dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT entity_type, count(*) FROM nodes GROUP BY entity_type")
        counts = {r[0]: r[1] for r in cursor.fetchall()}

        cursor.execute("SELECT count(*) FROM edges WHERE edge_type = 'CALLS'")
        call_edges = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT n.qualified_name, count(e.source_id) as in_degree
            FROM edges e
            JOIN nodes n ON e.target_id = n.id
            WHERE e.edge_type = 'CALLS' AND n.qualified_name != ''
            GROUP BY n.qualified_name
            ORDER BY in_degree DESC
            LIMIT 10;
            """
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

    def graph_search(self, repo_id: str, intent: GraphIntent, depth: int = 3, top_k: int = 20) -> list[dict]:
        target = intent.target
        if not target:
            return []
        if intent.type == "impact":
            return self.trace_impact_radius(target, max_depth=depth)[:top_k]
        if intent.type == "callers":
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT n.id AS node_id, n.qualified_name AS name, n.entity_type AS label, n.file_path
                FROM edges e
                JOIN nodes target_n ON e.target_id = target_n.id
                JOIN nodes n ON e.source_id = n.id
                WHERE e.edge_type = 'CALLS' AND (target_n.qualified_name = ? OR target_n.id = ?)
                LIMIT ?
                """,
                (target, target, top_k),
            )
            return [dict(r, relation="calls", other=target) for r in cursor.fetchall()]
        if intent.type == "symbol_lookup":
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT id AS node_id, qualified_name AS name, entity_type AS label, file_path FROM nodes WHERE qualified_name = ? OR id = ? LIMIT ?",
                (target, target, top_k),
            )
            return [dict(r, relation="definition") for r in cursor.fetchall()]
        return []

    def clear_repo(self, repo_id: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM nodes WHERE id LIKE ? OR id LIKE ?", (f"repo:{repo_id}%", f"file:{repo_id}%"))

    def stats(self) -> dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT count(*) FROM nodes")
        n = cursor.fetchone()[0]
        cursor.execute("SELECT count(*) FROM edges")
        e = cursor.fetchone()[0]
        return {"backend": "sqlite_cte", "total_nodes": n, "total_edges": e}

    def persist(self, path=None):
        pass

    @classmethod
    def load(cls, path):
        return cls(db_path=path)


# Backward compatibility alias
CodeGraphStore = SQLiteGraphStore

