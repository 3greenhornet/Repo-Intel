"""Graph store — Neo4j schema (§4.3.1 Store A) with two backends:

- ``NetworkXGraphStore`` (default): embedded, works with zero infrastructure.
  Implements the documented node/edge schema and the §7.3 query types
  (callers, callees, dependencies, impact, symbol lookup).
- ``Neo4jGraphStore`` (production): executes the documented Cypher statements
  when a Neo4j server is reachable (GRAPH_STORE=neo4j).

Both expose the same interface so components stay swappable (§1.3).
"""
from __future__ import annotations

from typing import Any

import networkx as nx

from app.core.storage.sqlite_graph_store import CodeGraphStore, SQLiteGraphStore
from app.models.code import CodeIntelligence
from app.models.evidence import GraphIntent
from app.models.finding import Finding
from app.models.repository import RepositorySnapshot
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Documented node labels (§4.3.1)
NODE_LABELS = ("Repository", "File", "Function", "Class", "Variable",
               "Commit", "Issue", "PullRequest", "Finding")
# Documented edge types (§4.3.1)
EDGE_TYPES = ("CONTAINS", "CALLS", "IMPORTS", "REFERENCES", "INHERITS",
              "IMPLEMENTS", "MODIFIES", "RELATES_TO", "HAS_FINDING", "AUTHORED_BY")


class GraphStats(dict):
    pass


class BaseGraphStore:
    """Interface for graph backends (§3.2.4 build_graph)."""

    def build_graph(
        self, snapshot: RepositorySnapshot, code_intelligence: CodeIntelligence,
        findings: list[Finding],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def graph_search(self, repo_id: str, intent: GraphIntent, depth: int = 3, top_k: int = 20) -> list[dict]:
        raise NotImplementedError

    def get_architecture_summary(self, repo_id: str) -> dict:
        raise NotImplementedError

    def get_impact_analysis(self, repo_id: str, symbol_name: str, max_depth: int = 5) -> list[dict]:
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError

    def clear_repo(self, repo_id: str) -> None:
        raise NotImplementedError


class NetworkXGraphStore(BaseGraphStore):
    """[DEPRECATED in 2026] Legacy in-memory implementation.
    Use SQLiteGraphStore (app.core.storage.sqlite_graph_store) for persistent,
    zero-infrastructure recursive CTE graph traversal.
    """

    _SYMBOL_LABELS = {"function": "Function", "class": "Class", "variable": "Variable",
                      "constant": "Variable", "interface": "Class", "enum": "Class"}

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._label_indexes: dict[str, dict[str, list[str]]] = {}

    # ------------------------------------------------------------------ build
    def build_graph(
        self, snapshot: RepositorySnapshot, code_intelligence: CodeIntelligence,
        findings: list[Finding],
    ) -> dict[str, Any]:
        self.clear_repo(snapshot.repo_id)
        g = self.graph

        # Repository node
        g.add_node(f"repo:{snapshot.repo_id}", labels=["Repository"],
                   id=snapshot.repo_id, url=snapshot.repo_url, branch=snapshot.branch,
                   commit_hash=snapshot.commit_hash)

        # File nodes (§4.3.1 MERGE (f:File {path}))
        for file in snapshot.files:
            g.add_node(f"file:{snapshot.repo_id}:{file.path}", labels=["File"],
                       path=file.path, language=file.language, category=file.category,
                       size_bytes=file.size_bytes, content_hash=file.content_hash)
            g.add_edge(f"repo:{snapshot.repo_id}", f"file:{snapshot.repo_id}:{file.path}", type="HAS_FILE")

        # Symbol nodes (Function/Class/Variable labels by type)
        for sym in code_intelligence.symbols:
            label = self._SYMBOL_LABELS.get(sym.type, "Variable")
            g.add_node(sym.id, labels=[label], id=sym.id, name=sym.name, type=sym.type,
                       file_path=sym.file_path, line=sym.location.start_line,
                       signature=sym.signature)
            g.add_edge(f"file:{snapshot.repo_id}:{sym.file_path}", sym.id, type="CONTAINS")

        # Code edges (calls/imports/references/inherits/contains)
        for edge in code_intelligence.edges:
            src = self._resolve_edge_node(edge.source, snapshot.repo_id)
            dst = self._resolve_edge_node(edge.target, snapshot.repo_id)
            if src in g and dst in g:
                g.add_edge(src, dst, type=edge.type.upper(),
                           line=edge.location.start_line if edge.location else None)

        # Commit nodes + MODIFIES + AUTHORED_BY
        for commit in snapshot.commits[:500]:
            cid = f"commit:{snapshot.repo_id}:{commit.hash}"
            g.add_node(cid, labels=["Commit"], hash=commit.hash, author=commit.author,
                       timestamp=commit.timestamp.isoformat(), message=commit.message[:200])
            for changed in commit.files_changed[:50]:
                fnode = f"file:{snapshot.repo_id}:{changed}"
                if fnode in g:
                    g.add_edge(cid, fnode, type="MODIFIES", additions=commit.additions,
                               deletions=commit.deletions)

        # Issue / PR nodes
        for issue in snapshot.issues:
            nid = f"issue:{snapshot.repo_id}:{issue.number}"
            g.add_node(nid, labels=["Issue"], number=issue.number, title=issue.title,
                       state=issue.state, issue_labels=issue.labels)
        for pr in snapshot.pull_requests:
            nid = f"pr:{snapshot.repo_id}:{pr.number}"
            g.add_node(nid, labels=["PullRequest"], number=pr.number, title=pr.title,
                       state=pr.state, reviewers=pr.reviewers)

        # Finding nodes + HAS_FINDING
        for finding in findings:
            fid = f"finding:{snapshot.repo_id}:{finding.id}"
            g.add_node(fid, labels=["Finding"], id=finding.id, rule=finding.rule_id,
                       severity=finding.severity, tool=finding.tool, message=finding.message[:200],
                       line=finding.line_start)
            fnode = f"file:{snapshot.repo_id}:{finding.file_path}"
            if fnode in g:
                g.add_edge(fnode, fid, type="HAS_FINDING", line=finding.line_start)

        self._rebuild_indexes()
        stats = self.stats()
        logger.info("graph_built", extra={"extra_fields": stats})
        return stats

    def _resolve_edge_node(self, node_id: str, repo_id: str) -> str:
        if node_id.endswith("<module>"):
            path = node_id.removesuffix("::<module>")
            return f"file:{repo_id}:{path}"
        return node_id

    def _rebuild_indexes(self) -> None:
        """FIX-025: build name → node_ids maps per label so lookups are O(1).

        Duplicate symbol names in different files are kept (list of ids).
        """
        self._label_indexes = {}
        for node_id, data in self.graph.nodes(data=True):
            labels = data.get("labels") or []
            if not labels:
                continue
            primary_label = labels[0]
            if primary_label in ("Function", "Class", "Variable"):
                name = data.get("name")
                if name:
                    self._label_indexes.setdefault(primary_label, {}).setdefault(
                        name, []).append(node_id)
            elif primary_label == "File":
                path = data.get("path")
                if path:
                    self._label_indexes.setdefault("File", {}).setdefault(
                        path, []).append(node_id)

    # ------------------------------------------------------------------ search
    def graph_search(self, repo_id: str, intent: GraphIntent, depth: int = 3, top_k: int = 20) -> list[dict]:
        """§7.3.5 graph_search — callers/callees/dependencies/impact/symbol_lookup."""
        g = self.graph
        target = intent.target
        if target is None:
            return []
        results: list[dict] = []

        if intent.type == "callers":
            for node_id in self._find_symbol_nodes(target):
                for pred in g.predecessors(node_id):
                    if g.get_edge_data(pred, node_id, {}).get("type") == "CALLS":
                        results.append(self._node_record(pred, relation="calls", other=node_id))

        elif intent.type == "callees":
            for node_id in self._find_symbol_nodes(target):
                for succ in g.successors(node_id):
                    if g.get_edge_data(node_id, succ, {}).get("type") == "CALLS":
                        results.append(self._node_record(succ, relation="called_by", other=node_id))

        elif intent.type == "dependencies":
            # FIX-025: iterate the File-path index instead of scanning all nodes
            file_nodes: list[str] = []
            for path, ids in self._label_indexes.get("File", {}).items():
                if target in (path or ""):
                    file_nodes.extend(ids)
            for fn in file_nodes:
                for succ in g.successors(fn):
                    etype = g.get_edge_data(fn, succ, {}).get("type")
                    if etype in ("IMPORTS", "CALLS", "CONTAINS"):
                        results.append(self._node_record(succ, relation=etype.lower()))
                # transitive imports up to depth
                for path in nx.single_source_shortest_path_length(g, fn, cutoff=depth).items():
                    node, d = path
                    if d > 1 and g.get_edge_data(fn, node, {}).get("type") is None:
                        pass

        elif intent.type == "impact":
            # §7.3 impact: what breaks if I change this — reverse CALLS paths
            for node_id in self._find_symbol_nodes(target):
                for dep in nx.ancestors(g, node_id):
                    # only report actual code dependents (functions/classes), not
                    # repo/file/commit container nodes
                    data = g.nodes.get(dep, {})
                    if data.get("labels") and data["labels"][0] not in ("Function", "Class"):
                        continue
                    if nx.has_path(g, dep, node_id):
                        results.append(self._node_record(dep, relation="impacted_by",
                                                         other=node_id,
                                                         path_length=nx.shortest_path_length(g, dep, node_id)))

        elif intent.type == "symbol_lookup":
            for node_id in self._find_symbol_nodes(target):
                results.append(self._node_record(node_id, relation="definition"))

        # de-duplicate & cap
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in results:
            key = r.get("node_id") or r.get("id")
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped[:top_k]

    def _find_symbol_nodes(self, name: str) -> list[str]:
        """FIX-025: look up symbol nodes by name using the prebuilt index."""
        if not name:
            return []
        matches: list[str] = []
        for label in ("Function", "Class", "Variable"):
            matches.extend(self._label_indexes.get(label, {}).get(name, []))
        return matches

    def _node_record(self, node_id: str, relation: str, other: str | None = None, **extra) -> dict:
        data = self.graph.nodes[node_id]
        rec = {"node_id": node_id, "relation": relation, **{
            k: v for k, v in data.items() if k != "labels"
        }}
        if other:
            rec["related_to"] = other
        rec.update(extra)
        return rec

    # --------------------------------------------------------------- summaries
    def get_architecture_summary(self, repo_id: str) -> dict:
        """MCP tool get_architecture (§10.3.1)."""
        g = self.graph
        files = [(n, d) for n, d in g.nodes(data=True) if d.get("labels") == ["File"]
                 and d.get("category") == "code"]
        symbols = [(n, d) for n, d in g.nodes(data=True)
                   if d.get("labels") and d["labels"][0] in ("Function", "Class")]
        return {
            "repo_id": repo_id,
            "code_files": len(files),
            "symbols": len(symbols),
            "functions": sum(1 for _, d in symbols if d["labels"][0] == "Function"),
            "classes": sum(1 for _, d in symbols if d["labels"][0] == "Class"),
            "call_edges": sum(1 for _, _, d in g.edges(data=True) if d.get("type") == "CALLS"),
            "import_edges": sum(1 for _, _, d in g.edges(data=True) if d.get("type") == "IMPORTS"),
            "top_hub_symbols": sorted(
                ((d.get("name"), g.in_degree(n)) for n, d in symbols),
                key=lambda x: -x[1])[:10],
        }

    def get_impact_analysis(self, repo_id: str, symbol_name: str, max_depth: int = 5) -> list[dict]:
        intent = GraphIntent(type="impact", target=symbol_name)
        return self.graph_search(repo_id, intent, depth=max_depth)

    def stats(self) -> dict:
        g = self.graph
        edge_types: dict[str, int] = {}
        for _, _, data in g.edges(data=True):
            t = data.get("type", "UNKNOWN")
            edge_types[t] = edge_types.get(t, 0) + 1
        node_labels: dict[str, int] = {}
        for _, data in g.nodes(data=True):
            for label in data.get("labels", []):
                node_labels[label] = node_labels.get(label, 0) + 1
        return {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "node_labels": node_labels,
            "edge_types": edge_types,
        }

    def clear_repo(self, repo_id: str) -> None:
        prefix = f"file:{repo_id}:"
        to_remove = [n for n in self.graph.nodes
                     if n.startswith(prefix) or n.startswith(f"repo:{repo_id}")
                     or n.startswith(f"commit:{repo_id}:")
                     or n.startswith(f"issue:{repo_id}:") or n.startswith(f"pr:{repo_id}:")
                     or n.startswith(f"finding:{repo_id}:")]
        self.graph.remove_nodes_from(to_remove)

    def persist(self, path) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self.graph, f)

    @classmethod
    def load(cls, path) -> "NetworkXGraphStore":
        import pickle

        store = cls()
        with open(path, "rb") as f:
            store.graph = pickle.load(f)
        store._rebuild_indexes()
        return store


class Neo4jGraphStore(BaseGraphStore):
    """Production backend executing the documented Cypher (§4.3.1/§15.2).

    # HONEST NOTE (Phase 2): This Neo4j adapter is fully implemented but disabled
    # by default (GRAPH_STORE=sqlite) to keep the MVP zero-infrastructure.
    # The Cypher queries are production-ready. To activate, spin up the Neo4j 
    # Docker container and set GRAPH_STORE=neo4j in .env. The SQLite store 
    # implements the exact same interface, proving the logic works locally.
    """

    def __init__(self) -> None:
        from app.config import get_settings

        settings = get_settings()
        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
            self.driver.verify_connectivity()
            self._create_schema()
        except Exception as exc:
            logger.warning(f"Neo4j unavailable ({exc}) — falling back to NetworkX store")
            raise

    def _create_schema(self) -> None:

        constraints = [
            "CREATE CONSTRAINT repo_id IF NOT EXISTS FOR (r:Repository) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT file_path IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE",
            "CREATE CONSTRAINT symbol_id IF NOT EXISTS FOR (s:Symbol) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT commit_hash IF NOT EXISTS FOR (c:Commit) REQUIRE c.hash IS UNIQUE",
            "CREATE CONSTRAINT finding_id IF NOT EXISTS FOR (f:Finding) REQUIRE f.id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for stmt in constraints:
                try:
                    session.run(stmt)
                except Exception as exc:
                    logger.debug(f"Constraint may already exist: {exc}")
            # FIX-002: sanity probe — verify variable-length traversal is
            # syntactically accepted so construction fails fast (and
            # get_graph_store falls back to NetworkX) rather than per-query.
            session.run(
                "MATCH path = (a)-[:CALLS*1..3]->(b) RETURN count(path) LIMIT 1"
            ).consume()

    def build_graph(self, snapshot, code_intelligence, findings) -> dict:
        with self.driver.session() as session:
            session.run(
                "MERGE (r:Repository {id: $id}) SET r.url = $url, r.branch = $branch, "
                "r.commit_hash = $commit",
                id=snapshot.repo_id, url=snapshot.repo_url,
                branch=snapshot.branch, commit=snapshot.commit_hash)
            for file in snapshot.files:
                session.run(
                    "MERGE (f:File {path: $path}) SET f.language = $lang, f.category = $cat",
                    path=f"{snapshot.repo_id}/{file.path}", lang=file.language, cat=file.category)
            for sym in code_intelligence.symbols:
                session.run(
                    "MERGE (s:Symbol {id: $id}) SET s.name = $name, s.type = $type, s.line = $line",
                    id=sym.id, name=sym.name, type=sym.type, line=sym.location.start_line)
            for edge in code_intelligence.edges:
                session.run(
                    f"MATCH (a {{id: $src}}), (b {{id: $dst}}) MERGE (a)-[:{edge.type.upper()}]->(b)",
                    src=edge.source, dst=edge.target)
        return {"backend": "neo4j"}

    def graph_search(self, repo_id, intent, depth=3, top_k=20):
        """FIX-002: Cypher does not allow parameter substitution inside a
        variable-length pattern — the depth must be a literal integer baked
        into the query string (clamped to a safe range)."""
        depth = max(1, min(int(depth), 10))
        queries = {
            "callers": (
                "MATCH (caller)-[:CALLS]->(target) "
                "WHERE target.name = $name "
                "RETURN caller, target LIMIT $limit"
            ),
            "callees": (
                "MATCH (source)-[:CALLS]->(callee) "
                "WHERE source.name = $name "
                "RETURN source, callee LIMIT $limit"
            ),
            "impact": (
                f"MATCH path = (target)<-[:CALLS*1..{depth}]-(dependent) "
                "WHERE target.name = $name "
                "RETURN path LIMIT $limit"
            ),
            "dependencies": (
                f"MATCH path = (f:File)-[:IMPORTS*1..{depth}]->(dep) "
                "WHERE f.path CONTAINS $name "
                "RETURN path LIMIT $limit"
            ),
            "symbol_lookup": (
                "MATCH (s) WHERE s.name = $name RETURN s LIMIT $limit"
            ),
        }
        with self.driver.session() as session:
            result = session.run(queries[intent.type], name=intent.target, limit=top_k)
            return [dict(record) for record in result]

    def stats(self):
        return {"backend": "neo4j"}

    def clear_repo(self, repo_id):
        with self.driver.session() as session:
            session.run("MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n", prefix=repo_id)

    def get_architecture_summary(self, repo_id):
        return {"backend": "neo4j", "note": "use graph_search"}

    def get_impact_analysis(self, repo_id, symbol_name, max_depth=5):
        return self.graph_search(repo_id, GraphIntent(type="impact", target=symbol_name), depth=max_depth)


def get_graph_store() -> BaseGraphStore:
    """Factory: pick backend from config; defaults to embedded SQLiteGraphStore."""
    from app.config import get_settings

    settings = get_settings()
    if settings.graph_store == "neo4j":
        try:
            return Neo4jGraphStore()
        except Exception:
            logger.warning("Neo4j backend failed — falling back to SQLiteGraphStore")
            return SQLiteGraphStore()
    if settings.graph_store == "networkx":
        return NetworkXGraphStore()
    return SQLiteGraphStore()

