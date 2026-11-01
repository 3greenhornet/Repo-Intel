"""AST Merkle Tree Fingerprinting for Incremental Indexing (§Task 2).

Replaces raw SHA-256 file hashing with structural AST fingerprinting.
When a file is modified (e.g. comments, docstrings, formatting changes),
AST Merkle hashes remain identical, preventing unnecessary re-parsing,
re-embedding, and vector re-indexing. Only functions and classes with
genuine syntactic or control-flow changes trigger re-indexing.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

from app.core.ingestion.file_walker import read_file_content
from app.core.ingestion.git_parser import (
    get_changed_files,
    get_head_commit,
    incremental_update,
)
from app.core.parsing.tree_sitter_parser import ParsingComponent, iter_nodes
from app.core.storage.sqlite_graph_store import SQLiteGraphStore
from app.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "get_changed_files",
    "get_head_commit",
    "incremental_update",
    "compute_ast_structural_hash",
    "compute_ast_fingerprint",
    "extract_ast_subtrees",
    "find_structurally_changed_nodes",
    "filter_structurally_modified_files",
    "IncrementalASTIndexer",
]


def compute_ast_fingerprint(node: Any) -> str:
    """Computes a structural hash of the AST, ignoring comments/whitespace (§Task 7 & Etch Detail 3)."""
    return compute_ast_structural_hash(node)


def compute_ast_structural_hash(node: Any) -> str:
    """Generate a stable structural Merkle hash for an AST subtree.

    Includes node types and structural hierarchy; ignores comment tokens,
    raw string values, formatting, and docstrings.
    """
    # 1. Tree-sitter node
    if hasattr(node, "type") and hasattr(node, "children"):
        tokens: list[str] = []
        stack = [node]
        while stack:
            curr = stack.pop()
            ntype = curr.type
            # Skip comments and whitespace tokens
            if "comment" in ntype or ntype in ("\n", " ", ""):
                continue
            child_types = [c.type for c in curr.children if "comment" not in c.type]
            tokens.append(f"{ntype}:[{','.join(child_types)}]")
            for c in reversed(curr.children):
                stack.append(c)
        return hashlib.sha256(";".join(tokens).encode("utf-8")).hexdigest()

    # 2. Python standard library `ast.AST` fallback
    if isinstance(node, ast.AST):
        tokens: list[str] = []
        for child in ast.walk(node):
            # Skip docstrings/string constants and comments
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                continue
            child_fields = [f for f, _ in ast.iter_fields(child)]
            tokens.append(f"{type(child).__name__}:[{','.join(child_fields)}]")
        return hashlib.sha256(";".join(tokens).encode("utf-8")).hexdigest()

    return hashlib.sha256(str(node).encode("utf-8")).hexdigest()


def extract_ast_subtrees(file_path: str, language: str, source_code: str) -> list[dict[str, Any]]:
    """Extract top-level functions and classes with their AST Merkle hashes."""
    subtrees: list[dict[str, Any]] = []

    # Try Tree-sitter first
    parser = ParsingComponent()
    ast_tree = parser.parse_file(file_path, language, source_code)

    if ast_tree and ast_tree.root:
        for node in iter_nodes(ast_tree.root):
            if node.type in ("function_definition", "class_definition", "function_declaration", "class_declaration"):
                # Extract identifier child
                name = ""
                for c in node.children:
                    if c.type in ("identifier", "name"):
                        name = source_code.encode("utf-8")[c.start_byte:c.end_byte].decode("utf-8", errors="ignore")
                        break
                if not name:
                    name = f"anonymous_{node.start_point[0]}"

                entity_type = "Class" if "class" in node.type else "Function"
                s_hash = compute_ast_structural_hash(node)
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                node_code = source_code.encode("utf-8")[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

                subtrees.append({
                    "id": f"{file_path}::{name}",
                    "qualified_name": name,
                    "entity_type": entity_type,
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "structural_hash": s_hash,
                    "source_code": node_code,
                })
        if subtrees:
            return subtrees

    # Fallback to Python standard library `ast`
    if language == "python" or file_path.endswith(".py"):
        try:
            tree = ast.parse(source_code)
            lines = source_code.splitlines()
            for child in ast.iter_child_nodes(tree):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    s_hash = compute_ast_structural_hash(child)
                    start_line = getattr(child, "lineno", 1)
                    end_line = getattr(child, "end_lineno", start_line)
                    node_code = "\n".join(lines[start_line - 1:end_line])
                    subtrees.append({
                        "id": f"{file_path}::{child.name}",
                        "qualified_name": child.name,
                        "entity_type": "Class" if isinstance(child, ast.ClassDef) else "Function",
                        "file_path": file_path,
                        "start_line": start_line,
                        "end_line": end_line,
                        "structural_hash": s_hash,
                        "source_code": node_code,
                    })
        except SyntaxError:
            pass

    return subtrees


def find_structurally_changed_nodes(
    repo_id: str,
    file_path: str,
    new_source: str,
    language: str,
    graph_store: SQLiteGraphStore,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compare AST Merkle hashes of current code against hashes in SQLite `nodes` table.

    Returns:
        (changed_or_new_nodes, deleted_node_ids)
    """
    current_subtrees = extract_ast_subtrees(file_path, language, new_source)

    # Fetch stored hashes from SQLite
    cursor = graph_store.conn.cursor()
    cursor.execute(
        "SELECT id, qualified_name, structural_hash FROM nodes WHERE file_path = ?",
        (file_path,),
    )
    stored = {r["id"]: r["structural_hash"] for r in cursor.fetchall()}

    changed_nodes: list[dict[str, Any]] = []
    current_ids: set[str] = set()

    for item in current_subtrees:
        node_id = item["id"]
        current_ids.add(node_id)
        stored_hash = stored.get(node_id)
        if stored_hash is None or stored_hash != item["structural_hash"]:
            # Structural change detected
            changed_nodes.append(item)
            logger.debug(f"AST Merkle divergence on {node_id}: {stored_hash} -> {item['structural_hash']}")

    deleted_ids = [nid for nid in stored if nid not in current_ids]
    return changed_nodes, deleted_ids


class IncrementalASTIndexer:
    """Orchestrates granular AST-level re-indexing over changed files."""

    def __init__(self, graph_store: SQLiteGraphStore | None = None) -> None:
        self.graph_store = graph_store or SQLiteGraphStore()

    def process_file_diff(
        self,
        repo_id: str,
        repo_path: Path | str,
        file_path: str,
        language: str = "python",
    ) -> dict[str, Any]:
        content, ok = read_file_content(repo_path, file_path)
        if not ok:
            return {"status": "read_failed", "file": file_path}

        changed_nodes, deleted_ids = find_structurally_changed_nodes(
            repo_id, file_path, content, language, self.graph_store
        )

        # Update SQLite nodes table with updated structural hashes
        for node in changed_nodes:
            self.graph_store.add_node(
                node_id=node["id"],
                qualified_name=node["qualified_name"],
                entity_type=node["entity_type"],
                file_path=node["file_path"],
                source_code=node["source_code"],
                structural_hash=node["structural_hash"],
            )

        if deleted_ids:
            with self.graph_store.conn:
                placeholders = ",".join("?" for _ in deleted_ids)
                self.graph_store.conn.execute(
                    f"DELETE FROM nodes WHERE id IN ({placeholders})",
                    deleted_ids,
                )

        return {
            "file": file_path,
            "total_subtrees": len(changed_nodes),
            "reindexed_nodes": [n["id"] for n in changed_nodes],
            "deleted_nodes": deleted_ids,
        }


def filter_structurally_modified_files(
    repo_path: Path | str,
    modified_files: list[Any],
    db: Any,
    repo_id: str,
) -> list[Any]:
    """Etch Detail 3: Skip re-embedding files where only comments/docstrings/formatting changed."""
    parser = ParsingComponent()
    structurally_changed = []

    for file_change in modified_files:
        f_path = getattr(file_change, "path", str(file_change))
        content, ok = read_file_content(repo_path, f_path)
        if not ok:
            structurally_changed.append(file_change)
            continue

        ext = Path(f_path).suffix.lstrip(".")
        lang = "python" if ext == "py" else ("javascript" if ext in ("js", "jsx") else ("typescript" if ext in ("ts", "tsx") else ext))
        ast_node = parser.parse_file(f_path, lang, content)
        if ast_node and ast_node.root:
            new_fp = compute_ast_fingerprint(ast_node.root)
            try:
                cursor = db.conn.cursor()
                cursor.execute("SELECT content_hash FROM files WHERE repo_id = ? AND path = ?", (repo_id, f_path))
                row = cursor.fetchone()
                if row and row[0] and row[0].startswith("ast:") and row[0][4:] == new_fp:
                    logger.info(f"Skipping re-embedding for {f_path}: AST fingerprint unchanged (comments/formatting only)")
                    continue
                db.conn.execute("UPDATE files SET content_hash = ? WHERE repo_id = ? AND path = ?", (f"ast:{new_fp}", repo_id, f_path))
                db.conn.commit()
            except Exception:
                pass

        structurally_changed.append(file_change)

    return structurally_changed

