"""Tree-sitter AST extraction (§4.2.1 Component A).

Supports Python, JavaScript, TypeScript out of the box; additional grammars
register via GRAMMARS. Implements the documented error handling:
- unsupported language → skip, log, continue
- parse error → record partial AST, flag file
- empty file → skip
- file too large (>1MB) → skip AST, record metadata
"""
from __future__ import annotations

from typing import Iterator

from app.models.code import ASTNode
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import tree_sitter_python  # type: ignore
    import tree_sitter_javascript  # type: ignore
    import tree_sitter_typescript as ts_typescript  # type: ignore
    from tree_sitter import Language, Parser

    _PY = Language(tree_sitter_python.language())
    _JS = Language(tree_sitter_javascript.language())
    _TSX = Language(ts_typescript.language_typescript())
    _GRAMMARS: dict[str, Language] = {"python": _PY, "javascript": _JS, "typescript": _TSX}
    TREE_SITTER_AVAILABLE = True
except Exception as exc:  # pragma: no cover - grammar pack missing
    logger.warning(f"Tree-sitter grammars unavailable ({exc}); AST parsing degraded")
    _GRAMMARS = {}
    TREE_SITTER_AVAILABLE = False

MAX_AST_FILE_BYTES = 1024 * 1024  # §4.2.1: file too large (>1MB) → skip AST


def _iter_nodes(node) -> "Iterator":
    """Depth-first iteration over tree-sitter nodes."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        for child in reversed(current.children):
            stack.append(child)


class ParsingComponent:
    """Concrete ParsingComponent (§3.2.2): parse_file / extract helpers."""

    def parse_file(self, file_path: str, language: str, source: str) -> ASTNode | None:
        """§4.2.1 extract_ast — returns ASTNode or None when skipped."""
        if not source.strip():
            logger.debug(f"Empty file, skipping: {file_path}")
            return None
        if len(source.encode("utf-8", errors="ignore")) > MAX_AST_FILE_BYTES:
            logger.warning(f"File too large for AST, skipping: {file_path}")
            return None
        lang = _GRAMMARS.get(language)
        if lang is None:
            logger.debug(f"Unsupported language '{language}' for {file_path}")
            return None

        parser = Parser(lang)
        tree = parser.parse(source.encode("utf-8"))
        root = tree.root_node

        node_count = 0
        depth = 0
        for _ in _iter_nodes(root):
            node_count += 1
        depth = _max_depth(root)

        return ASTNode(
            file_path=file_path,
            language=language,
            root=root,
            source=source,
            node_count=node_count,
            depth=depth,
            has_errors=bool(root.has_error),
        )

    def supported_languages(self) -> list[str]:
        return sorted(_GRAMMARS.keys())


def _max_depth(node, depth: int = 1) -> int:
    if not node.children:
        return depth
    return max(_max_depth(child, depth + 1) for child in node.children)


def node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def iter_nodes(root):
    """Public depth-first node iterator."""
    return _iter_nodes(root)
