"""Symbol + reference extraction (§4.2.2 Component B — the 'GitNexus' role).

Extracts Symbols (function/class/variable/constant), References (calls,
imports, inheritance) and builds resolved Edges including cross-file links,
directly from Tree-sitter ASTs.
"""
from __future__ import annotations

import re

from app.models.code import (
    ASTNode, CodeIntelligence, Edge, Reference, SourceLocation, Symbol,
)
from app.core.parsing.tree_sitter_parser import iter_nodes, node_text
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------- language config
SYMBOL_KINDS = {
    "python": {
        "function": {"function_definition", "decorated_definition"},
        "class": {"class_definition"},
        "variable": {"assignment"},
        "constant": {"assignment"},
    },
    "javascript": {
        "function": {"function_declaration", "method_definition", "arrow_function", "function_expression"},
        "class": {"class_declaration", "class"},
        "variable": {"variable_declarator"},
    },
    "typescript": {
        "function": {"function_declaration", "method_definition", "arrow_function", "function_expression"},
        "class": {"class_declaration", "class"},
        "variable": {"variable_declarator"},
        "interface": {"interface_declaration"},
    },
}

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CONST_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
CAMEL_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")


class SymbolExtractor:
    """Extract symbols and references from a single AST (§4.2.2)."""

    def extract_symbols(self, ast: ASTNode) -> list[Symbol]:
        """Functions, classes, variables (§3.2.2 extract_symbols)."""
        if ast.root is None:
            return []
        source_bytes = ast.source.encode("utf-8")
        config = SYMBOL_KINDS.get(ast.language, {})
        symbols: list[Symbol] = []
        enclosing: dict[int, str] = {}  # node id → enclosing symbol id

        for node in iter_nodes(ast.root):
            kind = node.type
            parent_id = self._nearest_enclosing(node, enclosing)

            name = None
            sym_type = None
            signature = None
            docstring = None

            if kind in config.get("class", set()) or kind == "class":
                name = self._child_text(node, "name", source_bytes)
                if kind == "class_definition" or kind == "class_declaration":
                    bases = [node_text(c, source_bytes) for c in node.children
                             if c.type in ("argument_list", "superclass", "heritage")]
                    signature = f"class {name}" + (f"({', '.join(bases)})" if bases else "")
                sym_type = "class"

            elif kind == "decorated_definition" and ast.language == "python":
                # handle decorated functions: find the inner function
                for child in node.children:
                    if child.type == "function_definition":
                        inner_name = self._child_text(child, "name", source_bytes)
                        if inner_name:
                            symbols.append(self._make_symbol(
                                ast, inner_name, "function", child, parent_id, source_bytes,
                                signature=self._signature(child, inner_name, source_bytes),
                            ))
                        break
                continue

            elif kind in config.get("function", set()) and kind != "decorated_definition":
                if kind == "method_definition" or kind == "function_definition":
                    name = self._child_text(node, "name", source_bytes)
                    signature = self._signature(node, name, source_bytes)
                elif kind in ("arrow_function", "function_expression"):
                    # best-effort: variable name from parent declarator
                    name = self._parent_declarator_name(node, source_bytes)
                else:
                    name = self._child_text(node, "name", source_bytes)
                sym_type = "function"

            elif kind in config.get("interface", set()):
                name = self._child_text(node, "name", source_bytes)
                sym_type = "interface"

            elif kind in config.get("variable", set()) and ast.language == "python":
                # only top-level module variables & constants
                if node.parent is not None and node.parent.type not in ("module", "class_definition", "function_definition"):
                    continue
                lhs_node = self._first_child(node, "identifier", "left")
                if lhs_node is None:
                    continue
                left = node_text(lhs_node, source_bytes)
                m = IDENT_RE.match(left)
                if not m:
                    continue
                name = m.group(0)
                sym_type = "constant" if CONST_RE.match(name) else "variable"

            if name and sym_type:
                symbol = self._make_symbol(ast, name, sym_type, node, parent_id, source_bytes, signature, docstring)
                symbols.append(symbol)
                enclosing[id(node)] = symbol.id

        return symbols

    # ---------------------------------------------------------------- references
    def extract_references(self, ast: ASTNode, symbols: list[Symbol]) -> list[Reference]:
        """Where symbols are used (§3.2.2 extract_references).

        Calls are attributed to their ENCLOSING function/method so the graph
        gets function→function CALLS edges (needed for impact analysis, §7.3).
        """
        if ast.root is None:
            return []
        source_bytes = ast.source.encode("utf-8")
        module_id = f"{ast.file_path}::<module>"
        refs: list[Reference] = []
        seen: set[tuple[str, int]] = set()

        FUNCTION_NODES = ("function_definition", "function_declaration",
                          "method_definition", "arrow_function", "function_expression")

        def walk(node, enclosing: str | None):
            sym_id = enclosing
            # a function/class node itself becomes the enclosing scope
            if node.type in FUNCTION_NODES or node.type in ("class_definition", "class_declaration", "class"):
                nm = self._child_text(node, "name", source_bytes) or self._parent_declarator_name(node, source_bytes)
                if nm:
                    for s in symbols:
                        if s.name == nm and s.file_path == ast.file_path:
                            sym_id = s.id
                            break
            if node.type in ("decorated_definition",):
                sym_id = sym_id or enclosing

            if node.type in ("call", "call_expression"):
                func_node = self._first_child(node, "identifier", "function")
                if func_node is not None:
                    target_name = node_text(func_node, source_bytes)
                    key = ("calls", node.start_byte)
                    if key not in seen:
                        seen.add(key)
                        refs.append(self._make_reference(ast, sym_id or module_id, target_name, "calls", node, source_bytes))

            elif node.type in ("import_statement", "import_from_statement", "import_declaration"):
                target_name = self._import_module_name(node, source_bytes, ast.language)
                if target_name:
                    key = ("imports", node.start_byte)
                    if key not in seen:
                        seen.add(key)
                        refs.append(self._make_reference(ast, module_id, target_name, "imports", node, source_bytes))

            elif node.type in ("superclass", "heritage") and node.parent is not None:
                if node.parent.type in ("class_definition", "class_declaration", "class"):
                    for n in IDENT_RE.findall(node_text(node, source_bytes)):
                        if CAMEL_RE.match(n):
                            refs.append(self._make_reference(ast, sym_id or module_id, n, "inherits", node, source_bytes))

            for child in node.children:
                walk(child, sym_id)

        walk(ast.root, None)
        return refs

    # ---------------------------------------------------------------------- edges
    def build_code_intelligence(self, asts: list[ASTNode], symbols_by_file: dict[str, list[Symbol]],
                                 refs_by_file: dict[str, list[Reference]]) -> CodeIntelligence:
        """§4.2.2 extract_code_intelligence — resolve edges + cross-file links."""
        all_symbols = [s for lst in symbols_by_file.values() for s in lst]
        all_refs = [r for lst in refs_by_file.values() for r in lst]

        # name → symbol resolution table (cross-file)
        by_name: dict[str, Symbol] = {}
        by_module: dict[str, list[Symbol]] = {}
        for s in all_symbols:
            by_name.setdefault(s.name, s)
            module = s.file_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            by_module.setdefault(module, []).append(s)

        edges: list[Edge] = []
        for ref in all_refs:
            target: Symbol | None = None
            if ref.type == "imports":
                module_name = ref.target_name.split(".")[0].removesuffix(".py")
                candidates = by_module.get(module_name, [])
                if candidates:
                    # module-level edge: file → file
                    src_module = ref.source_symbol
                    for c in candidates:
                        edges.append(Edge(
                            source=src_module if src_module.endswith("<module>") else f"{ref.id}::site",
                            target=c.id, type="imports",
                        ))
                continue

            target = by_name.get(ref.target_name)
            if target is not None and target.id != ref.source_symbol:
                ref.target_symbol = target.id
                edges.append(Edge(source=ref.source_symbol, target=target.id, type=ref.type,
                                  location=ref.location))

        # contains edges: module → top-level symbols
        for file_path, syms in symbols_by_file.items():
            module_id = f"{file_path}::<module>"
            for s in syms:
                edges.append(Edge(source=module_id, target=s.id, type="contains"))

        logger.info("code_intelligence_built", extra={"extra_fields": {
            "symbols": len(all_symbols), "references": len(all_refs), "edges": len(edges),
        }})
        return CodeIntelligence(symbols=all_symbols, references=all_refs, edges=edges)

    # ------------------------------------------------------------------- helpers
    def _make_symbol(self, ast, name, sym_type, node, parent_id, source_bytes, signature=None, docstring=None):
        loc = SourceLocation(
            file_path=ast.file_path,
            start_line=node.start_point[0] + 1, start_column=node.start_point[1],
            end_line=node.end_point[0] + 1, end_column=node.end_point[1],
        )
        sid = f"{ast.file_path}::{name}::{loc.start_line}"
        return Symbol(id=sid, name=name, type=sym_type, file_path=ast.file_path,
                      location=loc, parent=parent_id, signature=signature, docstring=docstring)

    def _make_reference(self, ast, source_symbol, target_name, ref_type, node, source_bytes):
        loc = SourceLocation(
            file_path=ast.file_path,
            start_line=node.start_point[0] + 1, start_column=node.start_point[1],
            end_line=node.end_point[0] + 1, end_column=node.end_point[1],
        )
        return Reference(
            id=f"{ast.file_path}::{loc.start_line}::{target_name}",
            source_symbol=source_symbol, target_symbol=None,
            target_name=target_name, type=ref_type, location=loc,
        )

    def _nearest_enclosing(self, node, enclosing: dict[int, str]) -> str | None:
        current = node.parent
        while current is not None:
            if id(current) in enclosing:
                return enclosing[id(current)]
            current = current.parent
        return None

    def _child_text(self, node, field: str, source_bytes) -> str | None:
        child = node.child_by_field_name(field)
        if child is None:
            for c in node.children:
                if c.type in (field, "property_identifier"):
                    return node_text(c, source_bytes)
            return None
        return node_text(child, source_bytes)

    def _first_child(self, node, *types: str):
        for c in node.children:
            if c.type in types:
                return c
            for gc in c.children:
                if gc.type in types:
                    return gc
        return None

    def _parent_declarator_name(self, node, source_bytes) -> str | None:
        parent = node.parent
        while parent is not None:
            if parent.type in ("variable_declarator", "assignment"):
                name_node = parent.child_by_field_name("name") or parent.child_by_field_name("left")
                if name_node is not None:
                    text = node_text(name_node, source_bytes)
                    m = IDENT_RE.match(text)
                    return m.group(0) if m else None
            parent = parent.parent
        return None

    def _signature(self, node, name: str | None, source_bytes) -> str | None:
        if name is None:
            return None
        params_node = node.child_by_field_name("parameters")
        if params_node is not None:
            params = node_text(params_node, source_bytes)
            return f"{name}{params}"
        return f"{name}(…)"

    def _import_module_name(self, node, source_bytes, language) -> str | None:
        text = node_text(node, source_bytes)
        m = re.search(r"from\s+['\"]?([\w\.]+)", text) or re.search(r"import\s+['\"]?([\w\./]+)", text)
        if m:
            return m.group(1)
        return None
