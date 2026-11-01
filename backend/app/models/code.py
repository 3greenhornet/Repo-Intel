"""Code intelligence data models (§4.2.2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SourceLocation:
    """A source span (§4.2.1)."""

    file_path: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int


@dataclass
class Symbol:
    """A code symbol: function / class / variable / constant (§4.2.2)."""

    id: str  # unique: file_path::name::line
    name: str
    type: str  # function, class, variable, constant, module, interface, enum
    file_path: str
    location: SourceLocation
    parent: Optional[str] = None  # enclosing symbol id
    signature: Optional[str] = None
    docstring: Optional[str] = None


@dataclass
class Reference:
    """A symbol usage site (§4.2.2)."""

    id: str
    source_symbol: str  # symbol id
    target_symbol: Optional[str]  # resolved symbol id
    target_name: str  # unresolved name
    type: str  # calls, imports, references, inherits
    location: SourceLocation


@dataclass
class Edge:
    """A code-graph edge (§4.2.2 Edge Types table)."""

    source: str  # symbol id
    target: str  # symbol id
    type: str  # calls, imports, references, inherits, implements, contains, modifies
    location: Optional[SourceLocation] = None


@dataclass
class ASTNode:
    """Tree-sitter parse result wrapper (§4.2.1)."""

    file_path: str
    language: str
    root: Any  # tree_sitter.Node (kept transient; not serialized)
    source: str
    node_count: int
    depth: int
    has_errors: bool = False

    # Transient objects cannot be deep-copied/pickled — exclude root.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["root"] = None
        return state


@dataclass
class CodeIntelligence:
    """Aggregated symbols / references / edges (§4.2.2)."""

    symbols: list[Symbol] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        for s in self.symbols:
            by_type[s.type] = by_type.get(s.type, 0) + 1
        edge_types: dict[str, int] = {}
        for e in self.edges:
            edge_types[e.type] = edge_types.get(e.type, 0) + 1
        return {
            "symbols": len(self.symbols),
            "references": len(self.references),
            "edges": len(self.edges),
            "symbols_by_type": by_type,
            "edges_by_type": edge_types,
        }
