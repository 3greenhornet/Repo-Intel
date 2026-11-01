"""AST extraction façade — parse a set of code files into ASTNode list (§4.2.1)."""
from __future__ import annotations

from pathlib import Path

from app.core.ingestion.file_walker import read_file_content
from app.core.parsing.symbol_extractor import SymbolExtractor
from app.core.parsing.tree_sitter_parser import ParsingComponent, TREE_SITTER_AVAILABLE
from app.models.code import ASTNode, CodeIntelligence
from app.models.repository import FileRecord
from app.utils.logger import get_logger

logger = get_logger(__name__)


def parse_code_files(repo_path: Path | str, files: list[FileRecord]) -> list[ASTNode]:
    """Parse all category='code' files into ASTNodes (Component A driver)."""
    parser = ParsingComponent()
    asts: list[ASTNode] = []
    parsed, skipped = 0, 0
    for file in files:
        if file.category != "code" or not file.language:
            continue
        content, ok = read_file_content(repo_path, file.path)
        if not ok:
            continue
        ast = parser.parse_file(file.path, file.language, content)
        if ast is None:
            skipped += 1
            continue
        if ast.has_errors:
            logger.debug(f"Partial AST (parse errors): {file.path}")
        parsed += 1
        asts.append(ast)
    logger.info("ast_extraction_complete", extra={"extra_fields": {
        "parsed": parsed, "skipped": skipped, "tree_sitter": TREE_SITTER_AVAILABLE,
    }})
    return asts


def extract_code_intelligence(asts: list[ASTNode]) -> CodeIntelligence:
    """Component B driver: symbols → references → edges (§4.2.2 algorithm)."""
    extractor = SymbolExtractor()
    symbols_by_file: dict[str, list] = {}
    refs_by_file: dict[str, list] = {}

    for ast in asts:
        file_symbols = extractor.extract_symbols(ast)
        symbols_by_file[ast.file_path] = file_symbols
        file_refs = extractor.extract_references(ast, file_symbols)
        refs_by_file[ast.file_path] = file_refs

    return extractor.build_code_intelligence(asts, symbols_by_file, refs_by_file)
