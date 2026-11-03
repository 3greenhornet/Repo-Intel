"""Gap analysis — what's MISSING from a codebase, not just what's wrong.

Analyzers:
1. Test coverage gap — which modules have no corresponding test coverage or test file imports
2. Documentation gap — which public functions/classes have no docstrings
3. Incomplete features — NotImplementedError / pass-only / stub bodies / TODO markers
4. Error handling gap — network/filesystem/database I/O calls not wrapped in try/except
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from app.core.ingestion.file_walker import read_file_content
from app.models.finding import Finding
from app.models.repository import FileRecord
from app.utils.logger import get_logger

logger = get_logger(__name__)

IO_CALLS = {
    "open",
    "requests.get", "requests.post", "requests.put", "requests.delete", "requests.request",
    "httpx.get", "httpx.post", "httpx.put", "httpx.delete", "httpx.request",
    "urllib.request.urlopen", "socket.socket",
    "subprocess.run", "subprocess.call", "subprocess.Popen", "subprocess.check_output",
    "os.system", "shutil.rmtree", "shutil.copy",
    "db.execute", "cursor.execute", "sqlite3.connect",
}

_STUB_PATTERNS = re.compile(
    r"\b(NotImplementedError|TODO|FIXME|coming soon|not yet implemented|placeholder)\b",
    re.I,
)


def analyze_gaps(
    repo_path: Path,
    files: list[FileRecord],
    code_intelligence: Any = None,
) -> list[Finding]:
    """Run all four gap analyzers and return a unified list of findings."""
    findings: list[Finding] = []
    code_files = [f for f in files if f.category == "code" and f.language == "python"]
    test_files = [f for f in files if f.category == "tests" or "test" in f.path.lower()]

    findings.extend(_test_coverage_gap(repo_path, code_files, test_files))
    findings.extend(_documentation_gap(repo_path, code_files, code_intelligence))
    findings.extend(_incomplete_features(repo_path, code_files))
    findings.extend(_error_handling_gap(repo_path, code_files))

    logger.info("gap_analysis_complete", extra={"extra_fields": {
        "findings": len(findings),
        "by_type": _count_by(findings, "finding_type"),
    }})
    return findings


# ------------------------------------------------------------- 1. test coverage
def _test_coverage_gap(
    repo_path: Path,
    code_files: list[FileRecord],
    test_files: list[FileRecord],
) -> list[Finding]:
    """Identify source modules without corresponding test imports or test files."""
    findings: list[Finding] = []
    covered: set[str] = set()

    for tf in test_files:
        content, ok = read_file_content(repo_path, tf.path)
        if not ok or not content:
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod_path = node.module.replace(".", "/")
                covered.add(mod_path + ".py")
                covered.add(mod_path)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    mod_path = alias.name.replace(".", "/")
                    covered.add(mod_path + ".py")
                    covered.add(mod_path)

    for f in code_files:
        norm_path = f.path.replace("\\", "/")
        module_path = norm_path.removesuffix(".py")
        base_name = Path(norm_path).stem

        # Skip __init__, migrations, setup, and root wrapper entrypoints
        if base_name in ("__init__", "setup", "conftest") or norm_path.startswith("migrations/"):
            continue

        # Match either direct path or submodule match
        covered_match = (
            norm_path in covered
            or module_path in covered
            or any(norm_path.endswith(c) for c in covered)
            or any(c.endswith(norm_path) for c in covered)
            or any(base_name in c for c in covered)
        )

        if not covered_match:
            findings.append(Finding(
                tool="gap_analyzer",
                rule_id="gap:untested-module",
                message=f"Module '{norm_path}' has no observed test file imports or coverage.",
                severity="major",
                finding_type="test_gap",
                file_path=norm_path,
                line_start=1,
                line_end=1,
                metadata={"gap_type": "test_coverage", "module": base_name},
            ))
    return findings


# ----------------------------------------------------------- 2. documentation
def _documentation_gap(
    repo_path: Path,
    code_files: list[FileRecord],
    code_intelligence: Any = None,
) -> list[Finding]:
    """Identify public functions and classes that lack docstrings."""
    findings: list[Finding] = []

    # Pathway A: use code_intelligence symbols if present
    if code_intelligence and hasattr(code_intelligence, "symbols") and code_intelligence.symbols:
        for sym in code_intelligence.symbols:
            if getattr(sym, "type", "") not in ("function", "class"):
                continue
            name = getattr(sym, "name", "")
            if name.startswith("_") or name.startswith("test_"):
                continue
            doc = getattr(sym, "docstring", "") or ""
            if doc.strip():
                continue
            file_p = getattr(sym, "file_path", "")
            if "/test" in file_p or "\\test" in file_p:
                continue

            loc = getattr(sym, "location", None)
            line_start = getattr(loc, "start_line", 1) if loc else 1
            line_end = getattr(loc, "end_line", line_start) if loc else line_start
            sym_id = getattr(sym, "id", None)

            findings.append(Finding(
                tool="gap_analyzer",
                rule_id="gap:missing-docstring",
                message=f"{sym.type.capitalize()} '{name}' has no docstring.",
                severity="minor",
                finding_type="doc_gap",
                file_path=file_p,
                line_start=line_start,
                line_end=line_end,
                symbol_id=sym_id,
                metadata={"gap_type": "documentation", "symbol": name},
            ))
        return findings

    # Pathway B: direct AST walk if code_intelligence not available
    for f in code_files:
        norm_path = f.path.replace("\\", "/")
        if "/test" in norm_path or "test_" in norm_path:
            continue
        content, ok = read_file_content(repo_path, f.path)
        if not ok or not content:
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_") or node.name.startswith("test_"):
                    continue
                doc = ast.get_docstring(node)
                if not doc or not doc.strip():
                    kind = "Class" if isinstance(node, ast.ClassDef) else "Function"
                    findings.append(Finding(
                        tool="gap_analyzer",
                        rule_id="gap:missing-docstring",
                        message=f"{kind} '{node.name}' has no docstring.",
                        severity="minor",
                        finding_type="doc_gap",
                        file_path=norm_path,
                        line_start=node.lineno,
                        line_end=node.lineno,
                        metadata={"gap_type": "documentation", "symbol": node.name},
                    ))
    return findings


# ------------------------------------------------------ 3. incomplete features
def _incomplete_features(repo_path: Path, code_files: list[FileRecord]) -> list[Finding]:
    """Detect stubs, raise NotImplementedError, pass-only bodies, and TODO markers."""
    findings: list[Finding] = []

    for f in code_files:
        norm_path = f.path.replace("\\", "/")
        content, ok = read_file_content(repo_path, f.path)
        if not ok or not content:
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Case 1: body is only `pass`
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                findings.append(_stub_finding(
                    f, node, "empty-body",
                    f"Function '{node.name}' has an empty body (only 'pass').",
                ))
                continue

            # Case 2: body raises NotImplementedError
            has_nie = False
            for stmt in node.body:
                if isinstance(stmt, ast.Raise):
                    exc = stmt.exc
                    name = ""
                    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                        name = exc.func.id
                    elif isinstance(exc, ast.Name):
                        name = exc.id
                    if "NotImplementedError" in name:
                        has_nie = True
                        break
            if has_nie:
                findings.append(_stub_finding(
                    f, node, "not-implemented",
                    f"Function '{node.name}' raises NotImplementedError.",
                ))
                continue

            # Case 3: body only returns None
            if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
                val = node.body[0].value
                if val is None or (isinstance(val, ast.Constant) and val.value is None):
                    findings.append(_stub_finding(
                        f, node, "return-none-stub",
                        f"Function '{node.name}' only returns None (stub body).",
                    ))

        # Case 4: TODO/FIXME inside function code (avoiding docstrings)
        lines = content.splitlines()
        current_func: str | None = None
        in_docstring = False

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if '"""' in stripped or "'''" in stripped:
                count = stripped.count('"""') + stripped.count("'''")
                if count % 2 == 1:
                    in_docstring = not in_docstring

            m = re.match(r"^\s*(?:async\s+)?def\s+(\w+)", line)
            if m:
                current_func = m.group(1)
                continue

            # If inside a function body and not within multiline docstring
            if current_func and not in_docstring and _STUB_PATTERNS.search(stripped):
                findings.append(Finding(
                    tool="gap_analyzer",
                    rule_id="gap:incomplete-marker",
                    message=f"Function '{current_func}' contains an incomplete marker: {stripped[:80]}",
                    severity="minor",
                    finding_type="incomplete_feature",
                    file_path=norm_path,
                    line_start=i,
                    line_end=i,
                    metadata={"gap_type": "incomplete", "marker": stripped[:40]},
                ))

    return findings


def _stub_finding(file: FileRecord, node: Any, kind: str, message: str) -> Finding:
    return Finding(
        tool="gap_analyzer",
        rule_id=f"gap:stub-{kind}",
        message=message,
        severity="major",
        finding_type="incomplete_feature",
        file_path=file.path.replace("\\", "/"),
        line_start=getattr(node, "lineno", 1),
        line_end=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
        metadata={"gap_type": "incomplete", "kind": kind},
    )


# ---------------------------------------------------------- 4. error handling
def _error_handling_gap(repo_path: Path, code_files: list[FileRecord]) -> list[Finding]:
    """I/O calls not wrapped in try/except within the same function."""
    findings: list[Finding] = []

    for f in code_files:
        norm_path = f.path.replace("\\", "/")
        # Test files purposefully don't catch exceptions to trigger test failures
        if "/test" in norm_path or "test_" in norm_path:
            continue

        content, ok = read_file_content(repo_path, f.path)
        if not ok or not content:
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Collect line numbers of try/except handlers
            try_ranges: list[tuple[int, int]] = []
            for child in ast.walk(node):
                if isinstance(child, ast.Try):
                    start = getattr(child, "lineno", 0)
                    end = getattr(child, "end_lineno", start)
                    try_ranges.append((start, end))

            # Find I/O calls in this function
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                call_name = _call_name(child)
                if not call_name or not _is_io_call(call_name):
                    continue

                line = getattr(child, "lineno", 0)
                if any(start <= line <= end for start, end in try_ranges):
                    continue

                findings.append(Finding(
                    tool="gap_analyzer",
                    rule_id="gap:unguarded-io",
                    message=f"'{call_name}()' called in '{node.name}' without try/except — unhandled I/O error could crash service.",
                    severity="minor",
                    finding_type="error_handling_gap",
                    file_path=norm_path,
                    line_start=line,
                    line_end=line,
                    metadata={"gap_type": "error_handling", "call": call_name},
                ))

    return findings


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        parts: list[str] = []
        cur: Any = node.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts)) if parts else None
    return None


def _is_io_call(name: str) -> bool:
    return any(name == io or name.endswith("." + io.split(".")[-1]) for io in IO_CALLS)


def _count_by(findings: list[Finding], attr: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in findings:
        key = getattr(f, attr, "unknown") or "unknown"
        out[key] = out.get(key, 0) + 1
    return out

