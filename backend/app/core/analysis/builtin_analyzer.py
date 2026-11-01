"""Built-in static analyzer — "repointel" tool (gap-filler for §4.2.3).

2026 Structural Analysis Architecture:
- Tier 1: `ast-grep` (Rust-powered `ast_grep_py`) sub-15ms concrete syntax tree matching.
- Tier 2: Python `ast` standard library walker for exact semantic node checking.
- Tier 3: Regex fallbacks for non-compiling/broken code syntax.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

try:
    from ast_grep_py import SgRoot  # type: ignore
    HAS_AST_GREP = True
except ImportError:
    HAS_AST_GREP = False

from app.core.ingestion.file_walker import read_file_content
from app.models.code import CodeIntelligence
from app.models.finding import Finding
from app.models.repository import FileRecord
from app.utils.logger import get_logger

logger = get_logger(__name__)

SATD_PATTERNS = [
    (re.compile(r"TODO\(?([\w\- ]*)\)?", re.I), "TODO"),
    (re.compile(r"FIXME", re.I), "FIXME"),
    (re.compile(r"XXX[:\s]", re.I), "XXX"),
    (re.compile(r"HACK[:\s]", re.I), "HACK"),
    (re.compile(r"DEBT[:\s]", re.I), "DEBT"),
]

SECURITY_RULES = [
    ("repointel:security/sql-string-concat",
     re.compile(r"(SELECT|INSERT|UPDATE|DELETE).{0,80}?[%f].{0,80}?(FROM|INTO|SET|WHERE|VALUES)", re.I | re.S),
     "critical", "vulnerability",
     "Possible SQL injection: SQL query built via string concatenation/formatting. Use parameterized queries."),
    ("repointel:security/eval-usage",
     re.compile(r"\beval\s*\("),
     "critical", "vulnerability",
     "Use of eval() executes arbitrary code — remove or replace with ast.literal_eval."),
    ("repointel:security/exec-usage",
     re.compile(r"\bexec\s*\("),
     "critical", "vulnerability",
     "Use of exec() executes arbitrary code."),
    ("repointel:security/shell-true",
     re.compile(r"shell\s*=\s*True"),
     "major", "vulnerability",
     "subprocess with shell=True is unsafe with untrusted input."),
    ("repointel:security/hardcoded-secret",
     re.compile(r"(?i)(password|passwd|secret|api_key|apikey|token)\s*=\s*['\"][^'\"]{6,}['\"]"),
     "blocker", "vulnerability",
     "Hardcoded credential detected — move to environment variable or secret store."),
    ("repointel:security/weak-hash",
     re.compile(r"\b(hashlib\.(md5|sha1))\b|\"md5\"|'md5'"),
     "major", "vulnerability",
     "Weak hash algorithm (MD5/SHA1) — use SHA-256 or better."),
]

BUG_RULES = [
    ("repointel:bug/bare-except",
     re.compile(r"^\s*except\s*:"),
     "major", "bug",
     "Bare `except:` swallows all errors including KeyboardInterrupt/SystemExit."),
    ("repointel:bug/mutable-default-arg",
     re.compile(r"def\s+\w+\s*\([^)]*=\s*(\[\]|\{\})"),
     "major", "bug",
     "Mutable default argument is shared across calls — use None + default inside."),
    ("repointel:bug/eq-none",
     re.compile(r"[=!]=\s*None|None\s*[!=]="),
     "minor", "bug",
     "Comparison to None should use `is None` / `is not None`."),
]

COMPLEXITY_THRESHOLD = 10
LONG_FUNCTION_LINES = 50
DEEP_NESTING = 5
MANY_PARAMS = 7


class BuiltinAnalyzer:
    """AST/regex static analyzer producing Finding objects in documented format."""

    name = "repointel"

    def analyze_file(self, file: FileRecord, source: str, symbols_for_file: list) -> list[Finding]:
        findings: list[Finding] = []
        lines = source.splitlines()

        # ---- Tier 1 ast-grep & Tier 2 AST / Regex structural rules ------------
        findings.extend(self._ast_structural_rules(file, source, lines))

        # ---- SATD comments (§4.2.4) -------------------------------------------
        findings.extend(self._satd_findings(file, lines))

        # ---- per-symbol structural metrics -------------------------------------
        for sym in symbols_for_file:
            if sym.type not in ("function", "class"):
                continue
            body = "\n".join(lines[sym.location.start_line - 1: sym.location.end_line])
            findings.extend(self._symbol_metrics(file, sym, body))

        # ---- file-level metrics ------------------------------------------------
        findings.extend(self._file_metrics(file, source, lines, symbols_for_file))
        return findings

    # -------------------------------------------------------------------- rules
    def _ast_structural_rules(self, file: FileRecord, source: str, lines: list[str]) -> list[Finding]:
        """Multi-tier structural analysis: ast-grep (Rust) -> Python AST -> Regex."""
        if file.language != "python":
            out = []
            out.extend(self._regex_rules(file, lines, SECURITY_RULES))
            out.extend(self._regex_rules(file, lines, BUG_RULES))
            return out

        # 1. Try ast-grep (Rust-powered CST pattern matching)
        if HAS_AST_GREP:
            try:
                findings = self._ast_grep_rules(file, source)
                findings.extend(self._regex_rules(file, lines, [
                    r for r in SECURITY_RULES if r[0] in (
                        "repointel:security/sql-string-concat",
                        "repointel:security/hardcoded-secret",
                        "repointel:security/weak-hash",
                    )
                ]))
                return findings
            except Exception as exc:
                logger.debug(f"ast-grep execution failed, falling back to python ast: {exc}")

        # 2. Try Python AST walker (Standard library, 100% accurate, zero dependency)
        try:
            findings = self._python_ast_rules(file, source)
            findings.extend(self._regex_rules(file, lines, [
                r for r in SECURITY_RULES if r[0] in (
                    "repointel:security/sql-string-concat",
                    "repointel:security/hardcoded-secret",
                    "repointel:security/weak-hash",
                )
            ]))
            return findings
        except SyntaxError:
            out = []
            out.extend(self._regex_rules(file, lines, SECURITY_RULES))
            out.extend(self._regex_rules(file, lines, BUG_RULES))
            return out

    def _ast_grep_rules(self, file: FileRecord, source: str) -> list[Finding]:
        findings: list[Finding] = []
        sg = SgRoot(source, "python")
        root = sg.root()

        # Rule: bare except
        for node in root.find_all(pattern="except:"):
            rng = node.range()
            line = rng.start.line + 1
            findings.append(Finding(
                tool=self.name,
                rule_id="repointel:bug/bare-except",
                message="Bare `except:` swallows all errors including KeyboardInterrupt/SystemExit.",
                severity="major",
                finding_type="bug",
                file_path=file.path,
                line_start=line,
                line_end=rng.end.line + 1,
            ))

        # Rule: mutable default args
        for pat in (
            "def $NAME($$$PRE, $ARG = [], $$$POST): $$$BODY",
            "def $NAME($$$PRE, $ARG = {}, $$$POST): $$$BODY",
            "def $NAME($$$PRE, $ARG = set(), $$$POST): $$$BODY",
        ):
            for node in root.find_all(pattern=pat):
                rng = node.range()
                line = rng.start.line + 1
                findings.append(Finding(
                    tool=self.name,
                    rule_id="repointel:bug/mutable-default-arg",
                    message="Mutable default argument is shared across calls — use None + default inside.",
                    severity="major",
                    finding_type="bug",
                    file_path=file.path,
                    line_start=line,
                    line_end=rng.end.line + 1,
                ))

        # Rule: comparison to None
        for pat in ("$X == None", "None == $X", "$X != None", "None != $X"):
            for node in root.find_all(pattern=pat):
                rng = node.range()
                line = rng.start.line + 1
                findings.append(Finding(
                    tool=self.name,
                    rule_id="repointel:bug/eq-none",
                    message="Comparison to None should use `is None` / `is not None`.",
                    severity="minor",
                    finding_type="bug",
                    file_path=file.path,
                    line_start=line,
                    line_end=rng.end.line + 1,
                ))

        # Rule: eval & exec usage
        for node in root.find_all(pattern="eval($$$ARGS)"):
            rng = node.range()
            findings.append(Finding(
                tool=self.name,
                rule_id="repointel:security/eval-usage",
                message="Use of eval() executes arbitrary code — remove or replace with ast.literal_eval.",
                severity="critical",
                finding_type="vulnerability",
                file_path=file.path,
                line_start=rng.start.line + 1,
                line_end=rng.end.line + 1,
            ))

        for node in root.find_all(pattern="exec($$$ARGS)"):
            rng = node.range()
            findings.append(Finding(
                tool=self.name,
                rule_id="repointel:security/exec-usage",
                message="Use of exec() executes arbitrary code.",
                severity="critical",
                finding_type="vulnerability",
                file_path=file.path,
                line_start=rng.start.line + 1,
                line_end=rng.end.line + 1,
            ))

        # Rule: shell=True
        for node in root.find_all(pattern="$CALL($$$PRE, shell=True, $$$POST)"):
            rng = node.range()
            findings.append(Finding(
                tool=self.name,
                rule_id="repointel:security/shell-true",
                message="subprocess with shell=True is unsafe with untrusted input.",
                severity="major",
                finding_type="vulnerability",
                file_path=file.path,
                line_start=rng.start.line + 1,
                line_end=rng.end.line + 1,
            ))

        return findings

    def _python_ast_rules(self, file: FileRecord, source: str) -> list[Finding]:
        findings: list[Finding] = []
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append(Finding(
                    tool=self.name,
                    rule_id="repointel:bug/bare-except",
                    message="Bare `except:` swallows all errors including KeyboardInterrupt/SystemExit.",
                    severity="major",
                    finding_type="bug",
                    file_path=file.path,
                    line_start=getattr(node, "lineno", 1),
                    line_end=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                ))

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults = node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
                for default in defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        findings.append(Finding(
                            tool=self.name,
                            rule_id="repointel:bug/mutable-default-arg",
                            message=f"Mutable default argument in function '{node.name}' is shared across calls — use None.",
                            severity="major",
                            finding_type="bug",
                            file_path=file.path,
                            line_start=getattr(default, "lineno", node.lineno),
                            line_end=getattr(default, "end_lineno", node.lineno),
                        ))

            if isinstance(node, ast.Compare):
                for op, comp in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.Eq, ast.NotEq)):
                        is_left_none = isinstance(node.left, ast.Constant) and node.left.value is None
                        is_right_none = isinstance(comp, ast.Constant) and comp.value is None
                        if is_left_none or is_right_none:
                            findings.append(Finding(
                                tool=self.name,
                                rule_id="repointel:bug/eq-none",
                                message="Comparison to None should use `is None` / `is not None`.",
                                severity="minor",
                                finding_type="bug",
                                file_path=file.path,
                                line_start=getattr(node, "lineno", 1),
                                line_end=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                            ))

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec"):
                    findings.append(Finding(
                        tool=self.name,
                        rule_id=f"repointel:security/{node.func.id}-usage",
                        message=f"Use of {node.func.id}() executes arbitrary code — remove or replace with ast.literal_eval.",
                        severity="critical",
                        finding_type="vulnerability",
                        file_path=file.path,
                        line_start=getattr(node, "lineno", 1),
                        line_end=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                    ))

            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        findings.append(Finding(
                            tool=self.name,
                            rule_id="repointel:security/shell-true",
                            message="subprocess with shell=True is unsafe with untrusted input.",
                            severity="major",
                            finding_type="vulnerability",
                            file_path=file.path,
                            line_start=getattr(node, "lineno", 1),
                            line_end=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                        ))

        return findings


    # -------------------------------------------------------------------- rules
    def _regex_rules(self, file, lines, rules) -> list[Finding]:
        out: list[Finding] = []
        for rule_id, pattern, severity, ftype, message in rules:
            for lineno, line in enumerate(lines, start=1):
                if pattern.search(line):
                    out.append(Finding(
                        tool=self.name, rule_id=rule_id, message=message,
                        severity=severity, finding_type=ftype,
                        file_path=file.path, line_start=lineno, line_end=lineno,
                    ))
        return out

    def _satd_findings(self, file, lines) -> list[Finding]:
        out: list[Finding] = []
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped.startswith(("#", "//", "*", "/*", "<!--")):
                # still scan inline trailing comments in python
                if "#" not in line:
                    continue
            for pattern, tag in SATD_PATTERNS:
                m = pattern.search(line)
                if m:
                    out.append(Finding(
                        tool=self.name, rule_id=f"repointel:satd/{tag.lower()}",
                        message=f"Self-admitted technical debt ({tag}): {stripped[:120]}",
                        severity="minor", finding_type="td_prediction",
                        file_path=file.path, line_start=lineno, line_end=lineno,
                    ))
                    break
        return out

    def _symbol_metrics(self, file, sym, body: str) -> list[Finding]:
        out: list[Finding] = []
        line_count = sym.location.end_line - sym.location.start_line + 1
        complexity = self._cyclomatic(body)
        params = self._param_count(sym.signature or "")

        if sym.type == "function" and complexity > COMPLEXITY_THRESHOLD:
            out.append(Finding(
                tool=self.name, rule_id="repointel:smell/high-complexity",
                message=f"Function '{sym.name}' has cyclomatic complexity {complexity} (threshold {COMPLEXITY_THRESHOLD}).",
                severity="major" if complexity > COMPLEXITY_THRESHOLD * 2 else "minor",
                finding_type="code_smell", file_path=file.path,
                line_start=sym.location.start_line, line_end=sym.location.end_line,
                symbol_id=sym.id, confidence=complexity / (COMPLEXITY_THRESHOLD * 3),
            ))
        if sym.type == "function" and line_count > LONG_FUNCTION_LINES:
            out.append(Finding(
                tool=self.name, rule_id="repointel:smell/long-function",
                message=f"Function '{sym.name}' spans {line_count} lines (threshold {LONG_FUNCTION_LINES}).",
                severity="minor", finding_type="code_smell",
                file_path=file.path, line_start=sym.location.start_line,
                line_end=sym.location.end_line, symbol_id=sym.id,
            ))
        if sym.type == "function" and params > MANY_PARAMS:
            out.append(Finding(
                tool=self.name, rule_id="repointel:smell/long-parameter-list",
                message=f"Function '{sym.name}' takes {params} parameters (threshold {MANY_PARAMS}).",
                severity="minor", finding_type="code_smell",
                file_path=file.path, line_start=sym.location.start_line,
                symbol_id=sym.id,
            ))
        nesting = self._max_nesting(body)
        if nesting > DEEP_NESTING:
            out.append(Finding(
                tool=self.name, rule_id="repointel:smell/deep-nesting",
                message=f"'{sym.name}' nests {nesting} levels deep (threshold {DEEP_NESTING}).",
                severity="minor", finding_type="code_smell",
                file_path=file.path, line_start=sym.location.start_line, symbol_id=sym.id,
            ))
        return out

    def _file_metrics(self, file, source, lines, symbols) -> list[Finding]:
        out: list[Finding] = []
        functions = [s for s in symbols if s.type == "function"]
        classes = [s for s in symbols if s.type == "class"]
        loc = sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))
        if loc > 500 and (len(functions) + len(classes)) / max(loc, 1) < 0.01:
            out.append(Finding(
                tool=self.name, rule_id="repointel:smell/god-file",
                message=f"File has {loc} LOC but only {len(functions)} functions / {len(classes)} classes.",
                severity="major", finding_type="code_smell",
                file_path=file.path, line_start=1, line_end=loc,
            ))
        return out

    # ------------------------------------------------------------------ metrics
    @staticmethod
    def _cyclomatic(body: str) -> int:
        score = 1
        for kw in ("if ", "elif ", "else:", "for ", "while ", "except", "case ", "and ", "or ", "?:"):
            score += body.count(kw)
        return score

    @staticmethod
    def _param_count(signature: str) -> int:
        inner = signature[signature.find("(") + 1: signature.rfind(")")] if "(" in signature else ""
        parts = [p for p in inner.split(",") if p.strip() and p.strip() not in ("self", "cls")]
        return len(parts)

    @staticmethod
    def _max_nesting(body: str) -> int:
        depth = max_depth = 0
        for ch in body:
            if ch == "{":
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == "}":
                depth = max(0, depth - 1)
        # Python: indentation-based
        indents: list[int] = []
        for line in body.splitlines():
            if line.strip():
                indents.append(len(line) - len(line.lstrip()))
        if indents:
            max_depth = max(max_depth, max(indents) // 4)
        return max_depth


def run_builtin_analysis(
    repo_path: Path | str, files: list[FileRecord], code_intelligence: CodeIntelligence
) -> list[Finding]:
    """Run the built-in analyzer over all code files (driver)."""
    analyzer = BuiltinAnalyzer()
    symbols_by_file: dict[str, list] = {}
    for sym in code_intelligence.symbols:
        symbols_by_file.setdefault(sym.file_path, []).append(sym)

    findings: list[Finding] = []
    analyzed = 0
    for file in files:
        if file.category not in ("code",):
            # SATD also lives in comments of config/docs — scan .py only for AST rules
            continue
        content, ok = read_file_content(repo_path, file.path)
        if not ok:
            continue
        analyzed += 1
        file_findings = analyzer.analyze_file(file, content, symbols_by_file.get(file.path, []))
        findings.extend(file_findings)

    logger.info("builtin_analysis_complete", extra={"extra_fields": {
        "files_analyzed": analyzed, "findings": len(findings),
        "by_severity": _count_by(findings, "severity"),
        "by_type": _count_by(findings, "finding_type"),
    }})
    return findings


def _count_by(findings, attr) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in findings:
        key = getattr(f, attr, "unknown") or "unknown"
        out[key] = out.get(key, 0) + 1
    return out
