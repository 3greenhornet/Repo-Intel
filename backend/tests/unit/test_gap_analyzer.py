from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tempfile
from app.core.analysis.gap_analyzer import (
    analyze_gaps, _incomplete_features, _error_handling_gap, _test_coverage_gap,
)
from app.models.repository import FileRecord
from app.utils.time import utc_now


def _file(path: str, lang: str = "python", category: str = "code") -> FileRecord:
    return FileRecord(
        path=path,
        category=category,
        language=lang,
        size_bytes=100,
        content_hash="x",
        last_modified=utc_now(),
    )


def test_incomplete_stub_detection():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / "stub.py").write_text("def todo_function():\n    pass\n", encoding="utf-8")
        findings = _incomplete_features(repo, [_file("stub.py")])
        assert any("stub" in f.rule_id or "empty-body" in f.rule_id for f in findings), [f.rule_id for f in findings]


def test_not_implemented_detection():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / "ni.py").write_text("def feature():\n    raise NotImplementedError('coming soon')\n", encoding="utf-8")
        findings = _incomplete_features(repo, [_file("ni.py")])
        assert any("not-implemented" in f.rule_id for f in findings)


def test_unguarded_io():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / "io.py").write_text("def read():\n    f = open('x.txt')\n    return f.read()\n", encoding="utf-8")
        findings = _error_handling_gap(repo, [_file("io.py")])
        assert any("unguarded-io" in f.rule_id for f in findings)


def test_guarded_io_is_ok():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / "io.py").write_text(
            "def read():\n    try:\n        f = open('x.txt')\n        return f.read()\n    except OSError:\n        return ''\n",
            encoding="utf-8",
        )
        findings = _error_handling_gap(repo, [_file("io.py")])
        assert not findings


def test_coverage_gap_detects_untested_file():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / "service.py").write_text("def run():\n    return 42\n", encoding="utf-8")
        (repo / "tests").mkdir()
        (repo / "tests" / "test_other.py").write_text("import os\n", encoding="utf-8")
        code_files = [_file("service.py")]
        test_files = [_file("tests/test_other.py", category="tests")]
        findings = _test_coverage_gap(repo, code_files, test_files)
        assert any("untested-module" in f.rule_id for f in findings)

