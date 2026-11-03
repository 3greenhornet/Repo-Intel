from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tempfile
from app.core.analysis.intent_checker import check_intent, load_intent
from app.core.storage.sqlite_graph_store import SQLiteGraphStore


def test_load_intent():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / ".repointel").mkdir()
        (repo / ".repointel" / "architecture.yml").write_text(
            "components:\n  api:\n    paths: ['app/api.py']\n"
            "dependencies:\n  forbidden:\n    - from: utils\n      to: auth\n",
            encoding="utf-8",
        )
        intent = load_intent(repo)
        assert intent is not None
        assert "components" in intent


def test_intent_violation_detected():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        store = SQLiteGraphStore(db_path)
        # Add nodes and a forbidden edge
        store.add_node("utils:helper", "helper", "Function", "app/utils.py")
        store.add_node("auth:login", "login", "Function", "app/auth.py")
        store.add_edge("utils:helper", "auth:login", "CALLS")

        intent = {
            "components": {
                "utils": {"paths": ["app/utils.py"]},
                "auth": {"paths": ["app/auth.py"]},
            },
            "dependencies": {
                "forbidden": [{"from": "utils", "to": "auth"}]
            },
        }

        findings = check_intent(Path(tmp), store, intent)
        assert any("forbidden" in f.rule_id for f in findings)
        assert any(f.finding_type == "architecture_violation" for f in findings)
        store.conn.close()
