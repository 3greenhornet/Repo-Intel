from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.agent.planner import _is_single_hop, decompose


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def invoke(self, prompt: str, **kw):
        return self.response


def test_single_hop_fast_path():
    result = decompose("Who calls login?", _FakeLLM("should not be called"))
    assert len(result.subtasks) == 1
    assert result.subtasks[0].goal == "Who calls login?"


def test_multi_hop_decomposes():
    llm = _FakeLLM(
        '{"subtasks": [{"goal": "Find auth code", "rationale": "needed first"},'
        ' {"goal": "Find security findings", "rationale": "second step"}]}'
    )
    result = decompose("Modernize the authentication module end to end", llm)
    assert len(result.subtasks) == 2
    assert result.subtasks[0].goal == "Find auth code"
    assert result.subtasks[1].goal == "Find security findings"


def test_bad_json_falls_back():
    result = decompose(
        "Complex architectural overhaul about the entire pipeline flow",
        _FakeLLM("not json at all"),
    )
    assert len(result.subtasks) == 1
    assert result.subtasks[0].goal == "Complex architectural overhaul about the entire pipeline flow"

