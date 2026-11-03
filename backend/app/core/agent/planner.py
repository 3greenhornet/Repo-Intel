"""Goal decomposition — break a large goal into atomic subtasks, execute each with Investigator."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

DECOMPOSE_PROMPT = """You are a software codebase investigation planner. Break the GOAL into 2-5 atomic subtasks that can each be investigated independently by a code intelligence tool that can (a) search code, (b) list findings, (c) trace callers, (d) calculate blast radius, (e) summarize architecture.

RULES:
- Each subtask must be answerable with ONE focused tool sequence.
- Subtasks must not overlap in purpose.
- Order them so earlier subtasks provide context for later ones.
- Output ONLY valid JSON matching the schema below, with no surrounding commentary or markdown.

Output schema:
{{"subtasks": [{{"goal": "<subtask question>", "rationale": "<one sentence explanation>"}}]}}

GOAL:
{goal}
"""


@dataclass
class Subtask:
    goal: str
    rationale: str = ""


@dataclass
class DecompositionResult:
    original_goal: str
    subtasks: list[Subtask] = field(default_factory=list)
    decomposition_elapsed: float = 0.0


def decompose(goal: str, llm: Any) -> DecompositionResult:
    """Decompose a goal into subtasks. Falls back to single-subtask on failure or simple queries."""
    started = time.monotonic()

    # Fast path: single-hop direct queries skip LLM decomposition
    if _is_single_hop(goal):
        logger.info("planner: single-hop goal, skipping decomposition")
        return DecompositionResult(
            original_goal=goal,
            subtasks=[Subtask(goal=goal, rationale="Single-hop — direct execution")],
            decomposition_elapsed=time.monotonic() - started,
        )

    try:
        raw = llm.invoke(DECOMPOSE_PROMPT.format(goal=goal), max_tokens=600)
        data = _parse_json(raw)
        if not data or "subtasks" not in data or not isinstance(data["subtasks"], list):
            raise ValueError("missing valid 'subtasks' array in decomposition output")

        subtasks = [
            Subtask(goal=str(s.get("goal", "")).strip(), rationale=str(s.get("rationale", "")).strip())
            for s in data["subtasks"]
            if isinstance(s, dict) and s.get("goal")
        ]

        if not subtasks:
            raise ValueError("empty subtasks list after filtering")

        elapsed = time.monotonic() - started
        logger.info(f"planner: decomposed into {len(subtasks)} subtasks in {elapsed:.2f}s")
        return DecompositionResult(
            original_goal=goal,
            subtasks=subtasks,
            decomposition_elapsed=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - started
        logger.warning(f"planner: decomposition failed ({exc}), falling back to single task")
        return DecompositionResult(
            original_goal=goal,
            subtasks=[Subtask(goal=goal, rationale="Fallback — direct execution")],
            decomposition_elapsed=elapsed,
        )


def _is_single_hop(goal: str) -> bool:
    """Fast-path detection for direct, single-hop queries."""
    single_hop_keywords = (
        "who calls", "what is", "where is", "show me",
        "find the", "list the", "what does", "how many",
        "is there", "which function", "which file",
    )
    g = goal.lower().strip()
    if any(k in g for k in single_hop_keywords):
        return True
    return len(goal.split()) < 7


def _parse_json(raw: str) -> dict[str, Any] | None:
    """Robustly parse JSON object from LLM output with potential markdown fences or commentary."""
    from json import JSONDecoder

    if not raw or not raw.strip():
        return None

    text = raw.strip()
    if text.startswith("```"):
        import re
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    decoder = JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)

    return None

