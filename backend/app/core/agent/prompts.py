"""Prompt templates for the RepoIntel Investigator agent."""
from __future__ import annotations

import json

from app.core.agent.tools import render_tool_list

PLANNER_SYSTEM = """You are a code investigation agent. You receive a goal and a running
trace of your own previous thoughts, actions, and observations. On each turn you
decide exactly one next action from the available tool list.

Rules:
- Call exactly ONE tool per turn.
- Output ONLY valid JSON, no prose, no markdown fences.
- Never repeat the same (action, args) pair on two consecutive turns.
- Use evidence IDs from previous search_code results when calling check_relations.
- When you have enough evidence to answer the goal, call finish with a full report.
- Every claim in the final report must cite file:line (e.g. demo_shop/auth.py:22-28).

Output schema:
{"thought": "<one sentence of reasoning>", "action": "<tool_name>", "args": {...}}
"""


def build_planner_prompt(goal: str, scratchpad: list[dict], step: int, max_steps: int) -> str:
    tool_list = render_tool_list()
    trace_lines = []
    for entry in scratchpad[-6:]:
        obs = entry.get("observation", {})
        obs_str = json.dumps(obs, default=str)[:800]
        trace_lines.append(
            f"[Step {entry['step']}]\n"
            f"Thought: {entry.get('thought', '')}\n"
            f"Action: {entry['action']}\n"
            f"Args: {json.dumps(entry.get('args', {}), default=str)}\n"
            f"Observation: {obs_str}\n"
        )
    trace = "\n".join(trace_lines) if trace_lines else "(no steps yet)"
    return (
        f"GOAL:\n{goal}\n\n"
        f"AVAILABLE TOOLS:\n{tool_list}\n\n"
        f"TRACE SO FAR (last 6 steps):\n{trace}\n\n"
        f"Step {step} of {max_steps}. Output the JSON for your next action."
    )


REFLECTION_SYSTEM = """You judge whether a code investigation agent is making
progress. Output ONLY valid JSON, no prose.

Output schema:
{"decision": "CONTINUE" | "CHANGE" | "FINISH" | "STUCK", "reason": "<one sentence>"}

- CONTINUE: the current direction is productive; keep going.
- CHANGE: the last 2 steps did not add new information; try a different tool.
- FINISH: enough evidence to write the report.
- STUCK: no productive path without external help.
"""


def build_reflection_prompt(goal: str, scratchpad: list[dict]) -> str:
    if not scratchpad:
        return f"GOAL:\n{goal}\n\nNo steps taken yet. Decide CONTINUE."
    last = scratchpad[-1]
    obs_str = json.dumps(last.get("observation", {}), default=str)[:600]
    return (
        f"GOAL:\n{goal}\n\n"
        f"Steps taken: {len(scratchpad)}\n"
        f"Last step:\n"
        f" Thought: {last.get('thought', '')}\n"
        f" Action: {last['action']}\n"
        f" Observation: {obs_str}\n\n"
        f"Decide the agent's next move."
    )


SYNTHESIS_SYSTEM = """You write concise, evidence-grounded investigation reports.
Output plain text only, no JSON, no markdown fences, no preamble.
Every factual claim must cite file:line, e.g. (demo_shop/auth.py:22-28).
If the investigation is incomplete, say so explicitly and recommend next steps.
"""


def build_synthesis_prompt(goal: str, scratchpad: list[dict]) -> str:
    lines = []
    for entry in scratchpad:
        obs = entry.get("observation", {})
        obs_str = json.dumps(obs, default=str)[:1000]
        lines.append(
            f"[Step {entry['step']}]\n"
            f"Action: {entry['action']} Args: {json.dumps(entry.get('args', {}), default=str)}\n"
            f"Observation: {obs_str}\n"
        )
    return (
        f"GOAL:\n{goal}\n\n"
        f"FULL TRACE:\n{chr(10).join(lines)}\n\n"
        f"Write the final report (200-400 words)."
    )
