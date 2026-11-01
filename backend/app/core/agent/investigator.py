"""ReAct-style investigation agent built on RepoIntel."""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.agent.prompts import (
    PLANNER_SYSTEM,
    REFLECTION_SYSTEM,
    SYNTHESIS_SYSTEM,
    build_planner_prompt,
    build_reflection_prompt,
    build_synthesis_prompt,
)
from app.core.agent.tools import ToolExecutor
from app.core.agent.tracer import Tracer, slugify

logger = logging.getLogger(__name__)


class Investigator:
    def __init__(
        self,
        phase1,
        phase2,
        llm_client,
        max_steps: int = 8,
        max_time_seconds: float = 120.0,
        trace_dir: str | Path = "data/traces",
    ) -> None:
        self.phase1 = phase1
        self.phase2 = phase2
        self.llm = llm_client
        self.max_steps = max_steps
        self.max_time = max_time_seconds
        self.trace_dir = Path(trace_dir)
        self.executor = ToolExecutor(phase1, phase2)

    def run(self, repo_id: str, goal: str) -> dict[str, Any]:
        started = time.monotonic()
        scratchpad: list[dict] = []
        tokens_before = self.llm.tokens_in + self.llm.tokens_out
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        trace_path = self.trace_dir / f"{ts}_{slugify(goal)}.jsonl"
        tracer = Tracer(trace_path)

        final_report: str | None = None
        budget_exhausted = False

        for step in range(1, self.max_steps + 1):
            if time.monotonic() - started > self.max_time:
                budget_exhausted = True
                break

            prompt = build_planner_prompt(goal, scratchpad, step, self.max_steps)
            raw = self._safe_invoke(prompt, PLANNER_SYSTEM)
            parsed = self._parse_action(raw)
            if parsed is None:
                logger.warning("Step %d: unparseable planner output, breaking", step)
                budget_exhausted = True
                break

            action_name = parsed["action"]
            args = parsed.get("args") or {}
            thought = parsed.get("thought", "")

            if action_name == "finish":
                final_report = str(args.get("report", "")).strip() or self._synthesize(goal, scratchpad)
                tracer.log(step, thought, "finish", args, {"finalized": True})
                scratchpad.append({
                    "step": step,
                    "thought": thought,
                    "action": "finish",
                    "args": args,
                    "observation": {"finalized": True},
                })
                break

            try:
                observation = self.executor.execute(action_name, args, repo_id)
            except Exception as exc:
                observation = {"error": str(exc)[:300]}

            scratchpad.append({
                "step": step,
                "thought": thought,
                "action": action_name,
                "args": args,
                "observation": observation,
            })
            tracer.log(step, thought, action_name, args, observation)

            decision = self._reflect(goal, scratchpad)
            if decision == "FINISH":
                final_report = self._synthesize(goal, scratchpad)
                break
            if decision == "STUCK":
                logger.info("Step %d: agent reports STUCK", step)
                break

        if final_report is None:
            final_report = self._synthesize(goal, scratchpad)

        elapsed = time.monotonic() - started
        tokens_used = (self.llm.tokens_in + self.llm.tokens_out) - tokens_before

        return {
            "goal": goal,
            "report": final_report,
            "steps": len(scratchpad),
            "elapsed_seconds": round(elapsed, 2),
            "tokens_used": tokens_used,
            "budget_exhausted": budget_exhausted,
            "trace_path": str(trace_path),
            "trace": scratchpad,
        }

    def _safe_invoke(self, prompt: str, system: str) -> str:
        try:
            return self.llm.invoke(prompt, system_prompt=system, max_tokens=1200)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return ""

    def _parse_action(self, raw: str) -> dict | None:
        if not raw or not raw.strip():
            return None
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        if not isinstance(data, dict) or "action" not in data:
            return None
        return data

    def _reflect(self, goal: str, scratchpad: list[dict]) -> str:
        prompt = build_reflection_prompt(goal, scratchpad)
        raw = self._safe_invoke(prompt, REFLECTION_SYSTEM)
        parsed = self._parse_action(raw)
        if parsed and parsed.get("decision") in ("CONTINUE", "CHANGE", "FINISH", "STUCK"):
            return parsed["decision"]
        if len(scratchpad) >= self.max_steps - 1:
            return "FINISH"
        return "CONTINUE"

    def _synthesize(self, goal: str, scratchpad: list[dict]) -> str:
        if not scratchpad:
            return "Investigation produced no steps; unable to answer."
        prompt = build_synthesis_prompt(goal, scratchpad)
        try:
            return self.llm.invoke(prompt, system_prompt=SYNTHESIS_SYSTEM, max_tokens=1600)
        except Exception as exc:
            logger.error("Synthesis failed: %s", exc)
            return "\n".join(
                f"[{s['action']}] {json.dumps(s['observation'], default=str)[:400]}"
                for s in scratchpad
            )
