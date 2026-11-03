"""ReAct-style investigation agent built on RepoIntel."""
from __future__ import annotations

import json
import logging
import os
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
        """Public entry: gate on ENABLE_GOAL_DECOMPOSITION."""
        if os.environ.get("ENABLE_GOAL_DECOMPOSITION", "0") == "1":
            return self._run_decomposed(repo_id, goal)
        return self._run_single(repo_id, goal)

    def _run_decomposed(self, repo_id: str, goal: str) -> dict[str, Any]:
        """Decompose goal into subtasks, run each via _run_single, aggregate."""
        from app.core.agent.planner import decompose

        started = time.monotonic()
        decomposition = decompose(goal, self.llm)

        # Single-subtask fast path: no aggregation overhead
        if len(decomposition.subtasks) == 1:
            result = self._run_single(repo_id, decomposition.subtasks[0].goal)
            result["decomposition"] = {
                "subtasks": [{"goal": decomposition.subtasks[0].goal,
                              "rationale": decomposition.subtasks[0].rationale}],
                "decomposition_elapsed": round(decomposition.decomposition_elapsed, 2),
            }
            return result

        results = []
        all_reports = []
        all_steps = []
        total_tokens = 0
        all_citations = []

        for i, subtask in enumerate(decomposition.subtasks, 1):
            logger.info(f"decomposed-subtask {i}/{len(decomposition.subtasks)}: {subtask.goal[:80]}")
            try:
                result = self._run_single(repo_id, subtask.goal)
            except Exception as exc:
                logger.warning(f"subtask {i} failed: {exc}")
                result = {"report": f"[subtask failed: {exc}]", "steps": 0, "tokens_used": 0, "trace": [], "citations": []}

            results.append({"subtask": subtask.goal, "result": result})
            all_reports.append(f"### {i}. {subtask.goal}\n\n{result.get('report', '')}")
            all_steps.extend(result.get("trace", []))
            all_citations.extend(result.get("citations", []))
            total_tokens += result.get("tokens_used", 0)

        return {
            "goal": goal,
            "report": "\n\n".join(all_reports),
            "steps": len(all_steps),
            "tokens_used": total_tokens,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "budget_exhausted": False,
            "decomposition": {
                "subtasks": [{"goal": s.goal, "rationale": s.rationale}
                             for s in decomposition.subtasks],
                "decomposition_elapsed": round(decomposition.decomposition_elapsed, 2),
            },
            "subtask_results": results,
            "trace": all_steps,
            "trace_path": "",
            "citations": all_citations,
            "evidence_count": len(all_citations),
        }

    def _run_single(self, repo_id: str, goal: str) -> dict[str, Any]:
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

        # Extract structured citations from all Evidence objects the agent saw
        from app.core.reasoning.output_generator import extract_citations

        all_evidence: list = []
        seen_ids: set[str] = set()
        for entry in scratchpad:
            obs = entry.get("observation", {}) or {}
            for ev_data in obs.get("evidence", []) or []:
                if not isinstance(ev_data, dict):
                    continue
                eid = ev_data.get("id")
                if not eid or eid in seen_ids:
                    continue
                ev_obj = self.executor.evidence_cache.get(eid)
                if ev_obj is not None:
                    all_evidence.append(ev_obj)
                    seen_ids.add(eid)

        citations = extract_citations(final_report or "", all_evidence)

        return {
            "goal": goal,
            "report": final_report,
            "steps": len(scratchpad),
            "elapsed_seconds": round(elapsed, 2),
            "tokens_used": tokens_used,
            "budget_exhausted": budget_exhausted,
            "trace_path": str(trace_path),
            "trace": scratchpad,
            "citations": [c.__dict__ for c in citations],
            "evidence_count": len(all_evidence),
        }

    def _safe_invoke(self, prompt: str, system: str) -> str:
        try:
            return self.llm.invoke(prompt, system_prompt=system, max_tokens=1200)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return ""

    def _parse_action(self, raw: str) -> dict | None:
        r"""Parse the planner's JSON action.

        FIX-2026: The greedy r"\{.*\}" regex grabbed from the first { to the last }
        in the string. Any prose before/after or multiple JSON blocks broke the payload.
        This version scans for the first valid JSON object with an "action" key
        using JSONDecoder.raw_decode, which stops at the matching close brace.
        """
        if not raw or not raw.strip():
            return None
        text = raw.strip()

        # Strip markdown fences if the model emitted them
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)

        # Fast path: whole string is valid JSON
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "action" in data:
                return data
        except json.JSONDecodeError:
            pass

        # Scan for the first valid JSON object with an "action" key
        from json import JSONDecoder
        decoder = JSONDecoder()
        n = len(text)
        for i in range(n):
            if text[i] != "{":
                continue
            try:
                obj, _end = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "action" in obj:
                return obj
        return None

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
            from app.core.reasoning.prompt_templates import sanitize_repository_content
            safe_prompt = sanitize_repository_content(prompt)
        except Exception:
            safe_prompt = prompt
        try:
            return self.llm.invoke(safe_prompt, system_prompt=SYNTHESIS_SYSTEM, max_tokens=1600)
        except Exception as exc:
            logger.error("Synthesis failed: %s", exc)
            return "\n".join(
                f"[{s['action']}] {json.dumps(s['observation'], default=str)[:400]}"
                for s in scratchpad
            )
