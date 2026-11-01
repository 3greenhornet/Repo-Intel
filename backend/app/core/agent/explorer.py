"""Component 5D: Iterative exploration loop (§8.4) — the AgentComponent (§3.2.6).

Budgets: max_iterations=5, max_total_tokens=10000, max_time_seconds,
min_evidence_count=3 (§8.4.2 configuration table).
"""
from __future__ import annotations

import time

from app.core.agent.gap_detector import check_sufficiency, identify_gaps, classify_query_complexity
from app.core.agent.query_formulator import formulate_query
from app.models.evidence import Evidence, ExplorationResult
from app.utils.logger import get_logger
from app.utils.token_counter import count_tokens_batch

logger = get_logger(__name__)


class AgenticExplorer:
    """§3.2.6 AgentComponent: check_sufficiency / identify_gaps / formulate_query / explore."""

    def __init__(self, step4_executor, llm_client=None, max_iterations: int = 5,
                 max_total_tokens: int = 10000, max_time_seconds: float = 60.0,
                 min_evidence_count: int = 3,
                 use_llm_sufficiency: bool = False) -> None:
        self.step4 = step4_executor            # callable(repo_id, query, top_k) -> RerankedEvidence
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.max_total_tokens = max_total_tokens
        self.max_time_seconds = max_time_seconds
        self.min_evidence_count = min_evidence_count
        self.use_llm_sufficiency = use_llm_sufficiency  # FIX-037

    # -------------------------------------------------------------- 5A checks
    def _check_sufficiency(self, query: str, evidence: list[Evidence]) -> tuple[bool, float, str]:
        """FIX-037: LLM-based check when configured, heuristic otherwise."""
        if self.use_llm_sufficiency and self.llm_client is not None:
            from app.core.agent.gap_detector import check_sufficiency_llm

            return check_sufficiency_llm(query, evidence, self.llm_client)
        return check_sufficiency(query, evidence)
    def check_sufficiency(self, evidence: list[Evidence], query: str = "") -> bool:
        sufficient, _, _ = check_sufficiency(query, evidence)
        return sufficient

    def identify_gaps(self, evidence: list[Evidence], query: str = "") -> list[str]:
        return identify_gaps(query, evidence)

    def formulate_query(self, gaps: list[str], original_query: str) -> str:
        return formulate_query(gaps, original_query, self.llm_client)

    # ---------------------------------------------------------------- 5D loop
    def explore(self, repo_id: str, query: str, initial_evidence: list[Evidence]) -> ExplorationResult:
        """§8.4.3 agentic_exploration with 2026 Adaptive RAG routing."""
        evidence = initial_evidence
        all_evidence = list(initial_evidence)
        iteration = 0
        start_time = time.monotonic()
        total_tokens = count_tokens_batch([e.chunk.content for e in evidence])
        gap_history: list[str] = []
        follow_ups: list[str] = []

        # Adaptive RAG: classify complexity before looping
        complexity, c_reason = classify_query_complexity(
            query, self.llm_client if self.use_llm_sufficiency else None)
        logger.info(f"Adaptive RAG routing: query complexity={complexity} ({c_reason})")

        # Fast single-pass path for simple factual queries with strong initial evidence
        if complexity == "simple" and initial_evidence:
            is_suff, conf, s_reason = self._check_sufficiency(query, initial_evidence)
            if is_suff and conf >= 0.7:
                return ExplorationResult(
                    evidence=initial_evidence, iterations=1, sufficient=True,
                    reason=f"Adaptive RAG (single-pass): {s_reason}",
                    elapsed_seconds=time.monotonic() - start_time,
                    gap_history=[], follow_up_queries=[])

        while iteration < self.max_iterations:
            iteration += 1
            is_sufficient, confidence, reason = self._check_sufficiency(query, evidence)
            if is_sufficient and len(all_evidence) >= self.min_evidence_count:
                return ExplorationResult(
                    evidence=all_evidence, iterations=iteration, sufficient=True,
                    reason=reason, elapsed_seconds=time.monotonic() - start_time,
                    gap_history=gap_history, follow_up_queries=follow_ups)

            elapsed = time.monotonic() - start_time
            if elapsed > self.max_time_seconds:
                return self._stop(all_evidence, iteration, "Time budget exceeded",
                                  elapsed, gap_history, follow_ups)
            if total_tokens > self.max_total_tokens:
                return self._stop(all_evidence, iteration, "Token budget exceeded",
                                  elapsed, gap_history, follow_ups)

            gaps = identify_gaps(query, evidence)
            if not gaps:
                return self._stop(all_evidence, iteration,
                                  "No gaps identified but evidence insufficient",
                                  elapsed, gap_history, follow_ups)
            gap_history.extend(g[:60] for g in gaps[:5])

            new_query = formulate_query(gaps, query, self.llm_client)
            follow_ups.append(new_query[:100])

            new_evidence = self.step4(repo_id, new_query, top_k=10).evidence
            new_evidence = self._deduplicate(new_evidence, all_evidence)

            if not new_evidence:
                return self._stop(all_evidence, iteration, "No new evidence found",
                                  elapsed, gap_history, follow_ups)

            evidence = new_evidence
            all_evidence.extend(new_evidence)
            total_tokens += count_tokens_batch([e.chunk.content for e in new_evidence])

        return self._stop(all_evidence, iteration, "Max iterations reached",
                          time.monotonic() - start_time, gap_history, follow_ups)

    def _stop(self, evidence, iterations, reason, elapsed, gaps, follow_ups) -> ExplorationResult:
        logger.info("exploration_stopped", extra={"extra_fields": {
            "iterations": iterations, "reason": reason, "evidence": len(evidence),
            "elapsed": round(elapsed, 2),
        }})
        return ExplorationResult(
            evidence=evidence, iterations=iterations, sufficient=False,
            reason=reason, elapsed_seconds=elapsed,
            gap_history=gaps, follow_up_queries=follow_ups)

    @staticmethod
    def _deduplicate(new_evidence: list[Evidence], existing: list[Evidence]) -> list[Evidence]:
        """§8.4.3 deduplicate."""
        existing_ids = {e.chunk.id for e in existing}
        return [e for e in new_evidence if e.chunk.id not in existing_ids]
