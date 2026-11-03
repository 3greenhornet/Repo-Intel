"""Phase 2 orchestrator — Query & Reason (§12 end-to-end flow).

`execute_full_pipeline(repo_id, question)` runs:
STEP 4 hybrid retrieval → STEP 5 agentic exploration → STEP 6 evidence
orchestration → STEP 7 LLM reasoning → STEP 8 developer output.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from app.core.agent.explorer import AgenticExplorer
from app.core.evidence.context_builder import (
    assemble_context, budget_context, select_evidence,
)
from app.core.evidence.nli_scorer import NLIScorer
from app.core.evidence.provenance import ProvenanceAttacher
from app.core.knowledge.knowledge_service import Phase1Pipeline
from app.core.reasoning.llm_client import LLMClient
from app.core.reasoning.output_generator import (
    extract_citations, generate_followups,
)
from app.core.reasoning.prompt_templates import (
    SYSTEM_PROMPT, SECURITY_SUFFIX, build_user_prompt, validate_input_length,
)
from app.core.retrieval.retrieval_service import RetrievalComponent
from app.models.query import QueryOptions, QueryResponse
from app.utils.logger import get_logger
from app.utils.time import utc_now
from app.utils.token_counter import count_tokens

logger = get_logger(__name__)

# FIX-001 (corrected): the doc's literal check ("not enough evidence" in
# answer) fails on its own test case "I don't have enough evidence" — that
# sentence contains "enough evidence" but not the substring "not enough".
# Match the family of abstention phrasings instead.
ABSTENTION_RE = re.compile(
    r"(?:not enough evidence|don'?t have enough evidence|do not have enough evidence|"
    r"insufficient evidence|no (?:relevant )?evidence (?:to|for)|cannot answer|"
    r"can'?t answer (?:this|that|the))",
    re.IGNORECASE,
)

_EXACT_ABSTENTION = re.compile(
    r"^\s*(?:based on the (?:provided )?evidence,?\s*)?"
    r"(?:i\s+(?:don'?t|do\s+not)\s+have\s+enough\s+evidence|"
    r"the\s+evidence\s+is\s+insufficient|"
    r"there\s+is\s+(?:not\s+enough|insufficient|no)\s+evidence\s+to\s+answer|"
    r"cannot\s+answer\s+this)",
    re.IGNORECASE,
)


def is_abstention(answer: str) -> bool:
    """Detect whether the answer is an abstention (insufficient evidence).

    Resolves both false-positives and false-negatives:
    1. If the answer explicitly declares insufficient evidence at the start, it is an
       abstention even if it mentions weak/speculative citations afterwards.
    2. If no citation markers exist, any general abstention phrase marks it as abstention.
    3. If citations exist without an opening refusal, it is a grounded substantive answer.
    """
    if not answer or not answer.strip():
        return True
    trimmed = answer.strip()
    if _EXACT_ABSTENTION.match(trimmed):
        return True
    if not re.search(r"\[Evidence\s+\d+", trimmed, re.IGNORECASE):
        return bool(ABSTENTION_RE.search(trimmed))
    return False


class Phase2Pipeline:
    """The runtime query pipeline bound to a Phase 1 knowledge base."""

    def __init__(self, phase1: Phase1Pipeline, llm_client: LLMClient | None = None) -> None:
        self.phase1 = phase1
        self.llm = llm_client or LLMClient()
        self.retrieval = RetrievalComponent(phase1.vector_indexer, phase1.graph_store)
        self.explorer = AgenticExplorer(
            step4_executor=self._run_step4,
            llm_client=self.llm if _cfg().use_llm_sufficiency else None,
            use_llm_sufficiency=_cfg().use_llm_sufficiency,  # FIX-037
            max_iterations=_cfg().max_iterations,
            max_total_tokens=_cfg().max_total_tokens,
            max_time_seconds=_cfg().max_time_seconds,
            min_evidence_count=_cfg().min_evidence_count,
        )
        self.nli = NLIScorer()
        self._provenance_attacher: dict[str, ProvenanceAttacher] = {}
        self._findings_cache: dict[str, list] = {}
        # FIX-014: subscribe to Phase 1 invalidation events
        self.phase1.register_invalidation_listener(self.invalidate_repo)

    # --------------------------------------------------------------- full query
    def execute_full_pipeline(self, repo_id: str, question: str,
                              options: QueryOptions | None = None) -> QueryResponse:
        """§12 Phase 2 flow. Returns the documented QueryResponse."""
        options = options or QueryOptions()
        validate_input_length(question)
        started = time.monotonic()  # FIX-042

        repo = self.phase1.db.get_repository(repo_id)
        if repo is None or repo.get("status") != "ready":
            raise ValueError(f"Repository {repo_id} not ready — run Phase 1 ingestion first")

        # ---- STEP 4: hybrid retrieval ----------------------------------------
        step4 = self._run_step4(repo_id, question, top_k=options.top_k)

        # ---- STEP 5: agentic exploration (optional) ---------------------------
        if options.use_agentic:
            exploration = self.explorer.explore(repo_id, question, step4.evidence)
            evidence = exploration.evidence
            iterations = exploration.iterations
        else:
            evidence = step4.evidence
            iterations = 0

        # ---- STEP 6: evidence orchestration -------------------------------------
        selected = select_evidence(evidence, token_budget=_cfg().evidence_token_budget)
        selected = self._attach_provenance(repo_id, selected)
        nli_relations = self.nli.score_nli(selected) if options.use_nli else []
        # FIX-029: extract contradictions for response + confidence penalty
        contradictions = [
            r for r in nli_relations if r.relation == "contradiction"]
        budget = budget_context(
            selected, llm_context_window=_cfg().llm_context_window,
            reserved_for_response=_cfg().reserved_for_response)
        context = assemble_context(budget.evidence, nli_relations)

        # FIX-031: surface related prior investigations as background hints
        related = self._recent_related_investigations(repo_id, question)
        if related:
            context += "\n\n=== PRIOR INVESTIGATIONS (reference only) ===\n"
            for i, inv in enumerate(related, 1):
                context += f"\n[Investigation {i}] Q: {(inv.get('query') or '')[:200]}\n"
                if inv.get("answer"):
                    context += f"A: {inv['answer'][:400]}\n"

        # ---- STEP 7: LLM reasoning -----------------------------------------------
        answer, grounded, confidence, abstained = self._reason(
            question, context, budget, contradictions=contradictions)

        # ---- STEP 8: developer output ----------------------------------------------
        citations = extract_citations(answer, budget.evidence)
        follow_ups = (generate_followups(self.llm, answer, question)
                      if answer and options.generate_followups else [])
        elapsed = time.monotonic() - started

        response = QueryResponse(
            answer=answer,
            citations=citations,
            repo_id=repo_id,
            question=question,
            exploration_iterations=iterations,
            elapsed_seconds=elapsed,
            confidence=confidence,
            grounded=grounded,
            abstained=abstained,
            evidence_count=len(budget.evidence),
            follow_up_questions=follow_ups,
            retrieval_sources=self._source_counts(budget.evidence),
            evidence_files=_dedup_files(budget.evidence),
            contradictions=[
                {"evidence_a": r.evidence_a, "evidence_b": r.evidence_b,
                 "confidence": round(r.confidence, 3)}
                for r in (contradictions or [])
            ],
            prior_investigations=[
                {"id": inv.get("id"), "query": inv.get("query")}
                for inv in related
            ],
            timestamp=utc_now(),
        )

        self.phase1.db.log_query(
            repo_id, question, answer, confidence, grounded, iterations, elapsed,
            count_tokens(context), count_tokens(answer))
        self.phase1.db.store_investigation(
            repo_id, question, answer, [e.chunk.id for e in budget.evidence])
        logger.info("query_complete", extra={"extra_fields": {
            "repo_id": repo_id, "question": question[:80], "iterations": iterations,
            "elapsed": round(elapsed, 2), "citations": len(citations),
            "grounded": grounded, "confidence": round(confidence, 2),
        }})
        return response

    # ------------------------------------------------------------------ internals
    def _run_step4(self, repo_id: str, query: str, top_k: int = 10):
        try:
            return self.retrieval.execute_step_4(repo_id, query, top_k=top_k)
        except RuntimeError:
            # FIX-011: self-healing — if the lexical index is unbound because
            # knowledge was never restored in this process, attempt an
            # on-demand restore and retry once.
            if self.phase1.restore_knowledge(repo_id):
                return self.retrieval.execute_step_4(repo_id, query, top_k=top_k)
            raise

    def _attach_provenance(self, repo_id: str, evidence):
        attacher = self._provenance_attacher.get(repo_id)
        if attacher is None:
            findings = self._get_findings(repo_id)
            repo_path = self.phase1._repo_paths.get(repo_id) or self.phase1.ingestion.repo_path_for(repo_id)
            if repo_path is None:
                record = self.phase1.db.get_repository(repo_id) or {}
                url = record.get("url") or ""
                if url and Path(url).exists():
                    repo_path = Path(url).resolve()
            attacher = ProvenanceAttacher(repo_path, findings)
            self._provenance_attacher[repo_id] = attacher
        return attacher.attach(evidence)

    def _get_findings(self, repo_id: str) -> list:
        """Findings from DB (dicts) → Finding objects."""
        if repo_id not in self._findings_cache:
            from app.models.finding import Finding

            rows = self.phase1.db.get_findings(repo_id, limit=500)
            findings = []
            for r in rows:
                try:
                    findings.append(Finding(
                        id=r["id"], tool=r["tool"], rule_id=r["rule_id"],
                        message=r["message"], severity=r["severity"],
                        finding_type=r["finding_type"], file_path=r["file_path"],
                        line_start=r["line_start"] or 0, line_end=r["line_end"] or 0,
                        symbol_id=r["symbol_id"], confidence=r["confidence"] or 1.0,
                        status=r["status"], commit_hash=r["commit_hash"],
                        metadata=_json_or_empty(r["metadata"]), repo_id=repo_id))
                except (KeyError, TypeError) as exc:
                    logger.debug(f"Skipping malformed finding row: {exc}")
            self._findings_cache[repo_id] = findings
        return self._findings_cache[repo_id]

    def _reason(self, question: str, context: str, budget,
                contradictions: list | None = None) -> tuple[str, bool, float, bool]:
        """§10.2 evidence-grounded reasoning call.

        FIX-001: an answer is grounded iff it cites at least one evidence item
        and is not an abstention. Abstentions get confidence 0.0, ungrounded
        non-abstaining answers (hallucination risk) get 0.1.
        FIX-029: grounded answers are penalized proportionally to the number
        of contradictions detected between retrieved evidence chunks.
        """
        try:
            prompt = build_user_prompt(question, context)
            answer = self.llm.invoke(prompt, system_prompt=SYSTEM_PROMPT + SECURITY_SUFFIX)
        except Exception as exc:
            logger.error("reasoning_failed", extra={"extra_fields": {"error": str(exc)[:200]}})
            return ("I don't have enough evidence to answer this "
                    "(reasoning backend unavailable).", False, 0.0, True)

        citations = extract_citations(answer, budget.evidence)
        abstained = is_abstention(answer)
        grounded = bool(citations) and not abstained

        if abstained:
            confidence = 0.0
        elif grounded:
            confidence = min(1.0, 0.5 + 0.1 * len(citations))
            if budget.truncated:
                confidence *= 0.95
            if contradictions:
                penalty = min(0.3, 0.1 * len(contradictions))
                confidence *= (1.0 - penalty)
        else:
            confidence = 0.1  # ungrounded and not abstaining → hallucination risk

        return answer, grounded, confidence, abstained

    @staticmethod
    def _source_counts(evidence) -> dict[str, int]:
        """FIX-008: report retrieval ORIGIN counts, not the post-rerank label."""
        counts: dict[str, int] = {}
        for e in evidence:
            origin = e.chunk.metadata.get("retrieval_origin") or e.source
            counts[origin] = counts.get(origin, 0) + 1
        return counts

    def invalidate_repo(self, repo_id: str) -> None:
        """FIX-014: drop all cached state for a repo after Phase 1 mutations."""
        self._findings_cache.pop(repo_id, None)
        self._provenance_attacher.pop(repo_id, None)
        if hasattr(self.retrieval, "invalidate_repo"):
            self.retrieval.invalidate_repo(repo_id)
        logger.info("phase2_cache_invalidated", extra={"extra_fields": {"repo_id": repo_id}})

    def _recent_related_investigations(self, repo_id: str, question: str,
                                       top_n: int = 3) -> list[dict]:
        """FIX-031: past investigations whose query shares content words with
        the current question. Cheap keyword overlap — a safe MVP; a semantic
        matcher is future work."""
        try:
            if not hasattr(self.phase1.db, "get_recent_investigations"):
                return []
            from app.core.retrieval.lexical import tokenize_code

            current_tokens = set(tokenize_code(question))
            if not current_tokens:
                return []
            rows = self.phase1.db.get_recent_investigations(repo_id, limit=50)
            scored: list[tuple[float, dict]] = []
            for row in rows:
                past_tokens = set(tokenize_code(row.get("query") or ""))
                if not past_tokens:
                    continue
                overlap = len(current_tokens & past_tokens)
                if overlap == 0:
                    continue
                scored.append((overlap / len(current_tokens), row))
            scored.sort(key=lambda x: -x[0])
            return [row for _, row in scored[:top_n]]
        except Exception as exc:
            logger.debug(f"Prior-investigation lookup skipped: {exc}")
            return []


def _cfg():
    from app.config import get_settings

    return get_settings()


def _json_or_empty(value):
    try:
        return json.loads(value) if value else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _dedup_files(evidence) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for e in evidence:
        f = e.provenance.source_file if e.provenance else e.chunk.file_path
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out
