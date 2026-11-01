"""Component 9A: Baselines (§14.1) + evaluation runner (§14.4).

Four configurations share one pipeline, exactly as documented:
- Baseline A: vector RAG (dense only, no fusion/rerank/agentic)
- Baseline B: graph retrieval only
- Baseline C: hybrid (lexical+dense+graph, RRF, rerank) — no agentic
- Variant D: full RepoIntel (hybrid + agentic exploration + NLI)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.core.evidence.context_builder import (
    assemble_context, budget_context, select_evidence,
)
from app.core.reasoning.prompt_templates import (
    SECURITY_SUFFIX, SYSTEM_PROMPT, build_user_prompt,
)
from app.utils.logger import get_logger
from app.utils.time import utc_now_iso

logger = get_logger(__name__)

BASELINE_CONFIGS = {  # §14.1 yaml blocks
    "baseline_a": {"retrieval": "dense_only", "fusion": "none", "reranking": "none",
                   "agentic": False},
    "baseline_b": {"retrieval": "graph_only", "fusion": "none", "reranking": "none",
                   "agentic": False},
    "baseline_c": {"retrieval": "hybrid", "fusion": "rrf", "reranking": "cross_encoder",
                   "agentic": False},
    "variant_d": {"retrieval": "hybrid", "fusion": "rrf", "reranking": "cross_encoder",
                   "agentic": True},
}


@dataclass
class EvalResult:
    baseline: str
    question: str
    answer: str = ""
    evidence: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    iterations: int = 0
    error: str = ""


def response_metadata_evidence(response) -> list[dict]:
    """QueryResponse selected evidence → evidence dicts for metric computation.

    Includes citation entries (chunk ids + snippets, used by citation
    correctness / groundedness) plus the deduped selected-file list (used by
    retrieval metrics — apples-to-apples with baselines A/B/C).
    """
    entries = [
        {"evidence_id": c.evidence_id, "source_file": c.source_file,
         "line_start": c.line_start, "line_end": c.line_end, "content": c.snippet}
        for c in response.citations]
    entries += [
        {"evidence_id": f"file:{f}", "source_file": f,
         "line_start": 0, "line_end": 0, "content": ""}
        for f in response.evidence_files]
    return entries


class BaselineRunner:
    """§14.1: every baseline shares the same interface as RepoIntel."""

    def __init__(self, phase2) -> None:
        self.p2 = phase2
        self.llm = phase2.llm

    def run(self, baseline: str, repo_id: str, query: str, top_k: int = 10) -> EvalResult:
        started = time.monotonic()  # FIX-042: monotonic for durations
        result = EvalResult(baseline=baseline, question=query)
        try:
            if baseline == "baseline_a":  # §14.1.1 dense only
                evidence = self.p2.retrieval.dense_search(repo_id, query, top_k)
            elif baseline == "baseline_b":  # §14.1.2 graph only
                evidence = self.p2.retrieval.graph_search(repo_id, query, top_k)
            elif baseline == "baseline_c":  # §14.1.3 hybrid, no agentic
                step4 = self.p2.retrieval.execute_step_4(repo_id, query, top_k=top_k)
                evidence = step4.evidence
            else:  # variant_d §14.1.4 full RepoIntel
                from app.models.query import QueryOptions

                response = self.p2.execute_full_pipeline(
                    repo_id, query, QueryOptions(generate_followups=False))
                result.answer = response.answer
                result.citations = [c.__dict__ for c in response.citations]
                result.iterations = response.exploration_iterations
                result.elapsed_seconds = response.elapsed_seconds
                # populate evidence for retrieval-quality metrics
                result.evidence = response_metadata_evidence(response)
                return result

            # shared LLM step for A/B/C (§14.1: direct LLM call, no agentic)
            selected = select_evidence(evidence)
            selected = self.p2._attach_provenance(repo_id, selected)
            budget = budget_context(selected)
            context = assemble_context(budget.evidence)
            prompt = build_user_prompt(query, context)
            answer = self.llm.invoke(prompt, SYSTEM_PROMPT + SECURITY_SUFFIX)
            result.answer = answer
            result.evidence = [
                {"evidence_id": e.chunk.id, "source_file": e.chunk.file_path,
                 "line_start": e.chunk.start_line, "line_end": e.chunk.end_line,
                 "content": e.chunk.content}
                for e in budget.evidence]
            from app.core.reasoning.output_generator import extract_citations

            result.citations = [c.__dict__ for c in extract_citations(answer, budget.evidence)]
        except Exception as exc:
            result.error = str(exc)[:300]
        result.elapsed_seconds = round(time.monotonic() - started, 2)
        return result


def run_evaluation(phase2, repo_id: str, questions: list[dict],
                   baselines: list[str] | None = None) -> dict:
    """§14.4 run_evaluation — all baselines over a question set.

    Each question dict: {"question": str, "relevant_files": [str],
    "relevant_symbols": [str]} used for retrieval-quality metrics.
    """

    from app.core.evaluation.metrics import (
        citation_correctness, compute_cost, groundedness_score, latency_stats,
        mrr, ndcg_at_k, recall_at_k,
    )

    runner = BaselineRunner(phase2)
    baselines = baselines or list(BASELINE_CONFIGS)
    report: dict = {
        "timestamp": utc_now_iso(),
        "repo_id": repo_id,
        "questions": len(questions),
        "benchmarks": {},
    }

    for baseline in baselines:
        results: list[EvalResult] = []
        for q in questions:
            results.append(runner.run(baseline, repo_id, q["question"]))

        latencies = [r.elapsed_seconds for r in results if not r.error]
        answers = [r for r in results if not r.error]

        # retrieval quality: relevant_files coverage in evidence order
        # (deduplicate file paths — several chunks can come from one file)
        recalls, mrrs, ndcgs = [], [], []
        for r, q in zip(results, questions):
            seen: set[str] = set()
            retrieved_ids: list[str] = []
            for e in r.evidence:
                f = e["source_file"]
                if f not in seen:
                    seen.add(f)
                    retrieved_ids.append(f)
            relevant = q.get("relevant_files", [])
            if relevant:
                recalls.append(recall_at_k(retrieved_ids, relevant, 10))
                mrrs.append(mrr(retrieved_ids, relevant))
                ndcgs.append(ndcg_at_k(retrieved_ids, relevant, 10))

        # answer quality
        g_scores = []
        for r, q in zip(results, questions):
            if r.answer:
                g_scores.append(groundedness_score(
                    r.answer, [e["content"] for e in r.evidence]))
        c_scores = [
            citation_correctness(r.answer, r.citations, r.evidence)
            for r in answers if r.citations]

        report["benchmarks"][baseline] = {
            "config": BASELINE_CONFIGS[baseline],
            "completed": len(answers),
            "errors": sum(1 for r in results if r.error),
            "recall@10": round(sum(recalls) / len(recalls), 3) if recalls else None,
            "mrr": round(sum(mrrs) / len(mrrs), 3) if mrrs else None,
            "ndcg@10": round(sum(ndcgs) / len(ndcgs), 3) if ndcgs else None,
            "groundedness": round(sum(g_scores) / len(g_scores), 3) if g_scores else None,
            "citation_correctness": round(sum(c_scores) / len(c_scores), 3) if c_scores else None,
            "avg_citations": round(
                sum(len(r.citations) for r in answers) / len(answers), 2) if answers else 0,
            "avg_iterations": round(
                sum(r.iterations for r in answers) / len(answers), 2) if answers else 0,
            "latency": latency_stats(latencies),
        }

    # cost accounting (§14.3.4)
    report["cost"] = compute_cost(
        [{"input_tokens": phase2.llm.tokens_in, "output_tokens": phase2.llm.tokens_out}])
    report["llm_calls"] = phase2.llm.calls

    # FIX-034 + 2026 correction: RAGAS when installed + an LLM judge always
    # attempted via the configured provider (works with the z-ai bridge too).
    if report["benchmarks"].get("variant_d"):
        judge_scores: list[dict] = []
        try:
            from app.core.evaluation.ragas_metrics import (
                llm_judge_scores, ragas_available,
            )

            if ragas_available():
                report["ragas"] = _ragas_report(phase2, repo_id, questions)
            # LLM-as-a-Judge on the full-pipeline variant (first 5 questions
            # to bound cost — documented limitation)
            for q in questions[:5]:
                resp = runner.run("variant_d", repo_id, q["question"])
                if resp.answer and not resp.error:
                    ev_texts = [e.get("content", "") for e in resp.evidence]
                    scores = llm_judge_scores(phase2.llm, q["question"], resp.answer, ev_texts)
                    if scores:
                        judge_scores.append(scores)
            if judge_scores:
                report["llm_judge"] = {
                    k: round(sum(s[k] for s in judge_scores) / len(judge_scores), 3)
                    for k in judge_scores[0]}
        except Exception as exc:
            logger.warning(f"RAGAS/judge evaluation failed: {exc}")
            report["ragas_error"] = str(exc)[:200]
    return report


def _ragas_report(phase2, repo_id: str, questions: list[dict]) -> dict:
    """FIX-034: RAGAS metrics over the full-pipeline variant."""
    from app.core.evaluation.ragas_metrics import evaluate_rag_quality

    runner = BaselineRunner(phase2)
    answers, q_texts, contexts, truths = [], [], [], []
    for q in questions:
        r = runner.run("variant_d", repo_id, q["question"])
        if r.answer and not r.error:
            q_texts.append(q["question"])
            answers.append(r.answer)
            contexts.append([e.get("content", "") for e in r.evidence])
            truths.append("")
    if not answers:
        return {}
    return evaluate_rag_quality(q_texts, answers, contexts, truths)
