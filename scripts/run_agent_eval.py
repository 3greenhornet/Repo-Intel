#!/usr/bin/env python3
"""Evaluate the planner against two baselines on the 8 demo questions."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("LLM_PROVIDER", os.environ.get("LLM_PROVIDER", "openai"))
if os.environ.get("GROQ_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["GROQ_API_KEY"]
os.environ.setdefault("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
os.environ.setdefault("LLM_MODEL", "openai/gpt-oss-120b")

from app.core.agent.investigator import Investigator  # noqa: E402
from app.core.evaluation.benchmarks import DEMO_BENCHMARK  # noqa: E402
from app.core.evaluation.metrics import groundedness_score  # noqa: E402
from app.core.ingestion.github_client import resolve_repo_id  # noqa: E402
from app.core.knowledge.knowledge_service import Phase1Pipeline  # noqa: E402
from app.core.reasoning.llm_client import LLMClient  # noqa: E402
from app.core.reasoning.query_pipeline import Phase2Pipeline  # noqa: E402
from app.models.query import QueryOptions  # noqa: E402

DEMO_REPO = Path(__file__).resolve().parents[1] / "data" / "repos" / "demo-web-shop"
OUT_JSON = Path(__file__).resolve().parents[1] / "data" / "eval_results.json"
OUT_PLOT = Path(__file__).resolve().parents[1] / "data" / "eval_coverage_vs_tokens.png"


def coverage(text: str, relevant_files: list[str]) -> float:
    if not relevant_files:
        return 0.0
    return sum(1 for f in relevant_files if f in text) / len(relevant_files)


def run_single_shot(phase2, repo_id, question):
    opts = QueryOptions(use_agentic=False, use_nli=False, generate_followups=False)
    tokens_before = phase2.llm.tokens_in + phase2.llm.tokens_out
    t0 = time.monotonic()
    resp = phase2.execute_full_pipeline(repo_id, question, opts)
    return {
        "answer": resp.answer,
        "steps": 1,
        "tokens": (phase2.llm.tokens_in + phase2.llm.tokens_out) - tokens_before,
        "elapsed": time.monotonic() - t0,
        "groundedness": groundedness_score(resp.answer, [c.snippet for c in resp.citations]),
    }


def run_fixed_explorer(phase2, repo_id, question):
    opts = QueryOptions(use_agentic=True, use_nli=False, generate_followups=False)
    tokens_before = phase2.llm.tokens_in + phase2.llm.tokens_out
    t0 = time.monotonic()
    resp = phase2.execute_full_pipeline(repo_id, question, opts)
    return {
        "answer": resp.answer,
        "steps": resp.exploration_iterations,
        "tokens": (phase2.llm.tokens_in + phase2.llm.tokens_out) - tokens_before,
        "elapsed": time.monotonic() - t0,
        "groundedness": groundedness_score(resp.answer, [c.snippet for c in resp.citations]),
    }


def run_planner(inv, repo_id, question):
    r = inv.run(repo_id, question)
    evidence_texts = []
    for entry in r.get("trace", []):
        for ev in entry.get("observation", {}).get("evidence", []):
            if isinstance(ev, dict) and ev.get("content"):
                evidence_texts.append(ev["content"])
    return {
        "answer": r["report"],
        "steps": r["steps"],
        "tokens": r["tokens_used"],
        "elapsed": r["elapsed_seconds"],
        "budget_exhausted": r["budget_exhausted"],
        "trace_path": r["trace_path"],
        "groundedness": groundedness_score(r["report"], evidence_texts),
    }


def main() -> int:
    if not DEMO_REPO.exists():
        print(f"Demo repo missing. Run: python scripts/create_demo_repo.py {DEMO_REPO}",
              file=sys.stderr)
        return 1

    repo_id = resolve_repo_id(str(DEMO_REPO))
    phase1 = Phase1Pipeline()
    if phase1.db.get_repository(repo_id) is None:
        phase1.build_knowledge(str(DEMO_REPO), fetch_github_data=False)
    else:
        phase1.restore_knowledge(repo_id)

    phase2 = Phase2Pipeline(phase1, llm_client=LLMClient())
    inv = Investigator(phase1, phase2, phase2.llm, max_steps=8, max_time_seconds=120.0)

    results: list[dict] = []
    for i, q in enumerate(DEMO_BENCHMARK, 1):
        question = q["question"]
        rel_files = q.get("relevant_files", [])
        print(f"\n[{i}/{len(DEMO_BENCHMARK)}] {question}")

        for config, fn in (
            ("single_shot", lambda: run_single_shot(phase2, repo_id, question)),
            ("fixed_explorer", lambda: run_fixed_explorer(phase2, repo_id, question)),
            ("planner", lambda: run_planner(inv, repo_id, question)),
        ):
            print(f"  running {config} ...", end="", flush=True)
            try:
                r = fn()
            except Exception as exc:
                print(f" FAILED: {exc}")
                results.append({"config": config, "question": question,
                                "error": str(exc)[:300]})
                continue
            r["config"] = config
            r["question"] = question
            r["coverage"] = coverage(r["answer"], rel_files)
            results.append(r)
            print(f" cov={r['coverage']:.2f} tok={r['tokens']} steps={r['steps']}")
            time.sleep(1.0)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {len(results)} rows to {OUT_JSON}")

    print("\n=== SUMMARY ===")
    print(f"{'config':<16}{'mean_cov':>12}{'mean_tokens':>14}{'mean_steps':>12}{'mean_s':>10}")
    for config in ("single_shot", "fixed_explorer", "planner"):
        rows = [r for r in results if r.get("config") == config and "error" not in r]
        if not rows:
            continue
        print(
            f"{config:<16}"
            f"{sum(r['coverage'] for r in rows)/len(rows):>12.3f}"
            f"{sum(r['tokens'] for r in rows)/len(rows):>14.0f}"
            f"{sum(r['steps'] for r in rows)/len(rows):>12.1f}"
            f"{sum(r['elapsed'] for r in rows)/len(rows):>10.2f}"
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"single_shot": "#888", "fixed_explorer": "#3b82f6", "planner": "#dc2626"}
        for config in colors:
            xs = [r["tokens"] for r in results if r.get("config") == config and "error" not in r]
            ys = [r["coverage"] for r in results if r.get("config") == config and "error" not in r]
            if xs:
                ax.scatter(xs, ys, label=config, color=colors[config], s=60, alpha=0.8)
        ax.set_xlabel("LLM tokens per investigation")
        ax.set_ylabel("Coverage (fraction of relevant files cited)")
        ax.set_title("Coverage vs. token cost: three investigation configs")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_PLOT, dpi=150)
        print(f"\nPlot saved to {OUT_PLOT}")
    except ImportError:
        print("\nmatplotlib not installed; skipping plot")

    print("\n=== FAILURES (planner only) ===")
    for r in results:
        if r.get("config") != "planner" or "error" in r:
            continue
        if r.get("budget_exhausted"):
            print(f"[budget_exhausted] {r['question'][:70]}")
        elif r.get("coverage", 0) < 0.5:
            print(f"[low_coverage={r['coverage']:.2f}] {r['question'][:70]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
