#!/usr/bin/env python3
"""CLI entry: run the Phase 3 evaluation (§14.4).

Usage:
  python scripts/run_eval.py [--repo <path_or_id>] [--baselines all|a,b,c,d] [--out file.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.evaluation.baselines import run_evaluation  # noqa: E402
from app.core.evaluation.benchmarks import get_benchmark  # noqa: E402
from app.core.ingestion.github_client import resolve_repo_id  # noqa: E402
from app.core.knowledge.knowledge_service import Phase1Pipeline  # noqa: E402
from app.core.reasoning.query_pipeline import Phase2Pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="RepoIntel AI — Phase 3 evaluation")
    parser.add_argument("--repo", default="data/repos/demo-web-shop")
    parser.add_argument("--baselines", default="all", help="all or comma list: a,b,c,d")
    parser.add_argument("--limit", type=int, default=0, help="limit question count")
    parser.add_argument("--out", default="data/evaluation_results.json")
    args = parser.parse_args()

    repo_id = resolve_repo_id(args.repo) if "/" in args.repo else args.repo
    phase1 = Phase1Pipeline()
    if phase1.db.get_repository(repo_id) is None:
        phase1.build_knowledge(args.repo, fetch_github_data=False)
    else:
        phase1.restore_knowledge(repo_id)
    phase2 = Phase2Pipeline(phase1)

    questions = get_benchmark("repointel-local")
    if args.limit:
        questions = questions[:args.limit]

    if args.baselines == "all":
        baselines = ["baseline_a", "baseline_b", "baseline_c", "variant_d"]
    else:
        baselines = [f"baseline_{x.strip()}" if x.strip() in "abc" else "variant_d"
                     for x in args.baselines.split(",")]

    print(f"Running evaluation: {len(baselines)} baselines × {len(questions)} questions", file=sys.stderr)
    report = run_evaluation(phase2, repo_id, questions, baselines=baselines)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["benchmarks"], indent=2))
    print(f"\nSaved to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
