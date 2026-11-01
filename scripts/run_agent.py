#!/usr/bin/env python3
"""Run one investigation with the RepoIntel Investigator agent."""
import argparse
import json
import os
import sys
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
from app.core.ingestion.github_client import resolve_repo_id  # noqa: E402
from app.core.knowledge.knowledge_service import Phase1Pipeline  # noqa: E402
from app.core.reasoning.llm_client import LLMClient  # noqa: E402
from app.core.reasoning.query_pipeline import Phase2Pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="repo path, URL, or repo_id")
    parser.add_argument("goal")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    args = parser.parse_args()

    repo_ref = args.repo
    repo_id = resolve_repo_id(repo_ref) if ("/" in repo_ref or repo_ref.startswith("http")) else repo_ref

    phase1 = Phase1Pipeline()
    if phase1.db.get_repository(repo_id) is None:
        print(f"Repository not ingested; building knowledge for {repo_ref} ...", file=sys.stderr)
        phase1.build_knowledge(repo_ref, fetch_github_data=False)
    else:
        phase1.restore_knowledge(repo_id)

    phase2 = Phase2Pipeline(phase1, llm_client=LLMClient())
    inv = Investigator(phase1, phase2, phase2.llm,
                       max_steps=args.max_steps, max_time_seconds=args.max_seconds)

    result = inv.run(repo_id, args.goal)
    print("\n=== REPORT ===\n")
    print(result["report"])
    print("\n=== SUMMARY ===")
    print(json.dumps({
        "steps": result["steps"],
        "elapsed_seconds": result["elapsed_seconds"],
        "tokens_used": result["tokens_used"],
        "budget_exhausted": result["budget_exhausted"],
        "trace_path": result["trace_path"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
