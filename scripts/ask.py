#!/usr/bin/env python3
"""CLI entry: ask a question through the full Phase 2 pipeline.

Usage:
  python scripts/ask.py <repo_url_or_id> "question"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.knowledge.knowledge_service import Phase1Pipeline  # noqa: E402
from app.core.reasoning.query_pipeline import Phase2Pipeline  # noqa: E402
from app.core.ingestion.github_client import resolve_repo_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="RepoIntel AI — ask a question")
    parser.add_argument("repo", help="Repo URL (must be ingested) or repo_id")
    parser.add_argument("question")
    parser.add_argument("--no-agentic", action="store_true")
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    repo_id = resolve_repo_id(args.repo) if ("/" in args.repo or args.repo.startswith("http")) else args.repo

    phase1 = Phase1Pipeline()
    if phase1.db.get_repository(repo_id) is None:
        print(f"Repository not ingested, building knowledge for {args.repo}...", file=sys.stderr)
        phase1.build_knowledge(args.repo, fetch_github_data=False)
    else:
        phase1.restore_knowledge(repo_id)  # reload persisted vector+graph indexes

    from app.models.query import QueryOptions

    options = QueryOptions(top_k=args.top_k, use_agentic=not args.no_agentic,
                           use_nli=not args.no_nli)
    pipeline = Phase2Pipeline(phase1)
    response = pipeline.execute_full_pipeline(repo_id, args.question, options)
    print(json.dumps(response.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
