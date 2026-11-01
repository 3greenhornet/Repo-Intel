#!/usr/bin/env python3
"""CLI entry: run Phase 1 (build knowledge) on a repository.

Usage:
  python scripts/build_knowledge.py <repo_url_or_local_path> [--branch main] [--no-github]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.knowledge.knowledge_service import Phase1Pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="RepoIntel AI — Phase 1 knowledge builder")
    parser.add_argument("repo", help="GitHub URL or local path")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--no-github", action="store_true", help="Skip GitHub API fetches")
    args = parser.parse_args()

    pipeline = Phase1Pipeline()
    report = pipeline.build_knowledge(
        args.repo, branch=args.branch, fetch_github_data=not args.no_github)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
