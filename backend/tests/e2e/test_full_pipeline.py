"""End-to-end tests (§17.3) — full pipeline with the real LLM.

Run:  pytest tests/e2e/ -v -m e2e
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.ingestion.github_client import resolve_repo_id  # noqa: E402
from app.core.knowledge.knowledge_service import Phase1Pipeline  # noqa: E402
from app.core.reasoning.query_pipeline import Phase2Pipeline  # noqa: E402
from app.models.query import QueryOptions  # noqa: E402

DEMO_REPO = Path(__file__).resolve().parents[3] / "data" / "repos" / "demo-web-shop"


@pytest.fixture(scope="module")
def phase2():
    if not DEMO_REPO.exists():
        pytest.skip("demo repo fixture not created")
    p1 = Phase1Pipeline()
    repo_id = resolve_repo_id(str(DEMO_REPO))
    if p1.db.get_repository(repo_id) is None:
        p1.build_knowledge(str(DEMO_REPO), fetch_github_data=False)
    else:
        p1.restore_knowledge(repo_id)
    return Phase2Pipeline(p1), repo_id


@pytest.mark.e2e
def test_full_pipeline_query(phase2):
    pipeline, repo_id = phase2
    question = "What does the login function do?"
    started = time.time()
    result = pipeline.execute_full_pipeline(
        repo_id, question, QueryOptions(generate_followups=False))
    elapsed = time.time() - started

    assert result.answer and len(result.answer) > 30
    assert result.grounded is True
    assert len(result.citations) > 0
    assert elapsed < 120  # documented budget with generous margin
    assert result.evidence_count > 0
    # citations must point at real files in the repo
    for c in result.citations:
        assert c.source_file, "citation has a source file"


@pytest.mark.e2e
def test_e2e_repeat_query_surfaces_prior_investigation(phase2):
    """FIX-031: a repeated question sees the first investigation as context,
    and the response reports it in prior_investigations."""
    pipeline, repo_id = phase2
    opts = QueryOptions(generate_followups=False)
    first = pipeline.execute_full_pipeline(repo_id, "What does the cart do?", opts)
    second = pipeline.execute_full_pipeline(repo_id, "What does the cart do?", opts)
    assert second.answer
    assert second.prior_investigations, "repeat query should surface the prior investigation"
    assert any("cart" in (p.get("query") or "") for p in second.prior_investigations)


@pytest.mark.e2e
def test_e2e_retrieval_sources_reports_origins(phase2):
    """FIX-008: retrieval_sources reflects lexical/dense/graph origins, not
    a single 'reranked' label."""
    pipeline, repo_id = phase2
    result = pipeline.execute_full_pipeline(
        repo_id, "Why is login slow?", QueryOptions(generate_followups=False))
    sources = set(result.retrieval_sources)
    assert sources, "retrieval_sources must not be empty"
    assert sources - {"reranked"}, f"only post-rerank labels found: {sources}"
