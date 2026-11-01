"""Integration tests (§17.2) — require an ingested knowledge base.

Run:  pytest tests/integration/ -v
(slow: builds a fixture knowledge base once per session)
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.knowledge.knowledge_service import Phase1Pipeline  # noqa: E402
from app.core.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from app.core.ingestion.github_client import resolve_repo_id  # noqa: E402

DEMO_REPO = Path(__file__).resolve().parents[3] / "data" / "repos" / "demo-web-shop"


@pytest.fixture(scope="session")
def pipeline():
    if not DEMO_REPO.exists():
        pytest.skip("demo repo fixture not created (run scripts/create_demo_repo.py)")
    p1 = Phase1Pipeline()
    if p1.db.get_repository(resolve_repo_id(str(DEMO_REPO))) is None:
        p1.build_knowledge(str(DEMO_REPO), fetch_github_data=False)
    else:
        p1.restore_knowledge(resolve_repo_id(str(DEMO_REPO)))
    return p1


@pytest.mark.integration
def test_lexical_search_returns_hits(pipeline):
    from app.core.retrieval.retrieval_service import RetrievalComponent

    retrieval = RetrievalComponent(pipeline.vector_indexer, pipeline.graph_store)
    repo_id = resolve_repo_id(str(DEMO_REPO))
    lexical = retrieval.lexical_search(repo_id, "login password hash", top_k=10)
    assert len(lexical) > 0
    assert lexical[0].score > 0
    assert lexical[0].source == "lexical"


@pytest.mark.integration
def test_dense_search_returns_hits(pipeline):
    from app.core.retrieval.retrieval_service import RetrievalComponent

    retrieval = RetrievalComponent(pipeline.vector_indexer, pipeline.graph_store)
    repo_id = resolve_repo_id(str(DEMO_REPO))
    dense = retrieval.dense_search(repo_id, "authentication and session handling", top_k=10)
    assert len(dense) > 0
    assert dense[0].source == "dense"


@pytest.mark.integration
def test_graph_search_callers(pipeline):
    from app.core.retrieval.retrieval_service import RetrievalComponent

    retrieval = RetrievalComponent(pipeline.vector_indexer, pipeline.graph_store)
    repo_id = resolve_repo_id(str(DEMO_REPO))
    graph = retrieval.graph_search(repo_id, "who calls validate_credentials?", top_k=10)
    assert len(graph) > 0
    assert graph[0].source == "graph"
    assert graph[0].score == 1.0  # §7.3.5 graph matches are exact


@pytest.mark.integration
def test_hybrid_retrieval_fusion(pipeline):
    from app.core.retrieval.retrieval_service import RetrievalComponent

    retrieval = RetrievalComponent(pipeline.vector_indexer, pipeline.graph_store)
    repo_id = resolve_repo_id(str(DEMO_REPO))
    step4 = retrieval.execute_step_4(repo_id, "login lockout retry", top_k=5)
    assert step4.total_retrieved > 0
    assert len(step4.evidence) > 0
    fused = reciprocal_rank_fusion([step4.evidence, []])
    assert len(fused) > 0


@pytest.mark.integration
def test_findings_store_roundtrip(pipeline):
    repo_id = resolve_repo_id(str(DEMO_REPO))
    findings = pipeline.db.get_findings(repo_id, severity="critical")
    assert len(findings) >= 2
    paths = {f["file_path"] for f in findings}
    assert "demo_shop/auth.py" in paths


@pytest.mark.integration
def test_impact_analysis(pipeline):
    repo_id = resolve_repo_id(str(DEMO_REPO))
    impact = pipeline.graph_store.get_impact_analysis(repo_id, "validate_credentials")
    names = {r.get("name") for r in impact}
    assert "login" in names        # direct caller
    assert "handle_login" in names  # transitive caller (cross-file)
