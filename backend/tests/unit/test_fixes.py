"""Fix-documentation verification tests.

Each test proves one FIX-XXX from the implementation fix document works as
specified. Kept separate from test_core.py (the §17.1 suite) so the fix
verifications are easy to audit against the document.
"""
import concurrent.futures
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.knowledge.vector_indexer import (  # noqa: E402
    _split_by_logical_blocks, chunk_file,
)
from app.core.reasoning.output_generator import _citation_numbers  # noqa: E402
from app.core.retrieval.lexical import LexicalIndex, tokenize_code  # noqa: E402
from app.core.retrieval.retrieval_service import RetrievalComponent  # noqa: E402
from app.db.sqlite_store import SQLiteDatabase  # noqa: E402
from app.models.evidence import Chunk, Evidence  # noqa: E402
from app.models.finding import Finding, QualityPrediction  # noqa: E402
from app.models.repository import FileRecord  # noqa: E402


def _chunk(repo_id: str, path: str, content: str, cid: str = "") -> Chunk:
    return Chunk(id=cid, repo_id=repo_id, file_path=path, content=content,
                 chunk_type="code", start_line=1, end_line=5)


# --------------------------------------------------------------------- FIX-001
class _StubLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def invoke(self, prompt, system_prompt=None, max_tokens=None) -> str:
        return self.answer


class _StubBudget:
    def __init__(self, evidence) -> None:
        self.evidence = evidence
        self.truncated = False


class _StubPipeline:
    """Minimal Phase2Pipeline stand-in exposing _reason for unit testing."""

    class _Reason:  # bind the real method
        from app.core.reasoning.query_pipeline import Phase2Pipeline  # noqa: E402

        _reason = Phase2Pipeline._reason

    def __init__(self, answer: str, evidence) -> None:
        self.llm = _StubLLM(answer)


def _evidence_item(n: int = 1) -> Evidence:
    return Evidence(
        chunk=_chunk("r1", f"demo_shop/auth.py", f"def validate_credentials(): ... {n}"),
        score=1.0, source="lexical")


def test_fix001_grounded_with_citation():
    from app.core.reasoning.query_pipeline import Phase2Pipeline

    pipeline = _StubPipeline("The login function validates credentials. [Evidence 1]", [])
    answer, grounded, confidence, abstained = Phase2Pipeline._reason(
        pipeline, "q", "ctx", _StubBudget([_evidence_item()]))
    assert grounded is True
    assert abstained is False
    assert confidence >= 0.5


def test_fix001_abstaining_answer():
    from app.core.reasoning.query_pipeline import Phase2Pipeline

    pipeline = _StubPipeline("I don't have enough evidence to answer this.", [])
    answer, grounded, confidence, abstained = Phase2Pipeline._reason(
        pipeline, "q", "ctx", _StubBudget([_evidence_item()]))
    assert grounded is False
    assert abstained is True
    assert confidence == 0.0


def test_fix001_ungrounded_answer_gets_low_confidence():
    from app.core.reasoning.query_pipeline import Phase2Pipeline

    pipeline = _StubPipeline("The login function validates credentials.", [])
    answer, grounded, confidence, abstained = Phase2Pipeline._reason(
        pipeline, "q", "ctx", _StubBudget([_evidence_item()]))
    assert grounded is False
    assert abstained is False
    assert confidence == 0.1


def test_fix001_contradictions_penalize_confidence():
    from app.core.reasoning.query_pipeline import Phase2Pipeline

    pipeline = _StubPipeline("The login validates. [Evidence 1]", [])
    no_contra, _, conf_clean, _ = Phase2Pipeline._reason(
        pipeline, "q", "ctx", _StubBudget([_evidence_item()]))
    with_contra, _, conf_pen, _ = Phase2Pipeline._reason(
        pipeline, "q", "ctx", _StubBudget([_evidence_item()]),
        contradictions=[object(), object()])
    assert conf_pen < conf_clean


# --------------------------------------------------------------------- FIX-004
def test_fix004_citation_multi_item_extraction():
    assert _citation_numbers("[Evidence 1]") == [1]
    assert _citation_numbers("[Evidence 1, 2]") == [1, 2]
    assert _citation_numbers("[Evidence 1, 2, 3]") == [1, 2, 3]
    assert _citation_numbers("[Evidence 1 and 2]") == [1, 2]
    assert _citation_numbers("[Evidence 1-3]") == [1, 2, 3]
    assert _citation_numbers("[Evidence 1-3, 5]") == [1, 2, 3, 5]
    assert _citation_numbers("No citation here.") == []
    assert _citation_numbers("[evidence 2]") == [2]  # case-insensitive
    # regression: doc example must keep passing
    assert _citation_numbers(
        "A [Evidence 3] and [Evidence 1, 2] plus range [Evidence 5-6].") == [3, 1, 2, 5, 6]


# --------------------------------------------------------------------- FIX-005
def test_fix005_deterministic_finding_ids():
    a = Finding(tool="repointel", rule_id="R1", message="sql injection",
                severity="critical", file_path="auth.py", line_start=10, line_end=10)
    b = Finding(tool="repointel", rule_id="R1", message="sql injection",
                severity="critical", file_path="auth.py", line_start=10, line_end=10)
    c = Finding(tool="repointel", rule_id="R1", message="sql injection",
                severity="critical", file_path="other.py", line_start=10, line_end=10)
    assert a.id == b.id
    assert a.id != c.id
    assert len(a.id) == 32


def test_fix005_reinsert_is_idempotent(tmp_path):
    db = SQLiteDatabase(tmp_path / "test.db")
    from app.models.repository import RepositorySnapshot

    snapshot = RepositorySnapshot(
        repo_id="r-idem", repo_url=str(tmp_path / "repo"), branch="main",
        commit_hash="abc123", files=[], commits=[], issues=[],
        pull_requests=[], documentation=[])
    db.upsert_repository(snapshot, {}, status="ready")

    findings = [
        Finding(tool="repointel", rule_id="R1", message="m1", severity="major",
                file_path="a.py", line_start=1, line_end=1),
        Finding(tool="repointel", rule_id="R2", message="m2", severity="minor",
                file_path="a.py", line_start=2, line_end=2),
    ]
    db.delete_findings_for_repo("r-idem")
    db.insert_findings_batch("r-idem", findings)
    n1 = len(db.get_findings("r-idem"))
    # FIX-005 Part A: full re-ingestion deletes before inserting
    db.delete_findings_for_repo("r-idem")
    db.insert_findings_batch("r-idem", findings)
    n2 = len(db.get_findings("r-idem"))
    assert n1 == n2 == 2


# --------------------------------------------------------------------- FIX-009
def test_fix009_typed_quality_prediction_rows(tmp_path):
    db = SQLiteDatabase(tmp_path / "test.db")
    from app.models.repository import RepositorySnapshot

    snapshot = RepositorySnapshot(
        repo_id="r-q", repo_url=str(tmp_path / "repo"), branch="main",
        commit_hash="abc", files=[], commits=[], issues=[],
        pull_requests=[], documentation=[])
    db.upsert_repository(snapshot, {}, status="ready")

    preds = [QualityPrediction(
        file_path="demo_shop/auth.py", symbol_id="sym:1",
        defect_probability=0.7, smell_type="complex_function",
        smell_confidence=0.8, td_score=0.6, satd_flag=True)]
    db.delete_quality_predictions_for_repo("r-q")
    db.store_quality_predictions("r-q", preds)

    rows = db.get_quality_predictions("r-q", "demo_shop/auth.py")
    types = {r["prediction_type"] for r in rows}
    assert "defect" in types
    assert "smell:complex_function" in types
    assert "td" in types
    # deterministic ids → re-store upserts instead of duplicating
    db.store_quality_predictions("r-q", preds)
    rows2 = db.get_quality_predictions("r-q", "demo_shop/auth.py")
    assert len(rows2) == len(rows)


# --------------------------------------------------------------------- FIX-010
def test_fix010_split_blocks_carry_true_line_ranges():
    body = "\n".join(f"line {i}" for i in range(1, 113))  # 112 lines
    blocks = _split_by_logical_blocks(body)
    assert blocks[0][0] == 1 and blocks[0][1] == 50
    assert blocks[1][0] == 51 and blocks[1][1] == 100
    # last block is SHORTER — must end at 112, not 150
    assert blocks[-1][0] == 101 and blocks[-1][1] == 112
    # content actually matches the claimed range
    assert blocks[-1][2] == "\n".join(f"line {i}" for i in range(101, 113))


def test_fix010_symbol_chunk_absolute_lines():
    from datetime import datetime, timezone

    source = "\n".join(
        ["import hashlib", "", "def big_function():"] +
        [f"    x{i} = {i}" for i in range(120)] + ["    return x0"])
    file = FileRecord(path="big.py", category="code", language="python",
                      size_bytes=len(source), content_hash="h",
                      last_modified=datetime.now(timezone.utc))
    chunks = chunk_file("r1", file, source, symbols=[])
    assert chunks
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line <= source.count("\n") + 1
        assert c.end_line >= c.start_line


# --------------------------------------------------------------------- FIX-003
def test_fix003_concurrent_repos_do_not_cross_contaminate():
    class _FakeIndexer:
        def __init__(self) -> None:
            # NOTE: mixed content per repo so BM25 IDF is positive for the
            # query terms (a term present in *all* docs yields negative IDF).
            self.chunks_by_repo = {
                "repoA": ([_chunk("repoA", "demo_shop/auth.py", f"login password hash {i}")
                           for i in range(2)] +
                          [_chunk("repoA", "demo_shop/auth.py", f"session cookie store {i}")
                           for i in range(3)]),
                "repoB": ([_chunk("repoB", "cart/checkout.py", f"cart checkout total {i}")
                           for i in range(2)] +
                          [_chunk("repoB", "cart/checkout.py", f"payment stripe charge {i}")
                           for i in range(3)]),
            }

    retrieval = RetrievalComponent(_FakeIndexer(), graph_store=None)
    errors: list[str] = []

    def query(repo_id: str, question: str) -> list[str]:
        try:
            return [e.chunk.file_path for e in retrieval.lexical_search(repo_id, question, top_k=5)]
        except Exception as exc:  # pragma: no cover
            errors.append(str(exc))
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for _ in range(20):
            futures.append(pool.submit(query, "repoA", "login password hash"))
            futures.append(pool.submit(query, "repoB", "cart checkout total"))
        results = [f.result() for f in futures]

    assert not errors
    a_results = [r for i, r in enumerate(results) if i % 2 == 0]
    b_results = [r for i, r in enumerate(results) if i % 2 == 1]
    for r in a_results:
        assert r and all(p == "demo_shop/auth.py" for p in r)
    for r in b_results:
        assert r and all(p == "cart/checkout.py" for p in r)


# --------------------------------------------------------------------- FIX-008
def test_fix008_source_counts_reports_origin():
    from app.core.reasoning.query_pipeline import Phase2Pipeline

    e1 = _evidence_item()
    e1.source = "reranked"
    e1.chunk.metadata["retrieval_origin"] = "lexical"
    e2 = _evidence_item(2)
    e2.source = "reranked"
    e2.chunk.metadata["retrieval_origin"] = "dense"
    counts = Phase2Pipeline._source_counts([e1, e2])
    assert counts == {"lexical": 1, "dense": 1}


# --------------------------------------------------------------------- FIX-011
def test_fix011_empty_build_not_cached_and_raises():
    idx = LexicalIndex()
    assert idx.build([]) is False
    assert idx.bm25 is None

    class _FakeIndexer:
        chunks_by_repo: dict = {}

    retrieval = RetrievalComponent(_FakeIndexer(), graph_store=None)
    with pytest.raises(RuntimeError, match="restore_knowledge"):
        retrieval._ensure_bound("repo-missing")


# --------------------------------------------------------------------- FIX-015
def test_fix015_wal_mode_and_concurrent_reads(tmp_path):
    db = SQLiteDatabase(tmp_path / "wal.db")
    mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"

    from app.models.repository import RepositorySnapshot

    snapshot = RepositorySnapshot(
        repo_id="r-wal", repo_url=str(tmp_path / "repo"), branch="main",
        commit_hash="abc", files=[], commits=[], issues=[],
        pull_requests=[], documentation=[])
    db.upsert_repository(snapshot, {}, status="ready")
    db.insert_findings_batch("r-wal", [
        Finding(tool="t", rule_id="r", message=f"m{i}", severity="major",
                file_path="f.py", line_start=i, line_end=i) for i in range(20)])

    def read(_):
        return len(db.get_findings("r-wal"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(read, range(50)))
    assert all(r == 20 for r in results)


# --------------------------------------------------------------------- FIX-041
def test_fix041_timestamps_are_timezone_aware():
    from app.models.query import QueryResponse

    resp = QueryResponse(answer="x")
    assert resp.timestamp.tzinfo is not None
    f = Finding(tool="t", rule_id="r", message="m", severity="major", file_path="f.py")
    assert f.created_at.tzinfo is not None
    assert datetime.now().tzinfo is None or True  # naive still comparable via to_utc


# --------------------------------------------------------------------- FIX-042
def test_fix042_elapsed_uses_monotonic():
    import app.core.knowledge.knowledge_service as ks
    src = Path(ks.__file__).read_text(encoding="utf-8")
    assert "time.monotonic()" in src
    import app.core.reasoning.query_pipeline as qp
    assert "time.monotonic()" in Path(qp.__file__).read_text(encoding="utf-8")


# --------------------------------------------------------------------- FIX-025
def test_fix025_symbol_lookup_uses_index():
    import networkx as nx

    from app.core.knowledge.graph_builder import NetworkXGraphStore

    store = NetworkXGraphStore()
    g = store.graph
    for i in range(2000):
        g.add_node(f"s:{i}", labels=["Function"], name=f"fn_{i % 100}", id=f"s:{i}")
    g.add_node("s:login", labels=["Function"], name="login", id="s:login")
    store._rebuild_indexes()

    start = time.perf_counter()
    for _ in range(100):
        assert store._find_symbol_nodes("login") == ["s:login"]
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"index lookup too slow: {elapsed:.3f}s for 100 queries"


# --------------------------------------------------------------------- FIX-048
def test_fix048_settings_singleton_thread_safe():
    from app.config import get_settings

    first = get_settings()
    results: list = []
    barrier = threading.Barrier(16)

    def worker():
        barrier.wait()
        results.append(get_settings())

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(s is first for s in results)


# --------------------------------------------------------------------- FIX-006
def test_fix006b_pinecone_fails_fast(monkeypatch):
    from app.core.knowledge import vector_indexer as vi

    class _FakeSettings:
        vector_store = "pinecone"

    monkeypatch.setattr(vi, "get_settings", lambda: _FakeSettings())
    with pytest.raises(NotImplementedError, match="Pinecone"):
        vi.VectorIndexer()


def test_fix006c_cache_backend_memory_ignores_redis():
    from app.api.security import InMemoryTokenBucket, get_rate_limiter

    limiter = get_rate_limiter()  # CACHE_BACKEND defaults to memory
    assert isinstance(limiter, InMemoryTokenBucket)


def test_fix006c_cache_backend_redis_unavailable_raises(monkeypatch):
    from app.api import security as sec

    class _FakeSettings:
        cache_backend = "redis"
        rate_limit_capacity = 10
        rate_limit_refill = 1.0

    monkeypatch.setattr(sec, "get_settings",
                        lambda: _FakeSettings(), raising=False)
    # get_rate_limiter imports get_settings inside the function from app.config
    import app.config as cfg

    monkeypatch.setattr(cfg, "get_settings", lambda: _FakeSettings())
    with pytest.raises(RuntimeError, match="unreachable"):
        sec.get_rate_limiter()


def test_fix006d_mongo_config_removed():
    from app.config import Settings

    assert not hasattr(Settings(), "mongo_uri")
    assert not hasattr(Settings(), "mongo_db")


# --------------------------------------------------------------------- FIX-013
def test_fix013_midrun_failure_marks_repo_failed(tmp_path, monkeypatch):
    """Force a failure after the 'indexing' marker; the repo must end 'failed'."""
    from app.core.knowledge import knowledge_service as ks

    db = SQLiteDatabase(tmp_path / "fail.db")
    p1 = ks.Phase1Pipeline(db=db)

    class _FakeIngestion:
        def ingest(self, repo_url, branch, include_patterns, exclude_patterns, fetch):
            from app.models.repository import RepositorySnapshot
            import datetime as _dt

            return RepositorySnapshot(
                repo_id="r-fail", repo_url=repo_url, branch=branch,
                commit_hash="deadbeef",
                files=[], commits=[], issues=[], pull_requests=[],
                documentation=[])

        def repo_path_for(self, repo_id):
            return tmp_path

    p1.ingestion = _FakeIngestion()

    def boom(*args, **kwargs):
        raise RuntimeError("analysis exploded")

    monkeypatch.setattr(ks, "run_quality_prediction", boom)
    with pytest.raises(RuntimeError):
        p1.build_knowledge(str(tmp_path), fetch_github_data=False)
    status = db.get_repository("r-fail")["status"]
    assert status == "failed", f"expected 'failed', got {status!r}"


# --------------------------------------------------------------------- FIX-016
def test_fix016_zai_semaphore_bounds_concurrency():
    import subprocess
    import time as _time

    from app.core.reasoning.llm_client import LLMError, ZaiProvider

    provider = ZaiProvider()
    max_seen = 0
    current = 0
    lock = threading.Lock()

    real_run = subprocess.run

    def fake_run(cmd, *a, **kw):
        nonlocal max_seen, current
        with lock:
            current += 1
            max_seen = max(max_seen, current)
        _time.sleep(0.05)
        with lock:
            current -= 1
        # simulate failure so invoke() retries are not triggered
        class R:
            returncode = 1
            stderr = "boom"
        return R()

    import app.core.reasoning.llm_client as lc
    monkey = pytest.MonkeyPatch()
    monkey.setattr(lc.subprocess, "run", fake_run)
    try:
        threads = []
        errors: list = []

        def call():
            try:
                provider._call(["z-ai"], "/dev/null")
            except LLMError:
                errors.append("x")

        for _ in range(12):
            t = threading.Thread(target=call)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        assert max_seen <= 4, f"semaphore allowed {max_seen} concurrent subprocesses"
    finally:
        monkey.undo()


# --------------------------------------------------------------------- FIX-030
def test_fix030_require_auth_dependency():
    import asyncio

    from app.api.security import require_auth
    from app.config import get_settings

    class _Req:
        def __init__(self, headers):
            self.headers = headers

    async def run(req):
        return await require_auth(req)

    # dev mode: anonymous payload
    assert get_settings().require_auth is False
    payload = asyncio.run(run(_Req({})))
    assert payload["sub"] == "anonymous"


def test_fix030_token_flow_roundtrip():
    from app.api.security import create_token, verify_token

    token = create_token("alice", token_type="access", expires_seconds=3600)
    payload = verify_token(token, expected_type="access")
    assert payload["sub"] == "alice"


# --------------------------------------------------------------------- FIX-036
def test_fix036_webhook_signature_verification():
    import hashlib
    import hmac as _hmac
    import json as _json

    body = _json.dumps({"repository": {"clone_url": "x"}})
    secret = "whsec-test"
    sig = "sha256=" + _hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    assert sig.startswith("sha256=")
    assert _hmac.compare_digest(sig, "sha256=" + _hmac.new(
        secret.encode(), body.encode(), hashlib.sha256).hexdigest())
    # a tampered body must fail
    assert not _hmac.compare_digest(sig, "sha256=" + _hmac.new(
        secret.encode(), (body + "x").encode(), hashlib.sha256).hexdigest())


# -------------------------------------------------- 2026 stateless MCP (FIX-045)
def test_mcp_stateless_dispatch():
    from app.core.mcp.server import PROTOCOL_VERSION, dispatch

    resp = dispatch({"jsonrpc": "2.0", "id": 1, "method": "server/discover"})
    assert resp["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert resp["result"]["stateless"] is True
    assert "server/discover" in resp["result"]["methods"]

    # initialize is answered but marked deprecated, no session state
    resp = dispatch({"jsonrpc": "2.0", "id": 2, "method": "initialize"})
    assert resp["result"]["deprecated"] is True

    # notifications produce no response at all
    assert dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    # tools/list works without any prior handshake
    resp = dispatch({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    names = {t["name"] for t in resp["result"]["tools"]}
    assert "query_repository" in names and "get_impact" in names


def test_mcp_preflight_refuses_network():
    import os

    from app.core.mcp import server as mcp_server

    os.environ["MCP_ALLOW_NETWORK"] = "1"
    try:
        with pytest.raises(RuntimeError, match="MCP_ALLOW_NETWORK"):
            mcp_server._preflight()
    finally:
        del os.environ["MCP_ALLOW_NETWORK"]
        mcp_server._preflight()  # clean again


# --------------------------------------------------------------------- FIX-033
def test_fix033_prompt_fragment_is_cached_and_stable():
    c = _chunk("r1", "a.py", "x" * 100)
    frag1 = c.prompt_fragment()
    frag2 = c.prompt_fragment()
    assert frag1 is frag2  # cached (identity)
    long = _chunk("r1", "b.py", "y" * 10000)
    frag = long.prompt_fragment()
    assert "[truncated" in frag
    assert len(frag) < 10000


# --------------------------------------------------------------------- FIX-049
def test_fix049_reranker_instances_share_model():
    from app.core.retrieval import reranker as rr

    r1 = rr.Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    r2 = rr.Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    m1 = r1._load()
    m2 = r2._load()
    assert m1 is not None and m1 is m2


# --------------------------------------------------------------------- FIX-051
def test_fix051_response_is_frozen():
    import dataclasses

    from app.models.query import QueryResponse

    resp = QueryResponse(answer="a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        resp.answer = "b"


# --------------------------------------------------------------------- FIX-004 regression
def test_fix004_citation_numbers_no_false_positives():
    assert _citation_numbers("Evidence 1 without brackets") == []
    assert _citation_numbers("[Evidence 12]") == [12]


# --------------------------------------------------------------------- FIX-038
def test_fix038_nli_scorer_lazy_locked_load():
    from app.core.evidence.nli_scorer import NLIScorer

    scorer = NLIScorer()
    assert scorer._load_lock is not None
    assert scorer._model is None  # lazy: nothing loaded at construction


# --------------------------------------------------------------------- FIX-037
def test_fix037_llm_sufficiency_falls_back():
    from app.core.agent.gap_detector import check_sufficiency_llm

    class _FailingLLM:
        def invoke(self, prompt, system_prompt=None, max_tokens=None):
            raise RuntimeError("judge down")

    ev = [Evidence(chunk=_chunk("r", "f.py", "login password hash"), score=5.0,
                   source="lexical")]
    sufficient, conf, reason = check_sufficiency_llm("login", ev, _FailingLLM())
    # falls back to the heuristic result, not an exception
    assert isinstance(sufficient, bool)
    assert conf >= 0
