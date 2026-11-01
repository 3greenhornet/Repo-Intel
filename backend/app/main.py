"""FastAPI main application — documented REST API (§10.3.2) + dashboard.

Endpoints:
- POST /api/v1/query           — full pipeline Q&A
- POST /api/v1/query/stream    — SSE streaming variant (§23.8)
- POST /api/v1/ingest          — trigger Phase 1 ingestion
- POST /api/v1/ingest/update   — incremental update (§4.1.5)
- GET  /api/v1/findings/{repo} — findings browser
- GET  /api/v1/findings/{repo}/recommendations — §11.2 recommendations
- GET  /api/v1/repos           — indexed repositories
- GET  /api/v1/architecture/{repo} — graph summary (MCP get_architecture)
- GET  /api/v1/impact/{repo}   — impact analysis
- GET  /api/v1/health          — dependency-validating health (§24.5)
- GET  /metrics                — Prometheus metrics (§18.2)
- GET  / — web dashboard
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.api.security import create_token, get_rate_limiter, require_auth, verify_token
from app.core.knowledge.knowledge_service import Phase1Pipeline
from app.core.reasoning.output_generator import compute_priority
from app.core.reasoning.query_pipeline import Phase2Pipeline
from app.core.retrieval.graph_retriever import extract_symbol
from app.utils.logger import get_logger, set_request_id
from app.utils.time import utc_now_iso

logger = get_logger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "app" / "static"

# --------------------------------------------------------------------- metrics (§18.2)
try:
    from prometheus_client import (Counter, Gauge, Histogram, generate_latest,
                                   CONTENT_TYPE_LATEST)

    queries_total = Counter("repointel_queries_total", "Total queries processed",
                            ["repo_id", "status"])
    query_latency = Histogram("repointel_query_latency_seconds", "Query latency",
                              buckets=[0.5, 1, 2, 5, 10, 30, 60, 120])
    findings_total = Gauge("repointel_findings_total", "Total findings stored")
    PROMETHEUS = True
except ImportError:  # pragma: no cover
    PROMETHEUS = False

# --------------------------------------------------------------------- tracing (OTel §Task 6)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("repointel")
    OTEL_AVAILABLE = True
except Exception:
    tracer = None
    OTEL_AVAILABLE = False

# ------------------------------------------------------------------------- app state
app = FastAPI(
    title="RepoIntel AI",
    version="1.0.0",
    description="LLM-powered repository intelligence platform (implementation of the "
                "RepoIntel AI specification)",
)
_rate_limiter = get_rate_limiter()
_phase1: Phase1Pipeline | None = None
_phase2: Phase2Pipeline | None = None


def get_pipelines() -> tuple[Phase1Pipeline, Phase2Pipeline]:
    global _phase1, _phase2
    if _phase1 is None:
        _phase1 = Phase1Pipeline()
        _phase2 = Phase2Pipeline(_phase1)
    return _phase1, _phase2


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """FIX-046: propagate a request id through every log line of a request.
    Must be registered BEFORE the rate limiter so ids exist for rate-limit logs."""
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    set_request_id(rid)
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """§24.1 — skip health checks, bucket per API key or client IP."""
    if request.url.path in ("/api/v1/health", "/metrics", "/"):
        return await call_next(request)
    user = request.headers.get("x-api-key") or (request.client.host if request.client else "anon")
    if not _rate_limiter.is_allowed(user):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
            headers={"Retry-After": "1"},
        )
    return await call_next(request)


# ------------------------------------------------------------------ auth (FIX-030)
class TokenRequestDTO(BaseModel):
    username: str
    api_key: str


class RefreshRequestDTO(BaseModel):
    refresh_token: str


@app.post("/api/v1/auth/token")
async def issue_token(req: TokenRequestDTO):
    """FIX-030: exchange a shared API key for access/refresh JWTs.

    NOTE: `api_key_salt` is a demo-grade shared secret — swap for a proper
    user store (hashed per-user keys) in production.
    """
    from app.config import get_settings

    settings = get_settings()
    if req.api_key != settings.api_key_salt:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {
        "access_token": create_token(req.username, token_type="access"),
        "refresh_token": create_token(req.username, token_type="refresh"),
        "token_type": "bearer",
    }


@app.post("/api/v1/auth/refresh")
async def refresh_token(req: RefreshRequestDTO):
    """FIX-030: exchange a valid refresh token for a new access token."""
    payload = verify_token(req.refresh_token, expected_type="refresh")
    return {"access_token": create_token(payload["sub"], token_type="access"),
            "token_type": "bearer"}


# ------------------------------------------------------- webhook (FIX-036)
@app.post("/api/v1/webhooks/github")
async def github_webhook(request: Request):
    """FIX-036: GitHub push webhook — verify HMAC-SHA256, trigger incremental.

    503 when GITHUB_WEBHOOK_SECRET is unset, 401 on bad/missing signature.
    """
    from app.config import get_settings
    from app.core.ingestion.github_client import resolve_repo_id

    settings = get_settings()
    secret = settings.github_webhook_secret
    if not secret:
        raise HTTPException(status_code=503, detail="GITHUB_WEBHOOK_SECRET not configured")

    signature = request.headers.get("x-hub-signature-256", "")
    if not signature.startswith("sha256="):
        raise HTTPException(status_code=401, detail="Missing signature")

    body = await request.body()
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="Invalid signature")

    event = request.headers.get("x-github-event", "")
    payload = await request.json()

    if event == "push":
        repo_url = payload.get("repository", {}).get("clone_url", "")
        if not repo_url:
            return {"status": "ignored", "reason": "no repository.clone_url"}
        repo_id = resolve_repo_id(repo_url)
        phase1, _ = get_pipelines()
        try:
            result = await asyncio.to_thread(phase1.incremental_update, repo_id)
            return {"status": "accepted", "repo_id": repo_id, "delta": result}
        except ValueError as exc:
            # Repo not yet ingested — acknowledge but note the skip
            return {"status": "skipped", "reason": str(exc)}
    if event == "ping":
        return {"status": "pong"}
    return {"status": "ignored", "event": event}


# ---------------------------------------------------------------------- request DTOs
class QueryRequestDTO(BaseModel):
    repo_id: str = Field(..., description="Repository UUID")
    question: str = Field(..., max_length=1000)
    options: dict = Field(default_factory=dict)


class InvestigateRequestDTO(BaseModel):
    repo_id: str
    question: str = Field(..., max_length=1000)
    max_steps: int = 8
    max_time_seconds: float = 120.0


class IngestRequestDTO(BaseModel):
    repo_url: str = Field(..., description="GitHub URL or local path")
    branch: str = "main"
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    fetch_github_data: bool = False


# --------------------------------------------------------------------------- endpoints
@app.post("/api/v1/query")
async def query(dto: QueryRequestDTO, user: dict = Depends(require_auth)):
    """§10.3.2 POST /api/v1/query — full pipeline response."""
    from app.models.query import QueryOptions

    phase1, phase2 = get_pipelines()
    phase1.restore_knowledge(dto.repo_id)
    options = QueryOptions(
        max_iterations=dto.options.get("max_iterations", 5),
        top_k=dto.options.get("top_k", 10),
        include_provenance=dto.options.get("include_provenance", True),
        use_agentic=dto.options.get("use_agentic", True),
        use_nli=dto.options.get("use_nli", True),
    )
    started = time.monotonic()
    span = tracer.start_span("query_pipeline") if tracer else None
    if span:
        span.set_attribute("repo_id", dto.repo_id)
        span.set_attribute("question_length", len(dto.question))

    try:
        response = await asyncio.to_thread(
            phase2.execute_full_pipeline, dto.repo_id, dto.question, options)
        if PROMETHEUS:
            queries_total.labels(repo_id=dto.repo_id, status="ok").inc()
            query_latency.observe(time.monotonic() - started)
        return response.to_dict()
    except ValueError as exc:
        if span:
            span.record_exception(exc)
        if PROMETHEUS:
            queries_total.labels(repo_id=dto.repo_id, status="error").inc()
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        if span:
            span.record_exception(exc)
        logger.error("query_endpoint_failed", extra={"extra_fields": {"error": str(exc)[:300]}})
        if PROMETHEUS:
            queries_total.labels(repo_id=dto.repo_id, status="error").inc()
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")
    finally:
        if span:
            span.end()


@app.post("/api/v1/query/stream")
async def query_stream(dto: QueryRequestDTO, user: dict = Depends(require_auth)):
    """§23.8 SSE streaming variant — streams the LLM tokens.

    FIX-028: uses `LLMClient.astream` — real token streaming for
    OpenAI/Anthropic providers; sentence-chunk pseudo-streaming for the
    z-ai bridge (request/response subprocess). The final event carries
    citations, contradictions (FIX-029) and iteration count.
    """
    phase1, phase2 = get_pipelines()
    phase1.restore_knowledge(dto.repo_id)

    async def event_generator():
        import json as _json

        try:
            # Step 1: retrieval + context (non-streaming, §23.8)
            yield f"data: {_json.dumps({'stage': 'retrieval', 'status': 'started'})}\n\n"
            from app.core.evidence.context_builder import (
                assemble_context, select_evidence,
            )
            from app.core.reasoning.prompt_templates import (
                SECURITY_SUFFIX, SYSTEM_PROMPT, build_user_prompt, validate_input_length,
            )
            validate_input_length(dto.question)
            step4 = await asyncio.to_thread(
                phase2._run_step4, dto.repo_id, dto.question, 10)
            exploration = await asyncio.to_thread(
                phase2.explorer.explore, dto.repo_id, dto.question, step4.evidence)
            selected = select_evidence(exploration.evidence)
            selected = await asyncio.to_thread(
                phase2._attach_provenance, dto.repo_id, selected)
            yield f"data: {_json.dumps({'stage': 'retrieval', 'status': 'done', 'evidence': len(selected)})}\n\n"

            # Step 2: NLI verification stage
            yield f"data: {_json.dumps({'stage': 'nli', 'status': 'started'})}\n\n"
            nli_relations = await asyncio.to_thread(phase2.nli.score_nli, selected)
            yield f"data: {_json.dumps({'stage': 'nli', 'status': 'done', 'relations': len(nli_relations)})}\n\n"

            context = assemble_context(selected, nli_relations)

            # Step 3: stream LLM output (FIX-028: real tokens when supported)
            yield f"data: {_json.dumps({'stage': 'answer', 'status': 'started'})}\n\n"
            prompt = build_user_prompt(dto.question, context)
            answer_buffer: list[str] = []
            async for token in phase2.llm.astream(prompt, SYSTEM_PROMPT + SECURITY_SUFFIX):
                answer_buffer.append(token)
                yield f"data: {_json.dumps({'token': token})}\n\n"
            answer = "".join(answer_buffer)

            # Step 3: final metadata (citations per §10.3.2 + contradictions FIX-029)
            from app.core.reasoning.output_generator import extract_citations

            citations = extract_citations(answer, selected)
            contradictions = [
                {"evidence_a": r.evidence_a, "evidence_b": r.evidence_b,
                 "confidence": round(r.confidence, 3)}
                for r in nli_relations if r.relation == "contradiction"
            ]
            yield "data: " + _json.dumps({
                "done": True, "citations": [
                    {"source_file": c.source_file, "line_start": c.line_start,
                     "line_end": c.line_end, "snippet": c.snippet}
                    for c in citations],
                "iterations": exploration.iterations,
                "contradictions": contradictions,
            }) + "\n\n"
        except Exception as exc:
            yield f"data: {_json.dumps({'error': str(exc)[:200]})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/v1/query/investigate")
async def query_investigate(dto: InvestigateRequestDTO, user: dict = Depends(require_auth)):
    """ReAct Investigator — exposes the full thought → action → observation trace."""
    from app.core.agent.investigator import Investigator

    phase1, phase2 = get_pipelines()
    phase1.restore_knowledge(dto.repo_id)

    investigator = Investigator(
        phase1, phase2, phase2.llm,
        max_steps=dto.max_steps,
        max_time_seconds=dto.max_time_seconds,
    )

    try:
        result = await asyncio.to_thread(investigator.run, dto.repo_id, dto.question)
        return result
    except Exception as exc:
        logger.error("investigate_failed", extra={"extra_fields": {"error": str(exc)[:300]}})
        raise HTTPException(status_code=500, detail=f"Investigation failed: {exc}")


@app.post("/api/v1/ingest")
async def ingest(dto: IngestRequestDTO, user: dict = Depends(require_auth)):
    """Trigger Phase 1 ingestion (§4). Runs in a worker thread."""
    phase1, _ = get_pipelines()
    try:
        report = await asyncio.to_thread(
            phase1.build_knowledge, dto.repo_url, dto.branch,
            dto.include_patterns or None, dto.exclude_patterns or None,
            dto.fetch_github_data)
        if PROMETHEUS:
            findings_total.set(report["phases"]["analysis"]["findings"])
        return report
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {exc}")


@app.post("/api/v1/ingest/update/{repo_id}")
async def ingest_update(repo_id: str, user: dict = Depends(require_auth)):
    """§4.1.5 incremental update."""
    phase1, _ = get_pipelines()
    try:
        return await asyncio.to_thread(phase1.incremental_update, repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Update failed: {exc}")


@app.get("/api/v1/findings/{repo_id}")
async def findings(repo_id: str, severity: str | None = None,
                   finding_type: str | None = None, limit: int = 200):
    """§10.3.2 GET /api/v1/findings/{repo_id}."""
    phase1, _ = get_pipelines()
    rows = phase1.db.get_findings(repo_id, severity=severity,
                                  finding_type=finding_type, limit=limit)
    return {"findings": rows, "total": len(rows)}


@app.get("/api/v1/findings/{repo_id}/recommendations")
async def recommendations(repo_id: str, limit: int = 10):
    """§11.2 explainable recommendations for top findings."""
    phase1, phase2 = get_pipelines()
    phase1.restore_knowledge(repo_id)
    rows = phase1.db.get_findings(repo_id, limit=limit)
    if not rows:
        return {"recommendations": []}

    from app.models.finding import Finding

    graph = phase1.graph_store
    recs = []
    for r in rows[:limit]:
        finding = Finding(
            id=r["id"], tool=r["tool"], rule_id=r["rule_id"] or "",
            message=r["message"], severity=r["severity"] or "info",
            finding_type=r["finding_type"], file_path=r["file_path"],
            line_start=r["line_start"] or 0, line_end=r["line_end"] or 0)
        symbol_name = extract_symbol(r["message"]) or ""
        impact = graph.get_impact_analysis(repo_id, symbol_name) if symbol_name else []
        impact_list = [i.get("node_id", "") for i in impact][:10] or [r["file_path"]]
        recs.append({
            "finding_id": finding.id, "severity": finding.severity,
            "rule": finding.rule_id, "file": finding.file_path,
            "lines": f"{finding.line_start}-{finding.line_end}",
            "impact": impact_list, "priority": compute_priority(finding, impact_list),
            "message": finding.message,
        })
    recs.sort(key=lambda x: -x["priority"])
    return {"recommendations": recs, "total": len(recs)}


@app.get("/api/v1/findings/{repo_id}/predictions/explain")
async def explain_quality_prediction(repo_id: str, file_path: str):
    """XAI (§Task 4 & Etch Detail 4): explain defect/smell prediction using TreeSHAP."""
    phase1, _ = get_pipelines()
    phase1.restore_knowledge(repo_id)

    from app.core.analysis.quality_predictor import QualityPredictor
    from app.models.repository import FileRecord
    from app.utils.time import utc_now

    try:
        pred_rows = phase1.db.get_quality_predictions(repo_id, file_path)
    except Exception:
        pred_rows = []
    features = {}
    for p in pred_rows:
        if p.get("file_path") == file_path:
            try:
                import json as _json
                features = _json.loads(p.get("features") or "{}")
            except Exception:
                pass
            break

    if not features:
        repo_path = phase1._repo_paths.get(repo_id)
        if repo_path and (Path(repo_path) / file_path).is_file():
            content = (Path(repo_path) / file_path).read_text(encoding="utf-8", errors="ignore")
            file_rec = FileRecord(path=file_path, language="python", category="code", size_bytes=len(content), content_hash="", last_modified=utc_now())
            features = QualityPredictor.extract_features(file_rec, content, [], [], [])

    predictor = QualityPredictor()
    explanation = predictor.explain_prediction(features)
    return {"file_path": file_path, "repo_id": repo_id, "explanation": explanation}



@app.get("/api/v1/repos")
async def list_repos():
    """Indexed repositories with stats."""
    phase1, _ = get_pipelines()
    repos = phase1.db.list_repositories()
    for repo in repos:
        repo.pop("stats", None)
    return {"repositories": repos, "total": len(repos)}


@app.get("/api/v1/repos/{repo_id}")
async def repo_detail(repo_id: str):
    phase1, _ = get_pipelines()
    repo = phase1.db.get_repository(repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    stats = {}
    import json as _json

    try:
        stats = _json.loads(repo.get("stats") or "{}")
    except _json.JSONDecodeError:
        pass
    findings_rows = phase1.db.get_findings(repo_id, limit=1000)
    by_severity: dict[str, int] = {}
    for f in findings_rows:
        by_severity[f["severity"] or "info"] = by_severity.get(f["severity"] or "info", 0) + 1
    
    # Backward compatibility enrichment for existing indexed repos
    if "ingestion" not in stats and "files" in stats:
        stats["ingestion"] = {"files": stats.get("files", 0)}
    if "findings" not in stats:
        stats["findings"] = len(findings_rows)

    qstats = phase1.db.query_stats()
    return {**repo, "stats": stats, "findings_by_severity": by_severity,
            "queries": qstats}


@app.get("/api/v1/architecture/{repo_id}")
async def architecture(repo_id: str):
    """MCP tool get_architecture (§10.3.1) exposed over REST."""
    phase1, _ = get_pipelines()
    phase1.restore_knowledge(repo_id)
    return phase1.graph_store.get_architecture_summary(repo_id)


@app.get("/api/v1/impact/{repo_id}")
async def impact(repo_id: str, symbol: str):
    """Impact analysis for a symbol (§7.3 impact query)."""
    phase1, _ = get_pipelines()
    phase1.restore_knowledge(repo_id)
    results = phase1.graph_store.get_impact_analysis(repo_id, symbol)
    return {"symbol": symbol, "impacted": results, "total": len(results)}


@app.get("/api/v1/health")
async def health():
    """§24.5 — dependency-validating health check."""
    components: dict[str, str] = {}
    overall = "healthy"
    phase1, phase2 = get_pipelines()

    try:
        phase1.db.conn.execute("SELECT 1")
        components["database"] = "ok"
    except Exception as exc:
        components["database"] = f"error: {exc}"
        overall = "degraded"

    repos = phase1.db.list_repositories()
    components["knowledge_base"] = "ok" if repos else "empty (no repositories ingested)"

    components["vector_index"] = phase1.vector_indexer.stats()["backend"]
    components["graph_store"] = type(phase1.graph_store).__name__
    components["llm_provider"] = phase2.llm.provider_name
    components["nli_model"] = phase2.nli.model_name
    try:
        m = phase2.nli._load()
        components["nli_status"] = "ok" if m else "degraded"
    except Exception as exc:
        components["nli_status"] = f"degraded: {exc}"
        overall = "degraded"
    if phase2.nli._model is False:
        overall = "degraded"

    # Return 200 so monitoring / health probes succeed while reporting component health
    status_code = 200

    return JSONResponse(
        status_code=status_code,
        content={
            "status": overall,
            "version": app.version,
            "components": components,
            "repositories": len(repos),
            "timestamp": utc_now_iso(),
        },
    )


if PROMETHEUS:

    @app.get("/metrics")
    async def metrics():
        return generate_latest(), {"content-type": CONTENT_TYPE_LATEST}


# ----------------------------------------------------------------------- dashboard UI
@app.get("/")
async def dashboard():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"name": "RepoIntel AI API", "docs": "/docs", "health": "/api/v1/health"}


if (FRONTEND_DIR / "index.html").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
