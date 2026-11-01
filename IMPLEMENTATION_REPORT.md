# RepoIntel AI — Implementation Report

*What was built, what was improvised, what couldn't run here, and what I'd
do next. This is the honest audit the spec itself models in Part 4 (§22).*

## 1. What was built (and verified by actually running it)

### Phase 1 — Build & Maintain Repository Knowledge (§4)
- **Step 1 Ingestion** (§4.1): GitHub clone client (token-aware, branch
  fallback, 3× retry per §4.1.7), file walker with the documented
  classification table (code/config/docs/tests/generated/binary/vendored/
  data), git log parsing with per-commit numstat, GitHub issues/PR REST
  fetch with rate-limit backoff, local fixture fallback, and the §4.1.5
  incremental algorithm (git diff → A/M/D/R delta → targeted reprocess).
  *Verified: 11 files, 7 commits, 4 issues, 2 PRs ingested in ~0.1s.*
- **Step 2 Analysis** (§4.2): real Tree-sitter parsing (python/js/ts),
  symbol + reference extraction with **function-level, cross-file CALLS
  edges** (`handle_login → login → validate_credentials → hash_password`
  verified), built-in static analyzer (SQLi string-concat, eval/exec,
  hardcoded secrets, weak hash, bare except, mutable default args,
  complexity, long functions, deep nesting, god file, SATD TODO/FIXME),
  SARIF parser with the §23.4 edge cases, SonarQube runner with §23.3
  pagination, CodeQL runner, trained sklearn quality models (defect
  0.85 acc / smell 1.00 acc / TD R² 0.93), history intelligence
  (hotspots, authors, coupling, finding↔commit correlation).
  *Verified: 12 findings (2 critical), 27 symbols, 58 references, 38 edges.*
- **Step 3 Knowledge base** (§4.3): graph store implementing the documented
  node/edge schema (NetworkX default, Neo4j backend included), vector index
  with §4.3.2 chunking (code by symbol, docs by section, 500-token cap),
  real MiniLM embeddings (384-dim), findings + metadata stores with the
  §15.1 schema (SQLite default, PostgreSQL DDL included), persistence +
  cross-process restore. *Verified: 63 nodes, 69 edges, 45 chunks, 12.5s
  total for the whole phase.*

### Phase 2 — Query & Reason (§7–§12)
- **Step 4 Hybrid retrieval** (§7): BM25 with the documented code-aware
  tokenizer, dense retrieval, graph retrieval (callers/callees/dependencies/
  impact/symbol lookup with the §7.3.4 intent parser), RRF fusion
  (§23.6 unequal-length safe), real cross-encoder reranking. *Verified:
  parallel retrieval + fusion + rerank.*
- **Step 5 Agentic exploration** (§8): sufficiency check (multi-hop aware),
  gap detection (term/relationship/context gaps), query reformulation,
  iteration loop with the documented budgets (5 iterations / 10k tokens /
  time limit), dedup. *Verified: iterations 1.25 avg on benchmark.*
- **Step 6 Evidence orchestration** (§9): token-budget selection, real NLI
  scoring with the §23.7 optimization (contradictions actually detected and
  surfaced), provenance attachment (file/lines/symbol/commit/author/rule),
  context budgeting, §9.5.2 context assembly format.
- **Step 7 Reasoning** (§10): multi-provider LLM client (z-ai bridge —
  works keyless here; OpenAI/Anthropic per §10.1.3), the documented §10.2
  system prompt with the §23.9 injection defense (4-layer sanitizer,
  XML-tagged prompt boundaries), SSE streaming endpoint (§23.8).
- **Step 8 Output** (§11): Q&A with multi-form citation parsing
  (`[Evidence 3]`, `[Evidence 1, 2]`, `[Evidence 5-6]`), recommendations
  with §11.2 priority computation, evidence-linked explanations, LLM
  follow-up generation. *Verified: 9-file citations with commit hashes on
  security questions; "Who calls login?" → correct answer citing
  api.py:10-16.*

### Phase 3 — Evaluation & Operations (§14–§19)
- **Step 9 Evaluation** (§14): all 4 documented baselines (A dense-only,
  B graph-only, C hybrid, D agentic full) sharing one interface, the §14.3
  metric implementations (Recall@K, MRR, NDCG, citation correctness,
  groundedness, localization, latency percentiles, token/cost), benchmark
  harness with hooks for SWE-bench-lite/RepoQA loaders. *Verified: real
  4×8 evaluation run, results table in README.*
- **API** (§10.3.2): POST /query, POST /query/stream, POST /ingest,
  POST /ingest/update, GET /findings (+filters), GET /recommendations,
  GET /repos, GET /architecture, GET /impact, GET /health (§24.5
  dependency-validating), GET /metrics (Prometheus).
- **Hardening** (§24): token-bucket rate limiting (Redis or in-memory),
  JWT (HS256 now, RS256 path documented), §24.7 log redaction (tested),
  graceful degradation everywhere (any component failing is logged and
  skipped per the §7.4.4-style tables).
- **Dashboard**: self-contained web UI (Q&A with live streaming, findings
  browser with severity filters, prioritized recommendations, architecture
  summary, repo ingestion) — no external JS dependencies.
- **MCP server** (§10.3.1): all 5 documented tools over stdio JSON-RPC.
  *Verified roundtrip: get_architecture returns live graph stats.*
- **Infra** (§16, §18–§19): docker-compose (full production stack),
  Dockerfile, k8s deployment + PVC, prometheus.yml, alerts.yml, GitHub
  Actions CI — ready to use once you have a Docker host.
- **Tests** (§17): 18 unit (all passing), 6 integration (all passing),
  1 e2e (full-pipeline LLM query).

## 2. What I improvised / filled in (gap-filling, per your request)

1. **Built-in static analyzer ("repointel" tool)** — the spec relies on
   SonarQube + CodeQL servers that can't run in this sandbox. I implemented
   an AST/regex analyzer with equivalent finding categories, severities and
   SARIF-compatible output, and wired the documented runners to activate
   automatically when their servers/CLIs become reachable. The pipeline
   always produces real findings either way.
2. **z-ai LLM bridge** — the spec assumes an OpenAI/Anthropic key. This
   environment has the z-ai CLI (glm-4-plus), so I added a provider that
   bridges to it with 429-aware backoff. OpenAI/Anthropic providers are
   implemented and activate via config — the documented path is intact.
3. **Embedded store adapters** — NetworkX graph, SQLite relational,
   in-memory cache, numpy vectors as defaults, each with the documented
   production backend behind the same interface. The spec's own design
   principle ("components must be swappable") made this clean.
4. **Function-level call edges** — the spec's §4.2.2 shows calls attributed
   loosely; I made references carry their enclosing function so impact
   analysis (§7.3 "what breaks if I change this") actually works
   transitively across files.
5. **Citation parsing for all LLM citation forms** — the spec only parses
   `[Evidence N]`; real models emit `[Evidence 1, 2]` and ranges, which I
   now parse.
6. **Deterministic model training script** — §4.2.4 says "models are
   pre-trained and loaded from disk" without providing training; I built
   `scripts/train_quality_models.py` (synthetic distribution following the
   documented feature semantics) so the models exist, persist, and load
   exactly as documented (runtime never trains).
7. **Local benchmark** — external benchmarks (SWE-bench et al.) are
   multi-GB downloads; I shipped loaders plus a labeled 8-question
   benchmark so §14 evaluation runs for real immediately.
8. **Persistence + restore** — cross-process restore of vector/graph
   indexes so the API server, CLI and tests share one knowledge base.

## 3. What I could NOT do here (and what you need to do)

| Blocker | Why | What unblocks it |
|---|---|---|
| **No Docker** in this sandbox | Neo4j/Postgres/Redis/Mongo/SonarQube/Prometheus/Grafana can't run | Run `docker compose -f infra/docker-compose.yml up` on any Docker host; the code paths switch via env vars (`GRAPH_STORE=neo4j`, etc.) |
| **No SonarQube server / CodeQL CLI** | Heavy services; CodeQL needs license acceptance (H26/H27) | Set `ENABLE_SONAR=1` (+SONAR_URL/TOKEN) or `ENABLE_CODEQL=1` — runners are implemented |
| **No OpenAI/Anthropic key** (H1) | You must supply one for those providers | Set `LLM_PROVIDER=openai` + `OPENAI_API_KEY` (or anthropic) — providers implemented; the z-ai bridge is the local default |
| **No real GitHub API calls in the demo** (H11) | The demo repo is a local fixture to stay deterministic | `GITHUB_TOKEN` set + any GitHub URL works through `scripts/build_knowledge.py <url>` — client is implemented (issues, PRs, clone) |
| **SWE-bench / RepoQA / BigVul / Juliet full runs** (H18) | Multi-GB downloads + external harnesses | Loaders are implemented (`datasets` package); point the eval at them on a networked host |
| **RAGAS** (§14.3.2) | Requires an OpenAI-keyed judge LLM | `pip install ragas` + run with an LLM provider; the heuristic groundedness metric stands in meanwhile |
| **Celery worker fleet** | Needs Redis + broker | Worker loop implemented (`app/workers/ingestion_worker.py`) with Redis queue; docker-compose includes it |
| **VS Code extension / Slack bot** (§27.2–27.3) | Need marketplace/workspace setup to be meaningful | Straightforward against the documented REST API (endpoints all live) — not built to keep scope on the core system |
| **MCP registration** (H10/H15) | Requires your Claude Code/Pi config | `cd backend && python -m app.core.mcp.server` — JSON-RPC stdio works, verified |
| **JS/TS parse coverage in the demo** | Fixture repo is Python (spec's own L1: MVP is Python-first) | JS/TS tree-sitter grammars are installed and the parser handles them — ingest a JS/TS repo and it works |

## 4. Honest limitations of this implementation

- **Embeddings are MiniLM for both code and docs** (spec wants CodeBERT for
  code). Swap `EMBEDDING_MODEL_CODE` in config — the engine supports it.
- **NLI model is nli-deberta-v3-base** (same labels, lighter than
  roberta-large-MNLI). Config switch `NLI_MODEL`.
- **Quality-prediction models are trained on synthetic distributions**, not
  PROMISE/DeleSmell/SATD datasets (§14.2.5 downloads). The feature
  extraction is the documented one; the datasets would improve the labels.
- **Hotspot detection threshold** (churn > 3) found 0 hotspots on the
  7-commit demo (auth.py churns 3×). It's the documented default; lower
  `churn_threshold` for small repos.
- **Groundedness metric** is a lexical-overlap approximation of RAGAS
  faithfulness (needs LLM-as-judge for the real thing).
- **Dashboard is a dependency-free single-file app**, not the documented
  React 18 project — deliberate scope call so it always works offline;
  the REST API it consumes is exactly the documented one a React app would.
- **Impact analysis explores CONTAINS + CALLS paths**, so container nodes
  (files) are filtered from results; commits that only *modified* a file
  don't count as dependents (correct semantics, noted for transparency).
- **One LLM per request** — no batch caching of identical contexts, and
  the z-ai bridge re-invokes per call (429s observed under load; backoff
  handles it but throughput is provider-limited).

## 5. How well the documented design held up

The spec was genuinely implementable — the component contracts (§3.2),
data models (§4.1.6) and error tables mapped almost 1:1 to code. Three
places where the documentation was thin and I had to decide:
1. §4.2.2 doesn't define call-edge attribution (module vs enclosing
   function) — I chose enclosing function (see above).
2. §14.1 baselines don't specify how retrieval metrics are compared for
   variant D (citations vs selected evidence) — I made all baselines
   measure selected evidence files.
3. §9.4 context budgeting assumed an 8k window; modern models have 32k+ —
   I made it configurable and defaulted larger.

## 6. What I would do next (in order)

1. **Ingest a real GitHub repo** (e.g. a popular Python project) with a
   real `GITHUB_TOKEN` — the whole pipeline is ready; this validates
   scale (10k files, §4.1.8 performance targets).
2. **Turn on the production stack** via docker-compose; run the §14
   external benchmarks (SWE-bench-lite) for comparable published numbers.
3. **Swap in CodeBERT** code embeddings and roberta-large-MNLI for exact
   spec fidelity.
4. **RAGAS** with an LLM judge for real faithfulness/answer-relevancy.
5. **GitHub webhooks** (§27.1 handler is sketched in the docs; the
   incremental pipeline it feeds is implemented and tested).
6. **React 18 dashboard** replacing the static one, against the same API.
7. **FAISS HNSW** index for >10k-chunk repos (§23.5 selection logic is
   documented; numpy exact search is optimal below that).

---

# Fix Worklist Execution Report (this session)

## Outcome

All 54 fixes from the 150-page implementation fix document were addressed,
plus the four 2026 modernization corrections. 62/62 tests pass (59
unit/integration + 3 e2e against the live z-ai LLM); the F-category lint
gate (real bugs, E722) is clean; a clean-state `make demo` and a
four-baseline evaluation were re-verified after the changes.

## Fixed-in-transit (found while applying the worklist)

1. **FIX-001 contains its own bug**: the doc's abstention check
   (`"not enough evidence" in answer`) fails the doc's own test case 2
   ("I don't have enough evidence…"). Replaced with an abstention-phrase
   regex.
2. **Latency corruption**: `execute_full_pipeline` mixed `time.time()` start
   with `time.monotonic()` end, producing −1.79e9 s latencies for variant D
   in the eval (caught in the post-fix eval run; both ends now monotonic).
3. **`symbol_extractor` module_id scoping** (ruff F821): the module id was
   defined in one method and referenced in another; plus 20 unused imports
   and 3 dead locals swept to zero.

## Partial / could not complete here

- **FIX-021 (hash-locked requirements)**: exact versions are pinned from the
  verified environment, but `--generate-hashes` could not finish within the
  build window (torch dependency tree). `make compile-deps` regenerates the
  hash-locked form in an unrestricted environment; Docker/CI install the
  pinned lock today.
- **Live external services** (unchanged from the original report): Neo4j,
  PostgreSQL, Redis, SonarQube, CodeQL, RAGAS and Docker are implemented and
  CI-wired but not runnable in this sandbox — embedded fallbacks were used
  and verified. The Neo4j Cypher fix (FIX-002) is code-verified but not
  integration-verified against a live server.
- **LLM-judge strictness**: the judge scores answers against the citation
  snippets exposed on the response (160 chars), so its faithfulness numbers
  are deliberately harsh. Feeding it the full evidence set is tracked
  follow-up work.
- **Style lints**: E501 (line length) and several S/B idiom warnings remain
  as tracked, non-blocking debt (reported in CI via `--statistics`).

## Environment notes

- Runtime Python is 3.12 in this sandbox; the doc pinned 3.13-only. The
  version contract was widened to `>=3.12,<3.14` (no 3.13-only features are
  used) so both the documented CI matrix and this runtime are valid.
- Embeddings moved 384-dim MiniLM → 768-dim UniXcoder per the 2026
  correction; a full re-index was performed (`rm -rf data/faiss && make demo`).
