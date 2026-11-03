# Changelog — RepoIntel AI

All notable changes and architectural versions are documented here.

## [1.2.0] — Codebase Health & First-Week Platform - 2026-11-02

### Added
- **Codebase Health & Gap Analysis (`gap_analyzer.py`):** AST-driven detection of untested modules (`test_gap`), missing/incomplete docstrings (`doc_gap`), unfinished stub features (`incomplete_feature` via `NotImplementedError`, pass-only bodies, TODO/FIXME markers), and unguarded file I/O operations (`error_handling_gap`).
- **Architectural Intent Layer (`intent_checker.py`):** Declarative validation of `.repointel/architecture.yml` against observed call and import edges in `repointel.db`, detecting forbidden cross-layer dependencies.
- **Goal Decomposition & L3 Agentic Planner (`planner.py`):** Multi-hop goal decomposition with fast-path bypass for direct queries; decomposes complex architectural missions into structured subtasks.
- **REST Endpoints & MCP Tools:** Added `GET /api/v1/repos/{repo_id}/gaps`, `GET /api/v1/repos/{repo_id}/health-score`, `GET /api/v1/repos/{repo_id}/intent`, plus MCP stdio tools `list_gaps` and `get_health_score`.
- **Health Tab & KPI Console (`index.html`):** 5-column dashboard bento grid with codebase health letter grade (A-F) and score (0-100), dedicated Codebase Health tab (`06 // CODEBASE HEALTH`) with severity ASCII bars, gap filters, and architecture rule violation inspector.
- **Safety Feature Flags:** All new capabilities gated behind environment flags (`ENABLE_GAP_ANALYSIS=0`, `ENABLE_INTENT_LAYER=0`, `ENABLE_GOAL_DECOMPOSITION=0`) ensuring 100% backward compatibility for the base demo flow.

## [0.9.0-MVP] - 2026-10-30 (Review 1 / Pitch Stage - Current)

### Added
- **Core Intelligence:** Tree-sitter AST parsing, BM25 + Dense + Graph hybrid retrieval, RRF fusion, FlashRank ONNX reranking (`ms-marco-MiniLM-L-12-v2`), and LLM synthesis with citations.
- **Persistent Code Graph:** Unified `SQLiteGraphStore` (`repointel.db`) with `WITH RECURSIVE` CTEs for sub-millisecond upstream blast radius & call tree analysis.
- **AST Merkle Fingerprinting:** Structural subtree hashing in `incremental.py` to prevent redundant re-parsing and re-embedding on comments/formatting edits.
- **Dual-Engine SAST:** `SASTOrchestrator` combining sub-15ms `ast-grep` syntactic linting with local offline `Semgrep` taint tracking (`config/semgrep/security-audit.yaml`).
- **Explainable AI (XAI):** `QualityPredictor.explain_prediction` powered by TreeSHAP for defect attribution, exposed via `GET /api/v1/findings/{repo}/predictions/explain`.
- **Observability:** OpenTelemetry native spans and LiteLLM OTel callbacks.
- **Tactile UI Console:** Industrial dot-matrix interface featuring live SSE stepper (`retrieval` -> `nli` -> `answer`), Atom One Dark code highlighting, and interactive force-directed `vis-network` knowledge and blast radius graphs.
- **ICE Bridge:** One-directional adapter (`backend/app/core/bridge/ice_adapter.py`) mapping RepoIntel findings to ICE evidence payloads.

### Deferred to v1.0.0 (Final Showcase on High-End GPU/RAM Workstation)
- **Neo4j & PostgreSQL:** Adapters are fully written and tested in code; disabled by default (`GRAPH_STORE=sqlite`, `RELATIONAL_STORE=sqlite`) to eliminate multi-service Docker friction during evaluation.
- **FAISS HNSW Vector Store:** NumPy exact search is used for mathematical precision on local repositories (<10k chunks).
- **CodeQL & SonarQube:** Enterprise runners coded; built-in AST + Semgrep provide zero-CLI instant local analysis.
- **React Dashboard:** Zero-dependency Vanilla JS ensures zero build steps, zero node_modules, and sub-10ms UI load latency.
- **Full Cloud SWE-bench:** 8-question local evaluation benchmark is active for rapid local iteration.
