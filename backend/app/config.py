"""RepoIntel AI — Configuration (§5.1 / §16.4 of the implementation documentation).

All settings are read from environment variables with safe local-development
defaults so the system runs with zero external services (embedded SQLite,
networkx graph, in-memory cache).  Production values switch the adapters to
Neo4j / PostgreSQL / Redis / Pinecone exactly as documented.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_env_file() -> None:
    for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
        env_file = base / ".env"
        if env_file.is_file():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip()
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass
            break

_load_env_file()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    """Central typed configuration object (documented §5.1)."""

    # ---- GitHub -------------------------------------------------------
    github_token: str = field(default_factory=lambda: _env("GITHUB_TOKEN"))
    github_api_url: str = field(default_factory=lambda: _env("GITHUB_API_URL", "https://api.github.com"))

    # ---- Storage backends ---------------------------------------------
    # graph_store: "networkx" (embedded, default) | "neo4j" (production)
    graph_store: str = field(default_factory=lambda: _env("GRAPH_STORE", "networkx"))
    # relational_store: "sqlite" (embedded, default) | "postgres" (production)
    relational_store: str = field(default_factory=lambda: _env("RELATIONAL_STORE", "sqlite"))
    # cache: "memory" (default) | "redis" (production)
    cache_backend: str = field(default_factory=lambda: _env("CACHE_BACKEND", "memory"))

    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "repointel"))

    postgres_host: str = field(default_factory=lambda: _env("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: _env_int("POSTGRES_PORT", 5432))
    postgres_db: str = field(default_factory=lambda: _env("POSTGRES_DB", "repointel"))
    postgres_user: str = field(default_factory=lambda: _env("POSTGRES_USER", "repointel"))
    postgres_password: str = field(default_factory=lambda: _env("POSTGRES_PASSWORD", "repointel"))

    redis_host: str = field(default_factory=lambda: _env("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: _env_int("REDIS_PORT", 6379))

    # FIX-006d: mongo_uri / mongo_db removed — no code touched MongoDB; a
    # provisioned-but-unused database is capability theatre. The MongoDB
    # service was also removed from infra/docker-compose.yml.

    # ---- Vector store ---------------------------------------------------
    vector_store: str = field(default_factory=lambda: _env("VECTOR_STORE", "numpy"))  # numpy | faiss | pinecone
    # empty → resolved to <data_dir>/faiss (stable, CWD-independent)
    faiss_index_path: str = field(default_factory=lambda: _env("FAISS_INDEX_PATH", ""))
    pinecone_api_key: str = field(default_factory=lambda: _env("PINECONE_API_KEY"))
    pinecone_env: str = field(default_factory=lambda: _env("PINECONE_ENV"))
    # FIX-026: memory budget (MB) for adaptive FAISS index selection
    vector_memory_budget_mb: int = field(default_factory=lambda: _env_int("VECTOR_MEMORY_BUDGET_MB", 4096))

    # ---- LLM ------------------------------------------------------------
    # provider: "openai" (default), "litellm", "anthropic", "google", "local", "zai"
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 4096))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    google_api_key: str = field(default_factory=lambda: _env("GOOGLE_API_KEY"))

    # ---- Embeddings -------------------------------------------------------
    # FIX-027 + 2026 correction: UniXcoder is the default for BOTH channels
    # (768-dim, AST + data-flow aware). MiniLM remains a documented lighter
    # fallback: EMBEDDING_MODEL_CODE=...all-MiniLM-L6-v2, EMBEDDING_DIMENSION=384.
    embedding_model_code: str = field(default_factory=lambda: _env("EMBEDDING_MODEL_CODE", "microsoft/unixcoder-base"))
    embedding_model_doc: str = field(default_factory=lambda: _env("EMBEDDING_MODEL_DOC", "microsoft/unixcoder-base"))
    embedding_dimension: int = field(default_factory=lambda: _env_int("EMBEDDING_DIMENSION", 768))

    # ---- Reranker / NLI ----------------------------------------------------
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2"))
    nli_model: str = field(default_factory=lambda: _env("NLI_MODEL", "dleemiller/EttinX-nli-xs"))

    # ---- Static analysis ----------------------------------------------------
    # analyzers enabled: built-in analyzer always available ("repointel" tool).
    # sonarqube / codeql activated when their servers/CLIs are reachable.
    sonar_url: str = field(default_factory=lambda: _env("SONAR_URL", "http://localhost:9000"))
    sonar_token: str = field(default_factory=lambda: _env("SONAR_TOKEN"))
    codeql_path: str = field(default_factory=lambda: _env("CODEQL_PATH", "codeql"))
    enable_sonar: bool = field(default_factory=lambda: _env("ENABLE_SONAR", "0") == "1")
    enable_codeql: bool = field(default_factory=lambda: _env("ENABLE_CODEQL", "0") == "1")

    # ---- Agentic exploration budgets (§8.4.2) -------------------------------
    max_iterations: int = field(default_factory=lambda: _env_int("MAX_ITERATIONS", 5))
    max_total_tokens: int = field(default_factory=lambda: _env_int("MAX_TOTAL_TOKENS", 10000))
    max_time_seconds: float = field(default_factory=lambda: _env_float("MAX_TIME_SECONDS", 60.0))
    min_evidence_count: int = field(default_factory=lambda: _env_int("MIN_EVIDENCE_COUNT", 3))
    # FIX-037: LLM-based sufficiency check (slower, more accurate)
    use_llm_sufficiency: bool = field(default_factory=lambda: _env("USE_LLM_SUFFICIENCY", "0") == "1")

    # ---- Context budgeting (§9.4) ---------------------------------------------
    llm_context_window: int = field(default_factory=lambda: _env_int("LLM_CONTEXT_WINDOW", 32768))
    reserved_for_response: int = field(default_factory=lambda: _env_int("RESERVED_FOR_RESPONSE", 4096))
    evidence_token_budget: int = field(default_factory=lambda: _env_int("EVIDENCE_TOKEN_BUDGET", 8000))

    # ---- Logging / performance -------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: _env("LOG_FORMAT", "json"))
    max_workers: int = field(default_factory=lambda: _env_int("MAX_WORKERS", 4))
    batch_size: int = field(default_factory=lambda: _env_int("BATCH_SIZE", 100))
    cache_ttl: int = field(default_factory=lambda: _env_int("CACHE_TTL", 3600))

    # ---- Security (§24) ------------------------------------------------------------
    jwt_secret: str = field(default_factory=lambda: _env("JWT_SECRET", "dev-secret-change-me"))
    api_key_salt: str = field(default_factory=lambda: _env("API_KEY_SALT", "dev-salt"))
    rate_limit_capacity: int = field(default_factory=lambda: _env_int("RATE_LIMIT_CAPACITY", 100))
    rate_limit_refill: float = field(default_factory=lambda: _env_float("RATE_LIMIT_REFILL", 2.0))
    # FIX-030: when 1, mutating + query endpoints require a Bearer JWT
    require_auth: bool = field(default_factory=lambda: _env("REQUIRE_AUTH", "0") == "1")
    # FIX-036: HMAC secret for GitHub push webhooks
    github_webhook_secret: str = field(default_factory=lambda: _env("GITHUB_WEBHOOK_SECRET"))
    # FIX-016/039: bound on parallel z-ai subprocesses
    zai_max_concurrent: int = field(default_factory=lambda: _env_int("ZAI_MAX_CONCURRENT", 4))

    # ---- Paths -------------------------------------------------------------------
    # Stable project data dir: <repo root>/data (independent of CWD)
    data_dir: Path = field(default_factory=lambda: (
        Path(os.environ["REPOINTEL_DATA_DIR"]).resolve()
        if os.environ.get("REPOINTEL_DATA_DIR")
        else Path(__file__).resolve().parents[2].joinpath("data")
    ))
    repos_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2].joinpath("data", "repos"))
    models_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2].joinpath("backend", "models", "trained"))

    # ---- Prompt injection limits (§23.9) -----------------------------------------------
    max_query_length: int = field(default_factory=lambda: _env_int("MAX_QUERY_LENGTH", 1000))
    max_context_length: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_LENGTH", 50000))

    @property
    def faiss_dir(self) -> Path:
        """Stable vector-index directory: override or <data_dir>/faiss."""
        return Path(self.faiss_index_path).resolve() if self.faiss_index_path else self.data_dir / "faiss"


_settings: Settings | None = None
_settings_lock = __import__("threading").Lock()


def get_settings() -> Settings:
    """Singleton settings accessor (FIX-048: double-checked locking)."""
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                s = Settings()
                for p in (s.data_dir, s.repos_dir, s.models_dir):
                    p.mkdir(parents=True, exist_ok=True)
                _settings = s
    return _settings
