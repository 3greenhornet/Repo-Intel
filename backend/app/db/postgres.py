"""Production PostgreSQL DDL (§15.1) — used when RELATIONAL_STORE=postgres.

Requires a reachable PostgreSQL server (docker-compose service `postgres`).
asyncpg pool parameters follow §24.3 G12 fix.
"""
from __future__ import annotations

POSTGRES_DDL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgvector;

CREATE TABLE IF NOT EXISTS repositories (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  url VARCHAR(500) NOT NULL UNIQUE,
  name VARCHAR(200) NOT NULL,
  branch VARCHAR(100) DEFAULT 'main',
  last_indexed_commit VARCHAR(40),
  last_indexed_at TIMESTAMPTZ,
  status VARCHAR(20) DEFAULT 'pending',
  config JSONB DEFAULT '{}',
  stats JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_repositories_status ON repositories(status);

CREATE TABLE IF NOT EXISTS files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  repo_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
  path VARCHAR(500) NOT NULL,
  language VARCHAR(50),
  category VARCHAR(50),
  size_bytes INTEGER,
  content_hash VARCHAR(64),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(repo_id, path)
);
CREATE INDEX IF NOT EXISTS idx_files_repo ON files(repo_id);

CREATE TABLE IF NOT EXISTS findings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  repo_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
  tool VARCHAR(50) NOT NULL,
  rule_id VARCHAR(100),
  finding_type VARCHAR(50) NOT NULL,
  severity VARCHAR(20),
  message TEXT NOT NULL,
  file_path VARCHAR(500) NOT NULL,
  line_start INTEGER,
  line_end INTEGER,
  symbol_id VARCHAR(200),
  confidence FLOAT,
  status VARCHAR(20) DEFAULT 'open',
  commit_hash VARCHAR(40),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_findings_repo ON findings(repo_id);
CREATE INDEX IF NOT EXISTS idx_findings_file ON findings(file_path);
CREATE INDEX IF NOT EXISTS idx_findings_severity ON findings(severity);
CREATE INDEX IF NOT EXISTS idx_findings_type ON findings(finding_type);

CREATE TABLE IF NOT EXISTS quality_predictions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  repo_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
  file_path VARCHAR(500) NOT NULL,
  symbol_id VARCHAR(200),
  prediction_type VARCHAR(50) NOT NULL,
  probability FLOAT,
  confidence FLOAT,
  features JSONB DEFAULT '{}',
  model_version VARCHAR(50),
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_quality_repo ON quality_predictions(repo_id);

CREATE TABLE IF NOT EXISTS commits (
  hash VARCHAR(40),
  repo_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
  author VARCHAR(200),
  author_email VARCHAR(200),
  timestamp TIMESTAMPTZ,
  message TEXT,
  additions INTEGER,
  deletions INTEGER,
  files_changed TEXT[],
  PRIMARY KEY (hash, repo_id)
);
CREATE INDEX IF NOT EXISTS idx_commits_repo ON commits(repo_id);
CREATE INDEX IF NOT EXISTS idx_commits_timestamp ON commits(timestamp);

CREATE TABLE IF NOT EXISTS issues (
  number INTEGER,
  repo_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
  title TEXT,
  body TEXT,
  labels TEXT[],
  state VARCHAR(20),
  created_at TIMESTAMPTZ,
  closed_at TIMESTAMPTZ,
  PRIMARY KEY (number, repo_id)
);

CREATE TABLE IF NOT EXISTS pull_requests (
  number INTEGER,
  repo_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
  title TEXT,
  body TEXT,
  reviewers TEXT[],
  state VARCHAR(20),
  merged_at TIMESTAMPTZ,
  files_changed TEXT[],
  PRIMARY KEY (number, repo_id)
);

CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  repo_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
  file_path VARCHAR(500),
  doc_type VARCHAR(50),
  content TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS investigation_records (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  repo_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
  query TEXT NOT NULL,
  finding TEXT,
  hypothesis TEXT,
  recommendation TEXT,
  evidence_ids UUID[],
  answer TEXT,
  status VARCHAR(20) DEFAULT 'open',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_investigation_repo ON investigation_records(repo_id);

CREATE TABLE IF NOT EXISTS query_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  repo_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
  user_id VARCHAR(100),
  query TEXT NOT NULL,
  answer TEXT,
  confidence FLOAT,
  grounded BOOLEAN,
  iterations INTEGER,
  elapsed_seconds FLOAT,
  input_tokens INTEGER,
  output_tokens INTEGER,
  cost_usd FLOAT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_query_logs_repo ON query_logs(repo_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_created ON query_logs(created_at);
"""


async def create_postgres_pool():
    """§24.3 G12: asyncpg pool with documented sizing/timeouts."""
    import asyncpg

    from app.config import get_settings

    settings = get_settings()
    return await asyncpg.create_pool(
        host=settings.postgres_host, port=settings.postgres_port,
        database=settings.postgres_db, user=settings.postgres_user,
        password=settings.postgres_password,
        min_size=5, max_size=20, max_queries=50000,
        max_inactive_connection_lifetime=300, command_timeout=60,
    )


def _is_connection_error(exc: Exception) -> bool:
    """FIX-006a: classify driver-level unavailability vs. code bugs."""
    name = type(exc).__name__
    return name in ("OperationalError", "InterfaceError", "ConnectionError",
                    "ConnectionRefusedError", "ConnectionResetError") or isinstance(
        exc, (ConnectionError, OSError))


def get_relational_store():
    """Factory: SQLite (default) or PostgreSQL (production).

    FIX-006a: only genuine unavailability (missing driver, unreachable
    server) triggers the SQLite fallback. A bug inside PostgresStore (e.g. a
    KeyError) propagates instead of silently downgrading the deployment.
    """
    from app.config import get_settings
    from app.utils.logger import get_logger

    settings = get_settings()
    if settings.relational_store == "postgres":
        try:
            from app.db.postgres_store import PostgresStore

            return PostgresStore()
        except ImportError as exc:
            get_logger(__name__).warning(
                f"PostgreSQL driver not installed ({exc}) — using SQLite. "
                "Install psycopg2-binary to enable RELATIONAL_STORE=postgres.")
        except Exception as exc:
            if _is_connection_error(exc):
                raise RuntimeError(
                    f"RELATIONAL_STORE=postgres but the server is unreachable "
                    f"({exc}). Fix connectivity or set RELATIONAL_STORE=sqlite.") from exc
            raise
    from app.db.sqlite_store import SQLiteDatabase

    return SQLiteDatabase()
