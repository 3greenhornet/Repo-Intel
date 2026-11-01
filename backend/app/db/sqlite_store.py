"""SQLite/PostgreSQL persistence — Stores C & D (§4.3.3, §4.3.4, §15.1).

The documented PostgreSQL schema (§15.1) is translated 1:1 into SQLite for
zero-infrastructure runs; `backend/app/db/postgres_store.py` carries the
production implementation. Table and column names match the documentation so
migration is mechanical.

Concurrency model (FIX-015): per-thread connections + WAL journaling.
Reads run in parallel; writes serialize at the SQLite file level with a
5s busy timeout. `isolation_level=None` enables explicit BEGIN/COMMIT
transactions used by the Phase 1 all-or-nothing write (FIX-013).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.models.finding import Finding, QualityPrediction
from app.models.repository import RepositorySnapshot
from app.utils.logger import get_logger
from app.utils.time import utc_now_iso

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS repositories (
  id TEXT PRIMARY KEY,
  url TEXT NOT NULL UNIQUE,
  name TEXT,
  branch TEXT DEFAULT 'main',
  last_indexed_commit TEXT,
  last_indexed_at TIMESTAMP,
  status TEXT DEFAULT 'pending',
  config TEXT DEFAULT '{}',
  stats TEXT DEFAULT '{}',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS files (
  id TEXT PRIMARY KEY,
  repo_id TEXT REFERENCES repositories(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  language TEXT,
  category TEXT,
  size_bytes INTEGER,
  content_hash TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(repo_id, path)
);
CREATE INDEX IF NOT EXISTS idx_files_repo ON files(repo_id);
CREATE TABLE IF NOT EXISTS findings (
  id TEXT PRIMARY KEY,
  repo_id TEXT REFERENCES repositories(id) ON DELETE CASCADE,
  tool TEXT NOT NULL,
  rule_id TEXT,
  finding_type TEXT NOT NULL,
  severity TEXT,
  message TEXT NOT NULL,
  file_path TEXT NOT NULL,
  line_start INTEGER,
  line_end INTEGER,
  symbol_id TEXT,
  confidence REAL,
  status TEXT DEFAULT 'open',
  commit_hash TEXT,
  metadata TEXT DEFAULT '{}',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_findings_repo ON findings(repo_id);
CREATE INDEX IF NOT EXISTS idx_findings_file ON findings(file_path);
CREATE INDEX IF NOT EXISTS idx_findings_severity ON findings(severity);
CREATE INDEX IF NOT EXISTS idx_findings_type ON findings(finding_type);
CREATE TABLE IF NOT EXISTS finding_correlations (
  id TEXT PRIMARY KEY,
  finding_id TEXT REFERENCES findings(id) ON DELETE CASCADE,
  commit_hash TEXT,
  introduced_at TIMESTAMP,
  fixed_at TIMESTAMP,
  status TEXT
);
CREATE TABLE IF NOT EXISTS quality_predictions (
  id TEXT PRIMARY KEY,
  repo_id TEXT REFERENCES repositories(id) ON DELETE CASCADE,
  file_path TEXT NOT NULL,
  symbol_id TEXT,
  prediction_type TEXT NOT NULL,
  probability REAL,
  confidence REAL,
  features TEXT DEFAULT '{}',
  model_version TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_quality_repo ON quality_predictions(repo_id);
CREATE INDEX IF NOT EXISTS idx_quality_type ON quality_predictions(repo_id, prediction_type);
CREATE TABLE IF NOT EXISTS commits (
  hash TEXT,
  repo_id TEXT REFERENCES repositories(id) ON DELETE CASCADE,
  author TEXT,
  author_email TEXT,
  timestamp TIMESTAMP,
  message TEXT,
  additions INTEGER,
  deletions INTEGER,
  files_changed TEXT,
  PRIMARY KEY (hash, repo_id)
);
CREATE INDEX IF NOT EXISTS idx_commits_repo ON commits(repo_id);
CREATE TABLE IF NOT EXISTS issues (
  number INTEGER,
  repo_id TEXT REFERENCES repositories(id) ON DELETE CASCADE,
  title TEXT,
  body TEXT,
  labels TEXT,
  state TEXT,
  created_at TIMESTAMP,
  closed_at TIMESTAMP,
  PRIMARY KEY (number, repo_id)
);
CREATE TABLE IF NOT EXISTS pull_requests (
  number INTEGER,
  repo_id TEXT REFERENCES repositories(id) ON DELETE CASCADE,
  title TEXT,
  body TEXT,
  reviewers TEXT,
  state TEXT,
  merged_at TIMESTAMP,
  files_changed TEXT,
  PRIMARY KEY (number, repo_id)
);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  repo_id TEXT REFERENCES repositories(id) ON DELETE CASCADE,
  file_path TEXT,
  doc_type TEXT,
  content TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS investigation_records (
  id TEXT PRIMARY KEY,
  repo_id TEXT REFERENCES repositories(id) ON DELETE CASCADE,
  query TEXT NOT NULL,
  finding TEXT,
  hypothesis TEXT,
  recommendation TEXT,
  evidence_ids TEXT,
  answer TEXT,
  status TEXT DEFAULT 'open',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_investigation_repo ON investigation_records(repo_id);
CREATE TABLE IF NOT EXISTS query_logs (
  id TEXT PRIMARY KEY,
  repo_id TEXT,
  user_id TEXT,
  query TEXT NOT NULL,
  answer TEXT,
  confidence REAL,
  grounded INTEGER,
  iterations INTEGER,
  elapsed_seconds REAL,
  input_tokens INTEGER,
  output_tokens INTEGER,
  cost_usd REAL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_query_logs_repo ON query_logs(repo_id);
"""


def _qpred_id(repo_id: str, file_path: str, prediction_type: str) -> str:
    """FIX-009: deterministic quality-prediction row id."""
    return hashlib.sha1(
        f"{repo_id}|{file_path}|{prediction_type}".encode()).hexdigest()[:32]


class SQLiteDatabase:
    """Embedded relational store implementing the documented FindingsStore API.

    FIX-015: every thread gets its own SQLite connection (WAL mode); reads no
    longer serialize behind a global lock. FIX-044: the old
    `_LockedConnection` proxy is gone entirely.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        settings = get_settings()
        self.db_path = db_path or (settings.data_dir / "repointel.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize the schema on a bootstrap connection
        boot = self._new_conn()
        try:
            boot.executescript(SCHEMA)
            boot.commit()
        finally:
            boot.close()

        self._local = threading.local()
        logger.info(f"SQLite store ready at {self.db_path} (WAL, thread-local connections)")

    def _new_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # FIX-013: explicit BEGIN/COMMIT control
            timeout=10.0,
        )
        conn.row_factory = sqlite3.Row
        # Pragmas are per-connection, so reapply on each new connection
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        """Return the calling thread's connection, creating one if needed."""
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._new_conn()
            self._local.conn = c
        return c

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    # ------------------------------------------------------------- repositories
    def upsert_repository(self, snapshot: RepositorySnapshot, stats: dict,
                          status: str = "indexing") -> None:
        """FIX-013: status is a parameter — 'indexing' while building, then
        flipped to 'ready' only after all writes commit."""
        name = snapshot.repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git") or snapshot.repo_id[:8]
        self.conn.execute(
            """INSERT INTO repositories (id, url, name, branch, last_indexed_commit, last_indexed_at, status, stats)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET last_indexed_commit=excluded.last_indexed_commit,
               last_indexed_at=excluded.last_indexed_at, status=excluded.status, stats=excluded.stats,
               updated_at=CURRENT_TIMESTAMP""",
            (snapshot.repo_id, snapshot.repo_url, name, snapshot.branch,
             snapshot.commit_hash, utc_now_iso(), status, json.dumps(stats)),
        )
        self.conn.commit()

    def get_repository(self, repo_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM repositories WHERE id = ?", (repo_id,)).fetchone()
        return dict(row) if row else None

    def list_repositories(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM repositories ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

    def store_files(self, repo_id: str, files) -> None:
        self.conn.execute("DELETE FROM files WHERE repo_id = ?", (repo_id,))
        self.conn.executemany(
            "INSERT OR REPLACE INTO files (id, repo_id, path, language, category, size_bytes, content_hash)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(str(uuid.uuid4()), repo_id, f.path, f.language, f.category,
              f.size_bytes, f.content_hash) for f in files],
        )
        self.conn.commit()

    # ------------------------------------------------------------------ findings
    def insert_finding(self, finding: Finding) -> str:
        self.conn.execute(
            """INSERT OR REPLACE INTO findings
               (id, repo_id, tool, rule_id, finding_type, severity, message, file_path,
                line_start, line_end, symbol_id, confidence, status, commit_hash, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (finding.id, finding.repo_id, finding.tool, finding.rule_id, finding.finding_type,
             finding.severity, finding.message, finding.file_path, finding.line_start,
             finding.line_end, finding.symbol_id, finding.confidence, finding.status,
             finding.commit_hash, json.dumps(finding.metadata)),
        )
        self.conn.commit()
        return finding.id

    def insert_findings_batch(self, repo_id: str, findings: list[Finding]) -> list[str]:
        for f in findings:
            f.repo_id = repo_id
        self.conn.executemany(
            """INSERT OR REPLACE INTO findings
               (id, repo_id, tool, rule_id, finding_type, severity, message, file_path,
                line_start, line_end, symbol_id, confidence, status, commit_hash, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(f.id, repo_id, f.tool, f.rule_id, f.finding_type, f.severity, f.message,
              f.file_path, f.line_start, f.line_end, f.symbol_id, f.confidence, f.status,
              f.commit_hash, json.dumps(f.metadata)) for f in findings],
        )
        self.conn.commit()
        logger.info(f"Stored {len(findings)} findings for {repo_id}")
        return [f.id for f in findings]

    def delete_findings_for_repo(self, repo_id: str) -> int:
        """FIX-005: full re-ingestion replaces the previous finding set."""
        cur = self.conn.execute("DELETE FROM findings WHERE repo_id = ?", (repo_id,))
        self.conn.commit()
        return cur.rowcount

    def delete_findings_by_files(self, repo_id: str, file_paths: list[str]) -> int:
        """FIX-012: incremental updates delete findings for touched files only."""
        if not file_paths:
            return 0
        placeholders = ",".join("?" * len(file_paths))
        cur = self.conn.execute(
            f"DELETE FROM findings WHERE repo_id = ? AND file_path IN ({placeholders})",
            [repo_id, *file_paths],
        )
        self.conn.commit()
        return cur.rowcount

    def get_findings_by_file(self, repo_id: str, file_path: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM findings WHERE repo_id = ? AND file_path = ? ORDER BY line_start",
            (repo_id, file_path)).fetchall()
        return [dict(r) for r in rows]

    def get_findings_by_severity(self, repo_id: str, severity: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM findings WHERE repo_id = ? AND severity = ?",
            (repo_id, severity)).fetchall()
        return [dict(r) for r in rows]

    def get_findings(self, repo_id: str, severity: str | None = None,
                     finding_type: str | None = None, limit: int = 500) -> list[dict]:
        query = "SELECT * FROM findings WHERE repo_id = ?"
        params: list[Any] = [repo_id]
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if finding_type:
            query += " AND finding_type = ?"
            params.append(finding_type)
        query += " ORDER BY CASE severity WHEN 'blocker' THEN 0 WHEN 'critical' THEN 1" \
                 " WHEN 'major' THEN 2 WHEN 'minor' THEN 3 ELSE 4 END, line_start LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    def update_status(self, finding_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE findings SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, finding_id))
        self.conn.commit()

    # -------------------------------------------------------- quality predictions
    def delete_quality_predictions_for_repo(self, repo_id: str) -> int:
        """FIX-005: full re-ingestion replaces the previous prediction set."""
        cur = self.conn.execute("DELETE FROM quality_predictions WHERE repo_id = ?", (repo_id,))
        self.conn.commit()
        return cur.rowcount

    def store_quality_predictions(self, repo_id: str, predictions: list[QualityPrediction]) -> None:
        """FIX-009 Option A: one row per (file, prediction_type).

        A file yields a 'defect' row always, a 'smell:<type>' row when a smell
        is predicted, and a 'td' row when td_score is meaningfully nonzero.
        satd_flag is encoded in the td row's confidence (1.0 = SATD present).
        Ids are deterministic so re-ingestion upserts instead of duplicating.
        """
        rows = []
        for p in predictions:
            features = json.dumps(p.features)
            # Defect row (always written)
            rows.append((
                _qpred_id(repo_id, p.file_path, "defect"), repo_id, p.file_path,
                p.symbol_id, "defect", p.defect_probability, p.defect_probability,
                p.model_version, features,
            ))
            # Smell row (only if a smell was predicted)
            if p.smell_type:
                rows.append((
                    _qpred_id(repo_id, p.file_path, f"smell:{p.smell_type}"), repo_id,
                    p.file_path, p.symbol_id, f"smell:{p.smell_type}",
                    p.smell_confidence, p.smell_confidence, p.model_version, features,
                ))
            # TD row (only if td_score is meaningfully nonzero)
            if p.td_score > 0.05:
                rows.append((
                    _qpred_id(repo_id, p.file_path, "td"), repo_id, p.file_path,
                    p.symbol_id, "td", p.td_score,
                    1.0 if p.satd_flag else 0.7, p.model_version, features,
                ))

        self.conn.executemany(
            """INSERT OR REPLACE INTO quality_predictions
               (id, repo_id, file_path, symbol_id, prediction_type, probability,
                confidence, model_version, features) VALUES (?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self.conn.commit()

    def get_quality_predictions(self, repo_id: str, file_path: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM quality_predictions WHERE repo_id = ? AND file_path = ?",
            (repo_id, file_path)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ commits/issues
    def store_commits(self, repo_id: str, commits) -> None:
        self.conn.executemany(
            """INSERT OR REPLACE INTO commits
               (hash, repo_id, author, author_email, timestamp, message, additions, deletions, files_changed)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [(c.hash, repo_id, c.author, c.author_email, c.timestamp.isoformat(),
              c.message, c.additions, c.deletions, json.dumps(c.files_changed)) for c in commits])
        self.conn.commit()

    def store_issues(self, repo_id: str, issues) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO issues (number, repo_id, title, body, labels, state, created_at, closed_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            [(i.number, repo_id, i.title, i.body, json.dumps(i.labels), i.state,
              i.created_at.isoformat() if i.created_at else None,
              i.closed_at.isoformat() if i.closed_at else None) for i in issues])
        self.conn.commit()

    def store_pull_requests(self, repo_id: str, prs) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO pull_requests (number, repo_id, title, body, reviewers, state, merged_at, files_changed)"
            " VALUES (?,?,?,?,?,?,?,?)",
            [(p.number, repo_id, p.title, p.body, json.dumps(p.reviewers), p.state,
              p.merged_at.isoformat() if p.merged_at else None, json.dumps(p.files_changed))
             for p in prs])
        self.conn.commit()

    def store_documents(self, repo_id: str, docs) -> None:
        self.conn.execute("DELETE FROM documents WHERE repo_id = ?", (repo_id,))
        self.conn.executemany(
            "INSERT INTO documents (id, repo_id, file_path, doc_type, content) VALUES (?,?,?,?,?)",
            [(str(uuid.uuid4()), repo_id, d.file_path, d.doc_type, d.content[:20000]) for d in docs])
        self.conn.commit()

    # -------------------------------------------------------- investigation/log
    def store_investigation(self, repo_id: str, query: str, answer: str,
                            evidence_ids: list[str], recommendation: str = "") -> str:
        iid = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO investigation_records
               (id, repo_id, query, recommendation, evidence_ids, answer, status)
               VALUES (?,?,?,?,?,?, 'resolved')""",
            (iid, repo_id, query, recommendation, json.dumps(evidence_ids), answer))
        self.conn.commit()
        return iid

    def get_recent_investigations(self, repo_id: str, limit: int = 20) -> list[dict]:
        """FIX-031: recent investigations for a repository, newest first."""
        rows = self.conn.execute(
            """SELECT id, query, answer, recommendation, evidence_ids, status, created_at
               FROM investigation_records WHERE repo_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (repo_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def log_query(self, repo_id: str, question: str, answer: str, confidence: float,
                  grounded: bool, iterations: int, elapsed: float,
                  input_tokens: int, output_tokens: int) -> None:
        self.conn.execute(
            """INSERT INTO query_logs
               (id, repo_id, query, answer, confidence, grounded, iterations,
                elapsed_seconds, input_tokens, output_tokens)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), repo_id, question, answer, confidence,
             int(grounded), iterations, elapsed, input_tokens, output_tokens))
        self.conn.commit()

    def query_stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) as n, AVG(elapsed_seconds) as avg_latency, AVG(confidence) as avg_conf"
            " FROM query_logs").fetchone()
        return dict(row) if row else {}
