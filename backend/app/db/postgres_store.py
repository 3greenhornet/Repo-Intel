"""Production PostgreSQL store — implements the same API as SQLiteDatabase.

# HONEST NOTE (Phase 2): This PostgreSQL store is fully implemented but 
# disabled by default (RELATIONAL_STORE=sqlite). The DDL matches the 
# documented schema exactly. The SQLite store proves the relational logic 
# works locally with zero infrastructure. To activate, set RELATIONAL_STORE=postgres
# and start the Postgres container.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from app.config import get_settings
from app.db.postgres import POSTGRES_DDL
from app.models.finding import Finding, QualityPrediction
from app.models.repository import RepositorySnapshot
from app.utils.logger import get_logger
from app.utils.time import utc_now_iso

logger = get_logger(__name__)


class PostgresStore:
    """Synchronous Postgres store with the SQLiteDatabase interface."""

    def __init__(self) -> None:
        import psycopg2
        import psycopg2.extras

        self._psycopg2 = psycopg2
        self._psycopg2_extras = psycopg2.extras
        s = get_settings()
        self.conn = psycopg2.connect(
            host=s.postgres_host, port=s.postgres_port,
            dbname=s.postgres_db, user=s.postgres_user, password=s.postgres_password,
        )
        self.conn.autocommit = False
        with self.conn.cursor() as cur:
            cur.execute(POSTGRES_DDL)
        self.conn.commit()
        logger.info("Postgres store ready at %s:%s", s.postgres_host, s.postgres_port)

    def _cursor(self):
        return self.conn.cursor(cursor_factory=self._psycopg2_extras.RealDictCursor)

    # ---- interface parity with SQLiteDatabase ----
    def upsert_repository(self, snapshot: RepositorySnapshot, stats: dict,
                          status: str = "indexing") -> None:
        name = snapshot.repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git") or snapshot.repo_id[:8]
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO repositories (id, url, name, branch, last_indexed_commit,
                                          last_indexed_at, status, stats)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    last_indexed_commit = EXCLUDED.last_indexed_commit,
                    last_indexed_at = EXCLUDED.last_indexed_at,
                    status = EXCLUDED.status, stats = EXCLUDED.stats,
                    updated_at = NOW()
            """, (snapshot.repo_id, snapshot.repo_url, name, snapshot.branch,
                  snapshot.commit_hash, utc_now_iso(), status, json.dumps(stats)))
        self.conn.commit()

    def get_repository(self, repo_id: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM repositories WHERE id = %s", (repo_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_repositories(self) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM repositories ORDER BY updated_at DESC")
            return [dict(r) for r in cur.fetchall()]

    def store_files(self, repo_id: str, files) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM files WHERE repo_id = %s", (repo_id,))
            self._psycopg2_extras.execute_values(
                cur,
                """INSERT INTO files (id, repo_id, path, language, category, size_bytes,
                                      content_hash)
                VALUES %s""",
                [(str(uuid.uuid4()), repo_id, f.path, f.language, f.category,
                  f.size_bytes, f.content_hash) for f in files],
            )
        self.conn.commit()

    def delete_findings_for_repo(self, repo_id: str) -> int:
        with self._cursor() as cur:
            cur.execute("DELETE FROM findings WHERE repo_id = %s", (repo_id,))
            count = cur.rowcount
        self.conn.commit()
        return count

    def delete_findings_by_files(self, repo_id: str, file_paths: list[str]) -> int:
        if not file_paths:
            return 0
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM findings WHERE repo_id = %s AND file_path = ANY(%s)",
                (repo_id, file_paths))
            count = cur.rowcount
        self.conn.commit()
        return count

    def delete_quality_predictions_for_repo(self, repo_id: str) -> int:
        with self._cursor() as cur:
            cur.execute("DELETE FROM quality_predictions WHERE repo_id = %s", (repo_id,))
            count = cur.rowcount
        self.conn.commit()
        return count

    def insert_finding(self, finding: Finding) -> str:
        return self.insert_findings_batch(finding.repo_id or "", [finding])[0]

    def insert_findings_batch(self, repo_id: str, findings: list[Finding]) -> list[str]:
        for f in findings:
            f.repo_id = repo_id
        rows = [
            (f.id, repo_id, f.tool, f.rule_id, f.finding_type, f.severity, f.message,
             f.file_path, f.line_start, f.line_end, f.symbol_id, f.confidence,
             f.status, f.commit_hash, json.dumps(f.metadata))
            for f in findings
        ]
        with self._cursor() as cur:
            self._psycopg2_extras.execute_values(
                cur,
                """INSERT INTO findings
                    (id, repo_id, tool, rule_id, finding_type, severity, message,
                     file_path, line_start, line_end, symbol_id, confidence,
                     status, commit_hash, metadata)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    severity = EXCLUDED.severity,
                    message = EXCLUDED.message,
                    status = EXCLUDED.status,
                    updated_at = NOW()""",
                rows,
            )
        self.conn.commit()
        return [f.id for f in findings]

    def get_findings(self, repo_id: str, severity: str | None = None,
                     finding_type: str | None = None, limit: int = 500) -> list[dict]:
        sql = "SELECT * FROM findings WHERE repo_id = %s"
        params: list[Any] = [repo_id]
        if severity:
            sql += " AND severity = %s"
            params.append(severity)
        if finding_type:
            sql += " AND finding_type = %s"
            params.append(finding_type)
        sql += (" ORDER BY CASE severity"
                " WHEN 'blocker' THEN 0 WHEN 'critical' THEN 1"
                " WHEN 'major' THEN 2 WHEN 'minor' THEN 3 ELSE 4 END,"
                " line_start LIMIT %s")
        params.append(limit)
        with self._cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def get_findings_by_file(self, repo_id: str, file_path: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM findings WHERE repo_id=%s AND file_path=%s ORDER BY line_start",
                (repo_id, file_path))
            return [dict(r) for r in cur.fetchall()]

    def get_findings_by_severity(self, repo_id: str, severity: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM findings WHERE repo_id=%s AND severity=%s",
                        (repo_id, severity))
            return [dict(r) for r in cur.fetchall()]

    def update_status(self, finding_id: str, status: str) -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE findings SET status=%s, updated_at=NOW() WHERE id=%s",
                        (status, finding_id))
        self.conn.commit()

    def store_quality_predictions(self, repo_id: str, predictions: list[QualityPrediction]) -> None:
        """FIX-009 parity: one row per (file, prediction_type)."""
        rows = []
        for p in predictions:
            features = json.dumps(p.features)
            rows.append((p.id, repo_id, p.file_path, p.symbol_id, "defect",
                         p.defect_probability, p.defect_probability, features, p.model_version))
            if p.smell_type:
                rows.append((f"{p.id}:{p.smell_type}", repo_id, p.file_path, p.symbol_id,
                             f"smell:{p.smell_type}", p.smell_confidence, p.smell_confidence,
                             features, p.model_version))
            if p.td_score > 0.05:
                rows.append((f"{p.id}:td", repo_id, p.file_path, p.symbol_id, "td",
                             p.td_score, 1.0 if p.satd_flag else 0.7, features, p.model_version))
        with self._cursor() as cur:
            self._psycopg2_extras.execute_values(
                cur,
                """INSERT INTO quality_predictions
                    (id, repo_id, file_path, symbol_id, prediction_type,
                     probability, confidence, features, model_version)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    probability = EXCLUDED.probability,
                    confidence = EXCLUDED.confidence,
                    features = EXCLUDED.features""",
                rows,
            )
        self.conn.commit()

    def get_quality_predictions(self, repo_id: str, file_path: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM quality_predictions WHERE repo_id=%s AND file_path=%s",
                (repo_id, file_path))
            return [dict(r) for r in cur.fetchall()]

    def store_commits(self, repo_id: str, commits) -> None:
        with self._cursor() as cur:
            self._psycopg2_extras.execute_values(
                cur,
                """INSERT INTO commits
                    (hash, repo_id, author, author_email, timestamp, message,
                     additions, deletions, files_changed)
                VALUES %s
                ON CONFLICT (hash, repo_id) DO NOTHING""",
                [(c.hash, repo_id, c.author, c.author_email, c.timestamp,
                  c.message, c.additions, c.deletions, c.files_changed) for c in commits],
            )
        self.conn.commit()

    def store_issues(self, repo_id: str, issues) -> None:
        with self._cursor() as cur:
            self._psycopg2_extras.execute_values(
                cur,
                """INSERT INTO issues (number, repo_id, title, body, labels, state,
                                       created_at, closed_at)
                VALUES %s
                ON CONFLICT (number, repo_id) DO NOTHING""",
                [(i.number, repo_id, i.title, i.body, i.labels, i.state,
                  i.created_at, i.closed_at) for i in issues],
            )
        self.conn.commit()

    def store_pull_requests(self, repo_id: str, prs) -> None:
        with self._cursor() as cur:
            self._psycopg2_extras.execute_values(
                cur,
                """INSERT INTO pull_requests
                    (number, repo_id, title, body, reviewers, state, merged_at, files_changed)
                VALUES %s
                ON CONFLICT (number, repo_id) DO NOTHING""",
                [(p.number, repo_id, p.title, p.body, p.reviewers, p.state,
                  p.merged_at, p.files_changed) for p in prs],
            )
        self.conn.commit()

    def store_documents(self, repo_id: str, docs) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM documents WHERE repo_id = %s", (repo_id,))
            self._psycopg2_extras.execute_values(
                cur,
                "INSERT INTO documents (id, repo_id, file_path, doc_type, content) VALUES %s",
                [(str(uuid.uuid4()), repo_id, d.file_path, d.doc_type, d.content[:20000])
                 for d in docs],
            )
        self.conn.commit()

    def store_investigation(self, repo_id: str, query: str, answer: str,
                            evidence_ids: list[str], recommendation: str = "") -> str:
        iid = str(uuid.uuid4())
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO investigation_records
                    (id, repo_id, query, recommendation, evidence_ids, answer, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'resolved')""",
                (iid, repo_id, query, recommendation, evidence_ids, answer),
            )
        self.conn.commit()
        return iid

    def get_recent_investigations(self, repo_id: str, limit: int = 20) -> list[dict]:
        """FIX-031 parity with SQLiteDatabase."""
        with self._cursor() as cur:
            cur.execute(
                """SELECT id, query, answer, recommendation, evidence_ids, status, created_at
                   FROM investigation_records WHERE repo_id = %s
                   ORDER BY created_at DESC LIMIT %s""",
                (repo_id, limit))
            return [dict(r) for r in cur.fetchall()]

    def log_query(self, repo_id: str, question: str, answer: str, confidence: float,
                  grounded: bool, iterations: int, elapsed: float,
                  input_tokens: int, output_tokens: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO query_logs
                    (id, repo_id, query, answer, confidence, grounded, iterations,
                     elapsed_seconds, input_tokens, output_tokens)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (str(uuid.uuid4()), repo_id, question, answer, confidence,
                 bool(grounded), iterations, elapsed, input_tokens, output_tokens),
            )
        self.conn.commit()

    def query_stats(self) -> dict:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n, AVG(elapsed_seconds) AS avg_latency,"
                        " AVG(confidence) AS avg_conf FROM query_logs")
            return dict(cur.fetchone())

    def close(self) -> None:
        self.conn.close()
