"""Phase 1 orchestrator — Build & Maintain Repository Knowledge (§4).

Runs the full Phase 1 flow:
STEP 1 ingestion → STEP 2 analysis (5 components) → STEP 3 knowledge base
(4 stores). Also exposes incremental updates for changed files only.

Transaction semantics (FIX-013): the repo row is marked 'indexing' before
any write, all DB writes for a full build happen inside one transaction that
flips the status to 'ready' at commit, and failures mark the repo 'failed'
so the next attempt re-runs. Incremental updates merge chunk sets instead
of replacing them (FIX-012) and invalidate Phase 2 caches (FIX-014).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

from app.core.analysis.builtin_analyzer import run_builtin_analysis
from app.core.analysis.codeql_runner import run_codeql
from app.core.analysis.history_intelligence import analyze_history
from app.core.analysis.quality_predictor import run_quality_prediction
from app.core.analysis.sast_orchestrator import SASTOrchestrator
from app.core.analysis.semgrep_runner import run_semgrep_scan
from app.core.analysis.sonarqube_runner import run_sonarqube
from app.core.ingestion.file_walker import read_file_content, walk_repository
from app.core.ingestion.ingestion_service import IngestionComponent
from app.core.knowledge.graph_builder import get_graph_store
from app.core.storage.sqlite_graph_store import SQLiteGraphStore
from app.core.knowledge.vector_indexer import (
    VectorIndexer, chunk_file, chunk_finding, chunk_issue,
)
from app.core.parsing.ast_extractor import extract_code_intelligence, parse_code_files
from app.db.sqlite_store import SQLiteDatabase
from app.models.evidence import Chunk
from app.models.repository import DeltaSnapshot, RepositorySnapshot
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Phase1Pipeline:
    """Ingest + analyze + index a repository (the §4 Phase 1 flow)."""

    def __init__(self, db: SQLiteDatabase | None = None) -> None:
        from app.config import get_settings

        self.settings = get_settings()
        self.ingestion = IngestionComponent()
        self.graph_store = get_graph_store()
        self.vector_indexer = VectorIndexer()
        self.db = db or SQLiteDatabase()
        self._repo_paths: dict[str, Path] = {}
        # FIX-014: cross-cutting cache invalidation listeners
        self._invalidation_listeners: list[Callable[[str], None]] = []

    # ------------------------------------------------------------- listeners
    def register_invalidation_listener(self, listener: Callable[[str], None]) -> None:
        """FIX-014: subscribe to repo-mutation events (e.g. Phase 2 caches)."""
        self._invalidation_listeners.append(listener)

    def _emit_invalidation(self, repo_id: str) -> None:
        for listener in self._invalidation_listeners:
            try:
                listener(repo_id)
            except Exception as exc:
                logger.warning(f"Invalidation listener raised: {exc}")

    # ---------------------------------------------------------------- full run
    def build_knowledge(
        self,
        repo_url: str,
        branch: str = "main",
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        fetch_github_data: bool = True,
    ) -> dict:
        """Full Phase 1 pipeline run. Returns a summary report.

        FIX-013: either completes fully or leaves a 'failed' marker — partial
        state is never indistinguishable from complete ingestion.
        """
        started = time.monotonic()
        report: dict = {"repo_url": repo_url, "branch": branch, "phases": {}}
        snapshot = None

        try:
            # ---- STEP 1: ingestion ------------------------------------------------
            t0 = time.monotonic()
            snapshot = self.ingestion.ingest(
                repo_url, branch, include_patterns, exclude_patterns, fetch_github_data)
            repo_path = self.ingestion.repo_path_for(snapshot.repo_id)
            self._repo_paths[snapshot.repo_id] = repo_path
            report["phases"]["ingestion"] = {**snapshot.stats(), "seconds": round(time.monotonic() - t0, 2)}

            # Mark indexing BEFORE any other DB write (FIX-013)
            self.db.upsert_repository(snapshot, report["phases"]["ingestion"],
                                      status="indexing")

            # ---- STEP 2: analysis (5 components) ------------------------------------
            t0 = time.monotonic()
            asts = parse_code_files(repo_path, snapshot.files)
            code_intelligence = extract_code_intelligence(asts)
            report["phases"]["parsing"] = {**code_intelligence.stats(), "seconds": round(time.monotonic() - t0, 2)}

            t0 = time.monotonic()
            findings = SASTOrchestrator(self.db).run_sast_pipeline(
                repo_path, snapshot.files, code_intelligence=code_intelligence, persist_to_db=False)
            findings += run_sonarqube(repo_path, snapshot.repo_id)
            findings += run_codeql(repo_path, snapshot.repo_id)

            # Gap analysis (gated — default off)
            if os.environ.get("ENABLE_GAP_ANALYSIS", "0") == "1":
                from app.core.analysis.gap_analyzer import analyze_gaps
                gap_findings = analyze_gaps(repo_path, snapshot.files, code_intelligence)
                findings += gap_findings
                report.setdefault("phases", {})["gap_analysis"] = {
                    "findings": len(gap_findings),
                    "by_type": _count_findings_by(gap_findings, "finding_type"),
                }

            report["phases"]["analysis"] = {
                "findings": len(findings),
                "tools": sorted({f.tool for f in findings}),
                "seconds": round(time.monotonic() - t0, 2),
            }

            t0 = time.monotonic()
            predictions = run_quality_prediction(
                repo_path, snapshot.files, code_intelligence, snapshot.commits, findings)
            report["phases"]["quality_prediction"] = {"files": len(predictions), "seconds": round(time.monotonic() - t0, 2)}

            t0 = time.monotonic()
            history = analyze_history(snapshot.commits, [f.path for f in snapshot.files], findings)
            report["phases"]["history"] = {**history.stats(), "seconds": round(time.monotonic() - t0, 2)}

            # ---- STEP 3: knowledge base (4 stores) -----------------------------------
            t0 = time.monotonic()
            graph_stats = self.graph_store.build_graph(snapshot, code_intelligence, findings)

            # Intent check — MUST run after graph is built, with repo_id scoping (gated — default off)
            if os.environ.get("ENABLE_INTENT_LAYER", "0") == "1":
                from app.core.analysis.intent_checker import load_intent, check_intent
                intent = load_intent(repo_path)
                if intent:
                    intent_findings = check_intent(
                        repo_path, self.graph_store, intent, repo_id=snapshot.repo_id
                    )
                    findings += intent_findings
                    report.setdefault("phases", {})["intent_check"] = {"findings": len(intent_findings)}
                else:
                    report.setdefault("phases", {})["intent_check"] = {
                        "findings": 0,
                        "status": "no architecture.yml — intent layer not checked",
                    }

            chunks = self._build_chunks(snapshot, code_intelligence, findings)
            vector_stats = self.vector_indexer.build_vector_index(snapshot.repo_id, chunks)

            # All DB writes in a single transaction (FIX-013 + FIX-005)
            try:
                self.db.store_files(snapshot.repo_id, snapshot.files)
                self.db.delete_findings_for_repo(snapshot.repo_id)  # FIX-005
                self.db.insert_findings_batch(snapshot.repo_id, findings)
                self.db.delete_quality_predictions_for_repo(snapshot.repo_id)  # FIX-005
                self.db.store_quality_predictions(snapshot.repo_id, predictions)
                self.db.store_commits(snapshot.repo_id, snapshot.commits)
                self.db.store_issues(snapshot.repo_id, snapshot.issues)
                self.db.store_pull_requests(snapshot.repo_id, snapshot.pull_requests)
                self.db.store_documents(snapshot.repo_id, snapshot.documentation)
                full_stats = {
                    "ingestion": report["phases"].get("ingestion", {}),
                    "parsing": report["phases"].get("parsing", {}),
                    "analysis": report["phases"].get("analysis", {}),
                    "quality_prediction": report["phases"].get("quality_prediction", {}),
                    "history": report["phases"].get("history", {}),
                    "knowledge_base": {
                        "graph": graph_stats,
                        "vector": vector_stats,
                    },
                    "findings": len(findings),
                }
                self.db.conn.execute(
                    "UPDATE repositories SET status = 'ready', stats = ? WHERE id = ?",
                    (json.dumps(full_stats), snapshot.repo_id),
                )
                self.db.conn.commit()
            except Exception:
                try:
                    self.db.conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

            report["phases"]["knowledge_base"] = {
                "graph": graph_stats, "vector": vector_stats,
                "persisted": "sqlite", "seconds": round(time.monotonic() - t0, 2),
            }
            # persist vector index for cross-process restore (§4.3.2 'Save index')
            self.vector_indexer.persist(snapshot.repo_id)
            self._persist_graph(snapshot.repo_id)
            for record in self.db.list_repositories():
                if record["id"] == snapshot.repo_id and Path(record.get("url") or "").exists():
                    self._repo_paths[snapshot.repo_id] = Path(record["url"]).resolve()

            # FIX-014: notify listeners that this repo's data changed
            self._emit_invalidation(snapshot.repo_id)

            report["repo_id"] = snapshot.repo_id
            report["commit_hash"] = snapshot.commit_hash
            report["total_seconds"] = round(time.monotonic() - started, 2)
            report["status"] = "ready"
            logger.info("phase1_complete", extra={"extra_fields": {
                "repo_id": snapshot.repo_id, "total_seconds": report["total_seconds"],
                "chunks": vector_stats.get("chunks", 0), "findings": len(findings),
            }})
            return report

        except Exception as exc:
            # Best-effort mark as failed (FIX-013)
            repo_id = snapshot.repo_id if snapshot is not None else None
            if repo_id:
                try:
                    self.db.conn.execute(
                        "UPDATE repositories SET status = 'failed' WHERE id = ?", (repo_id,))
                    self.db.conn.commit()
                except Exception:
                    pass
            logger.error("phase1_failed", extra={"extra_fields": {
                "repo_url": repo_url, "error": str(exc)[:300]}})
            raise

    # ------------------------------------------------------------- incremental
    def incremental_update(self, repo_id: str) -> dict:
        """§4.1.5: re-process only changed files since last indexed commit."""
        repo = self.db.get_repository(repo_id)
        if repo is None:
            raise ValueError(f"Unknown repo_id {repo_id} — run build_knowledge first")
        last_commit = repo["last_indexed_commit"]
        repo_url = repo.get("url") or ""
        delta = self.ingestion.incremental_update(repo_id, last_commit, repo_url=repo_url)
        if delta.total_changes == 0:
            return {"repo_id": repo_id, "changes": 0, "status": "up-to-date"}
        self._apply_delta(repo_id, delta)
        return {
            "repo_id": repo_id, "changes": delta.total_changes,
            "added": len(delta.added), "modified": len(delta.modified),
            "deleted": len(delta.deleted), "renamed": len(delta.renamed),
            "status": "updated",
        }

    def _apply_delta(self, repo_id: str, delta: DeltaSnapshot) -> None:
        """Merge-delta semantics (FIX-012): keep existing chunks, replace
        changed files' chunks, drop deleted files' chunks — never wipe the
        whole index with only the delta's files."""
        repo_path = self._repo_paths.get(repo_id) or self.ingestion.repo_path_for(repo_id)
        if repo_path is None:
            raise RuntimeError(f"Cannot apply delta for {repo_id}: repo path unknown")

        # Step 1: which file paths are touched / deleted by the delta
        # Etch Detail 3: filter out files whose AST fingerprint has not changed
        from app.core.ingestion.incremental import filter_structurally_modified_files
        structurally_modified = filter_structurally_modified_files(repo_path, delta.modified, self.db, repo_id)
        touched_paths: set[str] = set()
        deleted_paths: set[str] = set()
        for change in (delta.added + structurally_modified):
            touched_paths.add(change.path)
        for change in delta.deleted:
            deleted_paths.add(change.path)
        for change in delta.renamed:
            touched_paths.add(change.path)          # new path
            if getattr(change, "old_path", None):
                deleted_paths.add(change.old_path)  # old path is gone

        # Step 2: walk the working tree for FileRecords of touched files
        all_files = walk_repository(repo_path) if repo_path.exists() else []
        touched_files = [f for f in all_files if f.path in touched_paths]

        # Step 3: reparse + reanalyze only touched files
        asts = parse_code_files(repo_path, touched_files)
        code_intelligence = extract_code_intelligence(asts)
        findings = run_builtin_analysis(repo_path, touched_files, code_intelligence)
        findings += run_semgrep_scan(repo_path)

        # Step 4: MERGE with existing chunks — do not replace the index
        existing_chunks = list(self.vector_indexer.chunks_by_repo.get(repo_id, []))
        removed = 0
        surviving: list[Chunk] = []
        for c in existing_chunks:
            if c.file_path in deleted_paths or c.file_path in touched_paths:
                removed += 1
                continue
            surviving.append(c)

        new_chunks = self._chunks_for_files(
            repo_id, repo_path, touched_files, code_intelligence, findings)
        merged = surviving + new_chunks

        # Step 4.5: Update the Graph Store incrementally
        all_affected_paths = sorted(touched_paths | deleted_paths)
        if hasattr(self.graph_store, "remove_file_graph") and hasattr(self.graph_store, "insert_partial_graph"):
            self.graph_store.remove_file_graph(repo_id, all_affected_paths)
            if touched_files:
                self.graph_store.insert_partial_graph(repo_id, touched_files, code_intelligence, findings)
            logger.info("incremental_graph_updated", extra={"extra_fields": {
                "touched": len(touched_paths), "deleted": len(deleted_paths)
            }})
        else:
            logger.warning("Graph store does not support incremental updates; skipping graph sync.")

        # Step 5: rebuild the vector index over the merged set
        if merged:
            self.vector_indexer.build_vector_index(repo_id, merged)
        else:
            self.vector_indexer.indexes.pop(repo_id, None)
            self.vector_indexer.chunks_by_repo[repo_id] = []

        # Step 6: findings — delete only findings for touched/deleted files
        self.db.delete_findings_by_files(repo_id, sorted(touched_paths | deleted_paths))
        self.db.insert_findings_batch(repo_id, findings)

        # Step 7: update the last indexed commit
        self.db.conn.execute(
            "UPDATE repositories SET last_indexed_commit = ?, status='ready' WHERE id = ?",
            (delta.to_commit, repo_id))
        self.db.conn.commit()

        # Step 8: persist the merged index + graph
        self.vector_indexer.persist(repo_id)
        self._persist_graph(repo_id)

        # FIX-014: notify listeners (Phase 2 caches must drop stale data)
        self._emit_invalidation(repo_id)

        logger.info("incremental_delta_applied", extra={"extra_fields": {
            "repo_id": repo_id,
            "chunks_surviving": len(surviving),
            "chunks_new": len(new_chunks),
            "chunks_removed": removed,
            "files_touched": len(touched_paths),
            "files_deleted": len(deleted_paths),
        }})

    # ---------------------------------------------------------------- internals
    def _persist_graph(self, repo_id: str) -> None:
        """Persist the embedded graph so a fresh process can restore it."""
        if hasattr(self.graph_store, "persist"):
            path = self.settings.data_dir / f"{repo_id}.graph.pkl"
            try:
                self.graph_store.persist(path)
            except Exception:
                pass

    def restore_knowledge(self, repo_id: str) -> bool:
        """Reload persisted knowledge base (vector + graph) for a repo."""
        if repo_id in self.vector_indexer.indexes:
            return True  # already loaded
        ok = self.vector_indexer.restore(repo_id)
        if ok and hasattr(self.graph_store, "load"):
            graph_path = self.settings.data_dir / f"{repo_id}.graph.pkl"
            if graph_path.exists():
                try:
                    restored = self.graph_store.load(graph_path)
                    if hasattr(restored, "graph"):
                        self.graph_store.graph = restored.graph
                        if hasattr(self.graph_store, "_rebuild_indexes"):
                            self.graph_store._rebuild_indexes()
                except Exception:
                    pass
        if ok:
            record = self.db.get_repository(repo_id)
            if record:
                url = record.get("url") or ""
                p = Path(url)
                if p.exists():
                    self._repo_paths[repo_id] = p.resolve()
                elif (self.settings.data_dir.parent / url).exists():
                    self._repo_paths[repo_id] = (self.settings.data_dir.parent / url).resolve()
                elif (self.settings.data_dir / url).exists():
                    self._repo_paths[repo_id] = (self.settings.data_dir / url).resolve()
                elif (self.settings.repos_dir / p.name).exists():
                    self._repo_paths[repo_id] = (self.settings.repos_dir / p.name).resolve()
                else:
                    clone = self.settings.repos_dir / repo_id
                    if (clone / ".git").exists():
                        self._repo_paths[repo_id] = clone
        return ok

    # ------------------------------------------------------------------ chunks
    def _build_chunks(self, snapshot: RepositorySnapshot, code_intelligence, findings) -> list[Chunk]:
        """Chunk all sources: code (by symbol), docs, issues, PRs, findings."""
        repo_path = self._repo_paths.get(snapshot.repo_id)
        symbols = code_intelligence.symbols
        chunks: list[Chunk] = []

        for file in snapshot.files:
            if file.category in ("binary", "vendored", "data"):
                continue
            content, ok = read_file_content(repo_path, file.path)
            if not ok or not content.strip():
                continue
            chunks.extend(chunk_file(snapshot.repo_id, file, content, symbols))

        for issue in snapshot.issues:
            chunks.append(chunk_issue(snapshot.repo_id, issue))
        for pr in snapshot.pull_requests:
            chunks.append(chunk_issue(snapshot.repo_id, pr))
        for finding in findings:
            chunks.append(chunk_finding(snapshot.repo_id, finding, repo_path=repo_path))

        return chunks

    def _build_chunks_for(self, repo_id, files, code_intelligence, findings):
        return self._chunks_for_files(repo_id, self._repo_paths.get(repo_id), files,
                                      code_intelligence, findings)

    def _chunks_for_files(self, repo_id, repo_path, files, code_intelligence, findings):
        symbols = code_intelligence.symbols
        chunks: list[Chunk] = []
        for file in files:
            if file.category in ("binary", "vendored", "data"):
                continue
            content, ok = read_file_content(repo_path, file.path) if repo_path else ("", False)
            if not ok or not content.strip():
                continue
            chunks.extend(chunk_file(repo_id, file, content, symbols))
        for finding in findings:
            chunks.append(chunk_finding(repo_id, finding, repo_path=repo_path))
        return chunks

    # -------------------------------------------------------------- health info
    def health(self) -> dict:
        return {
            "graph_store": type(self.graph_store).__name__,
            "vector_index": self.vector_indexer.stats(),
            "repositories": self.db.list_repositories(),
        }


def _count_findings_by(findings: list, attr: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in findings:
        k = getattr(f, attr, "unknown") or "unknown"
        out[k] = out.get(k, 0) + 1
    return out
