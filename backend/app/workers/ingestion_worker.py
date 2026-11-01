"""Background ingestion worker (docker-compose `worker` service).

Pulls ingestion requests from a queue (Redis when available, a simple
local file queue otherwise) and runs the Phase 1 pipeline.
"""
from __future__ import annotations

import time

from app.core.knowledge.knowledge_service import Phase1Pipeline
from app.utils.logger import get_logger

logger = get_logger("repointel.worker")


def get_queue():
    """Redis-backed queue when available; None → direct mode."""
    try:
        import redis

        from app.config import get_settings

        settings = get_settings()
        client = redis.Redis(host=settings.redis_host, port=settings.redis_port)
        client.ping()
        return client
    except Exception:
        return None


def main(poll_seconds: float = 5.0) -> None:
    logger.info("ingestion_worker_started")
    queue = get_queue()
    pipeline = Phase1Pipeline()

    while True:
        repo_url = None
        if queue is not None:
            item = queue.blpop("celery:queue:ingestion", timeout=int(poll_seconds))
            if item:
                repo_url = item[1].decode()
        if repo_url:
            logger.info("worker_ingesting", extra={"extra_fields": {"repo": repo_url}})
            try:
                report = pipeline.build_knowledge(repo_url, fetch_github_data=True)
                logger.info("worker_ingest_done", extra={"extra_fields": {
                    "repo_id": report.get("repo_id"),
                    "seconds": report.get("total_seconds")}})
            except Exception as exc:
                logger.error("worker_ingest_failed", extra={"extra_fields": {
                    "repo": repo_url, "error": str(exc)[:200]}})
        else:
            time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
