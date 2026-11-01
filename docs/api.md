# API reference

Interactive docs: http://localhost:8000/docs (FastAPI/OpenAPI)

| Method | Path | Spec § |
|---|---|---|
| POST | /api/v1/query | 10.3.2 |
| POST | /api/v1/query/stream (SSE) | 23.8 |
| POST | /api/v1/ingest | 4 |
| POST | /api/v1/ingest/update/{repo_id} | 4.1.5 |
| GET | /api/v1/findings/{repo_id}?severity=&finding_type= | 10.3.2 |
| GET | /api/v1/findings/{repo_id}/recommendations | 11.2 |
| GET | /api/v1/repos | — |
| GET | /api/v1/repos/{repo_id} | — |
| GET | /api/v1/architecture/{repo_id} | 10.3.1 |
| GET | /api/v1/impact/{repo_id}?symbol= | 7.3 |
| GET | /api/v1/health | 24.5 |
| GET | /metrics | 18.2 |
| GET | / | dashboard |
