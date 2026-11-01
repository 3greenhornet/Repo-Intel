# Frontend

## Current state (MVP)

The dashboard served at `GET /` is a **dependency-free single-page app**
located at `backend/app/static/index.html`. It is mounted by the FastAPI
application and requires no separate build step, no npm, and no external
JavaScript.

It provides:
- Q&A with live SSE streaming and citation rendering
- Findings browser with severity and type filters
- Prioritized recommendations with impact lists
- Architecture summary (symbol counts, call edges, hub symbols)
- Repository ingestion and status
- Contradiction surfacing in answers (FIX-029)

Because it has no external dependencies, it works fully offline and starts
with the backend — no separate dev server, no CORS configuration.

## Future: React 18 port

A React 18 port that consumes the same REST API is future work. The API is
stable and documented in `docs/api.md`; the endpoints the React port would
consume are:

- `POST /api/v1/query` and `POST /api/v1/query/stream` (SSE)
- `GET /api/v1/findings/{repo_id}` and `GET /api/v1/findings/{repo_id}/recommendations`
- `GET /api/v1/architecture/{repo_id}`
- `POST /api/v1/ingest` and `POST /api/v1/ingest/update/{repo_id}`
- `POST /api/v1/auth/token` (when REQUIRE_AUTH=1)

The static dashboard implements the exact same views and can be used as a
functional reference for the port.

## Why not React for the MVP

The MVP's primary goal is validating the pipeline end-to-end. Introducing
npm, a build step, and a separate dev server adds operational friction that
does not advance the pipeline. The static dashboard demonstrates the same
functionality with zero build infrastructure.
