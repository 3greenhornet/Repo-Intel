# Contributing to RepoIntel AI

## Setup

1. Install Python 3.12+ (3.13 recommended — see `pyproject.toml`).
2. `pip install -r backend/requirements.txt`
3. `make demo` — creates the fixture repo and builds the knowledge base.
4. `make test` — runs unit + integration tests.
5. `make api` — serves the dashboard on http://localhost:8000.

## Adding a dependency

1. Edit `backend/requirements.in`.
2. `make compile-deps` — regenerates the lock file with hashes (FIX-021).
3. Commit both `requirements.in` and `requirements.txt`.

## Making a change

1. Create a feature branch.
2. Write a test that fails before your change and passes after.
3. Run `make test` and `ruff check backend/app/ scripts/ --select E,W,F,B,S`.
4. Open a PR; CI (unit + integration + lint + pip-audit) must be green.
   E2E runs on `main` merges and nightly.

## Code style

- `ruff` for linting (config in `pyproject.toml`).
- `black` for formatting (line length 100).
- Type hints on all public functions.
- Docstrings reference the implementation documentation section (`§X.Y`)
  or the fix document ID (`FIX-NNN`) where relevant.
- No bare `except:` in production code (E722 is lint-enforced; the demo
  fixture is the only exempted file — deliberately, so the analyzer flags it).
- Timezone-aware datetimes only — use `app.utils.time.utc_now()` (FIX-041).
- Durations via `time.monotonic()`, wall clock only for expiries/JWT (FIX-042).
