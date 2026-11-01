# RepoIntel AI — Setup & Verification

**Last verified:** 2026-11-01 on Windows 11, Python 3.14.5, torch 2.15.0+cpu
**Time to working demo:** ~10 minutes from clean machine

---

## TL;DR — The One Command That Works

If you have Python 3.14 + pip working, this is the whole install:

```powershell
# From repo root
py -3.14 -m pip install -r backend/requirements.txt --only-binary=:all:
pipx install semgrep
py -3.14 -c "import torch, fastapi, httpx, uvicorn, tree_sitter, sentence_transformers, rank_bm25, networkx, sklearn, flashrank, litellm; print('OK | torch', torch.__version__)"
```

Expected final line: `OK | torch 2.15.0+cpu`

Then start the server:
```powershell
cd backend
py -3.14 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**First start takes 40–60 seconds** — the NLI model (~128 MB) downloads on first `/health` call. This is normal. Wait, don't Ctrl+C.

Open `http://localhost:8000/` — you're done.

---

## Requirements

| Item | Minimum | Verified |
|---|---|---|
| Python | 3.12 | 3.14.5 ✅ |
| OS | Windows / Linux / macOS | Windows 11 ✅ |
| Disk | 3 GB (torch + models + DB) | |
| RAM | 4 GB | |
| Network | Needed once for model downloads | |

---

## Detailed Setup

### Step 0 — Check what Python you have

```powershell
py -0
```

Look for `3.12`, `3.13`, or `3.14` in the list. If you see `*` next to one of those, good.

**If you only see 3.8, 3.6, or Anaconda** → install Python 3.12+ from python.org or `winget install Python.Python.3.13`.

**Critical Windows gotcha:** `python` (bare) may point to `C:\Users\<user>\AppData\Local\Microsoft\WindowsApps\python.exe` — a **broken Store stub** that:
- Reports packages as installed via `pip show` but can't `import` them
- Has no site-packages
- Silently swallows `pip install`

**Always use `py -3.14` (or `py -3.13`), never bare `python`.**

Test it once:
```powershell
py -3.14 --version
py -3.14 -c "import sys; print(sys.executable)"
# Must print ...\Programs\Python\Python314\python.exe (NOT WindowsApps)
```

### Step 1 — Comment out semgrep from requirements

Semgrep's pip package fails to build on Windows (`Semgrep does not support Windows yet`). It's a CLI tool for this project, not a Python import — install via `pipx` instead.

```powershell
# From repo root
(Get-Content backend/requirements.txt) -replace '^semgrep.*', '# semgrep: install via pipx (CLI only on Windows)' | Set-Content backend/requirements.txt
```

Verify:
```powershell
Select-String -Path "backend/requirements.txt" -Pattern "semgrep"
# Should show: # semgrep: install via pipx (CLI only on Windows)
```

### Step 2 — Install torch first, from the CPU index

The default PyPI index for torch is CUDA (~2.5 GB). CPU-only is 128 MB and works fine here.

```powershell
py -3.14 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Expected: `Successfully installed torch-2.15.0+cpu ...`

### Step 3 — Install the rest

```powershell
py -3.14 -m pip install -r backend/requirements.txt --only-binary=:all:
```

**Why `--only-binary=:all:`** — without it, pip will backtrack through ~70 old versions of a package when one fails, wasting 5+ minutes. With it, pip fails fast on any package missing a wheel.

Expected: `Successfully installed fastapi-0.162.0 ... transformers-5.22.1 ... flashrank-0.2.10 ...` (71 packages)

**Ignore this warning** (it's about pipx, not the project):
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
pipx 1.18.11 requires filelock>=4.2, but you have filelock 3.32.7
```
Pipx will keep working. Not a problem.

### Step 4 — Install semgrep via pipx

```powershell
py -3.14 -m pip install --user pipx
py -3.14 -m pipx ensurepath
# Close and reopen terminal (PATH update)
pipx install semgrep
semgrep --version
```

Expected: `1.183.0`

### Step 5 — Verify everything

**Option A (recommended):** Run `verify.ps1` from repo root:
```powershell
.\verify.ps1
```
Should print `ALL CHECKS PASSED` at the end.

**Option B (manual one-liner):**
```powershell
py -3.14 -c "import torch, fastapi, httpx, uvicorn, tree_sitter, sentence_transformers, rank_bm25, networkx, sklearn, flashrank, litellm; print('OK | torch', torch.__version__)"
```

Expected: `OK | torch 2.15.0+cpu`

### Step 6 — Build the demo knowledge base (optional, once)

If `data/repointel.db` doesn't exist or the repos aren't indexed:

```powershell
python scripts/create_demo_repo.py data/repos/demo-web-shop
py -3.14 scripts/build_knowledge.py data/repos/demo-web-shop --no-github
```

Expected last line: `phase1_complete ... files: 11 ... chunks: 45 ... findings: 12`

### Step 7 — Start the server

```powershell
cd backend
py -3.14 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Expected startup sequence:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Then wait 40–60 seconds before the first `/health` call.** The NLI model downloads on first query. You'll see:
```
Downloading model.safetensors: 100%|...| 128MB
NLI model loaded successfully: dleemiller/EttinX-nli-xs on cpu
```

After that, health checks return in <100 ms.

### Step 8 — Confirm healthy

**Second terminal:**
```powershell
py -3.14 -c "import httpx; r = httpx.get('http://127.0.0.1:8000/api/v1/health', timeout=30); print(r.status_code, r.json()['status'])"
```

Expected: `200 healthy`

If `200 degraded`, inspect:
```powershell
py -3.14 -c "import httpx; print(httpx.get('http://127.0.0.1:8000/api/v1/health').json()['components'])"
```

---

## Demo

- **Dashboard:** `http://localhost:8000/`
- **Interactive API docs:** `http://localhost:8000/docs`
- **Pick repo `demo-web-shop`** (not `data\repos\demo-web-shop` — that's a duplicate)
- You should see: 12 findings (2 critical), 27 symbols, 38 call edges, 45 chunks

---

## Known Gotchas (things that WILL bite you)

### 1. First `/health` blocks for ~40s
The `/health` endpoint calls `phase2.nli._load()` synchronously. First call triggers the model download. **Fix in code:** make NLI load lazy on first query. **Workaround:** wait for `NLI model loaded successfully` in the server log before calling `/health`.

### 2. `python` vs `py -3.14`
`python` → WindowsApps stub (broken). `py -3.14` → real Python. **Always use `py -3.14`.**

### 3. The bare `verify_all.ps1` file (if present) is LYING
If you have a `verify_all.ps1` in the repo root, it uses `pip show torch` which returns metadata from the WindowsApps stub. It will say `PASS torch installed` while `import torch` fails. **Ignore or delete it.** Use `verify.ps1` (this repo's version) instead.

### 4. Semgrep pip package fails on Windows
Always use `pipx install semgrep`, never `pip install semgrep` on Windows. The pip package will fail with `Semgrep does not support Windows yet`.

### 5. pip backtracking waste
Without `--only-binary=:all:`, pip backtracks through hundreds of old versions of a failing package. On a slow connection this is 10+ minutes of "Collecting X ... Using cached ...".

### 6. HF symlink warning (harmless)
```
UserWarning: huggingface_hub cache-system uses symlinks by default... your machine does not support them
```
Silence with: `$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"`

### 7. Duplicate repo in dropdown
If you ran `make demo` AND ingested manually with a path, you'll see both `demo-web-shop` and `data\repos\demo-web-shop`. **Pick `demo-web-shop`.** Clean up with: `rm data/repointel.db` then re-run `make demo`.

### 8. Fresh shell = fresh PATH
After `winget install` or `pipx ensurepath`, you must close and reopen PowerShell, OR run:
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

---

## Troubleshooting Table

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` in health check | Used `python` (WindowsApps) instead of `py -3.14` | Use `py -3.14 -m uvicorn ...` |
| Health returns `degraded` | NLI/embeddings not loaded (still downloading) | Wait 60s, check server log for `NLI model loaded successfully` |
| Health returns 503 | Server running on `python` (WindowsApps) with no packages | Restart with `py -3.14 -m uvicorn` |
| `httpx.ReadTimeout` on health check | First call while NLI downloading | Use `timeout=30`, or wait for server log first |
| `semgrep` build fails with `does not support Windows yet` | Installing semgrep via pip | `pipx install semgrep` instead |
| `pip install` hangs for 10+ minutes | Backtracking through semgrep versions | Add `--only-binary=:all:` or remove semgrep |
| All packages install, but `import fastapi` fails | Running in WindowsApps stub Python | Use `py -3.14` explicitly |
| `codebase-rag.egg-info` or similar pollution | Copied files from another project | Safe to ignore / delete |
| Dashboard blank page | `backend/app/static/index.html` missing | `git status` — file should be tracked |
| Dashboard 404s on CSS/JS | Static mount failed | Check `FRONTEND_DIR` in `main.py` line ~45 |

---

## What NOT to Do

These are the wrong turns made on 2026-11-01 that cost 2+ hours. Do not repeat.

| Do NOT | Why | Do this instead |
|---|---|---|
| Create a venv | Only matters if you have multiple Python projects — you have one | Use `py -3.14` directly |
| Install Python 3.13 | torch has cp314 wheels since 2.15.0 — confirmed 2026-11-01 | Stay on 3.14 |
| Use bare `python` | It's the WindowsApps stub, lies about packages | Always `py -3.14` |
| Install semgrep via pip | Build fails on Windows, backtracks 70 versions | `pipx install semgrep` |
| Run `pip install -r` without `--only-binary=:all:` | pip backtracks for minutes on failure | Add the flag |
| Trust `verify_all.ps1` output | It checks `pip show` not `import` | Use `verify.ps1` from this repo |
| Try to `import torch` while in `(.venv)` with empty venv | The prompt lies — venv has no packages | `deactivate` first, or use `py -3.14` |
| Run health check immediately after starting uvicorn | First call blocks on NLI download | Wait for `NLI model loaded` log line |
| Delete `data/repointel.db` thinking it's a cache | It's the source of truth for all repos | Only delete if you want to re-ingest |

---

## What's NOT Verified (honest note)

These are documented in the repo but **not exercised in this setup**:

- **Neo4j backend** — adapter coded, no server to test against
- **PostgreSQL backend** — adapter coded, no server
- **Redis rate limiter** — adapter coded, defaults to in-memory
- **FAISS vector index** — adapter coded, uses NumPy by default
- **SonarQube runner** — needs external server + `ENABLE_SONAR=1`
- **CodeQL runner** — needs external CLI + `ENABLE_CODEQL=1`
- **Real GitHub ingestion** — only `--no-github` fixture runs verified
- **E2E tests** — need `GROQ_API_KEY` env var (3 tests skipped without it)
- **`scripts/verify_setup.py`** — the file referenced in docs **does not exist in the repo**

---

## Verification Script

`verify.ps1` (in this repo root) runs all checks in 15 seconds and prints `ALL CHECKS PASSED` or the exact failure. Use it after any environment change.

---

## Version Lock (as of 2026-11-01)

Recorded for reproducibility. Update this section if you change versions.

```
Python:              3.14.5
torch:               2.15.0+cpu
fastapi:             0.162.0
uvicorn:             0.48.0
transformers:        5.22.1
sentence-transformers: 6.2.0
flashrank:           0.2.10
litellm:             1.106.6
tree-sitter:         0.26.0
semgrep (pipx):      1.183.0
NLI model:           dleemiller/EttinX-nli-xs (128 MB, CPU)
```

---

**Repo:** `repointel-ai-2`
**Setup doc version:** 1.0 — 2026-11-01