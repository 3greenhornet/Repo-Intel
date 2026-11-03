<USER_REQUEST>
## Yes. Run these in this exact order. Stop at the first failure.

The pattern all conversation: too many fixes, not enough verification. This test list is deliberately short. If everything passes, **you're demo-ready and stop touching code.**

---

## Phase 1 — Verify what you just implemented (5 minutes)

### Test 1.1 — Semantic abstention edge case

**Run:**
```powershell
& "C:\Users\sreen\AppData\Local\Python\pythoncore-3.14-64\python.exe" -c "
import sys; sys.path.insert(0, 'backend')
from app.core.reasoning.query_pipeline import is_abstention

a1 = 'There is no evidence of SQL injection in auth.py [Evidence 1], as parameterized queries are enforced [Evidence 2].'
assert is_abstention(a1) == False, 'a1 FAILED: substantive answer with citations flagged as abstention'

a2 = 'I do not have enough evidence to answer this. [Evidence 13] mentions login_user which might be related.'
assert is_abstention(a2) == True, 'a2 FAILED: hedge-then-cite not flagged as abstention'

a3 = 'The login function validates credentials and calls hash_password [Evidence 1].'
assert is_abstention(a3) == False, 'a3 FAILED: normal grounded answer flagged as abstention'

a4 = 'I don\'t have enough evidence to answer this.'
assert is_abstention(a4) == True, 'a4 FAILED: pure abstention not flagged'

print('PASS: all 4 abstention cases correct')
"
```

**PASS:** `PASS: all 4 abstention cases correct`
**FAIL:** any assertion error → the regex in `_EXACT_ABSTENTION` needs adjustment. Report the failing case.

**Why this matters:** if a1 fails, every substantive answer on stage gets `confidence: 0.0` in the JSON. If a2 fails, you can't demonstrate abstention behavior. Both are visible if anyone inspects responses.

---

### Test 1.2 — Path traversal guard

**Run:**
```powershell
$repo = (Invoke-RestMethod http://127.0.0.1:8000/api/v1/repos).repositories[0].id

Write-Host "`n[1] Valid file (expect 200 with content)"
$r1 = Invoke-WebRequest "http://127.0.0.1:8000/api/v1/repos/$repo/file?path=demo_shop/auth.py&start=20&end=30"
Write-Host "  Status: $($r1.StatusCode)"

Write-Host "`n[2] Relative traversal (expect 400)"
try {
    $r2 = Invoke-WebRequest "http://127.0.0.1:8000/api/v1/repos/$repo/file?path=../../../Windows/win.ini"
    Write-Host "  Status: $($r2.StatusCode) -- FAILED, should be 400"
} catch {
    Write-Host "  Status: $($_.Exception.Response.StatusCode.value__) -- OK (rejected)"
}

Write-Host "`n[3] Absolute Windows path (expect 400)"
try {
    $r3 = Invoke-WebRequest "http://127.0.0.1:8000/api/v1/repos/$repo/file?path=C:/Windows/win.ini"
    Write-Host "  Status: $($r3.StatusCode) -- FAILED, should be 400"
} catch {
    Write-Host "  Status: $($_.Exception.Response.StatusCode.value__) -- OK (rejected)"
}

Write-Host "`n[4] URL-encoded traversal (expect 400)"
try {
    $r4 = Invoke-WebRequest "http://127.0.0.1:8000/api/v1/repos/$repo/file?path=..%2F..%2F..%2Fetc%2Fpasswd"
    Write-Host "  Status: $($r4.StatusCode) -- FAILED, should be 400"
} catch {
    Write-Host "  Status: $($_.Exception.Response.StatusCode.value__) -- OK (rejected)"
}

Write-Host "`n[5] Doubled traversal (expect 400)"
try {
    $r5 = Invoke-WebRequest "http://127.0.0.1:8000/api/v1/repos/$repo/file?path=....//....//etc/passwd"
    Write-Host "  Status: $($r5.StatusCode) -- FAILED, should be 400"
} catch {
    Write-Host "  Status: $($_.Exception.Response.StatusCode.value__) -- OK (rejected)"
}
```

**PASS criteria:**
- `[1]` returns `200`
- `[2]`, `[3]`, `[4]`, `[5]` all return `400` or `404` (rejected)

**FAIL:** if any of tests 2-5 returns `200` with file content → **the guard is broken, do not demo the citation modal.**

**Why this matters:** if any attack string succeeds, you should not show the citation viewer at all in the demo. Better to hide the modal than to have a security researcher in the audience spot it.

---

### Test 1.3 — Citation modal click-through

**Manual browser test — no command:**

1. Open `http://localhost:8000/`
2. Select `demo-web-shop`
3. Query: `What security vulnerabilities exist in the authentication code?`
4. Press **EXECUTE**
5. Wait for citations to appear
6. **Click a citation** on `demo_shop/auth.py:23-28`

**PASS criteria:**
- Modal opens
- File content visible with line numbers
- Lines 23-28 highlighted (target lines visually distinct from gutter)
- `COPY SNIPPET` copies the right text (paste in notepad to confirm)
- `COPY PATH` copies `demo_shop/auth.py:23-28`
- `OPEN IN VS CODE` triggers a "dispatched" toast or opens VS Code (either is fine — the graceful failure is expected)
- `ESC` closes the modal

**FAIL:** any step missing or broken → don't demo the modal live; screenshot it instead.

---

## Phase 2 — The 30-second check that could crash the server (2 minutes)

### Test 2.1 — `.encode()` vs `.embed()` regression

**Run:**
```powershell
Select-String -Path "backend/app/core/knowledge/vector_indexer.py" -Pattern "self\.engine\.(encode|embed)"
```

**PASS:** output shows `self.engine.embed(` (the correct method)
**FAIL:** output shows `self.engine.encode(` → **one-word fix required**: change `encode` to `embed` in `restore()`.

**Why this matters:** if `.encode()` is present, `phase1.restore_knowledge()` raises `AttributeError` on any cold start when the vector index needs reloading from disk. That kills every query on a fresh server start — including the one you demo first.

---

### Test 2.2 — Cold restart doesn't lose the index

**Run:**
```powershell
# 1. Kill server
Get-Process | Where-Object { $_.ProcessName -eq "python" -and $_.Path -like "*pythoncore*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 2

# 2. Restart
cd backend
Start-Process -FilePath "py" -ArgumentList "-3.14", "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"
Start-Sleep 10

# 3. Query immediately after restart
$repo = (Invoke-RestMethod http://127.0.0.1:8000/api/v1/repos).repositories[0].id
$r = Invoke-RestMethod -Method POST "http://127.0.0.1:8000/api/v1/query" -ContentType "application/json" -Body (@{repo_id=$repo; question="Who calls login?"} | ConvertTo-Json)
Write-Host "Answer length: $($r.answer.Length)"
Write-Host "Citations: $($r.citations.Count)"
```

**PASS:** `Answer length: > 50`, `Citations: > 0`
**FAIL:** empty answer or zero citations → the index is not being restored on cold start. **This is the highest-priority bug to fix if present.**

**Why this matters:** on stage you'll restart the server at least once (fresh shell, or after a crash). If queries return empty after restart, that's a total demo failure.

---

## Phase 3 — The 3 pre-demo smoke tests (10 minutes)

### Test 3.1 — LLM provider

**Run:**
```powershell
py -3.14 verify_gemini.py
```

**PASS:** at least one line contains `[+] ... SUCCESS`
**FAIL:** all lines are `FAILURE` or `SKIPPED` → set a key now:
```powershell
$env:GROQ_API_KEY = "gsk_..."
# or GEMINI_API_KEY, OPENAI_API_KEY, etc.
```

---

### Test 3.2 — Dashboard EXECUTE

**Manual browser test:**

1. Reload `http://localhost:8000/`
2. Select `demo-web-shop`
3. Query: `What security vulnerabilities exist in the authentication code?`
4. Click **EXECUTE [ENTER]**

**PASS:**
- Pipeline stepper advances: `[01 RETRIEVAL]` → `[02 NLI]` → `[03 SYNTHESIS]` (all three become green)
- Answer streams into left pane within 30 seconds
- Right pane populates with ≥ 3 citation cards
- vis-network graph renders below citations

**FAIL:** any of the above missing → check server logs for errors before proceeding.

---

### Test 3.3 — ReAct Investigator

**Run:**
```powershell
& "C:\Users\sreen\AppData\Local\Python\pythoncore-3.14-64\python.exe" -c "
import httpx, time
repos = httpx.get('http://127.0.0.1:8000/api/v1/repos').json()['repositories']
d = [r for r in repos if r['name'] == 'demo-web-shop'][0]
started = time.time()
r = httpx.post('http://127.0.0.1:8000/api/v1/query/investigate',
    json={'repo_id': d['id'], 'question': 'Who calls login?'},
    timeout=180)
elapsed = time.time() - started
j = r.json()
print('STATUS:', r.status_code)
print('STEPS:', j.get('steps'))
print('BUDGET_EXHAUSTED:', j.get('budget_exhausted'))
print('CITATIONS:', len(j.get('citations', [])))
print('ELAPSED:', round(elapsed, 1))
"
```

**PASS criteria (all must be true):**
- `STATUS: 200`
- `STEPS: >= 2`
- `BUDGET_EXHAUSTED: false`
- `CITATIONS: > 0`
- `ELAPSED: < 90`

**Run it 3 times.** All 3 must PASS.

**If 3/3 PASS:** demo INVESTIGATE live — you earned it.
**If 1 or 2 out of 3 fail:** INVESTIGATE is not demo-stable → **screenshot it in advance and don't click it on stage.**
**If 3/3 fail:** something's wrong with the fix. Investigate before the demo.

**Why run 3 times:** a single success could be a fluke. 3/3 is the threshold for "safe to click on stage."

---

## Phase 4 — Verify which of the 8 demo-critical items are actually applied (2 minutes)

Run this quick status check. It tells you if any of the 8 bugs from earlier are still present:

```powershell
cd backend
py -3.14 -c "
import inspect, sys
print('=== Demo-critical 8-item status ===\n')

# C1 — LexicalIndex.load restores chunks
from app.core.retrieval.lexical import LexicalIndex
src = inspect.getsource(LexicalIndex.load)
print('C1 LexicalIndex.load restores chunks:', 'chunks' in src and 'self.chunks' in src)

# C2 — _last_contradictions not a singleton attribute
from app.core.reasoning import query_pipeline
src = inspect.getsource(query_pipeline)
print('C2 _last_contradictions removed:', '_last_contradictions' not in src)

# E1 — health endpoint doesn't call nli._load()
from app.main import health
import ast
src = inspect.getsource(health)
print('E1 health non-blocking:', 'nli._load()' not in src or '# Do NOT call' in src)

# E2 — get_pipelines has lock
from app import main as mainmod
src = inspect.getsource(mainmod.get_pipelines)
print('E2 get_pipelines locked:', 'lock' in src.lower())

# J1 — Evidence has origin field
from app.models.evidence import Evidence
e = Evidence(chunk=None, score=0, source='test')
print('J1 Evidence.origin present:', hasattr(e, 'origin'))

# J3 — resolve_repo_id preserves path case
from app.core.ingestion.github_client import resolve_repo_id
a = resolve_repo_id('/home/Alice/proj')
b = resolve_repo_id('/home/alice/proj')
print('J3 resolve_repo_id case-sensitive for paths:', a != b)

# J5 — middleware order
src = open('app/main.py', encoding='utf-8').read()
rl_pos = src.find('async def rate_limit_middleware')
rid_pos = src.find('async def request_id_middleware')
print('J5 middleware order correct:', rid_pos > rl_pos and rid_pos > 0)

# N1 — rate limiter lazy init
print('N1 rate_limiter lazy init:', 'def _get_rate_limiter' in src or '_rate_limiter = None' in src)
"
```

**PASS:** all 8 lines show `True`
**FAIL:** any line shows `False` → that's one of the 8 demo-critical fixes NOT applied.

**If any of the 8 fail:** these are the only additions worth making before the demo. Each is small (~5-30 min). Do them. Then re-run this check.

**If all 8 pass:** you're as ready as you'll be. Nothing else to add.

---

## Phase 5 — Rehearse (10 minutes, no code changes)

**Run through the demo script 2 times out loud.** Not just click it — actually say the words. Time yourself.

**Script (from Section 16.2):**

| Time | Action | Words |
|---|---|---|
| 0:00 | Dashboard visible, KPIs loaded | "RepoIntel — evidence-grounded repository intelligence." |
| 0:15 | Point at KPI grid | "Files, symbols, findings, critical alerts — real numbers." |
| 0:30 | EXECUTE security query | "Hybrid retrieval: BM25 + dense + graph, RRF, FlashRank on CPU." |
| 0:50 | Point at stepper | "Retrieval → NLI verification → synthesis." |
| 1:10 | Point at citations + vis-network | "Every claim cites file:line." |
| 1:30 | Switch to Findings, filter Critical | "SQL injection, eval — deterministic, no LLM." |
| 1:50 | Architecture tab, click hub | "Blast radius — SQLite CTE, sub-millisecond." |
| **2:10 (optional if Test 3.3 passed 3/3)** | Query → INVESTIGATE [REACT] | "ReAct agent: 7 tools, step-by-step reasoning." |
| 2:35 | Point at steps + tokens | "Bounded — N steps, N seconds." |
| **STOP** | Return to Q&A, leave security answer on screen | "Full pipeline, one LLM key, zero infrastructure." |

**PASS:** completes in under 3 minutes without stumbling over the click path.
**FAIL:** you fumble a click or can't remember the next step → practice again. The audience notices hesitation more than content.

**Do not edit code between rehearsal and the actual demo.** A "small fix" after rehearsal is the highest-probability way to break what you just proved works.

---

## What to do based on test results

| Scenario | Action |
|---|---|
| **All phases pass** | You're done. Do not touch code. Show up and demo. |
| **Phase 1 any fail** | Fix only that specific test, re-run it. Don't run other phases until fixed. |
| **Phase 2 any fail** | **This is the highest priority.** The `.encode()` check and cold-restart test are non-negotiable. Fix these first. |
| **Phase 3.3 fails 2 of 3 runs** | Don't demo INVESTIGATE live. Screenshot it in advance, show the screenshot during the demo. |
| **Phase 4 any fails** | Fix only the specific failing items from the 8-item list. Ignore everything else. |
| **Phase 5 fumbles** | Practice the script again. Do not add features. |

---

## The one thing to know

**If Phase 1 + Phase 2 + Phase 3 all pass, you have a working demo right now.** The remaining failures in Phase 4 (if any) are small enough to fix in an hour. The remaining risk after that is not code — it's your rehearsal.

**Report the test results back.** If everything passes, I'll tell you to stop and demo. If something fails, name the test and I'll give the exact fix. No new features, no work orders, no documents — just the one failing test.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-02T10:41:12+05:30.

The user's current state is as follows:
Other open documents:
- c:\Users\sreen\OneDrive\Pictures\Bharath Gitam\ai !411\repo intel ai\repointel-ai-2\.gitignore (LANGUAGE_UNSPECIFIED)
- c:\Users\sreen\OneDrive\Pictures\Bharath Gitam\ai !411\repo intel ai\repointel-ai-2\.env.example (LANGUAGE_UNSPECIFIED)
- c:\Users\sreen\OneDrive\Pictures\Bharath Gitam\ai !411\repo intel ai\repointel-ai-2\scripts\run_agent_eval.py (LANGUAGE_UNSPECIFIED)
- c:\Users\sreen\OneDrive\Pictures\Bharath Gitam\ai !411\repo intel ai\repointel-ai-2\scripts\run_agent.py (LANGUAGE_UNSPECIFIED)
- c:\Users\sreen\OneDrive\Pictures\Bharath Gitam\ai !411\repo intel ai\repointel-ai-2\backend\app\core\agent\investigator.py (LANGUAGE_UNSPECIFIED)
</ADDITIONAL_METADATA>