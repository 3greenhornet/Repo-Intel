# verify.ps1 — RepoIntel AI environment verification
# Usage:  .\verify.ps1
# Prints PASS/FAIL per check, exits 0 if all pass.

$ErrorActionPreference = "Continue"
$script:fails = @()

function Check {
    param($name, [scriptblock]$test)
    try {
        $result = & $test
        if ($result) {
            Write-Host "  PASS  $name" -ForegroundColor Green
        } else {
            Write-Host "  FAIL  $name" -ForegroundColor Red
            $script:fails += $name
        }
    } catch {
        Write-Host "  FAIL  $name ($($_.Exception.Message.Split("`n")[0]))" -ForegroundColor Red
        $script:fails += $name
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  RepoIntel AI - Environment Verification" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ---------------------------------------------------------------- Python
Write-Host ""
Write-Host "[1] Python interpreter" -ForegroundColor Yellow

$py = "py"
$pyArgs = @("-3.14")

Check "py -3.14 is available" {
    (py -3.14 --version 2>&1) -match "Python 3\.(1[2-4])"
}

Check "py -3.14 resolves to Programs\Python (NOT WindowsApps)" {
    $exe = & py -3.14 -c "import sys; print(sys.executable)"
    $exe -notlike "*WindowsApps*" -and $exe -like "*Python*"
}

Check "Python version is 3.12, 3.13, or 3.14" {
    $v = (& py -3.14 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    $v -in @("3.12", "3.13", "3.14")
}

# ---------------------------------------------------------------- Packages
Write-Host ""
Write-Host "[2] Required packages" -ForegroundColor Yellow

$packages = @(
    "torch", "fastapi", "uvicorn", "httpx", "pydantic", "pydantic_settings",
    "tree_sitter", "tree_sitter_python", "tree_sitter_javascript", "tree_sitter_typescript",
    "sentence_transformers", "transformers", "rank_bm25", "networkx", "sklearn",
    "flashrank", "litellm", "prometheus_client", "jwt", "structlog"
)

foreach ($pkg in $packages) {
    Check "import $pkg" {
        & py -3.14 -c "import $pkg" 2>$null
        $LASTEXITCODE -eq 0
    }
}

# ---------------------------------------------------------------- Semgrep
Write-Host ""
Write-Host "[3] Semgrep (CLI)" -ForegroundColor Yellow

Check "semgrep CLI installed" {
    $null = Get-Command semgrep -ErrorAction Stop
    $true
}

Check "semgrep --version returns a version" {
    $v = (semgrep --version 2>&1) | Select-Object -First 1
    $v -match "^\d+\.\d+"
}

Check "semgrep rules validate" {
    $r = semgrep --validate --config=config/semgrep/security-audit.yaml 2>&1
    $r -match "found 0 configuration error"
}

# ---------------------------------------------------------------- Files
Write-Host ""
Write-Host "[4] Required files" -ForegroundColor Yellow

$requiredFiles = @(
    "backend/app/main.py",
    "backend/app/core/knowledge/knowledge_service.py",
    "backend/app/core/knowledge/vector_indexer.py",
    "backend/app/core/retrieval/reranker.py",
    "backend/app/core/agent/investigator.py",
    "backend/app/core/reasoning/query_pipeline.py",
    "backend/app/static/index.html",
    "backend/requirements.txt",
    "demo_evidence/eval_results.json"
)

foreach ($f in $requiredFiles) {
    Check "file exists: $f" { Test-Path $f }
}

# ---------------------------------------------------------------- Data
Write-Host ""
Write-Host "[5] Data & knowledge base" -ForegroundColor Yellow

Check "data/repointel.db exists" { Test-Path "data/repointel.db" }
Check "data/faiss exists" { Test-Path "data/faiss" }
Check "demo repo exists" { Test-Path "data/repos/demo-web-shop/.git" }

# ---------------------------------------------------------------- Server
Write-Host ""
Write-Host "[6] Live server (port 8000)" -ForegroundColor Yellow

$serverRunning = $false
try {
    $r = & py -3.14 -c "import httpx; print(httpx.get('http://127.0.0.1:8000/api/v1/health', timeout=3).status_code)" 2>$null
    $serverRunning = ($r -eq "200")
} catch { }

if ($serverRunning) {
    Check "server responds on :8000" { $true }
    Check "health status is healthy or degraded" {
        $s = (& py -3.14 -c "import httpx; print(httpx.get('http://127.0.0.1:8000/api/v1/health', timeout=30).json()['status'])" 2>$null).Trim()
        $s -in @("healthy", "degraded")
    }
    Check "GET /api/v1/repos returns data" {
        $j = & py -3.14 -c "import httpx, json; print(httpx.get('http://127.0.0.1:8000/api/v1/repos', timeout=5).text)" 2>$null
        $j -match '"repositories"'
    }
    Check "GET /docs (Swagger) responds" {
        (& py -3.14 -c "import httpx; print(httpx.get('http://127.0.0.1:8000/docs', timeout=5).status_code)" 2>$null).Trim() -eq "200"
    }
} else {
    Write-Host "  SKIP  server checks (server not running - start with: cd backend; py -3.14 -m uvicorn app.main:app --port 8000)" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------- Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
if ($script:fails.Count -eq 0) {
    Write-Host "  ALL CHECKS PASSED" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor White
    Write-Host "  cd backend" -ForegroundColor Gray
    Write-Host "  py -3.14 -m uvicorn app.main:app --host 0.0.0.0 --port 8000" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Then open http://localhost:8000/ (wait ~40s on first start for NLI model)" -ForegroundColor Gray
    Write-Host ""
    exit 0
} else {
    Write-Host "  $($script:fails.Count) CHECK(S) FAILED" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Failed checks:" -ForegroundColor Red
    foreach ($f in $script:fails) {
        Write-Host "  - $f" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "See SETUP.md troubleshooting section." -ForegroundColor White
    Write-Host ""
    exit 1
}