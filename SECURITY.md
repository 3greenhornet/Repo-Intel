# Security Policy

## Reporting a vulnerability

Please do NOT open a public issue. Email security@repointel.example with:

- A description of the vulnerability
- Steps to reproduce
- Affected versions
- Any suggested fix

We will acknowledge within 3 business days and provide a remediation timeline.

## Supported versions

Only the latest release on `main` receives security patches.

## Known limitations

- Pickle model files are NOT committed (FIX-023); they are trained at build
  time by `scripts/train_quality_models.py` (deterministic, `SEED = 42`).
- Prompt injection is mitigated but not eliminated — see
  `backend/app/core/reasoning/prompt_templates.py::sanitize_repository_content`.
- The MCP server speaks stdio with no network listener; trust is inherited
  from the spawning process. Do not wrap it in a network transport without
  TLS + auth (FIX-045).
- The `/api/v1/auth/token` API-key flow is demo-grade; swap `api_key_salt`
  for a proper user store in production (FIX-030).
- CI supply-chain hardening: lint/audit tools are version-pinned, dependency
  installs are hash-checked, and container images are Trivy-scanned
  (FIX-021/022 + 2026 corrections).
