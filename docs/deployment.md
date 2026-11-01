# Deployment

Local (zero infra): make api  (SQLite + networkx + in-memory cache + z-ai LLM)

Production:
1. docker compose -f infra/docker-compose.yml up
   (backend + worker + neo4j + postgres + redis + mongo + prometheus + grafana)
2. Set env in .env (copy .env.example): NEO4J_PASSWORD, POSTGRES_PASSWORD,
   GITHUB_TOKEN, LLM_PROVIDER + API key.
3. Kubernetes: kubectl apply -f infra/kubernetes/backend-deployment.yaml
   (image from §19 CI pipeline).
4. Backups: see §24.4 scripts (pg_dump / neo4j-admin dump / copy faiss files).

## Required GitHub Actions secrets for deploy (FIX-017)

| Secret | Purpose |
|--------|---------|
| `CONTAINER_REGISTRY` | e.g. `ghcr.io/your-org` |
| `CONTAINER_REGISTRY_USER` | Registry username |
| `CONTAINER_REGISTRY_PASSWORD` | Registry token / PAT |
| `CONTAINER_IMAGE_NAME` | e.g. `repointel-backend` |
| `KUBECONFIG` | base64-encoded kubeconfig for the target cluster |

If unset, the deploy job skips gracefully (logs `skip=true`, exits 0).

## MCP server (FIX-045 + 2026-07-28 protocol)

RepoIntel exposes five tools via MCP over stdio (stateless core — no
`Mcp-Session-Id`, no handshake state; discovery via `server/discover`):

    cd backend && python -m app.core.mcp.server

Registration in Claude Code (`~/.config/claude/mcp.json` or equivalent):

    {
        "mcpServers": {
            "repointel": {
                "command": "python",
                "args": ["-m", "app.core.mcp.server"],
                "cwd": "/path/to/repointel-ai/backend"
            }
        }
    }

Security: the server has no network listener. Trust is inherited from the
spawning process. Do not wrap in a network transport without TLS + auth.
Setting `MCP_ALLOW_NETWORK=1` is refused at startup by design.
