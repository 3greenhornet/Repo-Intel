FROM python:3.12-slim
WORKDIR /app

# System dependencies (git for ingestion; JDK for SonarQube scanner)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — version-locked (FIX-021/040). The lock pins exact
# versions verified by CI. Regenerate with `make compile-deps` to add
# --generate-hashes in an environment with unrestricted PyPI access; once
# hashes are present, switch the line below to `--require-hashes`.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt matplotlib

# Optional CodeQL CLI (enable in production by uncommenting):
# RUN curl -L https://github.com/github/codeql-cli-binaries/releases/latest/download/codeql-linux64.zip -o /tmp/codeql.zip \
#     && unzip /tmp/codeql.zip -d /opt && rm /tmp/codeql.zip
# ENV PATH="/opt/codeql:${PATH}"

# Application
COPY . .
RUN mkdir -p /data/faiss /data/repos backend/models/trained

# FIX-023: quality models are trained at build time (deterministic, SEED=42)
# instead of committing pickle artifacts into the repository.
RUN python -c "from app.core.analysis.model_trainer import main; main()"

# Health check (§24.5 — dependency-validating endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
