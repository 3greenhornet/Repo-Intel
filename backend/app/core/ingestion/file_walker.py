"""File walking + classification (§4.1.3 / §4.1.4)."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

from app.models.repository import FileRecord
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Lightweight regex-based secret redaction
SECRET_PATTERNS = [
    (re.compile(r"(?i)(?:api_key|apikey|secret|token|password|auth|jwt)[\s:=]+['\"]?([a-zA-Z0-9_\-\.]{16,})['\"]?"), "[REDACTED_SECRET]"),
    (re.compile(r"(gsk_[a-zA-Z0-9]{30,})"), "[REDACTED_GROQ_KEY]"),
    (re.compile(r"(sk-[a-zA-Z0-9]{20,})"), "[REDACTED_API_KEY]"),
    (re.compile(r"(ghp_[a-zA-Z0-9]{30,})"), "[REDACTED_GITHUB_KEY]"),
    (re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
]


def redact_secrets(text: str) -> str:
    """Mask sensitive credentials, tokens, and private keys before vectorization / LLM processing."""
    if not text:
        return text
    redacted = text
    for pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _load_gitignore_patterns(repo_path: Path) -> list[str]:
    """Parse local repository .gitignore file to respect developer exclusions."""
    gitignore = repo_path / ".gitignore"
    patterns = []
    if gitignore.is_file():
        try:
            for line in gitignore.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    pat = line.rstrip("/")
                    if pat.startswith("/"):
                        pat = pat[1:]
                    patterns.extend([pat, f"{pat}/**", f"**/{pat}"])
        except OSError:
            pass
    return patterns

# §4.1.3 File Type Classification table
CODE_EXTENSIONS = {
    ".py": "python", ".java": "java", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".rs": "rust",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".cs": "csharp", ".rb": "ruby", ".php": "php",
}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".ini", ".env", ".cfg"}
DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc"}
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".exe", ".so", ".dll", ".pdf",
    ".zip", ".tar", ".gz", ".jar", ".class", ".pyc", ".woff", ".woff2", ".ttf",
    ".mp3", ".mp4", ".bin", ".wasm",
}
DATA_EXTENSIONS = {".csv", ".parquet", ".sql", ".db", ".sqlite"}
VENDORED_DIRS = {"node_modules", "vendor", "third_party", "dist", "build", ".venv", "venv", "__pycache__"}
LARGE_FILE_BYTES = 10 * 1024 * 1024  # §4.1.7: skip parsing > 10MB

DEFAULT_EXCLUDES = [
    "node_modules/**", "vendor/**", "third_party/**", "dist/**", "build/**",
    ".git/**", ".venv/**", "venv/**", "__pycache__/**", "*.lock", "package-lock.json",
    ".scannerwork/**",  # FIX-007: SonarQube scanner working dir
    "sonar-project.properties",  # FIX-007: defense-in-depth vs committed props
]


def classify_file(rel_path: str) -> str:
    """Classify a file into §4.1.3 categories."""
    path = rel_path.lower()
    name = path.rsplit("/", 1)[-1]

    parts = set(path.split("/"))
    if parts & VENDORED_DIRS:
        return "vendored"

    ext = Path(name).suffix
    if name.startswith(("test_",)) or "_test" in name or ".spec." in name or name.startswith("conftest"):
        return "tests"
    if ext in CODE_EXTENSIONS:
        if name.endswith("_pb2.py") or "_generated" in name:
            return "generated"
        return "code"
    if ext in DOC_EXTENSIONS:
        return "docs"
    if ext in CONFIG_EXTENSIONS or name in {"makefile", "dockerfile", ".gitignore", ".env.example"}:
        return "config"
    if ext in BINARY_EXTENSIONS:
        return "binary"
    if ext in DATA_EXTENSIONS:
        return "data"
    return "other"


def detect_language(rel_path: str) -> str | None:
    ext = Path(rel_path).suffix.lower()
    return CODE_EXTENSIONS.get(ext)


def _detect_generated_header(path: Path) -> bool:
    """§4.1.3: files with 'DO NOT EDIT' header are generated."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(512)
        return "DO NOT EDIT" in head or "@generated" in head
    except OSError:
        return False


def walk_repository(
    repo_path: str | Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    read_contents: bool = False,
) -> list[FileRecord]:
    """§4.1.4 File Walking Algorithm — verbatim structure.

    Returns FileRecords with hash/size/mtime; optionally loads content.
    """
    repo_path = Path(repo_path)
    include_patterns = include_patterns or []
    base_excludes = list(exclude_patterns or DEFAULT_EXCLUDES)
    # Automatically load and respect .gitignore rules
    gitignore_rules = _load_gitignore_patterns(repo_path)
    all_excludes = base_excludes + gitignore_rules
    files: list[FileRecord] = []

    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(repo_path).as_posix()

        if any(fnmatch(relative_path, pat) for pat in all_excludes):
            continue
        if include_patterns and not any(fnmatch(relative_path, pat) for pat in include_patterns):
            continue

        category = classify_file(relative_path)
        if category == "vendored":
            continue
        if category == "code" and _detect_generated_header(path):
            category = "generated"

        try:
            size_bytes = path.stat().st_size
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if size_bytes > LARGE_FILE_BYTES:  # §4.1.7 large-file rule
                logger.debug(f"Skipping large file: {relative_path} ({size_bytes} bytes)")
                continue
            content_hash = _sha256_file(path)
        except OSError as exc:
            logger.warning(f"File read error on {relative_path}: {exc} — skipping")
            continue

        files.append(
            FileRecord(
                path=relative_path,
                category=category,
                language=detect_language(relative_path),
                size_bytes=size_bytes,
                content_hash=content_hash,
                last_modified=mtime,
            )
        )

    logger.info(f"Walked {len(files)} files in {repo_path}")
    return files


def read_file_content(repo_path: str | Path, rel_path: str, sanitize: bool = True) -> tuple[str, bool]:
    """Read a file with §4.1.7 invalid-encoding fallback and secret redaction."""
    path = Path(repo_path) / rel_path
    content = ""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="latin-1")
        except OSError:
            return "", False
    except OSError:
        return "", False

    if sanitize:
        content = redact_secrets(content)
    return content, True


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]  # truncated: only identity matters, not security
