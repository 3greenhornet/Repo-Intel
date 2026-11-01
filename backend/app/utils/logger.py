"""Structured logging with sensitive-data redaction (§24.7 G16 fix).

FIX-046: a `request_id` contextvar is propagated into every JSON log line
and into worker threads spawned via `asyncio.to_thread` (contexts are
copied at submission time on Python 3.9+).
"""
from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

_SENSITIVE_KEYS = (
    "password", "token", "api_key", "secret", "authorization",
    "github_token", "openai_api_key", "anthropic_api_key", "pinecone_api_key",
)

_PATTERNS = [
    (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "ghp_[REDACTED]"),
    (re.compile(r"gho_[a-zA-Z0-9]{36}"), "gho_[REDACTED]"),
    (re.compile(r"sk-[a-zA-Z0-9]{48}"), "sk-[REDACTED]"),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "sk-[REDACTED]"),
    (re.compile(r"Bearer\s+[a-zA-Z0-9\-._~+/]+=*"), "Bearer [REDACTED]"),
]


def redact_value(value: Any) -> Any:
    """Redact sensitive substrings inside string values."""
    if isinstance(value, str):
        for pattern, replacement in _PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value


def redact_event(event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog-style processor: redact sensitive keys and patterns."""
    for key in list(event_dict.keys()):
        if any(s in key.lower() for s in _SENSITIVE_KEYS):
            event_dict[key] = "[REDACTED]"
        else:
            event_dict[key] = redact_value(event_dict[key])
    return event_dict


class JSONFormatter(logging.Formatter):
    """JSON log formatter with redaction + UTC timestamps (§24.11)."""

    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": get_request_id(),  # FIX-046
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            event.update(extra)
        event = redact_event(event)
        if record.exc_info:
            event["exc"] = self.formatException(record.exc_info)
        return json.dumps(event, default=str)


def get_logger(name: str = "repointel") -> logging.Logger:
    """Logger factory — JSON to stdout (LOG_FORMAT=json) or plain text."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        import os

        if os.environ.get("LOG_FORMAT", "json") == "json":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
            )
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger


def log_extra(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    logger.log(level, event, extra={"extra_fields": fields})


# ------------------------------------------------- request id (FIX-046)
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-")


def set_request_id(rid: str) -> None:
    request_id_var.set(rid)


def get_request_id() -> str:
    return request_id_var.get()
