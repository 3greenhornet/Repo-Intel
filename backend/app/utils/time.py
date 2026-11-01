"""Timezone-aware UTC helpers (FIX-041).

`datetime.utcnow()` is deprecated since Python 3.12 and returns a naive
datetime; every timestamp in RepoIntel is now timezone-aware UTC.
"""
from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Current UTC time as an ISO 8601 string with timezone offset."""
    return utc_now().isoformat()


def to_utc(dt: datetime) -> datetime:
    """Coerce any datetime to timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
