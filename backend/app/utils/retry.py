"""Retry with exponential backoff (§4.1.7 error-handling tables)."""
from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def retry(
    fn: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 0.5,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    logger: Any = None,
    label: str = "operation",
) -> T:
    """Retry `fn` up to `attempts` times with exponential backoff."""
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except exceptions as exc:  # noqa: PERF203
            last_error = exc
            if logger:
                logger.warning(
                    f"{label} failed (attempt {attempt}/{attempts}): {exc}"
                )
            if attempt < attempts:
                time.sleep(base_delay * (2 ** (attempt - 1)))
    if last_error:
        raise last_error
    raise RuntimeError(f"{label}: retry loop exhausted unexpectedly")
