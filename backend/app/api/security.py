"""Security middleware — rate limiting (§24.1) + JWT (§24.2).

Token-bucket rate limiter with a documented Redis backend and an in-memory
fallback (no Redis in this environment). JWT helpers implement the documented
HS256 path; RS256 keys can be dropped in for production.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import Request

from app.utils.logger import get_logger

logger = get_logger(__name__)


class InMemoryTokenBucket:
    """§24.1 TokenBucketRateLimiter — in-memory backend (single node)."""

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._state: dict[str, tuple[float, float]] = {}  # key → (tokens, last_refill)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        tokens, last = self._state.get(key, (float(self.capacity), now))
        elapsed = now - last
        tokens = min(self.capacity, tokens + elapsed * self.refill_rate)
        if tokens >= 1:
            tokens -= 1
            self._state[key] = (tokens, now)
            return True
        self._state[key] = (tokens, now)
        return False

    def prune(self, max_age: float = 3600) -> None:
        now = time.time()
        stale = [k for k, (_, t) in self._state.items() if now - t > max_age]
        for k in stale:
            self._state.pop(k, None)


class RedisTokenBucket(InMemoryTokenBucket):
    """§24.1 Redis-backed bucket — used when Redis is reachable."""

    def __init__(self, capacity: int, refill_rate: float) -> None:
        super().__init__(capacity, refill_rate)
        import redis

        from app.config import get_settings

        settings = get_settings()
        self.redis = redis.Redis(host=settings.redis_host, port=settings.redis_port)
        self.redis.ping()

    def is_allowed(self, key: str) -> bool:
        redis_key = f"ratelimit:{key}"
        pipe = self.redis.pipeline()
        pipe.hgetall(redis_key)
        state = pipe.execute()[0]
        now = time.time()
        if not state:
            self.redis.hset(redis_key, mapping={
                "tokens": self.capacity - 1, "last_refill": now})
            self.redis.expire(redis_key, 3600)
            return True
        tokens = float(state.get(b"tokens", self.capacity))
        last_refill = float(state.get(b"last_refill", now))
        elapsed = now - last_refill
        tokens = min(self.capacity, tokens + elapsed * self.refill_rate)
        if tokens >= 1:
            tokens -= 1
            self.redis.hset(redis_key, mapping={"tokens": tokens, "last_refill": now})
            return True
        return False


def get_rate_limiter():
    """Factory (FIX-006c): honor the operator's explicit CACHE_BACKEND choice.

    - CACHE_BACKEND=memory → always in-memory, regardless of Redis availability
    - CACHE_BACKEND=redis  → Redis, and a RuntimeError if Redis is unreachable
      (a silent downgrade would mask a misconfigured production deployment)
    """
    from app.config import get_settings

    settings = get_settings()

    if settings.cache_backend == "memory":
        logger.info("Rate limiter: in-memory backend (CACHE_BACKEND=memory)")
        return InMemoryTokenBucket(settings.rate_limit_capacity, settings.rate_limit_refill)

    if settings.cache_backend == "redis":
        try:
            return RedisTokenBucket(settings.rate_limit_capacity, settings.rate_limit_refill)
        except Exception as exc:
            logger.error(f"Redis backend requested but unavailable: {exc}")
            raise RuntimeError(
                "CACHE_BACKEND=redis but Redis is unreachable. "
                "Set CACHE_BACKEND=memory or fix Redis connectivity.") from exc

    raise ValueError(f"Unknown CACHE_BACKEND: {settings.cache_backend!r}")


# ------------------------------------------------- JWT auth dependency (FIX-030)
async def require_auth(request: Request) -> dict[str, Any]:
    """FastAPI dependency that validates the bearer token when REQUIRE_AUTH=1.

    Returns the decoded token payload. In dev mode (REQUIRE_AUTH=0), returns
    a synthetic payload identifying the request as anonymous.
    """
    from app.config import get_settings

    settings = get_settings()

    if not settings.require_auth:
        return {"sub": "anonymous", "type": "dev"}

    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = auth[7:].strip()
    return verify_token(token, expected_type="access")


# ------------------------------------------------------------------------ JWT (§24.2)
def create_token(user_id: str, token_type: str = "access",
                 expires_seconds: int | None = None) -> str:
    """HS256 JWT with documented claims (sub/iat/nbf/exp/iss/aud/type)."""
    import time as _time
    import uuid

    import jwt

    from app.config import get_settings

    settings = get_settings()
    now = _time.time()
    default_exp = 15 * 60 if token_type == "access" else 7 * 24 * 3600
    payload = {
        "sub": user_id, "iat": now, "nbf": now,
        "exp": now + (expires_seconds or default_exp),
        "iss": "repointel-ai", "aud": "repointel-api",
        "type": token_type,
    }
    if token_type == "refresh":
        payload["jti"] = str(uuid.uuid4())
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def verify_token(token: str, expected_type: str = "access") -> dict[str, Any]:
    """§24.2 verify: exp/nbf/iss/aud + algorithm pinning."""
    import jwt

    from app.config import get_settings
    from fastapi import HTTPException

    try:
        payload = jwt.decode(
            token, get_settings().jwt_secret, algorithms=["HS256"],
            issuer="repointel-ai", audience="repointel-api")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")
    if payload.get("type") != expected_type:
        raise HTTPException(status_code=401, detail=f"Expected {expected_type} token")
    return payload
