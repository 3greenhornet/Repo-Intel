"""Component 7A: LLM Client (§10.1) — unified multi-provider interface.

Providers:
- ``zai``     (default here): bridges to the z-ai CLI (glm-4-plus) — works in
  this environment with no external keys.
- ``openai``  / ``anthropic``: documented clients, activated by API key env.
- ``local``: any OpenAI-compatible endpoint via OPENAI_BASE_URL (vLLM/Ollama).

All providers implement `invoke(prompt, system_prompt) -> str` with the
documented retry/temperature/max_tokens semantics.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMError(Exception):
    pass


class BaseProvider:
    name = "base"
    supports_streaming = False  # FIX-028

    def invoke(self, prompt: str, system_prompt: str | None = None,
               max_tokens: int | None = None) -> str:
        raise NotImplementedError

    async def astream(self, prompt: str, system_prompt: str | None = None,
                      max_tokens: int | None = None):
        """Yield tokens as they arrive. Only streaming providers implement this."""
        raise NotImplementedError("Provider does not support streaming")


class ZaiProvider(BaseProvider):
    """Bridge to `z-ai chat` CLI — JSON output parsed from a temp file.

    Handles upstream 429 rate limits with patient exponential backoff
    (5s → 15s → 45s) since batch evaluation fires many calls in succession.

    FIX-016: every subprocess call is bounded by a semaphore so concurrent
    HTTP requests cannot pin the FastAPI threadpool with dozens of blocking
    z-ai processes (max default 4, ZAI_MAX_CONCURRENT). `ainvoke` keeps the
    backoff sleeps off the event loop.
    """

    name = "zai"
    TIMEOUT = 180
    BACKOFFS = (5.0, 15.0, 45.0)

    def __init__(self, model: str = "glm-4-plus") -> None:
        self.model = model
        self._semaphore = threading.Semaphore(get_settings().zai_max_concurrent)

    def _call(self, cmd: list[str], out_path: str) -> str:
        with self._semaphore:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.TIMEOUT)
        if result.returncode != 0:
            raise LLMError(
                f"z-ai chat failed ({result.returncode}): {result.stderr[-300:]}")
        data = json.loads(Path(out_path).read_text(encoding="utf-8"))
        content = data["choices"][0]["message"]["content"]
        if not content or not content.strip():
            raise LLMError("Empty response from z-ai")
        return content

    def invoke(self, prompt: str, system_prompt: str | None = None,
               max_tokens: int | None = None) -> str:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out_path = tmp.name
        cmd = ["z-ai", "chat", "-p", prompt, "-o", out_path]
        if system_prompt:
            cmd.extend(["-s", system_prompt])
        try:
            last_error: Exception | None = None
            for attempt in range(len(self.BACKOFFS) + 1):
                try:
                    return self._call(cmd, out_path)
                except (LLMError, subprocess.TimeoutExpired) as exc:
                    last_error = exc
                    msg = str(exc)
                    is_rate_limit = "429" in msg or "Too many requests" in msg
                    if attempt < len(self.BACKOFFS):
                        delay = self.BACKOFFS[attempt] if is_rate_limit else min(attempt + 1, 2.0)
                        logger.warning(
                            f"z-ai call failed (attempt {attempt + 1}), "
                            f"backing off {delay}s (rate_limit={is_rate_limit})")
                        time.sleep(delay)
            raise last_error or LLMError("z-ai failed after retries")
        finally:
            Path(out_path).unlink(missing_ok=True)

    async def ainvoke(self, prompt: str, system_prompt: str | None = None,
                      max_tokens: int | None = None) -> str:
        """FIX-016: async wrapper — blocking subprocess runs in the threadpool
        (bounded by the semaphore); backoff sleeps are non-blocking."""
        import asyncio

        for attempt in range(len(self.BACKOFFS) + 1):
            try:
                return await asyncio.to_thread(self.invoke, prompt, system_prompt, max_tokens)
            except (LLMError, subprocess.TimeoutExpired) as exc:
                msg = str(exc)
                is_rate_limit = "429" in msg or "Too many requests" in msg
                if attempt < len(self.BACKOFFS):
                    delay = self.BACKOFFS[attempt] if is_rate_limit else min(attempt + 1, 2.0)
                    logger.warning(f"z-ai ainvoke failed (attempt {attempt + 1}), sleeping {delay}s")
                    await asyncio.sleep(delay)
                else:
                    raise


class OpenAIProvider(BaseProvider):
    """Documented OpenAI / Groq path (§10.1.3). Requires OPENAI_API_KEY or GROQ_API_KEY."""

    name = "openai"
    supports_streaming = True  # FIX-028

    def __init__(self, model: str = "gpt-4o") -> None:
        import httpx

        self.settings = get_settings()
        self.model = model
        self._client = httpx.Client(
            base_url=self._base_url(),
            headers={"Authorization": f"Bearer {self._api_key()}"},
            timeout=180)

    def _api_key(self) -> str:
        import os

        key = os.environ.get("OPENAI_API_KEY", self.settings.openai_api_key)
        if not key:
            key = os.environ.get("GROQ_API_KEY", getattr(self.settings, "groq_api_key", ""))
        if not key:
            raise LLMError("OPENAI_API_KEY or GROQ_API_KEY not set")
        return key

    def _base_url(self) -> str:
        import os

        if "OPENAI_BASE_URL" in os.environ:
            return os.environ["OPENAI_BASE_URL"].rstrip("/")
        if os.environ.get("GROQ_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            return "https://api.groq.com/openai/v1"
        return "https://api.openai.com/v1"

    def invoke(self, prompt: str, system_prompt: str | None = None,
               max_tokens: int | None = None) -> str:
        import random
        import time

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        models_to_try = [self.model]
        if "120b" in self.model:
            models_to_try.append(self.model.replace("120b", "20b"))
        elif "20b" not in self.model and "gpt-oss" not in self.model:
            models_to_try.append("openai/gpt-oss-20b")

        resp = None
        for model in models_to_try:
            for attempt in range(4):
                try:
                    resp = self._client.post("/chat/completions", json={
                        "model": model, "messages": messages,
                        "temperature": self.settings.llm_temperature,
                        "max_tokens": max_tokens or self.settings.llm_max_tokens,
                    })
                except Exception as exc:
                    logger.warning("HTTP post error: %s", exc)
                    time.sleep(2)
                    continue
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                if resp.status_code == 429:
                    retry_after = resp.headers.get("retry-after")
                    if retry_after:
                        try:
                            sleep_time = min(float(retry_after) + random.uniform(0.5, 1.5), 30.0)
                        except ValueError:
                            sleep_time = min(2.0 * (2 ** attempt) + random.uniform(0.5, 2.0), 30.0)
                    else:
                        sleep_time = min(2.0 * (2 ** attempt) + random.uniform(0.5, 2.0), 30.0)
                    logger.warning(f"Rate limited (429) on {model}, sleeping {sleep_time:.1f}s (attempt {attempt+1})...")
                    time.sleep(sleep_time)
                    # If this is the larger model and it's already failed twice, immediately pivot to the 20b fallback model
                    if attempt >= 1 and len(models_to_try) > 1 and model != models_to_try[-1]:
                        logger.info(f"Flipping to fallback model: {models_to_try[-1]}")
                        break
                    continue
                break

        if resp is not None and resp.status_code != 200:
            raise LLMError(f"OpenAI API {resp.status_code}: {resp.text[:200]}")
        raise LLMError("OpenAI invocation failed after retries")

    async def astream(self, prompt: str, system_prompt: str | None = None,
                      max_tokens: int | None = None):
        """FIX-028: real token streaming over SSE lines with fallback on 429."""
        import httpx

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        models_to_try = [self.model]
        if "120b" in self.model:
            models_to_try.append(self.model.replace("120b", "20b"))

        for model in models_to_try:
            payload = {
                "model": model, "messages": messages,
                "temperature": self.settings.llm_temperature,
                "max_tokens": max_tokens or self.settings.llm_max_tokens,
                "stream": True,
            }
            async with httpx.AsyncClient(
                base_url=self._base_url(),
                headers={"Authorization": f"Bearer {self._api_key()}"},
                timeout=180.0,
            ) as client:
                async with client.stream("POST", "/chat/completions", json=payload) as resp:
                    if resp.status_code == 429 and model != models_to_try[-1]:
                        logger.warning(f"astream rate limited on {model}, falling back to {models_to_try[-1]}")
                        continue
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise LLMError(f"OpenAI {resp.status_code}: {body[:200]!r}")
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        delta = event["choices"][0].get("delta", {}) if event.get("choices") else {}
                        content = delta.get("content")
                        if content:
                            yield content
                    return


class AnthropicProvider(BaseProvider):
    """Documented Anthropic path (§10.1.3). Requires ANTHROPIC_API_KEY."""

    name = "anthropic"
    supports_streaming = True  # FIX-028

    def __init__(self, model: str = "claude-sonnet-4-latest") -> None:
        import httpx

        self.settings = get_settings()
        self.model = model
        self._client = httpx.Client(
            base_url="https://api.anthropic.com/v1",
            headers={"x-api-key": self._require_key(),
                     "anthropic-version": "2023-06-01"},
            timeout=180)

    def _require_key(self) -> str:
        import os

        key = os.environ.get("ANTHROPIC_API_KEY", self.settings.anthropic_api_key)
        if not key:
            raise LLMError("ANTHROPIC_API_KEY not set")
        return key

    def invoke(self, prompt: str, system_prompt: str | None = None,
               max_tokens: int | None = None) -> str:
        body = {
            "model": self.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.llm_temperature,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
        }
        if system_prompt:
            # FIX-033: mark the system prompt as cacheable — prompt-prefix
            # stability plus cache_control enables provider-side prompt caching.
            body["system"] = [{"type": "text", "text": system_prompt,
                               "cache_control": {"type": "ephemeral"}}]
        resp = self._client.post("/messages", json=body)
        if resp.status_code != 200:
            raise LLMError(f"Anthropic API {resp.status_code}: {resp.text[:200]}")
        return resp.json()["content"][0]["text"]

    async def astream(self, prompt: str, system_prompt: str | None = None,
                      max_tokens: int | None = None):
        """FIX-028: real token streaming over Anthropic SSE events."""
        import httpx

        body = {
            "model": self.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.llm_temperature,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "stream": True,
        }
        if system_prompt:
            body["system"] = system_prompt
        async with httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": self._require_key(),
                "anthropic-version": "2023-06-01",
            },
            timeout=180.0,
        ) as client:
            async with client.stream("POST", "/messages", json=body) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    raise LLMError(f"Anthropic {resp.status_code}: {err[:200]!r}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")


class LitellmProvider(BaseProvider):
    """Multi-provider resilient LLM client powered by LiteLLM with automatic fallback chain."""
    name = "litellm"
    supports_streaming = True

    DEFAULT_CHAIN = [
        "openrouter/meta-llama/llama-3.3-70b-instruct",
        "groq/openai/gpt-oss-120b",
        "groq/openai/gpt-oss-20b",
        "gemini/gemini-2.5-flash",
        "mistral/mistral-small-latest",
    ]

    def __init__(self, model: str = "") -> None:
        try:
            import litellm
            self._litellm = litellm
            # Suppress noisy LiteLLM logs
            self._litellm.suppress_debug_info = True
            import os
            # Sync GOOGLE_API_KEY to GEMINI_API_KEY if needed
            if not os.getenv("GEMINI_API_KEY") and os.getenv("GOOGLE_API_KEY"):
                os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
            # Etch Detail 1: Enable OpenTelemetry native integration if configured
            if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv("OTEL_SERVICE_NAME"):
                try:
                    self._litellm.success_callback = ["otel"]
                    self._litellm.failure_callback = ["otel"]
                except Exception:
                    pass
        except ImportError:
            raise LLMError("litellm is not installed; install litellm or use LLM_PROVIDER=openai")
        chain = [m.strip() for m in model.split(",") if m.strip()] if model else self.DEFAULT_CHAIN
        self.chain = [m for m in chain if self._key_present(m)]
        if not self.chain:
            raise LLMError(
                "No LLM provider configured. Set one of: GROQ_API_KEY, "
                "GEMINI_API_KEY, OPENROUTER_API_KEY, MISTRAL_API_KEY, OPENAI_API_KEY."
            )
        logger.info(f"LiteLLM initialized with active fallback chain: {self.chain}")

    @staticmethod
    def _key_present(model: str) -> bool:
        import os
        provider = model.split("/", 1)[0]
        if provider == "gemini":
            return bool(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip())
        env_map = {
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        return bool(os.getenv(env_map.get(provider, ""), "").strip())

    def invoke(self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Try with native LiteLLM fallbacks first
        try:
            resp = self._litellm.completion(
                model=self.chain[0],
                messages=messages,
                fallbacks=self.chain[1:] if len(self.chain) > 1 else None,
                temperature=get_settings().llm_temperature,
                max_tokens=max_tokens or get_settings().llm_max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            logger.warning(f"LiteLLM completion with fallbacks failed: {exc}, attempting manual chain iteration...")
            last_err = exc
            for model in self.chain:
                try:
                    resp = self._litellm.completion(
                        model=model,
                        messages=messages,
                        temperature=get_settings().llm_temperature,
                        max_tokens=max_tokens or get_settings().llm_max_tokens,
                    )
                    return resp.choices[0].message.content
                except Exception as inner_exc:
                    last_err = inner_exc
                    continue
            raise LLMError(f"All models failed in fallback chain. Last error: {last_err}")

    async def astream(self, prompt: str, system_prompt: str | None = None,
                      max_tokens: int | None = None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._litellm.acompletion(
                model=self.chain[0],
                messages=messages,
                fallbacks=self.chain[1:] if len(self.chain) > 1 else None,
                temperature=get_settings().llm_temperature,
                max_tokens=max_tokens or get_settings().llm_max_tokens,
                stream=True,
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield delta
            return
        except Exception as exc:
            logger.warning(f"LiteLLM astream with fallbacks failed: {exc}, trying sequential model stream...")

        # Fallback to streaming through chain sequentially
        for model in self.chain[1:]:
            try:
                response = await self._litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=get_settings().llm_temperature,
                    max_tokens=max_tokens or get_settings().llm_max_tokens,
                    stream=True,
                )
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            yield delta
                return
            except Exception:
                continue

        # Last resort fallback: invoke synchronously and yield tokens
        import asyncio
        answer = await asyncio.to_thread(self.invoke, prompt, system_prompt, max_tokens)
        for part in answer.split(" "):
            yield part + " "


class LLMClient:
    """§10.1 unified client. Provider chosen from config; lazy fallbacks."""

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        self.settings = get_settings()
        self.provider_name = provider or self.settings.llm_provider
        self.model = model or self.settings.llm_model
        self._provider: BaseProvider | None = None
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def _get_provider(self) -> BaseProvider:
        if self._provider is not None:
            return self._provider
        candidates = [self.provider_name]
        for fallback in ("openai", "litellm", "zai"):
            if fallback not in candidates:
                candidates.append(fallback)
        for name in candidates:
            try:
                if name == "zai":
                    self._provider = ZaiProvider(self.model)
                elif name == "openai":
                    self._provider = OpenAIProvider(self.model)
                elif name == "anthropic":
                    self._provider = AnthropicProvider(self.model)
                elif name == "litellm":
                    self._provider = LitellmProvider(self.model)
                else:
                    continue
                return self._provider
            except LLMError as exc:
                logger.warning(f"Provider {name} unavailable: {exc}")
        raise LLMError("No LLM provider available — set OPENAI_API_KEY / GROQ_API_KEY / ANTHROPIC_API_KEY")

    def invoke(self, prompt: str, system_prompt: str | None = None,
               max_tokens: int | None = None) -> str:
        """§10.1.3 invoke — with token accounting (providers retry internally)."""
        provider = self._get_provider()
        self.calls += 1
        self.tokens_in += len(prompt) // 4
        started = time.monotonic()  # FIX-042
        try:
            content = provider.invoke(prompt, system_prompt, max_tokens)
            self.tokens_out += len(content) // 4
            logger.info("llm_invoke", extra={"extra_fields": {
                "provider": provider.name, "seconds": round(time.monotonic() - started, 2),
                "out_chars": len(content)}})
            return content
        except Exception as exc:
            logger.error("llm_invoke_failed", extra={"extra_fields": {
                "provider": provider.name, "error": str(exc)[:200]}})
            raise

    async def ainvoke(self, prompt: str, system_prompt: str | None = None,
                      max_tokens: int | None = None) -> str:
        """FIX-016: async path for FastAPI endpoints."""
        import asyncio

        provider = self._get_provider()
        self.calls += 1
        self.tokens_in += len(prompt) // 4
        started = time.monotonic()
        if hasattr(provider, "ainvoke"):
            content = await provider.ainvoke(prompt, system_prompt, max_tokens)
        else:
            content = await asyncio.to_thread(provider.invoke, prompt, system_prompt, max_tokens)
        self.tokens_out += len(content) // 4
        logger.info("llm_ainvoke", extra={"extra_fields": {
            "provider": provider.name, "seconds": round(time.monotonic() - started, 2),
            "out_chars": len(content)}})
        return content

    async def astream(self, prompt: str, system_prompt: str | None = None,
                      max_tokens: int | None = None):
        """FIX-028: real token streaming when the provider supports it;
        falls back to sentence-chunked pseudo-streaming otherwise (z-ai)."""
        import asyncio
        import re

        provider = self._get_provider()
        if not getattr(provider, "supports_streaming", False):
            answer = await asyncio.to_thread(provider.invoke, prompt, system_prompt, max_tokens)
            for part in re.split(r"(?<=[.!?:])\s+", answer):
                if part.strip():
                    yield part.strip() + " "
                    await asyncio.sleep(0.01)
            return
        self.calls += 1
        self.tokens_in += len(prompt) // 4
        async for token in provider.astream(prompt, system_prompt, max_tokens):
            self.tokens_out += max(1, len(token) // 4)
            yield token

    def stats(self) -> dict:
        return {"provider": self.provider_name, "calls": self.calls,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out}
