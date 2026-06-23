"""RU: Провайдеры LLM (HTTP-клиенты) для разных платформ.

EN: LLM providers (HTTP clients) for multiple platforms.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import re
import threading
import time
from typing import Any, Protocol

import aiohttp
import requests


LOGGER = logging.getLogger("forge")

# Module-level cache: base_url -> template type ("gemma4" | "gemma3" | "qwen" | "raw")
_template_cache: dict[str, str] = {}
_template_lock = threading.Lock()


class LLMProvider(Protocol):
    """RU: Протокол, описывающий минимальный интерфейс LLM-провайдера.

    EN: Protocol describing the minimal LLM provider interface.
    """

    async def generate(self, prompt: str, *, temperature: float, timeout: int) -> str: ...


@dataclass(frozen=True)
class LlamaCppConfig:
    """RU: Конфиг для llama.cpp server API.

    EN: Config for llama.cpp server API.
    """

    url: str
    model: str

    # Retry / logging controls
    max_retries: int = 2
    log_interval_s: int = 10

    # Legacy fields kept for call-site compat; unused by native /completion path.
    watchdog_enabled: bool = True
    first_token_timeout_s: int = 120
    stall_timeout_s: int = 120
    fallback_models: tuple[str, ...] = ()


def _base_url(url: str) -> str:
    """Strip any known endpoint suffix to get the server base URL."""
    for suffix in ("/v1/chat/completions", "/v1/completions", "/completion"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url.rstrip("/")


def _completion_url(url: str) -> str:
    """Return the /completion endpoint URL regardless of what was configured."""
    return _base_url(url) + "/completion"


def _detect_template(base: str) -> str:
    """Query /props and return "gemma4" | "gemma3" | "qwen" | "raw".

    Result is cached permanently once a non-raw template is detected.
    """
    with _template_lock:
        cached = _template_cache.get(base)
        if cached and cached != "raw":
            return cached

    try:
        resp = requests.get(f"{base}/props", timeout=5)
        resp.raise_for_status()
        template = resp.json().get("chat_template", "")
    except Exception:
        template = ""

    if "<|turn>" in template:
        ttype = "gemma4"
    elif "<start_of_turn>" in template:
        ttype = "gemma3"
    elif "<|im_start|>" in template:
        ttype = "qwen"
    else:
        ttype = "raw"

    with _template_lock:
        _template_cache[base] = ttype
    return ttype


def _wrap_prompt(prompt: str, ttype: str) -> str:
    """Wrap prompt with server chat-template tokens.

    For gemma4: the <|channel>thought\\n<channel|> assistant prefill suppresses
    the reasoning/thinking block so the model starts answering immediately.
    """
    if ttype == "gemma4":
        return f"<|turn>user\n{prompt}\n<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    if ttype == "gemma3":
        return f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"
    return prompt


class LlamaCppProvider:
    """RU: Провайдер для llama.cpp /completion (native, non-streaming) через aiohttp.

    EN: Provider for llama.cpp /completion (native, non-streaming) via aiohttp.
    """

    def __init__(self, cfg: LlamaCppConfig) -> None:
        self.cfg = cfg
        self._endpoint = _completion_url(cfg.url)
        self._base = _base_url(cfg.url)

    async def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
        attempts = max(1, 1 + int(self.cfg.max_retries))
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            if attempt > 1:
                LOGGER.warning(
                    "Retrying llama.cpp (%d/%d) endpoint=%s",
                    attempt,
                    attempts,
                    self._endpoint,
                )
            try:
                return await self._call(
                    prompt=prompt,
                    temperature=temperature,
                    max_total_s=int(timeout),
                )
            except _Retryable as exc:
                last_exc = exc.cause
                LOGGER.warning("llama.cpp retryable error (attempt %d): %s", attempt, exc)
                continue
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                LOGGER.warning("llama.cpp connection error (attempt %d): %s", attempt, exc)
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("llama.cpp generate failed without a captured exception")

    async def _call(self, *, prompt: str, temperature: float, max_total_s: int) -> str:
        ttype = _detect_template(self._base)
        wrapped = _wrap_prompt(prompt, ttype)

        payload: dict[str, Any] = {
            "prompt": wrapped,
            "stream": False,
            "temperature": float(temperature),
            "n_predict": 2048,
            "json_schema": {"type": "object"},
        }

        start = time.monotonic()

        timeout_obj = aiohttp.ClientTimeout(total=max_total_s if max_total_s > 0 else None)
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.post(self._endpoint, json=payload) as r:
                if r.status == 503:
                    await asyncio.sleep(30)
                    raise _Retryable(RuntimeError("503: Model loading"))

                if r.status >= 400:
                    text = await r.text()
                    preview = text[:500]
                    LOGGER.error(
                        "llama.cpp HTTP %d at %s -- body: %s",
                        r.status, self._endpoint, preview,
                    )
                    raise aiohttp.ClientResponseError(
                        r.request_info,
                        r.history,
                        status=r.status,
                        message=f"HTTP {r.status}",
                    )

                data = await r.json()

        text = str(
            data.get("content")
            or data.get("response")
            or data.get("message", {}).get("content")
            or ""
        ).strip()

        # Strip any leaked <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        if not text:
            raise RuntimeError("llama.cpp returned an empty response")

        elapsed = time.monotonic() - start
        LOGGER.info(
            "llama.cpp ok: endpoint=%s elapsed=%.1fs chars=%d",
            self._endpoint,
            elapsed,
            len(text),
        )
        return text


class _Retryable(Exception):
    """Internal signal: this error is safe to retry."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


# -- Cloud providers (legacy compat paths, wrapped in asyncio.to_thread) --


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    model: str


class OpenAIProvider:
    def __init__(self, cfg: OpenAIConfig) -> None:
        self.cfg = cfg

    async def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
        def _sync_call() -> str:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.cfg.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.cfg.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that always responds with valid JSON. Never include explanatory text outside the JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                },
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        return await asyncio.to_thread(_sync_call)


@dataclass(frozen=True)
class AnthropicConfig:
    api_key: str
    model: str


class AnthropicProvider:
    def __init__(self, cfg: AnthropicConfig) -> None:
        self.cfg = cfg

    async def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
        def _sync_call() -> str:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.cfg.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.cfg.model,
                    "max_tokens": 4096,
                    "temperature": temperature,
                    "system": "You are a helpful assistant that always responds with valid JSON. Never include explanatory text outside the JSON.",
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            parts = data.get("content") or []
            if parts and isinstance(parts, list) and isinstance(parts[0], dict):
                return parts[0].get("text", "")
            return str(data)

        return await asyncio.to_thread(_sync_call)


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str


class GeminiProvider:
    def __init__(self, cfg: GeminiConfig) -> None:
        self.cfg = cfg

    async def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
        def _sync_call() -> str:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.cfg.model}:generateContent"
            r = requests.post(
                url,
                params={"key": self.cfg.api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature},
                },
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            cands = data.get("candidates") or []
            if cands:
                content = cands[0].get("content") or {}
                parts = content.get("parts") or []
                if parts:
                    return parts[0].get("text", "")
            return str(data)

        return await asyncio.to_thread(_sync_call)
