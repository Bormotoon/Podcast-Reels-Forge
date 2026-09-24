"""RU: Провайдеры LLM (HTTP-клиенты) для разных платформ.

EN: LLM providers (HTTP clients) for multiple platforms.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import random
import re
import threading
import time
from typing import Any, Mapping, Protocol

import aiohttp
import requests

from podcast_reels_forge.llm.schemas import ANY_OBJECT_SCHEMA


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

    # Maximum tokens to generate. Must leave room for the response inside the
    # server ctx_size; too low truncates the JSON mid-token and breaks parsing.
    n_predict: int = 4096

    # Response schema llama.cpp turns into a sampling grammar. None keeps the
    # permissive "any object" schema.
    json_schema: Mapping[str, Any] | None = None

    # Simplified schema tried when the server rejects `json_schema`, before
    # giving up on structure altogether. None skips straight to "any object".
    fallback_schema: Mapping[str, Any] | None = None

    # Whether to constrain sampling to JSON at all. Stages that want prose (the
    # article editor) must turn this off: the grammar makes the model emit JSON
    # no matter what the prompt asks for, so a markdown request comes back empty.
    json_output: bool = True

    # Retry / logging controls
    max_retries: int = 2
    log_interval_s: int = 10
    # Exponential backoff between transport retries:
    #   delay = min(retry_max_delay_s, retry_base_delay_s * 2**n) + jitter
    # A 503 (model still loading) uses a longer floor.
    retry_base_delay_s: float = 2.0
    retry_max_delay_s: float = 30.0

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


# Schema downgrade levels: the configured schema, the stage's simplified
# fallback, then the permissive "any object".
SCHEMA_FULL = 0
SCHEMA_FALLBACK = 1
SCHEMA_ANY = 2


def build_completion_payload(
    cfg: LlamaCppConfig,
    wrapped_prompt: str,
    *,
    temperature: float,
    schema_downgraded: bool = False,
    schema_level: int = SCHEMA_FULL,
) -> dict[str, Any]:
    """RU: Собирает тело запроса к /completion.

    EN: Build the /completion request body. Split out from the request itself
    so the schema wiring is testable without a server. ``schema_downgraded``
    is the legacy switch and means "any object".
    """

    payload: dict[str, Any] = {
        "prompt": wrapped_prompt,
        "stream": False,
        "temperature": float(temperature),
        "n_predict": int(cfg.n_predict),
    }
    if not cfg.json_output:
        return payload

    level = SCHEMA_ANY if schema_downgraded else int(schema_level)
    schema: Mapping[str, Any] = cfg.json_schema or ANY_OBJECT_SCHEMA
    if level == SCHEMA_FALLBACK:
        schema = cfg.fallback_schema or ANY_OBJECT_SCHEMA
    elif level >= SCHEMA_ANY:
        schema = ANY_OBJECT_SCHEMA
    payload["json_schema"] = dict(schema)
    return payload


def retry_delay(
    attempt: int,
    *,
    base_s: float,
    max_s: float,
    jitter: float = 0.25,
    rng: random.Random | None = None,
) -> float:
    """Exponential backoff with proportional jitter for retry ``attempt`` (1-based).

    Jitter keeps parallel scout requests that failed together from hammering
    a recovering llama.cpp server in lockstep.
    """

    if base_s <= 0:
        return 0.0
    delay = min(max_s, base_s * (2 ** max(0, attempt - 1)))
    spread = delay * max(0.0, jitter)
    return max(0.0, delay + (rng or random).uniform(-spread, spread))


class LlmHttpError(RuntimeError):
    """A non-retryable HTTP answer from the LLM server (e.g. 400, 404)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


# 4xx answers worth retrying: request timeout and rate limiting.
_RETRYABLE_4XX = frozenset({408, 429})


def _looks_like_schema_rejection(body: str) -> bool:
    """Whether an HTTP 400 body blames the json_schema/grammar."""

    lowered = body.lower()
    return any(token in lowered for token in ("schema", "grammar", "gbnf"))


class LlamaCppProvider:
    """RU: Провайдер для llama.cpp /completion (native, non-streaming) через aiohttp.

    EN: Provider for llama.cpp /completion (native, non-streaming) via aiohttp.

    One ``aiohttp.ClientSession`` is kept for the provider's lifetime, so a
    run pays the connection setup once instead of once per request. Call
    :meth:`aclose` when done; a session left over from a finished event loop
    is replaced transparently.
    """

    def __init__(self, cfg: LlamaCppConfig) -> None:
        self.cfg = cfg
        self._endpoint = _completion_url(cfg.url)
        self._base = _base_url(cfg.url)
        # Raised one level each time the server rejects the current schema, so
        # a downgrade costs one failed request per provider, not one per call.
        self._schema_level = SCHEMA_FULL
        self._session: aiohttp.ClientSession | None = None
        self._session_loop: asyncio.AbstractEventLoop | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        if (
            self._session is None
            or self._session.closed
            or self._session_loop is not loop
        ):
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=8),
            )
            self._session_loop = loop
        return self._session

    async def aclose(self) -> None:
        """Close the pooled HTTP session, if one is open on this loop."""

        session, self._session = self._session, None
        if session is not None and not session.closed:
            try:
                await session.close()
            except RuntimeError:
                # Bound to an event loop that is already gone; nothing to do.
                pass

    def _next_schema_level(self) -> int:
        if self._schema_level == SCHEMA_FULL and self.cfg.fallback_schema is not None:
            return SCHEMA_FALLBACK
        return SCHEMA_ANY

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
                delay_floor = exc.min_delay_s
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                LOGGER.warning("llama.cpp connection error (attempt %d): %s", attempt, exc)
                delay_floor = None
            if attempt < attempts and delay_floor is not None and delay_floor <= 0:
                # A schema downgrade retries immediately: nothing is overloaded.
                continue
            if attempt < attempts:
                base = self.cfg.retry_base_delay_s
                cap = self.cfg.retry_max_delay_s
                if delay_floor:
                    base = max(base, delay_floor)
                    cap = max(cap, delay_floor * 4)
                await asyncio.sleep(retry_delay(attempt, base_s=base, max_s=cap))

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("llama.cpp generate failed without a captured exception")

    async def _call(self, *, prompt: str, temperature: float, max_total_s: int) -> str:
        ttype = _detect_template(self._base)
        wrapped = _wrap_prompt(prompt, ttype)

        payload = build_completion_payload(
            self.cfg,
            wrapped,
            temperature=temperature,
            schema_level=self._schema_level,
        )

        start = time.monotonic()

        timeout_obj = aiohttp.ClientTimeout(total=max_total_s if max_total_s > 0 else None)
        session = await self._get_session()
        async with session.post(self._endpoint, json=payload, timeout=timeout_obj) as r:
            if r.status == 503:
                # Model still loading: back off with a longer floor.
                raise _Retryable(RuntimeError("503: Model loading"), min_delay_s=5.0)

            if r.status >= 400:
                text = await r.text()
                preview = text[:500]
                # Older llama.cpp builds reject anything beyond a trivial
                # json_schema. Step down to the stage's simplified schema,
                # then to the permissive one, rather than failing the stage.
                if (
                    r.status == 400
                    and self._schema_level < SCHEMA_ANY
                    and self.cfg.json_output
                    and self.cfg.json_schema is not None
                    and _looks_like_schema_rejection(text)
                ):
                    self._schema_level = self._next_schema_level()
                    LOGGER.warning(
                        "llama.cpp rejected the response schema at %s; "
                        "falling back to a %s schema -- body: %s",
                        self._endpoint,
                        "simplified" if self._schema_level == SCHEMA_FALLBACK else "permissive",
                        preview,
                    )
                    raise _Retryable(RuntimeError("400: schema rejected"), min_delay_s=0.0)

                LOGGER.error(
                    "llama.cpp HTTP %d at %s -- body: %s",
                    r.status, self._endpoint, preview,
                )
                if 400 <= r.status < 500 and r.status not in _RETRYABLE_4XX:
                    # The same request will fail the same way; retrying only
                    # delays the stage.
                    raise LlmHttpError(r.status, f"HTTP {r.status}: {preview[:200]}")
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

        # llama.cpp sets stopped_limit=True when it hit n_predict before the
        # model was done, so the JSON tail is cut off. The JSON salvage in
        # extract_first_json_value recovers every complete item, and the
        # pipeline over-generates then filters down anyway. Log at INFO so it
        # is visible under --verbose for tuning but does not read as an error.
        if data.get("stopped_limit") or data.get("truncated"):
            LOGGER.info(
                "llama.cpp output reached the n_predict=%d token budget and was "
                "truncated; complete JSON items are still recovered "
                "(endpoint=%s, chars=%d)",
                int(self.cfg.n_predict),
                self._endpoint,
                len(text),
            )

        return text


class _Retryable(Exception):
    """Internal signal: this error is safe to retry.

    ``min_delay_s`` sets the backoff floor: 0 retries immediately, None uses
    the configured backoff as is.
    """

    def __init__(self, cause: Exception, *, min_delay_s: float | None = None) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.min_delay_s = min_delay_s


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


async def close_provider(provider: object) -> None:
    """Release a provider's pooled resources, if it holds any."""

    aclose = getattr(provider, "aclose", None)
    if aclose is not None:
        await aclose()
