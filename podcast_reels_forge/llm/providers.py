"""RU: Провайдеры LLM (HTTP-клиенты) для разных платформ.

EN: LLM providers (HTTP clients) for multiple platforms.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
from typing import Protocol

import requests
from requests import exceptions as requests_exceptions


LOGGER = logging.getLogger("forge")


class LLMProvider(Protocol):
    """RU: Протокол, описывающий минимальный интерфейс LLM-провайдера.

    EN: Protocol describing the minimal LLM provider interface.
    """

    def generate(self, prompt: str, *, temperature: float, timeout: int) -> str: ...


@dataclass(frozen=True)
class OllamaConfig:
    """RU: Конфиг для Ollama API.

    EN: Config for Ollama API.
    """

    url: str
    model: str

    # Watchdog / retry controls (optional)
    watchdog_enabled: bool = True
    first_token_timeout_s: int = 120
    stall_timeout_s: int = 120
    log_interval_s: int = 10
    max_retries: int = 2
    fallback_models: tuple[str, ...] = ()


class OllamaWatchdogTriggered(RuntimeError):
    """Raised when an Ollama request appears stalled or too slow."""

    def __init__(self, *, reason: str, model: str, elapsed_s: float) -> None:
        super().__init__(
            f"Ollama watchdog triggered ({reason}) for model '{model}' after {elapsed_s:.1f}s",
        )
        self.reason = reason
        self.model = model
        self.elapsed_s = elapsed_s


class OllamaProvider:
    """RU: Провайдер для Ollama (/api/generate).

    EN: Provider for Ollama (/api/generate).
    """

    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg

    def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
        # RU: Для локальных моделей важно видеть прогресс и уметь отменять.
        # EN: For local models we want progress + the ability to abort on stalls.

        models: list[str] = [self.cfg.model]
        models += [m for m in self.cfg.fallback_models if m and m.strip()]

        # Total attempts = 1 + max_retries, but we also try each fallback model in order.
        total_attempts = max(1, 1 + int(self.cfg.max_retries))
        attempt_models: list[str] = []
        for m in models:
            if len(attempt_models) >= total_attempts:
                break
            attempt_models.append(m)
        while len(attempt_models) < total_attempts:
            attempt_models.append(attempt_models[-1])

        last_exc: Exception | None = None
        for attempt_idx, model in enumerate(attempt_models, 1):
            try:
                if attempt_idx > 1:
                    LOGGER.warning(
                        "Retrying Ollama (%d/%d) with model=%s",
                        attempt_idx,
                        len(attempt_models),
                        model,
                    )
                return self._generate_streaming(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_total_s=int(timeout),
                )
            except OllamaWatchdogTriggered as exc:
                last_exc = exc
                continue
            except (requests_exceptions.Timeout, requests_exceptions.RequestException) as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Ollama generate failed without a captured exception")

    def _generate_streaming(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_total_s: int,
    ) -> str:
        start = time.monotonic()
        last_log = start
        got_any = False
        chars = 0
        parts: list[str] = []

        # Requests: connect timeout + per-read timeout.
        connect_timeout = min(10, max_total_s) if max_total_s > 0 else 10
        read_timeout = (
            int(self.cfg.stall_timeout_s)
            if self.cfg.watchdog_enabled
            else max_total_s
        )
        if read_timeout <= 0:
            read_timeout = max_total_s if max_total_s > 0 else 900

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }

        with requests.post(
            self.cfg.url,
            json=payload,
            stream=True,
            timeout=(connect_timeout, read_timeout),
        ) as r:
            r.raise_for_status()

            for line in r.iter_lines(decode_unicode=True):
                now = time.monotonic()

                if max_total_s > 0 and (now - start) > float(max_total_s):
                    raise OllamaWatchdogTriggered(
                        reason="max_total_timeout",
                        model=model,
                        elapsed_s=now - start,
                    )

                if not line:
                    continue

                got_any = True
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Ignore malformed lines (shouldn't happen, but keep robust).
                    continue

                piece = obj.get("response")
                if isinstance(piece, str) and piece:
                    parts.append(piece)
                    chars += len(piece)

                if self.cfg.watchdog_enabled:
                    # Before first token, be stricter.
                    if chars == 0 and (now - start) > float(self.cfg.first_token_timeout_s):
                        raise OllamaWatchdogTriggered(
                            reason="first_token_timeout",
                            model=model,
                            elapsed_s=now - start,
                        )

                    if (now - last_log) >= float(max(1, int(self.cfg.log_interval_s))):
                        LOGGER.info(
                            "Ollama progress: model=%s, received=%d chars, elapsed=%.1fs",
                            model,
                            chars,
                            now - start,
                        )
                        last_log = now

                done = obj.get("done")
                if done is True:
                    break

        if not got_any and self.cfg.watchdog_enabled:
            now = time.monotonic()
            raise OllamaWatchdogTriggered(
                reason="no_response_data",
                model=model,
                elapsed_s=now - start,
            )

        return "".join(parts)


@dataclass(frozen=True)
class OpenAIConfig:
    """RU: Конфиг для OpenAI Chat Completions.

    EN: Config for OpenAI Chat Completions.
    """

    api_key: str
    model: str


class OpenAIProvider:
    """RU: Провайдер для OpenAI Chat Completions API.

    EN: Provider for OpenAI Chat Completions API.
    """

    def __init__(self, cfg: OpenAIConfig):
        self.cfg = cfg

    def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
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


@dataclass(frozen=True)
class AnthropicConfig:
    """RU: Конфиг для Anthropic Messages API.

    EN: Config for Anthropic Messages API.
    """

    api_key: str
    model: str


class AnthropicProvider:
    """RU: Минимальный провайдер Anthropic Messages API.

    Примечание: требуется ANTHROPIC_API_KEY и корректный идентификатор модели.

    EN: Minimal Anthropic Messages API provider.

    Note: requires ANTHROPIC_API_KEY and a valid model id.
    """

    def __init__(self, cfg: AnthropicConfig):
        self.cfg = cfg

    def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
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
        # content: [{type: 'text', text: '...'}]
        parts = data.get("content") or []
        if parts and isinstance(parts, list) and isinstance(parts[0], dict):
            return parts[0].get("text", "")
        return str(data)


@dataclass(frozen=True)
class GeminiConfig:
    """RU: Конфиг для Gemini Generative Language API.

    EN: Config for Gemini Generative Language API.
    """

    api_key: str
    model: str


class GeminiProvider:
    """RU: Минимальный провайдер Gemini через REST API generative language.

    Примечание: endpoint зависит от модели; здесь используется API v1beta.

    EN: Minimal Gemini provider via generative language REST.

    Note: endpoint varies by model; this uses the v1beta API.
    """

    def __init__(self, cfg: GeminiConfig):
        self.cfg = cfg

    def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
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
        # candidates[0].content.parts[0].text
        cands = data.get("candidates") or []
        if cands:
            content = cands[0].get("content") or {}
            parts = content.get("parts") or []
            if parts:
                return parts[0].get("text", "")
        return str(data)
