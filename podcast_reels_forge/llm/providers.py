"""RU: Провайдеры LLM (HTTP-клиенты) для разных платформ.

EN: LLM providers (HTTP clients) for multiple platforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests


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


class OllamaProvider:
    """RU: Провайдер для Ollama (/api/generate).

    EN: Provider for Ollama (/api/generate).
    """

    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg

    def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
        r = requests.post(
            self.cfg.url,
            json={
                "model": self.cfg.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["response"]


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
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
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
                "max_tokens": 2048,
                "temperature": temperature,
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
