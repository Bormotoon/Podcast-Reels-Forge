"""Tests for LLM provider wrappers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from podcast_reels_forge.llm import providers

if TYPE_CHECKING:
    import pytest

    MonkeyPatch = pytest.MonkeyPatch


class DummyResponse:
    """Minimal response stub to mimic requests.Response."""

    def __init__(self, payload: dict[str, object]) -> None:
        """Store the stub payload."""
        self._payload = payload

    def raise_for_status(self) -> None:
        """Mimic successful HTTP call."""
        return

    def json(self) -> dict[str, object]:
        """Return the stub payload."""
        return self._payload


def test_openai_provider_parses_chat_completion(monkeypatch: MonkeyPatch) -> None:
    """Ensure OpenAI provider extracts chat completion content."""

    def fake_post(_url: str, **_kwargs: object) -> DummyResponse:
        return DummyResponse({"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr(providers, "requests", SimpleNamespace(post=fake_post))
    provider = providers.OpenAIProvider(
        providers.OpenAIConfig(api_key="k", model="m"),
    )
    result = provider.generate("x", temperature=0.0, timeout=1)
    if result != "hi":
        message = "Expected OpenAI provider to return 'hi'"
        raise AssertionError(message)


def test_anthropic_provider_parses_messages(monkeypatch: MonkeyPatch) -> None:
    """Ensure Anthropic provider extracts message text content."""

    def fake_post(_url: str, **_kwargs: object) -> DummyResponse:
        return DummyResponse({"content": [{"type": "text", "text": "hello"}]})

    monkeypatch.setattr(providers, "requests", SimpleNamespace(post=fake_post))
    provider = providers.AnthropicProvider(
        providers.AnthropicConfig(api_key="k", model="m"),
    )
    result = provider.generate("x", temperature=0.0, timeout=1)
    if result != "hello":
        message = "Expected Anthropic provider to return 'hello'"
        raise AssertionError(message)


def test_gemini_provider_parses_candidates(monkeypatch: MonkeyPatch) -> None:
    """Ensure Gemini provider extracts text from candidates."""

    def fake_post(_url: str, **_kwargs: object) -> DummyResponse:
        return DummyResponse({"candidates": [{"content": {"parts": [{"text": "hey"}]}}]})

    monkeypatch.setattr(providers, "requests", SimpleNamespace(post=fake_post))
    provider = providers.GeminiProvider(
        providers.GeminiConfig(api_key="k", model="m"),
    )
    result = provider.generate("x", temperature=0.0, timeout=1)
    if result != "hey":
        message = "Expected Gemini provider to return 'hey'"
        raise AssertionError(message)
