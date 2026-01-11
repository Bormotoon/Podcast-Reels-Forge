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


class DummyStreamResponse:
    """Streaming response stub that supports `with` and `iter_lines()`."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __enter__(self) -> "DummyStreamResponse":
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc: object,
        _tb: object,
    ) -> bool:
        return False

    def raise_for_status(self) -> None:
        return

    def iter_lines(self, decode_unicode: bool = False, **_kwargs: object) -> list[str]:
        # requests returns an iterator; list is also iterable.
        _ = decode_unicode
        return self._lines


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


def test_ollama_provider_streaming_collects_response(monkeypatch: MonkeyPatch) -> None:
    """Ensure Ollama provider can consume NDJSON streaming responses."""

    def fake_post(_url: str, **kwargs: object) -> DummyStreamResponse:
        payload = kwargs.get("json")
        if not isinstance(payload, dict):
            raise AssertionError("Expected json payload dict")
        if payload.get("stream") is not True:
            raise AssertionError("Expected stream=True for Ollama")

        # Two streamed chunks + done marker.
        return DummyStreamResponse(
            [
                '{"response":"hi","done":false}',
                '{"response":" there","done":true}',
            ],
        )

    monkeypatch.setattr(providers, "requests", SimpleNamespace(post=fake_post))
    provider = providers.OllamaProvider(
        providers.OllamaConfig(url="http://x/api/generate", model="m"),
    )
    result = provider.generate("x", temperature=0.0, timeout=5)
    if result != "hi there":
        raise AssertionError("Expected Ollama provider to join streamed chunks")


def test_ollama_provider_retries_and_switches_model(monkeypatch: MonkeyPatch) -> None:
    """Ensure watchdog/timeout failures can trigger a retry with fallback model."""

    calls: list[str] = []

    def fake_post(_url: str, **kwargs: object) -> DummyStreamResponse:
        payload = kwargs.get("json")
        if not isinstance(payload, dict):
            raise AssertionError("Expected json payload dict")
        model = str(payload.get("model"))
        calls.append(model)

        if len(calls) == 1:
            raise providers.requests_exceptions.ReadTimeout("stall")

        return DummyStreamResponse(['{"response":"ok","done":true}'])

    monkeypatch.setattr(providers, "requests", SimpleNamespace(post=fake_post))
    provider = providers.OllamaProvider(
        providers.OllamaConfig(
            url="http://x/api/generate",
            model="m1",
            max_retries=1,
            fallback_models=("m2",),
        ),
    )
    out = provider.generate("x", temperature=0.0, timeout=5)
    if out != "ok":
        raise AssertionError("Expected retry to succeed")
    if calls != ["m1", "m2"]:
        raise AssertionError(f"Expected model switch on retry, got: {calls}")
