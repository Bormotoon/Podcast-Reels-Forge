"""Tests for LLM provider wrappers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING

from podcast_reels_forge.llm import providers

if TYPE_CHECKING:
    import pytest

    MonkeyPatch = pytest.MonkeyPatch


class DummyResponse:
    """Minimal response stub to mimic requests.Response."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict[str, object]:
        return self._payload


def test_openai_provider_parses_chat_completion(monkeypatch: MonkeyPatch) -> None:
    """Ensure OpenAI provider extracts chat completion content."""

    def fake_post(_url: str, **_kwargs: object) -> DummyResponse:
        return DummyResponse({"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr(providers, "requests", SimpleNamespace(post=fake_post))
    provider = providers.OpenAIProvider(
        providers.OpenAIConfig(api_key="k", model="m"),
    )
    result = asyncio.run(provider.generate("x", temperature=0.0, timeout=1))
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
    result = asyncio.run(provider.generate("x", temperature=0.0, timeout=1))
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
    result = asyncio.run(provider.generate("x", temperature=0.0, timeout=1))
    if result != "hey":
        message = "Expected Gemini provider to return 'hey'"
        raise AssertionError(message)


def test_prompt_cache_ram_is_bounded(monkeypatch) -> None:
    """llama-server allows its host-side prompt cache 8192 MiB by default.

    That cache, not the weights, grew to ~6 GB and became this pipeline's share
    of a host OOM: with full GPU offload the model is entirely in VRAM while the
    host held 32 context checkpoints of ~160 MB each.
    """
    from podcast_reels_forge.utils import llama_cpp_service as svc

    monkeypatch.setattr(svc, "_server_supports", lambda flag: True)
    cmd = svc._build_llama_server_cmd(
        model_path="m.gguf", host="127.0.0.1", port=1, threads=4, ctx_size=8192,
        n_gpu_layers=999, batch_size=1024, ubatch_size=512, main_gpu=0, parallel=1,
        extra_args=[], cache_ram_mb=1024, ctx_checkpoints=4,
    )

    assert cmd[cmd.index("--cache-ram") + 1] == "1024"
    assert cmd[cmd.index("--ctx-checkpoints") + 1] == "4"


def test_unsupported_flags_are_not_passed(monkeypatch) -> None:
    """An unknown option makes llama-server exit instead of start.

    llama.cpp adds and renames options often, so support is checked rather than
    assumed — a pipeline must not stop working after a server downgrade.
    """
    from podcast_reels_forge.utils import llama_cpp_service as svc

    monkeypatch.setattr(svc, "_server_supports", lambda flag: False)
    cmd = svc._build_llama_server_cmd(
        model_path="m.gguf", host="127.0.0.1", port=1, threads=4, ctx_size=8192,
        n_gpu_layers=999, batch_size=1024, ubatch_size=512, main_gpu=0, parallel=1,
        extra_args=[], cache_ram_mb=1024, ctx_checkpoints=4,
    )

    assert "--cache-ram" not in cmd
    assert "--ctx-checkpoints" not in cmd


def test_cache_ram_auto_uses_what_is_free_minus_a_reserve(monkeypatch) -> None:
    """The point of `auto`: spend the memory a shut-down VM just freed.

    A fixed number cannot know whether a 12 GB VM was stopped for this run or a
    browser has since eaten the difference.
    """
    from podcast_reels_forge.utils import llama_cpp_service as svc

    monkeypatch.setattr(svc, "available_ram_mb", lambda: 16384)

    resolved = svc.resolve_cache_ram_mb(
        "auto", reserve_mb=6144, min_mb=512, max_mb=8192,
    )

    assert resolved == 8192  # 16384 - 6144 = 10240, capped at the ceiling


def test_cache_ram_auto_shrinks_when_the_host_is_busy(monkeypatch) -> None:
    from podcast_reels_forge.utils import llama_cpp_service as svc

    monkeypatch.setattr(svc, "available_ram_mb", lambda: 8000)

    assert svc.resolve_cache_ram_mb(
        "auto", reserve_mb=6144, min_mb=512, max_mb=8192,
    ) == 1856


def test_cache_ram_auto_never_goes_below_the_floor(monkeypatch) -> None:
    """A cramped host still gets a working cache rather than none at all."""
    from podcast_reels_forge.utils import llama_cpp_service as svc

    monkeypatch.setattr(svc, "available_ram_mb", lambda: 4000)

    assert svc.resolve_cache_ram_mb(
        "auto", reserve_mb=6144, min_mb=512, max_mb=8192,
    ) == 512


def test_cache_ram_accepts_a_plain_number_and_null(monkeypatch) -> None:
    from podcast_reels_forge.utils import llama_cpp_service as svc

    monkeypatch.setattr(svc, "available_ram_mb", lambda: 16384)

    assert svc.resolve_cache_ram_mb(2048, reserve_mb=1, min_mb=1, max_mb=9999) == 2048
    assert svc.resolve_cache_ram_mb(None, reserve_mb=1, min_mb=1, max_mb=9999) is None
    assert svc.resolve_cache_ram_mb("junk", reserve_mb=1, min_mb=1, max_mb=9999) is None


def test_unreadable_meminfo_falls_back_to_the_floor(monkeypatch) -> None:
    """Guessing high when we cannot measure is how a host gets OOM-killed."""
    from podcast_reels_forge.utils import llama_cpp_service as svc

    monkeypatch.setattr(svc, "available_ram_mb", lambda: None)

    assert svc.resolve_cache_ram_mb(
        "auto", reserve_mb=6144, min_mb=512, max_mb=8192,
    ) == 512
