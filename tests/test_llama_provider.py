"""RU: Поведение LlamaCppProvider: откат схем, backoff, повторы, сессия.

EN: LlamaCppProvider behaviour: schema downgrade, backoff, retries, session.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest

from podcast_reels_forge.llm import providers
from podcast_reels_forge.llm.providers import (
    SCHEMA_ANY,
    SCHEMA_FALLBACK,
    LlamaCppConfig,
    LlamaCppProvider,
    LlmHttpError,
    build_completion_payload,
    retry_delay,
)
from podcast_reels_forge.llm.schemas import (
    ANY_OBJECT_SCHEMA,
    FALLBACK_SCHEMAS,
    SCOUT_JSON_SCHEMA,
)


def _cfg(**overrides: Any) -> LlamaCppConfig:
    base: dict[str, Any] = {
        "url": "http://127.0.0.1:1/completion",
        "model": "m",
        "json_schema": SCOUT_JSON_SCHEMA,
        "fallback_schema": FALLBACK_SCHEMAS["scout"],
        "max_retries": 2,
        "retry_base_delay_s": 0.0,
    }
    base.update(overrides)
    return LlamaCppConfig(**base)


def test_schema_levels_step_down_through_the_fallback() -> None:
    cfg = _cfg()
    full = build_completion_payload(cfg, "p", temperature=0.1)
    fallback = build_completion_payload(cfg, "p", temperature=0.1, schema_level=SCHEMA_FALLBACK)
    anything = build_completion_payload(cfg, "p", temperature=0.1, schema_level=SCHEMA_ANY)
    assert full["json_schema"] == SCOUT_JSON_SCHEMA
    assert fallback["json_schema"] == FALLBACK_SCHEMAS["scout"]
    assert anything["json_schema"] == ANY_OBJECT_SCHEMA


def test_fallback_schemas_still_require_the_evidence_fields() -> None:
    item = FALLBACK_SCHEMAS["scout"]["properties"]["candidates"]["items"]
    assert {"start", "end", "quote"} <= set(item["required"])


def test_retry_delay_grows_exponentially_and_is_capped() -> None:
    rng = random.Random(0)
    delays = [retry_delay(n, base_s=2.0, max_s=30.0, jitter=0.0, rng=rng) for n in range(1, 7)]
    assert delays[:4] == [2.0, 4.0, 8.0, 16.0]
    assert max(delays) == 30.0


def test_retry_delay_jitter_stays_within_bounds() -> None:
    rng = random.Random(1)
    for _ in range(50):
        assert 1.5 <= retry_delay(1, base_s=2.0, max_s=30.0, jitter=0.25, rng=rng) <= 2.5


class _FakeResponse:
    def __init__(self, status: int, body: str = "", payload: dict[str, Any] | None = None) -> None:
        self.status = status
        self._body = body
        self._payload = payload or {}
        self.request_info = None
        self.history = ()

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def text(self) -> str:
        return self._body

    async def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, Any]] = []
        self.closed = False

    def post(self, _url: str, *, json: dict[str, Any], timeout: Any) -> _FakeResponse:
        self.payloads.append(json)
        return self.responses.pop(0)

    async def close(self) -> None:
        self.closed = True


def _provider_with(monkeypatch: pytest.MonkeyPatch, session: _FakeSession, **cfg: Any) -> LlamaCppProvider:
    monkeypatch.setattr(providers, "_detect_template", lambda _base: "raw")
    provider = LlamaCppProvider(_cfg(**cfg))

    async def _session() -> _FakeSession:
        return session

    monkeypatch.setattr(provider, "_get_session", _session)
    return provider


def test_schema_rejection_retries_with_the_simplified_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(
        [
            _FakeResponse(400, "failed to parse json_schema grammar"),
            _FakeResponse(200, payload={"content": '{"candidates": []}'}),
        ],
    )
    provider = _provider_with(monkeypatch, session)
    text = asyncio.run(provider.generate("p", temperature=0.1, timeout=5))
    assert text == '{"candidates": []}'
    assert session.payloads[0]["json_schema"] == SCOUT_JSON_SCHEMA
    assert session.payloads[1]["json_schema"] == FALLBACK_SCHEMAS["scout"]


def test_plain_client_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession([_FakeResponse(404, "no such endpoint")])
    provider = _provider_with(monkeypatch, session)
    with pytest.raises(LlmHttpError):
        asyncio.run(provider.generate("p", temperature=0.1, timeout=5))
    assert len(session.payloads) == 1


def test_server_errors_are_retried_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(providers.asyncio, "sleep", _sleep)
    session = _FakeSession(
        [
            _FakeResponse(503),
            _FakeResponse(200, payload={"content": "{}"}),
        ],
    )
    provider = _provider_with(monkeypatch, session, retry_base_delay_s=1.0)
    assert asyncio.run(provider.generate("p", temperature=0.1, timeout=5)) == "{}"
    # 503 means "model loading": the floor is 5 s, not the 1 s base.
    assert sleeps and sleeps[0] >= 5.0 * 0.75


def test_aclose_closes_the_pooled_session() -> None:
    async def _run() -> bool:
        provider = LlamaCppProvider(_cfg())
        session = await provider._get_session()
        again = await provider._get_session()
        assert session is again, "one session per provider and loop"
        await provider.aclose()
        return session.closed

    assert asyncio.run(_run())
