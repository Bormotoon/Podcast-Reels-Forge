"""Tests for analysis logic helper functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import podcast_reels_forge.scripts.analyze as analyze
from podcast_reels_forge.utils import ollama_service


def test_fmt_hms() -> None:
    assert analyze.fmt_hms(0) == "00:00"
    assert analyze.fmt_hms(59) == "00:59"
    assert analyze.fmt_hms(61) == "01:01"
    assert analyze.fmt_hms(3600) == "01:00:00"
    assert analyze.fmt_hms(3661) == "01:01:01"


def test_segments_to_compact_text() -> None:
    segs = [
        {"start": 0, "end": 5, "text": "Hello"},
        {"start": 5, "end": 10, "text": "World"},
    ]
    res = analyze.segments_to_compact_text(segs, 100)
    assert res == "[0-5] Hello\n[5-10] World"

    res_cap = analyze.segments_to_compact_text(segs, 10)
    assert len(res_cap) == 10


def test_chunk_segments_by_time() -> None:
    segs = [
        {"start": 0, "end": 10, "text": "A"},
        {"start": 10, "end": 20, "text": "B"},
        {"start": 21, "end": 30, "text": "C"},
        {"start": 30, "end": 45, "text": "D"},
    ]
    chunks = analyze.chunk_segments_by_time(segs, 100)
    assert len(chunks) == 1
    assert len(chunks[0]) == 4

    chunks = analyze.chunk_segments_by_time(segs, 25)
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2


def test_normalize_prompt_lang() -> None:
    assert analyze._normalize_prompt_lang("ru", None) == "ru"
    assert analyze._normalize_prompt_lang("EN", "ru") == "en"
    assert analyze._normalize_prompt_lang("auto", "russian") == "ru"
    assert analyze._normalize_prompt_lang("auto", "english") == "en"
    assert analyze._normalize_prompt_lang(None, "ru-RU") == "ru"
    assert analyze._normalize_prompt_lang("", "en-US") == "en"


def test_assign_speakers() -> None:
    segments = [
        {"start": 0, "end": 10, "text": "Hello world"},
        {"start": 10, "end": 20, "text": "Goodbye"},
    ]
    diar = [
        {"start": 0, "end": 15, "speaker": "SPEAKER_01"},
        {"start": 15, "end": 25, "speaker": "SPEAKER_02"},
    ]

    analyze._assign_speakers(segments, diar, prefix=True)
    assert segments[0]["speaker"] == "SPEAKER_01"
    assert segments[0]["text"] == "(SPEAKER_01) Hello world"
    assert "speaker" in segments[1]


def test_find_moments_orchestration() -> None:
    provider = MagicMock()
    provider.generate.side_effect = [
        '{"moment": {"start": 10, "end": 40, "score": 9, "title": "C1", "why": "W1"}}',
        '{"moment": {"start": 60, "end": 90, "score": 8, "title": "C2", "why": "W2"}}',
        '{"moments": [{"start": 10, "end": 40, "score": 9, "title": "T1", "why": "W1", "quote": "Q1"}]}',
    ]

    segments = [
        {"start": i * 10, "end": (i + 1) * 10, "text": f"seg {i}"}
        for i in range(10)
    ]

    moments = analyze.find_moments(
        provider,
        segments,
        duration=100.0,
        r_min=20,
        r_max=60,
        count=1,
        chunk_sec=50,
        max_ch=1000,
        timeout=10,
        ch_prompt="prompt1 {r_min} {r_max} {transcript}",
        select_prompt="prompt2 {count} {candidates_json}",
    )

    assert len(moments) == 1
    # Final selection is done locally (no extra LLM call), so we keep the best
    # candidate from chunk analysis.
    assert moments[0].title == "C1"
    assert moments[0].start == 10.0


def test_parse_local_ollama_host_port() -> None:
    assert ollama_service.parse_local_ollama_host_port(
        "http://127.0.0.1:11434/api/generate",
    ) == ("127.0.0.1", 11434)
    assert ollama_service.parse_local_ollama_host_port(
        "http://localhost:11434/api/generate",
    ) == ("localhost", 11434)
    assert ollama_service.parse_local_ollama_host_port(
        "http://10.0.0.1:11434/api/generate",
    ) is None


def test_ollama_start_skips_if_port_open(monkeypatch) -> None:
    called: dict[str, int] = {"popen": 0}

    def fake_is_open(_host: str, _port: int) -> bool:
        return True

    def fake_popen(*_args: object, **_kwargs: object) -> object:
        called["popen"] += 1
        return object()

    monkeypatch.setattr(ollama_service, "is_tcp_open", fake_is_open)
    monkeypatch.setattr(ollama_service.subprocess, "Popen", fake_popen)

    proc = ollama_service.ollama_start(host="127.0.0.1", port=11434)
    assert proc is None
    assert called["popen"] == 0
