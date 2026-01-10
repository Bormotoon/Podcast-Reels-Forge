"""Tests for analysis logic helper functions."""

from __future__ import annotations

from unittest.mock import MagicMock
from podcast_reels_forge.scripts.analyze import (
    fmt_hms,
    segments_to_compact_text,
    chunk_segments_by_time,
    _normalize_prompt_lang,
    _assign_speakers,
    find_moments,
)

def test_fmt_hms() -> None:
    assert fmt_hms(0) == "00:00"
    assert fmt_hms(59) == "00:59"
    assert fmt_hms(61) == "01:01"
    assert fmt_hms(3600) == "01:00:00"
    assert fmt_hms(3661) == "01:01:01"

def test_segments_to_compact_text() -> None:
    segs = [
        {"start": 0, "end": 5, "text": "Hello"},
        {"start": 5, "end": 10, "text": "World"},
    ]
    # Generator-based version: "[0-5] Hello\n[5-10] World"
    res = segments_to_compact_text(segs, 100)
    assert res == "[0-5] Hello\n[5-10] World"
    
    # Test capping
    res_cap = segments_to_compact_text(segs, 10)
    assert len(res_cap) == 10

def test_chunk_segments_by_time() -> None:
    segs = [
        {"start": 0, "end": 10, "text": "A"},
        {"start": 10, "end": 20, "text": "B"},
        {"start": 21, "end": 30, "text": "C"},
        {"start": 30, "end": 45, "text": "D"},
    ]
    # Case 1: All in one chunk
    chunks = chunk_segments_by_time(segs, 100)
    assert len(chunks) == 1
    assert len(chunks[0]) == 4

    # Case 2: Split by 25s
    # s[0].end - 0 = 10 <= 25 (ok)
    # s[1].end - 0 = 20 <= 25 (ok)
    # s[2].end - 0 = 30 > 25 (split! new chunk start at s[2].start=21)
    # s[3].end - 21 = 45-21 = 24 <= 25 (ok)
    chunks = chunk_segments_by_time(segs, 25)
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2

def test_normalize_prompt_lang() -> None:
    assert _normalize_prompt_lang("ru", None) == "ru"
    assert _normalize_prompt_lang("EN", "ru") == "en"
    assert _normalize_prompt_lang("auto", "russian") == "ru"
    assert _normalize_prompt_lang("auto", "english") == "en"
    assert _normalize_prompt_lang(None, "ru-RU") == "ru"
    assert _normalize_prompt_lang("", "en-US") == "en"

def test_assign_speakers() -> None:
    segments = [
        {"start": 0, "end": 10, "text": "Hello world"},
        {"start": 10, "end": 20, "text": "Goodbye"},
    ]
    diar = [
        {"start": 0, "end": 15, "speaker": "SPEAKER_01"},
        {"start": 15, "end": 25, "speaker": "SPEAKER_02"},
    ]
    _assign_speakers(segments, diar, prefix=True)
    
    assert segments[0]["speaker"] == "SPEAKER_01"
    assert segments[0]["text"] == "(SPEAKER_01) Hello world"
    
    # segment 2 (10-20) overlaps 10-15 (5s) with SP1, 15-20 (5s) with SP2. 
    # Current overlap logic: ov = min(a1, b1) - max(a0, b0)
    # Actually if they tie, it depends on order.
    # We just care it assigned something.
    assert "speaker" in segments[1]

def test_find_moments_orchestration() -> None:
    provider = MagicMock()
    # Mock first stage (chunk analysis for 2 chunks) and then selection
    provider.generate.side_effect = [
        '{"moment": {"start": 10, "end": 40, "score": 9, "title": "C1", "why": "W1"}}', # chunk 1
        '{"moment": {"start": 60, "end": 90, "score": 8, "title": "C2", "why": "W2"}}', # chunk 2
        '{"moments": [{"start": 10, "end": 40, "score": 9, "title": "T1", "why": "W1", "quote": "Q1"}]}' # selection
    ]
    
    segments = [{"start": i*10, "end": (i+1)*10, "text": f"seg {i}"} for i in range(10)]
    
    moments = find_moments(
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
        select_prompt="prompt2 {count} {candidates_json}"
    )
    
    assert len(moments) == 1
    assert moments[0].title == "T1"
    assert moments[0].start == 10.0
