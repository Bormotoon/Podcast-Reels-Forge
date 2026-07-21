"""Tests for building speaker turns out of a transcript plus diarization."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_reels_forge.analysis.speaker_turns import (
    build_speaker_turns,
    distinct_speakers,
    load_diarization,
    render_turns,
    speaker_at,
)


def _segment(start: float, end: float, text: str, words: list[tuple[float, float, str]]):
    return {
        "start": start,
        "end": end,
        "text": text,
        "words": [{"start": s, "end": e, "word": w} for s, e, w in words],
    }


def test_speaker_at_covers_gaps() -> None:
    diar = [
        {"start": 0.0, "end": 2.0, "speaker": "A"},
        {"start": 3.0, "end": 5.0, "speaker": "B"},
    ]
    assert speaker_at(diar, 1.0) == "A"
    assert speaker_at(diar, 4.0) == "B"
    # A word landing in the gap between turns must still be attributed.
    assert speaker_at(diar, 2.4) == "A"
    assert speaker_at(diar, 2.9) == "B"
    assert speaker_at([], 1.0) == ""


def test_turns_split_a_segment_between_speakers() -> None:
    """Whisper merges speakers into one segment; the split must undo that."""
    segment = _segment(
        0.0, 4.0,
        "Привет, друзья! Это я. Мы идём дальше.",
        [
            (0.0, 0.5, "Привет,"), (0.5, 1.0, "друзья!"),
            (2.0, 2.4, "Это"), (2.4, 2.8, "я."),
            (3.0, 3.4, "Мы"), (3.4, 3.7, "идём"), (3.7, 4.0, "дальше."),
        ],
    )
    diar = [
        {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
        {"start": 1.5, "end": 2.9, "speaker": "SPEAKER_01"},
        {"start": 2.9, "end": 4.0, "speaker": "SPEAKER_00"},
    ]

    turns = build_speaker_turns(segment_list := [segment], diar)

    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert turns[0].text == "Привет, друзья!"
    assert turns[1].text == "Это я."
    assert turns[2].text == "Мы идём дальше."
    assert segment_list[0]["text"] == "Привет, друзья! Это я. Мы идём дальше."


def test_turns_use_the_proofread_text_not_the_raw_words() -> None:
    """Words keep the raw ASR spelling after proofreading; text must win."""
    segment = _segment(
        0.0, 2.0,
        "Настольных ролевых игр. Женю Курокрада.",
        [
            (0.0, 0.4, "настольних"), (0.4, 0.8, "ролевых"), (0.8, 1.0, "игр."),
            (1.2, 1.6, "Жену"), (1.6, 2.0, "Курократа."),
        ],
    )
    diar = [
        {"start": 0.0, "end": 1.1, "speaker": "A"},
        {"start": 1.1, "end": 2.0, "speaker": "B"},
    ]

    turns = build_speaker_turns([segment], diar)

    joined = " ".join(t.text for t in turns)
    assert "настольних" not in joined
    assert "Настольных" in joined
    assert "Курокрада" in joined


def test_turns_snap_to_sentence_boundaries() -> None:
    """A turn must not start mid-phrase because the split landed there."""
    segment = _segment(
        0.0, 3.0,
        "Идём с нами. От работников педобразования.",
        [
            (0.0, 0.3, "Идём"), (0.3, 0.6, "с"), (0.6, 0.9, "нами."),
            (1.0, 1.4, "От"), (1.4, 2.0, "работников"), (2.0, 3.0, "педобразования."),
        ],
    )
    # The diarization boundary falls one word late, inside "От работников".
    diar = [
        {"start": 0.0, "end": 1.2, "speaker": "A"},
        {"start": 1.2, "end": 3.0, "speaker": "B"},
    ]

    turns = build_speaker_turns([segment], diar)

    assert turns[0].text == "Идём с нами."
    assert turns[1].text == "От работников педобразования."


def test_consecutive_turns_of_one_speaker_merge() -> None:
    segments = [
        _segment(0.0, 1.0, "Первая фраза.", [(0.0, 0.5, "Первая"), (0.5, 1.0, "фраза.")]),
        _segment(1.0, 2.0, "Вторая фраза.", [(1.0, 1.5, "Вторая"), (1.5, 2.0, "фраза.")]),
    ]
    diar = [{"start": 0.0, "end": 2.0, "speaker": "A"}]

    turns = build_speaker_turns(segments, diar)

    assert len(turns) == 1
    assert turns[0].text == "Первая фраза. Вторая фраза."


def test_no_diarization_means_no_turns() -> None:
    segment = _segment(0.0, 1.0, "Текст.", [(0.0, 1.0, "Текст.")])
    assert build_speaker_turns([segment], []) == []


def test_render_and_distinct_speakers() -> None:
    segment = _segment(
        0.0, 2.0, "Раз два. Три четыре.",
        [(0.0, 0.5, "Раз"), (0.5, 1.0, "два."), (1.2, 1.6, "Три"), (1.6, 2.0, "четыре.")],
    )
    diar = [
        {"start": 0.0, "end": 1.1, "speaker": "SPEAKER_00"},
        {"start": 1.1, "end": 2.0, "speaker": "SPEAKER_01"},
    ]
    turns = build_speaker_turns([segment], diar)

    assert distinct_speakers(turns) == ["SPEAKER_00", "SPEAKER_01"]
    assert render_turns(turns) == "SPEAKER_00: Раз два.\nSPEAKER_01: Три четыре."
    named = render_turns(turns, names={"SPEAKER_00": "Егор"})
    assert named.startswith("Егор: Раз два.")
    # An unnamed label keeps its id rather than getting a made-up name.
    assert "SPEAKER_01:" in named


def test_load_diarization_tolerates_junk(tmp_path: Path) -> None:
    path = tmp_path / "diarization.json"
    path.write_text(
        json.dumps([
            {"start": 1.0, "end": 2.0, "speaker": "B"},
            {"start": 0.0, "end": 1.0, "speaker": "A"},
            {"start": 5.0, "end": 4.0, "speaker": "C"},   # inverted
            {"start": 6.0, "end": 7.0, "speaker": ""},    # unnamed
            "junk",
        ]),
        encoding="utf-8",
    )

    entries = load_diarization(path)

    assert [e["speaker"] for e in entries] == ["A", "B"]
    assert load_diarization(tmp_path / "missing.json") == []
    assert load_diarization(None) == []
