"""RU: Перенос таймингов слов на вычитанный текст.

EN: Carrying word timings over to proofread text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from podcast_reels_forge.analysis.contracts import coerce_moment_record
from podcast_reels_forge.analysis.transcript_index import TranscriptIndex
from podcast_reels_forge.analysis.validation import METHOD_EXACT, verify_quote
from podcast_reels_forge.utils.burned_subtitles import (
    _real_timed_words,
    load_transcript_segments,
)
from podcast_reels_forge.utils.word_alignment import (
    realign_transcript_words,
    realign_words,
    words_match_text,
)


def _words(*items: tuple[str, float, float]) -> list[dict[str, Any]]:
    return [{"word": w, "start": s, "end": e, "probability": 0.9} for w, s, e in items]


RAW = _words(("ну", 0.0, 0.4), ("по", 0.5, 0.7), ("этому", 0.7, 1.1), ("щас", 1.2, 1.5), ("скажу", 1.6, 2.0))


def test_matched_words_keep_their_timings() -> None:
    words = realign_words(RAW, "Ну, по этому щас скажу.")
    assert words is not None
    assert [w["word"] for w in words] == ["Ну,", "по", "этому", "щас", "скажу."]
    assert [(w["start"], w["end"]) for w in words] == [(0.0, 0.4), (0.5, 0.7), (0.7, 1.1), (1.2, 1.5), (1.6, 2.0)]


def test_replaced_run_shares_the_time_of_what_it_replaced() -> None:
    words = realign_words(RAW, "Ну, поэтому сейчас скажу.")
    assert words is not None
    assert [w["word"] for w in words] == ["Ну,", "поэтому", "сейчас", "скажу."]
    merged, respelled = words[1], words[2]
    # "по этому щас" (0.5 .. 1.5) became "поэтому сейчас".
    assert merged["start"] == 0.5
    assert respelled["end"] == 1.5
    assert merged["end"] <= respelled["start"]
    assert words[3]["start"] == 1.6


def test_inserted_punctuation_token_takes_the_gap() -> None:
    words = realign_words(RAW, "Ну — по этому щас скажу")
    assert words is not None
    dash = words[1]
    assert dash["word"] == "—"
    assert 0.4 <= dash["start"] <= dash["end"] <= 0.5


def test_timings_stay_monotonic() -> None:
    words = realign_words(RAW, "ну вот по этому самому щас скажу")
    assert words is not None
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)
    assert all(w["end"] >= w["start"] for w in words)


def test_no_timings_means_nothing_to_carry() -> None:
    assert realign_words([], "текст") is None
    assert realign_words(_words(("a", 1.0, 0.5)), "a") is None


def test_segments_are_realigned_in_place_and_raw_words_kept() -> None:
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Ну, поэтому сейчас скажу.", "words": list(RAW)},
        {"start": 3.0, "end": 4.0, "text": "да", "words": _words(("да", 3.0, 3.5))},
    ]
    assert realign_transcript_words(segments) == 1
    assert segments[0]["raw_words"] == RAW
    assert words_match_text(segments[0]["words"], segments[0]["text"])
    assert "raw_words" not in segments[1]


def test_karaoke_uses_real_timings_after_proofreading(tmp_path: Path) -> None:
    """The renderer only trusts word timings that spell the displayed text."""
    segment = {"start": 0.0, "end": 2.0, "text": "Ну, поэтому сейчас скажу.", "words": list(RAW)}
    data: dict[str, Any] = {
        "segments": [segment],
        "sentences": [{"start": 0.0, "end": 2.0, "text": segment["text"]}],
    }
    path = tmp_path / "t.json"

    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    before = load_transcript_segments(path)[0]
    assert _real_timed_words(before) is None, "raw words no longer spell the text"

    realign_transcript_words(data["segments"])
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    after = load_transcript_segments(path)[0]
    timed = _real_timed_words(after)
    assert timed is not None
    assert timed[-1].text == "скажу."


def test_quotes_from_corrected_text_verify_exactly() -> None:
    segments = [{"start": 0.0, "end": 2.0, "text": "Ну, поэтому сейчас скажу.", "words": list(RAW)}]
    realign_transcript_words(segments)
    index = TranscriptIndex.from_transcript({"segments": segments})
    record = coerce_moment_record(
        {"start": 0.0, "end": 2.0, "quote": "поэтому сейчас скажу", "title": "t", "score": 7},
    )
    assert record is not None
    assert verify_quote(record, index).method == METHOD_EXACT
