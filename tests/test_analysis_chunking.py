"""RU: Тесты нарезки транскрипта на чанки и индекса таймингов.

EN: Tests for transcript chunking and the timing index.
"""

from __future__ import annotations

from typing import Any

from podcast_reels_forge.analysis.chunking import (
    build_analysis_chunks,
    transcript_units_from_segments,
)
from podcast_reels_forge.analysis.transcript_index import TranscriptIndex


def _words(text: str, start: float, per_word: float) -> list[dict[str, Any]]:
    words = []
    cursor = start
    for word in text.split():
        words.append(
            {"start": round(cursor, 3), "end": round(cursor + per_word, 3), "word": word},
        )
        cursor += per_word
    return words


def test_units_use_real_word_timings_for_sentence_boundaries() -> None:
    """Sentence spans come from word timings, not character-count guesswork.

    The two sentences differ wildly in length, so interpolating by character
    share would put the boundary in the wrong place.
    """
    text = "Да. Это очень длинное второе предложение с большим количеством слов."
    segment = {
        "start": 0.0,
        "end": 10.0,
        "text": text,
        "words": _words(text, 0.0, 1.0),
    }
    units = transcript_units_from_segments([segment])

    assert [u.text for u in units] == [
        "Да.",
        "Это очень длинное второе предложение с большим количеством слов.",
    ]
    # "Да." is one word, so it ends one second in — not at its character share.
    assert units[0].end == 1.0
    assert units[1].start == 1.0


def test_units_fall_back_to_interpolation_without_words() -> None:
    """Transcripts lacking word timings still produce usable units."""
    text = "Первое предложение. Второе предложение."
    units = transcript_units_from_segments(
        [{"start": 0.0, "end": 10.0, "text": text}],
    )
    assert len(units) == 2
    assert units[0].start == 0.0
    assert units[-1].end == 10.0


def test_units_fall_back_when_words_do_not_line_up() -> None:
    """A word list that disagrees with the text must not produce bogus spans."""
    text = "Первое предложение. Второе предложение."
    segment = {
        "start": 0.0,
        "end": 10.0,
        "text": text,
        "words": _words("только два слова", 0.0, 1.0),
    }
    units = transcript_units_from_segments([segment])
    assert len(units) == 2
    assert units[-1].end == 10.0


def test_chunks_respect_time_and_char_budgets() -> None:
    segments = []
    for index in range(40):
        start = index * 10.0
        text = f"Предложение номер {index} про школу."
        segments.append(
            {
                "start": start,
                "end": start + 10.0,
                "text": text,
                "words": _words(text, start, 10.0 / max(1, len(text.split()))),
            },
        )

    chunks = build_analysis_chunks(segments, chunk_seconds=100, max_chars=100_000, overlap_seconds=20)
    assert chunks
    for chunk in chunks:
        assert chunk.end - chunk.start <= 100.0 + 1e-6
    assert chunks[0].start == 0.0
    # Chunks must advance and cover the material in order.
    assert all(b.start > a.start for a, b in zip(chunks, chunks[1:]))


def test_chunks_overlap_so_boundary_moments_are_seen_twice() -> None:
    segments = [
        {"start": i * 10.0, "end": i * 10.0 + 10.0, "text": f"Фраза {i}."}
        for i in range(30)
    ]
    chunks = build_analysis_chunks(segments, chunk_seconds=100, max_chars=100_000, overlap_seconds=30)
    assert len(chunks) > 1
    assert chunks[1].start < chunks[0].end, "adjacent chunks should share a tail"


# -- transcript index --------------------------------------------------------


def _index() -> TranscriptIndex:
    text = "Первое предложение здесь. Второе предложение тут."
    return TranscriptIndex.from_transcript(
        {
            "segments": [
                {"start": 0.0, "end": 8.0, "text": text, "words": _words(text, 0.0, 1.0)},
            ],
            "sentences": [
                {"start": 0.0, "end": 3.0, "text": "Первое предложение здесь."},
                {"start": 3.0, "end": 8.0, "text": "Второе предложение тут."},
            ],
        },
    )


def test_index_reads_words_and_sentences() -> None:
    index = _index()
    assert len(index.words) == 6
    assert len(index.sentences) == 2
    assert bool(index)


def test_index_is_falsy_and_safe_when_empty() -> None:
    empty = TranscriptIndex.from_transcript({})
    assert not empty
    assert empty.words_between(0, 10) == []
    assert empty.speech_rate(0, 10) is None
    assert empty.snap_start(5.0, max_shift=3.0) == 5.0


def test_words_between_returns_the_overlapping_span() -> None:
    index = _index()
    words = index.words_between(1.5, 3.5)
    assert [w.text for w in words] == ["предложение", "здесь.", "Второе"]


def test_text_between_truncates_on_a_word_boundary() -> None:
    index = _index()
    text = index.text_between(0.0, 8.0, max_chars=20)
    assert len(text) <= 21  # the ellipsis may push it one char over
    assert text.endswith("…")


def test_speech_rate_counts_words_per_second() -> None:
    index = _index()
    assert index.speech_rate(0.0, 6.0) == 1.0


def test_index_falls_back_to_segments_without_sentences() -> None:
    """Older transcripts have no sentence groups; segments stand in."""
    index = TranscriptIndex.from_transcript(
        {"segments": [{"start": 0.0, "end": 5.0, "text": "Одно предложение."}]},
    )
    assert len(index.sentences) == 1
    assert index.sentences[0].end == 5.0
