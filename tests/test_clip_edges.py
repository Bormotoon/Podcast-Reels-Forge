"""Clip edges placed by the speech: lead-in, tail, sentence edges, neighbours."""

from __future__ import annotations

from typing import Any

import pytest

from podcast_reels_forge.analysis.transcript_index import TranscriptIndex
from podcast_reels_forge.utils.clip_intervals import ClipEdges, padded_intervals

EDGES = ClipEdges(lead_in_s=0.6, tail_s=0.3, guard_s=0.1, max_extend_s=4.0)


def _transcript(sentences: list[list[tuple[float, float, str]]]) -> TranscriptIndex:
    """Sentences given as lists of (start, end, word)."""

    words: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    for sentence in sentences:
        words += [{"start": s, "end": e, "word": w} for s, e, w in sentence]
        spans.append({
            "start": sentence[0][0],
            "end": sentence[-1][1],
            "text": " ".join(w for _, _, w in sentence),
        })
    return TranscriptIndex.from_transcript({
        "segments": [{"start": 0.0, "end": words[-1]["end"], "text": "", "words": words}],
        "sentences": spans,
    })


# A 2 s pause before "Вот", 1 s after "сказал." and a breathless join later.
INDEX = _transcript([
    [(7.0, 7.5, "Раньше"), (7.5, 8.0, "было.")],
    [(10.0, 10.4, "Вот"), (10.4, 11.0, "что"), (11.0, 11.6, "я"), (11.6, 12.0, "сказал.")],
    [(13.0, 13.5, "А"), (13.5, 14.0, "потом"), (14.0, 14.5, "ушёл.")],
    [(14.55, 15.0, "Всё.")],
])


def _cut(start: float, end: float, padding: float = 5.0) -> tuple[float, float]:
    [interval] = padded_intervals([(start, end)], padding, index=INDEX, edges=EDGES)
    return interval


def test_edges_hug_the_speech_instead_of_fixed_padding() -> None:
    start, end = _cut(10.0, 12.0)
    assert start == pytest.approx(9.4)   # 0.6 s lead-in, the pause is long enough
    assert end == pytest.approx(12.3)    # 0.3 s tail, well before "А" at 13.0


def test_mid_sentence_edges_grow_to_the_whole_sentence() -> None:
    start, end = _cut(11.0, 11.6)        # "я" alone
    assert start == pytest.approx(9.4)
    assert end == pytest.approx(12.3)


def test_lead_in_and_tail_stop_short_of_neighbouring_words() -> None:
    # Clip "А потом ушёл." — the next word "Всё." follows 0.05 s later.
    start, end = _cut(13.0, 14.5)
    assert start == pytest.approx(12.4)   # 13.0 - 0.6, "сказал." ended at 12.0
    assert end == pytest.approx(14.5)     # no silence to add before "Всё."

    # A pause of 0.5 s before the clip: lead-in shrinks to 0.5 - guard.
    index = _transcript([[(0.0, 1.0, "Раз.")], [(1.5, 2.0, "Два.")]])
    [interval] = padded_intervals([(1.5, 2.0)], 5.0, index=index, edges=EDGES)
    assert interval[0] == pytest.approx(1.1)


def test_far_sentence_edge_is_not_chased() -> None:
    edges = ClipEdges(max_extend_s=0.3)
    [interval] = padded_intervals([(11.0, 11.6)], 5.0, index=INDEX, edges=edges)
    # Words on both sides touch the clip, so there is no silence to add.
    assert interval == (pytest.approx(11.0), pytest.approx(11.6))


def test_neighbouring_clips_split_the_silence() -> None:
    first, second = padded_intervals(
        [(10.0, 12.0), (13.0, 14.0)], 5.0, index=INDEX, edges=EDGES,
    )
    assert first[1] <= second[0]


def test_falls_back_to_padding_without_words() -> None:
    empty = TranscriptIndex()
    assert padded_intervals([(100.0, 130.0)], 5.0, index=empty, edges=EDGES) == [(95.0, 135.0)]


def test_config_tolerates_garbage() -> None:
    edges = ClipEdges.from_config({"lead_in_s": "0.8", "tail_s": "x", "guard_s": -1})
    assert edges == ClipEdges(lead_in_s=0.8, guard_s=0.0)
    assert ClipEdges.from_config(None) == ClipEdges()


def test_sentences_come_from_word_punctuation() -> None:
    # One transcript "sentence" spanning three real ones.
    words = [
        (0.0, 0.5, "Раз"), (0.5, 1.0, "два."), (1.2, 1.6, "Три"), (1.6, 2.0, "четыре"),
        (2.0, 2.5, "пять."), (2.7, 3.0, "Шесть"), (3.0, 3.5, "семь."),
    ]
    index = TranscriptIndex.from_transcript({
        "segments": [{"start": 0.0, "end": 3.5, "text": "", "words": [
            {"start": s, "end": e, "word": w} for s, e, w in words
        ]}],
        "sentences": [{"start": 0.0, "end": 3.5, "text": "…"}],
    })
    edges = ClipEdges(lead_in_s=0.6, tail_s=0.3, guard_s=0.1, max_extend_s=4.0)
    # Starts on "четыре": grows back to "Три" (0.4 s) rather than to "Раз".
    [interval] = padded_intervals([(1.6, 2.5)], 5.0, index=index, edges=edges)
    assert interval == (pytest.approx(1.1), pytest.approx(2.6))


def test_dangling_fragments_are_dropped() -> None:
    edges = ClipEdges(max_extend_s=0.5, max_trim_s=1.5)
    # Opens on "сказал." (the tail of a 2 s sentence): dropped, the clip
    # opens on "А".
    [interval] = padded_intervals([(11.6, 14.5)], 5.0, index=INDEX, edges=edges)
    assert interval == (pytest.approx(12.4), pytest.approx(14.5))
    # Closes on "А" (the head of a 1.5 s sentence): dropped.
    [interval] = padded_intervals([(10.0, 13.5)], 5.0, index=INDEX, edges=edges)
    assert interval == (pytest.approx(9.4), pytest.approx(12.3))


def test_a_trim_never_eats_most_of_the_clip() -> None:
    edges = ClipEdges(max_extend_s=0.0, max_trim_s=10.0)
    [interval] = padded_intervals([(11.6, 13.5)], 5.0, index=INDEX, edges=edges)
    assert interval == (pytest.approx(11.6), pytest.approx(13.5))


def test_analysis_snap_sees_sentences_inside_a_transcript_group() -> None:
    words = [(0.0, 0.5, "Раз"), (0.5, 1.0, "два."), (1.2, 1.6, "Три"), (1.6, 2.0, "четыре.")]
    index = TranscriptIndex.from_transcript({
        "segments": [{"start": 0.0, "end": 2.0, "text": "", "words": [
            {"start": s, "end": e, "word": w} for s, e, w in words
        ]}],
        "sentences": [{"start": 0.0, "end": 2.0, "text": "…"}],
    })
    assert index.snap_start(1.4, max_shift=1.0) == pytest.approx(1.2)
    assert index.snap_end(0.8, max_shift=1.0) == pytest.approx(1.0)
