"""RU: Тесты валидации кандидатов против транскрипта.

EN: Tests for validating candidates against the transcript.
"""

from __future__ import annotations

from typing import Any

from podcast_reels_forge.analysis.contracts import MomentRecord, coerce_moment_record
from podcast_reels_forge.analysis.transcript_index import TranscriptIndex
from podcast_reels_forge.analysis.validation import (
    apply_quote_verification,
    clamp_record_to_window,
    clamp_records_to_episode,
    filter_nonoverlapping_outputs,
    snap_record_boundaries,
    verify_quote,
)

SPOKEN = "мы потеряли половину класса за один год и никто этого не заметил"


def _record(start: float, end: float, *, quote: str = "", title: str = "Момент") -> MomentRecord:
    record = coerce_moment_record(
        {
            "start": start,
            "end": end,
            "title": title,
            "quote": quote,
            "why": "Причина, по которой момент работает",
            "score": 8.0,
        },
    )
    assert record is not None
    return record


def _index(text: str = SPOKEN, *, start: float = 100.0, word_seconds: float = 0.5) -> TranscriptIndex:
    """An index where each word of `text` occupies `word_seconds`."""

    words: list[dict[str, Any]] = []
    cursor = start
    for word in text.split():
        words.append(
            {"start": cursor, "end": cursor + word_seconds, "word": word, "probability": 0.9},
        )
        cursor += word_seconds
    return TranscriptIndex.from_transcript(
        {
            "segments": [{"start": start, "end": cursor, "text": text, "words": words}],
            "sentences": [{"start": start, "end": cursor, "text": text}],
        },
    )


# -- window clamping ---------------------------------------------------------


def test_candidate_inside_the_window_is_untouched() -> None:
    record = _record(120, 165)
    assert clamp_record_to_window(record, 100, 200) is record


def test_slight_overshoot_is_clamped_to_the_window() -> None:
    """Scouts drift a little past their chunk; absorb it rather than drop it."""
    clamped = clamp_record_to_window(_record(98, 205), 100, 200, tolerance_s=10)
    assert clamped is not None
    assert (clamped.start, clamped.end) == (100.0, 200.0)


def test_candidate_far_outside_the_window_is_dropped() -> None:
    """A timecode from elsewhere in the episode is a hallucination."""
    assert clamp_record_to_window(_record(900, 950), 100, 200, tolerance_s=3) is None


def test_clamp_to_episode_trims_and_drops() -> None:
    records = [_record(10, 50), _record(90, 130), _record(200, 240)]
    clamped = clamp_records_to_episode(records, duration=100.0)
    assert [(r.start, r.end) for r in clamped] == [(10.0, 50.0), (90.0, 100.0)]


def test_clamp_to_episode_is_a_noop_without_duration() -> None:
    records = [_record(10, 50)]
    assert clamp_records_to_episode(records, duration=0.0) == records


# -- invented-output guard ---------------------------------------------------


def test_output_overlapping_no_input_is_dropped() -> None:
    """Filtering stages may merge and re-rate, but not invent."""
    inputs = [_record(100, 150), _record(300, 350)]
    outputs = [_record(110, 145, title="Из входа"), _record(900, 940, title="Выдуманный")]
    kept = filter_nonoverlapping_outputs(outputs, inputs)
    assert [r.title for r in kept] == ["Из входа"]


def test_guard_is_a_noop_without_inputs() -> None:
    outputs = [_record(100, 150)]
    assert filter_nonoverlapping_outputs(outputs, []) == outputs


# -- quote verification ------------------------------------------------------


def test_exact_quote_matches_and_locates_the_span() -> None:
    index = _index()
    record = _record(100, 130, quote="половину класса за один год")
    match = verify_quote(record, index)
    assert match.ratio > 0.95
    assert match.found


def test_quote_matches_despite_punctuation_and_case() -> None:
    """Verification compares letter content, not formatting."""
    index = _index()
    record = _record(100, 130, quote="Половину класса, за один год!")
    assert verify_quote(record, index).ratio > 0.9


def test_hallucinated_quote_scores_low() -> None:
    index = _index()
    record = _record(100, 130, quote="совершенно другой текст о котором никто не говорил")
    assert verify_quote(record, index).ratio < 0.55


def test_empty_quote_is_not_a_match() -> None:
    assert verify_quote(_record(100, 130, quote=""), _index()).ratio == 0.0


def test_verification_annotates_records() -> None:
    index = _index()
    records = [
        _record(100, 130, quote="половину класса за один год", title="Настоящая"),
        _record(100, 130, quote="выдуманная фраза про пингвинов", title="Выдуманная"),
    ]
    verified = apply_quote_verification(records, index)
    by_title = {r.title: r for r in verified}
    assert by_title["Настоящая"].quote_match_ratio > by_title["Выдуманная"].quote_match_ratio


def test_verification_can_widen_bounds_to_the_quote() -> None:
    """A trusted quote must fit inside the clip that claims it."""
    index = _index()
    # The quote sits late in the span, past the candidate's end.
    record = _record(100.0, 102.0, quote="никто этого не заметил")
    verified = apply_quote_verification(records=[record], index=index)[0]
    assert verified.end > record.end
    assert verified.start <= record.start


def test_verification_disabled_leaves_records_alone() -> None:
    records = [_record(100, 130, quote="половину класса")]
    assert apply_quote_verification(records, _index(), enabled=False) == records


# -- boundary snapping -------------------------------------------------------


def test_boundaries_snap_to_sentence_edges() -> None:
    """Clips should open and close on real speech boundaries."""
    index = _index(start=100.0)
    sentence_end = index.sentences[0].end
    # Start slightly inside the sentence, end slightly before it finishes.
    record = _record(101.2, sentence_end - 0.7)
    snapped = snap_record_boundaries(record, index, max_shift_s=3.0)
    assert snapped.start == 100.0
    assert snapped.end == sentence_end


def test_snapping_respects_the_shift_cap() -> None:
    """A distant boundary must not drag the clip somewhere else."""
    index = _index(start=100.0)
    record = _record(140.0, 145.0)
    snapped = snap_record_boundaries(record, index, max_shift_s=0.5)
    assert (snapped.start, snapped.end) == (140.0, 145.0)


def test_snapping_without_word_timings_is_a_noop() -> None:
    empty = TranscriptIndex.from_transcript({})
    record = _record(10, 50)
    assert snap_record_boundaries(record, empty, max_shift_s=3.0) is record
