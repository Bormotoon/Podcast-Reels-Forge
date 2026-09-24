"""RU: Цитата как доказательство: поиск, пороги, границы и метаданные.

EN: The quote as evidence: lookup, thresholds, boundaries and metadata.
"""

from __future__ import annotations

from typing import Any

import pytest

from podcast_reels_forge.analysis.contracts import (
    MomentRecord,
    coerce_moment_record,
    has_quote_evidence,
)
from podcast_reels_forge.analysis.metadata import (
    MissingEvidenceError,
    finalize_moment_list,
    finalize_moment_metadata,
)
from podcast_reels_forge.analysis.transcript_index import TranscriptIndex
from podcast_reels_forge.analysis.validation import (
    METHOD_EXACT,
    METHOD_FUZZY,
    METHOD_NONE,
    apply_quote_verification,
    enforce_quote_containment,
    quote_verification_settings,
    snap_record_boundaries,
    split_by_quote_ratio,
    verify_quote,
)

SPOKEN = "мы потеряли половину класса за один год и никто этого не заметил"


def _record(start: float, end: float, *, quote: str = "", **extra: Any) -> MomentRecord:
    record = coerce_moment_record(
        {
            "start": start,
            "end": end,
            "title": "Момент",
            "quote": quote,
            "why": "Причина",
            "score": 8.0,
            **extra,
        },
    )
    assert record is not None
    return record


def _index(text: str = SPOKEN, *, start: float = 100.0, word_seconds: float = 0.5) -> TranscriptIndex:
    words: list[dict[str, Any]] = []
    cursor = start
    for word in text.split():
        words.append({"start": cursor, "end": cursor + word_seconds, "word": word})
        cursor += word_seconds
    return TranscriptIndex.from_transcript(
        {
            "segments": [{"start": start, "end": cursor, "text": text, "words": words}],
            "sentences": [{"start": start, "end": cursor, "text": text}],
        },
    )


# -- matcher -----------------------------------------------------------------


def test_exact_match_reports_method_and_span() -> None:
    match = verify_quote(_record(100, 110, quote="половину класса за один год"), _index())
    assert match.method == METHOD_EXACT
    assert match.ratio == 1.0
    # "половину" is the 3rd word (starts at 101.0), "год" the 7th (ends 103.5).
    assert (match.start, match.end) == (101.0, 103.5)
    assert match.missing_tokens == 0


def test_inflected_quote_falls_back_to_fuzzy() -> None:
    """A recognition slip at a word ending is tolerated, but labelled."""
    match = verify_quote(_record(100, 110, quote="половина класса за один год"), _index())
    assert match.method == METHOD_FUZZY
    assert match.ratio >= 0.75
    assert match.found


def test_frequent_short_words_do_not_fake_a_match() -> None:
    """Character similarity used to reward windows of common short words."""
    match = verify_quote(_record(100, 110, quote="и не за и не за и не"), _index())
    assert match.ratio < 0.55
    assert match.method == METHOD_NONE
    assert not match.found


def test_quote_longer_than_the_haystack_is_not_the_whole_range() -> None:
    long_quote = SPOKEN + " а потом директор ушёл на пенсию и школу закрыли навсегда"
    match = verify_quote(_record(100, 106, quote=long_quote), _index(), tolerance_s=0.0)
    assert match.ratio < 0.75
    assert match.missing_tokens > 0


def test_repeated_phrase_prefers_the_occurrence_inside_the_clip() -> None:
    text = "это важно сказать сейчас потому что это важно сказать сейчас"
    index = _index(text, start=0.0, word_seconds=1.0)
    # The second occurrence starts at word 6 (t=6s).
    match = verify_quote(_record(6.0, 11.0, quote="это важно сказать"), index)
    assert match.method == METHOD_EXACT
    assert match.start == 6.0


def test_verification_works_without_word_timings() -> None:
    """Older transcripts only have sentences; interpolation still grounds quotes."""
    index = TranscriptIndex.from_transcript(
        {"sentences": [{"start": 0.0, "end": 13.0, "text": SPOKEN}]},
    )
    match = verify_quote(_record(0, 13, quote="половину класса за один год"), index)
    assert match.method == METHOD_EXACT
    assert match.found


def test_annotation_records_method_and_located_span() -> None:
    verified = apply_quote_verification(
        [_record(100, 110, quote="половину класса за один год")], _index(),
    )[0]
    assert verified.quote_match_method == METHOD_EXACT
    assert verified.quote_start == 101.0
    assert verified.quote_end == 103.5
    round_tripped = coerce_moment_record(verified.to_dict())
    assert round_tripped is not None
    assert round_tripped.quote_start == 101.0


# -- thresholds --------------------------------------------------------------


def test_low_ratio_is_a_hard_reject_but_unmeasured_is_kept() -> None:
    measured_bad = _record(0, 10, quote="x y", quote_match_ratio=0.3)
    measured_ok = _record(20, 30, quote="x y", quote_match_ratio=0.9)
    unmeasured = _record(40, 50, quote="x y")
    kept, rejected = split_by_quote_ratio([measured_bad, measured_ok, unmeasured], 0.75)
    assert kept == [measured_ok, unmeasured]
    assert rejected == [measured_bad]


def test_final_threshold_is_never_looser_than_the_pool_threshold() -> None:
    settings = quote_verification_settings({"min_ratio": 0.8, "min_final_ratio": 0.6})
    assert settings["min_final_ratio"] == 0.8
    defaults = quote_verification_settings({})
    assert (defaults["min_ratio"], defaults["min_final_ratio"]) == (0.55, 0.75)


# -- boundaries --------------------------------------------------------------


def test_containment_widens_a_clip_back_over_its_quote() -> None:
    record = _record(105, 120, quote="a b", quote_start=101.0, quote_end=104.0)
    kept, rejected = enforce_quote_containment([record], duration=200.0)
    assert not rejected
    assert (kept[0].start, kept[0].end) == (101.0, 120.0)


def test_containment_rejects_a_quote_past_the_episode_end() -> None:
    record = _record(100, 120, quote="a b", quote_start=110.0, quote_end=130.0)
    kept, rejected = enforce_quote_containment([record], duration=125.0)
    assert not kept
    assert rejected == [record]


def test_snapping_never_trims_into_the_quote() -> None:
    """An inward sentence edge is attractive, but not past the evidence."""
    index = TranscriptIndex.from_transcript(
        {
            "sentences": [
                {"start": 90.0, "end": 100.0, "text": "раз два"},
                {"start": 100.5, "end": 120.0, "text": "три четыре"},
            ],
        },
    )
    free = snap_record_boundaries(_record(99.8, 119.0), index, max_shift_s=3.0)
    assert free.start == 100.5, "without a quote the dangling fragment is dropped"

    pinned = snap_record_boundaries(
        _record(99.8, 119.0, quote_start=100.0, quote_end=110.0), index, max_shift_s=3.0,
    )
    assert pinned.start <= 100.0, "the quote starts at 100.0 and must stay inside"


# -- evidence contract & metadata -------------------------------------------


def test_has_quote_evidence_needs_real_words() -> None:
    assert not has_quote_evidence({"quote": ""})
    assert not has_quote_evidence({"quote": "да"})
    assert has_quote_evidence({"quote": "это правда"})


def test_metadata_never_turns_a_hook_into_a_quote() -> None:
    with pytest.raises(MissingEvidenceError):
        finalize_moment_metadata({"start": 0, "end": 30, "title": "T", "hook": "Хук", "score": 7})
    assert finalize_moment_list([{"start": 0, "end": 30, "title": "T", "hook": "Хук", "score": 7}]) == []

    legacy = finalize_moment_metadata(
        {"start": 0, "end": 30, "title": "T", "hook": "Хук", "score": 7}, require_quote=False,
    )
    assert legacy.quote == "", "the legacy path tolerates a gap but never fills it"


def test_generated_metadata_is_marked_as_derived() -> None:
    record = finalize_moment_metadata(
        {"start": 0, "end": 30, "quote": "мы потеряли половину класса", "score": 7},
    )
    assert record.quote == "мы потеряли половину класса"
    for key in ("title", "hook", "why", "caption", "hashtags"):
        assert key in record.derived_fields
    assert "quote" not in record.derived_fields


def test_model_written_metadata_is_not_marked_derived() -> None:
    record = finalize_moment_metadata(
        {
            "start": 0,
            "end": 30,
            "quote": "мы потеряли половину класса",
            "title": "Половина класса",
            "hook": "Куда делись дети?",
            "why": "Шокирующая цифра",
            "score": 7,
        },
    )
    assert "title" not in record.derived_fields
    assert "hook" not in record.derived_fields


def test_crop_confidence_is_gone_and_does_not_leak_back() -> None:
    record = coerce_moment_record(
        {"start": 0, "end": 30, "title": "T", "quote": "a b", "crop_confidence": 0.9},
    )
    assert record is not None
    assert "crop_confidence" not in record.to_dict()


def test_scout_evidence_is_read_as_why() -> None:
    record = coerce_moment_record(
        {"start": 0, "end": 30, "quote": "a b c", "evidence": "сильная развязка", "score": 7},
    )
    assert record is not None
    assert record.why == "сильная развязка"
