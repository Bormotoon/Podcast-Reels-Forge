"""RU: Доделки по docs/ANALYSIS_REPORT.md после первой волны изменений.

EN: Follow-ups to docs/ANALYSIS_REPORT.md after the first round of changes:
padding that stays out of neighbouring clips, speaker turns in the digest,
the scout's token budget, transport metrics and the location of weak fuzzy
quote matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from podcast_reels_forge.analysis.transcript_index import TranscriptIndex
from podcast_reels_forge.analysis.validation import (
    METHOD_FUZZY,
    apply_quote_verification,
    enforce_quote_containment,
)
from podcast_reels_forge.stages.analyze_stage import (
    RetryBudget,
    build_transcript_digest,
    fit_scout_chunk_chars,
    llm_transport_metrics,
    speaker_turn_times,
)
from podcast_reels_forge.utils.clip_intervals import moment_bounds, padded_intervals
from tests.test_quote_evidence import _index, _record

# -- padding (render QA) -----------------------------------------------------


def test_isolated_clip_gets_full_padding() -> None:
    assert padded_intervals([(100.0, 130.0)], 5.0) == [(95.0, 135.0)]


def test_padding_never_reaches_into_a_neighbour() -> None:
    intervals = padded_intervals([(100.0, 130.0), (134.0, 160.0)], 5.0)
    # A 4 s gap is split in half between the two clips.
    assert intervals == [(95.0, 132.0), (132.0, 165.0)]
    assert intervals[0][1] <= intervals[1][0]


def test_no_padding_towards_an_already_overlapping_clip() -> None:
    intervals = padded_intervals([(100.0, 130.0), (125.0, 160.0)], 5.0)
    assert intervals == [(95.0, 130.0), (125.0, 165.0)]


def test_padding_is_clamped_at_zero_and_order_independent() -> None:
    first = padded_intervals([(2.0, 20.0), (60.0, 90.0)], 5.0)
    second = padded_intervals([(60.0, 90.0), (2.0, 20.0)], 5.0)
    assert first == [(0.0, 25.0), (55.0, 95.0)]
    assert second == [first[1], first[0]]


def test_moment_bounds_tolerates_garbage() -> None:
    assert moment_bounds({"start": "1.5", "end": 3}) == (1.5, 3.0)
    assert moment_bounds({"start": "x", "end": 3}) == (0.0, 0.0)


# -- digest: speaker turns ---------------------------------------------------


def test_speaker_turn_times() -> None:
    segments = [
        {"start": 0.0, "speaker": "A"},
        {"start": 5.0, "speaker": "A"},
        {"start": 9.0, "speaker": "B"},
        {"start": 12.0},
        {"start": 15.0, "speaker": "A"},
    ]
    assert speaker_turn_times(segments) == [9.0, 15.0]
    assert speaker_turn_times([{"start": 0.0}, {"start": 1.0}]) == []


def test_digest_includes_a_reply_an_even_sample_misses() -> None:
    sentences = [
        {"start": i * 30.0, "end": i * 30.0 + 30.0, "text": "Обычная фраза про школу."}
        for i in range(120)
    ]
    sentences[78]["text"] = "Нет, вот тут я категорически не согласен."
    index = TranscriptIndex.from_transcript({"sentences": sentences})

    without = build_transcript_digest(index, max_chars=1500)
    with_turns = build_transcript_digest(index, max_chars=1500, speaker_turns=[78 * 30.0])
    assert "категорически" not in without
    assert "категорически" in with_turns


# -- scout token budget ------------------------------------------------------


def _chunks(chars: int) -> list[Any]:
    return [SimpleNamespace(text="слово " * (chars // 6))]


def test_chunks_that_fit_are_left_alone() -> None:
    conf = {"service": {"ctx_size": 32768}}
    assert fit_scout_chunk_chars(
        _chunks(6000), max_chars=12000, prompt="p", llama_cpp_conf=conf, n_predict=2048,
    ) is None


def test_overflowing_chunks_are_shrunk_to_the_slot_budget() -> None:
    conf = {"service": {"ctx_size": 6144}}
    fitted = fit_scout_chunk_chars(
        _chunks(12000), max_chars=12000, prompt="p", llama_cpp_conf=conf, n_predict=2048,
    )
    assert fitted is not None and fitted < 12000

    # Parallel slots split the context, so the budget per request shrinks.
    split = fit_scout_chunk_chars(
        _chunks(12000),
        max_chars=12000,
        prompt="p",
        llama_cpp_conf={"service": {"ctx_size": 12288, "parallel": 2}},
        n_predict=2048,
    )
    assert split == fitted


def test_hopeless_budget_is_reported_not_shrunk() -> None:
    conf = {"service": {"ctx_size": 4096}}
    assert fit_scout_chunk_chars(
        _chunks(12000), max_chars=12000, prompt="p", llama_cpp_conf=conf, n_predict=4096,
    ) is None


def test_no_ctx_size_means_no_opinion() -> None:
    assert fit_scout_chunk_chars(
        _chunks(12000), max_chars=12000, prompt="p", llama_cpp_conf={}, n_predict=4096,
    ) is None


# -- transport metrics -------------------------------------------------------


@dataclass
class _FakeLlama:
    responses: int
    truncated: int
    retries: dict[str, int]


@dataclass
class _Wrapper:
    inner: Any


def test_transport_metrics_look_through_caching_wrappers() -> None:
    budget = RetryBudget(5)
    budget.take()
    metrics = llm_transport_metrics(
        [
            _Wrapper(_FakeLlama(10, 2, {"timeout": 1})),
            _FakeLlama(10, 0, {"timeout": 2, "http_503": 1}),
            object(),
        ],
        budget,
    )
    assert metrics["responses"] == 20
    assert metrics["truncated_at_n_predict"] == 2
    assert metrics["truncated_rate"] == 0.1
    assert metrics["retries_by_reason"] == {"timeout": 3, "http_503": 1, "invalid_json": 1}


# -- weak fuzzy matches keep their location -----------------------------------

WEAK_QUOTE = "ах потеряли шкаф стол за один стул и никто сего не заметил"


def test_weak_fuzzy_match_is_located_but_does_not_move_the_clip() -> None:
    [record] = apply_quote_verification([_record(101.0, 104.0, quote=WEAK_QUOTE)], _index())
    assert record.quote_match_method == METHOD_FUZZY
    assert record.quote_match_ratio is not None and 0.55 <= record.quote_match_ratio < 0.75
    assert record.quote_start is not None and record.quote_end is not None
    # Too weak to widen the clip over it...
    assert (record.start, record.end) == (101.0, 104.0)
    # ...but a lowered min_final_ratio still gets the containment check.
    [contained], lost = enforce_quote_containment([record], duration=3600.0)
    assert not lost
    assert contained.start <= record.quote_start and record.quote_end <= contained.end
