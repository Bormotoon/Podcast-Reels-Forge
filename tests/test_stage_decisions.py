"""RU: Решения cleanup/judge по candidate_id, батчи judge и обзор эпизода.

EN: Cleanup/judge decisions by candidate_id, judge batching, and the episode
overview.
"""

from __future__ import annotations

import json

from podcast_reels_forge.analysis.contracts import MomentRecord, coerce_moment_record
from podcast_reels_forge.analysis.decisions import (
    CLEANUP_EDITABLE,
    JUDGE_EDITABLE,
    apply_stage_decisions,
)
from podcast_reels_forge.analysis.transcript_index import TranscriptIndex
from podcast_reels_forge.stages.analyze_stage import (
    RetryBudget,
    build_cleanup_payload,
    build_transcript_digest,
    episode_context_cache_key,
    format_episode_context,
    stratified_batches,
    target_candidate_range,
)


def _record(cid: str, start: float, end: float, *, quote: str = "", score: float = 7.0) -> MomentRecord:
    record = coerce_moment_record(
        {
            "candidate_id": cid,
            "start": start,
            "end": end,
            "quote": quote or f"настоящая цитата номер {cid}",
            "why": "доказательство",
            "score": score,
        },
    )
    assert record is not None
    return record


INPUTS = [_record("a", 0, 40), _record("b", 30, 70), _record("c", 200, 240)]


# -- decisions ---------------------------------------------------------------


def test_judge_cannot_move_bounds_or_rewrite_the_quote() -> None:
    response = {
        "reviews": [
            {
                "candidate_id": "a",
                "keep": True,
                "score": 9,
                "title": "Новый заголовок",
                "quote": "переписанная цитата",
                "start": 500,
                "end": 560,
            },
        ],
    }
    outcome = apply_stage_decisions(INPUTS, response, stage="judge", editable=JUDGE_EDITABLE)
    judged = {r.candidate_id: r for r in outcome.records}["a"]
    assert judged.title == "Новый заголовок"
    assert judged.quote == INPUTS[0].quote
    assert (judged.start, judged.end) == (0.0, 40.0)
    assert judged.score == 9.0


def test_keep_false_drops_and_unmentioned_survive() -> None:
    """Truncation cuts the tail of the list; that is not a verdict."""
    response = {"decisions": [{"candidate_id": "b", "keep": False}]}
    outcome = apply_stage_decisions(INPUTS, response, stage="cleanup", editable=CLEANUP_EDITABLE)
    assert [r.candidate_id for r in outcome.records] == ["a", "c"]
    assert outcome.dropped == ["b"]
    assert outcome.unmentioned == 2


def test_merge_widens_over_overlapping_sources_and_records_lineage() -> None:
    response = {"decisions": [{"candidate_id": "a", "keep": True, "merge_ids": ["b"]}]}
    outcome = apply_stage_decisions(INPUTS, response, stage="cleanup", editable=CLEANUP_EDITABLE)
    merged = {r.candidate_id: r for r in outcome.records}
    assert "b" not in merged
    assert (merged["a"].start, merged["a"].end) == (0.0, 70.0)
    assert merged["a"].merged_ids == ("b",)
    assert merged["a"].quote == INPUTS[0].quote


def test_merge_of_a_disjoint_candidate_does_not_glue_footage() -> None:
    response = {"decisions": [{"candidate_id": "a", "keep": True, "merge_ids": ["c"]}]}
    outcome = apply_stage_decisions(INPUTS, response, stage="cleanup", editable=CLEANUP_EDITABLE)
    merged = {r.candidate_id: r for r in outcome.records}
    assert (merged["a"].start, merged["a"].end) == (0.0, 40.0)
    assert "c" not in merged


def test_unknown_ids_are_ignored_not_invented() -> None:
    response = {"reviews": [{"candidate_id": "zzz", "keep": True, "score": 10}]}
    outcome = apply_stage_decisions(INPUTS, response, stage="judge", editable=JUDGE_EDITABLE)
    assert outcome.untraceable == 1
    assert [r.candidate_id for r in outcome.records] == ["a", "b", "c"]


def test_empty_response_keeps_the_batch() -> None:
    outcome = apply_stage_decisions(INPUTS, [], stage="judge", editable=JUDGE_EDITABLE)
    assert outcome.records == INPUTS


def test_legacy_full_records_are_traced_and_evidence_restored() -> None:
    """Old custom prompts answer with full records; they still apply safely."""
    response = {
        "moments": [
            {"start": 1, "end": 39, "title": "Старый формат", "quote": INPUTS[0].quote, "score": 8},
        ],
    }
    outcome = apply_stage_decisions(INPUTS, response, stage="judge", editable=JUDGE_EDITABLE)
    assert outcome.mode == "legacy"
    assert [r.candidate_id for r in outcome.records] == ["a"]
    assert (outcome.records[0].start, outcome.records[0].end) == (0.0, 40.0)
    assert outcome.records[0].title == "Старый формат"


def test_string_booleans_are_understood() -> None:
    response = {"decisions": [{"candidate_id": "a", "keep": "false"}]}
    outcome = apply_stage_decisions(INPUTS, response, stage="cleanup", editable=CLEANUP_EDITABLE)
    assert "a" in outcome.dropped


def test_cleanup_payload_carries_evidence_not_prose() -> None:
    payload = build_cleanup_payload(INPUTS)
    assert payload[0]["candidate_id"] == "a"
    assert payload[0]["quote"] == INPUTS[0].quote
    assert "title" not in payload[0]
    assert "hashtags" not in payload[0]


# -- judge batching ----------------------------------------------------------


def test_stratified_batches_spread_quality_evenly() -> None:
    records = [_record(str(i), i * 100.0, i * 100.0 + 40.0, score=float(i)) for i in range(30)]
    batches = stratified_batches(records, 14)
    assert len(batches) == 3
    assert sum(len(batch) for batch in batches) == 30
    # Each batch gets one of the top three.
    tops = {max(r.score for r in batch) for batch in batches}
    assert tops == {29.0, 28.0, 27.0}


def test_stratified_batches_of_nothing() -> None:
    assert stratified_batches([], 14) == []


# -- scout target range ------------------------------------------------------


def test_target_candidate_range_scales_with_chunk_length() -> None:
    assert target_candidate_range(600) == "4-8"
    assert target_candidate_range(60) == "1-3"


# -- retry budget ------------------------------------------------------------


def test_retry_budget_is_shared_and_finite() -> None:
    budget = RetryBudget(2)
    assert budget.take() and budget.take()
    assert not budget.take()
    assert (budget.used, budget.refused) == (2, 1)


# -- episode overview --------------------------------------------------------


def _index() -> TranscriptIndex:
    sentences = []
    for i in range(120):
        text = f"Обычная фраза номер {'x' * 5} про школу."
        if i == 77:
            text = "Мы потеряли 40 процентов учеников!"
        sentences.append({"start": i * 30.0, "end": i * 30.0 + 30.0, "text": text})
    return TranscriptIndex.from_transcript({"sentences": sentences})


def test_digest_picks_up_signal_sentences_an_even_sample_misses() -> None:
    digest = build_transcript_digest(_index(), max_chars=1500)
    assert "40 процентов" in digest
    assert len(digest) <= 1501


def test_digest_covers_the_whole_episode() -> None:
    digest = build_transcript_digest(_index(), max_chars=800)
    lines = digest.splitlines()
    assert lines[0].startswith("[00:00]")
    # The last even-sample pick is in the second half of the hour.
    assert any(line.startswith(("[4", "[5")) for line in lines)


def test_context_cache_key_tracks_every_input() -> None:
    base = {"digest": "d", "prompt": "p", "model": "m", "lang": "ru"}
    key = episode_context_cache_key(**base)
    for field in base:
        assert episode_context_cache_key(**{**base, field: "changed"}) != key


def test_context_limits_are_rendered() -> None:
    text = format_episode_context({"summary": "S", "context_limits": ["кто такой Иван"]})
    assert "кто такой Иван" in text


def test_context_cache_is_json_serialisable() -> None:
    json.dumps({"cache_key": episode_context_cache_key(digest="d", prompt="p", model="m", lang="ru")})
