"""RU: Тесты дедупликации, квот и отбора кандидатов.

EN: Tests for candidate de-duplication, quotas and selection.
"""

from __future__ import annotations

from typing import Any

from podcast_reels_forge.analysis.contracts import MomentRecord, coerce_moment_record
from podcast_reels_forge.analysis.ranking import (
    dedupe_moments,
    rank_moments,
    ranking_value,
)


def _record(
    start: float,
    end: float,
    *,
    title: str = "Момент",
    clip_type: str = "reel",
    score: float = 8.0,
) -> MomentRecord:
    record = coerce_moment_record(
        {
            "start": start,
            "end": end,
            "title": title,
            "quote": f"Цитата для {title}",
            "why": "Понятная причина, почему это работает",
            "score": score,
            "clip_type": clip_type,
        },
    )
    assert record is not None
    return record


def _buckets(records: list[MomentRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.clip_type] = counts.get(record.clip_type, 0) + 1
    return counts


def test_zero_quota_excludes_the_bucket() -> None:
    """A quota of 0 must exclude that clip type, not leave it unlimited."""
    records = [
        _record(0, 12, title="Стори A", clip_type="story"),
        _record(20, 32, title="Стори B", clip_type="story"),
        _record(40, 90, title="Рилс", clip_type="reel"),
    ]
    selected = rank_moments(records, clip_type_quotas={"story": 0, "reel": 2})
    assert _buckets(selected) == {"reel": 1}


def test_bucket_missing_from_quotas_is_excluded() -> None:
    """Clip types absent from the quota mapping are not selected."""
    records = [
        _record(0, 12, title="Стори", clip_type="story"),
        _record(40, 90, title="Рилс", clip_type="reel"),
    ]
    selected = rank_moments(records, clip_type_quotas={"reel": 3})
    assert _buckets(selected) == {"reel": 1}


def test_quota_limits_count_per_bucket() -> None:
    """Selection stops once a bucket reaches its quota."""
    records = [
        _record(0, 50, title="Рилс 1"),
        _record(60, 110, title="Рилс 2"),
        _record(120, 170, title="Рилс 3"),
    ]
    selected = rank_moments(records, clip_type_quotas={"reel": 2})
    assert len(selected) == 2


def test_empty_quotas_fall_back_to_unlimited_reels() -> None:
    """Callers passing no quotas still get results (legacy behaviour)."""
    records = [_record(0, 50, title="Рилс 1"), _record(60, 110, title="Рилс 2")]
    assert len(rank_moments(records, clip_type_quotas={})) == 2
    assert len(rank_moments(records, clip_type_quotas={"reel": 0})) == 2


def test_overlapping_moments_are_not_both_selected() -> None:
    """Time-overlapping clips are mutually exclusive."""
    records = [
        _record(0, 60, title="Первый", score=9.0),
        _record(30, 90, title="Второй", score=8.0),
    ]
    selected = rank_moments(records, clip_type_quotas={"reel": 5})
    assert len(selected) == 1
    assert selected[0].title == "Первый"


def test_dedupe_drops_heavily_overlapping_duplicates() -> None:
    """The same moment scouted twice collapses into one record."""
    records = [
        _record(100, 150, title="Дубль", score=9.0),
        _record(102, 149, title="Дубль", score=7.0),
        _record(300, 350, title="Другой"),
    ]
    deduped = dedupe_moments(records)
    assert len(deduped) == 2


def test_dedupe_keeps_distinct_moments() -> None:
    """Non-overlapping candidates all survive."""
    records = [_record(0, 40), _record(100, 140), _record(200, 240)]
    assert len(dedupe_moments(records)) == 3


def test_ranking_preserves_the_model_score() -> None:
    """`score` stays on the LLM's 1-10 scale; ranking writes `priority`.

    The cut stage filters on `score` via quality_filters.min_score, so ranking
    must not repurpose that field.
    """
    records = [_record(0, 45, score=7.0)]
    selected = rank_moments(records, clip_type_quotas={"reel": 1})
    assert len(selected) == 1
    assert selected[0].score == 7.0
    assert selected[0].priority is not None
    assert selected[0].priority != 7.0


def test_ranking_is_idempotent() -> None:
    """Re-ranking an already-ranked list does not drift the priority.

    Ranking twice used to feed the combined priority back in as the next
    pass's base score, inflating it on every round.
    """
    records = [_record(0, 45, title="A"), _record(100, 150, title="B")]
    once = rank_moments(records, clip_type_quotas={"reel": 5})
    twice = rank_moments(once, clip_type_quotas={"reel": 5})

    assert [r.title for r in once] == [r.title for r in twice]
    for first, second in zip(once, twice):
        assert first.priority == second.priority
        assert first.score == second.score


def test_ranking_value_prefers_priority_then_score() -> None:
    """Ordering uses priority once present, and score before that."""
    unscored = _record(0, 45, score=6.0)
    assert ranking_value(unscored) == 6.0

    scored = rank_moments([unscored], clip_type_quotas={"reel": 1})[0]
    assert ranking_value(scored) == scored.priority


def test_priority_round_trips_through_to_dict() -> None:
    """The new field survives serialization back into a record."""
    ranked = rank_moments([_record(0, 45)], clip_type_quotas={"reel": 1})[0]
    restored = coerce_moment_record(ranked.to_dict())
    assert restored is not None
    assert restored.priority == ranked.priority
    assert restored.score == ranked.score
    assert "priority" not in restored.extra
