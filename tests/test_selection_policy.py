"""RU: Финальный отбор: MMR, политика пересечений, дубли по цитате, типы клипов.

EN: Final selection: MMR, the overlap policy, quote duplicates, clip types.
"""

from __future__ import annotations

from podcast_reels_forge.analysis.contracts import MomentRecord, coerce_moment_record
from podcast_reels_forge.analysis.ranking import (
    assign_clip_types,
    dedupe_moments,
    rank_moments,
)


def _record(
    start: float,
    end: float,
    *,
    title: str = "Момент",
    quote: str = "",
    clip_type: str = "reel",
    score: float = 8.0,
) -> MomentRecord:
    record = coerce_moment_record(
        {
            "start": start,
            "end": end,
            "title": title,
            "quote": quote or f"уникальная цитата для {title}",
            "why": "Понятная причина, почему это работает",
            "score": score,
            "clip_type": clip_type,
        },
    )
    assert record is not None
    return record


def test_a_small_overlap_no_longer_knocks_out_a_neighbour() -> None:
    """Snap and padding make good neighbours touch; that is not a duplicate."""
    records = [
        _record(0, 50, title="Первый про бюджет", score=9.0),
        _record(47, 97, title="Второй про учителей", score=8.0),
    ]
    assert len(rank_moments(records, clip_type_quotas={"reel": 5})) == 2
    strict = rank_moments(records, clip_type_quotas={"reel": 5}, max_overlap_ratio=0.0)
    assert len(strict) == 1


def test_heavy_overlap_is_still_refused() -> None:
    records = [
        _record(0, 60, title="Первый", score=9.0),
        _record(30, 90, title="Второй", score=8.0),
    ]
    assert len(rank_moments(records, clip_type_quotas={"reel": 5})) == 1


def test_same_quote_with_a_new_title_is_a_duplicate() -> None:
    quote = "мы потеряли половину класса за один год"
    records = [
        _record(100, 150, title="Половина класса", quote=quote, score=9.0),
        _record(140, 200, title="Куда ушли дети", quote=quote, score=8.0),
    ]
    deduped = dedupe_moments(records)
    assert [r.title for r in deduped] == ["Половина класса"]


def test_mmr_trades_a_little_quality_for_a_new_topic() -> None:
    records = [
        _record(0, 45, title="Школьная программа перегружена", quote="дети не успевают по программе", score=9.0),
        _record(100, 145, title="Программа школы перегружена", quote="по программе дети не успевают", score=9.0),
        _record(200, 245, title="Еда в фудкорте", quote="в фудкорте невозможно поесть", score=7.0),
    ]
    selected = rank_moments(records, clip_type_quotas={"reel": 2})
    assert any("фудкорт" in r.title.lower() for r in selected)


def test_clip_type_follows_the_final_duration() -> None:
    quotas = {"story": 2, "reel": 3, "long_reel": 1}
    records = [
        _record(0, 12, clip_type="reel"),        # too short for a reel -> story
        _record(100, 145, clip_type="reel"),     # a reel, keep
        _record(200, 320, clip_type="story"),    # 120 s -> long reel
    ]
    typed = assign_clip_types(records, quotas)
    assert [r.clip_type for r in typed] == ["story", "reel", "long_reel"]
    assert "clip_type" in typed[0].derived_fields
    assert "clip_type" not in typed[1].derived_fields


def test_clip_type_assignment_respects_disabled_buckets() -> None:
    typed = assign_clip_types([_record(0, 12)], {"reel": 3})
    assert typed[0].clip_type == "reel", "no story quota: nowhere else to go"
