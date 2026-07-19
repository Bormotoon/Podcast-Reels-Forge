"""RU: Тесты разнообразия тем и оценки против golden-разметки.

EN: Tests for topic diversity in selection and golden-set scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

from podcast_reels_forge.analysis.contracts import MomentRecord, coerce_moment_record
from podcast_reels_forge.analysis.ranking import rank_moments, topic_similarity
from podcast_reels_forge.scripts.evaluate_prompts import (
    default_golden_path,
    intervals_match,
    load_golden,
    score_against_golden,
)


def _record(start: float, end: float, title: str, quote: str = "") -> MomentRecord:
    record = coerce_moment_record(
        {
            "start": start,
            "end": end,
            "title": title,
            "quote": quote or f"Цитата про {title}",
            "why": "Причина, по которой момент работает",
            "score": 8.0,
        },
    )
    assert record is not None
    return record


# -- topic diversity ---------------------------------------------------------


def test_topic_similarity_detects_the_same_subject() -> None:
    same_a = _record(0, 45, "Школьная программа перегружена", "дети не успевают по программе")
    same_b = _record(100, 145, "Программа школы перегружена", "по программе дети не успевают")
    different = _record(200, 245, "Еда в фудкорте", "в фудкорте невозможно нормально поесть")

    assert topic_similarity(same_a, same_b) > topic_similarity(same_a, different)
    assert topic_similarity(same_a, different) < 0.2


def test_selection_prefers_varied_topics() -> None:
    """Four clips saying one thing is a worse set than four varied ones."""
    records = [
        _record(0, 45, "Школьная программа перегружена", "дети не успевают по программе"),
        _record(100, 145, "Программа школы перегружена", "по программе дети не успевают"),
        _record(200, 245, "Еда в фудкорте ужасна", "в фудкорте невозможно поесть"),
    ]
    selected = rank_moments(records, clip_type_quotas={"reel": 2})
    titles = [r.title for r in selected]

    assert len(titles) == 2
    assert any("фудкорт" in t.lower() for t in titles), (
        "the second slot should go to a different topic, not a near-duplicate"
    )


def test_deferred_duplicates_backfill_unused_quota() -> None:
    """Diversity must not leave quota unfilled when nothing else is available."""
    records = [
        _record(0, 45, "Школьная программа перегружена", "дети не успевают по программе"),
        _record(100, 145, "Программа школы перегружена", "по программе дети не успевают"),
    ]
    selected = rank_moments(records, clip_type_quotas={"reel": 2})
    assert len(selected) == 2, "a near-duplicate is better than an empty slot"


def test_diversity_can_be_disabled() -> None:
    records = [
        _record(0, 45, "Школьная программа перегружена", "дети не успевают по программе"),
        _record(100, 145, "Программа школы перегружена", "по программе дети не успевают"),
        _record(200, 245, "Еда в фудкорте ужасна", "в фудкорте невозможно поесть"),
    ]
    selected = rank_moments(
        records, clip_type_quotas={"reel": 2}, diversity_enabled=False,
    )
    assert len(selected) == 2


# -- golden set --------------------------------------------------------------


def test_intervals_match_uses_the_shorter_span() -> None:
    """A long clip containing a short golden moment has found it."""
    long_clip = {"start": 100.0, "end": 200.0}
    short_golden = {"start": 150.0, "end": 170.0}
    assert intervals_match(long_clip, short_golden)


def test_intervals_do_not_match_on_a_slight_touch() -> None:
    assert not intervals_match({"start": 100.0, "end": 160.0}, {"start": 155.0, "end": 220.0})


def test_intervals_do_not_match_when_disjoint() -> None:
    assert not intervals_match({"start": 0.0, "end": 50.0}, {"start": 300.0, "end": 350.0})


def test_intervals_match_tolerates_malformed_input() -> None:
    assert not intervals_match({"start": "x"}, {"start": 0.0, "end": 10.0})


def test_golden_scoring_reports_recall_and_precision() -> None:
    golden = [
        {"start": 100.0, "end": 150.0, "label": "must"},
        {"start": 300.0, "end": 350.0, "label": "good"},
        {"start": 500.0, "end": 550.0, "label": "ok"},
    ]
    predicted = [
        {"start": 105.0, "end": 155.0},   # matches the must-have
        {"start": 800.0, "end": 850.0},   # matches nothing
    ]
    metrics = score_against_golden(predicted, golden)

    assert metrics["recall_must"] == 1.0
    assert metrics["recall_all"] == round(1 / 3, 3)
    assert metrics["precision"] == 0.5


def test_golden_scoring_flags_a_missed_must_have() -> None:
    golden = [{"start": 100.0, "end": 150.0, "label": "must"}]
    metrics = score_against_golden([{"start": 800.0, "end": 850.0}], golden)
    assert metrics["recall_must"] == 0.0
    assert metrics["precision"] == 0.0


def test_golden_scoring_is_empty_without_a_reference() -> None:
    assert score_against_golden([{"start": 0.0, "end": 10.0}], []) == {}


def test_load_golden_accepts_both_shapes(tmp_path: Path) -> None:
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(
        json.dumps({"episode": "e", "moments": [{"start": 1.0, "end": 2.0}]}),
        encoding="utf-8",
    )
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps([{"start": 1.0, "end": 2.0}]), encoding="utf-8")

    assert len(load_golden(wrapped)) == 1
    assert len(load_golden(bare)) == 1


def test_load_golden_tolerates_missing_or_broken_files(tmp_path: Path) -> None:
    assert load_golden(tmp_path / "nope.json") == []
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_golden(broken) == []


def test_default_golden_path_ignores_the_proofread_suffix() -> None:
    """The reference belongs to the episode, not a transcript revision."""
    raw = default_golden_path(Path("output/ep/ep.json"))
    proofread = default_golden_path(Path("output/ep/ep.proofread.json"))
    assert raw == proofread
    assert raw.name == "ep.json"
    assert raw.parent.name == "golden"
