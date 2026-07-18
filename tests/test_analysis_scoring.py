"""RU: Тесты детерминированных эвристик скоринга.

EN: Tests for the deterministic scoring heuristics.
"""

from __future__ import annotations

from typing import Any

import pytest

from podcast_reels_forge.analysis.scoring import (
    DEFAULT_SCORING_WEIGHTS,
    _SHORT_COMMON_RE,
    combined_priority_score,
    penalize_mid_thought,
    resolve_scoring_weights,
    scoring_breakdown,
)


def _moment(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "start": 100.0,
        "end": 145.0,
        "title": "Почему школы теряют учеников",
        "quote": "Мы потеряли половину класса за один год",
        "why": "Неожиданная цифра и понятный конфликт",
        "hook": "Почему школы теряют учеников каждый год",
        "caption": "Разбираем, почему из школы уходит половина класса за год.",
        "score": 8.0,
    }
    base.update(overrides)
    return base


def test_short_common_regex_has_no_duplicate_alternative() -> None:
    """The pattern should list each keyword once."""
    alternatives = _SHORT_COMMON_RE.pattern.split("(?:")[1].split(")")[0].split("|")
    assert len(alternatives) == len(set(alternatives))


def test_mid_thought_penalty_lowers_priority() -> None:
    """A title trailing off mid-thought must rank below a complete one."""
    complete = _moment()
    truncated = _moment(title="Почему школы теряют...")

    assert penalize_mid_thought(truncated) > 0.0
    assert penalize_mid_thought(complete) == 0.0

    complete_priority = combined_priority_score(complete, target_min=30, target_max=60)
    truncated_priority = combined_priority_score(truncated, target_min=30, target_max=60)
    assert truncated_priority < complete_priority


def test_mid_thought_penalty_matches_breakdown_and_weight() -> None:
    """The penalty reported in the breakdown is the one actually subtracted."""
    truncated = _moment(title="Обрывается на полуслове —")
    breakdown = scoring_breakdown(truncated, target_min=30, target_max=60)
    penalty = breakdown["mid_thought_penalty"]
    assert penalty > 0.0

    with_penalty = combined_priority_score(truncated, target_min=30, target_max=60)
    without_penalty = combined_priority_score(
        truncated,
        target_min=30,
        target_max=60,
        weights={"mid_thought": 0.0},
    )
    expected = penalty * DEFAULT_SCORING_WEIGHTS["mid_thought"]
    assert without_penalty - with_penalty == pytest.approx(expected, abs=1e-4)


def test_combined_priority_score_never_negative() -> None:
    """An overwhelming penalty floors the priority instead of going negative."""
    weak = _moment(title="...", quote="a", why="no", score=0.0, hook="", caption="")
    assert combined_priority_score(
        weak,
        target_min=30,
        target_max=60,
        weights={"mid_thought": 100.0},
    ) == 0.0


def test_resolve_scoring_weights_merges_and_ignores_garbage() -> None:
    """Known keys override; unknown or non-numeric entries are dropped."""
    weights = resolve_scoring_weights({"hook": 3.0, "nonsense": 1.0, "duration": "abc"})
    assert weights["hook"] == 3.0
    assert weights["duration"] == DEFAULT_SCORING_WEIGHTS["duration"]
    assert "nonsense" not in weights


def test_resolve_scoring_weights_defaults_on_empty() -> None:
    """No overrides yields a copy of the defaults."""
    assert resolve_scoring_weights(None) == DEFAULT_SCORING_WEIGHTS
    assert resolve_scoring_weights({}) == DEFAULT_SCORING_WEIGHTS
    resolve_scoring_weights({})["hook"] = 99.0
    assert DEFAULT_SCORING_WEIGHTS["hook"] == 1.8
