"""RU: Тесты масштабирования квот от хронометража.

EN: Tests for duration-based clip quota scaling.
"""

from __future__ import annotations

from podcast_reels_forge.stages.analyze_stage import (
    _build_requirements_text,
    scale_quotas_to_duration,
)

# The default config mix: stories 2, reels 3, long reels 1, highlight moments 5.
DEFAULT_MIX = {"story": 2, "reel": 3, "long_reel": 1, "highlight": 5}


def test_one_hour_yields_clips_per_hour_total() -> None:
    scaled = scale_quotas_to_duration(
        DEFAULT_MIX, duration_s=3600.0, clips_per_hour=10.0,
    )
    assert sum(scaled.values()) == 10


def test_total_scales_with_runtime_not_per_hour_buckets() -> None:
    """1.5 hours at 10/hour is 15 clips — computed from the total, so a
    2h42m episode gives round(2.7 * 10) = 27, not 2 buckets of 10."""
    assert sum(scale_quotas_to_duration(
        DEFAULT_MIX, duration_s=5400.0, clips_per_hour=10.0,
    ).values()) == 15
    assert sum(scale_quotas_to_duration(
        DEFAULT_MIX, duration_s=9720.0, clips_per_hour=10.0,
    ).values()) == 27


def test_short_episode_still_yields_at_least_one_clip() -> None:
    scaled = scale_quotas_to_duration(
        DEFAULT_MIX, duration_s=120.0, clips_per_hour=10.0,
    )
    assert sum(scaled.values()) == 1


def test_mix_proportions_are_roughly_preserved() -> None:
    """The configured counts act as the type mix, not absolute numbers."""
    scaled = scale_quotas_to_duration(
        DEFAULT_MIX, duration_s=2 * 3600.0, clips_per_hour=11.0,
    )
    # 22 clips at the 2:3:1:5 mix is exactly double the base.
    assert scaled == {"story": 4, "reel": 6, "long_reel": 2, "highlight": 10}


def test_zero_disables_scaling() -> None:
    assert scale_quotas_to_duration(
        DEFAULT_MIX, duration_s=3600.0, clips_per_hour=0.0,
    ) == DEFAULT_MIX


def test_unknown_duration_disables_scaling() -> None:
    assert scale_quotas_to_duration(
        DEFAULT_MIX, duration_s=0.0, clips_per_hour=10.0,
    ) == DEFAULT_MIX


def test_empty_mix_falls_back_to_reels() -> None:
    scaled = scale_quotas_to_duration(
        {"story": 0, "reel": 0}, duration_s=3600.0, clips_per_hour=10.0,
    )
    assert scaled["reel"] == 10


def test_requirements_text_reflects_scaled_quotas() -> None:
    """The model is asked for the same numbers the selection will enforce."""
    processing_conf = {
        "clips": {
            "stories": {"count": 2, "max_duration": 15},
            "reels": {"count": 3, "max_duration": 60},
            "long_reels": {"count": 1, "max_duration": 180},
            "highlights": {"count": 1, "moments_count": 5},
        },
    }
    scaled = scale_quotas_to_duration(
        DEFAULT_MIX, duration_s=2 * 3600.0, clips_per_hour=11.0,
    )
    text = _build_requirements_text(processing_conf, quotas=scaled)
    assert "Stories: 4 clips up to 15s" in text
    assert "Reels: 6 clips up to 60s" in text
    assert "Long reels: 2 clips up to 180s" in text
    assert "Highlights: 10 moments" in text


def test_requirements_text_omits_zeroed_types() -> None:
    """A type scaled down to zero should not be asked for at all."""
    processing_conf = {
        "clips": {
            "stories": {"count": 1, "max_duration": 15},
            "reels": {"count": 9, "max_duration": 60},
        },
    }
    text = _build_requirements_text(
        processing_conf, quotas={"story": 0, "reel": 3},
    )
    assert "Stories" not in text
    assert "Reels: 3 clips" in text


def test_requirements_text_without_quotas_matches_legacy_config_path() -> None:
    processing_conf = {
        "clips": {"reels": {"count": 4, "max_duration": 60}},
    }
    assert "Reels: 4 clips up to 60s" in _build_requirements_text(processing_conf)
