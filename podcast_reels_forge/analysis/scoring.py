"""Deterministic heuristics used by the staged analysis pipeline."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_QUESTION_RE = re.compile(r"[?？]")
_EXCLAMATION_RE = re.compile(r"[!！]")
_NUMERIC_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_SHORT_COMMON_RE = re.compile(r"\b(?:why|how|what|why|лучше|почему|как|что)\b", re.IGNORECASE)


def _word_count(text: str) -> int:
    return len([part for part in re.split(r"\s+", text.strip()) if part])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded_score(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    if value < lo:
        return max(0.0, value / max(lo, 1.0))
    if value > hi:
        return max(0.0, hi / max(value, 1.0))
    return 1.0


def hook_strength(moment: Mapping[str, Any]) -> float:
    hook = str(moment.get("hook", "") or moment.get("title", "")).strip()
    if not hook:
        return 0.0
    score = 0.0
    words = _word_count(hook)
    score += _bounded_score(words, 4, 14) * 0.45
    if _QUESTION_RE.search(hook):
        score += 0.12
    if _EXCLAMATION_RE.search(hook):
        score += 0.08
    if _NUMERIC_RE.search(hook):
        score += 0.08
    if _SHORT_COMMON_RE.search(hook):
        score += 0.08
    if len(hook) <= 80:
        score += 0.12
    if hook[:1].isupper():
        score += 0.07
    return min(1.0, score)


def readability_score(moment: Mapping[str, Any]) -> float:
    caption = str(moment.get("caption", "") or "").strip()
    quote = str(moment.get("quote", "") or "").strip()
    base = caption or quote
    if not base:
        return 0.0
    words = _word_count(base)
    score = _bounded_score(words, 8, 28) * 0.55
    if len(base) <= 220:
        score += 0.15
    if len(base) >= 30:
        score += 0.1
    if "," in base or "—" in base or ":" in base:
        score += 0.1
    if _QUESTION_RE.search(base) or _EXCLAMATION_RE.search(base):
        score += 0.05
    return min(1.0, score)


def completeness_score(moment: Mapping[str, Any]) -> float:
    title = str(moment.get("title", "") or "").strip()
    why = str(moment.get("why", "") or "").strip()
    quote = str(moment.get("quote", "") or "").strip()
    score = 0.0
    if title:
        score += 0.25
    if why:
        score += 0.25
    if quote:
        score += 0.2
    duration = max(
        0.01,
        _safe_float(moment.get("end", 0.0)) - _safe_float(moment.get("start", 0.0)),
    )
    score += _bounded_score(duration, 12, 180) * 0.3
    return min(1.0, score)


def speaker_focus_score(moment: Mapping[str, Any]) -> float:
    speaker = str(moment.get("speaker", "") or "").strip()
    speaker_conf = moment.get("speaker_confidence")
    score = 0.0
    if speaker:
        score += 0.7
    if isinstance(speaker_conf, (int, float)):
        score += max(0.0, min(0.3, float(speaker_conf) * 0.3))
    return min(1.0, score)


def clip_duration_score(moment: Mapping[str, Any], *, target_min: float, target_max: float) -> float:
    duration = max(
        0.0,
        _safe_float(moment.get("end", 0.0)) - _safe_float(moment.get("start", 0.0)),
    )
    if duration <= 0:
        return 0.0
    if target_min <= 0 or target_max <= 0 or target_max < target_min:
        return _bounded_score(duration, 10, 180)
    if duration < target_min:
        return max(0.0, duration / max(target_min, 1.0)) * 0.6
    if duration > target_max:
        return max(0.0, target_max / max(duration, 1.0)) * 0.6
    center = (target_min + target_max) / 2.0
    distance = abs(duration - center) / max(center, 1.0)
    return max(0.0, 1.0 - distance)


def combined_priority_score(
    moment: Mapping[str, Any],
    *,
    target_min: float,
    target_max: float,
) -> float:
    base_score = float(moment.get("score", 0.0) or 0.0)
    hook = hook_strength(moment)
    readable = readability_score(moment)
    complete = completeness_score(moment)
    speaker = speaker_focus_score(moment)
    duration = clip_duration_score(moment, target_min=target_min, target_max=target_max)
    total = (
        base_score * 0.55
        + hook * 1.8
        + readable * 1.2
        + complete * 1.0
        + speaker * 0.6
        + duration * 1.4
    )
    return round(total, 4)


def clip_type_target_bounds(clip_type: str) -> tuple[float, float]:
    clip = clip_type.lower()
    if "story" in clip:
        return 5.0, 15.0
    if "highlight" in clip or "hot" in clip:
        return 10.0, 30.0
    if "long" in clip:
        return 60.0, 180.0
    return 30.0, 60.0


def penalize_mid_thought(moment: Mapping[str, Any]) -> float:
    title = str(moment.get("title", "") or "").strip()
    quote = str(moment.get("quote", "") or "").strip()
    why = str(moment.get("why", "") or "").strip()
    penalty = 0.0
    if title and title.endswith(("...", "—", "-")):
        penalty += 0.3
    if quote and len(quote.split()) < 3:
        penalty += 0.15
    if why and len(why.split()) < 4:
        penalty += 0.1
    return penalty


def scoring_breakdown(
    moment: Mapping[str, Any],
    *,
    target_min: float,
    target_max: float,
) -> dict[str, float]:
    return {
        "hook_score": hook_strength(moment),
        "readability_score": readability_score(moment),
        "completeness_score": completeness_score(moment),
        "speaker_focus_score": speaker_focus_score(moment),
        "duration_score": clip_duration_score(
            moment,
            target_min=target_min,
            target_max=target_max,
        ),
        "mid_thought_penalty": penalize_mid_thought(moment),
    }
