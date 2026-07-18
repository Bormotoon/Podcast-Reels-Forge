"""Deterministic heuristics used by the staged analysis pipeline."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_QUESTION_RE = re.compile(r"[?？]")
_EXCLAMATION_RE = re.compile(r"[!！]")
_NUMERIC_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_SHORT_COMMON_RE = re.compile(r"\b(?:why|how|what|лучше|почему|как|что)\b", re.IGNORECASE)

# RU: Веса факторов итогового приоритета. Переопределяются через
# processing.analysis.scoring.weights.
# EN: Weights of the combined priority factors. Overridable through
# processing.analysis.scoring.weights.
DEFAULT_SCORING_WEIGHTS: dict[str, float] = {
    "base": 0.55,
    "hook": 1.8,
    "readability": 1.2,
    "completeness": 1.0,
    "speaker": 0.6,
    "duration": 1.4,
    "mid_thought": 1.0,
    "quote": 0.8,
    "audio": 0.7,
    "speech_rate": 0.4,
}

# RU: Нейтральное значение для факторов, которые не удалось измерить.
# EN: Neutral stand-in for factors that could not be measured, so records with
# and without the signal stay comparable instead of the missing one losing.
NEUTRAL_FACTOR = 0.5


def quote_grounding_score(moment: Mapping[str, Any]) -> float:
    """RU: Насколько цитата подтверждена транскриптом.

    EN: How well the moment's quote is grounded in the transcript. Every other
    heuristic scores prose the model wrote about itself; this is the one
    factor tied to what was actually said.
    """

    ratio = moment.get("quote_match_ratio")
    if ratio is None:
        return NEUTRAL_FACTOR
    try:
        return max(0.0, min(1.0, float(ratio)))
    except (TypeError, ValueError):
        return NEUTRAL_FACTOR


def resolve_scoring_weights(overrides: Mapping[str, Any] | None = None) -> dict[str, float]:
    """RU: Сливает пользовательские веса с дефолтными, игнорируя мусор.

    EN: Merge caller-supplied weights over the defaults, ignoring unknown or
    non-numeric entries so a malformed config cannot break ranking.
    """

    weights = dict(DEFAULT_SCORING_WEIGHTS)
    if not overrides:
        return weights
    for key, value in overrides.items():
        if key not in weights:
            continue
        try:
            weights[key] = float(value)
        except (TypeError, ValueError):
            continue
    return weights


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


def audio_quality_score(moment: Mapping[str, Any]) -> float:
    """RU: Оценка по звуку: громкость и доля тишины на отрезке.

    EN: Score the span's audio: how loud it is and how much of it is silence.
    A quiet, pause-heavy stretch makes a limp clip regardless of how well the
    model described it. Neutral when nothing was measured.
    """

    energy = moment.get("audio_energy_db")
    silence = moment.get("audio_silence_ratio")
    if energy is None and silence is None:
        return NEUTRAL_FACTOR

    score = 0.0
    weight = 0.0
    if energy is not None:
        try:
            # Podcast speech typically sits around -25..-15 dBFS mean.
            loudness = (float(energy) + 35.0) / 20.0
            score += max(0.0, min(1.0, loudness)) * 0.6
        except (TypeError, ValueError):
            return NEUTRAL_FACTOR
        weight += 0.6
    if silence is not None:
        try:
            ratio = max(0.0, min(1.0, float(silence)))
        except (TypeError, ValueError):
            return NEUTRAL_FACTOR
        # Up to ~35% pauses is normal speech rhythm; beyond that it drags.
        score += (1.0 - max(0.0, (ratio - 0.35) / 0.65)) * 0.4
        weight += 0.4

    return round(score / weight, 4) if weight else NEUTRAL_FACTOR


def speech_rate_score(moment: Mapping[str, Any]) -> float:
    """RU: Темп речи: слишком медленно — скучно, слишком быстро — не читается.

    EN: Speech rate. Too slow drags; too fast outruns the burned subtitles.
    Neutral when the transcript carried no word timings.
    """

    rate = moment.get("speech_rate_wps")
    if rate is None:
        return NEUTRAL_FACTOR
    try:
        value = float(rate)
    except (TypeError, ValueError):
        return NEUTRAL_FACTOR
    if value <= 0:
        return 0.0
    return round(_bounded_score(value, 1.6, 3.4), 4)


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
    weights: Mapping[str, Any] | None = None,
) -> float:
    """RU: Итоговый приоритет момента для ранжирования.

    EN: Combined ranking priority for a moment. This is deliberately not the
    moment's ``score``: ``score`` stays on the LLM's 1-10 scale that the cut
    stage filters on, while this value only orders candidates against each
    other.
    """

    w = resolve_scoring_weights(weights)
    base_score = float(moment.get("score", 0.0) or 0.0)
    hook = hook_strength(moment)
    readable = readability_score(moment)
    complete = completeness_score(moment)
    speaker = speaker_focus_score(moment)
    duration = clip_duration_score(moment, target_min=target_min, target_max=target_max)
    total = (
        base_score * w["base"]
        + hook * w["hook"]
        + readable * w["readability"]
        + complete * w["completeness"]
        + speaker * w["speaker"]
        + duration * w["duration"]
        + quote_grounding_score(moment) * w["quote"]
        + audio_quality_score(moment) * w["audio"]
        + speech_rate_score(moment) * w["speech_rate"]
        # Clips that open or close mid-thought are the most common complaint
        # about auto-cut reels, so the penalty has to actually land here.
        - penalize_mid_thought(moment) * w["mid_thought"]
    )
    return round(max(0.0, total), 4)


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
        "quote_grounding_score": quote_grounding_score(moment),
        "audio_quality_score": audio_quality_score(moment),
        "speech_rate_score": speech_rate_score(moment),
        "mid_thought_penalty": penalize_mid_thought(moment),
    }
