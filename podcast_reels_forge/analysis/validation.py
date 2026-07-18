"""RU: Проверка кандидатов против транскрипта.

Модель может вернуть таймкод за пределами куска, выдумать момент, которого не
было во входе, или пересказать «цитату» своими словами. Здесь всё это
ловится: границы поджимаются к реальным, выдуманные записи отбрасываются, а
цитаты сверяются с текстом на своём же отрезке.

EN: Validate candidates against the transcript.

A model can return a timecode outside its chunk, invent a moment that was
never in its input, or paraphrase a "quote" it claims is verbatim. This
module catches all three: bounds are clamped to real ones, invented records
are dropped, and quotes are checked against the text of their own span.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from podcast_reels_forge.analysis.contracts import MomentRecord, coerce_moment_record
from podcast_reels_forge.analysis.transcript_index import (
    TranscriptIndex,
    normalized_tokens,
)

# RU: Минимальная длительность клипа после клампа — короче уже не клип.
# EN: Shortest clip worth keeping after clamping.
_MIN_CLIP_SECONDS = 1.0

# Quote matches at or above this ratio are trusted enough to move the clip
# boundaries onto the matched span.
_REFINE_MIN_RATIO = 0.75


def _replace(record: MomentRecord, **changes: Any) -> MomentRecord | None:
    payload = {**record.to_dict(), **changes}
    return coerce_moment_record(payload)


def clamp_record_to_window(
    record: MomentRecord,
    window_start: float,
    window_end: float,
    *,
    tolerance_s: float = 3.0,
) -> MomentRecord | None:
    """Clamp a candidate to the window it was found in.

    Scout models routinely drift a little past the chunk they were shown, and
    occasionally hallucinate a timecode from elsewhere in the episode. A small
    tolerance absorbs the former; anything beyond it is dropped.
    """

    low = window_start - max(0.0, tolerance_s)
    high = window_end + max(0.0, tolerance_s)
    if record.end <= low or record.start >= high:
        return None

    start = min(max(record.start, window_start), window_end)
    end = min(max(record.end, window_start), window_end)
    if end - start < _MIN_CLIP_SECONDS:
        return None
    if start == record.start and end == record.end:
        return record
    return _replace(record, start=start, end=end)


def clamp_records_to_episode(
    records: Sequence[MomentRecord],
    duration: float,
) -> list[MomentRecord]:
    """Clamp final moments to the episode, dropping anything outside it."""

    if duration <= 0:
        return list(records)

    clamped: list[MomentRecord] = []
    for record in records:
        if record.start >= duration:
            continue
        start = max(0.0, record.start)
        end = min(record.end, duration)
        if end - start < _MIN_CLIP_SECONDS:
            continue
        if start == record.start and end == record.end:
            clamped.append(record)
            continue
        updated = _replace(record, start=start, end=end)
        if updated is not None:
            clamped.append(updated)
    return clamped


def filter_nonoverlapping_outputs(
    outputs: Sequence[MomentRecord],
    inputs: Sequence[MomentRecord],
    *,
    min_overlap_s: float = 1.0,
) -> list[MomentRecord]:
    """Drop stage outputs that match no input candidate in time.

    The cleanup and judge stages are meant to filter, merge and re-rate what
    they are given. A record overlapping nothing they were shown is invented,
    and its timecodes point at footage nobody vetted.
    """

    if not inputs:
        return list(outputs)

    kept: list[MomentRecord] = []
    for record in outputs:
        for candidate in inputs:
            overlap = min(record.end, candidate.end) - max(record.start, candidate.start)
            if overlap >= min_overlap_s:
                kept.append(record)
                break
    return kept


@dataclass(frozen=True)
class QuoteMatch:
    """Result of looking for a candidate's quote in the transcript."""

    ratio: float
    start: float | None = None
    end: float | None = None

    @property
    def found(self) -> bool:
        return self.start is not None and self.end is not None


def verify_quote(
    record: MomentRecord,
    index: TranscriptIndex,
    *,
    tolerance_s: float = 10.0,
) -> QuoteMatch:
    """Find a candidate's quote in the transcript around its own span.

    Returns the best similarity found and, when there is a match, the span the
    quote actually occupies. A quote nobody said is the clearest signal that a
    candidate is hallucinated — and it is the only field tying a candidate to
    real content, since every other heuristic scores the model's own prose.
    """

    quote_tokens = normalized_tokens(record.quote)
    if not quote_tokens:
        return QuoteMatch(ratio=0.0)

    words = index.words_between(
        record.start - tolerance_s,
        record.end + tolerance_s,
    )
    if not words:
        return QuoteMatch(ratio=0.0)

    word_tokens = [normalized_tokens(word.text) for word in words]
    # Words normalize to at most one token each; keep the mapping simple by
    # dropping anything that normalizes away (pure punctuation).
    usable = [(word, tokens[0]) for word, tokens in zip(words, word_tokens) if tokens]
    if not usable:
        return QuoteMatch(ratio=0.0)

    haystack_tokens = [token for _word, token in usable]
    window = len(quote_tokens)
    needle = " ".join(quote_tokens)

    best_ratio = 0.0
    best_span: tuple[float, float] | None = None

    if window >= len(haystack_tokens):
        ratio = difflib.SequenceMatcher(None, needle, " ".join(haystack_tokens)).ratio()
        return QuoteMatch(
            ratio=round(ratio, 4),
            start=usable[0][0].start if ratio >= _REFINE_MIN_RATIO else None,
            end=usable[-1][0].end if ratio >= _REFINE_MIN_RATIO else None,
        )

    for offset in range(len(haystack_tokens) - window + 1):
        candidate = " ".join(haystack_tokens[offset : offset + window])
        ratio = difflib.SequenceMatcher(None, needle, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_span = (usable[offset][0].start, usable[offset + window - 1][0].end)
            if ratio == 1.0:
                break

    if best_span is None or best_ratio < _REFINE_MIN_RATIO:
        return QuoteMatch(ratio=round(best_ratio, 4))
    return QuoteMatch(ratio=round(best_ratio, 4), start=best_span[0], end=best_span[1])


def apply_quote_verification(
    records: Sequence[MomentRecord],
    index: TranscriptIndex,
    *,
    enabled: bool = True,
    min_ratio: float = 0.55,
    refine_boundaries: bool = True,
) -> list[MomentRecord]:
    """Annotate records with their quote match, optionally widening bounds.

    Records below ``min_ratio`` are kept but marked, so scoring can penalize
    them rather than the pipeline silently losing a moment to a bad quote.
    """

    if not enabled or not index:
        return list(records)

    verified: list[MomentRecord] = []
    for record in records:
        match = verify_quote(record, index)
        changes: dict[str, Any] = {"quote_match_ratio": match.ratio}

        if (
            refine_boundaries
            and match.found
            and match.ratio >= _REFINE_MIN_RATIO
            and match.start is not None
            and match.end is not None
        ):
            # Widen only: the quote is the payload of the clip, so it must fit
            # inside it, but the surrounding setup is worth keeping too.
            changes["start"] = min(record.start, match.start)
            changes["end"] = max(record.end, match.end)

        updated = _replace(record, **changes)
        verified.append(updated if updated is not None else record)
    return verified


def snap_record_boundaries(
    record: MomentRecord,
    index: TranscriptIndex,
    *,
    max_shift_s: float = 3.0,
) -> MomentRecord:
    """Anchor a clip's bounds to real sentence/word boundaries."""

    if not index or max_shift_s <= 0:
        return record

    start = index.snap_start(record.start, max_shift=max_shift_s)
    end = index.snap_end(record.end, max_shift=max_shift_s)
    if start == record.start and end == record.end:
        return record
    if end - start < _MIN_CLIP_SECONDS:
        return record
    return _replace(record, start=start, end=end) or record


def snap_records(
    records: Sequence[MomentRecord],
    index: TranscriptIndex,
    *,
    enabled: bool = True,
    max_shift_s: float = 3.0,
) -> list[MomentRecord]:
    """Snap every record's boundaries, if enabled."""

    if not enabled:
        return list(records)
    return [
        snap_record_boundaries(record, index, max_shift_s=max_shift_s)
        for record in records
    ]


def annotate_speech_rate(
    records: Sequence[MomentRecord],
    index: TranscriptIndex,
) -> list[MomentRecord]:
    """Attach words-per-second for each record's span."""

    if not index:
        return list(records)

    annotated: list[MomentRecord] = []
    for record in records:
        rate = index.speech_rate(record.start, record.end)
        if rate is None:
            annotated.append(record)
            continue
        updated = _replace(record, speech_rate_wps=rate)
        annotated.append(updated if updated is not None else record)
    return annotated


def quote_verification_settings(conf: Mapping[str, Any]) -> dict[str, Any]:
    """Read the quote-verification knobs with their defaults."""

    def _bool(key: str, default: bool) -> bool:
        value = conf.get(key, default)
        return bool(value) if isinstance(value, bool) else default

    try:
        min_ratio = float(conf.get("min_ratio", 0.55))
    except (TypeError, ValueError):
        min_ratio = 0.55

    return {
        "enabled": _bool("enabled", True),
        "min_ratio": min_ratio,
        "refine_boundaries": _bool("refine_boundaries", True),
    }
