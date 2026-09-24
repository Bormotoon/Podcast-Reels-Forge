"""RU: Проверка кандидатов против транскрипта.

Модель может вернуть таймкод за пределами куска, выдумать момент, которого не
было во входе, или пересказать «цитату» своими словами. Здесь всё это
ловится: границы поджимаются к реальным, выдуманные записи отбрасываются, а
цитата — единственное доказательство того, что момент вообще был сказан, —
ищется в транскрипте сначала дословно, и лишь затем ограниченным нечётким
сравнением. Кандидат с неподтверждённой цитатой в финал не попадает.

EN: Validate candidates against the transcript.

A model can return a timecode outside its chunk, invent a moment that was
never in its input, or paraphrase a "quote" it claims is verbatim. This
module catches all three: bounds are clamped to real ones, invented records
are dropped, and the quote — the only evidence a moment was actually said —
is looked up verbatim first and only then with a bounded fuzzy alignment. A
candidate whose quote is not confirmed never reaches the final cut.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from podcast_reels_forge.analysis.contracts import (
    MomentRecord,
    coerce_moment_record,
    replace_record,
)
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

# RU: Пороги по умолчанию. Ниже min_ratio кандидат исключается из пула ещё до
# judge; ниже min_final_ratio — не допускается в финальный отбор.
# EN: Default thresholds. Below min_ratio a candidate leaves the pool before
# the judge sees it; below min_final_ratio it may not enter the final cut.
DEFAULT_MIN_RATIO = 0.55
DEFAULT_MIN_FINAL_RATIO = 0.75

METHOD_EXACT = "exact"
METHOD_FUZZY = "fuzzy"
METHOD_NONE = "none"


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


def quote_similarity(first: MomentRecord, second: MomentRecord) -> float:
    """Token overlap of two records' quotes (0..1, over the shorter quote)."""

    left = set(normalized_tokens(first.quote))
    right = set(normalized_tokens(second.quote))
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def filter_nonoverlapping_outputs(
    outputs: Sequence[MomentRecord],
    inputs: Sequence[MomentRecord],
    *,
    min_overlap_s: float = 1.0,
    min_quote_similarity: float = 0.5,
) -> list[MomentRecord]:
    """Drop stage outputs that cannot be traced back to an input candidate.

    The cleanup and judge stages are meant to filter, merge and re-rate what
    they are given. Lineage is checked first: an output carrying a known
    ``candidate_id`` is accepted. Otherwise it must overlap an input in time
    *and*, when both sides carry quotes, talk about the same words — a model
    can pick a different moment inside a long input interval, and that
    footage was never vetted.
    """

    if not inputs:
        return list(outputs)

    known_ids = {
        identity
        for record in inputs
        for identity in (record.candidate_id, *record.merged_ids)
        if identity
    }
    kept: list[MomentRecord] = []
    for record in outputs:
        if record.candidate_id and record.candidate_id in known_ids:
            kept.append(record)
            continue
        for candidate in inputs:
            overlap = min(record.end, candidate.end) - max(record.start, candidate.start)
            if overlap < min_overlap_s:
                continue
            if (
                record.quote
                and candidate.quote
                and quote_similarity(record, candidate) < min_quote_similarity
            ):
                continue
            kept.append(record)
            break
    return kept


@dataclass(frozen=True)
class QuoteMatch:
    """Result of looking for a candidate's quote in the transcript."""

    ratio: float
    start: float | None = None
    end: float | None = None
    method: str = METHOD_NONE
    matched_tokens: int = 0
    missing_tokens: int = 0
    extra_tokens: int = 0

    @property
    def found(self) -> bool:
        return self.start is not None and self.end is not None


def _fold(token: str) -> str:
    """Crude stem used only by the fuzzy fallback.

    Absorbs inflection ("школы"/"школа") and small recognition slips at word
    endings, which is what the fallback exists for; the exact pass never uses
    it.
    """

    return token[: max(4, min(6, len(token) - 2))]


def _exact_occurrences(needle: Sequence[str], haystack: Sequence[str]) -> list[int]:
    width = len(needle)
    if width == 0 or width > len(haystack):
        return []
    first = needle[0]
    return [
        offset
        for offset in range(len(haystack) - width + 1)
        if haystack[offset] == first and list(haystack[offset : offset + width]) == list(needle)
    ]


def verify_quote(
    record: MomentRecord,
    index: TranscriptIndex,
    *,
    tolerance_s: float = 10.0,
) -> QuoteMatch:
    """Find a candidate's quote in the transcript around its own span.

    Two passes, strictest first:

    1. **exact** — the normalized quote tokens occur contiguously in the
       transcript. When they occur more than once, the occurrence closest to
       the candidate's own interval wins.
    2. **fuzzy** — a bounded token alignment over stemmed tokens, tried only
       from positions where a quote token actually occurs. The ratio is a
       Dice score over the matched span, so missing *and* extra tokens both
       cost; unlike a character-level similarity it cannot be satisfied by a
       window of frequent short words.
    """

    quote_tokens = normalized_tokens(record.quote)
    if not quote_tokens:
        return QuoteMatch(ratio=0.0)

    timed = index.timed_tokens(record.start - tolerance_s, record.end + tolerance_s)
    if not timed:
        return QuoteMatch(ratio=0.0, missing_tokens=len(quote_tokens))

    haystack = [token for token, _start, _end in timed]
    width = len(quote_tokens)

    def _distance(first: int, last: int) -> float:
        span_start, span_end = timed[first][1], timed[last][2]
        overlap = min(span_end, record.end) - max(span_start, record.start)
        return -overlap if overlap > 0 else abs(span_start - record.start)

    occurrences = _exact_occurrences(quote_tokens, haystack)
    if occurrences:
        best = min(occurrences, key=lambda offset: _distance(offset, offset + width - 1))
        return QuoteMatch(
            ratio=1.0,
            start=timed[best][1],
            end=timed[best + width - 1][2],
            method=METHOD_EXACT,
            matched_tokens=width,
        )

    folded_quote = [_fold(token) for token in quote_tokens]
    folded_haystack = [_fold(token) for token in haystack]
    vocabulary = set(folded_quote)
    slack = max(2, width // 4)

    best_key: tuple[float, float] | None = None
    best_match = QuoteMatch(ratio=0.0, missing_tokens=width)
    for offset, token in enumerate(folded_haystack):
        if token not in vocabulary:
            continue
        window = folded_haystack[offset : offset + width + slack]
        blocks = [
            block
            for block in difflib.SequenceMatcher(
                None, folded_quote, window, autojunk=False,
            ).get_matching_blocks()
            if block.size
        ]
        if not blocks:
            continue
        matched = sum(block.size for block in blocks)
        first = offset + blocks[0].b
        last = offset + blocks[-1].b + blocks[-1].size - 1
        span_len = last - first + 1
        ratio = 2.0 * matched / (width + span_len)
        key = (ratio, -_distance(first, last))
        if best_key is None or key > best_key:
            best_key = key
            best_match = QuoteMatch(
                ratio=round(ratio, 4),
                start=timed[first][1],
                end=timed[last][2],
                method=METHOD_FUZZY,
                matched_tokens=matched,
                missing_tokens=width - matched,
                extra_tokens=span_len - matched,
            )

    if best_match.ratio < _REFINE_MIN_RATIO:
        # Too weak to say where the quote is; report the ratio only.
        return QuoteMatch(
            ratio=best_match.ratio,
            method=METHOD_NONE,
            matched_tokens=best_match.matched_tokens,
            missing_tokens=best_match.missing_tokens,
            extra_tokens=best_match.extra_tokens,
        )
    return best_match


def apply_quote_verification(
    records: Sequence[MomentRecord],
    index: TranscriptIndex,
    *,
    enabled: bool = True,
    refine_boundaries: bool = True,
    **_thresholds: Any,
) -> list[MomentRecord]:
    """Annotate records with their quote match, optionally widening bounds.

    Annotation only — rejection is :func:`split_by_quote_ratio`'s job, so the
    caller decides at which point of the pipeline a threshold applies. Extra
    keyword arguments (the thresholds from :func:`quote_verification_settings`)
    are accepted and ignored so the settings dict can be splatted in.
    """

    if not enabled or not index:
        return list(records)

    verified: list[MomentRecord] = []
    for record in records:
        match = verify_quote(record, index)
        changes: dict[str, Any] = {
            "quote_match_ratio": match.ratio,
            "quote_match_method": match.method,
            "quote_start": match.start,
            "quote_end": match.end,
        }
        if refine_boundaries and match.found and match.start is not None and match.end is not None:
            # Widen only: the quote is the payload of the clip, so it must fit
            # inside it, but the surrounding setup is worth keeping too.
            changes["start"] = min(record.start, match.start)
            changes["end"] = max(record.end, match.end)
        verified.append(replace_record(record, **changes))
    return verified


def split_by_quote_ratio(
    records: Sequence[MomentRecord],
    min_ratio: float,
) -> tuple[list[MomentRecord], list[MomentRecord]]:
    """Split records into (kept, rejected) by quote match ratio.

    Records that were never measured (``quote_match_ratio is None`` — no
    transcript index to check against) are kept: the absence of a check is
    not evidence of invention. A record that *was* measured and fell short is
    rejected outright; a soft penalty used to let a fully invented quote
    through on the strength of its title and score.
    """

    kept: list[MomentRecord] = []
    rejected: list[MomentRecord] = []
    for record in records:
        ratio = record.quote_match_ratio
        if ratio is not None and ratio < min_ratio:
            rejected.append(record)
        else:
            kept.append(record)
    return kept, rejected


def enforce_quote_containment(
    records: Sequence[MomentRecord],
    *,
    duration: float = 0.0,
) -> tuple[list[MomentRecord], list[MomentRecord]]:
    """Make sure every located quote sits inside its final clip.

    Runs after every transformation (refine, snap, clamp) and right before
    ``moments.json`` is written. A clip that lost part of its quote is widened
    back over it when that stays inside the episode; otherwise it is
    rejected, since the rendered clip would not contain the words it is
    selected for.
    """

    kept: list[MomentRecord] = []
    rejected: list[MomentRecord] = []
    for record in records:
        q_start, q_end = record.quote_start, record.quote_end
        if q_start is None or q_end is None:
            kept.append(record)
            continue
        if record.start <= q_start and q_end <= record.end:
            kept.append(record)
            continue
        start = min(record.start, q_start)
        end = max(record.end, q_end)
        if start < 0 or (duration > 0 and end > duration):
            rejected.append(record)
            continue
        kept.append(replace_record(record, start=start, end=end))
    return kept, rejected


def snap_record_boundaries(
    record: MomentRecord,
    index: TranscriptIndex,
    *,
    max_shift_s: float = 3.0,
) -> MomentRecord:
    """Anchor a clip's bounds to real sentence/word boundaries.

    A located quote bounds how far a boundary may move inward, so snapping
    never trims the evidence out of the clip.
    """

    if not index or max_shift_s <= 0:
        return record

    start = index.snap_start(record.start, max_shift=max_shift_s, limit=record.quote_start)
    end = index.snap_end(record.end, max_shift=max_shift_s, limit=record.quote_end)
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
        annotated.append(replace_record(record, speech_rate_wps=rate))
    return annotated


def quote_verification_settings(conf: Mapping[str, Any]) -> dict[str, Any]:
    """Read the quote-verification knobs with their defaults."""

    def _bool(key: str, default: bool) -> bool:
        value = conf.get(key, default)
        return bool(value) if isinstance(value, bool) else default

    def _ratio(key: str, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(conf.get(key, default))))
        except (TypeError, ValueError):
            return default

    min_ratio = _ratio("min_ratio", DEFAULT_MIN_RATIO)
    return {
        "enabled": _bool("enabled", True),
        "min_ratio": min_ratio,
        # The final gate can only be stricter than the pool gate.
        "min_final_ratio": max(min_ratio, _ratio("min_final_ratio", DEFAULT_MIN_FINAL_RATIO)),
        "refine_boundaries": _bool("refine_boundaries", True),
    }
