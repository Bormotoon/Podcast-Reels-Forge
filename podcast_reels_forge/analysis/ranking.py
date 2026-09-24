"""Ranking and de-duplication helpers for analysis candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from podcast_reels_forge.analysis.contracts import (
    MomentRecord,
    coerce_moment_record,
    replace_record,
)
from podcast_reels_forge.analysis.transcript_index import normalized_tokens
from podcast_reels_forge.analysis.scoring import (
    clip_type_target_bounds,
    combined_priority_score,
    scoring_breakdown,
)
from podcast_reels_forge.analysis.validation import quote_similarity

# RU: Два клипа могут пересекаться не больше чем на эту долю более короткого.
# Раньше запрещалось любое пересечение, и после snap/padding соседние хорошие
# моменты выбивали друг друга.
# EN: Two selected clips may overlap by at most this share of the shorter one.
# Any overlap used to be forbidden, so after snapping and padding two good
# neighbouring moments knocked each other out.
DEFAULT_MAX_OVERLAP_RATIO = 0.2

# Weight of topic redundancy in MMR selection (utility = quality - lambda *
# similarity to what is already selected).
DEFAULT_MMR_LAMBDA = 0.7

# Quotes at least this similar describe the same words.
_SAME_QUOTE_SIMILARITY = 0.8
_MIN_QUOTE_TOKENS_FOR_IDENTITY = 4

BUCKETS = ("story", "highlight", "reel", "long_reel")


def _overlap_seconds(a: MomentRecord, b: MomentRecord) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _jaccard_like_overlap(a: MomentRecord, b: MomentRecord) -> float:
    overlap = _overlap_seconds(a, b)
    if overlap <= 0:
        return 0.0
    span = max(a.end - a.start, b.end - b.start, 0.01)
    return overlap / span


def overlap_ratio_of_shorter(a: MomentRecord, b: MomentRecord) -> float:
    """Overlap as a share of the shorter clip (0 when disjoint)."""

    overlap = _overlap_seconds(a, b)
    if overlap <= 0:
        return 0.0
    shorter = max(min(a.end - a.start, b.end - b.start), 0.01)
    return overlap / shorter


def ranking_value(moment: MomentRecord) -> float:
    """RU: Значение, по которому кандидаты сравниваются между собой.

    EN: The value candidates are ordered by. Prefers the combined `priority`
    once ranking has computed it, and falls back to the model's raw `score`
    for records that have not been through scoring yet.
    """

    if moment.priority is not None:
        return float(moment.priority)
    return float(moment.score)


def topic_tokens(moment: MomentRecord) -> frozenset[str]:
    """Content words identifying what a moment is about."""

    text = f"{moment.title} {moment.quote}"
    return frozenset(
        token for token in normalized_tokens(text) if len(token) >= 3
    )


def topic_similarity(first: MomentRecord, second: MomentRecord) -> float:
    """Jaccard overlap of two moments' topic vocabulary."""

    left = topic_tokens(first)
    right = topic_tokens(second)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _dedupe_key(moment: MomentRecord) -> tuple[int, int, str, str]:
    return (
        round(moment.start * 10),
        round(moment.end * 10),
        moment.clip_type.lower(),
        moment.title.lower().strip(),
    )


def same_quote(first: MomentRecord, second: MomentRecord) -> bool:
    """Whether two records carry the same (non-trivial) quote."""

    if min(
        len(normalized_tokens(first.quote)),
        len(normalized_tokens(second.quote)),
    ) < _MIN_QUOTE_TOKENS_FOR_IDENTITY:
        return False
    return quote_similarity(first, second) >= _SAME_QUOTE_SIMILARITY


def is_duplicate(
    record: MomentRecord,
    existing: MomentRecord,
    *,
    overlap_threshold: float = 0.35,
) -> bool:
    """Whether ``record`` repeats ``existing``.

    Either they cover mostly the same time, or they quote the same words in
    overlapping spans — the latter catches the same moment re-titled with
    slightly different bounds, which time overlap alone let through.
    """

    if _jaccard_like_overlap(record, existing) >= overlap_threshold:
        return True
    return _overlap_seconds(record, existing) > 0 and same_quote(record, existing)


def dedupe_moments(records: Sequence[MomentRecord], *, overlap_threshold: float = 0.35) -> list[MomentRecord]:
    """Remove near-duplicate or heavily overlapping candidates.

    Of two duplicates the one ranked higher survives; the ranking already
    rewards a better quote match and a more complete clip.
    """

    ordered = sorted(
        records,
        key=lambda record: (
            -ranking_value(record),
            -(record.quote_match_ratio or 0.0),
            record.start,
            record.end,
            record.title,
        ),
    )
    selected: list[MomentRecord] = []
    seen_keys: set[tuple[int, int, str, str]] = set()
    for record in ordered:
        key = _dedupe_key(record)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if any(
            is_duplicate(record, existing, overlap_threshold=overlap_threshold)
            for existing in selected
        ):
            continue
        selected.append(record)
    return selected


def bucket_name(clip_type: str) -> str:
    """Quota bucket a clip type belongs to."""

    ct = clip_type.lower()
    if "story" in ct:
        return "story"
    if "highlight" in ct or "hot" in ct:
        return "highlight"
    if "long" in ct:
        return "long_reel"
    return "reel"


def assign_clip_types(
    records: Sequence[MomentRecord],
    quotas: Mapping[str, int],
    *,
    tolerance: float = 0.25,
) -> list[MomentRecord]:
    """Give every record a clip type its final duration actually fits.

    A scout sees one chunk and cannot reliably judge how long a story runs,
    and boundaries move afterwards anyway. The model's type is kept when it
    is enabled and the duration fits it (within ``tolerance``); otherwise the
    enabled bucket whose target range fits best is assigned, and the change
    is recorded in ``derived_fields``.
    """

    enabled = [bucket for bucket in BUCKETS if int(quotas.get(bucket, 0) or 0) > 0]
    if not enabled:
        return list(records)

    def _distance(duration: float, bucket: str, slack: float) -> float:
        low, high = clip_type_target_bounds(bucket)
        low *= 1.0 - slack
        high *= 1.0 + slack
        if duration < low:
            return low - duration
        if duration > high:
            return duration - high
        center = (low + high) / 2.0
        # Inside the range: prefer the bucket whose centre is closest, scaled
        # well below any out-of-range distance.
        return -1.0 + abs(duration - center) / max(high, 1.0)

    assigned: list[MomentRecord] = []
    for record in records:
        duration = record.end - record.start
        current = bucket_name(record.clip_type)
        if current in enabled and _distance(duration, current, tolerance) < 0:
            assigned.append(record)
            continue
        best = min(enabled, key=lambda bucket: (_distance(duration, bucket, 0.0), BUCKETS.index(bucket)))
        if best == current:
            assigned.append(record)
            continue
        derived = list(record.derived_fields)
        if "clip_type" not in derived:
            derived.append("clip_type")
        assigned.append(replace_record(record, clip_type=best, derived_fields=derived))
    return assigned


def _with_scoring_fields(
    record: MomentRecord,
    *,
    target_min: float,
    target_max: float,
    stage: str,
    weights: Mapping[str, Any] | None = None,
) -> MomentRecord:
    breakdown = scoring_breakdown(
        record.to_dict(),
        target_min=target_min,
        target_max=target_max,
    )
    total = combined_priority_score(
        record.to_dict(),
        target_min=target_min,
        target_max=target_max,
        weights=weights,
    )
    # `score` deliberately keeps whatever the model rated this moment on its
    # 1-10 scale — the cut stage filters on it. Overwriting it here used to
    # both break that filter and feed the combined total back into itself as
    # `base_score` on the next ranking pass.
    data = {
        **record.to_dict(),
        "priority": total,
        "hook_score": breakdown["hook_score"],
        "completeness_score": breakdown["completeness_score"],
        "speaker_focus": breakdown["speaker_focus_score"],
        "subtitle_readability_score": breakdown["readability_score"],
        "duration_fit_score": breakdown["duration_score"],
        "selection_stage": stage,
    }
    coerced = coerce_moment_record(data)
    return coerced or record


def rank_moments(
    records: Sequence[MomentRecord],
    *,
    clip_type_quotas: Mapping[str, int],
    scoring_weights: Mapping[str, Any] | None = None,
    diversity_enabled: bool = True,
    max_topic_similarity: float = 0.5,
    fill_to_total: int | None = None,
    max_overlap_ratio: float = DEFAULT_MAX_OVERLAP_RATIO,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
) -> list[MomentRecord]:
    """Apply scoring, dedupe and quota-aware MMR selection.

    Selection is iterative maximal-marginal-relevance: each step takes the
    candidate with the best ``quality - mmr_lambda * similarity`` to what is
    already selected, among those that fit their quota and the overlap
    policy. A candidate at or above ``max_topic_similarity`` to a selected
    clip is additionally pushed behind every distinct candidate, so a
    repeated topic only fills a slot nothing else can.

    ``fill_to_total`` makes the per-type quotas soft: when set (the
    duration-scaled path), slots a bucket could not fill spill over to the
    best remaining candidates of any type, so a mix mismatch between the
    quotas and what the episode actually contains costs composition, not
    clip count.
    """

    if not records:
        return []

    scored: list[MomentRecord] = []
    for record in records:
        target_min, target_max = clip_type_target_bounds(record.clip_type)
        scored.append(
            _with_scoring_fields(
                record,
                target_min=target_min,
                target_max=target_max,
                stage=record.selection_stage or "judge",
                weights=scoring_weights,
            ),
        )

    deduped = dedupe_moments(scored)
    quotas = {key.lower(): max(0, int(value)) for key, value in clip_type_quotas.items()}
    # A quota of 0 — or a bucket missing from the mapping — excludes that clip
    # type entirely. Callers that configure no quotas at all still expect
    # results, so treat an empty/all-zero mapping as "reels, unlimited".
    if not any(quotas.values()):
        quotas = {"reel": len(deduped)}

    ordered = sorted(deduped, key=lambda r: (-ranking_value(r), r.start, r.end))
    values = [ranking_value(record) for record in ordered]
    low, high = min(values), max(values)
    spread = high - low

    def _quality(record: MomentRecord) -> float:
        return (ranking_value(record) - low) / spread if spread > 0 else 1.0

    selected: list[MomentRecord] = []
    bucket_counts: dict[str, int] = {}

    def _no_conflict(record: MomentRecord) -> bool:
        return all(
            overlap_ratio_of_shorter(record, existing) <= max(0.0, max_overlap_ratio)
            and not is_duplicate(record, existing)
            for existing in selected
        )

    def _fits_quota(record: MomentRecord) -> bool:
        bucket = bucket_name(record.clip_type)
        limit = quotas.get(bucket, 0)
        return limit > 0 and bucket_counts.get(bucket, 0) < limit

    def _utility(record: MomentRecord) -> float:
        if not diversity_enabled or not selected:
            return _quality(record)
        redundancy = max(topic_similarity(record, existing) for existing in selected)
        # Non-overlapping in time, but about the same thing: a reel set that
        # says one thing four ways is worse than a varied one.
        repeat_penalty = 1.0 if redundancy >= max_topic_similarity else 0.0
        return _quality(record) - mmr_lambda * redundancy - repeat_penalty

    def _take(record: MomentRecord) -> None:
        bucket = bucket_name(record.clip_type)
        selected.append(record)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    remaining = list(ordered)
    while remaining:
        eligible = [r for r in remaining if _fits_quota(r) and _no_conflict(r)]
        if not eligible:
            break
        # max() keeps the first of equal utilities, i.e. the higher-ranked.
        best = max(eligible, key=_utility)
        _take(best)
        remaining.remove(best)

    if fill_to_total is not None and len(selected) < fill_to_total:
        # Quota spillover: the type mix ran out of matching candidates before
        # the episode ran out of good moments. Fill the remaining slots by
        # priority, still under the overlap policy.
        for record in ordered:
            if len(selected) >= fill_to_total:
                break
            if record in selected or not _no_conflict(record):
                continue
            _take(record)

    return selected


def coerce_ranking_candidates(raw: Sequence[Mapping[str, Any]]) -> list[MomentRecord]:
    records: list[MomentRecord] = []
    for item in raw:
        record = coerce_moment_record(item)
        if record is not None:
            records.append(record)
    return records
