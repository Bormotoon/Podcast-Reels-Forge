"""Ranking and de-duplication helpers for analysis candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from podcast_reels_forge.analysis.contracts import MomentRecord, coerce_moment_record
from podcast_reels_forge.analysis.transcript_index import normalized_tokens
from podcast_reels_forge.analysis.scoring import (
    clip_type_target_bounds,
    combined_priority_score,
    scoring_breakdown,
)


def _overlap_seconds(a: MomentRecord, b: MomentRecord) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _jaccard_like_overlap(a: MomentRecord, b: MomentRecord) -> float:
    overlap = _overlap_seconds(a, b)
    if overlap <= 0:
        return 0.0
    span = max(a.end - a.start, b.end - b.start, 0.01)
    return overlap / span


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


def dedupe_moments(records: Sequence[MomentRecord], *, overlap_threshold: float = 0.35) -> list[MomentRecord]:
    """Remove near-duplicate or heavily overlapping candidates."""

    ordered = sorted(
        records,
        key=lambda record: (
            -ranking_value(record),
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

        duplicate = False
        for existing in selected:
            if _jaccard_like_overlap(record, existing) >= overlap_threshold:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(record)
    return selected


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
        "crop_confidence": breakdown["duration_score"],
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
) -> list[MomentRecord]:
    """Apply scoring, dedupe and quota-aware selection."""

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

    def _bucket_name(clip_type: str) -> str:
        ct = clip_type.lower()
        if "story" in ct:
            return "story"
        if "highlight" in ct or "hot" in ct:
            return "highlight"
        if "long" in ct:
            return "long_reel"
        return "reel"

    selected: list[MomentRecord] = []
    bucket_counts: dict[str, int] = {}
    # Clips held back only because they repeat an already-selected topic. They
    # are still valid picks, so they backfill any quota the first pass could
    # not fill rather than being lost.
    deferred: list[MomentRecord] = []

    def _fits(record: MomentRecord) -> bool:
        bucket = _bucket_name(record.clip_type)
        limit = quotas.get(bucket, 0)
        if limit <= 0:
            return False
        if bucket_counts.get(bucket, 0) >= limit:
            return False
        return all(_overlap_seconds(record, existing) <= 0 for existing in selected)

    def _take(record: MomentRecord) -> None:
        bucket = _bucket_name(record.clip_type)
        selected.append(record)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    ordered = sorted(deduped, key=lambda r: (-ranking_value(r), r.start, r.end))
    for record in ordered:
        if not _fits(record):
            continue
        if diversity_enabled and any(
            topic_similarity(record, existing) >= max_topic_similarity
            for existing in selected
        ):
            # Non-overlapping in time, but about the same thing. A reel set
            # that says one thing four ways is worse than a varied one.
            deferred.append(record)
            continue
        _take(record)

    for record in deferred:
        if _fits(record):
            _take(record)

    return selected


def coerce_ranking_candidates(raw: Sequence[Mapping[str, Any]]) -> list[MomentRecord]:
    records: list[MomentRecord] = []
    for item in raw:
        record = coerce_moment_record(item)
        if record is not None:
            records.append(record)
    return records
