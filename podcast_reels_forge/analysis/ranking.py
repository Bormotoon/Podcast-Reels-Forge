"""Ranking and de-duplication helpers for analysis candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from podcast_reels_forge.analysis.contracts import MomentRecord, coerce_moment_record
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
            -float(record.score),
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
    )
    data = {
        **record.to_dict(),
        "score": total,
        "judge_score": total,
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
    for record in sorted(deduped, key=lambda r: (-float(r.score), r.start, r.end)):
        bucket = _bucket_name(record.clip_type)
        limit = quotas.get(bucket, 0)
        if limit <= 0:
            continue
        if bucket_counts.get(bucket, 0) >= limit:
            continue
        if any(_overlap_seconds(record, existing) > 0 for existing in selected):
            continue
        selected.append(record)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    return selected


def coerce_ranking_candidates(raw: Sequence[Mapping[str, Any]]) -> list[MomentRecord]:
    records: list[MomentRecord] = []
    for item in raw:
        record = coerce_moment_record(item)
        if record is not None:
            records.append(record)
    return records
