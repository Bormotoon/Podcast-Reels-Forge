"""RU: Применение решений cleanup/judge к исходным кандидатам.

Cleanup и judge больше не пересылают записи целиком — они отвечают решениями
по ``candidate_id``: оставить, отбросить, слить, переоценить. Python
применяет их к исходным объектам, поэтому модель физически не может сдвинуть
таймкоды или переписать цитату: ``start``, ``end`` и ``quote`` всегда берутся
из исходного кандидата. Это ещё и резко сокращает выходные токены.

Ответ в старом формате (полные записи без ``candidate_id``) тоже принимается:
каждая запись привязывается к входному кандидату по времени и цитате, а
доказательные поля всё равно восстанавливаются из источника.

EN: Apply cleanup/judge decisions to the original candidates.

Cleanup and judge no longer echo records back — they answer with decisions
keyed by ``candidate_id``: keep, drop, merge, re-rate. Python applies them to
the original objects, so the model cannot move timecodes or rewrite a quote:
``start``, ``end`` and ``quote`` always come from the source candidate. It
also cuts output tokens sharply.

A legacy answer (full records without ``candidate_id``) is still accepted:
each record is traced to an input candidate by time and quote, and the
evidence fields are restored from that source regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from podcast_reels_forge.analysis.candidate_extraction import extract_candidate_payload
from podcast_reels_forge.analysis.contracts import (
    MomentRecord,
    coerce_moment_record,
    replace_record,
)
from podcast_reels_forge.analysis.validation import filter_nonoverlapping_outputs

# Fields each stage may change. Everything else — above all start, end and
# quote — is owned by Python.
CLEANUP_EDITABLE: tuple[str, ...] = ()
JUDGE_EDITABLE: tuple[str, ...] = ("title", "hook", "why", "clip_type", "caption", "hashtags")


@dataclass
class DecisionOutcome:
    """What applying one stage response did to its input."""

    records: list[MomentRecord]
    mode: str = "empty"  # "ids" | "legacy" | "empty"
    kept: int = 0
    dropped: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    unmentioned: int = 0
    untraceable: int = 0


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"false", "no", "0", "drop", "reject"}:
            return False
        if lowered in {"true", "yes", "1", "keep"}:
            return True
    return default


def _item_id(item: Mapping[str, Any]) -> str:
    return str(item.get("candidate_id") or item.get("id") or "").strip()


def _item_score(item: Mapping[str, Any]) -> float | None:
    for key in ("score", "quality_score", "judge_score"):
        value = item.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _trace_legacy(
    item: Mapping[str, Any],
    inputs: Sequence[MomentRecord],
) -> MomentRecord | None:
    """Find the input a legacy full-record answer refers to."""

    record = coerce_moment_record(item)
    if record is None:
        return None
    if not filter_nonoverlapping_outputs([record], inputs):
        return None
    return max(
        inputs,
        key=lambda source: min(record.end, source.end) - max(record.start, source.start),
    )


def apply_stage_decisions(
    inputs: Sequence[MomentRecord],
    response: Any,
    *,
    stage: str,
    editable: Sequence[str],
    record_judge_score: bool = False,
) -> DecisionOutcome:
    """Apply one cleanup/judge response to the candidates it was shown.

    * An empty or unusable response keeps the input unchanged: a failed batch
      must not cost that stretch of the episode.
    * In id mode, candidates the response does not mention are kept as they
      were — truncation at ``n_predict`` cuts the tail of the list, and that
      is not a verdict on those candidates.
    * In legacy mode (records without ids) the old semantics hold: what the
      model did not return is dropped.
    """

    items = extract_candidate_payload(response)
    if not items or not inputs:
        return DecisionOutcome(records=list(inputs), kept=len(inputs))

    by_id = {record.candidate_id: record for record in inputs if record.candidate_id}
    # Id mode as soon as the answer speaks in ids at all — even unknown ones:
    # hallucinated ids must not flip it into legacy "unmentioned = dropped".
    id_mode = any(_item_id(item) for item in items)

    decided: dict[int, tuple[MomentRecord, Mapping[str, Any]]] = {}
    outcome = DecisionOutcome(records=[], mode="ids" if id_mode else "legacy")
    position = {id(record): offset for offset, record in enumerate(inputs)}

    for item in items:
        source = by_id.get(_item_id(item))
        if source is None:
            source = _trace_legacy(item, inputs) if not id_mode or not _item_id(item) else None
        if source is None:
            outcome.untraceable += 1
            continue
        offset = position[id(source)]
        # First verdict on a candidate wins; repeats are model noise.
        decided.setdefault(offset, (source, item))

    if not decided:
        # Nothing in the answer refers to a real candidate: treat it like an
        # unusable answer and keep the batch.
        return DecisionOutcome(
            records=list(inputs),
            kept=len(inputs),
            untraceable=outcome.untraceable,
        )

    merged_into: dict[int, list[MomentRecord]] = {}
    absorbed: set[int] = set()
    for offset, (source, item) in sorted(decided.items()):
        if not _as_bool(item.get("keep"), True):
            continue
        raw_merge = item.get("merge_ids")
        if not isinstance(raw_merge, list):
            continue
        for merge_id in raw_merge:
            other = by_id.get(str(merge_id).strip())
            if other is None or other is source:
                continue
            other_offset = position[id(other)]
            if other_offset in absorbed:
                continue
            absorbed.add(other_offset)
            merged_into.setdefault(offset, []).append(other)

    for offset, source in enumerate(inputs):
        if offset in absorbed:
            outcome.merged.append(source.candidate_id)
            continue
        entry = decided.get(offset)
        if entry is None:
            if id_mode:
                outcome.unmentioned += 1
                outcome.records.append(source)
            else:
                outcome.dropped.append(source.candidate_id)
            continue
        _source, item = entry
        if not _as_bool(item.get("keep"), True):
            outcome.dropped.append(source.candidate_id)
            continue

        changes: dict[str, Any] = {"selection_stage": stage}
        for key in editable:
            value = item.get(key)
            if value not in (None, "", []):
                changes[key] = value
        reason_codes = item.get("reason_codes")
        if isinstance(reason_codes, list) and reason_codes:
            changes["reason_codes"] = list(
                dict.fromkeys([*source.reason_codes, *map(str, reason_codes)]),
            )
        score = _item_score(item)
        if score is not None:
            changes["score"] = score
            if record_judge_score:
                changes["judge_score"] = score
        others = merged_into.get(offset, [])
        if others:
            # Merging may widen the interval over overlapping sources only; a
            # disjoint "merge" is really a dedupe and must not glue unrelated
            # footage together. The quote stays the primary's.
            start, end = source.start, source.end
            for other in others:
                if min(end, other.end) - max(start, other.start) > 0:
                    start, end = min(start, other.start), max(end, other.end)
            changes["start"], changes["end"] = start, end
            changes["merged_ids"] = list(
                dict.fromkeys(
                    [*source.merged_ids, *(o.candidate_id for o in others), *(m for o in others for m in o.merged_ids)],
                ),
            )
        outcome.records.append(replace_record(source, **changes))

    outcome.kept = len(outcome.records)
    return outcome
