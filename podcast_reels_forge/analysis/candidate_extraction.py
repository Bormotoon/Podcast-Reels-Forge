"""Candidate extraction and normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from podcast_reels_forge.analysis.contracts import MomentRecord, coerce_moment_record


def extract_candidate_payload(value: Any) -> list[Mapping[str, Any]]:
    """Return a list of raw candidate mappings from an LLM response."""

    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        single = value.get("moment")
        if isinstance(single, Mapping):
            return [single]
        for key in ("moments", "candidates", "results", "items", "clips"):
            raw = value.get(key)
            if isinstance(raw, list):
                return [item for item in raw if isinstance(item, Mapping)]
    return []


def normalize_candidate_list(value: Any, *, stage: str) -> list[MomentRecord]:
    """Convert raw LLM output into validated moment records."""

    records: list[MomentRecord] = []
    for raw in extract_candidate_payload(value):
        record = coerce_moment_record(raw)
        if record is None:
            continue
        payload = {
            **record.to_dict(),
            "selection_stage": stage,
        }
        coerced = coerce_moment_record(payload)
        if coerced is not None:
            records.append(coerced)
    return records


def build_candidate_json(records: Sequence[MomentRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]
