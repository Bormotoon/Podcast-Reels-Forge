"""Metadata finalization helpers for analysis moments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from podcast_reels_forge.analysis.contracts import MomentRecord, coerce_moment_record
from podcast_reels_forge.utils.reel_markdown import build_description_text, build_hashtags


def _ensure_caption(moment: Mapping[str, Any]) -> str:
    caption = str(moment.get("caption", "") or "").strip()
    if caption:
        return caption
    description = build_description_text(moment)
    return description


def finalize_moment_metadata(moment: MomentRecord | Mapping[str, Any]) -> MomentRecord:
    raw = moment.to_dict() if isinstance(moment, MomentRecord) else dict(moment)
    description = build_description_text(raw)
    caption = _ensure_caption(raw)
    hashtags = build_hashtags(raw, description_text=description)
    if len(hashtags) != 5:
        hashtags = build_hashtags(
            {
                **raw,
                "caption": caption or description,
            },
            description_text=description,
        )

    payload = {
        **raw,
        "caption": caption or description,
        "hashtags": hashtags[:5],
        "selection_stage": raw.get("selection_stage") or "metadata",
    }
    if not str(payload.get("title", "")).strip():
        payload["title"] = description[:60]
    if not str(payload.get("hook", "")).strip():
        payload["hook"] = payload["title"]
    if not str(payload.get("quote", "")).strip():
        payload["quote"] = payload["hook"]
    if not str(payload.get("why", "")).strip():
        payload["why"] = description
    record = coerce_moment_record(payload)
    if record is None:
        raise ValueError("Failed to finalize moment metadata")
    return record


def finalize_moment_list(moments: Sequence[MomentRecord | Mapping[str, Any]]) -> list[MomentRecord]:
    final: list[MomentRecord] = []
    for moment in moments:
        final.append(finalize_moment_metadata(moment))
    return final
