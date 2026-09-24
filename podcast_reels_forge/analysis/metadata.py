"""Metadata finalization helpers for analysis moments.

Only presentation fields (title, hook, caption, hashtags, why) may be filled
in here, and every one that is filled in is listed in ``derived_fields``.
Evidence fields — ``start``, ``end`` and above all ``quote`` — are never
synthesized: a quote made up from the hook used to turn a missing piece of
evidence into plausible-looking text.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from podcast_reels_forge.analysis.contracts import (
    MomentRecord,
    coerce_moment_record,
    has_quote_evidence,
)
from podcast_reels_forge.utils.reel_markdown import build_description_text, build_hashtags

LOGGER = logging.getLogger("forge")


class MissingEvidenceError(ValueError):
    """A moment reached metadata finalization without a transcript quote."""


def finalize_moment_metadata(
    moment: MomentRecord | Mapping[str, Any],
    *,
    require_quote: bool = True,
) -> MomentRecord:
    raw = moment.to_dict() if isinstance(moment, MomentRecord) else dict(moment)
    if require_quote and not has_quote_evidence(raw):
        raise MissingEvidenceError(
            f"moment [{raw.get('start')}, {raw.get('end')}] has no transcript quote",
        )

    derived = list(raw.get("derived_fields") or [])

    def _derive(key: str, value: Any) -> None:
        raw[key] = value
        if key not in derived:
            derived.append(key)

    description = build_description_text(raw)
    if not str(raw.get("title", "") or "").strip():
        _derive("title", description[:60])
    if not str(raw.get("hook", "") or "").strip():
        _derive("hook", raw["title"])
    if not str(raw.get("why", "") or "").strip():
        _derive("why", description)
    if not str(raw.get("caption", "") or "").strip():
        _derive("caption", description)

    hashtags = build_hashtags(raw, description_text=description)
    if list(hashtags[:5]) != list(raw.get("hashtags") or []):
        _derive("hashtags", hashtags[:5])

    payload = {
        **raw,
        "derived_fields": derived,
        "selection_stage": raw.get("selection_stage") or "metadata",
    }
    record = coerce_moment_record(payload)
    if record is None:
        raise ValueError("Failed to finalize moment metadata")
    return record


def finalize_moment_list(
    moments: Sequence[MomentRecord | Mapping[str, Any]],
    *,
    require_quote: bool = True,
) -> list[MomentRecord]:
    """Finalize every moment, rejecting the ones without evidence."""

    final: list[MomentRecord] = []
    for moment in moments:
        try:
            final.append(finalize_moment_metadata(moment, require_quote=require_quote))
        except MissingEvidenceError as exc:
            LOGGER.warning("metadata: rejected a moment without evidence: %s", exc)
    return final
