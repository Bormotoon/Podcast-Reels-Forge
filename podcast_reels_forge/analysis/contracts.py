"""Typed analysis contracts used by the staged pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AnalysisChunkUnit:
    """Small deterministic unit of transcript used to build chunk prompts."""

    source_segment_index: int
    start: float
    end: float
    text: str
    speaker: str = ""


@dataclass(frozen=True)
class AnalysisChunk:
    """Chunk passed to the scout model."""

    chunk_id: str
    start: float
    end: float
    text: str
    speaker_set: tuple[str, ...] = ()
    sentence_count: int = 0
    overlap_left: bool = False
    overlap_right: bool = False
    source_segment_ids: tuple[int, ...] = ()

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "speaker_set": list(self.speaker_set),
            "sentence_count": self.sentence_count,
            "overlap_left": self.overlap_left,
            "overlap_right": self.overlap_right,
            "source_segment_ids": list(self.source_segment_ids),
        }


@dataclass(frozen=True)
class MomentRecord:
    """Final or intermediate moment record produced by the analysis stage."""

    start: float
    end: float
    title: str
    quote: str
    why: str
    # The model's own 1-10 rating. Stays on that scale end to end: the cut
    # stage filters on it via processing.quality_filters.min_score.
    score: float
    clip_type: str = "reel"
    hook: str = ""
    caption: str = ""
    hashtags: tuple[str, ...] = ()
    # Combined heuristic ranking value. Only meaningful when comparing
    # candidates against each other, so it is kept apart from `score`.
    priority: float | None = None
    judge_score: float | None = None
    hook_score: float | None = None
    completeness_score: float | None = None
    speaker_focus: float | None = None
    subtitle_readability_score: float | None = None
    crop_confidence: float | None = None
    selection_stage: str = ""
    source_chunk_ids: tuple[str, ...] = ()
    speaker: str = ""
    speaker_confidence: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "title": self.title,
            "quote": self.quote,
            "why": self.why,
            "score": float(self.score),
            "clip_type": self.clip_type,
            "hook": self.hook,
            "caption": self.caption,
            "hashtags": list(self.hashtags),
        }
        optional = {
            "priority": self.priority,
            "judge_score": self.judge_score,
            "hook_score": self.hook_score,
            "completeness_score": self.completeness_score,
            "speaker_focus": self.speaker_focus,
            "subtitle_readability_score": self.subtitle_readability_score,
            "crop_confidence": self.crop_confidence,
            "selection_stage": self.selection_stage,
            "source_chunk_ids": list(self.source_chunk_ids),
            "speaker": self.speaker,
            "speaker_confidence": self.speaker_confidence,
        }
        for key, value in optional.items():
            if value not in (None, "", [], ()):
                data[key] = value
        for key, value in self.extra.items():
            if key not in data and value is not None:
                data[key] = value
        return data


def coerce_moment_record(raw: Mapping[str, Any]) -> MomentRecord | None:
    """Convert a raw dict into a validated moment record."""

    try:
        start = float(raw.get("start", 0.0))
        end = float(raw.get("end", 0.0))
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    if not raw.get("title") and not raw.get("quote") and not raw.get("why"):
        return None

    hashtags_raw = raw.get("hashtags")
    hashtags: tuple[str, ...] = ()
    if isinstance(hashtags_raw, list):
        hashtags = tuple(str(item).strip() for item in hashtags_raw if str(item).strip())

    source_chunk_ids_raw = raw.get("source_chunk_ids")
    source_chunk_ids: tuple[str, ...] = ()
    if isinstance(source_chunk_ids_raw, list):
        source_chunk_ids = tuple(
            str(item).strip()
            for item in source_chunk_ids_raw
            if str(item).strip()
        )

    def _float_or_none(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    extra = {
        key: value
        for key, value in raw.items()
        if key not in {
            "start",
            "end",
            "title",
            "quote",
            "why",
            "score",
            "clip_type",
            "hook",
            "caption",
            "hashtags",
            "priority",
            "judge_score",
            "hook_score",
            "completeness_score",
            "speaker_focus",
            "subtitle_readability_score",
            "crop_confidence",
            "selection_stage",
            "source_chunk_ids",
            "speaker",
            "speaker_confidence",
        }
    }

    return MomentRecord(
        start=start,
        end=end,
        title=str(raw.get("title", "")).strip(),
        quote=str(raw.get("quote", "")).strip(),
        why=str(raw.get("why", "")).strip(),
        score=_float_or_none(raw.get("score")) or 0.0,
        clip_type=str(raw.get("clip_type", "reel") or "reel").strip() or "reel",
        hook=str(raw.get("hook", "")).strip(),
        caption=str(raw.get("caption", "")).strip(),
        hashtags=hashtags,
        priority=_float_or_none(raw.get("priority")),
        judge_score=_float_or_none(raw.get("judge_score")),
        hook_score=_float_or_none(raw.get("hook_score")),
        completeness_score=_float_or_none(raw.get("completeness_score")),
        speaker_focus=_float_or_none(raw.get("speaker_focus")),
        subtitle_readability_score=_float_or_none(
            raw.get("subtitle_readability_score"),
        ),
        crop_confidence=_float_or_none(raw.get("crop_confidence")),
        selection_stage=str(raw.get("selection_stage", "")).strip(),
        source_chunk_ids=source_chunk_ids,
        speaker=str(raw.get("speaker", "")).strip(),
        speaker_confidence=_float_or_none(raw.get("speaker_confidence")),
        extra=extra,
    )
