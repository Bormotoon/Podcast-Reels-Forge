"""Typed analysis contracts used by the staged pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
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
    # Stable identity assigned by Python when the scout reports a candidate.
    # Cleanup and judge refer to candidates by this id instead of echoing the
    # records back, so they cannot move a clip or rewrite its quote.
    candidate_id: str = ""
    # Ids of candidates folded into this one by cleanup (lineage).
    merged_ids: tuple[str, ...] = ()
    # Short machine-readable reasons ("surprise", "payoff", ...).
    reason_codes: tuple[str, ...] = ()
    # Combined heuristic ranking value. Only meaningful when comparing
    # candidates against each other, so it is kept apart from `score`.
    priority: float | None = None
    judge_score: float | None = None
    hook_score: float | None = None
    completeness_score: float | None = None
    speaker_focus: float | None = None
    subtitle_readability_score: float | None = None
    # How well the clip length fits its clip type (0..1). Used to live under
    # the misleading name `crop_confidence`, which has nothing to do with it.
    duration_fit_score: float | None = None
    # How well `quote` matches what was actually said in [start, end]. None
    # when verification did not run (no word timings, or disabled).
    quote_match_ratio: float | None = None
    # "exact" (contiguous normalized match), "fuzzy" (bounded token
    # alignment) or "none". Empty when verification did not run.
    quote_match_method: str = ""
    # Where the quote was actually found in the transcript.
    quote_start: float | None = None
    quote_end: float | None = None
    speech_rate_wps: float | None = None
    # Measured from the source audio; None when it was unavailable.
    audio_energy_db: float | None = None
    audio_silence_ratio: float | None = None
    selection_stage: str = ""
    source_chunk_ids: tuple[str, ...] = ()
    speaker: str = ""
    speaker_confidence: float | None = None
    # Metadata fields Python generated rather than the model or transcript
    # supplied (e.g. a caption built from the title). Kept explicit so
    # derived prose is never mistaken for transcript evidence.
    derived_fields: tuple[str, ...] = ()
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
            "candidate_id": self.candidate_id,
            "merged_ids": list(self.merged_ids),
            "reason_codes": list(self.reason_codes),
            "priority": self.priority,
            "judge_score": self.judge_score,
            "hook_score": self.hook_score,
            "completeness_score": self.completeness_score,
            "speaker_focus": self.speaker_focus,
            "subtitle_readability_score": self.subtitle_readability_score,
            "duration_fit_score": self.duration_fit_score,
            "quote_match_ratio": self.quote_match_ratio,
            "quote_match_method": self.quote_match_method,
            "quote_start": _rounded(self.quote_start),
            "quote_end": _rounded(self.quote_end),
            "speech_rate_wps": self.speech_rate_wps,
            "audio_energy_db": self.audio_energy_db,
            "audio_silence_ratio": self.audio_silence_ratio,
            "selection_stage": self.selection_stage,
            "source_chunk_ids": list(self.source_chunk_ids),
            "speaker": self.speaker,
            "speaker_confidence": self.speaker_confidence,
            "derived_fields": list(self.derived_fields),
        }
        for key, value in optional.items():
            if value not in (None, "", [], ()):
                data[key] = value
        for key, value in self.extra.items():
            if key not in data and value is not None:
                data[key] = value
        return data


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)


# Every named field; anything else in a raw dict is carried in `extra`.
_KNOWN_KEYS = frozenset(f.name for f in fields(MomentRecord)) - {"extra"}
# Keys that must never ride along in `extra`: `crop_confidence` is the old
# name of `duration_fit_score` and would reintroduce the wrong data from
# stale intermediate files; `evidence` is folded into `why`.
_DROPPED_KEYS = frozenset({"crop_confidence", "evidence"})


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def has_quote_evidence(raw: Mapping[str, Any] | MomentRecord) -> bool:
    """Whether a candidate carries a non-trivial quote to verify.

    A candidate without one cannot be checked against the transcript at all,
    so it is never allowed to proceed as a scout result.
    """

    quote = raw.quote if isinstance(raw, MomentRecord) else raw.get("quote", "")
    return len(str(quote or "").split()) >= 2


def coerce_moment_record(raw: Mapping[str, Any]) -> MomentRecord | None:
    """Convert a raw dict into a validated moment record."""

    try:
        start = float(raw.get("start", 0.0))
        end = float(raw.get("end", 0.0))
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    # The scout's `evidence` is the same thing the older prompts called `why`.
    why = str(raw.get("why", "") or raw.get("evidence", "") or "").strip()
    if not raw.get("title") and not raw.get("quote") and not why:
        return None

    extra = {
        key: value
        for key, value in raw.items()
        if key not in _KNOWN_KEYS and key not in _DROPPED_KEYS
    }

    return MomentRecord(
        start=start,
        end=end,
        title=str(raw.get("title", "") or "").strip(),
        quote=str(raw.get("quote", "") or "").strip(),
        why=why,
        score=_float_or_none(raw.get("score")) or 0.0,
        clip_type=str(raw.get("clip_type", "reel") or "reel").strip() or "reel",
        hook=str(raw.get("hook", "") or "").strip(),
        caption=str(raw.get("caption", "") or "").strip(),
        hashtags=_str_tuple(raw.get("hashtags")),
        candidate_id=str(raw.get("candidate_id", "") or "").strip(),
        merged_ids=_str_tuple(raw.get("merged_ids")),
        reason_codes=_str_tuple(raw.get("reason_codes")),
        priority=_float_or_none(raw.get("priority")),
        judge_score=_float_or_none(raw.get("judge_score")),
        hook_score=_float_or_none(raw.get("hook_score")),
        completeness_score=_float_or_none(raw.get("completeness_score")),
        speaker_focus=_float_or_none(raw.get("speaker_focus")),
        subtitle_readability_score=_float_or_none(
            raw.get("subtitle_readability_score"),
        ),
        duration_fit_score=_float_or_none(raw.get("duration_fit_score")),
        quote_match_ratio=_float_or_none(raw.get("quote_match_ratio")),
        quote_match_method=str(raw.get("quote_match_method", "") or "").strip(),
        quote_start=_float_or_none(raw.get("quote_start")),
        quote_end=_float_or_none(raw.get("quote_end")),
        speech_rate_wps=_float_or_none(raw.get("speech_rate_wps")),
        audio_energy_db=_float_or_none(raw.get("audio_energy_db")),
        audio_silence_ratio=_float_or_none(raw.get("audio_silence_ratio")),
        selection_stage=str(raw.get("selection_stage", "") or "").strip(),
        source_chunk_ids=_str_tuple(raw.get("source_chunk_ids")),
        speaker=str(raw.get("speaker", "") or "").strip(),
        speaker_confidence=_float_or_none(raw.get("speaker_confidence")),
        derived_fields=_str_tuple(raw.get("derived_fields")),
        extra=extra,
    )


def replace_record(record: MomentRecord, **changes: Any) -> MomentRecord:
    """A copy of ``record`` with ``changes`` applied, re-validated.

    Falls back to the original when the change would make the record invalid
    (e.g. an inverted interval), so callers never lose a record to a bad edit.
    """

    return coerce_moment_record({**record.to_dict(), **changes}) or record
