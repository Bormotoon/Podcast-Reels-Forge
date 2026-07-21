"""Deterministic transcript chunking helpers for analysis prompts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from podcast_reels_forge.analysis.contracts import AnalysisChunk, AnalysisChunkUnit

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _split_text_to_sentences(text: str) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return parts or [text]


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _segment_word_timings(segment: Mapping[str, Any]) -> list[tuple[float, float, str]]:
    """Usable (start, end, text) triples from a segment's word list."""

    raw_words = segment.get("words")
    if not isinstance(raw_words, list):
        return []

    words: list[tuple[float, float, str]] = []
    for raw_word in raw_words:
        if not isinstance(raw_word, Mapping):
            continue
        text = str(raw_word.get("word", "")).strip()
        start = _coerce_float(raw_word.get("start"), -1.0)
        end = _coerce_float(raw_word.get("end"), -1.0)
        if not text or start < 0 or end < start:
            return []
        words.append((start, end, text))
    return words


def _sentence_spans_from_words(
    sentence_parts: Sequence[str],
    words: Sequence[tuple[float, float, str]],
) -> list[tuple[float, float]] | None:
    """Map sentences onto real word timings by consuming words in order.

    Returns None when the split does not line up with the word list, so the
    caller can fall back to interpolation rather than emit bogus timings.
    """

    if not words:
        return None

    spans: list[tuple[float, float]] = []
    cursor = 0
    for part in sentence_parts:
        part_word_count = len(part.split())
        if part_word_count <= 0 or cursor + part_word_count > len(words):
            return None
        first = words[cursor]
        last = words[cursor + part_word_count - 1]
        if last[1] <= first[0]:
            return None
        spans.append((first[0], last[1]))
        cursor += part_word_count

    # Every word must be accounted for; a leftover tail means the sentence
    # split and the word list disagree.
    if cursor != len(words):
        return None
    return spans


def transcript_units_from_segments(segments: Sequence[Mapping[str, Any]]) -> list[AnalysisChunkUnit]:
    """Split transcript segments into sentence-aware units.

    Sentence boundaries are taken from real word timings when the transcript
    carries them (timing_version 2), and only fall back to distributing the
    segment duration by character count when it does not.
    """

    units: list[AnalysisChunkUnit] = []
    for idx, segment in enumerate(segments):
        start = _coerce_float(segment.get("start", 0.0))
        end = _coerce_float(segment.get("end", 0.0))
        text = _clean_text(str(segment.get("text", "")))
        if end <= start or not text:
            continue

        # `or ""` matters: a transcript without diarization stores speaker=None,
        # and str(None) is the truthy "None", which prefixed every single line
        # sent to the model with a literal "(None)".
        speaker = str(segment.get("speaker") or "").strip()
        sentence_parts = _split_text_to_sentences(text)
        if len(sentence_parts) == 1:
            units.append(
                AnalysisChunkUnit(
                    source_segment_index=idx,
                    start=start,
                    end=end,
                    text=text,
                    speaker=speaker,
                ),
            )
            continue

        spans = _sentence_spans_from_words(sentence_parts, _segment_word_timings(segment))
        if spans is not None:
            for part, (part_start, part_end) in zip(sentence_parts, spans):
                units.append(
                    AnalysisChunkUnit(
                        source_segment_index=idx,
                        start=round(part_start, 3),
                        end=round(part_end, 3),
                        text=part,
                        speaker=speaker,
                    ),
                )
            continue

        duration = max(0.01, end - start)
        total_chars = max(1, sum(max(len(part), 1) for part in sentence_parts))
        cursor = start
        for part_index, part in enumerate(sentence_parts):
            share = max(len(part), 1) / total_chars
            next_cursor = end if part_index == len(sentence_parts) - 1 else cursor + duration * share
            next_cursor = max(cursor + 0.01, next_cursor)
            units.append(
                AnalysisChunkUnit(
                    source_segment_index=idx,
                    start=round(cursor, 3),
                    end=round(next_cursor, 3),
                    text=part,
                    speaker=speaker,
                ),
            )
            cursor = next_cursor

    units.sort(key=lambda unit: (unit.start, unit.end))
    return units


def _join_units(units: Sequence[AnalysisChunkUnit]) -> str:
    lines: list[str] = []
    for unit in units:
        prefix = f"({unit.speaker}) " if unit.speaker else ""
        lines.append(f"[{int(unit.start)}-{int(unit.end)}] {prefix}{unit.text}")
    return "\n".join(lines)


def build_analysis_chunks(
    segments: Sequence[Mapping[str, Any]],
    *,
    chunk_seconds: int,
    max_chars: int,
    overlap_seconds: int = 30,
) -> list[AnalysisChunk]:
    """Build deterministic chunk objects with overlap and speaker metadata."""

    units = transcript_units_from_segments(segments)
    if not units:
        return []

    chunks: list[AnalysisChunk] = []
    index = 0
    chunk_index = 1
    overlap_seconds = max(0, int(overlap_seconds))

    while index < len(units):
        start_unit = units[index]
        chunk_units = [start_unit]
        chunk_start = start_unit.start
        chunk_end = start_unit.end
        source_ids = {start_unit.source_segment_index}
        speaker_set = {start_unit.speaker} if start_unit.speaker else set()

        next_index = index + 1
        while next_index < len(units):
            candidate = units[next_index]
            proposed_end = max(chunk_end, candidate.end)
            proposed_text = _join_units(chunk_units + [candidate])
            if (
                proposed_end - chunk_start > float(chunk_seconds)
                or len(proposed_text) > int(max_chars)
            ):
                break
            chunk_units.append(candidate)
            chunk_end = proposed_end
            source_ids.add(candidate.source_segment_index)
            if candidate.speaker:
                speaker_set.add(candidate.speaker)
            next_index += 1

        overlap_left = index > 0
        overlap_right = next_index < len(units)
        chunk_text = _join_units(chunk_units)
        chunk = AnalysisChunk(
            chunk_id=f"chunk_{chunk_index:03d}",
            start=round(chunk_start, 3),
            end=round(chunk_end, 3),
            text=chunk_text,
            speaker_set=tuple(sorted(s for s in speaker_set if s)),
            sentence_count=len(chunk_units),
            overlap_left=overlap_left,
            overlap_right=overlap_right,
            source_segment_ids=tuple(sorted(source_ids)),
        )
        chunks.append(chunk)
        chunk_index += 1

        if next_index <= index:
            index += 1
            continue

        if overlap_seconds > 0 and next_index < len(units):
            overlap_start = max(chunk_start, chunk_end - float(overlap_seconds))
            new_index = next_index
            for candidate_index in range(index + 1, len(units)):
                if units[candidate_index].start >= overlap_start:
                    new_index = candidate_index
                    break
            index = max(index + 1, new_index)
        else:
            index = next_index

    return chunks
