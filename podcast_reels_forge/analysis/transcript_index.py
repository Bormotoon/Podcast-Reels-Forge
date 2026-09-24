"""RU: Индекс транскрипта: пословные и пофразовые тайминги для анализа.

Транскрипт (timing_version 2) уже содержит точные тайминги слов
(``segments[].words``) и границы предложений (``sentences``). Этот модуль
даёт к ним быстрый доступ: найти слова в интервале, собрать текст отрезка,
подтянуть границу клипа к ближайшей границе фразы или слова.

EN: Transcript index: word- and sentence-level timings for the analysis.

A timing_version-2 transcript already carries exact word timings
(``segments[].words``) and sentence boundaries (``sentences``). This module
makes them queryable: find the words inside an interval, build the text of a
span, and snap a clip boundary to the nearest sentence or word edge.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def normalize_for_compare(text: str) -> str:
    """RU: Нормализует текст: без пунктуации/регистра, ё→е.

    EN: Normalize text for comparison: strip punctuation and case, fold ё→е.
    Mirrors the proofread stage's guardrail so both compare text the same way.
    """

    lowered = str(text).lower().replace("ё", "е")
    return " ".join(_WORD_RE.findall(lowered))


def normalized_tokens(text: str) -> list[str]:
    """Word tokens of ``text`` after normalization."""

    normalized = normalize_for_compare(text)
    return normalized.split() if normalized else []


@dataclass(frozen=True)
class TimedWord:
    """A single word with its timing."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TimedSentence:
    """A sentence span."""

    start: float
    end: float
    text: str


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class TranscriptIndex:
    """Queryable view over a transcript's word and sentence timings."""

    def __init__(
        self,
        words: Sequence[TimedWord] = (),
        sentences: Sequence[TimedSentence] = (),
    ) -> None:
        self.words: list[TimedWord] = sorted(words, key=lambda w: (w.start, w.end))
        self.sentences: list[TimedSentence] = sorted(
            sentences, key=lambda s: (s.start, s.end),
        )
        self._word_starts = [w.start for w in self.words]
        self._sentence_starts = [s.start for s in self.sentences]
        self._sentence_ends = sorted(s.end for s in self.sentences)

    @classmethod
    def from_transcript(cls, data: Mapping[str, Any]) -> "TranscriptIndex":
        """Build an index from parsed transcript JSON.

        Tolerates transcripts without word timings (older timing_version, or a
        faster-whisper fallback path that dropped them): the index simply ends
        up with fewer anchors and the callers degrade to no-ops.
        """

        words: list[TimedWord] = []
        raw_segments = data.get("segments")
        if isinstance(raw_segments, list):
            for segment in raw_segments:
                if not isinstance(segment, Mapping):
                    continue
                raw_words = segment.get("words")
                if not isinstance(raw_words, list):
                    continue
                for raw_word in raw_words:
                    if not isinstance(raw_word, Mapping):
                        continue
                    text = str(raw_word.get("word", "")).strip()
                    start = _coerce_float(raw_word.get("start"), -1.0)
                    end = _coerce_float(raw_word.get("end"), -1.0)
                    if not text or start < 0 or end < start:
                        continue
                    words.append(TimedWord(start=start, end=end, text=text))

        sentences: list[TimedSentence] = []
        raw_sentences = data.get("sentences")
        if isinstance(raw_sentences, list):
            for raw_sentence in raw_sentences:
                if not isinstance(raw_sentence, Mapping):
                    continue
                text = str(raw_sentence.get("text", "")).strip()
                start = _coerce_float(raw_sentence.get("start"), -1.0)
                end = _coerce_float(raw_sentence.get("end"), -1.0)
                if not text or start < 0 or end <= start:
                    continue
                sentences.append(TimedSentence(start=start, end=end, text=text))

        # Fall back to segment spans when the transcript has no sentence groups.
        if not sentences and isinstance(raw_segments, list):
            for segment in raw_segments:
                if not isinstance(segment, Mapping):
                    continue
                text = str(segment.get("text", "")).strip()
                start = _coerce_float(segment.get("start"), -1.0)
                end = _coerce_float(segment.get("end"), -1.0)
                if not text or start < 0 or end <= start:
                    continue
                sentences.append(TimedSentence(start=start, end=end, text=text))

        return cls(words=words, sentences=sentences)

    def __bool__(self) -> bool:
        return bool(self.words or self.sentences)

    def words_between(self, start: float, end: float) -> list[TimedWord]:
        """Words whose span overlaps ``[start, end]``."""

        if not self.words or end <= start:
            return []
        # Words are short, so scanning back a little from the first word
        # starting at/after `start` catches any that straddle the boundary.
        index = max(0, bisect.bisect_left(self._word_starts, start) - 8)
        found: list[TimedWord] = []
        for word in self.words[index:]:
            if word.start >= end:
                break
            if word.end > start:
                found.append(word)
        return found

    def text_between(self, start: float, end: float, *, max_chars: int = 0) -> str:
        """Plain text of the span, optionally truncated on a word boundary."""

        words = self.words_between(start, end)
        if words:
            text = " ".join(word.text for word in words).strip()
        else:
            text = " ".join(
                sentence.text
                for sentence in self.sentences
                if sentence.end > start and sentence.start < end
            ).strip()

        if max_chars > 0 and len(text) > max_chars:
            clipped = text[:max_chars]
            last_space = clipped.rfind(" ")
            if last_space > max_chars * 0.6:
                clipped = clipped[:last_space]
            text = clipped.rstrip() + "…"
        return text

    # RU: Штраф (в секундах) за якорь на границе слова, а не фразы: граница
    # предложения выигрывает, пока она не дальше границы слова на эту величину.
    # EN: Cost, in seconds, of anchoring on a word edge instead of a sentence
    # edge: a sentence boundary wins unless it is this much further away.
    WORD_ANCHOR_PENALTY_S = 1.5
    # Moving a boundary inward trims the clip; it is allowed for sentence
    # edges only (dropping a dangling fragment), and costs a little extra so an
    # equally close outward edge wins.
    INWARD_PENALTY_S = 0.5

    def snap_start(
        self,
        value: float,
        *,
        max_shift: float,
        limit: float | None = None,
    ) -> float:
        """Move a clip start onto the cheapest nearby speech boundary.

        Clips that begin mid-word or mid-sentence are the visible symptom of
        LLM-picked boundaries. Candidates are sentence starts on either side
        of ``value`` and word starts before it; the cost is the shift itself,
        plus a penalty for word anchors and for moving inward. ``limit`` is
        the latest allowed start (the quote start), so snapping can never cut
        into the clip's evidence.
        """

        if max_shift <= 0:
            return value
        latest = value + max_shift if limit is None else min(value + max_shift, limit)
        options: list[tuple[float, float]] = []
        for anchor in self._values_within(self._sentence_starts, value - max_shift, latest):
            inward = self.INWARD_PENALTY_S if anchor > value else 0.0
            options.append((abs(value - anchor) + inward, anchor))
        word_start = self._nearest_at_or_before(self._word_starts, value)
        if word_start is not None and 0.0 <= value - word_start <= max_shift:
            options.append((value - word_start + self.WORD_ANCHOR_PENALTY_S, word_start))
        if not options:
            return value
        return min(options)[1]

    def snap_end(
        self,
        value: float,
        *,
        max_shift: float,
        limit: float | None = None,
    ) -> float:
        """Move a clip end onto the cheapest nearby speech boundary.

        Mirror of :meth:`snap_start`; ``limit`` is the earliest allowed end
        (the quote end).
        """

        if max_shift <= 0:
            return value
        earliest = value - max_shift if limit is None else max(value - max_shift, limit)
        options: list[tuple[float, float]] = []
        for anchor in self._values_within(self._sentence_ends, earliest, value + max_shift):
            inward = self.INWARD_PENALTY_S if anchor < value else 0.0
            options.append((abs(anchor - value) + inward, anchor))
        word_end = self._nearest_word_end_at_or_after(value)
        if word_end is not None and 0.0 <= word_end - value <= max_shift:
            options.append((word_end - value + self.WORD_ANCHOR_PENALTY_S, word_end))
        if not options:
            return value
        return min(options)[1]

    def timed_tokens(self, start: float, end: float) -> list[tuple[str, float, float]]:
        """Normalized tokens of ``[start, end]`` with their timings.

        Uses word timings when the transcript has them. Without them (older
        transcripts) the sentence text is tokenized and timed by even
        interpolation across the sentence — coarse, but enough to tell a real
        quote from an invented one.
        """

        tokens: list[tuple[str, float, float]] = []
        if self.words:
            for word in self.words_between(start, end):
                for token in normalized_tokens(word.text)[:1]:
                    tokens.append((token, word.start, word.end))
            return tokens

        for sentence in self.sentences:
            if sentence.end <= start or sentence.start >= end:
                continue
            parts = normalized_tokens(sentence.text)
            if not parts:
                continue
            step = (sentence.end - sentence.start) / len(parts)
            for offset, token in enumerate(parts):
                token_start = sentence.start + offset * step
                tokens.append((token, token_start, token_start + step))
        return tokens

    def speech_rate(self, start: float, end: float) -> float | None:
        """Words per second across the span, or None without word timings."""

        if not self.words or end <= start:
            return None
        words = self.words_between(start, end)
        if not words:
            return None
        return round(len(words) / (end - start), 4)

    @staticmethod
    def _nearest_at_or_before(values: Sequence[float], target: float) -> float | None:
        index = bisect.bisect_right(values, target)
        return values[index - 1] if index else None

    @staticmethod
    def _values_within(values: Sequence[float], low: float, high: float) -> list[float]:
        if high < low:
            return []
        return list(values[bisect.bisect_left(values, low) : bisect.bisect_right(values, high)])

    def _nearest_word_end_at_or_after(self, target: float) -> float | None:
        index = max(0, bisect.bisect_left(self._word_starts, target) - 8)
        for word in self.words[index:]:
            if word.end >= target:
                return word.end
        return None
