"""RU: Фактический интервал нарезки клипа с учётом padding.

Когда в транскрипте есть тайминги слов, клип режется по речи: старт — чуть
раньше первого слова фразы (но не раньше конца предыдущего слова), финиш —
чуть позже последнего слова (но не позже начала следующего). Фиксированный
``reel_padding`` остаётся только запасным вариантом для транскрипта без слов.

EN: The interval a clip is actually cut from, padding included. The cut, its
burned subtitles and later subtitle re-syncs must all agree on it.

With word timings the edges follow the speech: a short lead-in before the
first word of the opening sentence, a short tail after the last word of the
closing one, and neither reaches into a neighbouring word. The fixed
``reel_padding`` is only the fallback for a transcript without words.
"""

from __future__ import annotations

import bisect
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from podcast_reels_forge.analysis.transcript_index import TranscriptIndex

LOG = logging.getLogger("forge")


@dataclass(frozen=True)
class ClipEdges:
    """How a clip's edges are placed around its speech.

    ``lead_in_s`` — silence kept before the first word (lets the player start
    before anyone talks); ``tail_s`` — silence kept after the last word;
    ``guard_s`` — distance kept from a neighbouring word, whose Whisper timing
    is only accurate to ~0.1 s; ``max_extend_s`` — how far an edge may grow
    to reach the start or end of the sentence it falls into; ``max_trim_s`` —
    how long a dangling fragment of a neighbouring sentence may be for the
    edge to drop it instead.
    """

    lead_in_s: float = 0.6
    tail_s: float = 0.3
    guard_s: float = 0.1
    max_extend_s: float = 4.0
    max_trim_s: float = 1.5

    @classmethod
    def from_config(cls, conf: Any) -> "ClipEdges":
        """Build from ``processing.clip_edges``; missing or bad keys keep defaults."""

        if not isinstance(conf, Mapping):
            return cls()
        values: dict[str, float] = {}
        for key in ("lead_in_s", "tail_s", "guard_s", "max_extend_s", "max_trim_s"):
            try:
                if key in conf:
                    values[key] = max(0.0, float(conf[key]))
            except (TypeError, ValueError):
                continue
        return cls(**values)

    def cli_args(self) -> list[str]:
        """The video/re-render scripts' flags for these edges."""

        return [
            "--lead-in", str(self.lead_in_s),
            "--tail", str(self.tail_s),
            "--edge-guard", str(self.guard_s),
            "--max-extend", str(self.max_extend_s),
            "--max-trim", str(self.max_trim_s),
        ]

    @classmethod
    def from_args(cls, args: Any) -> "ClipEdges":
        """Build from the scripts' parsed flags (see :meth:`add_arguments`)."""

        return cls(
            lead_in_s=max(0.0, float(args.lead_in)),
            tail_s=max(0.0, float(args.tail)),
            guard_s=max(0.0, float(args.edge_guard)),
            max_extend_s=max(0.0, float(args.max_extend)),
            max_trim_s=max(0.0, float(args.max_trim)),
        )

    @classmethod
    def add_arguments(cls, parser: Any) -> None:
        """Register the flags :meth:`cli_args` emits on an argparse parser."""

        defaults = cls()
        parser.add_argument(
            "--lead-in", type=float, default=defaults.lead_in_s,
            help="Silence kept before the first word (needs a transcript with word timings)",
        )
        parser.add_argument(
            "--tail", type=float, default=defaults.tail_s,
            help="Silence kept after the last word",
        )
        parser.add_argument(
            "--edge-guard", type=float, default=defaults.guard_s,
            help="Distance kept from the neighbouring words",
        )
        parser.add_argument(
            "--max-extend", type=float, default=defaults.max_extend_s,
            help="How far an edge may grow to reach its sentence's start/end",
        )
        parser.add_argument(
            "--max-trim", type=float, default=defaults.max_trim_s,
            help="Longest fragment of a neighbouring sentence an edge drops",
        )


def moment_bounds(moment: Mapping[str, Any]) -> tuple[float, float]:
    """A moment's (start, end); (0, 0) when they are not numbers."""

    try:
        start = float(moment.get("start", 0) or 0)
        end = float(moment.get("end", 0) or 0)
    except (TypeError, ValueError):
        return 0.0, 0.0
    return start, end


def load_speech_index(path: Path | None) -> TranscriptIndex | None:
    """Word/sentence timings of a transcript JSON; None when unavailable."""

    if path is None or not path.exists():
        return None
    try:
        index = TranscriptIndex.from_transcript(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        LOG.warning("Transcript unreadable for clip edges (%s); using fixed padding", exc)
        return None
    return index if index.words else None


def speech_edges(
    index: TranscriptIndex,
    start: float,
    end: float,
    edges: ClipEdges,
) -> tuple[float, float, float, float] | None:
    """Speech span of a clip and the silence available around it.

    Returns ``(speech_start, speech_end, lead, tail)``, or None when the
    transcript has no words in the clip. The span opens on the first word of
    a sentence and closes on the last word of one: an edge that falls inside
    a sentence either grows to that sentence's start/end (up to
    ``max_extend_s``) or drops the dangling fragment (up to ``max_trim_s``),
    whichever moves it less. ``lead``/``tail`` are how much silence may be
    added on each side without touching the previous/next word.
    """

    words = index.words
    if not words or end <= start:
        return None
    inside = index.words_between(start, end)
    if not inside:
        return None
    first = words.index(inside[0])
    last = words.index(inside[-1])
    starts = index.sentence_start_words
    # Dropping a fragment must leave the clip most of its speech.
    max_trim = min(edges.max_trim_s, (words[last].end - words[first].start) / 2.0)

    if first not in starts:
        options: list[tuple[float, int]] = []
        opening = starts[bisect.bisect_right(starts, first) - 1]
        grow = words[first].start - words[opening].start
        if grow <= edges.max_extend_s:
            options.append((grow, opening))
        following = bisect.bisect_right(starts, first)
        if following < len(starts) and starts[following] <= last:
            trim = words[starts[following]].start - words[first].start
            if trim <= max_trim:
                options.append((trim + 1e-3, starts[following]))
        if options:
            first = min(options)[1]

    # A word closes a sentence when the next one opens a sentence.
    if last + 1 < len(words) and last + 1 not in starts:
        options = []
        following = bisect.bisect_right(starts, last)
        closing = (starts[following] if following < len(starts) else len(words)) - 1
        grow = words[closing].end - words[last].end
        if grow <= edges.max_extend_s:
            options.append((grow, closing))
        previous = starts[following - 1] - 1
        if previous >= first:
            trim = words[last].end - words[previous].end
            if trim <= max_trim:
                options.append((trim + 1e-3, previous))
        if options:
            last = min(options)[1]

    speech_start, speech_end = words[first].start, words[last].end
    lead = edges.lead_in_s
    if first > 0:
        previous_end = max(w.end for w in words[max(0, first - 8):first])
        lead = min(lead, max(0.0, speech_start - previous_end - edges.guard_s))
    tail = edges.tail_s
    if last + 1 < len(words):
        tail = min(tail, max(0.0, words[last + 1].start - speech_end - edges.guard_s))
    return speech_start, speech_end, lead, tail


def padded_intervals(
    bounds: Sequence[tuple[float, float]],
    padding: float,
    *,
    index: TranscriptIndex | None = None,
    edges: ClipEdges | None = None,
) -> list[tuple[float, float]]:
    """The interval each clip is actually cut from, padding included.

    With ``index`` (a transcript with word timings) and ``edges`` each clip
    is first fitted to its speech (see :func:`speech_edges`) and gets only the
    silence around it; ``padding`` then applies only to clips the transcript
    has no words for.

    Padding widens a clip on both sides, but never into a neighbouring clip:
    towards a neighbour it is capped at half the gap between them, and it is
    dropped on a side that already overlaps one (the selection allows a small
    overlap). Otherwise two adjacent moments would share the same seconds of
    footage — and the same subtitles — in two reels. The subtitles, the
    encode and the QA duration check all use this one interval.
    """

    padding = max(0.0, float(padding))
    spans: list[tuple[float, float, float, float]] = []
    for start, end in bounds:
        fitted = (
            speech_edges(index, start, end, edges)
            if index is not None and edges is not None
            else None
        )
        spans.append(fitted if fitted is not None else (start, end, padding, padding))
    bounds = [(start, end) for start, end, _, _ in spans]

    result: list[tuple[float, float]] = []
    for position, (start, end, before, after) in enumerate(spans):
        if before > 0 or after > 0:
            for other, (o_start, o_end) in enumerate(bounds):
                if other == position or o_end <= o_start:
                    continue
                if o_end <= start:
                    before = min(before, (start - o_end) / 2.0)
                elif o_start >= end:
                    after = min(after, (o_start - end) / 2.0)
                else:
                    if o_start < start:
                        before = 0.0
                    if o_end > end:
                        after = 0.0
        result.append((max(0.0, start - before), end + after))
    return result
