"""RU: Перенос пословных таймингов на исправленный текст.

Вычитка меняет `segments[].text`, а `words` (с таймингами Whisper) остаются
сырыми. Из-за этого караоке-субтитры исправленной фразы переходили на
пропорциональную интерполяцию, а цитаты сверялись не с тем текстом, который
видела модель. Здесь слова исправленного текста сопоставляются с сырыми по
токенам, и тайминги переносятся: совпавшее слово берёт свои, заменённый
кусок делит время своих исходных слов, вставка занимает паузу между
соседями.

EN: Carry word timings over to corrected text.

Proofreading changes `segments[].text` while `words` (Whisper's timings)
stay raw. Karaoke subtitles for every corrected sentence then fell back to
proportional interpolation, and quotes were checked against text the model
never saw. Here the corrected words are aligned to the raw ones token by
token and the timings carried over: a matched word keeps its own, a replaced
run shares the time of the words it replaced, an insertion takes the gap
between its neighbours.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from typing import Any

from podcast_reels_forge.analysis.transcript_index import normalize_for_compare


def _norm(text: str) -> str:
    return normalize_for_compare(text).replace(" ", "")


def _timing(word: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        start = float(word["start"])
        end = float(word["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if end < start:
        return None
    return start, end


def _spread(tokens: Sequence[str], start: float, end: float) -> list[tuple[float, float]]:
    """Split ``[start, end]`` across tokens in proportion to their length."""

    weights = [max(1, len(token)) for token in tokens]
    total = float(sum(weights))
    spans: list[tuple[float, float]] = []
    cursor = start
    for weight in weights:
        step = (end - start) * weight / total
        spans.append((round(cursor, 3), round(cursor + step, 3)))
        cursor += step
    return spans


def words_match_text(words: Sequence[Mapping[str, Any]], text: str) -> bool:
    """Whether the word list already spells ``text`` (token for token)."""

    tokens = text.split()
    if len(tokens) != len(words):
        return False
    return all(
        _norm(token) == _norm(str(word.get("word", "")))
        for token, word in zip(tokens, words)
    )


def realign_words(
    raw_words: Sequence[Mapping[str, Any]],
    text: str,
) -> list[dict[str, Any]] | None:
    """Timed words for ``text``, one per whitespace token, from ``raw_words``.

    Returns None when there are no usable timings to carry over. The output
    has exactly one entry per ``text.split()`` token with the token itself as
    ``word``, which is what the subtitle renderer checks before trusting
    real timings.
    """

    raw = [(word, span) for word in raw_words if (span := _timing(word)) is not None]
    tokens = text.split()
    if not raw:
        return None
    if not tokens:
        return []

    raw_norm = [_norm(str(word.get("word", ""))) for word, _span in raw]
    new_norm = [_norm(token) for token in tokens]
    matcher = difflib.SequenceMatcher(None, raw_norm, new_norm, autojunk=False)

    out: list[dict[str, Any] | None] = [None] * len(tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(j2 - j1):
                word, (start, end) = raw[i1 + offset]
                out[j1 + offset] = {
                    "start": start,
                    "end": end,
                    "word": tokens[j1 + offset],
                    "probability": word.get("probability"),
                }
        elif tag == "replace":
            start, end = raw[i1][1][0], raw[i2 - 1][1][1]
            for offset, (w_start, w_end) in enumerate(_spread(tokens[j1:j2], start, end)):
                out[j1 + offset] = {
                    "start": w_start, "end": w_end, "word": tokens[j1 + offset], "probability": None,
                }
        elif tag == "insert":
            before = raw[i1 - 1][1][1] if i1 > 0 else raw[0][1][0]
            after = raw[i1][1][0] if i1 < len(raw) else raw[-1][1][1]
            start, end = min(before, after), max(before, after)
            for offset, (w_start, w_end) in enumerate(_spread(tokens[j1:j2], start, end)):
                out[j1 + offset] = {
                    "start": w_start, "end": w_end, "word": tokens[j1 + offset], "probability": None,
                }
        # "delete": raw words the correction dropped simply lose their timing.

    words = [item for item in out if item is not None]
    # Keep the sequence monotonic: a zero-width insertion at a boundary must
    # not start before the word it follows ends.
    cursor = words[0]["start"] if words else 0.0
    for word in words:
        word["start"] = round(max(float(word["start"]), cursor), 3)
        word["end"] = round(max(float(word["end"]), float(word["start"])), 3)
        cursor = word["start"]
    return words


def realign_transcript_words(segments: Sequence[dict[str, Any]]) -> int:
    """Realign every segment whose text no longer matches its words, in place.

    The original list is kept as ``raw_words`` for audit. Returns the number
    of segments that were realigned.
    """

    changed = 0
    for segment in segments:
        raw = segment.get("words")
        text = str(segment.get("text", ""))
        if not isinstance(raw, list) or not raw or words_match_text(raw, text):
            continue
        realigned = realign_words(raw, text)
        if realigned is None:
            continue
        segment.setdefault("raw_words", raw)
        segment["words"] = realigned
        changed += 1
    return changed
