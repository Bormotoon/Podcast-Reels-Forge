"""RU: Разбиение транскрипта на реплики спикеров по данным диаризации.

Whisper режет речь по паузам, а не по говорящим: один сегмент на 36 секунд
запросто содержит реплики троих. Поэтому спикер назначается каждому слову
(пословные тайминги есть в транскрипте), и уже подряд идущие слова одного
голоса собираются в реплику.

EN: Splitting a transcript into speaker turns using diarization.

Whisper segments on pauses, not on speakers: a single 36-second segment happily
contains three people talking. So the speaker is assigned per word (the
transcript carries word timings) and consecutive words from one voice are then
collected into a turn.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# A turn shorter than this is almost always a diarization wobble in the middle
# of someone else's sentence ("да", "угу"), not a real change of speaker.
DEFAULT_MIN_TURN_WORDS = 2


@dataclass(frozen=True)
class SpeakerTurn:
    """One uninterrupted stretch of speech from a single speaker."""

    speaker: str
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }


def load_diarization(path: str | Path | None) -> list[dict[str, Any]]:
    """Read a diarization.json into sorted {start, end, speaker} entries."""

    if not path:
        return []
    diar_path = Path(path)
    if not diar_path.exists():
        return []
    try:
        raw = json.loads(diar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(raw, list):
        return []

    entries: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        speaker = str(item.get("speaker", "")).strip()
        if not speaker or end <= start:
            continue
        entries.append({"start": start, "end": end, "speaker": speaker})

    entries.sort(key=lambda e: (e["start"], e["end"]))
    return entries


def speaker_at(diarization: Sequence[Mapping[str, Any]], moment: float) -> str:
    """RU: Кто говорит в момент времени; при промахе — ближайший интервал.

    EN: Who is speaking at ``moment``; falls back to the nearest interval.

    Diarization leaves gaps around breaths and laughter, and a word landing in
    one must still be attributed rather than dropped.
    """
    if not diarization:
        return ""

    best_gap = float("inf")
    best_speaker = ""
    for entry in diarization:
        start, end = float(entry["start"]), float(entry["end"])
        if start <= moment <= end:
            return str(entry["speaker"])
        gap = start - moment if moment < start else moment - end
        if gap < best_gap:
            best_gap = gap
            best_speaker = str(entry["speaker"])
    return best_speaker


def _segment_words(segment: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_words = segment.get("words")
    if not isinstance(raw_words, list):
        return []
    words: list[dict[str, Any]] = []
    for raw in raw_words:
        if not isinstance(raw, Mapping):
            continue
        text = str(raw.get("word", "")).strip()
        raw_start, raw_end = raw.get("start"), raw.get("end")
        if raw_start is None or raw_end is None:
            continue
        try:
            start = float(raw_start)
            end = float(raw_end)
        except (TypeError, ValueError):
            continue
        if not text or end < start:
            continue
        words.append({"start": start, "end": end, "word": text})
    return words


#: How far a speaker boundary may be nudged to land on a sentence end.
_SNAP_WINDOW_WORDS = 4
_SENTENCE_END = (".", "!", "?", "…", ".»", "!»", "?»", '."', '!"', '?"')


def _snap_to_sentence(words: Sequence[str], index: int) -> int:
    """Move a split point onto the nearest sentence boundary, if one is close."""

    index = max(0, min(index, len(words)))
    if index in (0, len(words)):
        return index

    best = index
    best_distance = _SNAP_WINDOW_WORDS + 1
    for candidate in range(
        max(1, index - _SNAP_WINDOW_WORDS),
        min(len(words), index + _SNAP_WINDOW_WORDS) + 1,
    ):
        if not words[candidate - 1].endswith(_SENTENCE_END):
            continue
        distance = abs(candidate - index)
        if distance < best_distance:
            best, best_distance = candidate, distance
    return best


def _speaker_runs(
    words: Sequence[Mapping[str, Any]],
    diarization: Sequence[Mapping[str, Any]],
    *,
    min_turn_words: int,
) -> list[tuple[str, int, int]]:
    """Consecutive (speaker, first_word, last_word_exclusive) runs in a segment."""

    labels = [
        speaker_at(diarization, (float(w["start"]) + float(w["end"])) / 2.0)
        for w in words
    ]
    runs: list[list[Any]] = []
    for index, label in enumerate(labels):
        if runs and runs[-1][0] == label:
            runs[-1][2] = index + 1
        else:
            runs.append([label, index, index + 1])

    # Swallow one-word flickers ("да", "угу") between two runs of one speaker.
    merged: list[list[Any]] = []
    position = 0
    while position < len(runs):
        run = runs[position]
        nxt = runs[position + 1] if position + 1 < len(runs) else None
        if (
            merged
            and nxt is not None
            and run[2] - run[1] < min_turn_words
            and merged[-1][0] == nxt[0]
        ):
            merged[-1][2] = nxt[2]
            position += 2
            continue
        if merged and merged[-1][0] == run[0]:
            merged[-1][2] = run[2]
        else:
            merged.append(list(run))
        position += 1

    return [(str(r[0]), int(r[1]), int(r[2])) for r in merged]


def build_speaker_turns(
    segments: Sequence[Mapping[str, Any]],
    diarization: Sequence[Mapping[str, Any]],
    *,
    min_turn_words: int = DEFAULT_MIN_TURN_WORDS,
) -> list[SpeakerTurn]:
    """RU: Собирает реплики спикеров из сегментов и диаризации.

    EN: Build speaker turns out of transcript segments and diarization.

    Timings come from the word list, but the text comes from the segment's own
    ``text``: after proofreading those words still hold the raw ASR spelling,
    and rebuilding turns from them would quietly undo the proofreading stage.
    So a segment spanning several speakers is cut proportionally — the word
    counts barely move under proofreading, which is exactly what its own
    guardrail enforces.
    """
    if not diarization:
        return []

    turns: list[SpeakerTurn] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        try:
            seg_start = float(segment.get("start", 0.0))
            seg_end = float(segment.get("end", 0.0))
        except (TypeError, ValueError):
            continue

        words = _segment_words(segment)
        text_words = text.split()
        if not words or not text_words:
            turns.append(
                SpeakerTurn(
                    speaker=speaker_at(diarization, (seg_start + seg_end) / 2.0),
                    start=seg_start,
                    end=seg_end,
                    text=text,
                ),
            )
            continue

        runs = _speaker_runs(words, diarization, min_turn_words=min_turn_words)
        total_raw = len(words)
        total_text = len(text_words)
        # Proportional mapping lands mid-phrase, so nudge each boundary onto the
        # nearest sentence end: a turn that starts on "От | работников" reads as
        # a transcription glitch rather than as someone taking the floor.
        cuts = [
            _snap_to_sentence(text_words, round(run[1] * total_text / total_raw))
            for run in runs
        ]
        cuts.append(total_text)
        for index, (speaker, first, last) in enumerate(runs):
            text_from, text_to = cuts[index], cuts[index + 1]
            if text_to <= text_from:
                continue
            piece = " ".join(text_words[text_from:text_to]).strip()
            if not piece:
                continue
            turns.append(
                SpeakerTurn(
                    speaker=speaker,
                    start=float(words[first]["start"]),
                    end=float(words[last - 1]["end"]),
                    text=piece,
                ),
            )

    return _merge_consecutive(turns)


def _merge_consecutive(turns: Sequence[SpeakerTurn]) -> list[SpeakerTurn]:
    """Join neighbouring turns that belong to the same speaker."""

    out: list[SpeakerTurn] = []
    for turn in turns:
        if out and out[-1].speaker == turn.speaker:
            previous = out[-1]
            out[-1] = SpeakerTurn(
                speaker=previous.speaker,
                start=previous.start,
                end=turn.end,
                text=f"{previous.text} {turn.text}".strip(),
            )
        else:
            out.append(turn)
    return out


def _absorb_stray_groups(
    groups: list[list[tuple[str, str, float, float]]],
    *,
    min_turn_words: int,
) -> list[list[tuple[str, str, float, float]]]:
    """RU: Гасит однословные «перебивки» диаризации внутри чужой реплики.

    EN: Swallow one-word diarization flickers inside somebody else's turn.
    """
    if min_turn_words <= 1 or len(groups) < 3:
        return groups

    out: list[list[tuple[str, str, float, float]]] = [groups[0]]
    index = 1
    while index < len(groups):
        group = groups[index]
        neighbour = groups[index + 1] if index + 1 < len(groups) else None
        # A blip is short, and the same speaker resumes right after it.
        if (
            len(group) < min_turn_words
            and neighbour is not None
            and out[-1][0][0] == neighbour[0][0]
        ):
            out[-1] = out[-1] + group + neighbour
            index += 2
            continue
        if out and out[-1][0][0] == group[0][0]:
            out[-1] = out[-1] + group
        else:
            out.append(group)
        index += 1
    return out


def render_turns(
    turns: Sequence[SpeakerTurn],
    *,
    names: Mapping[str, str] | None = None,
) -> str:
    """RU: Текст реплик в виде «Имя: слова» — то, что видит модель.

    EN: Turns rendered as "Name: words" — what the model is shown.
    """
    mapping = dict(names or {})
    lines: list[str] = []
    for turn in turns:
        label = mapping.get(turn.speaker, turn.speaker)
        lines.append(f"{label}: {turn.text}")
    return "\n".join(lines)


def distinct_speakers(turns: Sequence[SpeakerTurn]) -> list[str]:
    """Speaker ids in order of first appearance."""

    seen: list[str] = []
    for turn in turns:
        if turn.speaker and turn.speaker not in seen:
            seen.append(turn.speaker)
    return seen
