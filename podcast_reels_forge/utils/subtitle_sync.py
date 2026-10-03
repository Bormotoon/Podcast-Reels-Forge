"""RU: Проверка и подгонка таймингов субтитров вторым проходом Whisper.

Тайминги слов из транскрипта эпизода — результат одного прохода по часовому
файлу. Перед вжиганием каждый клип слушается заново: его собственный
интервал вырезается из исходного аудио и распознаётся с таймингами слов.
Слова транскрипта сопоставляются с услышанными; если они заметно
расходятся, субтитры получают тайминги второго прохода, а несопоставленные
слова раскладываются между соседними опорными. Если сопоставилось слишком
мало, второму проходу не верим и оставляем исходные тайминги.

EN: Check and fix subtitle timing with a second Whisper pass.

Transcript word timings come from one pass over the whole episode. Before
burning, every clip is listened to again: its own interval is cut from the
source audio and recognized with word timestamps. Transcript words are matched
to the heard ones; when they drift apart noticeably the subtitles take the
second pass's timings, and unmatched words are laid out between the nearest
matched anchors. When too little matches, the second pass is not trusted and
the original timings stay.
"""

from __future__ import annotations

import bisect
import difflib
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from podcast_reels_forge.utils.ffmpeg import ffmpeg_bin

LOG = logging.getLogger("forge")

SAMPLE_RATE = 16000
DEFAULT_MODEL = "large-v3"
# Below this share of transcript words found in the second pass, the pass is
# not trusted (music, crosstalk, a different language) and nothing changes.
DEFAULT_MIN_MATCH_RATIO = 0.5
# Timings are replaced only when the drift is worth fixing: p95 of the
# per-word end difference at or above this.
DEFAULT_APPLY_THRESHOLD_S = 0.2
# Two differently spelled words still count as the same word above this
# similarity ("простится" / "упростится").
_FUZZY_WORD_RATIO = 0.6


# Whisper often lets the word after a pause absorb the pause: "Знаете" timed
# 5.48-8.12 when it was said in the last ~0.7 s of that. The end is reliable,
# the start is not, and for subtitles the start is what shows the cue early.
_WORD_BASE_S = 0.2
_WORD_PER_CHAR_S = 0.11


def plausible_start(start: float, end: float, text: str) -> float:
    """``start`` moved up to the latest point the word could plausibly begin."""

    letters = sum(1 for char in text if char.isalnum())
    longest = _WORD_BASE_S + _WORD_PER_CHAR_S * max(1, letters)
    return max(start, end - longest)


@dataclass(frozen=True)
class TimedWord:
    start: float
    end: float
    text: str


@dataclass
class ClipSyncReport:
    """What the second pass found for one clip."""

    clip_start: float
    clip_end: float
    reference_words: int = 0
    heard_words: int = 0
    matched_words: int = 0
    match_ratio: float = 0.0
    median_shift_s: float = 0.0
    mean_abs_shift_s: float = 0.0
    p95_abs_shift_s: float = 0.0
    max_abs_shift_s: float = 0.0
    applied: bool = False
    verdict: str = ""
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("extra")
        data.update(self.extra)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in data.items()}


def normalize_word(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum()).replace("ё", "е")


def extract_audio(source: Path, start: float, end: float) -> Any:
    """The [start, end] span of ``source`` as 16 kHz mono float32 samples."""

    import numpy as np

    cmd = [
        ffmpeg_bin(), "-v", "error", "-nostdin",
        "-ss", f"{max(0.0, start):.3f}", "-to", f"{max(start, end):.3f}",
        "-i", str(source),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-300:] or "ffmpeg failed")
    return np.frombuffer(result.stdout, dtype=np.float32)


def recognize_words(model: Any, audio: Any, *, offset: float, language: str | None) -> list[TimedWord]:
    """Words Whisper hears in ``audio``, on the episode's timeline."""

    segments, _info = model.transcribe(
        audio,
        language=language or None,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
        beam_size=5,
    )
    words: list[TimedWord] = []
    for segment in segments:
        for word in segment.words or ():
            text = str(word.word).strip()
            if text and word.end > word.start:
                start = plausible_start(float(word.start), float(word.end), text)
                words.append(TimedWord(offset + start, offset + float(word.end), text))
    return words


def match_words(
    reference: Sequence[TimedWord],
    heard: Sequence[TimedWord],
) -> dict[int, int]:
    """Map reference word index -> heard word index, in order.

    Exact matches of normalized tokens first; inside a replaced stretch of
    equal length, words are paired one to one when they are spelled alike.
    """

    ref_norm = [normalize_word(w.text) for w in reference]
    heard_norm = [normalize_word(w.text) for w in heard]
    pairs: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(None, ref_norm, heard_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                pairs[i1 + offset] = j1 + offset
        elif tag == "replace" and i2 - i1 == j2 - j1:
            for offset in range(i2 - i1):
                a, b = ref_norm[i1 + offset], heard_norm[j1 + offset]
                if a and b and difflib.SequenceMatcher(None, a, b).ratio() >= _FUZZY_WORD_RATIO:
                    pairs[i1 + offset] = j1 + offset
    return pairs


def retime(
    reference: Sequence[TimedWord],
    heard: Sequence[TimedWord],
    pairs: dict[int, int],
) -> list[TimedWord]:
    """Reference words with the heard timings.

    Matched words take the heard span. An unmatched word keeps its own
    duration and its place relative to the speech around it: it moves by a
    shift interpolated between the shifts of the matched words on either side
    (before the first / after the last matched word, by that word's shift).
    Spreading unmatched words over the gap between anchors used to put a word
    into the pause before it — and its cue on screen ahead of the speech.
    """

    if not pairs:
        return list(reference)
    anchors = sorted(pairs)
    shift_at = {i: heard[pairs[i]].start - reference[i].start for i in anchors}

    def _shift(i: int) -> float:
        if i <= anchors[0]:
            return shift_at[anchors[0]]
        if i >= anchors[-1]:
            return shift_at[anchors[-1]]
        position = bisect.bisect_left(anchors, i)
        left, right = anchors[position - 1], anchors[position]
        t0, t1 = reference[left].start, reference[right].start
        weight = (reference[i].start - t0) / (t1 - t0) if t1 > t0 else 0.5
        return shift_at[left] + (shift_at[right] - shift_at[left]) * weight

    out: list[TimedWord] = []
    previous_start = 0.0
    for i, word in enumerate(reference):
        if i in pairs:
            start, end = heard[pairs[i]].start, heard[pairs[i]].end
        else:
            delta = _shift(i)
            start, end = word.start + delta, word.end + delta
        start = max(start, previous_start)
        end = max(end, start + 0.01)
        out.append(TimedWord(round(start, 3), round(end, 3), word.text))
        previous_start = start
    return out


def _percentile(values: Sequence[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))]


def sync_clip(
    model: Any,
    source_audio: Path,
    reference: Sequence[TimedWord],
    *,
    clip_start: float,
    clip_end: float,
    language: str | None,
    min_match_ratio: float = DEFAULT_MIN_MATCH_RATIO,
    apply_threshold_s: float = DEFAULT_APPLY_THRESHOLD_S,
) -> tuple[list[TimedWord], ClipSyncReport]:
    """Check one clip's word timings; return the words to use and a report."""

    report = ClipSyncReport(clip_start=clip_start, clip_end=clip_end, reference_words=len(reference))
    if not reference:
        report.verdict = "no_words"
        return list(reference), report
    try:
        audio = extract_audio(source_audio, clip_start, clip_end)
        heard = recognize_words(model, audio, offset=clip_start, language=language)
    except Exception as exc:  # noqa: BLE001 - a failed check must not fail the cut
        report.verdict = "error"
        report.error = str(exc)[:300]
        return list(reference), report

    report.heard_words = len(heard)
    pairs = match_words(reference, heard)
    report.matched_words = len(pairs)
    report.match_ratio = len(pairs) / len(reference)
    # Drift is measured on word ends: they are what both passes agree on when
    # the timing is right (starts carry the pause-absorption noise above).
    shifts = [heard[j].end - reference[i].end for i, j in pairs.items()]
    if shifts:
        abs_shifts = [abs(s) for s in shifts]
        report.median_shift_s = _percentile(shifts, 0.5)
        report.mean_abs_shift_s = sum(abs_shifts) / len(abs_shifts)
        report.p95_abs_shift_s = _percentile(abs_shifts, 0.95)
        report.max_abs_shift_s = max(abs_shifts)

    if report.match_ratio < min_match_ratio:
        report.verdict = "unreliable"
        return list(reference), report
    if report.p95_abs_shift_s < apply_threshold_s:
        report.verdict = "in_sync"
        return list(reference), report
    report.verdict = "retimed"
    report.applied = True
    return retime(reference, heard, pairs), report


def load_model(model_name: str, *, device: str = "auto", compute_type: str = "auto") -> Any:
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:  # noqa: BLE001 - e.g. CUDA out of memory: fall back
        if device == "cpu":
            raise
        LOG.warning("subtitle sync: %s on %s failed (%s); using CPU", model_name, device, exc)
        return WhisperModel(model_name, device="cpu", compute_type="int8")
