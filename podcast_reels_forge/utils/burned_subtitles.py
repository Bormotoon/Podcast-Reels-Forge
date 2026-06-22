"""Helpers for rendering burned-in subtitles via .ass files."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from podcast_reels_forge.utils.reel_markdown import reel_index_from_path

LOG = logging.getLogger(__name__)

DEFAULT_SUBTITLE_FONT = Path("assets/fonts/bignoodletoooblique.ttf")
DEFAULT_SUBTITLE_CSS_TEMPLATE = Path("assets/subtitles/forge_subtitles.css")
DEFAULT_FONT_SIZE_PX = 36
DEFAULT_MAX_LINES = 2
DEFAULT_MAX_WIDTH_RATIO = 0.65
DEFAULT_WRAP_WORDS = True
DEFAULT_VERTICAL_ALIGN = "bottom"
DEFAULT_VERTICAL_OFFSET = 0.0
DEFAULT_WORD_X_SPACE = 6
DEFAULT_WORD_Y_SPACE = 8
DEFAULT_TEXT_OVERFLOW_STRATEGY = "exceed_width"
DEFAULT_AVOID_ENDING_WITH_SHORT_WORD_CHARS = 2

# BBC/Netflix subtitle guidelines (https://www.bbc.co.uk/accessibility/forproducts/guides/subtitles):
# - 25 chars/line for 9:16 vertical (equivalent to 37 chars in 75% 16:9)
# - Max 2 lines (landscape) / 3 lines (vertical 9:16)
# - 160-180 WPM → 0.33-0.375s per word
# - Min 0.3s per word → 4 words = 1.2s minimum
# - Min 1s gap between subtitles (preferably 1.5s)
# - Max 1.5s anticipation or lag behind speech
DEFAULT_CHARS_PER_LINE = 25
DEFAULT_MIN_DURATION_S = 1.5
DEFAULT_MAX_DURATION_S = 7.0
DEFAULT_GAP_BETWEEN_SUBTITLES_S = 0.15


@dataclass(frozen=True)
class SubtitleSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class _TimedSubtitleWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SubtitleRenderSettings:
    enabled: bool
    font_path: Path
    ass_style: Path | None = None
    font_size_px: int = DEFAULT_FONT_SIZE_PX
    max_lines: int = DEFAULT_MAX_LINES
    max_width_ratio: float = DEFAULT_MAX_WIDTH_RATIO
    wrap_words: bool = DEFAULT_WRAP_WORDS
    vertical_align: str = DEFAULT_VERTICAL_ALIGN
    vertical_offset: float = DEFAULT_VERTICAL_OFFSET
    word_x_space: int = DEFAULT_WORD_X_SPACE
    word_y_space: int = DEFAULT_WORD_Y_SPACE
    fade_in_duration: float = 0.18
    fade_out_duration: float = 0.12


def subtitle_settings_from_conf(
    conf: Mapping[str, Any] | None,
    *,
    repo_dir: Path,
) -> SubtitleRenderSettings:
    subtitles_conf = conf.get("subtitles", {}) if isinstance(conf, Mapping) else {}
    if not isinstance(subtitles_conf, Mapping):
        subtitles_conf = {}
    enabled = bool(subtitles_conf.get("enabled", True))
    font_value = subtitles_conf.get("font") or subtitles_conf.get("font_path")
    ass_style_value = subtitles_conf.get("ass_style")
    font_path = _resolve_config_path(
        font_value,
        repo_dir=repo_dir,
        default=DEFAULT_SUBTITLE_FONT,
    )
    ass_style = _resolve_config_path(
        ass_style_value,
        repo_dir=repo_dir,
        default=Path("assets/subtitles/forge_subtitles.ass"),
    ) if ass_style_value else None

    return SubtitleRenderSettings(
        enabled=enabled,
        font_path=font_path,
        ass_style=ass_style,
        font_size_px=_coerce_int(
            subtitles_conf.get("font_size_px"),
            default=DEFAULT_FONT_SIZE_PX,
            minimum=16,
        ),
        max_lines=_coerce_int(
            subtitles_conf.get("max_lines"),
            default=DEFAULT_MAX_LINES,
            minimum=1,
        ),
        max_width_ratio=_coerce_float(
            subtitles_conf.get("max_width_ratio"),
            default=DEFAULT_MAX_WIDTH_RATIO,
            minimum=0.1,
            maximum=1.0,
        ),
        wrap_words=_coerce_bool(
            subtitles_conf.get("wrap_words"),
            default=DEFAULT_WRAP_WORDS,
        ),
        vertical_align=_coerce_align(
            subtitles_conf.get("vertical_align"),
            default=DEFAULT_VERTICAL_ALIGN,
        ),
        vertical_offset=_coerce_float(
            subtitles_conf.get("vertical_offset"),
            default=DEFAULT_VERTICAL_OFFSET,
            minimum=-1.0,
            maximum=1.0,
        ),
        word_x_space=_coerce_int(
            subtitles_conf.get("word_x_space"),
            default=DEFAULT_WORD_X_SPACE,
            minimum=0,
        ),
        word_y_space=_coerce_int(
            subtitles_conf.get("word_y_space"),
            default=DEFAULT_WORD_Y_SPACE,
            minimum=0,
        ),
        fade_in_duration=_coerce_float(
            subtitles_conf.get("fade_in_duration"),
            default=0.18,
            minimum=0.01,
            maximum=1.0,
        ),
        fade_out_duration=_coerce_float(
            subtitles_conf.get("fade_out_duration"),
            default=0.12,
            minimum=0.01,
            maximum=1.0,
        ),
    )


def ensure_reel_burned_subtitles(
    moment: Mapping[str, Any],
    reel_path: Path,
    *,
    transcript_json_path: Path,
    padding: float,
    settings: SubtitleRenderSettings,
    verbose: bool = False,
) -> Path | None:
    if not settings.enabled:
        return None
    if not reel_path.exists():
        raise FileNotFoundError(f"Reel file not found: {reel_path}")
    if not transcript_json_path.exists():
        raise FileNotFoundError(f"Transcript JSON not found: {transcript_json_path}")
    if not settings.font_path.exists():
        raise FileNotFoundError(f"Subtitle font not found: {settings.font_path}")

    start = _coerce_float(moment.get("start"), default=0.0)
    end = _coerce_float(moment.get("end"), default=0.0)
    if end <= start:
        raise ValueError(f"Invalid reel boundaries for {reel_path.name}: start={start} end={end}")

    transcript_segments = load_transcript_segments(transcript_json_path)
    return _render_reel_with_subtitles_assets(
        moment=moment,
        reel_path=reel_path,
        transcript_segments=transcript_segments,
        padding=padding,
        settings=settings,
        template_dir=reel_path.parent,
        verbose=verbose,
    )


def sync_reel_burned_subtitles(
    moments: Sequence[Mapping[str, Any]],
    reels_root: Path,
    *,
    transcript_json_path: Path,
    padding: float,
    settings: SubtitleRenderSettings,
    verbose: bool = False,
) -> list[Path]:
    written: list[Path] = []
    if not settings.enabled or not reels_root.exists():
        return written

    import re as _re
    reel_files = sorted(
        p for p in reels_root.rglob("reel_*.mp4")
        if p.is_file() and _re.match(r"^reel_\d+\.mp4$", p.name)
    )
    if not reel_files:
        return written

    transcript_segments = load_transcript_segments(transcript_json_path)

    for reel_path in reel_files:
        index = reel_index_from_path(reel_path)
        if index is None:
            continue
        moment_index = index - 1
        if moment_index < 0 or moment_index >= len(moments):
            continue
        srt_path = _render_reel_with_subtitles_assets(
            moment=moments[moment_index],
            reel_path=reel_path,
            transcript_segments=transcript_segments,
            padding=padding,
            settings=settings,
            template_dir=reel_path.parent,
            verbose=verbose,
        )
        if srt_path is not None:
            written.append(srt_path)
    return written


def _render_reel_with_subtitles_assets(
    *,
    moment: Mapping[str, Any],
    reel_path: Path,
    transcript_segments: Sequence[SubtitleSegment],
    padding: float,
    settings: SubtitleRenderSettings,
    template_dir: Path,
    verbose: bool,
) -> Path | None:
    start = _coerce_float(moment.get("start"), default=0.0)
    end = _coerce_float(moment.get("end"), default=0.0)
    if end <= start:
        raise ValueError(f"Invalid boundaries: {start} - {end}")

    clip_segments = slice_segments_for_clip(
        transcript_segments,
        clip_start=max(0.0, start - float(padding)),
        clip_end=end + float(padding),
    )
    clip_segments = _prepare_subtitle_segments(clip_segments, settings=settings)

    if not clip_segments:
        return None

    # Write fallback .srt
    srt_path = reel_path.with_suffix(".srt")
    write_srt_file(srt_path, clip_segments)

    # Write .ass subtitles
    ass_path = reel_path.with_suffix(".ass")
    _write_ass_file(ass_path, clip_segments, settings)

    return ass_path


def _fmt_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    cs = int((s % 1) * 100)
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"


def _write_ass_file(path: Path, segments: Sequence[SubtitleSegment], settings: SubtitleRenderSettings) -> None:
    # Try to load ASS style from the style-editor output file
    ass_style_path = _find_ass_style_file()
    if ass_style_path and ass_style_path.exists():
        header = ass_style_path.read_text(encoding="utf-8").rstrip()
    else:
        font_name = _font_name_from_path(settings.font_path)
        header = _default_ass_header(font_name, settings.font_size_px)

    # Events section
    events = "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

    dialogue_lines: list[str] = []
    for seg in segments:
        words = _build_timed_words(seg)
        parts: list[str] = []
        for w in words:
            dur_cs = int((w.end - w.start) * 100)
            parts.append(f"{{\\kf{dur_cs}}}{w.text}")

        dialogue_text = " ".join(parts)
        dialogue_lines.append(
            f"Dialogue: 0,{_fmt_ass_time(seg.start)},{_fmt_ass_time(seg.end)},Default,,0,0,0,,{dialogue_text}"
        )

    path.write_text(header + events + "\n".join(dialogue_lines) + "\n", encoding="utf-8")


def _find_ass_style_file() -> Path | None:
    """Look for the style-editor output file (forge_subtitles.ass) in known locations."""
    base = Path(__file__).resolve().parents[2]
    candidates = [
        base / "assets" / "subtitles" / "forge_subtitles.ass",
        base / DEFAULT_SUBTITLE_CSS_TEMPLATE.with_suffix(".ass"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _font_name_from_path(font_path: Path) -> str:
    """Extract a readable font name from the font file path."""
    stem = font_path.stem
    # Convert file name to a more readable font name
    name = stem.replace("_", " ").replace("-", " ")
    # Capitalize each word
    return " ".join(word.capitalize() for word in name.split())


def _default_ass_header(font_name: str, font_size: int) -> str:
    """Generate a default ASS header with sensible defaults."""
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,4,3,2,40,40,150,1"""


def load_transcript_segments(path: Path) -> list[SubtitleSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []

    raw_sentences = data.get("sentences", [])
    if isinstance(raw_sentences, list) and raw_sentences:
        sentence_segments: list[SubtitleSegment] = []
        for raw in raw_sentences:
            if not isinstance(raw, Mapping):
                continue
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            start = _coerce_float(raw.get("start"), default=0.0)
            end = _coerce_float(raw.get("end"), default=0.0)
            if end <= start:
                continue
            sentence_segments.append(SubtitleSegment(start=start, end=end, text=text))
        if sentence_segments:
            return sentence_segments

    raw_segments = data.get("segments", [])
    segments: list[SubtitleSegment] = []
    for raw in raw_segments:
        if not isinstance(raw, Mapping):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        start = _coerce_float(raw.get("start"), default=0.0)
        end = _coerce_float(raw.get("end"), default=0.0)
        if end <= start:
            continue
        segments.append(SubtitleSegment(start=start, end=end, text=text))
    return segments


def slice_segments_for_clip(
    transcript_segments: Sequence[SubtitleSegment],
    *,
    clip_start: float,
    clip_end: float,
) -> list[SubtitleSegment]:
    out: list[SubtitleSegment] = []
    if clip_end <= clip_start:
        return out

    for seg in transcript_segments:
        overlap_start = max(float(clip_start), seg.start)
        overlap_end = min(float(clip_end), seg.end)
        if overlap_end <= overlap_start:
            continue
        shifted_start = max(0.0, overlap_start - clip_start)
        shifted_end = max(0.0, overlap_end - clip_start)
        if shifted_end - shifted_start < 0.05:
            continue
        out.append(
            SubtitleSegment(
                start=round(shifted_start, 3),
                end=round(shifted_end, 3),
                text=seg.text,
            ),
        )
    return out


def write_srt_file(path: Path, segments: Sequence[SubtitleSegment]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, seg in enumerate(segments, 1):
        lines.append(str(index))
        lines.append(f"{_format_srt_timestamp(seg.start)} --> {_format_srt_timestamp(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000.0)))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    secs = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _prepare_subtitle_segments(
    segments: Sequence[SubtitleSegment],
    *,
    settings: SubtitleRenderSettings,
) -> list[SubtitleSegment]:
    """Prepare subtitle segments for rendering following BBC/Netflix guidelines.

    Pipeline:
    1. Merge consecutive short segments into readable blocks
    2. Split blocks that exceed max_chars per line
    3. Remove overlapping blocks
    4. Enforce min/max duration and gap between subtitles
    5. Remove overlaps again (enforce may have created new ones)
    """
    if not segments:
        return []

    lines = settings.max_lines if settings.wrap_words else 1
    max_chars_per_line = DEFAULT_CHARS_PER_LINE
    max_chars = max(12, lines * max_chars_per_line)
    min_chars = max(8, max_chars // 3)

    # Step 1: Merge consecutive short segments into blocks
    merged = _merge_consecutive_segments(
        list(segments),
        max_duration=DEFAULT_MAX_DURATION_S,
        max_chars=max_chars,
    )

    # Step 2: Split oversized blocks (respecting sentence boundaries)
    prepared: list[SubtitleSegment] = []
    for seg in merged:
        prepared.extend(
            _split_long_segment(
                seg,
                max_chars=max_chars,
                min_chars=min_chars,
            ),
        )

    # Step 3: Remove any overlapping blocks
    prepared = _remove_overlaps(prepared)

    # Step 4: Enforce minimum duration and gap (after overlap removal)
    prepared = _enforce_timing_constraints(prepared)

    # Step 5: Remove overlaps again (enforce may have created new ones)
    prepared = _remove_overlaps(prepared)

    return prepared


def _merge_consecutive_segments(
    segments: list[SubtitleSegment],
    *,
    max_duration: float,
    max_chars: int,
) -> list[SubtitleSegment]:
    """Merge consecutive short segments into single subtitle blocks."""
    if not segments:
        return []

    merged: list[SubtitleSegment] = []
    current_text_parts: list[str] = []
    current_start: float | None = None
    current_end: float = 0.0

    def flush() -> None:
        if current_text_parts and current_start is not None:
            merged.append(
                SubtitleSegment(
                    start=round(current_start, 3),
                    end=round(current_end, 3),
                    text=" ".join(current_text_parts),
                )
            )

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        combined_text = " ".join(current_text_parts + [text]) if current_text_parts else text
        combined_duration = seg.end - (current_start if current_start is not None else seg.start)
        has_sentence_end = bool(_SENTENCE_END_RE.search(text))

        should_flush = False
        if current_start is not None:
            gap = seg.start - current_end
            if gap > 0.5:
                should_flush = True
            elif combined_duration > max_duration:
                should_flush = True
            elif len(combined_text) > max_chars:
                should_flush = True
            elif has_sentence_end and len(combined_text) >= max_chars // 2:
                should_flush = True

        if should_flush:
            flush()
            current_text_parts = []
            current_start = None

        if current_start is None:
            current_start = seg.start
        current_text_parts.append(text)
        current_end = seg.end

    flush()
    return merged


def _remove_overlaps(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
    """Remove overlapping blocks by pushing later blocks forward."""
    if not segments:
        return []

    result: list[SubtitleSegment] = []
    for seg in segments:
        if not result:
            result.append(seg)
            continue
        prev = result[-1]
        if seg.start < prev.end:
            new_start = prev.end + 0.05
            duration = seg.end - seg.start
            seg = SubtitleSegment(
                start=round(new_start, 3),
                end=round(new_start + duration, 3),
                text=seg.text,
            )
        result.append(seg)
    return result


def _enforce_timing_constraints(
    segments: list[SubtitleSegment],
) -> list[SubtitleSegment]:
    """Enforce minimum duration and gap between subtitle blocks."""
    if not segments:
        return []

    result: list[SubtitleSegment] = []

    for seg in segments:
        duration = seg.end - seg.start

        if duration < DEFAULT_MIN_DURATION_S:
            seg = SubtitleSegment(
                start=seg.start,
                end=round(seg.start + DEFAULT_MIN_DURATION_S, 3),
                text=seg.text,
            )

        if duration > DEFAULT_MAX_DURATION_S:
            seg = SubtitleSegment(
                start=seg.start,
                end=round(seg.start + DEFAULT_MAX_DURATION_S, 3),
                text=seg.text,
            )

        if result:
            prev = result[-1]
            gap = seg.start - prev.end
            if gap < DEFAULT_GAP_BETWEEN_SUBTITLES_S:
                new_prev_end = seg.start - DEFAULT_GAP_BETWEEN_SUBTITLES_S
                if new_prev_end > prev.start and (new_prev_end - prev.start) >= DEFAULT_MIN_DURATION_S:
                    result[-1] = SubtitleSegment(
                        start=prev.start,
                        end=round(new_prev_end, 3),
                        text=prev.text,
                    )

        result.append(seg)

    return result


import re as _re

_SENTENCE_END_RE = _re.compile(r"[.!?…]\s*$")

_NO_LINE_END = frozenset({
    "в", "на", "по", "из", "за", "к", "у", "о", "об", "от", "до", "со", "ко",
    "а", "и", "но", "ни", "да", "нет", "не", "то", "ли", "бы", "же", "вот",
    "ну", "или", "что", "как", "где", "когда", "чтобы", "пока", "тоже", "уже",
    "ещё", "еще", "просто", "только", "ведь", "если", "либо", "однако", "потом",
    "тогда", "сейчас", "потому", "раз", "хотя", "чтоб", "будто", "даже",
    "вообще", "именно", "конечно", "пожалуй", "пожалуйста",
    "сразу", "типа", "кроме", "после", "перед", "между", "через", "около",
})

_COMMA_AFTER = _re.compile(r"[,;:—–]\s*$")


def _split_long_segment(
    segment: SubtitleSegment,
    *,
    max_chars: int,
    min_chars: int,
) -> list[SubtitleSegment]:
    """Split a segment that exceeds max_chars into readable sub-segments."""
    text = segment.text.strip()
    if len(text) <= max_chars:
        return [segment]

    words = _build_timed_words(segment)
    if len(words) <= 1:
        return [segment]

    chunks: list[SubtitleSegment] = []
    word_index = 0

    while word_index < len(words):
        chunk_end = _find_split_point(
            words,
            start_index=word_index,
            max_chars=max_chars,
            min_chars=min_chars,
        )
        chunk_words = words[word_index:chunk_end]
        if not chunk_words:
            break
        chunks.append(
            SubtitleSegment(
                start=round(chunk_words[0].start, 3),
                end=round(chunk_words[-1].end, 3),
                text=" ".join(w.text for w in chunk_words).strip(),
            )
        )
        word_index = chunk_end

    return chunks or [segment]


def _find_split_point(
    words: Sequence[_TimedSubtitleWord],
    *,
    start_index: int,
    max_chars: int,
    min_chars: int,
) -> int:
    """Find the best word index to split a subtitle block."""
    current_index = start_index
    chars_count = 0
    total_words = len(words)

    best_sentence_end: int | None = None
    best_comma_end: int | None = None
    best_before_preposition: int | None = None
    best_normal: int | None = None

    while current_index < total_words:
        word_text = words[current_index].text
        word_len = len(word_text)

        if chars_count + word_len > max_chars:
            break

        chars_count += word_len
        current_index += 1

        if current_index < total_words:
            chars_count += 1

        words_so_far = current_index - start_index
        if words_so_far < 2:
            continue

        if _SENTENCE_END_RE.search(word_text):
            best_sentence_end = current_index

        if _COMMA_AFTER.search(word_text):
            best_comma_end = current_index

        if current_index < total_words:
            next_lower = words[current_index].text.lower().strip(".,!?…;:")
            if next_lower in _NO_LINE_END:
                best_before_preposition = current_index

        if current_index < total_words:
            this_lower = word_text.lower().strip(".,!?…;:")
            remaining_chars = sum(len(w.text) for w in words[current_index:])
            if this_lower not in _NO_LINE_END:
                best_normal = current_index
            elif remaining_chars > min_chars:
                best_normal = current_index

    if best_sentence_end is not None:
        return best_sentence_end
    if best_comma_end is not None:
        return best_comma_end
    if best_before_preposition is not None:
        return best_before_preposition
    if best_normal is not None:
        return best_normal

    if current_index == start_index:
        current_index += 1

    remaining = sum(len(w.text) for w in words[current_index:])
    if 0 < remaining < min_chars:
        return total_words

    return current_index


def _build_timed_words(segment: SubtitleSegment) -> list[_TimedSubtitleWord]:
    words_text = [word for word in segment.text.split() if word.strip()]
    if not words_text:
        return []

    start = float(segment.start)
    end = max(start + 0.01, float(segment.end))
    duration = end - start
    weights = [max(len(word), 1) for word in words_text]
    total_weight = sum(weights)
    consumed_weight = 0
    timed_words: list[_TimedSubtitleWord] = []

    for index, (word_text, weight) in enumerate(zip(words_text, weights)):
        word_start = start + duration * (consumed_weight / total_weight)
        consumed_weight += weight
        word_end = (
            end
            if index == len(words_text) - 1
            else start + duration * (consumed_weight / total_weight)
        )
        timed_words.append(
            _TimedSubtitleWord(
                start=round(word_start, 3),
                end=round(max(word_start + 0.01, word_end), 3),
                text=word_text,
            ),
        )
    return timed_words


def _resolve_config_path(
    value: object,
    *,
    repo_dir: Path,
    default: Path,
) -> Path:
    path = Path(str(value)).expanduser() if value else default
    if not path.is_absolute():
        path = repo_dir / path
    return path.resolve()


def _coerce_float(
    value: object,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        out = float(default)
    if minimum is not None:
        out = max(out, minimum)
    if maximum is not None:
        out = min(out, maximum)
    return out


def _coerce_int(value: object, *, default: int, minimum: int = 1) -> int:
    try:
        out = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        out = int(default)
    return max(out, minimum)


def _coerce_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_align(value: object, *, default: str) -> str:
    allowed = {"top", "center", "bottom"}
    align = str(value or default).strip().lower()
    return align if align in allowed else default
