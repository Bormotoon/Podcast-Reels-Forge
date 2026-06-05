"""Helpers for rendering burned-in subtitles with pycaps."""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from podcast_reels_forge.utils.reel_markdown import reel_index_from_path

LOG = logging.getLogger(__name__)

DEFAULT_SUBTITLE_FONT = Path("assets/fonts/bignoodletoooblique.ttf")
DEFAULT_SUBTITLE_CSS_TEMPLATE = Path("assets/subtitles/forge_subtitles.css")
DEFAULT_FONT_SIZE_PX = 44
DEFAULT_MAX_LINES = 2
DEFAULT_MAX_WIDTH_RATIO = 0.9
DEFAULT_WRAP_WORDS = True
DEFAULT_VERTICAL_ALIGN = "bottom"
DEFAULT_VERTICAL_OFFSET = 0.0
DEFAULT_WORD_X_SPACE = 6
DEFAULT_WORD_Y_SPACE = 8
DEFAULT_TEXT_OVERFLOW_STRATEGY = "exceed_width"
DEFAULT_SEGMENT_CHARS_PER_LINE = 14
DEFAULT_SEGMENT_MIN_CHARS = 8
DEFAULT_AVOID_ENDING_WITH_SHORT_WORD_CHARS = 2
_PYCAPS_TEMPLATE_DIRNAME = ".pycaps_template"


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
    css_path: Path = DEFAULT_SUBTITLE_CSS_TEMPLATE
    font_size_px: int = DEFAULT_FONT_SIZE_PX
    max_lines: int = DEFAULT_MAX_LINES
    max_width_ratio: float = DEFAULT_MAX_WIDTH_RATIO
    wrap_words: bool = DEFAULT_WRAP_WORDS
    vertical_align: str = DEFAULT_VERTICAL_ALIGN
    vertical_offset: float = DEFAULT_VERTICAL_OFFSET


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
    css_value = subtitles_conf.get("css") or subtitles_conf.get("css_path")
    font_path = _resolve_config_path(
        font_value,
        repo_dir=repo_dir,
        default=DEFAULT_SUBTITLE_FONT,
    )
    css_path = _resolve_config_path(
        css_value,
        repo_dir=repo_dir,
        default=DEFAULT_SUBTITLE_CSS_TEMPLATE,
    )

    return SubtitleRenderSettings(
        enabled=enabled,
        font_path=font_path,
        css_path=css_path,
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
    if not settings.css_path.exists():
        raise FileNotFoundError(f"Subtitle CSS template not found: {settings.css_path}")

    start = _coerce_float(moment.get("start"), default=0.0)
    end = _coerce_float(moment.get("end"), default=0.0)
    if end <= start:
        raise ValueError(f"Invalid reel boundaries for {reel_path.name}: start={start} end={end}")

    transcript_segments = load_transcript_segments(transcript_json_path)
    template_dir = prepare_pycaps_template(
        reel_path.parent / _PYCAPS_TEMPLATE_DIRNAME,
        settings=settings,
    )
    return _render_reel_with_subtitles_assets(
        moment=moment,
        reel_path=reel_path,
        transcript_segments=transcript_segments,
        padding=padding,
        settings=settings,
        template_dir=template_dir,
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

    reel_files = sorted(
        p for p in reels_root.rglob("reel_*.mp4") if p.is_file()
    )
    if not reel_files:
        return written

    transcript_segments = load_transcript_segments(transcript_json_path)
    template_dir = prepare_pycaps_template(
        reels_root / _PYCAPS_TEMPLATE_DIRNAME,
        settings=settings,
    )

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
            template_dir=template_dir,
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
        raise ValueError(f"Invalid reel boundaries for {reel_path.name}: start={start} end={end}")

    clip_segments = slice_segments_for_clip(
        transcript_segments,
        clip_start=max(0.0, start - float(padding)),
        clip_end=end + float(padding),
    )
    clip_segments = _prepare_subtitle_segments(clip_segments, settings=settings)
    if not clip_segments:
        LOG.warning(
            "No transcript segments overlap %s; leaving video without burned subtitles",
            reel_path.name,
        )
        return None

    srt_path = reel_path.with_suffix(".srt")
    write_srt_file(srt_path, clip_segments)

    tmp_output = reel_path.with_name(f"{reel_path.stem}.subtitled{reel_path.suffix}")
    if tmp_output.exists():
        tmp_output.unlink(missing_ok=True)

    _render_reel_with_pycaps(
        template_dir=template_dir,
        reel_path=reel_path,
        tmp_output=tmp_output,
        clip_segments=clip_segments,
        settings=settings,
        verbose=verbose,
    )
    tmp_output.replace(reel_path)
    return srt_path


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


def prepare_pycaps_template(
    template_dir: Path,
    *,
    settings: SubtitleRenderSettings,
) -> Path:
    resources_dir = template_dir / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)

    font_suffix = settings.font_path.suffix.lower() or ".ttf"
    font_name = f"subtitle_font{font_suffix}"
    shutil.copyfile(settings.font_path, resources_dir / font_name)

    (template_dir / "styles.css").write_text(
        _render_styles_css_template(
            settings.css_path,
            font_filename=font_name,
            font_size_px=settings.font_size_px,
            font_format=_font_format_for_suffix(font_suffix),
        ),
        encoding="utf-8",
    )
    (template_dir / "pycaps.template.json").write_text(
        json.dumps(
            _build_template_config(settings),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return template_dir


def _build_template_config(settings: SubtitleRenderSettings) -> dict[str, Any]:
    max_number_of_lines = settings.max_lines if settings.wrap_words else 1
    return {
        "css": "styles.css",
        "resources": "resources",
        "layout": {
            "x_words_space": DEFAULT_WORD_X_SPACE,
            "y_words_space": DEFAULT_WORD_Y_SPACE,
            "max_width_ratio": settings.max_width_ratio,
            "max_number_of_lines": max_number_of_lines,
            "min_number_of_lines": 1,
            "on_text_overflow_strategy": DEFAULT_TEXT_OVERFLOW_STRATEGY,
            "vertical_align": {
                "align": settings.vertical_align,
                "offset": settings.vertical_offset,
            },
        },
        "animations": [
            {
                "type": "fade_in",
                "when": "narration-starts",
                "what": "segment",
                "duration": 0.18,
            },
            {
                "type": "fade_out",
                "when": "narration-ends",
                "what": "segment",
                "duration": 0.12,
            },
        ],
    }


def _render_styles_css_template(
    template_path: Path,
    *,
    font_filename: str,
    font_size_px: int,
    font_format: str,
) -> str:
    default_template_path = _default_css_template_path()
    rendered_default = _render_single_stylesheet(
        default_template_path,
        font_filename=font_filename,
        font_size_px=font_size_px,
        font_format=font_format,
    )
    if template_path.resolve() == default_template_path:
        return rendered_default

    rendered_custom = _render_single_stylesheet(
        template_path,
        font_filename=font_filename,
        font_size_px=font_size_px,
        font_format=font_format,
    )
    return rendered_default.rstrip() + "\n\n" + rendered_custom.lstrip()


def _render_single_stylesheet(
    template_path: Path,
    *,
    font_filename: str,
    font_size_px: int,
    font_format: str,
) -> str:
    template = template_path.read_text(encoding="utf-8")
    return (
        template.replace("__FONT_FILENAME__", font_filename)
        .replace("__FONT_FORMAT__", font_format)
        .replace("__FONT_SIZE_PX__", str(int(font_size_px)))
    )


def _render_reel_with_pycaps(
    *,
    template_dir: Path,
    reel_path: Path,
    tmp_output: Path,
    clip_segments: Sequence[SubtitleSegment],
    settings: SubtitleRenderSettings,
    verbose: bool = False,
) -> None:
    try:
        from pycaps.common import Document, Line, Segment, TimeFragment, Word
        from pycaps.template import TemplateLoader
        from pycaps.transcriber import AudioTranscriber
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in runtime smoke tests
        raise RuntimeError(
            "pycaps is not installed in the current environment. Install project "
            "requirements and make sure Playwright Chromium is available.",
        ) from exc

    class TranscriptTranscriber(AudioTranscriber):
        def __init__(self, segments: Sequence[SubtitleSegment]) -> None:
            self._segments = list(segments)

        def transcribe(self, audio_path: str) -> Document:  # noqa: ARG002 - required by interface
            document = Document()
            for seg in self._segments:
                duration = max(0.01, float(seg.end) - float(seg.start))
                segment_time = TimeFragment(start=float(seg.start), end=float(seg.end))
                segment = Segment(time=segment_time)
                line = Line(time=segment_time)
                text_without_spaces = seg.text.replace(" ", "")
                if not text_without_spaces:
                    continue
                letter_duration = duration / len(text_without_spaces)
                last_word_end = float(seg.start)
                for word_text in seg.text.split():
                    end = last_word_end + len(word_text) * letter_duration
                    word = Word(
                        text=word_text,
                        time=TimeFragment(start=last_word_end, end=end),
                    )
                    last_word_end = end
                    line.words.add(word)
                segment.lines.add(line)
                document.segments.add(segment)
            return document

    old_cwd = Path.cwd()
    try:
        os.chdir(reel_path.parent)
        builder = TemplateLoader(template_dir.name).with_input_video(reel_path.name).load(False)
        builder.with_output_video(tmp_output.name)
        builder.with_custom_audio_transcriber(TranscriptTranscriber(clip_segments))
        pipeline = builder.build()
        pipeline.run()
    finally:
        os.chdir(old_cwd)


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000.0)))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    secs = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _default_css_template_path() -> Path:
    return (Path(__file__).resolve().parents[2] / DEFAULT_SUBTITLE_CSS_TEMPLATE).resolve()


def _prepare_subtitle_segments(
    segments: Sequence[SubtitleSegment],
    *,
    settings: SubtitleRenderSettings,
) -> list[SubtitleSegment]:
    prepared: list[SubtitleSegment] = []
    max_chars, min_chars = _subtitle_chunk_limits(settings)
    for seg in segments:
        prepared.extend(
            _split_subtitle_segment(
                seg,
                max_chars=max_chars,
                min_chars=min_chars,
                avoid_finishing_with_short_word_chars=DEFAULT_AVOID_ENDING_WITH_SHORT_WORD_CHARS,
            ),
        )
    return prepared


def _subtitle_chunk_limits(settings: SubtitleRenderSettings) -> tuple[int, int]:
    lines = settings.max_lines if settings.wrap_words else 1
    max_chars = max(12, int(lines) * DEFAULT_SEGMENT_CHARS_PER_LINE)
    min_chars = min(max_chars, max(DEFAULT_SEGMENT_MIN_CHARS, max_chars // 2))
    return max_chars, min_chars


def _split_subtitle_segment(
    segment: SubtitleSegment,
    *,
    max_chars: int,
    min_chars: int,
    avoid_finishing_with_short_word_chars: int,
) -> list[SubtitleSegment]:
    words = _build_timed_words(segment)
    if len(words) <= 1:
        return [segment]

    out: list[SubtitleSegment] = []
    word_index = 0
    while word_index < len(words):
        word_end_index = _find_subtitle_chunk_end(
            words,
            start_index=word_index,
            max_chars=max_chars,
            min_chars=min_chars,
            avoid_finishing_with_short_word_chars=avoid_finishing_with_short_word_chars,
        )
        chunk_words = words[word_index:word_end_index]
        if not chunk_words:
            break
        out.append(
            SubtitleSegment(
                start=round(chunk_words[0].start, 3),
                end=round(chunk_words[-1].end, 3),
                text=" ".join(word.text for word in chunk_words).strip(),
            ),
        )
        word_index = word_end_index
    return out or [segment]


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


def _find_subtitle_chunk_end(
    words: Sequence[_TimedSubtitleWord],
    *,
    start_index: int,
    max_chars: int,
    min_chars: int,
    avoid_finishing_with_short_word_chars: int,
) -> int:
    current_index = start_index
    chars_count = 0

    while current_index < len(words):
        word_len = len(words[current_index].text)
        if chars_count + word_len > max_chars:
            break
        chars_count += word_len
        current_index += 1

    if current_index == start_index:
        current_index += 1

    while (
        current_index < len(words)
        and len(words[current_index - 1].text) < avoid_finishing_with_short_word_chars
    ):
        current_index += 1

    remaining_chars = sum(len(word.text) for word in words[current_index:])
    if remaining_chars < min_chars:
        return len(words)

    return current_index


def _font_format_for_suffix(suffix: str) -> str:
    return {
        ".ttf": "truetype",
        ".otf": "opentype",
        ".woff": "woff",
        ".woff2": "woff2",
    }.get(suffix.lower(), "truetype")


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
