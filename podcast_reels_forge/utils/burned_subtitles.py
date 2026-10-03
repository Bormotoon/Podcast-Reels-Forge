"""Helpers for rendering burned-in subtitles via .ass files."""

from __future__ import annotations

import json
import logging
import re as _re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from podcast_reels_forge.utils.clip_intervals import (
    ClipEdges,
    load_speech_index,
    moment_bounds,
    padded_intervals,
)
from podcast_reels_forge.utils import subtitle_presets as presets
from podcast_reels_forge.utils.subtitle_layout import (
    CENSOR_STYLES,
    LINE_BALANCES,
    NO_LINE_END,
    PUNCTUATION_MODES,
    TEXT_CASES,
    TextMeasurer,
    apply_case,
    build_censor_matcher,
    censor_word,
    lines_fit,
    strip_punctuation,
    wrap_words,
)
from podcast_reels_forge.utils.subtitle_sync import plausible_start
from podcast_reels_forge.utils.reel_markdown import reel_index_from_path

LOG = logging.getLogger(__name__)

DEFAULT_SUBTITLE_FONT = Path("assets/fonts/bignoodletoooblique.ttf")
DEFAULT_SUBTITLE_CSS_TEMPLATE = Path("assets/subtitles/forge_subtitles.css")
# ~9% of the 1080px-wide frame: a readable viral-caption size that matches the
# style-editor default. The previous 36 was ~3% of width → unreadably small.
DEFAULT_FONT_SIZE_PX = 96
DEFAULT_MAX_LINES = 2
# Text spans the frame minus the platform's right-hand action rail (see
# DEFAULT_MARGIN_H): 1080 - 2*140 = 800px ≈ 0.74 of the frame.
DEFAULT_MAX_WIDTH_RATIO = 0.74
DEFAULT_WRAP_WORDS = True
# "style" keeps the alignment from the .ass style (editor or preset); top /
# center / bottom override its row and keep its left/centre/right column.
DEFAULT_VERTICAL_ALIGN = "style"
VERTICAL_ALIGNS = ("style", "top", "center", "bottom")
DEFAULT_VERTICAL_OFFSET = 0.0
DEFAULT_WORD_X_SPACE = 6
DEFAULT_WORD_Y_SPACE = 8

# --- Default caption look -------------------------------------------------
# Tuned for the format that dominates podcast shorts (Hormozi/MrBeast-style
# karaoke captions): a heavy condensed face, a thick black outline instead of a
# drop shadow, and a \kf sweep that fills each spoken word with an accent colour.
#
# ASS colours are &HAABBGGRR — alpha first, then blue/green/red.
DEFAULT_PRIMARY_COLOUR = "&H000AD6FF"    # #FFD60A amber — words already spoken
DEFAULT_SECONDARY_COLOUR = "&H00FFFFFF"  # white — words not yet spoken
DEFAULT_OUTLINE_COLOUR = "&H00000000"    # opaque black
DEFAULT_BACK_COLOUR = "&H80000000"       # half-transparent black (shadow/box)
# A heavy outline is what keeps captions legible over arbitrary footage; at
# Fontsize 96 anything below ~6 starts breaking up on bright frames.
DEFAULT_OUTLINE_WIDTH = 8
DEFAULT_SHADOW_DEPTH = 0
# Horizontal inset that clears the right-hand action rail on Reels/TikTok/Shorts
# (the narrowest safe zone is 140px). Applied on both sides to keep text centred.
DEFAULT_MARGIN_H = 140
# Sits above the caption/handle/audio chrome pinned to the bottom of the frame.
DEFAULT_MARGIN_V = 470
DEFAULT_FADE_IN_S = 0.12
DEFAULT_FADE_OUT_S = 0.08
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
# The 25 chars/line figure above assumes text spanning this share of the frame.
# Kept separate from DEFAULT_MAX_WIDTH_RATIO so retuning the shipped default
# rescales line length instead of silently moving the guideline itself.
CHARS_PER_LINE_REFERENCE_RATIO = 0.65
DEFAULT_MIN_DURATION_S = 1.5
DEFAULT_MAX_DURATION_S = 7.0
DEFAULT_GAP_BETWEEN_SUBTITLES_S = 0.15
# A pause at least this long ends a cue (stable-ts / auto-subs use 0.5 s).
DEFAULT_PAUSE_SPLIT_S = 0.5
# Fades only around a pause at least this long. On 127 real podcast clips 93%
# of cues follow the previous one within 0.2 s; fading every one of them made
# the text blink at each change instead of easing in after silence.
DEFAULT_FADE_MIN_GAP_S = 0.3
# Bottom-anchored captions: the shorter line on top hides less of the picture
# (BBC/Netflix "pyramid"), and it avoids a one-word bottom line.
DEFAULT_LINE_BALANCE = "bottom_heavy"

# How the spoken word is shown inside a cue:
#   none    — the cue appears whole, no per-word effect;
#   karaoke — \kf sweep from SecondaryColour to PrimaryColour;
#   word    — only the active word takes the Highlight style;
#   fill    — the active word and every word before it take it;
#   reveal  — words appear as they are spoken (typewriter);
#   pop     — like "word", and the active word briefly scales up.
HIGHLIGHT_MODES = ("none", "karaoke", "word", "fill", "reveal", "pop")
_POP_START_SCALE = 1.12
_POP_MS = 120


@dataclass(frozen=True)
class _TimedSubtitleWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SubtitleSegment:
    start: float
    end: float
    text: str
    # Real per-word timings from the transcript, when it carries them. Empty
    # means the karaoke timing has to be interpolated from the segment span.
    words: tuple[_TimedSubtitleWord, ...] = ()
    # Diarization label ("SPEAKER_00"); empty when the transcript has none.
    speaker: str = ""


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
    fade_in_duration: float = DEFAULT_FADE_IN_S
    fade_out_duration: float = DEFAULT_FADE_OUT_S
    # Word-by-word \kf highlighting. Off by default: the whole cue appears at
    # once, so a word-timing error is a small lag, not a visibly wrong word.
    # Legacy switch for highlight="karaoke".
    karaoke: bool = False
    # Built-in look (utils/subtitle_presets.py). Empty: the editor's .ass file,
    # or the "forge" look when there is none.
    preset: str = ""
    highlight: str = "none"
    # Active-word colour (#RRGGBB) on top of the Highlight style; empty = style.
    highlight_color: str = ""
    text_case: str = "none"
    strip_punctuation: str = "keep"
    censor_words: tuple[str, ...] = ()
    censor_style: str = "middle"
    line_balance: str = DEFAULT_LINE_BALANCE
    # Fade a cue in/out only when the gap to its neighbour is at least this
    # long (seconds); 0 fades every cue.
    fade_min_gap_s: float = DEFAULT_FADE_MIN_GAP_S
    # 0 = derived from the font, its size and the usable width.
    max_chars_per_line: int = 0
    # 0 = no limit; 1 shows one word at a time.
    max_words_per_cue: int = 0
    pause_split_s: float = DEFAULT_PAUSE_SPLIT_S
    min_duration_s: float = DEFAULT_MIN_DURATION_S
    max_duration_s: float = DEFAULT_MAX_DURATION_S
    min_gap_s: float = DEFAULT_GAP_BETWEEN_SUBTITLES_S
    # Edge softening (\blur): turns an outline into a glow.
    blur: float = 0.0
    # Text colour per speaker (#RRGGBB), in order of first appearance in a reel.
    speaker_colors: tuple[str, ...] = ()
    # A change of speaker always starts a new cue.
    split_on_speaker: bool = True

    @property
    def highlight_mode(self) -> str:
        if self.highlight != "none":
            return self.highlight
        return "karaoke" if self.karaoke else "none"


def _coerce_choice(value: object, allowed: Sequence[str], *, default: str) -> str:
    text = str(value if value is not None else default).strip().lower().replace("-", "_")
    return text if text in allowed else default


def _coerce_str_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [part for part in _re.split(r"[,\n]", value)]
    elif isinstance(value, Sequence):
        items = [str(part) for part in value]
    else:
        return ()
    return tuple(item.strip() for item in items if str(item).strip())


def _coerce_hex(value: object) -> str:
    text = str(value or "").strip()
    if _re.fullmatch(r"#?[0-9a-fA-F]{6}", text):
        return "#" + text.lstrip("#").upper()
    return ""


def subtitle_settings_from_conf(
    conf: Mapping[str, Any] | None,
    *,
    repo_dir: Path,
) -> SubtitleRenderSettings:
    subtitles_conf = conf.get("subtitles", {}) if isinstance(conf, Mapping) else {}
    if not isinstance(subtitles_conf, Mapping):
        subtitles_conf = {}
    preset_name = str(subtitles_conf.get("preset") or "").strip().lower().replace("-", "_")
    if preset_name and presets.get_preset(preset_name) is None:
        LOG.warning(
            "Unknown subtitles.preset %r; known: %s",
            preset_name, ", ".join(presets.preset_names()),
        )
        preset_name = ""
    if preset_name:
        # A preset's render settings are defaults; explicit keys win. Its font
        # applies when config.yaml names none.
        look = presets.preset_look(preset_name)
        subtitles_conf = {
            "font": look["fontPath"],
            **presets.preset_render(preset_name),
            **{k: v for k, v in subtitles_conf.items() if v is not None and v != ""},
        }
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
            default=DEFAULT_FADE_IN_S,
            # 0 is a meaningful value: it turns the fade off entirely.
            minimum=0.0,
            maximum=1.0,
        ),
        fade_out_duration=_coerce_float(
            subtitles_conf.get("fade_out_duration"),
            default=DEFAULT_FADE_OUT_S,
            minimum=0.0,
            maximum=1.0,
        ),
        karaoke=_coerce_bool(subtitles_conf.get("karaoke"), default=False),
        preset=preset_name,
        highlight=_coerce_choice(subtitles_conf.get("highlight"), HIGHLIGHT_MODES, default="none"),
        highlight_color=_coerce_hex(subtitles_conf.get("highlight_color")),
        text_case=_coerce_choice(subtitles_conf.get("text_case"), TEXT_CASES, default="none"),
        strip_punctuation=_coerce_choice(
            subtitles_conf.get("strip_punctuation"), PUNCTUATION_MODES, default="keep",
        ),
        censor_words=_coerce_str_list(subtitles_conf.get("censor_words")),
        censor_style=_coerce_choice(subtitles_conf.get("censor_style"), CENSOR_STYLES, default="middle"),
        line_balance=_coerce_choice(
            subtitles_conf.get("line_balance"), LINE_BALANCES, default=DEFAULT_LINE_BALANCE,
        ),
        fade_min_gap_s=_coerce_float(
            subtitles_conf.get("fade_min_gap_s"), default=DEFAULT_FADE_MIN_GAP_S, minimum=0.0, maximum=5.0,
        ),
        max_chars_per_line=_coerce_int(subtitles_conf.get("max_chars_per_line"), default=0, minimum=0),
        max_words_per_cue=_coerce_int(subtitles_conf.get("max_words_per_cue"), default=0, minimum=0),
        pause_split_s=_coerce_float(
            subtitles_conf.get("pause_split_s"), default=DEFAULT_PAUSE_SPLIT_S, minimum=0.05, maximum=5.0,
        ),
        min_duration_s=_coerce_float(
            subtitles_conf.get("min_duration_s"), default=DEFAULT_MIN_DURATION_S, minimum=0.0, maximum=5.0,
        ),
        max_duration_s=_coerce_float(
            subtitles_conf.get("max_duration_s"), default=DEFAULT_MAX_DURATION_S, minimum=0.5, maximum=30.0,
        ),
        min_gap_s=_coerce_float(
            subtitles_conf.get("min_gap_s"), default=DEFAULT_GAP_BETWEEN_SUBTITLES_S, minimum=0.0, maximum=2.0,
        ),
        blur=_coerce_float(subtitles_conf.get("blur"), default=0.0, minimum=0.0, maximum=30.0),
        speaker_colors=tuple(
            c for c in (_coerce_hex(v) for v in _coerce_str_list(subtitles_conf.get("speaker_colors"))) if c
        ),
        split_on_speaker=_coerce_bool(subtitles_conf.get("split_on_speaker"), default=True),
    )


def ensure_reel_burned_subtitles(
    moment: Mapping[str, Any],
    reel_path: Path,
    *,
    transcript_json_path: Path,
    padding: float,
    settings: SubtitleRenderSettings,
    verbose: bool = False,
    interval: tuple[float, float] | None = None,
) -> Path | None:
    """Write a reel's .srt/.ass. ``interval`` is the exact span the reel was
    cut from (see :func:`padded_intervals`); without it the moment is padded
    symmetrically by ``padding``."""

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
        interval=interval,
    )


def sync_reel_burned_subtitles(
    moments: Sequence[Mapping[str, Any]],
    reels_root: Path,
    *,
    transcript_json_path: Path,
    padding: float,
    settings: SubtitleRenderSettings,
    verbose: bool = False,
    edges: ClipEdges | None = None,
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

    transcript_segments = retime_segments(
        load_transcript_segments(transcript_json_path),
        load_saved_retiming(reels_root),
    )
    # The same per-clip interval the cut used, so re-synced subtitles line up
    # with the footage of reels that sit close together.
    intervals = padded_intervals(
        [moment_bounds(m) for m in moments],
        float(padding),
        index=load_speech_index(transcript_json_path) if edges is not None else None,
        edges=edges,
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
            template_dir=reel_path.parent,
            verbose=verbose,
            interval=intervals[moment_index],
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
    interval: tuple[float, float] | None = None,
) -> Path | None:
    start = _coerce_float(moment.get("start"), default=0.0)
    end = _coerce_float(moment.get("end"), default=0.0)
    if end <= start:
        raise ValueError(f"Invalid boundaries: {start} - {end}")

    clip_start, clip_end = interval or (max(0.0, start - float(padding)), end + float(padding))
    clip_segments = slice_segments_for_clip(
        transcript_segments,
        clip_start=clip_start,
        clip_end=clip_end,
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


def _ass_play_res_y(header: str) -> int:
    """Read PlayResY from an ASS header, defaulting to the 9:16 canvas."""

    match = _re.search(r"^PlayResY:\s*(\d+)", header, _re.MULTILINE)
    return int(match.group(1)) if match else 1920


def _ass_style_margin_v(header: str) -> int:
    """Read the Default style's MarginV, driven by the header's Format line."""

    style_match = _re.search(r"^Style:\s*([^\n]+)", header, _re.MULTILINE)
    if not style_match:
        return 0
    fields = [f.strip() for f in style_match.group(1).split(",")]

    format_match = _re.search(r"^Format:\s*([^\n]+)", header, _re.MULTILINE)
    index = None
    if format_match:
        names = [n.strip().lower() for n in format_match.group(1).split(",")]
        if "marginv" in names:
            index = names.index("marginv")
    if index is None:
        # V4+ default ordering, counting Name as field 0.
        index = 21

    if index >= len(fields):
        return 0
    try:
        return int(float(fields[index]))
    except (TypeError, ValueError):
        return 0


def _dialogue_margin_v(header: str, settings: SubtitleRenderSettings) -> int:
    """RU: MarginV для строки Dialogue с учётом ``subtitles.vertical_offset``.

    EN: Per-cue MarginV honouring ``subtitles.vertical_offset``.

    0 means "inherit the style", which is what we emit when the offset is zero —
    so the default config keeps producing byte-identical output. A positive
    offset pushes the text away from the edge it is anchored to.
    """

    offset = float(settings.vertical_offset)
    if offset == 0.0:
        return 0

    shift = int(round(offset * _ass_play_res_y(header)))
    margin_v = _ass_style_margin_v(header) + shift
    # 0 would read as "inherit"; keep at least 1px so the override survives.
    return max(1, margin_v)


def _fade_ms(
    seg: SubtitleSegment,
    settings: SubtitleRenderSettings,
    *,
    fade_in_ok: bool = True,
    fade_out_ok: bool = True,
) -> tuple[int, int]:
    """Fade-in / fade-out of a cue in ms, clamped so they never outlast it.

    ``fade_in_ok`` / ``fade_out_ok`` are False next to a back-to-back cue
    (see ``subtitles.fade_min_gap_s``): the text then cuts straight over.
    """

    fade_in = max(0.0, float(settings.fade_in_duration)) if fade_in_ok else 0.0
    fade_out = max(0.0, float(settings.fade_out_duration)) if fade_out_ok else 0.0
    if fade_in <= 0 and fade_out <= 0:
        return 0, 0

    duration = max(0.0, float(seg.end) - float(seg.start))
    total = fade_in + fade_out
    if total > duration and total > 0:
        # Scale both down proportionally so the cue still reaches full opacity.
        scale = duration / total
        fade_in *= scale
        fade_out *= scale
    return int(round(fade_in * 1000)), int(round(fade_out * 1000))


def _fade_tag(
    seg: SubtitleSegment,
    settings: SubtitleRenderSettings,
    *,
    fade_in_ok: bool = True,
    fade_out_ok: bool = True,
) -> str:
    """RU: Тег ``\\fad`` для плавного появления/исчезновения реплики.

    EN: The ``\\fad`` tag that fades a cue in and out.

    ``subtitles.fade_in_duration`` / ``fade_out_duration`` used to be parsed and
    then dropped on the floor, so the GUI sliders did nothing. Durations are
    clamped so the two fades can never outlast the cue itself.
    """

    in_ms, out_ms = _fade_ms(seg, settings, fade_in_ok=fade_in_ok, fade_out_ok=fade_out_ok)
    if in_ms <= 0 and out_ms <= 0:
        return ""
    return f"{{\\fad({in_ms},{out_ms})}}"


# --- Style header -------------------------------------------------------------

# Top-anchored text clears the platform's top bar (IG: 220px).
DEFAULT_TOP_MARGIN_V = 250


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


@lru_cache(maxsize=32)
def _font_family(font_path: str) -> str:
    """The family name libass/fontconfig will look the font up by.

    The style editor writes the file's stem ("MontserratBlack"), which only
    happens to work for fonts whose family is the stem in another case.
    """

    try:
        from fontTools.ttLib import TTFont  # noqa: PLC0415 - heavy, optional

        family = TTFont(font_path, lazy=True)["name"].getDebugName(1)
        if family:
            return str(family)
    except Exception:  # noqa: BLE001 - missing or unreadable font
        pass
    return _font_name_from_path(Path(font_path))


def _squash_name(name: str) -> str:
    return _re.sub(r"[\s_\-]+", "", name).lower()


def _normalise_header(header: str, settings: SubtitleRenderSettings) -> str:
    """Make an editor-written header render the way it was previewed."""

    family = _font_family(str(settings.font_path))
    aliases = {_squash_name(settings.font_path.stem), _squash_name(family)}
    lines = []
    for line in header.splitlines():
        if line.startswith("Style:"):
            parts = line[len("Style:"):].split(",")
            if len(parts) > 1 and _squash_name(parts[1]) in aliases:
                parts[1] = family
                line = "Style:" + ",".join(parts)
        lines.append(line)
    header = "\n".join(lines)
    for key, value in (("WrapStyle", "0"), ("ScaledBorderAndShadow", "yes")):
        if _re.search(rf"^{key}:", header, _re.MULTILINE):
            continue
        if _re.search(r"^PlayResY:", header, _re.MULTILINE):
            header = _re.sub(
                r"^(PlayResY:[^\n]*)$", rf"\1\n{key}: {value}", header, count=1, flags=_re.MULTILINE,
            )
        else:
            header = header.replace("[Script Info]", f"[Script Info]\n{key}: {value}", 1)
    return header


def _default_ass_header(font_name: str, font_size: int) -> str:
    """RU: Стиль субтитров по умолчанию — «вирусные» подписи (пресет forge).

    EN: The default caption style — the "forge" preset.

    Used when neither a preset nor a style file from the visual editor is set.
    Colours read as PrimaryColour = already spoken, SecondaryColour = still
    upcoming, because a ``\\kf`` sweep fills from secondary to primary.
    """

    look = presets.scale_look(presets.preset_look(presets.DEFAULT_PRESET), font_size)
    return presets.ass_header(look, font_name)


def _resolve_style_header(settings: SubtitleRenderSettings) -> str:
    """The ``[Script Info]`` + ``[V4+ Styles]`` part of a reel's .ass.

    An explicit ``subtitles.preset`` wins; otherwise the style editor's file
    (``subtitles.ass_style``, then the default location); otherwise "forge".
    """

    family = _font_family(str(settings.font_path))
    if settings.preset:
        look = presets.scale_look(presets.preset_look(settings.preset), settings.font_size_px)
        return presets.ass_header(look, family)
    for candidate in (settings.ass_style, _find_ass_style_file()):
        if candidate is not None and candidate.exists():
            return _normalise_header(candidate.read_text(encoding="utf-8").rstrip(), settings)
    return _default_ass_header(family, settings.font_size_px)


def _parse_ass_styles(header: str) -> dict[str, dict[str, str]]:
    """Styles by lowercased name, each as {lowercased field: value}."""

    default_names = [n.strip().lower() for n in presets.ASS_STYLE_FORMAT[len("Format:"):].split(",")]
    names = default_names
    styles: dict[str, dict[str, str]] = {}
    in_styles = False
    for raw in header.splitlines():
        line = raw.strip()
        if line.startswith("["):
            in_styles = line.lower() in ("[v4+ styles]", "[v4 styles]")
            continue
        if not in_styles:
            continue
        lower = line.lower()
        if lower.startswith("format:"):
            names = [n.strip().lower() for n in line[len("Format:"):].split(",")]
        elif lower.startswith("style:"):
            values = [v.strip() for v in line[len("Style:"):].split(",")]
            fields = dict(zip(names, values))
            styles[fields.get("name", "").lower()] = fields
    return styles


def _style_num(style: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(style.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _ass_play_res_x(header: str) -> int:
    match = _re.search(r"^PlayResX:\s*(\d+)", header, _re.MULTILINE)
    return int(match.group(1)) if match else 1080


@dataclass(frozen=True)
class _Layout:
    """Where and how wide a reel's cues are laid out."""

    measurer: TextMeasurer
    max_width: float
    max_lines: int
    play_x: int
    play_y: int
    alignment: int
    margin_l: int
    margin_r: int
    # Per-event MarginL/MarginR; 0 inherits the style's.
    event_margin_l: int = 0
    event_margin_r: int = 0


def _layout_for(header: str, settings: SubtitleRenderSettings) -> _Layout:
    style = _parse_ass_styles(header).get("default", {})
    play_x = _ass_play_res_x(header)
    play_y = _ass_play_res_y(header)
    margin_l = int(_style_num(style, "marginl", 0))
    margin_r = int(_style_num(style, "marginr", 0))
    usable = play_x - margin_l - margin_r
    if usable <= 0:
        usable = play_x
    font_path = settings.font_path if settings.font_path.exists() else None
    measurer = TextMeasurer(
        font_path,
        _style_num(style, "fontsize", settings.font_size_px),
        scale_x=_style_num(style, "scalex", 100),
        spacing=_style_num(style, "spacing", 0),
        # Outline (or the box padding, for BorderStyle 3/4) widens each line.
        outline=_style_num(style, "outline", 0),
    )
    alignment = int(_style_num(style, "alignment", 2)) or 2
    target = float(settings.max_width_ratio) * play_x
    event_margin = 0
    if target > usable and (alignment - 1) % 3 == 1:
        # A width ratio wider than the style's margins widens the text for
        # real: centred cues get symmetric per-event margins to match.
        event_margin = max(1, int(round((play_x - target) / 2)))
        usable = play_x - 2 * event_margin
    return _Layout(
        measurer=measurer,
        max_width=min(float(usable), target),
        max_lines=settings.max_lines if settings.wrap_words else 1,
        play_x=play_x,
        play_y=play_y,
        alignment=alignment,
        margin_l=event_margin or margin_l,
        margin_r=event_margin or margin_r,
        event_margin_l=event_margin,
        event_margin_r=event_margin,
    )


def _position_tag(layout: _Layout, settings: SubtitleRenderSettings) -> str:
    """``\\an`` (and ``\\pos`` for an offset centre) for subtitles.vertical_align."""

    align = settings.vertical_align
    if align == "style":
        return ""
    column = (layout.alignment - 1) % 3  # 0 left, 1 centre, 2 right
    row_base = {"bottom": 1, "center": 4, "top": 7}[align]
    tag = f"\\an{row_base + column}"
    if align == "center" and settings.vertical_offset:
        # libass ignores MarginV for middle rows, so an offset needs \pos.
        x = {
            0: layout.margin_l,
            1: layout.margin_l + (layout.play_x - layout.margin_l - layout.margin_r) / 2,
            2: layout.play_x - layout.margin_r,
        }[column]
        y = layout.play_y / 2 - float(settings.vertical_offset) * layout.play_y
        tag += f"\\pos({round(x)},{round(y)})"
    return tag


def _event_margin_v(header: str, settings: SubtitleRenderSettings, layout: _Layout) -> int:
    align = settings.vertical_align
    style_row = (
        "bottom" if layout.alignment in (1, 2, 3)
        else "center" if layout.alignment in (4, 5, 6)
        else "top"
    )
    if align in ("style", "center", style_row):
        return _dialogue_margin_v(header, settings)
    # Moving a centred style to an edge: its MarginV (usually 0) means nothing
    # there, so start from that edge's safe-zone default.
    base = DEFAULT_MARGIN_V if align == "bottom" else DEFAULT_TOP_MARGIN_V
    return max(1, base + int(round(float(settings.vertical_offset) * layout.play_y)))


# --- Cue text -----------------------------------------------------------------


def _escape_ass_text(text: str) -> str:
    """Transcript text must not open override blocks or start escapes."""

    return text.replace("\\", "/").replace("{", "(").replace("}", ")")


def _censor_for(settings: SubtitleRenderSettings) -> Callable[[str], bool] | None:
    return build_censor_matcher(settings.censor_words) if settings.censor_words else None


def _display_word(
    word: str,
    settings: SubtitleRenderSettings,
    censor: Callable[[str], bool] | None,
) -> str:
    text = word
    if censor is not None and censor(text):
        text = censor_word(text, settings.censor_style)
    text = strip_punctuation(text, settings.strip_punctuation)
    text = apply_case(text, settings.text_case)
    return _escape_ass_text(text)


def _split_colour(value: str) -> tuple[str, str]:
    """``&HAABBGGRR`` → (``AA``, ``BBGGRR``)."""

    digits = value.strip().lstrip("&").lstrip("Hh").rstrip("&").zfill(8)[-8:].upper()
    return digits[:2], digits[2:]


def _colour_tags(value: str, slot: int = 1) -> str:
    alpha, bgr = _split_colour(value)
    return f"\\{slot}c&H{bgr}&\\{slot}a&H{alpha}&"


def _hex_tag(hex_colour: str, slot: int = 1) -> str:
    return f"\\{slot}c{presets.ass_override_colour(hex_colour)}" if hex_colour else ""


def _speaker_colours(
    segments: Sequence[SubtitleSegment],
    settings: SubtitleRenderSettings,
) -> dict[str, str]:
    """Speaker label → colour, assigned in order of first appearance."""

    colours: dict[str, str] = {}
    if not settings.speaker_colors:
        return colours
    for seg in segments:
        label = seg.speaker
        if label and label not in colours:
            colours[label] = settings.speaker_colors[len(colours) % len(settings.speaker_colors)]
    return colours


@dataclass(frozen=True)
class _WordLook:
    """Override tags that dress the active and inactive words of a cue."""

    inactive: str
    active: str
    after_active: str
    pop: Callable[[int], str] | None


def _word_look(
    styles: Mapping[str, Mapping[str, str]],
    settings: SubtitleRenderSettings,
    *,
    blur: str,
    speaker_colour: str,
) -> _WordLook:
    default = styles.get("default", {})
    highlight = styles.get("highlight")
    mode = settings.highlight_mode
    if highlight is not None:
        # The look defines both: Default = inactive, Highlight = active.
        inactive = blur + _hex_tag(speaker_colour)
        active = "\\rHighlight" + blur + _hex_tag(settings.highlight_color)
        after = "\\r" + inactive
        scale_src = highlight
    else:
        # Same reading as the \kf sweep: SecondaryColour = not (yet) spoken,
        # PrimaryColour = the spoken word.
        secondary = default.get("secondarycolour", "&H00FFFFFF")
        primary = default.get("primarycolour", "&H000AD6FF")
        if mode == "reveal":
            # Typewriter keeps the cue's own colour; only visibility changes.
            secondary = primary
        inactive = blur + (_hex_tag(speaker_colour) or _colour_tags(secondary))
        active = blur + (_hex_tag(settings.highlight_color) or _colour_tags(primary))
        after = inactive
        scale_src = default
    pop = None
    if mode == "pop":
        sx = _style_num(scale_src, "scalex", 100)
        sy = _style_num(scale_src, "scaley", 100)
        after += f"\\fscx{sx:g}\\fscy{sy:g}"

        def pop(duration_ms: int, sx: float = sx, sy: float = sy) -> str:
            ms = max(1, min(_POP_MS, duration_ms))
            big_x, big_y = sx * _POP_START_SCALE, sy * _POP_START_SCALE
            return f"\\fscx{big_x:g}\\fscy{big_y:g}\\t(0,{ms},\\fscx{sx:g}\\fscy{sy:g})"

    return _WordLook(inactive=inactive, active=active, after_active=after, pop=pop)


def _join_rows(parts: Sequence[str], breaks: set[int]) -> str:
    out = ""
    for index, part in enumerate(parts):
        if index:
            out += "\\N" if index in breaks else " "
        out += part
    return out


def _tag(body: str) -> str:
    return f"{{{body}}}" if body else ""


def _cue_dialogues(
    seg: SubtitleSegment,
    *,
    settings: SubtitleRenderSettings,
    layout: _Layout,
    styles: Mapping[str, Mapping[str, str]],
    margin_v: int,
    position: str,
    censor: Callable[[str], bool] | None,
    speaker_colour: str,
    fade_in_ok: bool = True,
    fade_out_ok: bool = True,
) -> list[str]:
    timed = [
        (_display_word(w.text, settings, censor), w)
        for w in _build_timed_words(seg)
    ]
    timed = [(text, w) for text, w in timed if text]
    if not timed:
        return []
    words = [text for text, _ in timed]
    rows = wrap_words(
        words,
        layout.measurer,
        max_width=layout.max_width,
        max_lines=layout.max_lines,
        balance=settings.line_balance,
    )
    breaks: set[int] = set()
    index = 0
    for row in rows[:-1]:
        index += len(row)
        breaks.add(index)

    blur = f"\\blur{settings.blur:g}" if settings.blur > 0 else ""
    mode = settings.highlight_mode

    def dialogue(start: float, end: float, text: str) -> str:
        return (
            f"Dialogue: 0,{_fmt_ass_time(start)},{_fmt_ass_time(end)},Default,,"
            f"{layout.event_margin_l},{layout.event_margin_r},{margin_v},,{text}"
        )

    if mode == "none":
        base = position + blur + _hex_tag(speaker_colour)
        return [dialogue(seg.start, seg.end, _fade_tag(seg, settings, fade_in_ok=fade_in_ok, fade_out_ok=fade_out_ok) + _tag(base) + _join_rows(words, breaks))]

    if mode == "karaoke":
        # \kf durations run back to back from the cue's start, so a cue that
        # appears before its first word (merged blocks, min-duration padding)
        # needs that lead-in as an empty syllable, or every highlight is early.
        first = timed[0][1]
        lead_cs = int(round((first.start - seg.start) * 100))
        parts: list[str] = []
        for i, (text, w) in enumerate(timed):
            dur_cs = int(round((w.end - w.start) * 100))
            lead = f"{{\\k{lead_cs}}}" if i == 0 and lead_cs > 0 else ""
            parts.append(f"{lead}{{\\kf{dur_cs}}}{text}")
        base = position + blur + _hex_tag(speaker_colour, 2)
        return [dialogue(seg.start, seg.end, _fade_tag(seg, settings, fade_in_ok=fade_in_ok, fade_out_ok=fade_out_ok) + _tag(base) + _join_rows(parts, breaks))]

    # Per-word modes: one event per spoken word, each showing the whole cue
    # with that word dressed up (ai-video-captions / pycaps approach). The
    # layout of every event is identical, so the text does not jump.
    look = _word_look(styles, settings, blur=blur, speaker_colour=speaker_colour)
    # Hidden (not yet spoken) words in "reveal". \4a would also clear a
    # BorderStyle 4 box, which libass draws with the event's last alpha.
    box_per_event = int(_style_num(styles.get("default", {}), "borderstyle", 1)) == 4
    hide = "{\\1a&HFF&\\3a&HFF&}" if box_per_event else "{\\alpha&HFF&}"
    fade_in_ms, fade_out_ms = _fade_ms(seg, settings, fade_in_ok=fade_in_ok, fade_out_ok=fade_out_ok)
    events: list[str] = []
    n = len(timed)
    prev_end = float(seg.start)
    for i in range(n):
        start = float(seg.start) if i == 0 else max(prev_end, timed[i][1].start)
        end = float(seg.end) if i == n - 1 else max(start, timed[i + 1][1].start)
        if end - start < 0.01:
            continue
        prev_end = end
        duration_ms = int(round((end - start) * 1000))
        parts = []
        for j, text in enumerate(words):
            is_active = j == i or (mode == "fill" and j < i)
            if mode == "reveal" and j > i:
                parts.append((hide if j == i + 1 else "") + text)
                continue
            if is_active:
                extra = look.pop(duration_ms) if (look.pop and j == i) else ""
                parts.append(_tag(look.active + extra) + text + _tag(look.after_active))
            else:
                parts.append(text)
        fade = ""
        f_in = fade_in_ms if i == 0 else 0
        f_out = fade_out_ms if i == n - 1 else 0
        if f_in or f_out:
            fade = f"{{\\fad({f_in},{f_out})}}"
        text = fade + _tag(position + look.inactive) + _join_rows(parts, breaks)
        events.append(dialogue(start, end, text))
    return events


def _write_ass_file(path: Path, segments: Sequence[SubtitleSegment], settings: SubtitleRenderSettings) -> None:
    header = _resolve_style_header(settings)
    layout = _layout_for(header, settings)
    styles = _parse_ass_styles(header)
    margin_v = _event_margin_v(header, settings, layout)
    position = _position_tag(layout, settings)
    censor = _censor_for(settings)
    colours = _speaker_colours(segments, settings)

    events = "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    dialogue_lines: list[str] = []
    gap = float(settings.fade_min_gap_s)
    for index, seg in enumerate(segments):
        prev_end = segments[index - 1].end if index > 0 else None
        next_start = segments[index + 1].start if index + 1 < len(segments) else None
        dialogue_lines.extend(
            _cue_dialogues(
                seg,
                settings=settings,
                layout=layout,
                styles=styles,
                margin_v=margin_v,
                position=position,
                censor=censor,
                speaker_colour=colours.get(seg.speaker, ""),
                fade_in_ok=prev_end is None or seg.start - prev_end >= gap,
                fade_out_ok=next_start is None or next_start - seg.end >= gap,
            ),
        )

    path.write_text(header + "\n" + events + "\n".join(dialogue_lines) + "\n", encoding="utf-8")


def load_transcript_words(data: Mapping[str, Any]) -> list[_TimedSubtitleWord]:
    """Every word the transcript timed, in order.

    faster-whisper emits these for both modes, but the burn path used to
    ignore them and guess word timings from character lengths instead.
    """

    words: list[_TimedSubtitleWord] = []
    raw_segments = data.get("segments")
    if not isinstance(raw_segments, list):
        return words

    for raw_segment in raw_segments:
        if not isinstance(raw_segment, Mapping):
            continue
        raw_words = raw_segment.get("words")
        if not isinstance(raw_words, list):
            continue
        for raw_word in raw_words:
            if not isinstance(raw_word, Mapping):
                continue
            text = str(raw_word.get("word", "")).strip()
            start = _coerce_float(raw_word.get("start"), default=-1.0)
            end = _coerce_float(raw_word.get("end"), default=-1.0)
            if not text or start < 0 or end <= start:
                continue
            # A word right after a pause often starts at the pause in Whisper's
            # timing, which would show its cue a second or two early.
            start = plausible_start(start, end, text)
            words.append(_TimedSubtitleWord(start=start, end=end, text=text))

    words.sort(key=lambda word: (word.start, word.end))
    return words


def _words_within(
    words: Sequence[_TimedSubtitleWord],
    start: float,
    end: float,
) -> tuple[_TimedSubtitleWord, ...]:
    """Words whose midpoint falls inside the span."""

    if not words:
        return ()
    return tuple(
        word
        for word in words
        if start <= (word.start + word.end) / 2.0 <= end
    )


def load_transcript_segments(path: Path) -> list[SubtitleSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []

    all_words = load_transcript_words(data)

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
            sentence_segments.append(
                SubtitleSegment(
                    start=start,
                    end=end,
                    text=text,
                    words=_words_within(all_words, start, end),
                    speaker=_speaker_label(raw.get("speaker")),
                ),
            )
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
        segments.append(
            SubtitleSegment(
                start=start,
                end=end,
                text=text,
                words=_words_within(all_words, start, end),
                speaker=_speaker_label(raw.get("speaker")),
            ),
        )
    return segments


def _speaker_label(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"none", "null", "unknown"} else text


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
        if not seg.words:
            out.append(
                SubtitleSegment(
                    start=round(shifted_start, 3),
                    end=round(shifted_end, 3),
                    text=seg.text,
                    speaker=seg.speaker,
                ),
            )
            continue
        # Word timings are absolute in the episode; move them onto the clip's
        # own timeline. A word belongs to the clip when its midpoint does, so
        # one clipped at the boundary is shown once, not in both reels.
        clip_len = float(clip_end) - float(clip_start)
        kept = [
            _TimedSubtitleWord(
                start=round(max(0.0, word.start - clip_start), 3),
                end=round(min(clip_len, word.end - clip_start), 3),
                text=word.text,
            )
            for word in seg.words
            if clip_start <= (word.start + word.end) / 2.0 <= clip_end
        ]
        kept = [word for word in kept if word.end > word.start]
        if not kept:
            continue
        # The text is rebuilt from the words that are actually in the clip.
        # A sentence cut by the clip boundary used to keep its full text with
        # only part of its word timings; the mismatch sent the karaoke to
        # interpolation, which spread the whole sentence over the part of it
        # that fits — seconds of drift by the end of the cue.
        out.append(
            SubtitleSegment(
                start=kept[0].start,
                end=kept[-1].end,
                text=" ".join(word.text for word in kept),
                words=tuple(kept),
                speaker=seg.speaker,
            ),
        )
    return out


WordKey = tuple[float, str]


def word_key(word: _TimedSubtitleWord) -> WordKey:
    return (round(float(word.start), 3), word.text)


def clip_words(
    segments: Sequence[SubtitleSegment],
    *,
    clip_start: float,
    clip_end: float,
) -> list[_TimedSubtitleWord]:
    """Transcript words (episode timeline) that belong to a clip's interval."""

    return [
        word
        for seg in segments
        for word in seg.words
        if clip_start <= (word.start + word.end) / 2.0 <= clip_end
    ]


def retime_segments(
    segments: Sequence[SubtitleSegment],
    retimed: Mapping[WordKey, tuple[float, float]],
) -> list[SubtitleSegment]:
    """Segments with some word timings replaced (see utils/subtitle_sync.py)."""

    if not retimed:
        return list(segments)
    out: list[SubtitleSegment] = []
    for seg in segments:
        if not seg.words or not any(word_key(w) in retimed for w in seg.words):
            out.append(seg)
            continue
        words = tuple(
            _TimedSubtitleWord(*retimed[word_key(w)], w.text) if word_key(w) in retimed else w
            for w in seg.words
        )
        out.append(
            SubtitleSegment(
                start=min(seg.start, words[0].start),
                end=max(seg.end, words[-1].end),
                text=seg.text,
                words=words,
                speaker=seg.speaker,
            ),
        )
    return out


SUBTITLE_SYNC_FILE = "subtitle_sync.json"


def load_saved_retiming(reels_dir: Path) -> dict[WordKey, tuple[float, float]]:
    """Word timings a cut's Whisper check replaced, so a later subtitle
    re-sync (without re-cutting) keeps them."""

    path = reels_dir / SUBTITLE_SYNC_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    retimed: dict[WordKey, tuple[float, float]] = {}
    for row in data.get("retimed_words", []) if isinstance(data, dict) else []:
        try:
            key = (round(float(row["orig_start"]), 3), str(row["text"]))
            retimed[key] = (float(row["start"]), float(row["end"]))
        except (KeyError, TypeError, ValueError):
            continue
    return retimed


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

    header = _resolve_style_header(settings)
    layout = _layout_for(header, settings)
    lines = layout.max_lines
    max_chars_per_line = _chars_per_line(layout, settings)
    max_chars = max(8, lines * max_chars_per_line)
    min_chars = max(4, max_chars // 3)

    # Step 1: Merge consecutive short segments into blocks
    merged = _merge_consecutive_segments(
        list(segments),
        max_duration=settings.max_duration_s,
        max_chars=max_chars,
        pause_s=settings.pause_split_s,
        split_on_speaker=settings.split_on_speaker,
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

    # Step 2b: Short punchy cues (subtitles.max_words_per_cue).
    if settings.max_words_per_cue > 0:
        prepared = [
            chunk
            for seg in prepared
            for chunk in _split_by_word_count(seg, settings.max_words_per_cue)
        ]

    # Step 2c: Character counts only approximate the font; make sure every
    # cue really fits max_lines lines of the usable width.
    censor = _censor_for(settings)
    prepared = [
        chunk for seg in prepared for chunk in _split_to_fit(seg, layout, settings, censor)
    ]

    # Step 3: Remove any overlapping blocks
    prepared = _remove_overlaps(prepared)

    # Step 4: Enforce minimum duration and gap (after overlap removal)
    prepared = _enforce_timing_constraints(
        prepared,
        min_duration=settings.min_duration_s,
        max_duration=settings.max_duration_s,
        min_gap=settings.min_gap_s,
    )

    # Step 5: Remove overlaps again (enforce may have created new ones)
    prepared = _remove_overlaps(prepared)

    return prepared


def _chars_per_line(layout: _Layout, settings: SubtitleRenderSettings) -> int:
    """Characters per line: explicit, or how many average glyphs of this font
    fit the usable width (replaces the font-blind 25-per-line guideline)."""

    if settings.max_chars_per_line > 0:
        return settings.max_chars_per_line
    sample = apply_case("Разбираемся, почему этот выпуск вызывает споры", settings.text_case)
    inner = layout.max_width - 2.0 * layout.measurer.outline
    return max(4, int(inner / layout.measurer.average_char_width(sample)))


def _chunk_segment(segment: SubtitleSegment, chunk: Sequence[_TimedSubtitleWord]) -> SubtitleSegment:
    return SubtitleSegment(
        start=round(chunk[0].start, 3),
        end=round(chunk[-1].end, 3),
        text=" ".join(w.text for w in chunk).strip(),
        words=tuple(chunk) if segment.words else (),
        speaker=segment.speaker,
    )


def _split_by_word_count(segment: SubtitleSegment, limit: int) -> list[SubtitleSegment]:
    """At most ``limit`` words per cue; a sentence end also closes a chunk."""

    words = _build_timed_words(segment)
    if len(words) <= limit:
        return [segment]
    chunks: list[list[_TimedSubtitleWord]] = [[]]
    for word in words:
        if len(chunks[-1]) >= limit:
            chunks.append([])
        chunks[-1].append(word)
        if _SENTENCE_END_RE.search(word.text):
            chunks.append([])
    return [_chunk_segment(segment, chunk) for chunk in chunks if chunk]


def _best_split_index(texts: Sequence[str]) -> int:
    """Where to cut a cue in two: near the middle, at a natural point."""

    total = sum(len(t) for t in texts) or 1
    best_index, best_cost = len(texts) // 2, float("inf")
    left = 0
    for index in range(1, len(texts)):
        left += len(texts[index - 1])
        cost = abs(2 * left - total) / total
        prev = texts[index - 1]
        if _SENTENCE_END_RE.search(prev):
            cost -= 0.35
        elif _COMMA_AFTER.search(prev):
            cost -= 0.2
        if prev.lower().strip(".,!?…;:") in NO_LINE_END:
            cost += 0.4
        if cost < best_cost:
            best_index, best_cost = index, cost
    return best_index


def _split_to_fit(
    segment: SubtitleSegment,
    layout: _Layout,
    settings: SubtitleRenderSettings,
    censor: Callable[[str], bool] | None,
    depth: int = 0,
) -> list[SubtitleSegment]:
    words = _build_timed_words(segment)
    shown = [_display_word(w.text, settings, censor) for w in words]
    shown = [t for t in shown if t]
    if len(words) <= 1 or depth >= 8:
        return [segment]
    rows = wrap_words(
        shown,
        layout.measurer,
        max_width=layout.max_width,
        max_lines=layout.max_lines,
        balance=settings.line_balance,
    )
    if len(rows) <= layout.max_lines and lines_fit(rows, layout.measurer, layout.max_width):
        return [segment]
    cut = _best_split_index([w.text for w in words])
    out: list[SubtitleSegment] = []
    for chunk in (words[:cut], words[cut:]):
        if chunk:
            out.extend(_split_to_fit(_chunk_segment(segment, chunk), layout, settings, censor, depth + 1))
    return out


def _merge_consecutive_segments(
    segments: list[SubtitleSegment],
    *,
    max_duration: float,
    max_chars: int,
    pause_s: float = DEFAULT_PAUSE_SPLIT_S,
    split_on_speaker: bool = True,
) -> list[SubtitleSegment]:
    """Merge consecutive short segments into single subtitle blocks.

    A pause of ``pause_s`` or a change of speaker always ends a block: one cue
    must not run two people's words together (BBC: new speaker, new subtitle).
    """
    if not segments:
        return []

    merged: list[SubtitleSegment] = []
    current_text_parts: list[str] = []
    current_words: list[_TimedSubtitleWord] = []
    current_start: float | None = None
    current_end: float = 0.0
    current_speaker = ""

    def flush() -> None:
        if current_text_parts and current_start is not None:
            merged.append(
                SubtitleSegment(
                    start=round(current_start, 3),
                    end=round(current_end, 3),
                    text=" ".join(current_text_parts),
                    # Merging is pure concatenation in time order, so the
                    # word timings stay valid for the combined block.
                    words=tuple(current_words),
                    speaker=current_speaker,
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
            if gap > pause_s:
                should_flush = True
            elif split_on_speaker and seg.speaker != current_speaker:
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
            current_words = []
            current_start = None

        if current_start is None:
            current_start = seg.start
            current_speaker = seg.speaker
        current_text_parts.append(text)
        # If any part of a block lacks word timings the totals will not match
        # its text, and _real_timed_words falls back to interpolation.
        current_words.extend(seg.words)
        current_end = seg.end

    flush()
    return merged


_MIN_TRIMMED_DURATION_S = 0.3


def _remove_overlaps(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
    """Resolve overlapping blocks without moving speech later.

    The earlier block is cut short: its overlap is usually min-duration
    padding, not speech. Only when that would leave it too short to read does
    the later block start a little late — and then only its start moves, never
    its end. Pushing whole blocks forward used to accumulate across a run of
    short cues into seconds of lag.
    """
    if not segments:
        return []

    result: list[SubtitleSegment] = []
    for seg in segments:
        if not result:
            result.append(seg)
            continue
        prev = result[-1]
        if seg.start < prev.end:
            trimmed_end = seg.start - 0.05
            if trimmed_end - prev.start >= _MIN_TRIMMED_DURATION_S:
                result[-1] = replace(prev, end=round(trimmed_end, 3))
            else:
                new_start = prev.end + 0.05
                seg = replace(
                    seg,
                    start=round(new_start, 3),
                    end=round(max(seg.end, new_start + _MIN_TRIMMED_DURATION_S), 3),
                )
        result.append(seg)
    return result


def _enforce_timing_constraints(
    segments: list[SubtitleSegment],
    *,
    min_duration: float = DEFAULT_MIN_DURATION_S,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    min_gap: float = DEFAULT_GAP_BETWEEN_SUBTITLES_S,
) -> list[SubtitleSegment]:
    """Enforce minimum duration and gap between subtitle blocks."""
    if not segments:
        return []

    result: list[SubtitleSegment] = []

    for seg in segments:
        duration = seg.end - seg.start

        if duration < min_duration:
            seg = replace(seg, end=round(seg.start + min_duration, 3))

        if duration > max_duration:
            seg = replace(seg, end=round(seg.start + max_duration, 3))

        if result:
            prev = result[-1]
            gap = seg.start - prev.end
            if gap < min_gap:
                new_prev_end = seg.start - min_gap
                if new_prev_end > prev.start and (new_prev_end - prev.start) >= min_duration:
                    result[-1] = replace(prev, end=round(new_prev_end, 3))

        result.append(seg)

    return result


_SENTENCE_END_RE = _re.compile(r"[.!?…]\s*$")

_NO_LINE_END = NO_LINE_END

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
                # Carry the timings through the split so the sub-segments keep
                # real word timing instead of re-interpolating.
                words=tuple(chunk_words) if segment.words else (),
                speaker=segment.speaker,
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


def _normalize_word(text: str) -> str:
    """Compare-friendly form of a word: letters and digits only, lowercased."""

    return "".join(char for char in text.lower() if char.isalnum())


def _real_timed_words(segment: SubtitleSegment) -> list[_TimedSubtitleWord] | None:
    """The segment's own word timings, if they still match its text.

    The proofread stage rewrites segment text in place while keeping the
    original word list, so the two can legitimately disagree; when they do the
    caller falls back to interpolation rather than mistiming the karaoke.
    """

    if not segment.words:
        return None

    words_text = [word for word in segment.text.split() if word.strip()]
    if len(words_text) != len(segment.words):
        return None

    for expected, actual in zip(words_text, segment.words):
        if _normalize_word(expected) != _normalize_word(actual.text):
            return None

    # Keep the text as rendered, but take the timings from the transcript.
    # The ASS karaoke tag consumes durations back to back, so a word runs
    # until the next one actually starts — that way the highlight advances on
    # the real speech onset instead of drifting through the pauses.
    timed: list[_TimedSubtitleWord] = []
    for index, (expected, actual) in enumerate(zip(words_text, segment.words)):
        is_last = index == len(words_text) - 1
        end = actual.end if is_last else segment.words[index + 1].start
        timed.append(
            _TimedSubtitleWord(
                start=round(actual.start, 3),
                end=round(max(actual.start + 0.01, end), 3),
                text=expected,
            ),
        )
    return timed


def _build_timed_words(segment: SubtitleSegment) -> list[_TimedSubtitleWord]:
    real = _real_timed_words(segment)
    if real is not None:
        return real

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
    align = str(value or default).strip().lower()
    if align == "middle":
        align = "center"
    return align if align in VERTICAL_ALIGNS else default
