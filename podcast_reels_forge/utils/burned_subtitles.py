"""Helpers for rendering burned-in subtitles with pycaps."""

from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from podcast_reels_forge.utils.reel_markdown import reel_index_from_path

LOG = logging.getLogger(__name__)

DEFAULT_SUBTITLE_FONT = Path("assets/fonts/bignoodletoooblique.ttf")
DEFAULT_FONT_SIZE_PX = 72
DEFAULT_MAX_LINES = 3
DEFAULT_MAX_WIDTH_RATIO = 0.82
DEFAULT_VERTICAL_ALIGN = "bottom"
DEFAULT_VERTICAL_OFFSET = -0.08
_PYCAPS_TEMPLATE_DIRNAME = ".pycaps_template"


@dataclass(frozen=True)
class SubtitleSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SubtitleRenderSettings:
    enabled: bool
    font_path: Path
    font_size_px: int = DEFAULT_FONT_SIZE_PX
    max_lines: int = DEFAULT_MAX_LINES
    max_width_ratio: float = DEFAULT_MAX_WIDTH_RATIO
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
    font_path = Path(font_value) if font_value else DEFAULT_SUBTITLE_FONT
    if not font_path.is_absolute():
        font_path = repo_dir / font_path

    return SubtitleRenderSettings(
        enabled=enabled,
        font_path=font_path.resolve(),
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

    start = _coerce_float(moment.get("start"), default=0.0)
    end = _coerce_float(moment.get("end"), default=0.0)
    if end <= start:
        raise ValueError(f"Invalid reel boundaries for {reel_path.name}: start={start} end={end}")

    transcript_segments = load_transcript_segments(transcript_json_path)
    clip_segments = slice_segments_for_clip(
        transcript_segments,
        clip_start=max(0.0, start - float(padding)),
        clip_end=end + float(padding),
    )
    if not clip_segments:
        LOG.warning(
            "No transcript segments overlap %s; leaving video without burned subtitles",
            reel_path.name,
        )
        return None

    srt_path = reel_path.with_suffix(".srt")
    write_srt_file(srt_path, clip_segments)

    template_config = prepare_pycaps_template(
        reel_path.parent / _PYCAPS_TEMPLATE_DIRNAME,
        settings=settings,
    )

    tmp_output = reel_path.with_name(f"{reel_path.stem}.subtitled{reel_path.suffix}")
    if tmp_output.exists():
        tmp_output.unlink(missing_ok=True)

    cmd = [
        *_pycaps_base_command(),
        "render",
        "--input",
        str(reel_path),
        "--config",
        str(template_config),
        "--transcript",
        str(srt_path),
        "--transcript-format",
        "srt",
        "--output",
        str(tmp_output),
    ]
    if verbose:
        cmd.append("--verbose")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tmp_output.unlink(missing_ok=True)
        raise RuntimeError(_format_pycaps_error(result))
    tmp_output.replace(reel_path)
    return srt_path


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
    for reel_path in reel_files:
        index = reel_index_from_path(reel_path)
        if index is None:
            continue
        moment_index = index - 1
        if moment_index < 0 or moment_index >= len(moments):
            continue
        srt_path = ensure_reel_burned_subtitles(
            moments[moment_index],
            reel_path,
            transcript_json_path=transcript_json_path,
            padding=padding,
            settings=settings,
            verbose=verbose,
        )
        if srt_path is not None:
            written.append(srt_path)
    return written


def load_transcript_segments(path: Path) -> list[SubtitleSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = data.get("segments", []) if isinstance(data, dict) else []
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
        _build_styles_css(
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
    return template_dir / "pycaps.template.json"


def _build_template_config(settings: SubtitleRenderSettings) -> dict[str, Any]:
    return {
        "css": "styles.css",
        "resources": "resources",
        "layout": {
            "x_words_space": 8,
            "y_words_space": 18,
            "max_width_ratio": settings.max_width_ratio,
            "max_number_of_lines": settings.max_lines,
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


def _build_styles_css(*, font_filename: str, font_size_px: int, font_format: str) -> str:
    return f"""@font-face {{
    font-family: 'ForgeSubtitleFont';
    src: url('{font_filename}') format('{font_format}');
}}

.word {{
    font-family: 'ForgeSubtitleFont', 'Impact', sans-serif;
    font-size: {int(font_size_px)}px;
    line-height: 1.0;
    color: #ffffff;
    font-weight: 700;
    background-color: rgba(0, 0, 0, 0.58);
    border-radius: 12px;
    padding: 8px 12px;
    text-shadow:
        -2px -2px 0 rgba(0, 0, 0, 0.92),
         2px -2px 0 rgba(0, 0, 0, 0.92),
        -2px  2px 0 rgba(0, 0, 0, 0.92),
         2px  2px 0 rgba(0, 0, 0, 0.92),
         0    4px 10px rgba(0, 0, 0, 0.7);
}}

.word-being-narrated {{
    color: #ffd54a;
    background-color: rgba(18, 18, 18, 0.82);
}}

.word-already-narrated {{
    color: #ffffff;
}}

.first-word-in-line {{
    border-top-left-radius: 14px;
    border-bottom-left-radius: 14px;
}}

.last-word-in-line {{
    border-top-right-radius: 14px;
    border-bottom-right-radius: 14px;
}}
"""


def _pycaps_base_command() -> list[str]:
    pycaps_cli = shutil.which("pycaps")
    if pycaps_cli:
        return [pycaps_cli]
    if importlib.util.find_spec("pycaps.cli") is not None:
        return [sys.executable, "-c", "from pycaps.cli import app; app()"]
    raise RuntimeError(
        "pycaps is not installed. Install project requirements and then run "
        "`playwright install chromium`.",
    )


def _format_pycaps_error(result: subprocess.CompletedProcess[str]) -> str:
    detail = ((result.stderr or "").strip() or (result.stdout or "").strip() or "unknown pycaps error")
    if "playwright install chromium" in detail.lower() or "chromium browser is not installed" in detail.lower():
        return (
            "pycaps could not launch Chromium. Run `playwright install chromium` "
            f"and retry. Details: {detail[-800:]}"
        )
    if "no module named pycaps" in detail.lower():
        return (
            "pycaps is not installed in the current environment. Install project "
            f"requirements first. Details: {detail[-800:]}"
        )
    return f"pycaps failed to burn subtitles: {detail[-800:]}"


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000.0)))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    secs = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _font_format_for_suffix(suffix: str) -> str:
    return {
        ".ttf": "truetype",
        ".otf": "opentype",
        ".woff": "woff",
        ".woff2": "woff2",
    }.get(suffix.lower(), "truetype")


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
        out = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        out = int(default)
    return max(out, minimum)


def _coerce_align(value: object, *, default: str) -> str:
    allowed = {"top", "center", "bottom"}
    align = str(value or default).strip().lower()
    return align if align in allowed else default
