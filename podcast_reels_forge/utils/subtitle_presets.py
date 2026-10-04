"""Ready-made subtitle looks.

RU: Готовые стили субтитров.

A preset is a complete caption template in the spirit of pycaps' templates:
the look (the ``Default`` ASS style, plus an optional ``Highlight`` style for
the active word) and the render settings that go with it (highlight mode,
letter case, words per cue...). Looks are described in the style editor's own
terms (hex colour + opacity, outline, margins), so the GUI and the burner share
one definition: ``gui/assets/subtitle-presets.js`` is generated from this
module (``python3 -m podcast_reels_forge.utils.subtitle_presets``) and a test
keeps the two in sync.

Sources the looks are modelled on (see docs/COMPETITOR_REVIEW.md):

* nicolaigaina/ai-video-captions — Hormozi, MrBeast, Karaoke, Bounce;
* francozanardi/pycaps — word-focus, explosive (neon), vibrant, minimalist,
  retro-gaming, fast (one word);
* WEIFENG2333/VideoCaptioner — the padded box ("rounded" mode);
* the TikTok/Reels native caption looks (box, sticker).

Render settings in a preset are defaults: anything set explicitly under
``subtitles:`` in config.yaml wins.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# PlayRes the looks are designed for (9:16 reels).
PLAY_RES_X = 1080
PLAY_RES_Y = 1920
# Font sizes, outlines and shadows below are tuned at this subtitles.font_size_px;
# another value scales every preset proportionally.
REFERENCE_FONT_SIZE = 96

ASS_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
    "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
    "MarginR, MarginV, Encoding"
)

# Style-editor fields every look starts from: the shipped "forge" look.
LOOK_DEFAULTS: dict[str, Any] = {
    "fontPath": "assets/fonts/bignoodletoooblique.ttf",
    "fontSizePx": 96, "spacingPx": 0,
    "bold": True, "italic": False, "underline": False, "strikeout": False,
    # PrimaryColour = spoken / active, SecondaryColour = not yet spoken.
    "primaryColor": "#FFD60A", "primaryOp": 1.0,
    "secondaryColor": "#FFFFFF", "secondaryOp": 1.0,
    "borderStyle": 1,
    "outlineColor": "#000000", "outlineOp": 1.0, "outline": 8,
    "backColor": "#000000", "backOp": 0.5, "shadow": 0,
    # 140px side margins clear the Reels/TikTok/Shorts action rail; MarginV
    # 470 lifts the text above the caption/handle/audio chrome.
    "alignment": 2, "marginV": 470, "marginL": 140, "marginR": 140,
    "scaleX": 100, "scaleY": 100, "angle": 0,
    # Optional "Highlight" style for the active word (subtitles.highlight).
    "hlEnabled": False,
    "hlColor": "#FFD60A", "hlOp": 1.0,
    "hlBorderStyle": 1,
    "hlOutlineColor": "#000000", "hlOutlineOp": 1.0, "hlOutline": 8,
    "hlBackColor": "#000000", "hlBackOp": 0.5, "hlShadow": 0,
    "hlScale": 100,
}

PRESETS: dict[str, dict[str, Any]] = {
    "forge": {
        "title": {"ru": "Forge (по умолчанию)", "en": "Forge (default)"},
        "description": {
            "ru": "Янтарный BigNoodle с толстым чёрным контуром, реплика целиком.",
            "en": "Amber BigNoodle with a heavy black outline; whole cue at once.",
        },
        "look": {},
        "render": {"highlight": "none", "text_case": "none"},
    },
    "hormozi": {
        "title": {"ru": "Hormozi", "en": "Hormozi"},
        "description": {
            "ru": "Белый Montserrat Black капсом, активное слово жёлтое; 2–4 слова на экране.",
            "en": "White Montserrat Black caps, active word in yellow; 2–4 words on screen.",
        },
        "look": {
            "fontPath": "assets/fonts/MontserratBlack.ttf", "fontSizePx": 80, "bold": False,
            "primaryColor": "#FFFFFF", "secondaryColor": "#FFFFFF",
            "outline": 6, "shadow": 4, "backColor": "#000000", "backOp": 0.5,
            "hlEnabled": True, "hlColor": "#FFE600", "hlOutline": 6, "hlShadow": 4,
        },
        "render": {"highlight": "word", "text_case": "upper", "max_words_per_cue": 4,
                   "strip_punctuation": "periods"},
    },
    "mrbeast": {
        "title": {"ru": "MrBeast", "en": "MrBeast"},
        "description": {
            "ru": "Жёлтый Russo One, оранжевое слово «подпрыгивает»; очень толстый контур.",
            "en": "Yellow Russo One, the orange active word pops; extra-thick outline.",
        },
        "look": {
            "fontPath": "assets/fonts/RussoOne-Regular.ttf", "fontSizePx": 92, "bold": False,
            "primaryColor": "#FFFF00", "secondaryColor": "#FFFF00",
            "outline": 9, "shadow": 5, "backColor": "#000000", "backOp": 1.0,
            "hlEnabled": True, "hlColor": "#FF6600", "hlOutline": 9, "hlShadow": 5,
            "hlBackOp": 1.0, "hlScale": 110,
        },
        "render": {"highlight": "pop", "text_case": "upper", "max_words_per_cue": 3,
                   "strip_punctuation": "periods"},
    },
    "karaoke": {
        "title": {"ru": "Караоке", "en": "Karaoke"},
        "description": {
            "ru": "Слова плавно заливаются синим по мере произнесения (\\kf).",
            "en": "Words fill with blue as they are spoken (\\kf sweep).",
        },
        "look": {
            "fontPath": "assets/fonts/MontserratBlack.ttf", "fontSizePx": 76, "bold": False,
            "primaryColor": "#2B9BFF", "secondaryColor": "#FFFFFF",
            "outline": 5, "shadow": 3, "backOp": 0.5,
        },
        "render": {"highlight": "karaoke", "text_case": "upper"},
    },
    "tiktok": {
        "title": {"ru": "TikTok", "en": "TikTok"},
        "description": {
            "ru": "Белый текст с тонким контуром, активное слово бирюзовое.",
            "en": "White text with a thin outline, active word in TikTok cyan.",
        },
        "look": {
            "fontPath": "assets/fonts/MontserratBlack.ttf", "fontSizePx": 70, "bold": False,
            "primaryColor": "#FFFFFF", "secondaryColor": "#FFFFFF",
            "outline": 4, "shadow": 2, "backOp": 0.6, "marginV": 560,
            "hlEnabled": True, "hlColor": "#25F4EE", "hlOutline": 4, "hlShadow": 2,
            "hlBackOp": 0.6,
        },
        "render": {"highlight": "word", "text_case": "none"},
    },
    "box": {
        "title": {"ru": "Плашка", "en": "Box"},
        "description": {
            "ru": "Белый текст на одной полупрозрачной тёмной плашке с отступами.",
            "en": "White text on a single padded, translucent dark box.",
        },
        "look": {
            "fontPath": "assets/fonts/OswaldBold.ttf", "fontSizePx": 80, "bold": False,
            "primaryColor": "#FFFFFF", "secondaryColor": "#FFFFFF",
            # BorderStyle 4 (libass >= 0.17): one box per cue; Outline is its
            # padding, so the outline colour is made fully transparent.
            "borderStyle": 4, "outline": 18, "outlineOp": 0.0,
            "backColor": "#000000", "backOp": 0.75, "shadow": 0,
        },
        "render": {"highlight": "none", "text_case": "none", "line_balance": "bottom_heavy"},
    },
    "sticker": {
        "title": {"ru": "Стикер", "en": "Sticker"},
        "description": {
            "ru": "Чёрный текст на белой «наклейке» со скруглёнными краями.",
            "en": "Black text on a white rounded sticker.",
        },
        "look": {
            "fontPath": "assets/fonts/RussoOne-Regular.ttf", "fontSizePx": 78, "bold": False,
            "primaryColor": "#111111", "secondaryColor": "#111111",
            # A very thick outline in the box colour reads as a rounded pill.
            "outline": 20, "outlineColor": "#FFFFFF", "shadow": 0,
        },
        "render": {"highlight": "none", "text_case": "upper", "strip_punctuation": "periods"},
    },
    "word_box": {
        "title": {"ru": "Слово в плашке", "en": "Word box"},
        "description": {
            "ru": "Активное слово выделено оранжевой плашкой (pycaps word-focus).",
            "en": "The active word sits on an orange box (pycaps word-focus).",
        },
        "look": {
            "fontPath": "assets/fonts/MontserratBlack.ttf", "fontSizePx": 74, "bold": False,
            "primaryColor": "#FFFFFF", "secondaryColor": "#FFFFFF",
            "outline": 6, "shadow": 0,
            "hlEnabled": True, "hlColor": "#FFFFFF", "hlBorderStyle": 3,
            "hlOutline": 10, "hlOutlineColor": "#F76F00", "hlBackColor": "#F76F00",
            "hlBackOp": 1.0,
        },
        "render": {"highlight": "word", "text_case": "upper", "strip_punctuation": "periods"},
    },
    "neon": {
        "title": {"ru": "Неон", "en": "Neon"},
        "description": {
            "ru": "Жёлтый текст с оранжевым свечением, активное слово белое с красным ореолом.",
            "en": "Yellow text with an orange glow; the active word glows white-red.",
        },
        "look": {
            "fontPath": "assets/fonts/RussoOne-Regular.ttf", "fontSizePx": 80, "bold": False,
            "primaryColor": "#FFDD00", "secondaryColor": "#FFDD00",
            "outline": 5, "outlineColor": "#FF8800", "shadow": 0,
            "hlEnabled": True, "hlColor": "#FFFFFF", "hlOutline": 6,
            "hlOutlineColor": "#FF1A00",
        },
        "render": {"highlight": "word", "text_case": "upper", "blur": 6},
    },
    "vibrant": {
        "title": {"ru": "Вайб", "en": "Vibrant"},
        "description": {
            "ru": "Непрозвучавшие слова полупрозрачные, сказанные — яркие с розовой тенью.",
            "en": "Upcoming words are faded; spoken ones turn solid with a magenta shadow.",
        },
        "look": {
            "fontPath": "assets/fonts/UnboundedBlack.ttf", "fontSizePx": 62, "bold": False,
            "primaryColor": "#FFFFFF", "primaryOp": 0.45,
            "secondaryColor": "#FFFFFF", "secondaryOp": 0.45,
            "outline": 2, "outlineOp": 0.45, "shadow": 0,
            "hlEnabled": True, "hlColor": "#FFFFFF", "hlOutline": 3,
            "hlShadow": 4, "hlBackColor": "#FF00FF", "hlBackOp": 1.0,
        },
        "render": {"highlight": "fill", "text_case": "upper", "strip_punctuation": "periods"},
    },
    "minimal": {
        "title": {"ru": "Минимал", "en": "Minimal"},
        "description": {
            "ru": "Спокойный Oswald без капса, тонкий контур и мягкая тень, длинное появление.",
            "en": "Calm Oswald in sentence case, thin outline, soft shadow, slow fades.",
        },
        "look": {
            "fontPath": "assets/fonts/OswaldBold.ttf", "fontSizePx": 66, "bold": False,
            "primaryColor": "#FFFFFF", "primaryOp": 0.95,
            "secondaryColor": "#FFFFFF", "secondaryOp": 0.95,
            "outline": 2, "outlineOp": 0.6, "shadow": 2, "backOp": 0.5,
        },
        "render": {"highlight": "none", "text_case": "none", "strip_punctuation": "periods",
                   "fade_in_duration": 0.3, "fade_out_duration": 0.3,
                   "line_balance": "bottom_heavy"},
    },
    "classic": {
        "title": {"ru": "Классика", "en": "Classic"},
        "description": {
            "ru": "Как в кино: белый текст, контур и тень, «пирамида» строк (BBC/Netflix).",
            "en": "Broadcast style: white text, outline and shadow, bottom-heavy lines.",
        },
        "look": {
            "fontPath": "assets/fonts/OswaldBold.ttf", "fontSizePx": 72, "bold": False,
            "primaryColor": "#FFFFFF", "secondaryColor": "#FFFFFF",
            "outline": 3, "shadow": 2, "backOp": 0.7,
        },
        "render": {"highlight": "none", "text_case": "none", "line_balance": "bottom_heavy",
                   "fade_in_duration": 0.0, "fade_out_duration": 0.0},
    },
    "one_word": {
        "title": {"ru": "По слову", "en": "One word"},
        "description": {
            "ru": "Одно крупное слово в центре кадра — быстрый темп (pycaps fast).",
            "en": "One big word at a time in the middle of the frame (pycaps fast).",
        },
        "look": {
            "fontSizePx": 150, "primaryColor": "#FFFFFF", "secondaryColor": "#FFFFFF",
            "outline": 10, "alignment": 5, "marginV": 0,
        },
        "render": {"highlight": "none", "text_case": "upper", "max_words_per_cue": 1,
                   "strip_punctuation": "all", "fade_in_duration": 0.0,
                   "fade_out_duration": 0.0, "min_duration_s": 0.0},
    },
    "retro": {
        "title": {"ru": "Ретро-игра", "en": "Retro game"},
        "description": {
            "ru": "Пиксельный шрифт на синей плашке, слова печатаются по мере речи.",
            "en": "Pixel font on a navy box; words type in as they are spoken.",
        },
        "look": {
            "fontPath": "assets/fonts/PressStart2P-Regular.ttf", "fontSizePx": 46,
            "bold": False, "primaryColor": "#E0E0E0", "secondaryColor": "#E0E0E0",
            "borderStyle": 4, "outline": 18, "outlineOp": 0.0,
            "backColor": "#141450", "backOp": 0.85, "shadow": 0,
            "hlEnabled": True, "hlColor": "#FFFF88", "hlBorderStyle": 4,
            "hlOutline": 18, "hlOutlineOp": 0.0, "hlBackColor": "#141450", "hlBackOp": 0.85,
        },
        "render": {"highlight": "reveal", "text_case": "none", "max_lines": 3},
    },
    "bold_pop": {
        "title": {"ru": "Жирный поп", "en": "Bold pop"},
        "description": {
            "ru": "Широкий Unbounded капсом, зелёное слово увеличивается в такт.",
            "en": "Wide Unbounded caps; the green active word pops in time.",
        },
        "look": {
            "fontPath": "assets/fonts/UnboundedBlack.ttf", "fontSizePx": 64, "bold": False,
            "primaryColor": "#FFFFFF", "secondaryColor": "#FFFFFF",
            "outline": 6, "shadow": 0,
            "hlEnabled": True, "hlColor": "#39FF14", "hlOutline": 6, "hlScale": 112,
        },
        "render": {"highlight": "pop", "text_case": "upper", "max_words_per_cue": 3,
                   "strip_punctuation": "periods"},
    },
    "headline": {
        "title": {"ru": "Заголовок", "en": "Headline"},
        "description": {
            "ru": "Жёлтый Rubik Mono One, сказанные слова становятся белыми.",
            "en": "Yellow Rubik Mono One; spoken words turn white.",
        },
        "look": {
            "fontPath": "assets/fonts/RubikMonoOne-Regular.ttf", "fontSizePx": 58,
            "bold": False, "primaryColor": "#FFD60A", "secondaryColor": "#FFD60A",
            "outline": 7, "shadow": 0,
            "hlEnabled": True, "hlColor": "#FFFFFF", "hlOutline": 7,
        },
        "render": {"highlight": "fill", "text_case": "upper", "strip_punctuation": "periods"},
    },
}

DEFAULT_PRESET = "forge"


def preset_names() -> list[str]:
    return list(PRESETS)


def get_preset(name: object) -> dict[str, Any] | None:
    key = str(name or "").strip().lower().replace("-", "_")
    return PRESETS.get(key)


def preset_look(name: object) -> dict[str, Any]:
    preset = get_preset(name) or PRESETS[DEFAULT_PRESET]
    return {**LOOK_DEFAULTS, **preset["look"]}


def preset_render(name: object) -> dict[str, Any]:
    preset = get_preset(name)
    return dict(preset["render"]) if preset else {}


# --- ASS conversion -----------------------------------------------------------


def ass_colour(hex_colour: str, opacity: float = 1.0) -> str:
    """``#RRGGBB`` + opacity → ``&HAABBGGRR`` (ASS alpha is transparency)."""

    value = str(hex_colour or "#FFFFFF").lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    value = (value + "000000")[:6].upper()
    red, green, blue = value[0:2], value[2:4], value[4:6]
    alpha = round((1.0 - max(0.0, min(1.0, float(opacity)))) * 255)
    return f"&H{alpha:02X}{blue}{green}{red}"


def ass_override_colour(hex_colour: str) -> str:
    """``#RRGGBB`` → ``&HBBGGRR&`` for ``\\1c``-style override tags."""

    return "&H" + ass_colour(hex_colour)[4:] + "&" if hex_colour else ""


def _num(value: float) -> str:
    rounded = round(float(value), 2)
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)


def scale_look(look: Mapping[str, Any], font_size_px: float) -> dict[str, Any]:
    """Scale a look designed at :data:`REFERENCE_FONT_SIZE` to another base size."""

    factor = float(font_size_px) / REFERENCE_FONT_SIZE
    if abs(factor - 1.0) < 1e-6:
        return dict(look)
    scaled = dict(look)
    for key in ("fontSizePx", "outline", "shadow", "hlOutline", "hlShadow", "spacingPx"):
        if key in scaled:
            scaled[key] = round(float(scaled[key]) * factor, 1)
    return scaled


def style_line(name: str, look: Mapping[str, Any], font_name: str) -> str:
    fields = (
        name, font_name, _num(look["fontSizePx"]),
        ass_colour(look["primaryColor"], look["primaryOp"]),
        ass_colour(look["secondaryColor"], look["secondaryOp"]),
        ass_colour(look["outlineColor"], look["outlineOp"]),
        ass_colour(look["backColor"], look["backOp"]),
        -1 if look["bold"] else 0, -1 if look["italic"] else 0,
        -1 if look["underline"] else 0, -1 if look["strikeout"] else 0,
        _num(look["scaleX"]), _num(look["scaleY"]), _num(look["spacingPx"]),
        _num(look["angle"]), int(look["borderStyle"]),
        _num(look["outline"]), _num(look["shadow"]), int(look["alignment"]),
        int(look["marginL"]), int(look["marginR"]), int(look["marginV"]), 1,
    )
    return "Style: " + ",".join(str(f) for f in fields)


def highlight_look(look: Mapping[str, Any]) -> dict[str, Any]:
    """The ``Highlight`` style: the Default look with the active-word overrides."""

    scale = float(look.get("hlScale", 100)) / 100.0
    return {
        **look,
        "primaryColor": look["hlColor"], "primaryOp": look["hlOp"],
        "secondaryColor": look["hlColor"], "secondaryOp": look["hlOp"],
        "borderStyle": look["hlBorderStyle"],
        "outlineColor": look["hlOutlineColor"], "outlineOp": look["hlOutlineOp"],
        "outline": look["hlOutline"],
        "backColor": look["hlBackColor"], "backOp": look["hlBackOp"],
        "shadow": look["hlShadow"],
        "scaleX": float(look["scaleX"]) * scale,
        "scaleY": float(look["scaleY"]) * scale,
    }


def ass_header(look: Mapping[str, Any], font_name: str) -> str:
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {PLAY_RES_X}",
        f"PlayResY: {PLAY_RES_Y}",
        # We insert the line breaks ourselves (utils/subtitle_layout.py);
        # libass only wraps as a safety net if a line still overflows.
        "WrapStyle: 0",
        # Outline and shadow scale with the frame like the text does.
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        ASS_STYLE_FORMAT,
        style_line("Default", look, font_name),
    ]
    if look.get("hlEnabled"):
        lines.append(style_line("Highlight", highlight_look(look), font_name))
    return "\n".join(lines)


# --- GUI export -----------------------------------------------------------------

JS_PATH = Path("gui/assets/subtitle-presets.js")


def font_metrics(fonts_dir: Path | None = None) -> dict[str, float]:
    """``assets/fonts/X.ttf`` → (winAscent + winDescent) / unitsPerEm.

    libass sizes a face by its OS/2 win metrics, while browsers expose only
    hhea/typo metrics, which differ for fonts like Montserrat (1.56 vs 1.22
    em). The GUI preview needs this ratio to draw text at the burned size.
    """

    from fontTools.ttLib import TTFont  # noqa: PLC0415

    root = Path(__file__).resolve().parents[2]
    fonts_dir = fonts_dir or root / "assets/fonts"
    metrics: dict[str, float] = {}
    for path in sorted(fonts_dir.glob("*.[ot]tf")):
        face = TTFont(path, lazy=True)
        os2 = face["OS/2"]
        span = int(os2.usWinAscent) + int(os2.usWinDescent)
        if span > 0:
            key = path.relative_to(root).as_posix()
            metrics[key] = round(span / int(face["head"].unitsPerEm), 4)
    return metrics


def presets_js() -> str:
    payload = {
        "lookDefaults": LOOK_DEFAULTS,
        "presets": PRESETS,
        "fontMetrics": font_metrics(),
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        "// Generated by `python3 -m podcast_reels_forge.utils.subtitle_presets`.\n"
        "// Do not edit by hand: change podcast_reels_forge/utils/subtitle_presets.py.\n"
        f"window.FORGE_SUBTITLE_PRESETS = {body};\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    target = Path(args[0]) if args else Path(__file__).resolve().parents[2] / JS_PATH
    target.write_text(presets_js(), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
