"""Tests for subtitle line layout, text transforms and the built-in presets."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_reels_forge.utils import subtitle_presets as presets
from podcast_reels_forge.utils.subtitle_layout import (
    TextMeasurer,
    apply_case,
    bare_word,
    build_censor_matcher,
    censor_word,
    lines_fit,
    strip_punctuation,
    wrap_words,
)

REPO = Path(__file__).resolve().parents[1]
FONT = REPO / "assets/fonts/bignoodletoooblique.ttf"


def _measurer(font: Path | None = FONT, size: float = 96, outline: float = 8) -> TextMeasurer:
    return TextMeasurer(font, size, outline=outline)


def test_measurer_matches_libass_scale() -> None:
    """libass sizes a face by ascender+descender; measured 723px in a libass render."""
    m = _measurer(outline=0)
    assert m.width("РАЗБИРАЕМСЯ ПОЧЕМУ ЭТОТ") == pytest.approx(723, abs=6)


def test_measurer_falls_back_without_a_font(tmp_path: Path) -> None:
    m = _measurer(tmp_path / "missing.ttf", outline=0)
    assert m.width("абв") == pytest.approx(3 * 96 * 0.45)


def test_short_cue_stays_on_one_line() -> None:
    assert wrap_words("Да, конечно".split(), _measurer(), max_width=800, max_lines=2) == [["Да,", "конечно"]]


def test_balanced_lines_beat_greedy_tail() -> None:
    words = "Мы долго спорили о том кто прав но так и не договорились".split()
    m = _measurer()
    greedy = wrap_words(words, m, max_width=800, max_lines=3, balance="greedy")
    balanced = wrap_words(words, m, max_width=800, max_lines=3, balance="balanced")
    assert greedy[-1] == ["договорились"], "greedy leaves a one-word tail"
    assert len(balanced[-1]) > 1
    widths = [m.width(" ".join(line)) for line in balanced]
    assert max(widths) - min(widths) < 120
    assert lines_fit(balanced, m, 800)


def test_line_never_ends_on_a_preposition() -> None:
    words = "Я сегодня утром поехал на работу через весь город".split()
    for balance in ("balanced", "bottom_heavy", "top_heavy"):
        rows = wrap_words(words, _measurer(), max_width=800, max_lines=2, balance=balance)
        assert len(rows) == 2
        assert bare_word(rows[0][-1]) not in {"на", "через"}


def test_breaks_prefer_punctuation() -> None:
    words = "Вернулись очень поздно, сразу легли спать".split()
    rows = wrap_words(words, _measurer(), max_width=800, max_lines=2)
    assert rows[0][-1] == "поздно,"


def test_bottom_heavy_never_widens_the_top_line() -> None:
    words = "Это первая строка и чуть более длинная вторая строка".split()
    m = _measurer()
    rows = wrap_words(words, m, max_width=800, max_lines=2, balance="bottom_heavy")
    assert m.width(" ".join(rows[0])) <= m.width(" ".join(rows[1])) + 1


def test_text_transforms() -> None:
    assert apply_case("из-за", "title") == "Из-За"
    assert apply_case("Да", "upper") == "ДА"
    assert strip_punctuation("слово,", "periods") == "слово"
    assert strip_punctuation("что?", "periods") == "что?"
    assert strip_punctuation("ну...", "periods") == "ну..."
    assert strip_punctuation("«Да»,", "all") == "Да"
    assert strip_punctuation("из-за", "all") == "из-за"
    assert strip_punctuation("—", "all") == ""


def test_censor_matches_whole_words_and_prefixes() -> None:
    matches = build_censor_matcher(["бля*", "хрен"])
    assert matches("Блять,")
    assert matches("хрен")
    assert not matches("хрена")
    assert censor_word("Блять,", "middle") == "Б***ь,"
    assert censor_word("хрен", "whole") == "****"
    assert censor_word("хрен", "first") == "х***"


def test_every_preset_builds_a_valid_header() -> None:
    for name in presets.preset_names():
        look = presets.preset_look(name)
        header = presets.ass_header(look, "Font")
        assert "Style: Default,Font," in header
        assert ("Style: Highlight," in header) == bool(look["hlEnabled"])
        font = REPO / look["fontPath"]
        assert font.exists(), f"{name}: font {font} is missing"
        render = presets.preset_render(name)
        assert render.get("highlight", "none") in (
            "none", "karaoke", "word", "fill", "reveal", "pop",
        )


def test_ass_colour_conversion() -> None:
    assert presets.ass_colour("#FFD60A") == "&H000AD6FF"
    assert presets.ass_colour("#000000", 0.5) == "&H80000000"
    assert presets.ass_override_colour("#FF0000") == "&H0000FF&"


def test_scale_look_scales_size_and_borders() -> None:
    look = presets.preset_look("hormozi")
    scaled = presets.scale_look(look, presets.REFERENCE_FONT_SIZE * 1.5)
    assert scaled["fontSizePx"] == pytest.approx(look["fontSizePx"] * 1.5)
    assert scaled["outline"] == pytest.approx(look["outline"] * 1.5)


def test_gui_preset_file_is_in_sync() -> None:
    """gui/assets/subtitle-presets.js is generated from subtitle_presets.py."""
    js = (REPO / presets.JS_PATH).read_text(encoding="utf-8")
    assert js == presets.presets_js(), (
        "run `python3 -m podcast_reels_forge.utils.subtitle_presets` to regenerate"
    )
