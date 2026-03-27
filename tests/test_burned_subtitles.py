"""Tests for pycaps-based burned subtitle helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import podcast_reels_forge.utils.burned_subtitles as bs

if TYPE_CHECKING:
    import pytest

    MonkeyPatch = pytest.MonkeyPatch


def test_slice_segments_for_clip_rebases_and_clips() -> None:
    segments = [
        bs.SubtitleSegment(start=3.0, end=6.0, text="before"),
        bs.SubtitleSegment(start=8.0, end=12.0, text="inside"),
        bs.SubtitleSegment(start=14.5, end=18.0, text="tail"),
    ]

    clipped = bs.slice_segments_for_clip(segments, clip_start=5.0, clip_end=15.0)

    assert clipped == [
        bs.SubtitleSegment(start=0.0, end=1.0, text="before"),
        bs.SubtitleSegment(start=3.0, end=7.0, text="inside"),
        bs.SubtitleSegment(start=9.5, end=10.0, text="tail"),
    ]


def test_subtitle_settings_defaults_are_conservative(tmp_path: Path) -> None:
    settings = bs.subtitle_settings_from_conf(None, repo_dir=tmp_path)

    assert settings.font_size_px == 44
    assert settings.max_lines == 2
    assert settings.max_width_ratio == 0.9
    assert settings.vertical_align == "bottom"
    assert settings.vertical_offset == 0.0
    assert settings.css_path == (tmp_path / bs.DEFAULT_SUBTITLE_CSS_TEMPLATE).resolve()


def test_subtitle_settings_from_conf_resolves_css_path(tmp_path: Path) -> None:
    settings = bs.subtitle_settings_from_conf(
        {
            "subtitles": {
                "enabled": True,
                "font": "assets/fonts/custom.ttf",
                "css": "assets/subtitles/custom.css",
            },
        },
        repo_dir=tmp_path,
    )

    assert settings.enabled is True
    assert settings.font_path == (tmp_path / "assets/fonts/custom.ttf").resolve()
    assert settings.css_path == (tmp_path / "assets/subtitles/custom.css").resolve()


def test_prepare_pycaps_template_uses_external_css_template(tmp_path: Path) -> None:
    font_path = tmp_path / "font.ttf"
    font_path.write_text("font", encoding="utf-8")

    css_template = tmp_path / "subtitles.css"
    css_template.write_text(
        """/* custom template marker */
@font-face {
    font-family: 'ForgeSubtitleFont';
    src: url('__FONT_FILENAME__') format('__FONT_FORMAT__');
}

.word {
    font-size: __FONT_SIZE_PX__px;
}
""",
        encoding="utf-8",
    )

    template_dir = tmp_path / "template"
    out_dir = bs.prepare_pycaps_template(
        template_dir,
        settings=bs.SubtitleRenderSettings(
            enabled=True,
            font_path=font_path,
            css_path=css_template,
            font_size_px=37,
        ),
    )

    styles = (out_dir / "styles.css").read_text(encoding="utf-8")
    assert "custom template marker" in styles
    assert "src: url('subtitle_font.ttf') format('truetype');" in styles
    assert "--subtitle-font-size: 37px;" in styles
    assert "padding: var(--subtitle-padding-y) var(--subtitle-padding-x);" in styles


def test_ensure_reel_burned_subtitles_writes_srt_and_runs_pycaps(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    reel_path = tmp_path / "reel_01.mp4"
    reel_path.write_text("mp4")

    font_path = tmp_path / "font.ttf"
    font_path.write_text("font")

    transcript_path = tmp_path / "video.json"
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "Intro"},
                    {"start": 9.5, "end": 12.5, "text": "Key moment"},
                    {"start": 12.5, "end": 14.0, "text": "Closing words"},
                ],
            },
        ),
        encoding="utf-8",
    )

    commands: list[list[str]] = []

    def fake_run(
        cmd: list[str] | tuple[str, ...],
        *,
        capture_output: bool = False,
        text: bool = False,
        check: bool = False,
    ) -> SimpleNamespace:
        cmd_list = [str(part) for part in cmd]
        commands.append(cmd_list)
        Path(cmd_list[-1]).write_text("subtitled")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bs.shutil, "which", lambda name: "/usr/bin/pycaps" if name == "pycaps" else None)
    monkeypatch.setattr(bs.subprocess, "run", fake_run)

    srt_path = bs.ensure_reel_burned_subtitles(
        {"start": 10.0, "end": 13.0},
        reel_path,
        transcript_json_path=transcript_path,
        padding=1.0,
        settings=bs.SubtitleRenderSettings(
            enabled=True,
            font_path=font_path,
            css_path=(Path(__file__).resolve().parents[1] / bs.DEFAULT_SUBTITLE_CSS_TEMPLATE).resolve(),
        ),
    )

    assert srt_path == reel_path.with_suffix(".srt")
    assert srt_path.exists()
    assert "00:00:00,500 --> 00:00:03,500" in srt_path.read_text(encoding="utf-8")
    assert commands
    assert commands[0][0] == "/usr/bin/pycaps"
    assert commands[0][1] == "render"
    assert "--transcript-format" in commands[0]
    assert reel_path.read_text(encoding="utf-8") == "subtitled"
    template_dir = reel_path.parent / ".pycaps_template"
    assert (template_dir / "pycaps.template.json").exists()

    template = json.loads((template_dir / "pycaps.template.json").read_text(encoding="utf-8"))
    assert template["layout"]["max_number_of_lines"] == 2
    assert template["layout"]["min_number_of_lines"] == 1
    assert template["layout"]["on_text_overflow_strategy"] == "exceed_width"
    assert template["layout"]["vertical_align"] == {"align": "bottom", "offset": 0.0}
    assert template["layout"]["x_words_space"] == 6
    assert template["layout"]["y_words_space"] == 8

    styles = (template_dir / "styles.css").read_text(encoding="utf-8")
    assert "--subtitle-font-size: 44px;" in styles
    assert "padding: var(--subtitle-padding-y) var(--subtitle-padding-x);" in styles
