"""Tests for pycaps-based burned subtitle helpers."""

from __future__ import annotations

import json
from pathlib import Path
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
    assert settings.wrap_words is True
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
                "wrap_words": False,
            },
        },
        repo_dir=tmp_path,
    )

    assert settings.enabled is True
    assert settings.font_path == (tmp_path / "assets/fonts/custom.ttf").resolve()
    assert settings.css_path == (tmp_path / "assets/subtitles/custom.css").resolve()
    assert settings.wrap_words is False


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


def test_prepare_pycaps_template_disables_word_wrap_when_requested(tmp_path: Path) -> None:
    font_path = tmp_path / "font.ttf"
    font_path.write_text("font", encoding="utf-8")

    css_template = tmp_path / "subtitles.css"
    css_template.write_text(
        ".word { font-size: __FONT_SIZE_PX__px; }\n",
        encoding="utf-8",
    )

    template_dir = tmp_path / "template"
    out_dir = bs.prepare_pycaps_template(
        template_dir,
        settings=bs.SubtitleRenderSettings(
            enabled=True,
            font_path=font_path,
            css_path=css_template,
            wrap_words=False,
        ),
    )

    template = json.loads((out_dir / "pycaps.template.json").read_text(encoding="utf-8"))
    assert template["layout"]["max_number_of_lines"] == 1


def test_prepare_subtitle_segments_splits_long_cues_into_readable_chunks(
    tmp_path: Path,
) -> None:
    font_path = tmp_path / "font.ttf"
    font_path.write_text("font", encoding="utf-8")

    css_template = tmp_path / "subtitles.css"
    css_template.write_text(
        ".word { font-size: __FONT_SIZE_PX__px; }\n",
        encoding="utf-8",
    )

    prepared = bs._prepare_subtitle_segments(  # noqa: SLF001 - regression coverage for runtime bug
        [
            bs.SubtitleSegment(
                start=0.0,
                end=5.1,
                text=(
                    "разражает почему а первый потому что правда "
                    "почти всегда ведут себя неподобающе как я уже"
                ),
            ),
        ],
        settings=bs.SubtitleRenderSettings(
            enabled=True,
            font_path=font_path,
            css_path=css_template,
            max_lines=2,
            wrap_words=True,
        ),
    )

    assert [segment.text for segment in prepared] == [
        "разражает почему а первый потому",
        "что правда почти всегда ведут",
        "себя неподобающе как я уже",
    ]
    assert prepared[0].start == 0.0
    assert prepared[-1].end == 5.1


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

    render_calls: list[tuple[str, str, str, list[str]]] = []

    def fake_render(
        *,
        template_dir: Path,
        reel_path: Path,
        tmp_output: Path,
        clip_segments: list[bs.SubtitleSegment],
        settings: bs.SubtitleRenderSettings,
        verbose: bool = False,
    ) -> None:
        render_calls.append(
            (
                template_dir.name,
                reel_path.name,
                tmp_output.name,
                [segment.text for segment in clip_segments],
            ),
        )
        tmp_output.write_text("subtitled")

    monkeypatch.setattr(bs, "_render_reel_with_pycaps", fake_render)

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
    assert render_calls == [
        (
            ".pycaps_template",
            "reel_01.mp4",
            "reel_01.subtitled.mp4",
            ["Key moment", "Closing words"],
        ),
    ]
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
