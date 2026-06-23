"""Tests for .ass-based burned subtitle helpers."""

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

    assert settings.font_size_px == 96
    assert settings.max_lines == 2
    assert settings.max_width_ratio == 0.65
    assert settings.wrap_words is True
    assert settings.vertical_align == "bottom"
    assert settings.vertical_offset == 0.0
    assert settings.ass_style is None


def test_subtitle_settings_from_conf_resolves_css_path(tmp_path: Path) -> None:
    settings = bs.subtitle_settings_from_conf(
        {
            "subtitles": {
                "enabled": True,
                "font": "assets/fonts/custom.ttf",
                "ass_style": "assets/subtitles/custom.ass",
                "wrap_words": False,
            },
        },
        repo_dir=tmp_path,
    )

    assert settings.enabled is True
    assert settings.font_path == (tmp_path / "assets/fonts/custom.ttf").resolve()
    assert settings.ass_style == (tmp_path / "assets/subtitles/custom.ass").resolve()
    assert settings.wrap_words is False


def test_write_ass_file_creates_valid_ass(tmp_path: Path) -> None:
    """Ensure _write_ass_file creates a valid .ass file with karaoke tags."""
    ass_path = tmp_path / "test.ass"
    segments = [
        bs.SubtitleSegment(start=0.0, end=2.5, text="Hello world"),
        bs.SubtitleSegment(start=3.0, end=5.0, text="Second line"),
    ]
    settings = bs.SubtitleRenderSettings(
        enabled=True,
        font_path=tmp_path / "font.ttf",
    )

    bs._write_ass_file(ass_path, segments, settings)

    content = ass_path.read_text(encoding="utf-8")
    assert "[Script Info]" in content
    assert "[V4+ Styles]" in content
    assert "[Events]" in content
    assert "Dialogue:" in content
    assert "Hello" in content
    assert "world" in content
    assert "\\kf" in content  # karaoke tag


def test_fmt_ass_time_formats_correctly() -> None:
    assert bs._fmt_ass_time(0.0) == "0:00:00.00"
    assert bs._fmt_ass_time(65.5) == "0:01:05.50"
    result = bs._fmt_ass_time(3661.12)
    assert result.startswith("1:01:01.1")


def test_ensure_reel_burned_subtitles_writes_srt_and_ass(
    tmp_path: Path,
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

    ass_path = bs.ensure_reel_burned_subtitles(
        {"start": 10.0, "end": 13.0},
        reel_path,
        transcript_json_path=transcript_path,
        padding=1.0,
        settings=bs.SubtitleRenderSettings(
            enabled=True,
            font_path=font_path,
        ),
    )

    assert ass_path is not None
    assert ass_path.suffix == ".ass"
    assert ass_path.exists()
    assert "[Script Info]" in ass_path.read_text(encoding="utf-8")

    srt_path = reel_path.with_suffix(".srt")
    assert srt_path.exists()
    srt_content = srt_path.read_text(encoding="utf-8")
    assert "Key moment" in srt_content
