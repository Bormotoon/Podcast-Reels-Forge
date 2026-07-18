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


# -- word-level timing -------------------------------------------------------


def _word(start: float, end: float, text: str) -> dict[str, float | str]:
    return {"start": start, "end": end, "word": text, "probability": 0.9}


def _transcript_with_words(path: Path) -> Path:
    """A transcript whose words are deliberately uneven in duration.

    Interpolating by character length would spread them evenly, so any test
    asserting the real timings survived can tell the two apart.
    """
    path.write_text(
        json.dumps(
            {
                "language": "ru",
                "duration": 10.0,
                "timing_version": 2,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 6.0,
                        "text": "Да очень длинное слово",
                        "words": [
                            _word(0.0, 3.0, "Да"),
                            _word(3.0, 3.5, "очень"),
                            _word(3.5, 4.0, "длинное"),
                            _word(4.0, 6.0, "слово"),
                        ],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_load_transcript_segments_attaches_word_timings(tmp_path: Path) -> None:
    segments = bs.load_transcript_segments(_transcript_with_words(tmp_path / "t.json"))

    assert len(segments) == 1
    assert [w.text for w in segments[0].words] == ["Да", "очень", "длинное", "слово"]


def test_build_timed_words_uses_real_timings(tmp_path: Path) -> None:
    """"Да" is short but lasts 3s; interpolation would give it the least time."""
    segment = bs.load_transcript_segments(_transcript_with_words(tmp_path / "t.json"))[0]

    timed = bs._build_timed_words(segment)
    durations = {w.text: round(w.end - w.start, 3) for w in timed}

    assert durations["Да"] == 3.0
    assert durations["очень"] == 0.5
    assert durations["Да"] > durations["длинное"]


def test_build_timed_words_falls_back_without_word_data() -> None:
    """Transcripts with no word timings still get karaoke, just interpolated."""
    segment = bs.SubtitleSegment(start=0.0, end=4.0, text="раз два три")

    timed = bs._build_timed_words(segment)

    assert [w.text for w in timed] == ["раз", "два", "три"]
    assert timed[0].start == 0.0
    assert timed[-1].end == 4.0


def test_build_timed_words_falls_back_when_text_was_edited() -> None:
    """Proofread rewrites segment text while keeping the original word list."""
    segment = bs.SubtitleSegment(
        start=0.0,
        end=4.0,
        text="совершенно другой текст здесь",
        words=(
            bs._TimedSubtitleWord(0.0, 1.0, "раз"),
            bs._TimedSubtitleWord(1.0, 2.0, "два"),
        ),
    )

    timed = bs._build_timed_words(segment)

    assert [w.text for w in timed] == ["совершенно", "другой", "текст", "здесь"]
    assert timed[-1].end == 4.0


def test_slice_segments_for_clip_rebases_word_timings() -> None:
    segment = bs.SubtitleSegment(
        start=10.0,
        end=14.0,
        text="раз два",
        words=(
            bs._TimedSubtitleWord(10.0, 11.0, "раз"),
            bs._TimedSubtitleWord(12.0, 14.0, "два"),
        ),
    )

    clipped = bs.slice_segments_for_clip([segment], clip_start=8.0, clip_end=20.0)

    assert [(w.start, w.end) for w in clipped[0].words] == [(2.0, 3.0), (4.0, 6.0)]


def test_ass_karaoke_reflects_real_word_durations(tmp_path: Path) -> None:
    """The \\kf durations in the rendered ASS come from the transcript."""
    segment = bs.SubtitleSegment(
        start=0.0,
        end=6.0,
        text="Да очень",
        words=(
            bs._TimedSubtitleWord(0.0, 3.0, "Да"),
            bs._TimedSubtitleWord(3.0, 6.0, "очень"),
        ),
    )
    settings = bs.subtitle_settings_from_conf(None, repo_dir=tmp_path)
    ass_path = tmp_path / "out.ass"

    bs._write_ass_file(ass_path, [segment], settings)
    content = ass_path.read_text(encoding="utf-8")

    # 3.0s each, in centiseconds.
    assert "{\\kf300}Да" in content
    assert "{\\kf300}очень" in content


def test_word_timings_survive_segment_merging(tmp_path: Path) -> None:
    """Merging short blocks must not silently drop back to interpolation."""
    segments = [
        bs.SubtitleSegment(
            start=0.0, end=1.0, text="раз",
            words=(bs._TimedSubtitleWord(0.0, 1.0, "раз"),),
        ),
        bs.SubtitleSegment(
            start=1.0, end=2.0, text="два",
            words=(bs._TimedSubtitleWord(1.0, 2.0, "два"),),
        ),
    ]
    settings = bs.subtitle_settings_from_conf(None, repo_dir=tmp_path)

    prepared = bs._prepare_subtitle_segments(segments, settings=settings)

    merged = [seg for seg in prepared if seg.text == "раз два"]
    assert merged, "the two short blocks should merge"
    assert [w.text for w in merged[0].words] == ["раз", "два"]
