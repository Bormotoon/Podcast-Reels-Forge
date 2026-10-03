"""Tests for .ass-based burned subtitle helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import podcast_reels_forge.utils.burned_subtitles as bs

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
    # Text spans the frame minus the 140px action-rail inset on both sides.
    assert settings.max_width_ratio == 0.74
    assert settings.wrap_words is True
    # The position comes from the .ass style (editor or preset) unless overridden.
    assert settings.vertical_align == "style"
    assert settings.highlight_mode == "none"
    assert settings.preset == ""
    assert settings.vertical_offset == 0.0
    assert settings.ass_style is None
    assert settings.fade_in_duration == bs.DEFAULT_FADE_IN_S
    assert settings.fade_out_duration == bs.DEFAULT_FADE_OUT_S


def test_fade_can_be_disabled_from_config(tmp_path: Path) -> None:
    """Zero must survive parsing: it is how the fade gets turned off."""
    settings = bs.subtitle_settings_from_conf(
        {"subtitles": {"fade_in_duration": 0, "fade_out_duration": 0}},
        repo_dir=tmp_path,
    )

    assert settings.fade_in_duration == 0.0
    assert settings.fade_out_duration == 0.0
    seg = bs.SubtitleSegment(start=0.0, end=3.0, text="раз два")
    assert bs._fade_tag(seg, settings) == ""


def test_default_ass_style_is_the_viral_caption_look(tmp_path: Path) -> None:
    """The shipped style must stay legible over arbitrary footage."""
    header = bs._default_ass_header("Bignoodletoooblique", bs.DEFAULT_FONT_SIZE_PX)

    style = next(
        line for line in header.splitlines() if line.startswith("Style: Default,")
    )
    fields = style.split(",")
    # Format: Name, Fontname, Fontsize, Primary, Secondary, Outline, Back, ...
    assert fields[2] == "96"
    # \kf sweeps secondary -> primary, so spoken text is the accent colour.
    assert fields[3] == bs.DEFAULT_PRIMARY_COLOUR
    assert fields[4] == bs.DEFAULT_SECONDARY_COLOUR
    assert fields[7] == "-1", "captions are bold"

    assert bs._ass_style_margin_v(header) == bs.DEFAULT_MARGIN_V
    # A thick outline replaces the drop shadow.
    assert bs.DEFAULT_OUTLINE_WIDTH >= 6
    assert bs.DEFAULT_SHADOW_DEPTH == 0


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
        karaoke=True,
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
    """Timings come from the transcript, not from character lengths — except
    that a 3 s "Да" is Whisper folding the pause before it into the word: the
    word starts where it plausibly could, so its cue does not show early."""
    segment = bs.load_transcript_segments(_transcript_with_words(tmp_path / "t.json"))[0]

    timed = bs._build_timed_words(segment)
    by_text = {w.text: w for w in timed}

    assert by_text["Да"].end == 3.0
    assert by_text["Да"].start == pytest.approx(3.0 - 0.42)
    assert round(by_text["очень"].end - by_text["очень"].start, 3) == 0.5
    assert by_text["слово"].end == 6.0


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
    settings = bs.subtitle_settings_from_conf({"subtitles": {"karaoke": True}}, repo_dir=tmp_path)
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


def test_fade_tag_emitted_and_clamped_to_cue_length(tmp_path: Path) -> None:
    """subtitles.fade_* must reach the ASS instead of being parsed and dropped."""
    seg = bs.SubtitleSegment(start=0.0, end=4.0, text="раз два")

    settings = bs.SubtitleRenderSettings(
        enabled=True,
        font_path=tmp_path / "font.ttf",
        fade_in_duration=0.18,
        fade_out_duration=0.12,
    )
    assert bs._fade_tag(seg, settings) == r"{\fad(180,120)}"

    # Fades longer than the cue are scaled down so it still reaches full opacity.
    short = bs.SubtitleSegment(start=0.0, end=0.5, text="раз")
    greedy = bs.SubtitleRenderSettings(
        enabled=True,
        font_path=tmp_path / "font.ttf",
        fade_in_duration=1.0,
        fade_out_duration=1.0,
    )
    assert bs._fade_tag(short, greedy) == r"{\fad(250,250)}"

    # Zero means "no fade at all", not "\fad(0,0)".
    off = bs.SubtitleRenderSettings(
        enabled=True,
        font_path=tmp_path / "font.ttf",
        fade_in_duration=0.0,
        fade_out_duration=0.0,
    )
    assert bs._fade_tag(seg, off) == ""


def test_written_ass_carries_fade_before_karaoke(tmp_path: Path) -> None:
    """The fade override must lead the line, ahead of the \\kf word tags."""
    ass_path = tmp_path / "out.ass"
    segments = [bs.SubtitleSegment(start=0.0, end=2.0, text="раз два")]
    settings = bs.SubtitleRenderSettings(
        enabled=True,
        font_path=tmp_path / "font.ttf",
        fade_in_duration=0.2,
        fade_out_duration=0.1,
        karaoke=True,
    )

    bs._write_ass_file(ass_path, segments, settings)

    dialogue = [
        line for line in ass_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Dialogue:")
    ]
    assert len(dialogue) == 1
    # Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
    text = dialogue[0].split(",", 9)[9]
    assert text.startswith(r"{\fad(200,100)}")
    assert r"\kf" in text


def test_max_width_ratio_drives_line_length(tmp_path: Path) -> None:
    """Widening max_width_ratio must allow longer subtitle blocks."""
    long_text = " ".join(["слово"] * 40)
    segments = [bs.SubtitleSegment(start=0.0, end=20.0, text=long_text)]

    def longest_block(ratio: float) -> int:
        settings = bs.SubtitleRenderSettings(
            enabled=True,
            font_path=tmp_path / "font.ttf",
            max_width_ratio=ratio,
        )
        prepared = bs._prepare_subtitle_segments(segments, settings=settings)
        return max(len(seg.text) for seg in prepared)

    narrow = longest_block(0.35)
    default = longest_block(bs.DEFAULT_MAX_WIDTH_RATIO)
    wide = longest_block(0.95)

    assert narrow < default < wide


def test_vertical_offset_overrides_dialogue_margin(tmp_path: Path) -> None:
    """subtitles.vertical_offset must shift MarginV on the Dialogue lines."""
    segments = [bs.SubtitleSegment(start=0.0, end=2.0, text="раз два")]

    def margins(offset: float) -> list[str]:
        ass_path = tmp_path / f"out_{offset}.ass"
        bs._write_ass_file(
            ass_path,
            segments,
            bs.SubtitleRenderSettings(
                enabled=True,
                font_path=tmp_path / "font.ttf",
                vertical_offset=offset,
            ),
        )
        return [
            line.split(",", 9)[7]  # MarginV
            for line in ass_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("Dialogue:")
        ]

    # 0 keeps the historic "inherit from style" behaviour byte-for-byte.
    assert margins(0.0) == ["0"]
    # A positive offset pushes the cue away from its anchored edge.
    shifted = int(margins(0.1)[0])
    assert shifted > 0


def test_ass_header_parsing_reads_margin_and_res() -> None:
    """MarginV is located through the Format line, not a hardcoded index."""
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Big,96,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,1,5,2,2,45,45,470,1\n"
    )
    assert bs._ass_play_res_y(header) == 1920
    assert bs._ass_style_margin_v(header) == 470

    # No usable header: fall back rather than crash.
    assert bs._ass_play_res_y("") == 1920
    assert bs._ass_style_margin_v("") == 0


# --- Presets, layout and highlight modes ---------------------------------------

REPO = Path(__file__).resolve().parents[1]
FONT = REPO / "assets/fonts/bignoodletoooblique.ttf"


def _timed(text: str, step: float = 0.5, speaker: str = "") -> bs.SubtitleSegment:
    words = tuple(
        bs._TimedSubtitleWord(round(i * step, 3), round(i * step + step * 0.9, 3), w)
        for i, w in enumerate(text.split())
    )
    return bs.SubtitleSegment(
        start=words[0].start, end=words[-1].end, text=text, words=words, speaker=speaker,
    )


def _dialogues(tmp_path: Path, segments: list[bs.SubtitleSegment], **conf: object) -> list[str]:
    settings = bs.subtitle_settings_from_conf(
        {"subtitles": {"font": str(FONT), **conf}}, repo_dir=tmp_path,
    )
    ass_path = tmp_path / "out.ass"
    bs._write_ass_file(ass_path, segments, settings)
    return [
        line.split(",", 9)[9]
        for line in ass_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Dialogue:")
    ]


def test_preset_render_defaults_yield_to_explicit_config(tmp_path: Path) -> None:
    settings = bs.subtitle_settings_from_conf(
        {"subtitles": {"preset": "hormozi", "max_words_per_cue": 2}}, repo_dir=tmp_path,
    )
    assert settings.preset == "hormozi"
    assert settings.highlight_mode == "word"
    assert settings.text_case == "upper"
    assert settings.max_words_per_cue == 2  # explicit beats the preset's 4
    assert settings.font_path.name == "MontserratBlack.ttf"


def test_unknown_preset_is_ignored(tmp_path: Path) -> None:
    settings = bs.subtitle_settings_from_conf({"subtitles": {"preset": "nope"}}, repo_dir=tmp_path)
    assert settings.preset == ""


def test_long_cue_gets_explicit_balanced_line_break(tmp_path: Path) -> None:
    [text] = _dialogues(tmp_path, [_timed("Разбираемся почему этот выпуск вызывает споры")])
    assert text.count("\\N") == 1
    top, bottom = text.split("}")[-1].split("\\N")
    assert abs(len(top) - len(bottom)) <= 8


def test_header_turns_on_scaled_borders_and_wrap_style(tmp_path: Path) -> None:
    settings = bs.subtitle_settings_from_conf({"subtitles": {"font": str(FONT)}}, repo_dir=tmp_path)
    header = bs._resolve_style_header(settings)
    assert "ScaledBorderAndShadow: yes" in header
    assert "WrapStyle: 0" in header


def test_word_highlight_emits_one_event_per_word(tmp_path: Path) -> None:
    seg = _timed("раз два три")
    texts = _dialogues(tmp_path, [seg], preset="hormozi", max_words_per_cue=0)
    assert len(texts) == 3
    for i, text in enumerate(texts):
        assert text.count("\\rHighlight") == 1
        active = text.split("\\rHighlight", 1)[1].split("}", 1)[1]
        assert active.startswith(["РАЗ", "ДВА", "ТРИ"][i])


def test_fill_highlights_every_spoken_word(tmp_path: Path) -> None:
    texts = _dialogues(tmp_path, [_timed("раз два три")], highlight="fill")
    assert [t.count("{\\1c") for t in texts][-1] >= 3


def test_highlight_without_highlight_style_uses_style_colours(tmp_path: Path) -> None:
    [first, *_] = _dialogues(tmp_path, [_timed("раз два")], highlight="word")
    # Inactive = SecondaryColour (white), active = PrimaryColour (amber).
    assert "\\1c&HFFFFFF&" in first
    assert "\\1c&H0AD6FF&" in first


def test_reveal_hides_words_not_yet_spoken(tmp_path: Path) -> None:
    texts = _dialogues(tmp_path, [_timed("раз два три")], highlight="reveal")
    assert "\\alpha&HFF&" in texts[0]
    assert "\\alpha&HFF&" not in texts[-1]


def test_reveal_keeps_a_cue_box_visible(tmp_path: Path) -> None:
    texts = _dialogues(tmp_path, [_timed("раз два три")], preset="retro")
    assert "\\alpha&HFF&" not in texts[0]
    assert "\\1a&HFF&\\3a&HFF&" in texts[0]


def test_pop_scales_the_active_word_back_down(tmp_path: Path) -> None:
    texts = _dialogues(tmp_path, [_timed("раз два")], highlight="pop")
    assert "\\t(0," in texts[0]
    assert "\\fscx112" in texts[0]


def test_fade_only_on_the_outer_word_events(tmp_path: Path) -> None:
    texts = _dialogues(
        tmp_path, [_timed("раз два три")], highlight="word",
        fade_in_duration=0.1, fade_out_duration=0.1,
    )
    assert texts[0].startswith("{\\fad(100,0)}")
    assert "\\fad" not in texts[1]
    assert texts[-1].startswith("{\\fad(0,100)}")


def test_text_case_punctuation_and_censor_reach_the_render(tmp_path: Path) -> None:
    [text] = _dialogues(
        tmp_path, [_timed("ну, блин, вот так.")],
        text_case="upper", strip_punctuation="periods", censor_words=["блин"],
    )
    assert "НУ Б**Н ВОТ ТАК" in text


def test_speaker_change_starts_a_new_cue(tmp_path: Path) -> None:
    a = _timed("раз два", speaker="SPEAKER_00")
    b = bs.SubtitleSegment(
        start=1.0, end=1.9, text="три",
        words=(bs._TimedSubtitleWord(1.0, 1.9, "три"),), speaker="SPEAKER_01",
    )
    settings = bs.subtitle_settings_from_conf(None, repo_dir=tmp_path)
    prepared = bs._prepare_subtitle_segments([a, b], settings=settings)
    assert [s.speaker for s in prepared] == ["SPEAKER_00", "SPEAKER_01"]

    merged = bs._prepare_subtitle_segments(
        [a, b], settings=bs.subtitle_settings_from_conf(
            {"subtitles": {"split_on_speaker": False}}, repo_dir=tmp_path,
        ),
    )
    assert len(merged) == 1


def test_speaker_colours_follow_first_appearance(tmp_path: Path) -> None:
    a = _timed("раз", speaker="B")
    b = bs.SubtitleSegment(start=3.0, end=3.5, text="два",
                           words=(bs._TimedSubtitleWord(3.0, 3.5, "два"),), speaker="A")
    texts = _dialogues(tmp_path, [a, b], speaker_colors=["#FF0000", "#00FF00"])
    assert "\\1c&H0000FF&" in texts[0]
    assert "\\1c&H00FF00&" in texts[1]


def test_max_words_per_cue_splits_into_short_cues(tmp_path: Path) -> None:
    settings = bs.subtitle_settings_from_conf({"subtitles": {"max_words_per_cue": 1}}, repo_dir=tmp_path)
    prepared = bs._prepare_subtitle_segments([_timed("раз два три")], settings=settings)
    assert [s.text for s in prepared] == ["раз", "два", "три"]


def test_every_cue_fits_its_lines(tmp_path: Path) -> None:
    long_text = " ".join(["длиннословие"] * 30)
    settings = bs.subtitle_settings_from_conf(
        {"subtitles": {"font": str(FONT), "max_chars_per_line": 80}}, repo_dir=tmp_path,
    )
    prepared = bs._prepare_subtitle_segments([_timed(long_text, step=0.3)], settings=settings)
    layout = bs._layout_for(bs._resolve_style_header(settings), settings)
    for seg in prepared:
        rows = bs.wrap_words(seg.text.split(), layout.measurer,
                             max_width=layout.max_width, max_lines=layout.max_lines)
        assert len(rows) <= 2
        assert bs.lines_fit(rows, layout.measurer, layout.max_width)


def test_vertical_align_overrides_the_style_row(tmp_path: Path) -> None:
    [top] = _dialogues(tmp_path, [_timed("раз два")], vertical_align="top")
    assert "\\an8" in top
    [centre] = _dialogues(tmp_path, [_timed("раз два")], vertical_align="center", vertical_offset=0.1)
    assert "\\an5\\pos(540,768)" in centre
    [style] = _dialogues(tmp_path, [_timed("раз два")])
    assert "\\an" not in style


def test_wide_width_ratio_widens_the_event_margins(tmp_path: Path) -> None:
    settings = bs.subtitle_settings_from_conf(
        {"subtitles": {"font": str(FONT), "max_width_ratio": 0.9}}, repo_dir=tmp_path,
    )
    layout = bs._layout_for(bs._resolve_style_header(settings), settings)
    assert layout.max_width == pytest.approx(972, abs=1)
    assert layout.event_margin_l == layout.event_margin_r == 54


def test_configured_ass_style_file_is_used(tmp_path: Path) -> None:
    style = tmp_path / "custom.ass"
    style.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,bignoodletoooblique,77,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        "0,0,0,0,100,100,0,0,1,2,0,2,10,10,300,1\n",
        encoding="utf-8",
    )
    settings = bs.subtitle_settings_from_conf(
        {"subtitles": {"font": str(FONT), "ass_style": str(style)}}, repo_dir=tmp_path,
    )
    header = bs._resolve_style_header(settings)
    # The editor's file-stem font name is mapped to the font's real family.
    assert "Style: Default,BigNoodleTooOblique,77," in header
    assert "ScaledBorderAndShadow: yes" in header


def test_video_processor_reads_the_whole_subtitles_section() -> None:
    from podcast_reels_forge.scripts.video_processor import _subtitle_settings_from_json

    settings = _subtitle_settings_from_json('{"max_lines": 3, "preset": "box", "fade_in_duration": 0}')
    assert settings.max_lines == 3
    assert settings.preset == "box"
    assert settings.fade_in_duration == 0.0
    assert _subtitle_settings_from_json("not json").max_lines == bs.DEFAULT_MAX_LINES


def test_back_to_back_cues_cut_over_without_fading(tmp_path: Path) -> None:
    """Fades only around pauses: a cue right after another must not blink."""
    a = _timed("раз два")                         # 0.0 .. 0.95
    b = bs.SubtitleSegment(start=1.0, end=1.9, text="три",
                           words=(bs._TimedSubtitleWord(1.0, 1.9, "три"),))
    c = bs.SubtitleSegment(start=3.0, end=3.9, text="четыре",
                           words=(bs._TimedSubtitleWord(3.0, 3.9, "четыре"),))
    texts = _dialogues(tmp_path, [a, b, c], fade_in_duration=0.1, fade_out_duration=0.1)
    assert texts[0].startswith("{\\fad(100,0)}")   # after silence in, straight into b
    assert texts[1].startswith("{\\fad(0,100)}")   # a pause follows b
    assert texts[2].startswith("{\\fad(100,100)}")

    always = _dialogues(tmp_path, [a, b], fade_in_duration=0.1, fade_out_duration=0.1, fade_min_gap_s=0)
    assert always[0].startswith("{\\fad(100,100)}")


def test_default_line_balance_is_the_pyramid(tmp_path: Path) -> None:
    settings = bs.subtitle_settings_from_conf(None, repo_dir=tmp_path)
    assert settings.line_balance == "bottom_heavy"
    assert settings.fade_min_gap_s == bs.DEFAULT_FADE_MIN_GAP_S
