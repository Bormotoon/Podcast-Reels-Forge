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
        settings=bs.SubtitleRenderSettings(enabled=True, font_path=font_path),
    )

    assert srt_path == reel_path.with_suffix(".srt")
    assert srt_path.exists()
    assert "00:00:00,500 --> 00:00:03,500" in srt_path.read_text(encoding="utf-8")
    assert commands
    assert commands[0][0] == "/usr/bin/pycaps"
    assert commands[0][1] == "render"
    assert "--transcript-format" in commands[0]
    assert reel_path.read_text(encoding="utf-8") == "subtitled"
    assert (reel_path.parent / ".pycaps_template" / "pycaps.template.json").exists()
