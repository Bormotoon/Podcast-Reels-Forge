"""Tests for video processor logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from podcast_reels_forge.scripts.video_processor import (
    FfmpegOptions,
    main,
    ffmpeg_cut,
    create_concat_sample,
)

@patch("podcast_reels_forge.scripts.video_processor._run_subprocess")
def test_ffmpeg_cut_standard(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    opts = FfmpegOptions(
        vertical_crop=False,
        smart_crop_face=False,
        use_nvenc=True,
        v_bitrate="1M",
        a_bitrate="128k",
        preset="ultrafast",
        padding=0.0,
        face_samples=7,
        face_min_size=60,
    )
    video_in = Path("in.mp4")
    out_path = Path("out.mp4")
    
    success, out_p = ffmpeg_cut(video_in, 10.0, 20.0, out_path, opts)
    
    assert success is True
    assert out_p == out_path
    cmd = mock_run.call_args[0][0]
    assert "ffmpeg" in cmd
    # Check SS and TO
    assert "10.0" in cmd
    assert "20.0" in cmd
    assert "in.mp4" in cmd
    assert "out.mp4" in cmd

@patch("podcast_reels_forge.scripts.video_processor._run_subprocess")
def test_ffmpeg_cut_vertical(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    opts = FfmpegOptions(
        vertical_crop=True,
        smart_crop_face=False,
        use_nvenc=True,
        v_bitrate="1M",
        a_bitrate="128k",
        preset="ultrafast",
        padding=1.0,
        face_samples=7,
        face_min_size=60,
    )
    
    success, out_p = ffmpeg_cut(Path("in.mp4"), 10.0, 20.0, Path("out.mp4"), opts)
    
    assert success is True
    
    cmd = mock_run.call_args[0][0]
    assert "-vf" in cmd
    assert any("crop=1080:1920" in part for part in cmd)
    # ss should be 9.0 due to padding
    assert "9.0" in cmd

@patch("podcast_reels_forge.scripts.video_processor.Path.unlink")
@patch("podcast_reels_forge.scripts.video_processor._run_subprocess")
def test_create_concat_sample(mock_run: MagicMock, mock_unlink: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    reels = [tmp_path / "r1.mp4", tmp_path / "r2.mp4"]
    out = tmp_path / "preview.mp4"
    
    success = create_concat_sample(reels, out)
    
    assert success is True
    # Should have created a .txt file
    txt_file = tmp_path / "preview.mp4.txt"
    assert txt_file.exists()
    content = txt_file.read_text()
    assert "file '" in content
    assert "r1.mp4" in content
    assert mock_unlink.called


@patch("podcast_reels_forge.scripts.video_processor._run_subprocess")
def test_main_writes_reel_markdown(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = MagicMock(returncode=0)

    input_video = tmp_path / "input.mp4"
    input_video.write_text("video")

    moments_path = tmp_path / "moments.json"
    moments_path.write_text(
        json.dumps(
            [
                {
                    "start": 10.0,
                    "end": 20.0,
                    "title": "Strong moment",
                    "quote": "Key quote",
                    "why": "Because it is compelling",
                    "score": 9,
                    "hook": "Hook",
                    "caption": "A ready caption for the reel",
                    "hashtags": ["#podcast", "#reels"],
                }
            ],
        ),
        encoding="utf-8",
    )

    outdir = tmp_path / "out"
    main(
        [
            "--input",
            str(input_video),
            "--moments",
            str(moments_path),
            "--outdir",
            str(outdir),
            "--threads",
            "1",
        ],
    )

    md_path = outdir / "reels" / "reel_01.md"
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "A ready caption for the reel" in content
    assert "#podcast" in content


@patch("podcast_reels_forge.scripts.video_processor.sync_reel_burned_subtitles")
@patch("podcast_reels_forge.scripts.video_processor._run_subprocess")
def test_main_burns_subtitles_when_requested(
    mock_run: MagicMock,
    mock_sync_subtitles: MagicMock,
    tmp_path: Path,
) -> None:
    mock_run.return_value = MagicMock(returncode=0)

    input_video = tmp_path / "input.mp4"
    input_video.write_text("video")
    transcript_json = tmp_path / "video.json"
    transcript_json.write_text('{"segments": []}', encoding="utf-8")
    subtitle_font = tmp_path / "font.ttf"
    subtitle_font.write_text("font")
    subtitle_css = tmp_path / "subtitles.css"
    subtitle_css.write_text(".word { font-size: __FONT_SIZE_PX__px; }\n", encoding="utf-8")

    moments_path = tmp_path / "moments.json"
    moments_path.write_text(
        json.dumps([{"start": 10.0, "end": 20.0, "title": "Clip"}]),
        encoding="utf-8",
    )

    outdir = tmp_path / "out"
    main(
        [
            "--input",
            str(input_video),
            "--moments",
            str(moments_path),
            "--outdir",
            str(outdir),
            "--threads",
            "1",
            "--burn-subtitles",
            "--transcript-json",
            str(transcript_json),
            "--subtitle-font",
            str(subtitle_font),
            "--subtitle-css",
            str(subtitle_css),
            "--no-subtitle-wrap-words",
        ],
    )

    assert mock_sync_subtitles.called
    sync_args = mock_sync_subtitles.call_args.args
    sync_kwargs = mock_sync_subtitles.call_args.kwargs
    assert sync_args[0][0]["title"] == "Clip"
    assert sync_args[1] == outdir / "reels"
    assert sync_kwargs["transcript_json_path"] == transcript_json
    assert sync_kwargs["settings"].font_path == subtitle_font.resolve()
    assert sync_kwargs["settings"].css_path == subtitle_css.resolve()
    assert sync_kwargs["settings"].wrap_words is False
