"""Tests for video processor logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from podcast_reels_forge.scripts.video_processor import (
    FfmpegOptions,
    ffmpeg_cut,
    create_concat_sample,
)

@patch("podcast_reels_forge.scripts.video_processor._run_subprocess")
def test_ffmpeg_cut_standard(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    opts = FfmpegOptions(
        vertical_crop=False,
        v_bitrate="1M",
        a_bitrate="128k",
        preset="ultrafast",
        padding=0.0,
    )
    video_in = Path("in.mp4")
    out_path = Path("out.mp4")
    
    success = ffmpeg_cut(video_in, 10.0, 20.0, out_path, opts)
    
    assert success is True
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
        v_bitrate="1M",
        a_bitrate="128k",
        preset="ultrafast",
        padding=1.0,
    )
    
    ffmpeg_cut(Path("in.mp4"), 10.0, 20.0, Path("out.mp4"), opts)
    
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
