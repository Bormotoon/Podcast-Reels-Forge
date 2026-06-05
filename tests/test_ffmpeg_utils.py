"""Tests for the ffmpeg resolver and codec-argument builder."""

from __future__ import annotations

from unittest.mock import patch

from podcast_reels_forge.utils import ffmpeg as ffmod


def test_codec_args_nvenc_when_available() -> None:
    with patch.object(ffmod, "ffmpeg_has_nvenc", return_value=True):
        args = ffmod.build_video_codec_args(
            use_nvenc=True, v_bitrate="8M", preset="fast", nvenc_cq=21, nvenc_preset="p5",
        )
    assert "h264_nvenc" in args
    assert "-cq" in args and "21" in args
    assert "-maxrate" in args and "8M" in args
    assert "libx264" not in args


def test_codec_args_software_when_nvenc_missing() -> None:
    # use_nvenc requested but the resolved ffmpeg has no NVENC -> software.
    with patch.object(ffmod, "ffmpeg_has_nvenc", return_value=False):
        args = ffmod.build_video_codec_args(
            use_nvenc=True, v_bitrate="8M", preset="fast",
        )
    assert "libx264" in args
    assert "h264_nvenc" not in args


def test_codec_args_software_when_not_requested() -> None:
    with patch.object(ffmod, "ffmpeg_has_nvenc", return_value=True):
        args = ffmod.build_video_codec_args(
            use_nvenc=False, v_bitrate="8M", preset="medium",
        )
    assert "libx264" in args
    assert "-preset" in args and "medium" in args


def test_resolve_prefers_nvenc_build() -> None:
    ffmod.resolve_ffmpeg.cache_clear()
    # First candidate exists but lacks NVENC; second exists and has NVENC.
    def fake_which(name: str) -> str | None:
        return None

    def fake_exists(path: str) -> bool:
        return path in {"/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"}

    def fake_has_nvenc(path: str) -> bool:
        return path == "/usr/bin/ffmpeg"

    with patch.object(ffmod.shutil, "which", fake_which), \
         patch.object(ffmod.os.path, "exists", fake_exists), \
         patch.object(ffmod, "_has_nvenc", fake_has_nvenc):
        path, has = ffmod.resolve_ffmpeg()
    ffmod.resolve_ffmpeg.cache_clear()
    assert has is True
    assert path == "/usr/bin/ffmpeg"
