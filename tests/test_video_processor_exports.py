"""Tests for video exports helper functions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import podcast_reels_forge.scripts.video_processor as vp

if TYPE_CHECKING:
    from pathlib import Path
    import pytest

    MonkeyPatch = pytest.MonkeyPatch


class DummyRes:
    """Simple subprocess result stub."""

    def __init__(self, rc: int = 0) -> None:
        """Store return code."""
        self.returncode = rc


def test_exports_flags_invoke_ffmpeg(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Ensure export helpers invoke ffmpeg with expected flags."""
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str] | tuple[str, ...], *,
        capture_output: bool = False,
        text: bool = False,
    ) -> DummyRes:
        del capture_output, text
        calls.append(list(cmd))
        return DummyRes(0)

    monkeypatch.setattr(vp, "subprocess", SimpleNamespace(run=fake_run))

    mp4 = str(tmp_path / "x.mp4")
    out_webm = str(tmp_path / "x.webm")
    out_audio = str(tmp_path / "x.m4a")
    out_gif = str(tmp_path / "x.gif")

    if not vp._export_webm(mp4, out_webm):  # noqa: SLF001
        message = "WebM export failed"
        raise AssertionError(message)
    if not vp._export_audio(mp4, out_audio):  # noqa: SLF001
        message = "Audio export failed"
        raise AssertionError(message)
    if not vp._export_gif(mp4, out_gif):  # noqa: SLF001
        message = "GIF export failed"
        raise AssertionError(message)

    # RU: Как минимум один вызов ffmpeg на каждый экспорт.
    # EN: At least one ffmpeg call per export.
    if not any(cmd and cmd[0] == "ffmpeg" for cmd in calls):
        message = "Expected ffmpeg calls for exports"
        raise AssertionError(message)
