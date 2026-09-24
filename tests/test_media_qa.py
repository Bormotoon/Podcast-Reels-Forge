"""RU: Проверка готовых роликов. EN: Rendered clip QA."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from podcast_reels_forge.utils import media_qa


def _probe_result(*, duration: float, kinds: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    payload = {"format": {"duration": str(duration)}, "streams": [{"codec_type": k} for k in kinds]}
    return subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")


@pytest.fixture
def clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(media_qa, "ffprobe_bin", lambda: "ffprobe")
    path = tmp_path / "reel_01.mp4"
    path.write_bytes(b"x")
    return path


def test_a_good_clip_passes(clip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_qa.subprocess, "run", lambda *a, **k: _probe_result(duration=50.2, kinds=("video", "audio")))
    assert media_qa.check_clip(clip, expected_duration=50.0) == []


def test_missing_audio_and_short_duration_fail(clip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_qa.subprocess, "run", lambda *a, **k: _probe_result(duration=12.0, kinds=("video",)))
    problems = media_qa.check_clip(clip, expected_duration=50.0)
    assert problems is not None
    assert any("no audio" in p for p in problems)
    assert any("duration" in p for p in problems)


def test_mostly_black_clip_fails_when_asked(clip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if "blackdetect" in " ".join(cmd):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="black_start:0 black_end:40 black_duration:40.0")
        return _probe_result(duration=50.0, kinds=("video", "audio"))

    monkeypatch.setattr(media_qa.subprocess, "run", run)
    assert media_qa.check_clip(clip, expected_duration=50.0) == []
    assert any("black" in p for p in media_qa.check_clip(clip, expected_duration=50.0, blackdetect=True) or [])


def test_without_ffprobe_nothing_is_claimed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_qa, "ffprobe_bin", lambda: None)
    assert media_qa.check_clip(tmp_path / "missing.mp4", expected_duration=10.0) is None


@patch("podcast_reels_forge.scripts.video_processor.ffmpeg_cut")
@patch("podcast_reels_forge.scripts.video_processor._run_subprocess")
def test_a_broken_render_is_set_aside_and_fails_the_cut(
    mock_run: MagicMock, mock_ffmpeg_cut: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from podcast_reels_forge.scripts import video_processor

    def cut(_video: Path, _start: float, _end: float, out: Path, *_a: Any, **_k: Any) -> tuple[bool, Path, None]:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")
        return True, out, None

    mock_run.return_value = MagicMock(returncode=0)
    mock_ffmpeg_cut.side_effect = cut
    monkeypatch.setattr(video_processor, "check_clip", lambda *a, **k: ["qa: no audio stream"])
    monkeypatch.setattr(video_processor, "media_duration", lambda *a, **k: None)
    (tmp_path / "in.mp4").write_text("v")
    moments = tmp_path / "m.json"
    moments.write_text(json.dumps([{"start": 1.0, "end": 30.0, "title": "T", "score": 9}]))

    with pytest.raises(SystemExit) as excinfo:
        video_processor.main([
            "--input", str(tmp_path / "in.mp4"), "--moments", str(moments),
            "--outdir", str(tmp_path / "out"), "--threads", "1",
        ])

    assert excinfo.value.code == 2
    assert (tmp_path / "out" / "reels" / "rejected" / "reel_01.mp4").exists()
    assert not (tmp_path / "out" / "reels" / "reel_01.mp4").exists()
