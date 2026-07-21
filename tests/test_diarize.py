"""Tests for diarization script."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from podcast_reels_forge.scripts.diarize import main

def test_diarize_missing_token(tmp_path: Path) -> None:
    input_file = tmp_path / "audio.mp3"
    input_file.write_text("dummy")
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(SystemExit, match="PYANNOTE_TOKEN is not set"):
            main(["--input", str(input_file)])

def test_diarize_success(tmp_path: Path) -> None:
    # A .wav input is handed to pyannote as-is; see the mp3 test below for the
    # decoding path.
    input_file = tmp_path / "audio.wav"
    input_file.write_text("dummy")
    out_dir = tmp_path / "out"

    mock_pipeline_class = MagicMock()
    mock_pipeline_instance = MagicMock()
    mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

    mock_turn = MagicMock()
    mock_turn.start = 1.0
    mock_turn.end = 2.0
    mock_diar_result = MagicMock()
    mock_pipeline_instance.return_value = mock_diar_result
    mock_diar_result.itertracks.return_value = [(mock_turn, "track", "SPEAKER_01")]
    # RU: pyannote<4.0 отдаёт Annotation без .speaker_diarization; убираем
    #     авто-атрибут MagicMock, чтобы сработал fallback через getattr.
    # EN: pyannote<4.0 returns an Annotation without .speaker_diarization;
    #     drop the MagicMock auto-attribute so the getattr fallback kicks in.
    del mock_diar_result.speaker_diarization

    with patch.dict("os.environ", {"PYANNOTE_TOKEN": "fake_token"}):
        with patch.dict("sys.modules", {"pyannote.audio": MagicMock(Pipeline=mock_pipeline_class)}):
            main(["--input", str(input_file), "--outdir", str(out_dir)])
    
    out_json = out_dir / "diarization.json"
    assert out_json.exists()
    
    with out_json.open(encoding="utf-8") as f:
        data = json.load(f)
    assert data[0]["speaker"] == "SPEAKER_01"
    assert data[0]["start"] == 1.0


def test_diarize_decodes_non_pcm_input(tmp_path: Path) -> None:
    """pyannote fails on mp3, so a non-PCM input must be decoded first."""
    from podcast_reels_forge.scripts import diarize as diarize_mod

    input_file = tmp_path / "audio.mp3"
    input_file.write_text("dummy")

    ffmpeg_calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        cmd_list = list(cmd)
        ffmpeg_calls.append(cmd_list)
        Path(cmd_list[-1]).write_text("wav")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    seen: list[str] = []

    with patch.object(diarize_mod.subprocess, "run", fake_run):
        with diarize_mod._pcm_input(input_file, quiet=True) as path:
            seen.append(path.suffix)
            assert path.exists()

    assert seen == [".wav"]
    # ffmpeg_bin() probes the binary first, so pick the call that does the work.
    decode_calls = [cmd for cmd in ffmpeg_calls if "pcm_s16le" in cmd]
    assert decode_calls, "the mp3 must be decoded before pyannote sees it"
    cmd = decode_calls[0]
    assert "16000" in cmd, "the model runs at 16 kHz"
    assert "1" == cmd[cmd.index("-ac") + 1], "mono"


def test_diarize_passes_wav_through_untouched(tmp_path: Path) -> None:
    from podcast_reels_forge.scripts import diarize as diarize_mod

    wav = tmp_path / "audio.wav"
    wav.write_text("dummy")

    def explode(*_a: object, **_kw: object) -> None:
        raise AssertionError("a .wav must not be re-encoded")

    with patch.object(diarize_mod.subprocess, "run", explode):
        with diarize_mod._pcm_input(wav, quiet=True) as path:
            assert path == wav
