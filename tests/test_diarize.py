"""Tests for diarization script."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from podcast_reels_forge.scripts.diarize import main

def test_diarize_missing_token(tmp_path: Path) -> None:
    input_file = tmp_path / "audio.mp3"
    input_file.write_text("dummy")
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(SystemExit, match="PYANNOTE_TOKEN is not set"):
            main(["--input", str(input_file)])

@patch("podcast_reels_forge.scripts.diarize.os.environ.get")
def test_diarize_success(mock_env_get: MagicMock, tmp_path: Path) -> None:
    mock_env_get.return_value = "fake_token"
    input_file = tmp_path / "audio.mp3"
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
    
    with patch.dict("sys.modules", {"pyannote.audio": MagicMock(Pipeline=mock_pipeline_class)}):
        main(["--input", str(input_file), "--outdir", str(out_dir)])
    
    out_json = out_dir / "diarization.json"
    assert out_json.exists()
    
    with out_json.open(encoding="utf-8") as f:
        data = json.load(f)
    assert data[0]["speaker"] == "SPEAKER_01"
    assert data[0]["start"] == 1.0
