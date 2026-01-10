"""Tests for transcription stage logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from podcast_reels_forge.stages.transcribe_stage import (
    TranscribeConfig,
    resolve_device,
    _select_compute_type,
    transcribe_file,
)

def test_resolve_device_cpu() -> None:
    # Force cpu
    assert resolve_device("cpu") == "cpu"

@patch("podcast_reels_forge.stages.transcribe_stage.torch")
def test_resolve_device_cuda_available(mock_torch: MagicMock) -> None:
    mock_torch.cuda.is_available.return_value = True
    assert resolve_device("cuda") == "cuda"

@patch("podcast_reels_forge.stages.transcribe_stage.torch")
def test_resolve_device_cuda_not_available(mock_torch: MagicMock) -> None:
    mock_torch.cuda.is_available.return_value = False
    assert resolve_device("cuda") == "cpu"

def test_select_compute_type_explicit() -> None:
    assert _select_compute_type("cuda", "int8") == "int8"
    assert _select_compute_type("cpu", "float32") == "float32"

@patch("podcast_reels_forge.stages.transcribe_stage.torch")
def test_select_compute_type_default_new_gpu(mock_torch: MagicMock) -> None:
    mock_torch.cuda.get_device_capability.return_value = (8, 0)
    # CUDA_MAJOR_FLOAT16_THRESHOLD is 7, 8 >= 7 so float16
    assert _select_compute_type("cuda", None) == "float16"

@patch("podcast_reels_forge.stages.transcribe_stage.torch")
def test_select_compute_type_default_old_gpu(mock_torch: MagicMock) -> None:
    mock_torch.cuda.get_device_capability.return_value = (6, 1)
    # 6 < 7 so float32
    assert _select_compute_type("cuda", None) == "float32"


@patch("podcast_reels_forge.stages.transcribe_stage.WhisperModel")
@patch("podcast_reels_forge.stages.transcribe_stage.torch")
def test_transcribe_file_cleans_cuda_cache(
    mock_torch: MagicMock, mock_whisper: MagicMock, tmp_path: Path,
) -> None:
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.empty_cache.return_value = None

    input_file = tmp_path / "audio.mp3"
    input_file.write_text("dummy")

    config = TranscribeConfig(
        input_path=input_file,
        outdir=tmp_path / "out",
        model_name="tiny",
        device="cuda",
        language="en",
        beam_size=1,
        compute_type="float16",
        quiet=True,
        verbose=False,
    )

    mock_model_instance = mock_whisper.return_value
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.duration = 1.0

    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = " hi "

    mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)

    out_path = transcribe_file(config)
    assert out_path.exists()
    mock_torch.cuda.empty_cache.assert_called()

@patch("podcast_reels_forge.stages.transcribe_stage.WhisperModel")
def test_transcribe_file_orchestration(mock_whisper: MagicMock, tmp_path: Path) -> None:
    input_file = tmp_path / "audio.mp3"
    input_file.write_text("dummy")
    
    config = TranscribeConfig(
        input_path=input_file,
        outdir=tmp_path / "out",
        model_name="tiny",
        device="cpu",
        language="en",
        beam_size=1,
        compute_type="float32",
        quiet=True,
        verbose=False
    )
    
    # Mock model behaviors
    mock_model_instance = mock_whisper.return_value
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.duration = 10.0
    
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.0
    mock_segment.text = " Hello world "
    
    mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
    
    out_path = transcribe_file(config)
    
    assert out_path.exists()
    assert out_path.name == "audio.json"
    
    import json
    with out_path.open(encoding="utf-8") as f:
        data = json.load(f)
    
    assert data["language"] == "en"
    assert data["segments"][0]["text"] == "Hello world"
