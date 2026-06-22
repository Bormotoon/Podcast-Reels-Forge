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
    # Older CUDA now prefers the safer mixed-precision path.
    assert _select_compute_type("cuda", None) == "int8_float16"


@patch("podcast_reels_forge.stages.transcribe_stage.BatchedInferencePipeline")
@patch("podcast_reels_forge.stages.transcribe_stage.WhisperModel")
@patch("podcast_reels_forge.stages.transcribe_stage.torch")
def test_transcribe_file_cleans_cuda_cache(
    mock_torch: MagicMock,
    mock_whisper: MagicMock,
    mock_batched: MagicMock,
    tmp_path: Path,
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

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.duration = 1.0

    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = " hi "

    mock_batched.return_value.transcribe.return_value = ([mock_segment], mock_info)

    out_path = transcribe_file(config)
    assert out_path.exists()
    mock_torch.cuda.empty_cache.assert_called()

@patch("podcast_reels_forge.stages.transcribe_stage.BatchedInferencePipeline")
@patch("podcast_reels_forge.stages.transcribe_stage.WhisperModel")
def test_transcribe_file_orchestration(
    mock_whisper: MagicMock, mock_batched: MagicMock, tmp_path: Path,
) -> None:
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
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.duration = 10.0

    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.0
    mock_segment.text = " Hello world "

    mock_batched.return_value.transcribe.return_value = ([mock_segment], mock_info)

    out_path = transcribe_file(config)
    srt_path = out_path.with_suffix(".srt")
    
    assert out_path.exists()
    assert out_path.name == "audio.json"
    assert srt_path.exists()
    
    import json
    with out_path.open(encoding="utf-8") as f:
        data = json.load(f)
    
    assert data["language"] == "en"
    assert data["segments"][0]["text"] == "Hello world"

    srt_text = srt_path.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:02,000" in srt_text
    assert "Hello world" in srt_text


@patch("podcast_reels_forge.stages.transcribe_stage.BatchedInferencePipeline")
@patch("podcast_reels_forge.stages.transcribe_stage.WhisperModel")
def test_transcribe_file_quality_mode_uses_sequential(
    mock_whisper: MagicMock, mock_batched: MagicMock, tmp_path: Path,
) -> None:
    """Quality mode must use the sequential model.transcribe with context, not batching."""
    input_file = tmp_path / "audio.mp3"
    input_file.write_text("dummy")

    config = TranscribeConfig(
        input_path=input_file,
        outdir=tmp_path / "out",
        model_name="tiny",
        device="cpu",
        language="ru",
        beam_size=5,
        compute_type="float32",
        mode="quality",
        initial_prompt="Родительское собрание.",
        quality_beam_size=10,
        quiet=True,
        verbose=False,
    )

    mock_info = MagicMock()
    mock_info.language = "ru"
    mock_info.duration = 5.0
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = " привет "
    mock_whisper.return_value.transcribe.return_value = ([mock_segment], mock_info)

    out_path = transcribe_file(config)
    assert out_path.exists()

    # Sequential path used, batched pipeline NOT used.
    mock_whisper.return_value.transcribe.assert_called_once()
    mock_batched.return_value.transcribe.assert_not_called()

    # Quality settings were applied.
    _, kwargs = mock_whisper.return_value.transcribe.call_args
    assert kwargs["condition_on_previous_text"] is True
    assert kwargs["beam_size"] == 10
    assert kwargs["initial_prompt"] == "Родительское собрание."
    assert "vad_parameters" in kwargs

    import json
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["mode"] == "quality"
