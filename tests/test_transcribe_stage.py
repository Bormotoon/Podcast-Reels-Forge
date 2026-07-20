"""Tests for transcription stage logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from podcast_reels_forge.stages.transcribe_stage import (
    TranscribeConfig,
    resolve_device,
    _dump_srt_output,
    _segment_to_srt_cues,
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


def test_segment_to_srt_cues_splits_multiple_sentences() -> None:
    # Full-length sentences (each over the merge threshold) stay one per cue.
    seg = {
        "start": 0.0,
        "end": 12.0,
        "text": (
            "Сегодня мы обсудим очень важную и большую тему целиком. "
            "Наш гость расскажет о своём необычном профессиональном пути. "
            "А в конце выпуска мы ответим на вопросы наших слушателей."
        ),
    }
    cues = _segment_to_srt_cues(seg)

    assert [c[2] for c in cues] == [
        "Сегодня мы обсудим очень важную и большую тему целиком.",
        "Наш гость расскажет о своём необычном профессиональном пути.",
        "А в конце выпуска мы ответим на вопросы наших слушателей.",
    ]
    # Cues stay inside the segment span and advance monotonically.
    assert cues[0][0] == 0.0
    assert cues[-1][1] == 12.0
    for earlier, later in zip(cues, cues[1:]):
        assert earlier[1] <= later[0] + 1e-6


def test_segment_to_srt_cues_merges_two_short_sentences() -> None:
    # Two short sentences may share a cue (at most two), never three.
    seg = {"start": 0.0, "end": 6.0, "text": "Да. Нет. Может быть."}
    cues = _segment_to_srt_cues(seg)
    assert [c[2] for c in cues] == ["Да. Нет.", "Может быть."]


def test_segment_to_srt_cues_wraps_long_punctuationless_run() -> None:
    words = " ".join(f"слово{i}" for i in range(60))
    seg = {"start": 0.0, "end": 30.0, "text": words}
    cues = _segment_to_srt_cues(seg)
    assert len(cues) > 1
    assert all(len(c[2]) <= 140 for c in cues)
    # No text is lost during wrapping.
    assert " ".join(c[2] for c in cues) == words


def test_dump_srt_output_emits_one_sentence_per_cue(tmp_path: Path) -> None:
    first = "Это первое достаточно длинное предложение выпуска."
    second = "А это уже второе не менее длинное предложение выпуска."
    third = "Третье длинное предложение целиком здесь."
    segments = [
        {"start": 0.0, "end": 8.0, "text": f"{first} {second}"},
        {"start": 8.0, "end": 11.0, "text": third},
    ]
    srt_path = tmp_path / "out.srt"
    _dump_srt_output(srt_path, segments)
    text = srt_path.read_text(encoding="utf-8")

    # Three cues total, numbered 1..3, one sentence each.
    assert "1\n" in text
    assert "3\n" in text
    assert "4\n" not in text
    assert first in text
    assert second in text
    assert third in text
    # The two sentences of the first segment are on separate cues, not one line.
    assert f"{first} {second}" not in text
