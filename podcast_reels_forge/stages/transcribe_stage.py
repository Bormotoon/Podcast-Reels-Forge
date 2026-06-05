"""Transcription stage CLI and helpers."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# RU: Xet-протокол HF на некоторых сетях зависает (сокет в CLOSE-WAIT) — форсируем
#     обычный HTTP. Должно быть выставлено ДО импорта faster_whisper/huggingface_hub,
#     т.к. флаг читается в константу при их импорте. setdefault оставляет override.
# EN: HF Xet protocol hangs on some networks (CLOSE-WAIT socket) — force plain HTTP.
#     Must run BEFORE faster_whisper/huggingface_hub import (the flag is read into a
#     constant at their import time). setdefault keeps it overridable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from faster_whisper import BatchedInferencePipeline, WhisperModel

try:
    import torch
    _torch_available = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _torch_available = False

from podcast_reels_forge.utils.logging_utils import setup_logging

CUDA_MAJOR_FLOAT16_THRESHOLD: Final = 7

LOGGER = setup_logging()


@dataclass(frozen=True)
class TranscribeConfig:
    """Configuration for running a transcription job."""

    input_path: Path
    outdir: Path | None
    model_name: str
    device: str
    language: str
    beam_size: int
    compute_type: str | None
    word_timestamps: bool = True
    vad_filter: bool = True
    # RU: False ломает цепочку галлюцинаций (бесконечное "Спасибо." на тишине/музыке).
    # EN: False breaks the hallucination death-spiral (endless "Спасибо." on silence/music).
    condition_on_previous_text: bool = False
    best_of: int = 1
    patience: float = 1.0
    # RU: Батчевый инференс — основной рычаг скорости на GPU.
    # EN: Batched inference — the main GPU speed lever.
    batch_size: int = 16
    # RU: Прямой штраф за повторы внутри декодирования.
    # EN: Direct anti-repetition controls inside decoding.
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    quiet: bool = False
    verbose: bool = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """RU: Парсит аргументы CLI для стадии транскрибации.

    EN: Parse CLI args for the transcription stage.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video with faster-whisper.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input audio/video file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Directory to save the JSON transcript.",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Faster-whisper model name or path (default: large-v3).",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="auto",
        help="Run inference on CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--language",
        default="ru",
        help="Language code (ru/en) or 'auto' (default: ru).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5).",
    )
    parser.add_argument(
        "--compute-type",
        choices=("auto", "float32", "float16", "int8", "int8_float16", "int8_float32"),
        help=(
            "Override compute_type passed to faster-whisper. "
            "Default: int8_float16 on older GPUs, float16 on newer CUDA GPUs, float32 on CPU."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batched-inference batch size; main GPU speed lever (default: 16).",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Number of candidates when sampling (default: 1).",
    )
    parser.add_argument(
        "--patience",
        type=float,
        default=1.0,
        help="Beam search patience (default: 1.0).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty for repeated tokens; curbs hallucination loops (default: 1.1).",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="Block repeating n-grams of this size (default: 3).",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help=(
            "Feed prior text as context. OFF by default: leaving it on triggers the "
            "endless-repetition hallucination on silence/music."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error output")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args(argv)


def resolve_device(requested: str) -> str:
    """RU: Приводит желаемое устройство (cuda/cpu) к реально доступному.

    EN: Resolve requested device to an actually available device.
    """
    req = str(requested).strip().lower()
    if req == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if req == "cuda":
        if torch is None:
            LOGGER.warning("CUDA support requested but torch is not installed; using CPU")
        elif torch.cuda.is_available():
            return "cuda"
        else:
            LOGGER.warning("CUDA not available, falling back to CPU")
    return "cpu"


def _default_compute_type(resolved_device: str) -> str:
    """Select default compute type based on device capability."""
    if resolved_device != "cuda" or torch is None:
        return "float32"
    try:
        major, _minor = torch.cuda.get_device_capability()
    except (RuntimeError, AttributeError):
        return "float32"
    if major < CUDA_MAJOR_FLOAT16_THRESHOLD:
        return "int8_float16"
    return "float16"


def _select_compute_type(resolved_device: str, requested: str | None) -> str:
    """Return explicit compute type or fall back to default."""
    if requested:
        if str(requested).strip().lower() == "auto":
            return _default_compute_type(resolved_device)
        return requested
    return _default_compute_type(resolved_device)


def _load_model(model_name: str, resolved_device: str, compute_type: str) -> WhisperModel:
    """Load the Whisper model with chosen device and compute type."""
    return WhisperModel(model_name, device=resolved_device, compute_type=compute_type)


# RU: Лестница температур — главный предохранитель от галлюцинаций. Если сегмент
#     получает высокий compression_ratio (повторы) или низкий logprob, faster-whisper
#     повторяет его на следующей температуре.
# EN: Temperature fallback ladder — the main anti-hallucination safety net. A segment
#     with high compression_ratio (repeats) or low logprob is retried at the next temp.
TEMPERATURE_LADDER: Final = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _transcribe_with_optional_kwargs(
    model: WhisperModel,
    input_path: Path,
    *,
    language: str | None,
    beam_size: int,
    word_timestamps: bool,
    vad_filter: bool,
    condition_on_previous_text: bool,
    best_of: int,
    patience: float,
    batch_size: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> tuple[Any, Any]:
    """RU: Запускает батчевый pipeline faster-whisper с анти-галлюцинационными флагами.

    EN: Run faster-whisper batched pipeline with anti-hallucination flags.
    """
    kwargs: dict[str, Any] = {
        "language": language,
        "beam_size": beam_size,
        "word_timestamps": word_timestamps,
        "vad_filter": vad_filter,
        "condition_on_previous_text": condition_on_previous_text,
        "best_of": max(1, int(best_of)),
        "patience": max(1.0, float(patience)),
        "temperature": TEMPERATURE_LADDER,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "repetition_penalty": max(1.0, float(repetition_penalty)),
        "no_repeat_ngram_size": max(0, int(no_repeat_ngram_size)),
    }

    batched = BatchedInferencePipeline(model=model)
    try:
        return batched.transcribe(
            str(input_path),
            batch_size=max(1, int(batch_size)),
            **kwargs,
        )
    except TypeError:
        # RU: Запасной путь для несовместимой версии API.
        # EN: Fallback for an incompatible API version.
        return model.transcribe(str(input_path), language=language, beam_size=beam_size)


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        and ("cuda" in msg or "cudnn" in msg or "cublas" in msg or "gpu" in msg)
    )


def _dump_output(out_path: Path, output: dict[str, object]) -> None:
    """Write transcription output to disk."""
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp (HH:MM:SS,mmm)."""
    total_ms = max(0, int(round(seconds * 1000.0)))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder = remainder % 60_000
    secs = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _dump_srt_output(srt_path: Path, segments: list[dict[str, Any]]) -> None:
    """Write SRT subtitles based on transcription segments."""
    lines: list[str] = []
    for idx, seg in enumerate(segments, 1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = str(seg.get("text", "")).strip()
        lines.append(str(idx))
        lines.append(f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")

    with srt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _word_to_dict(word: Any) -> dict[str, Any]:
    probability = getattr(word, "probability", None)
    if isinstance(probability, (int, float)):
        probability_value: float | None = round(float(probability), 3)
    else:
        probability_value = None
    return {
        "start": round(float(getattr(word, "start", 0.0)), 3),
        "end": round(float(getattr(word, "end", 0.0)), 3),
        "word": str(getattr(word, "word", getattr(word, "text", ""))).strip(),
        "probability": probability_value,
    }


def _segment_confidence(segment: Any) -> float | None:
    avg_logprob = getattr(segment, "avg_logprob", None)
    if not isinstance(avg_logprob, (int, float)):
        return None
    try:
        # Convert rough average log-probability into a bounded confidence score.
        score = 1.0 + float(avg_logprob)
    except (TypeError, ValueError):
        return None
    return round(max(0.0, min(1.0, score)), 3)


def _build_sentence_groups(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build sentence-like groups from the transcript segments."""

    sentences: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(str(item.get("text", "")).strip() for item in current).strip()
        if not text:
            current.clear()
            return
        sentences.append(
            {
                "start": round(float(current[0].get("start", 0.0)), 3),
                "end": round(float(current[-1].get("end", 0.0)), 3),
                "text": text,
                "speaker": current[0].get("speaker") or "",
                "segment_count": len(current),
            },
        )
        current.clear()

    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        current.append(segment)
        if re.search(r"[.!?…！？]$", text) or len(" ".join(item["text"] for item in current)) >= 140:
            flush()

    flush()
    return sentences


def transcribe_file(config: TranscribeConfig) -> Path:
    """RU: Запускает транскрибацию faster-whisper и записывает JSON транскрипт.

    EN: Run faster-whisper transcription and write a transcript JSON.
    """
    if not config.input_path.exists():
        message = f"Input file not found: {config.input_path}"
        raise SystemExit(message)

    resolved_device = resolve_device(config.device)
    compute_type = _select_compute_type(resolved_device, config.compute_type)

    model: WhisperModel | None = None
    try:
        def _cleanup_cuda() -> None:
            gc.collect()
            if resolved_device == "cuda" and torch is not None:
                try:
                    torch.cuda.empty_cache()
                except (AttributeError, RuntimeError):
                    pass

        # RU: Пытаемся загрузить модель. При OOM на CUDA делаем деградацию.
        # EN: Try to load model. If CUDA OOM happens, degrade settings.
        try:
            try:
                model = _load_model(config.model_name, resolved_device, compute_type)
            except ValueError:
                compute_type = "float32"
                model = _load_model(config.model_name, resolved_device, compute_type)
        except RuntimeError as exc:
            if resolved_device == "cuda" and _is_cuda_oom(exc):
                LOGGER.warning(
                    "CUDA OOM during model init; falling back. model=%s compute_type=%s",
                    config.model_name,
                    compute_type,
                )
                _cleanup_cuda()

                # Prefer smaller types on CUDA first, then CPU.
                fallback_attempts = ["float16", "int8_float16", "int8"]
                loaded = False
                for ct in fallback_attempts:
                    try:
                        model = _load_model(config.model_name, "cuda", ct)
                        compute_type = ct
                        resolved_device = "cuda"
                        loaded = True
                        break
                    except Exception:
                        _cleanup_cuda()
                        continue

                if not loaded:
                    LOGGER.warning("CUDA OOM persists; switching transcription to CPU")
                    resolved_device = "cpu"
                    compute_type = "float32"
                    model = _load_model(config.model_name, resolved_device, compute_type)
            else:
                raise

        if model is None:
            raise RuntimeError("Whisper model failed to initialize")

        lang: str | None
        if str(config.language).strip().lower() == "auto":
            lang = None
        else:
            lang = config.language

        if config.verbose and not config.quiet:
            LOGGER.info("[transcribe] input=%s", config.input_path)

        # RU: Прогон с деградацией при OOM: GPU(batch) → GPU(batch/2) → … → CPU.
        #     Высокий batch_size даёт скорость; лесенка гарантирует, что мы не упадём.
        # EN: Run with OOM degradation: GPU(batch) → GPU(batch/2) → … → CPU.
        #     High batch_size buys speed; the ladder guarantees we never hard-fail.
        cur_batch = max(1, int(config.batch_size))
        while True:
            try:
                segments, info = _transcribe_with_optional_kwargs(
                    model,
                    config.input_path,
                    language=lang,
                    beam_size=config.beam_size,
                    word_timestamps=bool(config.word_timestamps),
                    vad_filter=bool(config.vad_filter),
                    condition_on_previous_text=bool(config.condition_on_previous_text),
                    best_of=int(config.best_of),
                    patience=float(config.patience),
                    batch_size=cur_batch,
                    repetition_penalty=float(config.repetition_penalty),
                    no_repeat_ngram_size=int(config.no_repeat_ngram_size),
                )
                break
            except RuntimeError as exc:
                if not (resolved_device == "cuda" and _is_cuda_oom(exc)):
                    raise
                _cleanup_cuda()
                if cur_batch > 1:
                    cur_batch = max(1, cur_batch // 2)
                    LOGGER.warning(
                        "CUDA OOM during transcription; retrying on GPU with batch_size=%d",
                        cur_batch,
                    )
                    continue
                LOGGER.warning(
                    "CUDA OOM persists at batch_size=1; switching transcription to CPU",
                )
                resolved_device = "cpu"
                compute_type = "float32"
                model = _load_model(config.model_name, resolved_device, compute_type)

        segments_list = list(segments)
        segment_dicts: list[dict[str, Any]] = []
        for seg in segments_list:
            raw_words = getattr(seg, "words", None)
            words = []
            if isinstance(raw_words, (list, tuple)):
                words = [
                    _word_to_dict(word)
                    for word in raw_words
                    if str(getattr(word, "word", getattr(word, "text", ""))).strip()
                ]
            speaker_raw = getattr(seg, "speaker", None)
            speaker = speaker_raw if isinstance(speaker_raw, str) and speaker_raw.strip() else None
            avg_logprob = getattr(seg, "avg_logprob", None)
            segment_dicts.append(
                {
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "text": str(seg.text).strip(),
                    "confidence": _segment_confidence(seg),
                    "avg_logprob": (
                        round(float(avg_logprob), 3)
                        if isinstance(avg_logprob, (int, float))
                        else None
                    ),
                    "speaker": speaker,
                    "words": words,
                },
            )

        sentences = _build_sentence_groups(segment_dicts)
        language_confidence = getattr(info, "language_probability", None)
        try:
            language_confidence = (
                round(float(language_confidence), 3)
                if language_confidence is not None
                else None
            )
        except (TypeError, ValueError):
            language_confidence = None

        output = {
            "audio": str(config.input_path.resolve()),
            "source_audio": str(config.input_path.resolve()),
            "model": config.model_name,
            "device": resolved_device,
            "compute_type": compute_type,
            "language": getattr(info, "language", config.language),
            "language_confidence": language_confidence,
            "duration": getattr(info, "duration", None),
            "timing_version": 2,
            "segments": segment_dicts,
            "sentences": sentences,
        }

        if config.outdir:
            config.outdir.mkdir(parents=True, exist_ok=True)
            out_path = config.outdir / config.input_path.with_suffix(".json").name
        else:
            out_path = config.input_path.with_suffix(".json")
        srt_path = out_path.with_suffix(".srt")

        _dump_output(out_path, output)
        _dump_srt_output(srt_path, output["segments"])

        if not config.quiet:
            LOGGER.info("[transcribe] saved=%s", out_path)
            LOGGER.info("[transcribe] saved=%s", srt_path)

        return out_path
    finally:
        # RU: Явно освобождаем ресурсы модели, чтобы не оставлять GPU память в процессе.
        # EN: Explicitly release model resources to avoid lingering GPU memory.
        model = None
        gc.collect()
        if resolved_device == "cuda" and torch is not None:
            try:
                torch.cuda.empty_cache()
            except (AttributeError, RuntimeError):
                pass


def main(argv: list[str] | None = None) -> None:
    """RU: CLI-точка входа для стадии транскрибации.

    EN: CLI entrypoint for the transcription stage.
    """
    args = parse_args(argv)
    config = TranscribeConfig(
        input_path=args.input,
        outdir=args.outdir,
        model_name=args.model,
        device=args.device,
        language=args.language,
        beam_size=args.beam_size,
        compute_type=args.compute_type,
        best_of=args.best_of,
        patience=args.patience,
        batch_size=args.batch_size,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        condition_on_previous_text=args.condition_on_previous_text,
        quiet=args.quiet,
        verbose=args.verbose,
    )
    transcribe_file(config)
