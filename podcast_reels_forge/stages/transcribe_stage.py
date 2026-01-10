"""Transcription stage CLI and helpers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from faster_whisper import WhisperModel

try:
    import torch
except ImportError:
    torch = None

from podcast_reels_forge.utils.logging_utils import setup_logging

CUDA_MAJOR_FLOAT16_THRESHOLD: Final = 7

LOGGER = setup_logging()


@dataclass(frozen=True)
class TranscribeConfig:
    input_path: Path
    outdir: Path | None
    model_name: str
    device: str
    language: str
    beam_size: int
    compute_type: str | None
    quiet: bool
    verbose: bool


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
        default="medium",
        help="Faster-whisper model name or path (default: medium).",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Run inference on CUDA when available.",
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
        choices=("float32", "float16", "int8", "int8_float16", "int8_float32"),
        help=(
            "Override compute_type passed to faster-whisper. "
            "Default: int8_float16 on older GPUs, float16 on newer CUDA GPUs, float32 on CPU."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress non-error output")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args(argv)


def resolve_device(requested: str) -> str:
    """RU: Приводит желаемое устройство (cuda/cpu) к реально доступному.

    EN: Resolve requested device to an actually available device.
    """
    if requested == "cuda":
        if torch is None:
            LOGGER.warning("CUDA support requested but torch is not installed; using CPU")
        elif torch.cuda.is_available():
            return "cuda"
        else:
            LOGGER.warning("CUDA not available, falling back to CPU")
    return "cpu"


def _default_compute_type(resolved_device: str) -> str:
    if resolved_device != "cuda" or torch is None:
        return "float32"
    try:
        major, _minor = torch.cuda.get_device_capability()
    except (RuntimeError, AttributeError):
        return "float32"
    if major < CUDA_MAJOR_FLOAT16_THRESHOLD:
        return "float32"
    return "float16"


def _select_compute_type(resolved_device: str, requested: str | None) -> str:
    if requested:
        return requested
    return _default_compute_type(resolved_device)


def _load_model(model_name: str, resolved_device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_name, device=resolved_device, compute_type=compute_type)


def _dump_output(out_path: Path, output: dict[str, object]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def transcribe_file(config: TranscribeConfig) -> Path:
    """RU: Запускает транскрибацию faster-whisper и записывает JSON транскрипт.

    EN: Run faster-whisper transcription and write a transcript JSON.
    """
    if not config.input_path.exists():
        message = f"Input file not found: {config.input_path}"
        raise SystemExit(message)

    resolved_device = resolve_device(config.device)
    compute_type = _select_compute_type(resolved_device, config.compute_type)

    try:
        model = _load_model(config.model_name, resolved_device, compute_type)
    except ValueError:
        compute_type = "float32"
        model = _load_model(config.model_name, resolved_device, compute_type)

    lang: str | None = None if str(config.language).strip().lower() == "auto" else config.language

    if config.verbose and not config.quiet:
        LOGGER.info("[transcribe] input=%s", config.input_path)

    segments, info = model.transcribe(
        str(config.input_path),
        language=lang,
        beam_size=config.beam_size,
    )

    output = {
        "audio": str(config.input_path.resolve()),
        "model": config.model_name,
        "device": resolved_device,
        "compute_type": compute_type,
        "language": info.language,
        "duration": info.duration,
        "segments": [
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            }
            for seg in segments
        ],
    }

    if config.outdir:
        config.outdir.mkdir(parents=True, exist_ok=True)
        out_path = config.outdir / config.input_path.with_suffix(".json").name
    else:
        out_path = config.input_path.with_suffix(".json")

    _dump_output(out_path, output)

    if not config.quiet:
        LOGGER.info("[transcribe] saved=%s", out_path)

    return out_path


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
        quiet=args.quiet,
        verbose=args.verbose,
    )
    transcribe_file(config)
