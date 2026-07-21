#!/usr/bin/env python3
"""RU: Опциональная стадия диаризации (определение спикеров).

Стадия намеренно опциональна, потому что требует дополнительных зависимостей и
скачивания моделей.

Включение через config.yaml:
    diarization:
        enabled: true

Требования:
- установлен `pyannote-audio` в окружении
- переменная окружения `PYANNOTE_TOKEN` для доступа к скачиванию моделей

EN: Optional speaker diarization stage.

This stage is intentionally optional because it requires extra dependencies and
model downloads.

Enable via config.yaml:
    diarization:
        enabled: true

Requirements:
- `pyannote-audio` installed in the environment
- `PYANNOTE_TOKEN` environment variable for model download access
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from podcast_reels_forge.utils.ffmpeg import ffmpeg_bin
from podcast_reels_forge.utils.logging_utils import setup_logging

LOGGER = setup_logging()


def _status(msg: str, *, quiet: bool) -> None:
    if not quiet:
        LOGGER.info(msg)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """RU: Парсит аргументы командной строки.

    EN: Parse command line arguments.
    """
    ap = argparse.ArgumentParser(
        description="Optional speaker diarization using pyannote-audio.",
    )
    ap.add_argument(
        "--input", type=Path, required=True, help="Path to input audio/video file",
    )
    ap.add_argument(
        "--outdir", type=Path, default=Path("out"), help="Output directory",
    )
    ap.add_argument(
        "--model", default="pyannote/speaker-diarization", help="pyannote model id",
    )
    ap.add_argument(
        "--num-speakers", type=int, default=None,
        help="Exact number of speakers, if known (curbs over-clustering on noise/overlap)",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress non-error output")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    return ap.parse_args(argv)


@contextlib.contextmanager
def _pcm_input(path: Path, *, quiet: bool) -> Iterator[Path]:
    """RU: Отдаёт 16 кГц моно WAV для входа, декодируя его при необходимости.

    EN: Yield a 16 kHz mono WAV for *path*, decoding it first when needed.
    """
    if path.suffix.lower() == ".wav":
        yield path
        return

    with tempfile.TemporaryDirectory(prefix="forge-diarize-") as tmp:
        wav_path = Path(tmp) / (path.stem + ".wav")
        _status(f"[diarize] decoding {path.name} to 16 kHz mono WAV", quiet=quiet)
        res = subprocess.run(
            [
                ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(path),
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path),
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            detail = (res.stderr or res.stdout or "unknown ffmpeg error").strip()
            raise SystemExit(
                f"Failed to decode {path.name} for diarization: {detail[-400:]}",
            )
        yield wav_path


def main(argv: list[str] | None = None) -> None:
    """RU: Точка входа для стадии диаризации.

    EN: Main entry point for diarization stage.
    """
    args = parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    token = os.environ.get("PYANNOTE_TOKEN")
    if not token:
        raise SystemExit(
            "PYANNOTE_TOKEN is not set (required when diarization is enabled)",
        )

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        message = (
            "pyannote-audio is not installed. Install it and its dependencies to use diarization. "
            f"Original error: {exc}"
        )
        raise SystemExit(message) from exc

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / "diarization.json"

    if args.verbose:
        _status(f"[diarize] model={args.model}", quiet=args.quiet)

    # The signature and return type differ across pyannote.audio majors, so
    # this whole block is checked at runtime rather than by mypy.
    # Untyped on purpose: the signature differs across pyannote majors (4.x
    # renamed use_auth_token to token), and with pyannote absent the class is
    # Any anyway — a type: ignore here would flip between needed and unused
    # depending on the environment.
    load_pipeline: Any = Pipeline.from_pretrained
    try:
        pipeline = load_pipeline(args.model, token=token)
    except TypeError:
        # RU: pyannote.audio <4.0 использует старое имя аргумента.
        # EN: pyannote.audio <4.0 uses the old argument name.
        pipeline = load_pipeline(args.model, use_auth_token=token)
    if pipeline is None:
        raise SystemExit(f"Failed to load diarization pipeline: {args.model}")

    try:
        import torch
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except ImportError:
        pass

    pipeline_kwargs: dict[str, Any] = {}
    if args.num_speakers:
        pipeline_kwargs["num_speakers"] = args.num_speakers

    # RU: pyannote читает файл кусками и на mp3 падает: обрезка возвращает на
    #     несколько сэмплов меньше запрошенного. Поэтому не-PCM вход сначала
    #     декодируем в 16 кГц моно WAV — ровно то, на чём работает модель.
    # EN: pyannote reads the file in chunks and fails on mp3: a crop comes back
    #     a few samples short of what it asked for. So a non-PCM input is
    #     decoded to 16 kHz mono WAV first — exactly what the model runs on.
    with _pcm_input(args.input, quiet=bool(args.quiet)) as audio_path:
        diarization = pipeline(str(audio_path), **pipeline_kwargs)
    # RU: pyannote.audio>=4.0 возвращает DiarizeOutput (.speaker_diarization),
    #     а не Annotation напрямую.
    # EN: pyannote.audio>=4.0 returns a DiarizeOutput (.speaker_diarization)
    #     instead of an Annotation directly.
    annotation: Any = getattr(diarization, "speaker_diarization", diarization)

    items: list[dict[str, object]] = []
    for turn, _track, speaker in annotation.itertracks(yield_label=True):
        try:
            items.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker),
                },
            )
        except (TypeError, ValueError):
            continue

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    _status(str(out_path), quiet=args.quiet)


if __name__ == "__main__":
    main()
