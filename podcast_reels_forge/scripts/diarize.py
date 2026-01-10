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
import json
import os
from pathlib import Path

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
    ap.add_argument("--quiet", action="store_true", help="Suppress non-error output")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    return ap.parse_args(argv)


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

    pipeline = Pipeline.from_pretrained(args.model, use_auth_token=token)
    diarization = pipeline(str(args.input))

    items: list[dict[str, object]] = []
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
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
