#!/usr/bin/env python3
"""RU: Отдельный лаунчер только для транскрибации аудио из input/.

EN: Standalone launcher for transcription-only runs on audio files from input/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

LOG = logging.getLogger("ForgeTranscribeOnly")
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


def ensure_venv() -> None:
    """Re-exec into the repo venv if not already in a venv."""
    env_flag = "WHISPER_VENV_ACTIVE"
    if os.environ.get(env_flag) == "1":
        return

    if sys.prefix != sys.base_prefix:
        os.environ[env_flag] = "1"
        return

    repo_dir = Path(__file__).resolve().parent
    venv_python = repo_dir / "whisper-env" / "bin" / "python"
    if venv_python.exists():
        os.environ[env_flag] = "1"
        os.execv(str(venv_python), [str(venv_python), *sys.argv])


def _configure_logging(*, quiet: bool, verbose: bool) -> None:
    level = logging.ERROR if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _is_valid_json(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _resolve_device(device_raw: object) -> str:
    device = str(device_raw or "cuda").strip().lower()
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if device in {"cuda", "cpu"}:
        return device
    return "cpu"


def _find_audio_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []

    files = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe all audio files from input/ using project config.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--quiet", action="store_true", help="Only errors")
    parser.add_argument("--verbose", action="store_true", help="Verbose logs")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip files if transcript JSON already exists",
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="Deprecated no-op: device=auto now always resolves to CUDA when available.",
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "quality"),
        default=None,
        help="Override transcription.mode: fast (batched) or quality (sequential, accurate, slow).",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Override transcription.initial_prompt: domain context to bias vocabulary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ensure_venv()

    import yaml

    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        conf: dict[str, Any] = yaml.safe_load(f) or {}

    cli_conf = conf.get("cli", {}) if isinstance(conf, dict) else {}
    quiet = bool(args.quiet or cli_conf.get("quiet", False))
    verbose = bool(args.verbose or cli_conf.get("verbose", False))
    _configure_logging(quiet=quiet, verbose=verbose)

    cache_conf = conf.get("cache", {}) if isinstance(conf, dict) else {}
    validate_json = bool(cache_conf.get("validate_json", True))
    skip_existing = bool(cache_conf.get("enabled", True)) and not args.no_skip_existing

    paths_conf = conf.get("paths", {}) if isinstance(conf, dict) else {}
    input_dir = Path(str(paths_conf.get("input_dir", "input")))
    output_dir = Path(str(paths_conf.get("output_dir", "output")))

    audio_files = _find_audio_files(input_dir)
    if not audio_files:
        if not quiet:
            print(f"No audio files found in {input_dir}", flush=True)
        return

    t_conf = conf.get("transcription", {}) if isinstance(conf, dict) else {}
    model_name = str(t_conf.get("model", "small"))
    language = str(t_conf.get("language", "ru"))
    beam_size = int(t_conf.get("beam_size", 5))
    best_of = int(t_conf.get("best_of", 1))
    patience = float(t_conf.get("patience", 1.0))
    batch_size = int(t_conf.get("batch_size", 16))
    repetition_penalty = float(t_conf.get("repetition_penalty", 1.1))
    no_repeat_ngram_size = int(t_conf.get("no_repeat_ngram_size", 3))
    condition_on_previous_text = bool(t_conf.get("condition_on_previous_text", False))
    mode = str(args.mode or t_conf.get("mode", "fast"))
    initial_prompt = args.initial_prompt or t_conf.get("initial_prompt") or None
    quality_beam_size = int(t_conf.get("quality_beam_size", 10))

    compute_type_raw = t_conf.get("compute_type")
    compute_type: str | None = None
    if isinstance(compute_type_raw, str) and compute_type_raw.strip():
        ct = compute_type_raw.strip()
        if ct.lower() != "auto":
            compute_type = ct

    device = _resolve_device(t_conf.get("device", "cuda"))

    if not quiet:
        print(f"Found {len(audio_files)} audio file(s) in {input_dir}", flush=True)

    from podcast_reels_forge.stages.transcribe_stage import (
        TranscribeConfig,
        transcribe_file,
    )

    done = 0
    skipped = 0
    failed = 0

    for audio_path in audio_files:
        file_outdir = output_dir / audio_path.stem
        transcript_path = file_outdir / audio_path.with_suffix(".json").name

        if skip_existing and transcript_path.exists():
            if not validate_json or _is_valid_json(transcript_path):
                skipped += 1
                if verbose and not quiet:
                    print(f"[skip] {audio_path.name} -> {transcript_path}", flush=True)
                continue

        if not quiet:
            print(f"[transcribe] {audio_path.name}", flush=True)

        cfg = TranscribeConfig(
            input_path=audio_path,
            outdir=file_outdir,
            model_name=model_name,
            device=device,
            language=language,
            beam_size=beam_size,
            compute_type=compute_type,
            best_of=best_of,
            patience=patience,
            batch_size=batch_size,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            condition_on_previous_text=condition_on_previous_text,
            mode=mode,
            initial_prompt=initial_prompt,
            quality_beam_size=quality_beam_size,
            quiet=quiet,
            verbose=verbose,
        )

        try:
            out_path = transcribe_file(cfg)
            done += 1
            if verbose and not quiet:
                print(f"[saved] {out_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            LOG.error("Failed to transcribe %s: %s", audio_path, exc)

    if not quiet:
        print(
            f"Completed. done={done} skipped={skipped} failed={failed}",
            flush=True,
        )

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
