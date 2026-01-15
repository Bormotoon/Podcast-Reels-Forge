"""RU: Оркестрация пайплайна Podcast Reels Forge.

Модуль содержит логику основного пайплайна и оркестрирует стадии:
1) Транскрибация (аудио/видео → JSON транскрипт)
2) Опциональная диаризация (распознавание спикеров)
3) Анализ (LLM ищет «вирусные» моменты)
4) Обработка видео (нарезка, эффекты, экспорт)

EN: Pipeline orchestration for Podcast Reels Forge.

This module contains the main pipeline logic that orchestrates all stages:
1) Transcription (audio/video → JSON transcript)
2) Optional diarization (speaker identification)
3) Analysis (LLM finds viral moments)
4) Video processing (cut reels, apply effects, export)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable: object, **_: object) -> object:
        return iterable

from podcast_reels_forge.utils.ollama_service import (
    ENV_MANAGED_BY_PIPELINE,
    ollama_start,
    ollama_stop,
    parse_local_ollama_host_port,
)

log = logging.getLogger("Forge")


def _model_folder_name(model: str) -> str:
    """Map a model id to a stable folder name.

    We keep folder names short and human-friendly.
    """
    m = (model or "").strip().lower()
    if m.startswith("qwen3"):
        return "qwen3"
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith("gemma3"):
        return "gemma3"
    if m.startswith("gemma2"):
        return "gemma2"
    if m.startswith("gemini-3"):
        return "gemini3"
    # Fallback: filesystem-safe-ish
    out = []
    for ch in m:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch in (":", "/", ".", " "):
            out.append("_")
    name = "".join(out).strip("_")
    return name or "model"


def _get_allowed_models() -> set[str]:
    return {
        "qwen3:latest",
        "deepseek-r1:8b",
        "gemma3:4b",
        "gemma2:9b",
        "gemini-3-flash-preview:latest",
    }


@dataclass(frozen=True)
class PipelineIO:
    """RU: Пути ввода/вывода для пайплайна.

    EN: Input/output paths for the pipeline.
    """

    input_dir: Path
    output_dir: Path


def pick_input_file(input_dir: Path, suffixes: tuple[str, ...]) -> Path | None:
    """RU: Выбирает самый новый файл с подходящим расширением из директории.

    Аргументы:
        input_dir: Директория для поиска файлов.
        suffixes: Допустимые расширения (с точкой, например, '.mp4').

    Возвращает:
        Путь к самому новому файлу или None, если файлов не найдено.

    EN: Pick the newest matching file from a directory.

    Args:
        input_dir: Directory to search for files.
        suffixes: Tuple of allowed file extensions (with dots, e.g., '.mp4').

    Returns:
        Path to the newest matching file, or None if no files found.

    """
    if not input_dir.exists():
        return None
    files = [
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes
    ]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def status(msg: str, *, quiet: bool) -> None:
    """RU: Печатает короткое статус-сообщение (если не quiet).

    EN: Emit a short status line.
    """
    if not quiet:
        print(msg, flush=True)


def _read_json_if_valid(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _ensure_placeholder_analyze_outputs(moments_path: Path, reels_md_path: Path) -> None:
    moments_path.parent.mkdir(parents=True, exist_ok=True)
    if not moments_path.exists():
        moments_path.write_text("[]\n", encoding="utf-8")
    if not reels_md_path.exists():
        reels_md_path.write_text("# Reels Suggestions\n\n(no moments)\n", encoding="utf-8")


def _outputs_ready(outputs: list[Path], *, validate_json: bool) -> bool:
    for p in outputs:
        if not p.exists():
            return False
        try:
            if p.stat().st_size <= 0:
                return False
        except OSError:
            return False
        if validate_json and p.suffix.lower() == ".json":
            if _read_json_if_valid(p) is None:
                return False
    return True


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _set_cli_arg(args: list[str], flag: str, value: str) -> bool:
    for i, v in enumerate(args):
        if v == flag and i + 1 < len(args):
            args[i + 1] = value
            return True
    return False


def _get_model_overrides(conf: dict[str, Any], model: str) -> dict[str, Any]:
    """Return per-model overrides for a config section, if present."""
    overrides = conf.get("model_overrides")
    if not isinstance(overrides, dict):
        return {}
    ov = overrides.get(model)
    return ov if isinstance(ov, dict) else {}


def _merge_ollama_conf(base: dict[str, Any], model: str) -> dict[str, Any]:
    """Merge base ollama config with per-model overrides.

    Supports shallow overrides, with a nested merge for the 'watchdog' dict.
    """
    merged: dict[str, Any] = dict(base)
    ov = _get_model_overrides(base, model)
    for k, v in ov.items():
        if k == "watchdog" and isinstance(v, dict):
            wd = base.get("watchdog")
            merged_wd: dict[str, Any] = dict(wd) if isinstance(wd, dict) else {}
            merged_wd.update(v)
            merged["watchdog"] = merged_wd
            continue
        merged[k] = v
    return merged


def _prompt_variant_for_model(prompts_conf: dict[str, Any], model: str) -> str:
    variant = str(prompts_conf.get("variant", "default"))
    mv = prompts_conf.get("model_variants")
    if isinstance(mv, dict):
        mvv = mv.get(model)
        if isinstance(mvv, str) and mvv.strip():
            return mvv.strip()
    return variant


def run_module(
    module: str,
    args: list[str],
    *,
    quiet: bool,
    verbose: bool,
    env: dict[str, str] | None = None,
) -> None:
    """RU: Запускает Python-модуль через текущий интерпретатор.

    Аргументы:
        module: Путь модуля (например, 'podcast_reels_forge.scripts.analyze').
        args: Аргументы командной строки для передачи модулю.
        quiet: Подавлять вывод.
        verbose: Показывать подробный вывод.

    EN: Run a Python module using the current interpreter.

    Args:
        module: Module path (e.g., 'podcast_reels_forge.scripts.analyze').
        args: Command line arguments to pass.
        quiet: Suppress output.
        verbose: Show detailed output.

    """
    # Use the current interpreter path. Avoid resolving symlinks: venv python
    # executables are often symlinks to system python, but must be invoked
    # via the venv path so sys.prefix and site-packages stay correct.
    cmd = [sys.executable, "-m", module] + args
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    if verbose:
        log.debug("Running: %s", " ".join(cmd))
        res = subprocess.run(cmd, text=True, env=proc_env)
    else:
        res = subprocess.run(cmd, capture_output=True, text=True, env=proc_env)
    if res.returncode != 0:
        if not verbose:
            if res.stdout:
                print(res.stdout.strip(), file=sys.stderr)
            if res.stderr:
                print(res.stderr.strip(), file=sys.stderr)
        raise SystemExit(res.returncode)


def run_pipeline(
    *,
    conf: dict[str, Any],
    repo_dir: Path,
    quiet: bool,
    verbose: bool,
    skip_existing: bool = True,
    autotune: bool = False,
    progress: bool = True,
) -> None:
    """RU: Запускает полный пайплайн на основе config.yaml и файлов на диске.

    Аргументы:
        conf: Словарь конфигурации, загруженный из config.yaml.
        repo_dir: Путь к корню репозитория (для поиска prompts и т.п.).
        quiet: Подавлять статус-сообщения.
        verbose: Включить подробный лог.

    EN: Run the full pipeline based on config and filesystem discovery.

    Args:
        conf: Configuration dictionary loaded from config.yaml.
        repo_dir: Path to the repository root (for locating prompts, etc.).
        quiet: Suppress status messages.
        verbose: Enable detailed logging.

    """
    paths = conf.get("paths", {})
    input_dir_value = str(paths.get("input_dir", "input"))
    output_dir_value = str(paths.get("output_dir", "output"))
    io = PipelineIO(
        input_dir=Path(input_dir_value), output_dir=Path(output_dir_value),
    )
    os.makedirs(io.output_dir, exist_ok=True)

    cache_conf = conf.get("cache", {}) if isinstance(conf, dict) else {}
    validate_json = bool(cache_conf.get("validate_json", True))
    if "enabled" in cache_conf:
        skip_existing = bool(cache_conf.get("enabled", True)) and skip_existing

    video_exts = (".mp4", ".mkv", ".mov", ".avi")
    audio_exts = (".mp3", ".wav", ".flac", ".m4a")

    video_path = pick_input_file(io.input_dir, video_exts)
    audio_path = pick_input_file(io.input_dir, audio_exts)

    if not video_path:
        raise SystemExit(
            f"No video file found in {io.input_dir}. Put an .mp4/.mkv/.mov/.avi into {io.input_dir}/",
        )

    target_audio = audio_path or video_path

    status(f"[forge] video: {video_path.name}", quiet=quiet)
    status(f"[forge] transcribe input: {Path(target_audio).name}", quiet=quiet)

    diar_conf = conf.get("diarization", {})
    diar_enabled = bool(diar_conf.get("enabled", False))

    # Determine analysis models.
    a_conf = conf.get("ollama", {})
    models_raw = a_conf.get("models")
    if isinstance(models_raw, list) and models_raw:
        models = [str(m).strip() for m in models_raw if str(m).strip()]
    else:
        models = [str(a_conf.get("model", "qwen3:latest")).strip()]

    allowed = _get_allowed_models()
    unknown = [m for m in models if m not in allowed]
    if unknown:
        raise SystemExit(
            "Unsupported models in config. Allowed: "
            + ", ".join(sorted(allowed))
            + "; got: "
            + ", ".join(unknown),
        )

    stages_count = 2 + (1 if diar_enabled else 0) + len(models) + len(models)
    stage_iter = iter(
        tqdm(
            range(stages_count),
            total=stages_count,
            disable=(not progress) or quiet,
            desc="Podcast Reels Forge",
        ),
    )

    # 1) Transcribe
    t_conf = conf.get("transcription", {})

    transcript_path = io.output_dir / target_audio.with_suffix(".json").name
    legacy_audio_json = io.output_dir / "audio.json"
    if not transcript_path.exists() and legacy_audio_json.exists():
        transcript_path = legacy_audio_json

    if skip_existing and _outputs_ready([transcript_path], validate_json=validate_json):
        status(f"[1/{stages_count}] transcribe: skip (exists)", quiet=quiet)
    else:
        status(f"[1/{stages_count}] transcribe: start", quiet=quiet)
        transcribe_args = [
            "--input",
            str(target_audio),
            "--outdir",
            str(io.output_dir),
            "--model",
            str(t_conf.get("model", "small")),
            "--device",
            str(t_conf.get("device", "cuda")),
            "--language",
            str(t_conf.get("language", "ru")),
            "--beam-size",
            str(t_conf.get("beam_size", 5)),
        ]

        compute_type_raw = t_conf.get("compute_type")
        compute_type = None
        if isinstance(compute_type_raw, str) and compute_type_raw.strip():
            compute_type = compute_type_raw.strip()

        device_raw = str(t_conf.get("device", "cuda")).strip().lower()
        if autotune and device_raw == "auto":
            transcribe_args[transcribe_args.index("--device") + 1] = (
                "cuda" if _has_cuda() else "cpu"
            )

        if autotune and compute_type and compute_type.lower() == "auto":
            compute_type = None
        if compute_type and compute_type.lower() != "auto":
            transcribe_args += ["--compute-type", compute_type]

        if quiet:
            transcribe_args.append("--quiet")
        if verbose:
            transcribe_args.append("--verbose")
        run_module(
            "podcast_reels_forge.scripts.transcribe",
            transcribe_args,
            quiet=quiet,
            verbose=verbose,
        )
        status("[transcribe] done", quiet=quiet)

    try:
        next(stage_iter)
    except StopIteration:
        pass

    # 2) Optional diarization
    diar_path = io.output_dir / "diarization.json"
    if diar_enabled:
        step = 2
        if skip_existing and _outputs_ready([diar_path], validate_json=validate_json):
            status(f"[{step}/{stages_count}] diarize: skip (exists)", quiet=quiet)
        else:
            status(f"[{step}/{stages_count}] diarize: start", quiet=quiet)
            diarize_args = [
                "--input",
                str(target_audio),
                "--outdir",
                str(io.output_dir),
                "--model",
                str(diar_conf.get("model", "pyannote/speaker-diarization")),
            ]
            if quiet:
                diarize_args.append("--quiet")
            if verbose:
                diarize_args.append("--verbose")
            run_module(
                "podcast_reels_forge.scripts.diarize",
                diarize_args,
                quiet=quiet,
                verbose=verbose,
            )
            status("[diarize] done", quiet=quiet)

        try:
            next(stage_iter)
        except StopIteration:
            pass

    # 3) Analyze (all models)
    prompts_conf = conf.get("prompts", {})
    p_conf = conf.get("processing", {})
    clips_conf = p_conf.get("clips", {})
    analyze_step = 2 if not diar_enabled else 3
    url = str(a_conf.get("url", "http://127.0.0.1:11434/api/generate")).strip()
    env: dict[str, str] = {}
    ollama_proc: subprocess.Popen | None = None
    local_ollama = parse_local_ollama_host_port(str(url))
    if local_ollama:
        env[ENV_MANAGED_BY_PIPELINE] = "1"
        host, port = local_ollama
        ollama_proc = ollama_start(host=host, port=port)

    try:
        for i, model in enumerate(models, 1):
            model_a_conf = _merge_ollama_conf(a_conf, model)
            model_dir = io.output_dir / _model_folder_name(model)
            model_dir.mkdir(parents=True, exist_ok=True)
            moments_path = model_dir / "moments.json"
            reels_md_path = model_dir / "reels.md"
            step = analyze_step + (i - 1)

            if skip_existing and _outputs_ready(
                [moments_path, reels_md_path],
                validate_json=validate_json,
            ):
                status(
                    f"[{step}/{stages_count}] analyze ({_model_folder_name(model)}): skip (exists)",
                    quiet=quiet,
                )
            else:
                status(
                    f"[{step}/{stages_count}] analyze ({_model_folder_name(model)}): start (may be slow)",
                    quiet=quiet,
                )

                analyze_args = [
                    "--transcript",
                    str(transcript_path),
                    "--outdir",
                    str(model_dir),
                    "--provider",
                    "ollama",
                    "--model",
                    str(model),
                    "--temperature",
                    str(model_a_conf.get("temperature", 0.3)),
                    "--reels",
                    str(p_conf.get("reels_count", 4)),
                    "--stories-count",
                    str(clips_conf.get("stories", {}).get("count", 0)),
                    "--reels-count",
                    str(clips_conf.get("reels", {}).get("count", 0)),
                    "--long-reels-count",
                    str(clips_conf.get("long_reels", {}).get("count", 0)),
                    "--highlights-moments",
                    str(clips_conf.get("highlights", {}).get("moments_count", 0)),
                    "--reel-min",
                    str(p_conf.get("reel_min_duration", 30)),
                    "--reel-max",
                    str(p_conf.get("reel_max_duration", 60)),
                    "--chunk-seconds",
                    str(model_a_conf.get("chunk_seconds", 600)),
                    "--max_chars_chunk",
                    str(model_a_conf.get("max_chars_chunk", 12000)),
                    "--timeout",
                    str(model_a_conf.get("timeout", 900)),
                    "--prompt-lang",
                    str(prompts_conf.get("language", "auto")),
                    "--prompt-variant",
                    _prompt_variant_for_model(prompts_conf, model),
                    "--url",
                    str(url),
                ]

                # Optional Ollama watchdog / fallback controls
                wd = model_a_conf.get("watchdog", {})
                if isinstance(wd, dict):
                    if "enabled" in wd:
                        if bool(wd.get("enabled")):
                            analyze_args.append("--ollama-watchdog")
                        else:
                            analyze_args.append("--no-ollama-watchdog")
                    if wd.get("first_token_timeout") is not None:
                        analyze_args += [
                            "--ollama-first-token-timeout",
                            str(wd.get("first_token_timeout")),
                        ]
                    if wd.get("stall_timeout") is not None:
                        analyze_args += [
                            "--ollama-stall-timeout",
                            str(wd.get("stall_timeout")),
                        ]
                    if wd.get("log_interval") is not None:
                        analyze_args += [
                            "--ollama-log-interval",
                            str(wd.get("log_interval")),
                        ]
                    if wd.get("max_retries") is not None:
                        analyze_args += [
                            "--ollama-max-retries",
                            str(wd.get("max_retries")),
                        ]

                fb = model_a_conf.get("fallback_models", [])
                if isinstance(fb, list):
                    for m in fb:
                        ms = str(m).strip()
                        if ms:
                            analyze_args += ["--ollama-fallback-model", ms]

                if diar_enabled and diar_path.exists():
                    analyze_args += ["--diarization", str(diar_path)]
                if quiet:
                    analyze_args.append("--quiet")
                if verbose:
                    analyze_args.append("--verbose")

                if autotune:
                    # Autotune for slow local models: fewer calls + smaller prompts.
                    try:
                        reels = int(p_conf.get("reels_count", 4))
                    except (TypeError, ValueError):
                        reels = 4
                    try:
                        timeout_s = int(model_a_conf.get("timeout", 900))
                    except (TypeError, ValueError):
                        timeout_s = 900
                    try:
                        chunk_s = int(model_a_conf.get("chunk_seconds", 600))
                    except (TypeError, ValueError):
                        chunk_s = 600
                    try:
                        max_chars = int(model_a_conf.get("max_chars_chunk", 12000))
                    except (TypeError, ValueError):
                        max_chars = 12000

                    chunk_s = max(chunk_s, 1800)
                    max_chars = min(max_chars, 6000)
                    reels = min(reels, 2)
                    timeout_s = min(timeout_s, 300)

                    _set_cli_arg(analyze_args, "--chunk-seconds", str(chunk_s))
                    _set_cli_arg(analyze_args, "--max_chars_chunk", str(max_chars))
                    _set_cli_arg(analyze_args, "--reels", str(reels))
                    _set_cli_arg(analyze_args, "--timeout", str(timeout_s))

                try:
                    run_module(
                        "podcast_reels_forge.scripts.analyze",
                        analyze_args,
                        quiet=quiet,
                        verbose=verbose,
                        env=env or None,
                    )
                    status(f"[analyze] done ({_model_folder_name(model)})", quiet=quiet)
                    _ensure_placeholder_analyze_outputs(moments_path, reels_md_path)
                except SystemExit as exc:
                    log.error(
                        "Analyze failed; continuing. model=%s code=%s",
                        model,
                        exc,
                    )
                    _ensure_placeholder_analyze_outputs(moments_path, reels_md_path)
                except Exception as exc:
                    log.exception(
                        "Analyze raised exception; continuing. model=%s error=%s",
                        model,
                        exc,
                    )
                    _ensure_placeholder_analyze_outputs(moments_path, reels_md_path)
            try:
                next(stage_iter)
            except StopIteration:
                pass
    finally:
        if ollama_proc:
            ollama_stop(ollama_proc)

    try:
        next(stage_iter)
    except StopIteration:
        pass

    # 4) Cut + exports (per model)
    v_conf = conf.get("video", {})
    exports_conf = conf.get("exports", {})

    padding = int(p_conf.get("reel_padding", 5))
    cut_start_step = analyze_step + len(models)
    for i, model in enumerate(models, 1):
        model_dir = io.output_dir / _model_folder_name(model)
        moments_path = model_dir / "moments.json"
        reels_dir = model_dir / "reels"
        existing_reels = list(reels_dir.glob("reel_*.mp4")) if reels_dir.exists() else []
        step = cut_start_step + (i - 1)

        moments_data = _read_json_if_valid(moments_path)
        if moments_data is None:
            status(
                f"[{step}/{stages_count}] cut ({_model_folder_name(model)}): skip (no moments)",
                quiet=quiet,
            )
            try:
                next(stage_iter)
            except StopIteration:
                pass
            continue

        skip_cut = skip_existing and bool(existing_reels)
        if skip_cut:
            status(
                f"[{step}/{stages_count}] cut ({_model_folder_name(model)}): skip (exists)",
                quiet=quiet,
            )
        else:
            status(
                f"[{step}/{stages_count}] cut ({_model_folder_name(model)}): start",
                quiet=quiet,
            )
            video_args = [
                "--input",
                str(video_path),
                "--moments",
                str(moments_path),
                "--outdir",
                str(model_dir),
                "--threads",
                str(v_conf.get("threads", 4)),
                "--v-bitrate",
                str(v_conf.get("video_bitrate", "5M")),
                "--a-bitrate",
                str(v_conf.get("audio_bitrate", "192k")),
                "--preset",
                str(v_conf.get("preset", "fast")),
                "--padding",
                str(padding),
            ]
            if "use_nvenc" in v_conf and not bool(v_conf.get("use_nvenc")):
                video_args.append("--no-nvenc")
            if v_conf.get("vertical_crop", True):
                video_args.append("--vertical")
            if exports_conf.get("webm", False):
                video_args.append("--export-webm")
            if exports_conf.get("gif", False):
                video_args.append("--export-gif")
            if exports_conf.get("audio_only", False):
                video_args.append("--export-audio")
            if quiet:
                video_args.append("--quiet")
            if verbose:
                video_args.append("--verbose")

            run_module(
                "podcast_reels_forge.scripts.video_processor",
                video_args,
                quiet=quiet,
                verbose=verbose,
            )
            status(f"[cut] done ({_model_folder_name(model)})", quiet=quiet)

        try:
            next(stage_iter)
        except StopIteration:
            pass

    status("[forge] done", quiet=quiet)
