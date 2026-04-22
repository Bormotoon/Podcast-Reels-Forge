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
    get_ollama_models,
    ollama_start,
    ollama_stop,
    parse_local_ollama_host_port,
    pull_ollama_model,
)
from podcast_reels_forge.config import (
    normalize_model_folder_name,
    resolve_ollama_role_mapping,
)
from podcast_reels_forge.stages.analyze_stage import run_staged_analysis
from podcast_reels_forge.stages.transcribe_stage import (
    TranscribeConfig,
    transcribe_file,
)
from podcast_reels_forge.utils.burned_subtitles import (
    subtitle_settings_from_conf,
    sync_reel_burned_subtitles,
)
from podcast_reels_forge.utils.reel_markdown import sync_reel_markdowns

log = logging.getLogger("Forge")


def _model_folder_name(model: str) -> str:
    """Backward-compatible wrapper around the shared folder-name helper."""

    return normalize_model_folder_name(model)


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


def _file_has_content(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _ensure_mp3_companion(video_path: Path) -> Path:
    """Create a same-stem MP3 companion from video if it does not exist yet."""
    mp3_path = video_path.with_suffix(".mp3")
    if _file_has_content(mp3_path):
        return mp3_path

    log.info(
        "Creating MP3 companion for %s -> %s",
        video_path.name,
        mp3_path.name,
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "320k",
        str(mp3_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise SystemExit("ffmpeg not found; required to create MP3 companions") from exc
    if res.returncode != 0:
        stderr = (res.stderr or "").strip()
        stdout = (res.stdout or "").strip()
        detail = stderr or stdout or "unknown ffmpeg error"
        raise SystemExit(
            f"Failed to create MP3 companion for {video_path.name}: {detail[-500:]}",
        )
    return mp3_path


def find_input_queue(input_dir: Path) -> list[dict[str, Any]]:
    """RU: Находит видео и гарантирует MP3-спутник для каждого ролика.

    EN: Find video files and ensure each one has a same-stem MP3 companion.
    """
    video_exts = {".mp4", ".mkv", ".mov", ".avi"}

    if not input_dir.exists():
        return []

    stems: dict[str, Path] = {}

    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in video_exts:
            continue
        stem = p.stem
        curr_v = stems.get(stem)
        # Pick newest if multiple video formats for the same stem.
        if curr_v is None or p.stat().st_mtime > curr_v.stat().st_mtime:
            stems[stem] = p

    queue = []
    for stem in sorted(stems.keys()):
        video_path = stems[stem]
        mp3_path = _ensure_mp3_companion(video_path)
        queue.append({
            "stem": stem,
            "video": video_path,
            "audio": mp3_path,
        })
    return queue


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
    input_dir_path = Path(str(paths.get("input_dir", "input")))
    base_output_dir = Path(str(paths.get("output_dir", "output")))

    cache_conf = conf.get("cache", {}) if isinstance(conf, dict) else {}
    validate_json = bool(cache_conf.get("validate_json", True))
    if "enabled" in cache_conf:
        skip_existing = bool(cache_conf.get("enabled", True)) and skip_existing

    queue = find_input_queue(input_dir_path)
    if not queue:
        status(f"RU: В папке {input_dir_path} не найдено видео-файлов.", quiet=quiet)
        status(f"EN: No video files found in {input_dir_path}", quiet=quiet)
        return

    diar_conf = conf.get("diarization", {})
    diar_enabled = bool(diar_conf.get("enabled", False))
    subtitle_settings = subtitle_settings_from_conf(conf, repo_dir=repo_dir)
    a_conf = conf.get("ollama", {})
    prompts_conf = conf.get("prompts", {})
    p_conf = conf.get("processing", {})
    v_conf = conf.get("video", {})
    exports_conf = conf.get("exports", {})

    roles = resolve_ollama_role_mapping(conf)
    final_model_folder = _model_folder_name(roles.judge)

    stages_per_file = 1 + (1 if diar_enabled else 0) + 2
    total_stages = len(queue) * stages_per_file
    stage_iter = iter(
        tqdm(
            range(total_stages),
            total=total_stages,
            disable=(not progress) or quiet,
            desc="Podcast Reels Forge",
        ),
    )

    for item in queue:
        video_path = item["video"]
        target_audio = item["audio"]
        stem = item["stem"]

        io = PipelineIO(
            input_dir=input_dir_path,
            output_dir=base_output_dir / stem,
        )
        os.makedirs(io.output_dir, exist_ok=True)
        analysis_model_folder = io.output_dir / final_model_folder

        status(f"\n[forge] processing: {stem}", quiet=quiet)
        status(f"[forge] video: {video_path.name}", quiet=quiet)
        status(f"[forge] transcribe input: {target_audio.name}", quiet=quiet)

        # 1) Transcribe
        t_conf = conf.get("transcription", {})

        transcript_path = io.output_dir / target_audio.with_suffix(".json").name
        transcript_srt_path = io.output_dir / target_audio.with_suffix(".srt").name
        legacy_audio_json = io.output_dir / "audio.json"
        if not transcript_path.exists() and legacy_audio_json.exists():
            transcript_path = legacy_audio_json
            transcript_srt_path = legacy_audio_json.with_suffix(".srt")

        if skip_existing and _outputs_ready(
            [transcript_path, transcript_srt_path],
            validate_json=validate_json,
        ):
            status("[transcribe] skip (exists)", quiet=quiet)
        else:
            status("[transcribe] start", quiet=quiet)
            device_raw = str(t_conf.get("device", "cuda")).strip().lower()
            if device_raw == "auto":
                device_raw = "cuda" if _has_cuda() else "cpu"
            if autotune and device_raw == "auto":
                device_raw = "cuda" if _has_cuda() else "cpu"

            compute_type_raw = t_conf.get("compute_type")
            compute_type = None
            if isinstance(compute_type_raw, str) and compute_type_raw.strip():
                compute_type = compute_type_raw.strip()
                if autotune and compute_type.lower() == "auto":
                    compute_type = None

            transcribe_config = TranscribeConfig(
                input_path=target_audio,
                outdir=io.output_dir,
                model_name=str(t_conf.get("model", "large-v3")),
                device=device_raw,
                language=str(t_conf.get("language", "ru")),
                beam_size=int(t_conf.get("beam_size", 5)),
                compute_type=compute_type,
                quiet=quiet,
                verbose=verbose,
            )
            transcript_path = transcribe_file(transcribe_config)
            status("[transcribe] done", quiet=quiet)

        try:
            next(stage_iter)
        except StopIteration:
            pass

        # 2) Optional diarization
        diar_path = io.output_dir / "diarization.json"
        if diar_enabled:
            if skip_existing and _outputs_ready([diar_path], validate_json=validate_json):
                status("[diarize] skip (exists)", quiet=quiet)
            else:
                status("[diarize] start", quiet=quiet)
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

        # 3) Analyze (staged, single final output folder)
        url = str(a_conf.get("url", "http://127.0.0.1:11434/api/generate")).strip()
        ollama_proc: subprocess.Popen | None = None
        local_ollama = parse_local_ollama_host_port(str(url))
        if local_ollama:
            host, port = local_ollama
            ollama_proc = ollama_start(host=host, port=port)

        # Proactively check/pull models if using Ollama
        if "ollama" in str(url).lower():
            try:
                available = get_ollama_models(url)
                for model in roles.unique_models():
                    if model not in available and f"{model}:latest" not in available:
                        status(f"[ollama] model '{model}' not found, pulling...", quiet=quiet)
                        pull_ollama_model(url, model)
            except Exception as e:
                log.warning("Could not verify/pull Ollama models: %s", e)

        try:
            analysis_model_folder.mkdir(parents=True, exist_ok=True)
            moments_path = analysis_model_folder / "moments.json"
            reels_md_path = analysis_model_folder / "reels.md"

            if skip_existing and _outputs_ready(
                [moments_path, reels_md_path],
                validate_json=validate_json,
            ):
                status(
                    f"[analyze] skip ({final_model_folder})",
                    quiet=quiet,
                )
            else:
                status(
                    f"[analyze] start ({final_model_folder})",
                    quiet=quiet,
                )
                try:
                    final_moments = run_staged_analysis(
                        transcript_path=transcript_path,
                        outdir=analysis_model_folder,
                        provider_name="ollama",
                        url=url,
                        api_key=None,
                        roles=roles,
                        ollama_conf=a_conf,
                        prompts_conf=prompts_conf,
                        processing_conf=p_conf,
                        diarization_path=diar_path if diar_enabled and diar_path.exists() else None,
                        quiet=quiet,
                        verbose=verbose,
                        progress=progress,
                    )
                    status(
                        f"[analyze] done ({final_model_folder}, moments={len(final_moments)})",
                        quiet=quiet,
                    )
                except SystemExit as exc:
                    log.error(
                        "Analyze failed; continuing. folder=%s code=%s",
                        final_model_folder,
                        exc,
                    )
                    _ensure_placeholder_analyze_outputs(moments_path, reels_md_path)
                except Exception as exc:
                    log.exception(
                        "Analyze raised exception; continuing. folder=%s error=%s",
                        final_model_folder,
                        exc,
                    )
                    _ensure_placeholder_analyze_outputs(moments_path, reels_md_path)
                else:
                    _ensure_placeholder_analyze_outputs(moments_path, reels_md_path)
            try:
                next(stage_iter)
            except StopIteration:
                pass
        finally:
            if ollama_proc:
                ollama_stop(ollama_proc)

        # 4) Cut + exports (single final output)
        padding = int(p_conf.get("reel_padding", 5))
        reels_dir = analysis_model_folder / "reels"
        moments_data = _read_json_if_valid(moments_path)
        if moments_data is None:
            status(
                f"[cut] skip ({final_model_folder}): no moments",
                quiet=quiet,
            )
            try:
                next(stage_iter)
            except StopIteration:
                pass
            continue

        existing_reels = list(reels_dir.glob("reel_*.mp4")) if reels_dir.exists() else []
        skip_cut = skip_existing and bool(existing_reels)
        if skip_cut:
            status(
                f"[cut] skip ({final_model_folder}): exists",
                quiet=quiet,
            )
        else:
            status(
                f"[cut] start ({final_model_folder})",
                quiet=quiet,
            )
            video_args = [
                "--input",
                str(video_path),
                "--moments",
                str(moments_path),
                "--outdir",
                str(analysis_model_folder),
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
            if v_conf.get("smart_crop_face", True):
                video_args.append("--smart-crop-face")
                video_args += ["--face-samples", str(v_conf.get("face_samples", 7))]
                video_args += ["--face-min-size", str(v_conf.get("face_min_size", 60))]

            q_conf = conf.get("processing", {}).get("quality_filters", {})
            if "min_score" in q_conf:
                video_args += ["--filter-min-score", str(q_conf["min_score"])]
            if "min_duration" in q_conf:
                video_args += ["--filter-min-duration", str(q_conf["min_duration"])]
            if "max_duration" in q_conf:
                video_args += ["--filter-max-duration", str(q_conf["max_duration"])]
            if "face_min_ratio" in q_conf:
                video_args += ["--filter-face-ratio", str(q_conf["face_min_ratio"])]

            if exports_conf.get("webm", False):
                video_args.append("--export-webm")
            if exports_conf.get("gif", False):
                video_args.append("--export-gif")
            if exports_conf.get("audio_only", False):
                video_args.append("--export-audio")
            if subtitle_settings.enabled:
                video_args.append("--burn-subtitles")
                video_args += ["--transcript-json", str(transcript_path)]
                video_args += ["--subtitle-font", str(subtitle_settings.font_path)]
                video_args += ["--subtitle-css", str(subtitle_settings.css_path)]
                if not subtitle_settings.wrap_words:
                    video_args.append("--no-subtitle-wrap-words")
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
            status(f"[cut] done ({final_model_folder})", quiet=quiet)

        if isinstance(moments_data, list):
            try:
                sync_reel_markdowns(
                    [m for m in moments_data if isinstance(m, dict)],
                    reels_dir,
                )
            except OSError as exc:
                log.warning(
                    "Failed to write reel markdowns for %s: %s",
                    final_model_folder,
                    exc,
                )
            if subtitle_settings.enabled and skip_cut:
                sync_reel_burned_subtitles(
                    [m for m in moments_data if isinstance(m, dict)],
                    reels_dir,
                    transcript_json_path=transcript_path,
                    padding=padding,
                    settings=subtitle_settings,
                    verbose=verbose and not quiet,
                )

        try:
            next(stage_iter)
        except StopIteration:
            pass

    status("[forge] done", quiet=quiet)
