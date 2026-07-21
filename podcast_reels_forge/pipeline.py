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

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Collection, Sequence
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable: object, **_: object) -> object:
        return iterable

from podcast_reels_forge.utils.llama_cpp_service import (
    is_tcp_open,
    llama_cpp_start,
    llama_cpp_stop,
    parse_local_llama_cpp_host_port,
    wait_for_server_ready,
)
from podcast_reels_forge.config import (
    normalize_model_folder_name,
    resolve_llama_cpp_role_mapping,
)
from podcast_reels_forge.stages.analyze_stage import run_staged_analysis
from podcast_reels_forge.stages.article_stage import run_article
from podcast_reels_forge.stages.proofread_stage import run_proofread
from podcast_reels_forge.stages.transcribe_stage import (
    TranscribeConfig,
    transcribe_file,
)
from podcast_reels_forge.utils.burned_subtitles import (
    subtitle_settings_from_conf,
    sync_reel_burned_subtitles,
)
from podcast_reels_forge.utils.ffmpeg import ffmpeg_bin
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


#: Listening copy. 320k keeps the MP3 transparent for a human ear.
MP3_BITRATE = "320k"
#: Working copy for the models. Both faster-whisper and pyannote decode to
#: 16 kHz mono internally, so this is exactly what they consume — a 48 kHz
#: stereo file would only make them do the downmix themselves. Storing it as
#: PCM also sidesteps a real pyannote failure: cropping an MP3 yields a chunk a
#: few samples short of what it asked for, and the pipeline raises.
WAV_SAMPLE_RATE = 16000


def _ensure_audio_companions(video_path: Path) -> tuple[Path, Path]:
    """RU: Готовит рядом с видео MP3 и WAV — одним проходом ffmpeg.

    EN: Produce the MP3 and WAV companions next to a video in one ffmpeg pass.

    Both are encoded from the video's own audio stream, decoded once. Deriving
    the WAV from the MP3 instead would bake the lossy artefacts into what the
    models hear.
    """
    mp3_path = video_path.with_suffix(".mp3")
    wav_path = video_path.with_suffix(".wav")

    need_mp3 = not _file_has_content(mp3_path)
    need_wav = not _file_has_content(wav_path)
    if not need_mp3 and not need_wav:
        return mp3_path, wav_path

    wanted = ", ".join(
        name for name, needed in (("MP3", need_mp3), ("WAV", need_wav)) if needed
    )
    log.info("Creating %s companion(s) for %s", wanted, video_path.name)

    cmd = [ffmpeg_bin(), "-y", "-i", str(video_path)]
    if need_mp3:
        cmd += ["-vn", "-c:a", "libmp3lame", "-b:a", MP3_BITRATE, str(mp3_path)]
    if need_wav:
        cmd += [
            "-vn",
            "-ac", "1",
            "-ar", str(WAV_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(wav_path),
        ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise SystemExit("ffmpeg not found; required to create audio companions") from exc
    if res.returncode != 0:
        stderr = (res.stderr or "").strip()
        stdout = (res.stdout or "").strip()
        detail = stderr or stdout or "unknown ffmpeg error"
        raise SystemExit(
            f"Failed to create audio companions for {video_path.name}: {detail[-500:]}",
        )
    return mp3_path, wav_path


def _ensure_mp3_companion(video_path: Path) -> Path:
    """Backward-compatible wrapper: the MP3 half of the companion pair."""

    return _ensure_audio_companions(video_path)[0]


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
        mp3_path, wav_path = _ensure_audio_companions(video_path)
        queue.append({
            "stem": stem,
            "video": video_path,
            # MP3 is the listening copy; the models get the PCM one.
            "audio": mp3_path,
            "wav": wav_path,
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


def _kill_llama_server(port: int) -> None:
    """Kill any llama-server process listening on *port* and wait for VRAM to free.

    Safe to call even if no server is running. Used to reclaim GPU VRAM before
    stages that need it (Whisper transcription, NVENC encoding). systemd services
    with Restart=always will restart the server automatically afterwards.
    """
    try:
        result = subprocess.run(
            ["pkill", "-f", f"llama-server.*--port.*{port}"],
            capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            log.info("Killed llama-server on port %d to free VRAM", port)
            time.sleep(3)
    except Exception as exc:
        log.debug("_kill_llama_server: %s", exc)


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _cpu_count() -> int:
    detected = os.cpu_count() or 4
    return max(1, int(detected))


def _gpu_vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return float(props.total_memory) / (1024.0**3)
    except Exception:
        return None


def _autotune_llama_cpp_conf(base_conf: dict[str, Any]) -> dict[str, Any]:
    tuned = dict(base_conf)
    service = tuned.get("service")
    service_dict: dict[str, Any] = dict(service) if isinstance(service, dict) else {}
    cpu = _cpu_count()
    vram = _gpu_vram_gb()

    service_dict["threads"] = int(service_dict.get("threads", max(6, min(16, cpu - 2))))
    if "ctx_size" not in service_dict:
        service_dict["ctx_size"] = 8192
    if "n_gpu_layers" not in service_dict:
        service_dict["n_gpu_layers"] = 99
    if "batch_size" not in service_dict:
        service_dict["batch_size"] = 1536 if (vram is not None and vram >= 14.0) else 1024
    if "ubatch_size" not in service_dict:
        service_dict["ubatch_size"] = 768 if (vram is not None and vram >= 14.0) else 512
    if "parallel" not in service_dict:
        service_dict["parallel"] = 2 if (vram is not None and vram >= 14.0) else 1

    tuned["service"] = service_dict

    scout_parallelism = int(tuned.get("scout_parallelism", 1))
    if scout_parallelism <= 0:
        scout_parallelism = 1
    if "scout_parallelism" not in tuned:
        tuned["scout_parallelism"] = min(int(service_dict.get("parallel", 1)), 3)
    return tuned


def _autotune_video_threads(v_conf: dict[str, Any]) -> int:
    cpu = _cpu_count()
    nvenc_enabled = bool(v_conf.get("use_nvenc", True))
    if nvenc_enabled:
        return max(1, min(3, cpu // 4 if cpu >= 4 else 1))
    return max(1, min(8, cpu // 2))


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


def _merge_llama_cpp_conf(base: dict[str, Any], model: str) -> dict[str, Any]:
    """Merge base llama.cpp config with per-model overrides.

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


#: Stages in execution order, as accepted by ``--only`` / ``--skip``.
PIPELINE_STAGES: tuple[str, ...] = (
    "transcribe",
    "diarize",
    "proofread",
    "article",
    "analyze",
    "cut",
)


def resolve_stages(
    *,
    only: str | Sequence[str] | None = None,
    skip: str | Sequence[str] | None = None,
) -> set[str]:
    """RU: Превращает --only/--skip в набор стадий к запуску.

    EN: Turn ``--only`` / ``--skip`` into the set of stages to run.

    Accepts comma-separated strings or sequences. Unknown names raise rather
    than being ignored: a typo must not silently skip half the pipeline.
    """

    def parse(value: str | Sequence[str] | None) -> list[str]:
        if value is None:
            return []
        items = value.split(",") if isinstance(value, str) else list(value)
        return [str(item).strip().lower() for item in items if str(item).strip()]

    only_names = parse(only)
    skip_names = parse(skip)

    unknown = sorted(set(only_names + skip_names) - set(PIPELINE_STAGES))
    if unknown:
        raise SystemExit(
            f"Unknown pipeline stage(s): {', '.join(unknown)}. "
            f"Available: {', '.join(PIPELINE_STAGES)}",
        )

    selected = set(only_names) if only_names else set(PIPELINE_STAGES)
    selected -= set(skip_names)
    if not selected:
        raise SystemExit("No pipeline stages left to run after --only/--skip")
    return selected


def run_pipeline(
    *,
    conf: dict[str, Any],
    repo_dir: Path,
    quiet: bool,
    verbose: bool,
    skip_existing: bool = True,
    autotune: bool = False,
    progress: bool = True,
    stages: Collection[str] | None = None,
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
    proofread_conf = conf.get("proofread", {})
    if not isinstance(proofread_conf, dict):
        proofread_conf = {}
    proofread_enabled = bool(proofread_conf.get("enabled", False))
    article_conf = conf.get("article", {})
    if not isinstance(article_conf, dict):
        article_conf = {}
    article_enabled = bool(article_conf.get("enabled", False))

    # A stage runs when the config enables it AND the caller selected it.
    active = set(stages) if stages is not None else set(PIPELINE_STAGES)
    diar_enabled = diar_enabled and "diarize" in active
    proofread_enabled = proofread_enabled and "proofread" in active
    article_enabled = article_enabled and "article" in active
    transcribe_enabled = "transcribe" in active
    analyze_enabled = "analyze" in active
    cut_enabled = "cut" in active
    if not quiet and stages is not None:
        status(
            "[forge] stages: " + ", ".join(s for s in PIPELINE_STAGES if s in active),
            quiet=quiet,
        )
    subtitle_settings = subtitle_settings_from_conf(conf, repo_dir=repo_dir)
    a_conf = conf.get("llama_cpp", {})
    prompts_conf = conf.get("prompts", {})
    p_conf = conf.get("processing", {})
    v_conf = conf.get("video", {})
    exports_conf = conf.get("exports", {})

    if not isinstance(a_conf, dict):
        a_conf = {}
    if not isinstance(v_conf, dict):
        v_conf = {}

    if autotune:
        a_conf = _autotune_llama_cpp_conf(a_conf)
        tuned_threads = _autotune_video_threads(v_conf)
        v_conf = {**v_conf, "threads": tuned_threads}
        if not quiet:
            service = a_conf.get("service", {}) if isinstance(a_conf, dict) else {}
            status(
                "[autotune] "
                f"llama threads={service.get('threads')} parallel={service.get('parallel')} "
                f"ctx={service.get('ctx_size')} video_jobs={v_conf.get('threads')}",
                quiet=quiet,
            )

    roles = resolve_llama_cpp_role_mapping(conf)
    final_model_folder = _model_folder_name(roles.judge_metadata)

    # Resolve local llama host/port once so all stages can reference it.
    _llama_url = str(a_conf.get("url", "")).strip()
    local_llama_global = parse_local_llama_cpp_host_port(_llama_url) if _llama_url else None

    stages_per_file = (
        (1 if transcribe_enabled else 0)
        + (1 if diar_enabled else 0)
        + (1 if proofread_enabled else 0)
        + (1 if article_enabled else 0)
        + (1 if analyze_enabled else 0)
        + (1 if cut_enabled else 0)
    )
    total_stages = len(queue) * stages_per_file
    # A plain progress object, advanced explicitly. Driving a tqdm iterator
    # with next() renders one step behind (tqdm counts an iteration when the
    # following next() arrives), so the bar sat at N-1 and finished at 3/4.
    stage_bar = tqdm(
        total=total_stages,
        disable=(not progress) or quiet,
        desc="Podcast Reels Forge",
    )

    for item in queue:
        video_path = item["video"]
        target_audio = item["audio"]
        # RU: Моделям отдаём PCM: у Whisper нет артефактов mp3, а pyannote на
        #     mp3 просто падает (обрезка даёт на несколько сэмплов меньше).
        # EN: The models get the PCM copy: Whisper avoids the mp3 artefacts and
        #     pyannote outright fails on mp3 (a crop comes back a few samples
        #     short of what it asked for).
        model_audio = item.get("wav") or target_audio
        if not _file_has_content(model_audio):
            model_audio = target_audio
        stem = item["stem"]

        io = PipelineIO(
            input_dir=input_dir_path,
            output_dir=base_output_dir / stem,
        )
        os.makedirs(io.output_dir, exist_ok=True)
        analysis_model_folder = io.output_dir / final_model_folder

        status(f"\n[forge] processing: {stem}", quiet=quiet)
        status(f"[forge] video: {video_path.name}", quiet=quiet)
        status(f"[forge] transcribe input: {model_audio.name}", quiet=quiet)

        # 1) Transcribe
        t_conf = conf.get("transcription", {})

        transcript_path = io.output_dir / target_audio.with_suffix(".json").name
        transcript_srt_path = io.output_dir / target_audio.with_suffix(".srt").name
        legacy_audio_json = io.output_dir / "audio.json"
        if not transcript_path.exists() and legacy_audio_json.exists():
            transcript_path = legacy_audio_json
            transcript_srt_path = legacy_audio_json.with_suffix(".srt")

        _transcribe_needed = transcribe_enabled and not (skip_existing and _outputs_ready(
            [transcript_path, transcript_srt_path],
            validate_json=validate_json,
        ))
        if _transcribe_needed and local_llama_global:
            # Whisper large-v3 needs ~10GB VRAM. Kill any llama-server that
            # holds the GPU before we start transcription.
            _kill_llama_server(local_llama_global[1])

        if not transcribe_enabled:
            status("[transcribe] skip (not selected)", quiet=quiet)
        elif not _transcribe_needed:
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
                input_path=model_audio,
                outdir=io.output_dir,
                model_name=str(t_conf.get("model", "large-v3")),
                device=device_raw,
                language=str(t_conf.get("language", "ru")),
                beam_size=int(t_conf.get("beam_size", 5)),
                compute_type=compute_type,
                best_of=int(t_conf.get("best_of", 1)),
                patience=float(t_conf.get("patience", 1.0)),
                batch_size=int(t_conf.get("batch_size", 16)),
                repetition_penalty=float(t_conf.get("repetition_penalty", 1.1)),
                no_repeat_ngram_size=int(t_conf.get("no_repeat_ngram_size", 3)),
                condition_on_previous_text=bool(
                    t_conf.get("condition_on_previous_text", False)
                ),
                mode=str(t_conf.get("mode", "fast")),
                initial_prompt=t_conf.get("initial_prompt") or None,
                quality_beam_size=int(t_conf.get("quality_beam_size", 10)),
                quiet=quiet,
                verbose=verbose,
            )
            transcript_path = transcribe_file(transcribe_config)
            status("[transcribe] done", quiet=quiet)

        if transcribe_enabled:
            stage_bar.update(1)

        # 2) Optional diarization
        diar_path = io.output_dir / "diarization.json"
        if diar_enabled:
            if skip_existing and _outputs_ready([diar_path], validate_json=validate_json):
                status("[diarize] skip (exists)", quiet=quiet)
            else:
                status("[diarize] start", quiet=quiet)
                diarize_args = [
                    "--input",
                    str(model_audio),
                    "--outdir",
                    str(io.output_dir),
                    "--model",
                    str(diar_conf.get("model", "pyannote/speaker-diarization")),
                ]
                num_speakers = diar_conf.get("num_speakers")
                if num_speakers:
                    diarize_args += ["--num-speakers", str(int(num_speakers))]
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

            stage_bar.update(1)

        # 3) Proofread + Analyze (share one llama-server session; proofread runs
        #    after diarization so pyannote gets the GPU before llama loads).
        url = str(a_conf.get("url", "http://127.0.0.1:8080/v1/chat/completions")).strip()
        llama_cpp_proc: subprocess.Popen | None = None
        local_llama = parse_local_llama_cpp_host_port(str(url))
        service_conf = a_conf.get("service", {}) if isinstance(a_conf, dict) else {}
        if local_llama and bool(service_conf.get("auto_start", True)):
            host, port = local_llama
            llama_cpp_proc = llama_cpp_start(
                host=host,
                port=port,
                service_conf=service_conf if isinstance(service_conf, dict) else {},
            )

        # Wait for the server to finish loading the model before sending requests.
        # This handles both our own auto-started instance and an external server
        # that is still loading (returns 503 "Loading model" until ready).
        if local_llama:
            host, port = local_llama
            startup_timeout = int((service_conf or {}).get("startup_timeout", 120))
            wait_for_server_ready(host, port, timeout_s=max(startup_timeout, 300))

        try:
            # 3a) Proofread transcript: LLM fixes spelling/punctuation, writes
            #     <stem>.proofread.json; downstream stages use the corrected file.
            if proofread_enabled:
                proofread_path = transcript_path.with_name(
                    transcript_path.stem + ".proofread.json",
                )
                proofread_srt_path = proofread_path.with_suffix(".srt")
                if skip_existing and _outputs_ready(
                    [proofread_path, proofread_srt_path],
                    validate_json=validate_json,
                ):
                    status("[proofread] skip (exists)", quiet=quiet)
                    transcript_path = proofread_path
                else:
                    status(f"[proofread] start ({roles.proofread})", quiet=quiet)
                    try:
                        asyncio.run(run_proofread(
                            transcript_path=transcript_path,
                            output_path=proofread_path,
                            url=url,
                            model=roles.proofread,
                            proofread_conf=proofread_conf,
                            prompts_conf=prompts_conf,
                            quiet=quiet,
                            verbose=verbose,
                        ))
                        transcript_path = proofread_path
                        status("[proofread] done", quiet=quiet)
                    except Exception as exc:
                        log.exception(
                            "Proofread failed; continuing with the raw transcript: %s",
                            exc,
                        )
                stage_bar.update(1)

            # RU: Вычитанный транскрипт мог быть сделан прошлым запуском. Даже
            #     если стадия сейчас не запускалась (--only analyze), дальше
            #     должен идти исправленный текст, а не сырой.
            # EN: The proofread transcript may come from an earlier run. Even
            #     when the stage did not execute this time (--only analyze), the
            #     corrected text is what the rest of the pipeline must use.
            _existing_proofread = transcript_path.with_name(
                transcript_path.stem + ".proofread.json",
            )
            if _existing_proofread.exists() and _outputs_ready(
                [_existing_proofread], validate_json=validate_json,
            ):
                transcript_path = _existing_proofread

            # 3b) Article: retell the (proofread) transcript as readable prose
            #     with meaning-based sections. Read-only for the transcript.
            if article_enabled:
                article_stem = transcript_path.stem.replace(".proofread", "")
                article_md_path = transcript_path.with_name(article_stem + ".article.md")
                article_json_path = article_md_path.with_suffix(".json")
                if skip_existing and _outputs_ready(
                    [article_md_path, article_json_path],
                    validate_json=validate_json,
                ):
                    status("[article] skip (exists)", quiet=quiet)
                else:
                    status(f"[article] start ({roles.article})", quiet=quiet)
                    try:
                        asyncio.run(run_article(
                            transcript_path=transcript_path,
                            output_path=article_md_path,
                            url=url,
                            model=roles.article,
                            article_conf=article_conf,
                            prompts_conf=prompts_conf,
                            diarization_path=(
                                diar_path if diar_path.exists() else None
                            ),
                            title=stem,
                            quiet=quiet,
                            verbose=verbose,
                        ))
                        status("[article] done", quiet=quiet)
                    except Exception as exc:
                        # The article is a side artefact; reels must still ship.
                        log.exception("Article stage failed; continuing: %s", exc)
                stage_bar.update(1)

            analysis_model_folder.mkdir(parents=True, exist_ok=True)
            moments_path = analysis_model_folder / "moments.json"
            reels_md_path = analysis_model_folder / "reels.md"

            if not analyze_enabled:
                status("[analyze] skip (not selected)", quiet=quiet)
            elif skip_existing and _outputs_ready(
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
                    final_moments = asyncio.run(run_staged_analysis(
                        transcript_path=transcript_path,
                        outdir=analysis_model_folder,
                        provider_name="llama_cpp",
                        url=url,
                        api_key=None,
                        roles=roles,
                        llama_cpp_conf=a_conf,
                        prompts_conf=prompts_conf,
                        processing_conf=p_conf,
                        diarization_path=diar_path if diar_enabled and diar_path.exists() else None,
                        quiet=quiet,
                        verbose=verbose,
                        progress=progress,
                    ))
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
            if analyze_enabled:
                stage_bar.update(1)
        finally:
            if llama_cpp_proc:
                llama_cpp_stop(llama_cpp_proc)
            # Free VRAM from any llama-server (ours or external) before the cut
            # stage so NVENC can use the GPU unimpeded.
            if local_llama:
                h, p = local_llama
                if is_tcp_open(h, p):
                    _kill_llama_server(p)

        # 4) Cut + exports (single final output)
        padding = int(p_conf.get("reel_padding", 5))
        reels_dir = analysis_model_folder / "reels"
        moments_data = _read_json_if_valid(moments_path)
        if moments_data is None:
            status(
                f"[cut] skip ({final_model_folder}): no moments",
                quiet=quiet,
            )
            if cut_enabled:
                stage_bar.update(1)
            continue

        import re as _re
        existing_reels = [
            p for p in (reels_dir.glob("reel_*.mp4") if reels_dir.exists() else [])
            if _re.match(r"^reel_\d+\.mp4$", p.name)
        ]
        skip_cut = (not cut_enabled) or (skip_existing and bool(existing_reels))
        if not cut_enabled:
            status("[cut] skip (not selected)", quiet=quiet)
        elif skip_cut:
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
            video_args += ["--nvenc-cq", str(v_conf.get("nvenc_cq", 21))]
            video_args += ["--nvenc-preset", str(v_conf.get("nvenc_preset", "p5"))]
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

        if cut_enabled:
            stage_bar.update(1)

    stage_bar.close()
    status("[forge] done", quiet=quiet)
