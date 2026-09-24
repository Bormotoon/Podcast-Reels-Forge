"""RU: Оркестрация пайплайна Podcast Reels Forge.

Модуль содержит логику основного пайплайна и оркестрирует стадии:
0) Опциональная загрузка с YouTube (ссылка/плейлист/канал → файл во входной папке)
1) Транскрибация (аудио/видео → JSON транскрипт)
2) Опциональная диаризация (распознавание спикеров)
3) Анализ (LLM ищет «вирусные» моменты)
4) Обработка видео (нарезка, эффекты, экспорт)

EN: Pipeline orchestration for Podcast Reels Forge.

This module contains the main pipeline logic that orchestrates all stages:
0) Optional YouTube fetch (link/playlist/channel → a file in the input folder)
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
import re
import shutil
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
from podcast_reels_forge import __version__
from podcast_reels_forge.config import (
    normalize_model_folder_name,
    resolve_llama_cpp_role_mapping,
)
from podcast_reels_forge.sources.youtube import (
    VideoFilters,
    YouTubeError,
    api_key_from_env,
    resolve_sources,
)
from podcast_reels_forge.run_report import CACHED, DONE, FAILED, SKIPPED, RunReport
from podcast_reels_forge.stages.analyze_stage import run_staged_analysis
from podcast_reels_forge.stages.article_stage import run_article
from podcast_reels_forge.stages.fetch_stage import (
    DEFAULT_FILENAME_TEMPLATE,
    FetchConfig,
    describe_videos,
    fetch_videos,
    is_download_fragment,
    locate_existing,
    resolve_download_dir,
    want_video_for_stages,
)
from podcast_reels_forge.stages.proofread_stage import run_proofread
from podcast_reels_forge.stages.transcribe_stage import (
    TranscribeConfig,
    transcribe_file,
    whisper_model_session,
)
from podcast_reels_forge.utils.burned_subtitles import (
    subtitle_settings_from_conf,
    sync_reel_burned_subtitles,
)
from podcast_reels_forge.utils.ffmpeg import ffmpeg_bin
from podcast_reels_forge.utils.fingerprint import (
    StageState,
    file_digest,
    file_identity,
    files_digest,
    fingerprint,
    subset,
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


#: Listening copy. 320k keeps the MP3 transparent for a human ear.
MP3_BITRATE = "320k"
#: Working copy for the models. Both faster-whisper and pyannote decode to
#: 16 kHz mono internally, so this is exactly what they consume — a 48 kHz
#: stereo file would only make them do the downmix themselves. Storing it as
#: PCM also sidesteps a real pyannote failure: cropping an MP3 yields a chunk a
#: few samples short of what it asked for, and the pipeline raises.
WAV_SAMPLE_RATE = 16000


def _ensure_audio_companions(source_path: Path, *, want_mp3: bool = True) -> tuple[Path, Path]:
    """RU: Готовит рядом с исходником MP3 и WAV — одним проходом ffmpeg.

    EN: Produce the MP3 and WAV companions next to a source in one ffmpeg pass.

    Both are encoded from the source's own audio stream, decoded once. Deriving
    the WAV from the MP3 instead would bake the lossy artefacts into what the
    models hear. ``want_mp3=False`` skips the listening copy: the models only
    ever read the WAV (``audio.listening_copy`` in the config).

    The source is usually a video, but an audio-only YouTube fetch lands an m4a
    here instead — ffmpeg does not care, and re-encoding that m4a to MP3 would be
    a second lossy pass, which is why the download keeps it as delivered.
    """
    mp3_path = source_path.with_suffix(".mp3")
    wav_path = source_path.with_suffix(".wav")

    need_mp3 = want_mp3 and not _file_has_content(mp3_path)
    need_wav = not _file_has_content(wav_path)
    if not need_mp3 and not need_wav:
        return mp3_path, wav_path

    wanted = ", ".join(
        name for name, needed in (("MP3", need_mp3), ("WAV", need_wav)) if needed
    )
    log.info("Creating %s companion(s) for %s", wanted, source_path.name)

    cmd = [ffmpeg_bin(), "-y", "-i", str(source_path)]
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
            f"Failed to create audio companions for {source_path.name}: {detail[-500:]}",
        )
    return mp3_path, wav_path


def _ensure_mp3_companion(video_path: Path) -> Path:
    """Backward-compatible wrapper: the MP3 half of the companion pair."""

    return _ensure_audio_companions(video_path)[0]


#: Containers treated as an episode's video source.
VIDEO_EXTS: tuple[str, ...] = (".mp4", ".mkv", ".mov", ".avi", ".webm")

#: RU: Контейнеры, которые годятся как исходник, когда видео нет вовсе — так
#:     выглядит эпизод, скачанный с YouTube без видеодорожки. Собственные
#:     спутники (.mp3/.wav) исключены: иначе они сами стали бы «эпизодами».
#: EN: Containers usable as a source when there is no video at all — what a
#:     YouTube fetch without the video track leaves behind. Our own companions
#:     (.mp3/.wav) are excluded, or they would register as episodes themselves.
AUDIO_ONLY_EXTS: tuple[str, ...] = (".m4a", ".opus", ".aac", ".ogg", ".flac")


def find_input_queue(
    input_dir: Path,
    *,
    only_stems: Collection[str] | None = None,
    ensure_companions: bool = True,
) -> list[dict[str, Any]]:
    """RU: Находит эпизоды и гарантирует MP3/WAV-спутники для каждого.

    EN: Find episodes and ensure each one has its MP3/WAV companions.

    The scan is recursive so a download folder such as ``input/youtube/`` is
    seen; every pre-existing file sits at the top level, so nothing changes for
    them. The stem stays the single identity key and the output layout stays
    flat — ids in YouTube filenames make collisions a non-issue.

    ``only_stems`` narrows the queue **before** any ffmpeg runs. That ordering is
    the point: building companions for a folder of untouched local episodes would
    cost an hour of transcoding that the caller never asked for.

    An entry whose stem has audio but no video gets ``video=None``; every stage
    except cutting works from the audio anyway.

    yt-dlp's per-format pieces (``… [id].f137.mp4``) are never episodes: they
    are what an interrupted download leaves before merging, and a video-only
    piece used to abort every following run at the companion step.

    ``ensure_companions=False`` only computes the companion paths; the
    pipeline builds them per episode, inside that episode's failure guard, so
    one broken file costs one episode instead of the whole queue.
    """
    if not input_dir.exists():
        return []

    wanted = set(only_stems) if only_stems is not None else None

    videos: dict[str, Path] = {}
    audio_only: dict[str, Path] = {}

    for p in sorted(input_dir.rglob("*")):
        if not p.is_file():
            continue
        stem = p.stem
        if wanted is not None and stem not in wanted:
            continue
        if is_download_fragment(p):
            continue
        suffix = p.suffix.lower()
        if suffix in VIDEO_EXTS:
            current = videos.get(stem)
            # Pick newest if multiple video formats for the same stem.
            if current is None or p.stat().st_mtime > current.stat().st_mtime:
                videos[stem] = p
        elif suffix in AUDIO_ONLY_EXTS:
            current = audio_only.get(stem)
            if current is None or p.stat().st_mtime > current.stat().st_mtime:
                audio_only[stem] = p

    queue = []
    for stem in sorted(set(videos) | set(audio_only)):
        video_path = videos.get(stem)
        source_path = video_path if video_path is not None else audio_only[stem]
        if ensure_companions:
            mp3_path, wav_path = _ensure_audio_companions(source_path)
        else:
            mp3_path, wav_path = source_path.with_suffix(".mp3"), source_path.with_suffix(".wav")
        queue.append({
            "stem": stem,
            "source": source_path,
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


def _analysis_outputs_ready(
    moments_path: Path,
    reels_md_path: Path,
    *,
    validate_json: bool,
) -> bool:
    """RU: Готов ли анализ на самом деле, а не только по наличию файлов.

    EN: Whether the analysis is genuinely done, not merely present on disk.

    RU: Упавшая стадия оставляет после себя плейсхолдер `[]` — по размеру и
        синтаксису он неотличим от удачного разбора, поэтому кэш навсегда
        пропускал анализ, и резать было нечего. Пустой список моментов — это не
        результат, а повод пересчитать.
    EN: A failed stage leaves a `[]` placeholder behind — by size and syntax it
        is indistinguishable from a successful parse, so the cache skipped the
        analysis forever and the cut stage had nothing to work with. An empty
        moments list is not a result; it is a reason to redo the analysis.
    """
    if not _outputs_ready([moments_path, reels_md_path], validate_json=validate_json):
        return False
    moments = _read_json_if_valid(moments_path)
    if isinstance(moments, list) and moments:
        return True
    # RU: Пустой список — результат, только если анализ дошёл до конца: иначе
    #     эпизод без подходящих клипов пересчитывался бы каждую ночь.
    # EN: An empty list is a result only when the analysis ran to the end;
    #     otherwise an episode with nothing worth cutting would be redone
    #     every night.
    marker = _read_json_if_valid(moments_path.with_name("analysis_complete.json"))
    return isinstance(moments, list) and isinstance(marker, dict) and marker.get("status") == "ok"


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
    "fetch",
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


def _youtube_conf(
    conf: dict[str, Any], overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """RU: Блок youtube из конфига поверх значений с CLI.

    EN: The youtube config block, with CLI overrides applied on top.

    Only keys the caller actually set are taken from ``overrides``: argparse
    hands us ``None`` for everything untouched, and letting those through would
    wipe the config values they are meant to override.
    """

    base = conf.get("youtube") if isinstance(conf, dict) else None
    merged: dict[str, Any] = dict(base) if isinstance(base, dict) else {}
    for key, value in (overrides or {}).items():
        if value is not None:
            merged[key] = value
    return merged


def _run_youtube_fetch(
    *,
    conf: dict[str, Any],
    input_dir: Path,
    active: set[str],
    sources: Sequence[str] | None,
    cli_exclude: Sequence[str] | None,
    overrides: dict[str, Any] | None,
    scope: bool,
    list_only: bool,
    skip_existing: bool,
    quiet: bool,
    verbose: bool,
    progress: bool,
    report: RunReport | None = None,
) -> set[str] | None:
    """RU: Разворачивает источники YouTube и скачивает недостающее.

    EN: Expand the YouTube sources and download whatever is missing.

    Returns the stems this run should be limited to, or ``None`` for "no
    narrowing" — either nothing was requested from YouTube or the caller asked
    for the whole input folder.

    Resolution happens even when the ``fetch`` stage is not selected: the run
    still has to know which episodes were meant, so ``--youtube <link> --only
    analyze`` works on already-downloaded material.
    """

    yt_conf = _youtube_conf(conf, overrides)

    cli_sources = [str(s).strip() for s in (sources or []) if str(s).strip()]
    conf_sources = yt_conf.get("sources")
    conf_sources = (
        [str(s).strip() for s in conf_sources if str(s).strip()]
        if isinstance(conf_sources, list)
        else []
    )
    # CLI first: "the link I just typed" should head the list.
    all_sources = cli_sources + [s for s in conf_sources if s not in cli_sources]
    if not all_sources:
        return None

    # RU: Исключения складываются, а не переопределяются: постоянный список в
    #     конфиге («никогда не брать этот подкаст») должен пережить разовое
    #     --yt-exclude, иначе одна команда молча снимет правило.
    # EN: Exclusions add up rather than override: a standing config list ("never
    #     take this show") must survive a one-off --yt-exclude, or a single
    #     command would quietly lift the rule.
    conf_exclude = yt_conf.get("exclude")
    exclude = [
        str(s).strip()
        for s in (conf_exclude if isinstance(conf_exclude, list) else [])
        if str(s).strip()
    ]
    exclude += [s for s in (cli_exclude or []) if s not in exclude]

    download_dir = resolve_download_dir(yt_conf, input_dir)

    filters = VideoFilters(
        limit=max(0, int(yt_conf.get("limit") or 0)),
        since=yt_conf.get("since") or None,
        until=yt_conf.get("until") or None,
        min_duration=max(0, int(yt_conf.get("min_duration") or 0)),
        max_duration=max(0, int(yt_conf.get("max_duration") or 0)),
        skip_live=bool(yt_conf.get("skip_live", True)),
    )
    api_key = api_key_from_env(str(yt_conf.get("api_key_env") or "YOUTUBE_API_KEY"))

    status(f"[fetch] источники: {', '.join(all_sources)}", quiet=quiet)
    if exclude:
        status(f"[fetch] исключения: {', '.join(exclude)}", quiet=quiet)
    if not api_key and not quiet:
        status(
            "[fetch] YOUTUBE_API_KEY не задан — перечисление делает yt-dlp "
            "(без дат публикации до скачивания)",
            quiet=quiet,
        )

    try:
        videos = resolve_sources(
            all_sources, api_key=api_key, filters=filters, exclude=exclude,
        )
    except YouTubeError as exc:
        if list_only:
            raise SystemExit(f"YouTube: {exc}") from exc
        # RU: Нет сети или YouTube отвечает ошибкой — это не повод бросать уже
        #     скачанное: обрабатываем то, что лежит в папке загрузок.
        # EN: No network, or YouTube errors out — no reason to abandon what is
        #     already downloaded: process whatever sits in the download folder.
        message = f"YouTube недоступен ({exc}); работаю с уже скачанным"
        log.error(message)
        status(f"[fetch] {message}", quiet=quiet)
        if report is not None:
            report.event("error", message)
        if not scope:
            return None
        return {
            p.stem
            for p in (download_dir.iterdir() if download_dir.exists() else [])
            if p.is_file() and not is_download_fragment(p)
            and p.suffix.lower() in VIDEO_EXTS + AUDIO_ONLY_EXTS
        }

    if list_only:
        print(describe_videos(videos, download_dir))
        return None

    if not videos:
        status("[fetch] под условия отбора не попал ни один ролик", quiet=quiet)
        return set() if scope else None

    if "fetch" not in active:
        status(f"[fetch] skip (not selected); роликов выбрано: {len(videos)}", quiet=quiet)
        fetched = locate_existing(videos, download_dir)
    else:
        want_video = want_video_for_stages(yt_conf, active)
        config = FetchConfig(
            download_dir=download_dir,
            want_video=want_video,
            max_height=max(0, int(yt_conf.get("max_height") or 0)),
            filename_template=str(
                yt_conf.get("filename_template") or DEFAULT_FILENAME_TEMPLATE,
            ),
            archive=(Path(str(yt_conf["archive"])) if yt_conf.get("archive") else None),
            skip_existing=skip_existing,
            cookies_file=(
                Path(str(yt_conf["cookies_file"])) if yt_conf.get("cookies_file") else None
            ),
            retries=int(yt_conf.get("retries") or 3),
            rate_limit=(str(yt_conf["rate_limit"]) if yt_conf.get("rate_limit") else None),
            # RU: Сквозной проброс опций yt-dlp. YouTube регулярно меняет отдачу
            #     видео, и лечится это обычно одной опцией — не хочется ради
            #     каждой такой правки трогать код.
            # EN: A pass-through for raw yt-dlp options. YouTube keeps changing
            #     how it serves video, and the fix is usually a single option —
            #     which should not require a code change every time.
            extra_options=(
                dict(yt_conf["ydl_options"])
                if isinstance(yt_conf.get("ydl_options"), dict)
                else {}
            ),
            quiet=quiet,
            verbose=verbose,
        )
        status(
            f"[fetch] start ({len(videos)} шт., "
            f"{'видео до ' + str(config.max_height) + 'p' if want_video else 'только аудио'})",
            quiet=quiet,
        )
        bar = tqdm(
            total=len(videos),
            disable=(not progress) or quiet,
            desc="YouTube",
        )
        try:
            fetched = fetch_videos(
                videos, config, on_progress=lambda _v: bar.update(1),
            )
        finally:
            bar.close()
        downloaded = sum(1 for f in fetched if f.downloaded)
        status(
            f"[fetch] done (скачано {downloaded}, уже было {len(fetched) - downloaded}, "
            f"не получилось {len(videos) - len(fetched)})",
            quiet=quiet,
        )

    if not scope:
        return None
    return {f.stem for f in fetched}


@dataclass
class EpisodeState:
    """RU: Пути и промежуточное состояние одного эпизода.

    EN: Paths and in-flight state of one episode.
    """

    stem: str
    source: Path
    video: Path | None
    audio: Path
    wav: Path
    output_dir: Path
    analysis_folder: Path
    transcript_path: Path
    transcript_srt_path: Path
    diar_path: Path
    #: Set when a stage the rest depends on failed; later phases skip it.
    broken: bool = False
    #: The transcript subtitles and captions are built from. None: the same
    #: one the analysis used (``transcript_path``).
    subtitle_transcript: Path | None = None

    @property
    def raw_transcript_path(self) -> Path:
        name = self.transcript_path.name.replace(".proofread.json", ".json")
        return self.transcript_path.with_name(name)

    @property
    def proofread_path(self) -> Path:
        raw = self.raw_transcript_path
        return raw.with_name(raw.stem + ".proofread.json")

    @property
    def state(self) -> StageState:
        return StageState(self.output_dir)

    @property
    def model_audio(self) -> Path:
        # RU: Моделям отдаём PCM: у Whisper нет артефактов mp3, а pyannote на
        #     mp3 просто падает (обрезка даёт на несколько сэмплов меньше).
        # EN: The models get the PCM copy: Whisper avoids the mp3 artefacts and
        #     pyannote outright fails on mp3 (a crop comes back a few samples
        #     short of what it asked for).
        if _file_has_content(self.wav):
            return self.wav
        return self.audio if _file_has_content(self.audio) else self.source

    @property
    def moments_path(self) -> Path:
        return self.analysis_folder / "moments.json"

    @property
    def reels_md_path(self) -> Path:
        return self.analysis_folder / "reels.md"


class _LlamaSession:
    """RU: Один запуск llama-server на группу LLM-стадий.

    EN: One llama-server session for a group of LLM stages.

    Starting a server that is already running is a no-op, and a server that
    could not be started at all is not waited on: the old flow spent up to
    300 s per episode polling a port nobody was going to open.
    """

    def __init__(self, a_conf: dict[str, Any]) -> None:
        self.url = str(a_conf.get("url", "http://127.0.0.1:8080/v1/chat/completions")).strip()
        self.local = parse_local_llama_cpp_host_port(self.url)
        service = a_conf.get("service", {}) if isinstance(a_conf, dict) else {}
        self.service_conf: dict[str, Any] = service if isinstance(service, dict) else {}
        self._proc: subprocess.Popen | None = None
        self.opened = False

    def open(self) -> None:
        if self.opened:
            return
        self.opened = True
        if not self.local:
            return
        host, port = self.local
        if bool(self.service_conf.get("auto_start", True)):
            self._proc = llama_cpp_start(host=host, port=port, service_conf=self.service_conf)
        if self._proc is None and not is_tcp_open(host, port):
            log.error(
                "llama-server на %s:%s не запущен и не отвечает; LLM-стадии будут падать",
                host, port,
            )
            return
        # Wait for the model to load: both our own instance and an external
        # server answer 503 "Loading model" until they are ready.
        startup_timeout = int(self.service_conf.get("startup_timeout", 120))
        wait_for_server_ready(host, port, timeout_s=max(startup_timeout, 300))

    def close(self) -> None:
        if not self.opened:
            return
        self.opened = False
        if self._proc is not None:
            llama_cpp_stop(self._proc)
            self._proc = None
        # Free VRAM from any llama-server (ours or external) before the cut
        # stage so NVENC — or the next Whisper load — gets the GPU.
        if self.local:
            host, port = self.local
            if is_tcp_open(host, port):
                _kill_llama_server(port)


def _describe_failure(exc: BaseException) -> str:
    if isinstance(exc, SystemExit):
        code = exc.code
        return f"exit {code}" if isinstance(code, int) else str(code)
    return f"{type(exc).__name__}: {exc}"


class _PipelineRun:
    """RU: Один прогон по очереди эпизодов, со стадиями и отчётом.

    EN: One run over the episode queue, with its stages and report.

    Every stage of every episode runs inside a guard: whatever it raises —
    including the ``SystemExit`` of a stage subprocess — is recorded in the
    report and costs that episode (or just that stage, for optional ones),
    never the rest of the queue.
    """

    def __init__(
        self,
        *,
        conf: dict[str, Any],
        repo_dir: Path,
        quiet: bool,
        verbose: bool,
        skip_existing: bool,
        autotune: bool,
        progress: bool,
        active: set[str],
        report: RunReport,
    ) -> None:
        self.conf = conf
        self.repo_dir = repo_dir
        self.quiet = quiet
        self.verbose = verbose
        self.skip_existing = skip_existing
        self.autotune = autotune
        self.progress = progress
        self.report = report

        cache_conf = conf.get("cache", {}) if isinstance(conf, dict) else {}
        self.validate_json = bool(cache_conf.get("validate_json", True))

        self.diar_conf = conf.get("diarization", {}) or {}
        proofread_conf = conf.get("proofread", {})
        self.proofread_conf = proofread_conf if isinstance(proofread_conf, dict) else {}
        article_conf = conf.get("article", {})
        self.article_conf = article_conf if isinstance(article_conf, dict) else {}

        self.transcribe_enabled = "transcribe" in active
        self.diar_enabled = bool(self.diar_conf.get("enabled", False)) and "diarize" in active
        self.proofread_enabled = bool(self.proofread_conf.get("enabled", False)) and "proofread" in active
        self.article_enabled = bool(self.article_conf.get("enabled", False)) and "article" in active
        self.analyze_enabled = "analyze" in active
        self.cut_enabled = "cut" in active

        self.subtitle_settings = subtitle_settings_from_conf(conf, repo_dir=repo_dir)
        a_conf = conf.get("llama_cpp", {})
        self.a_conf: dict[str, Any] = a_conf if isinstance(a_conf, dict) else {}
        self.prompts_conf = conf.get("prompts", {})
        self.p_conf = conf.get("processing", {})
        v_conf = conf.get("video", {})
        self.v_conf: dict[str, Any] = v_conf if isinstance(v_conf, dict) else {}
        self.exports_conf = conf.get("exports", {})

        if autotune:
            self.a_conf = _autotune_llama_cpp_conf(self.a_conf)
            self.v_conf = {**self.v_conf, "threads": _autotune_video_threads(self.v_conf)}
            service = self.a_conf.get("service", {})
            status(
                "[autotune] "
                f"llama threads={service.get('threads')} parallel={service.get('parallel')} "
                f"ctx={service.get('ctx_size')} video_jobs={self.v_conf.get('threads')}",
                quiet=quiet,
            )

        self.roles = resolve_llama_cpp_role_mapping(conf)
        self.final_model_folder = _model_folder_name(self.roles.judge_metadata)
        self.llama = _LlamaSession(self.a_conf)
        self.stage_bar: Any = None

        autonomy = conf.get("autonomy") if isinstance(conf.get("autonomy"), dict) else {}
        self.scheduling = str((autonomy or {}).get("scheduling", "stage")).strip().lower()
        audio_conf = conf.get("audio") if isinstance(conf.get("audio"), dict) else {}
        self.listening_copy = bool((audio_conf or {}).get("listening_copy", True))
        self.delete_wav = bool((audio_conf or {}).get("delete_wav_after_analysis", False))
        # RU: scope=clips — вычитывать только отрезки выбранных клипов (для
        #     субтитров и подписей). Статье нужен весь текст, поэтому при
        #     включённой статье вычитка всегда полная.
        # EN: scope=clips proofreads only the selected clips' spans (what
        #     subtitles and captions show). The article needs the whole text,
        #     so with the article enabled the scope is always full.
        scope = str(self.proofread_conf.get("scope", "full")).strip().lower()
        self.proofread_clips_only = self.proofread_enabled and scope == "clips"
        if self.proofread_clips_only and self.article_enabled:
            log.warning("proofread.scope=clips ignored: the article needs the whole transcript")
            self.proofread_clips_only = False
        self.llm_needed = self.proofread_enabled or self.article_enabled or self.analyze_enabled

    # -- plumbing -------------------------------------------------------------

    def _tick(self) -> None:
        if self.stage_bar is not None:
            self.stage_bar.update(1)

    def guard(self, ep: EpisodeState, stage: str, fn: Any) -> bool:
        """Run one stage; record its outcome. False when it failed."""

        started = time.monotonic()
        try:
            outcome = fn(ep)
        except KeyboardInterrupt:
            raise
        except (Exception, SystemExit) as exc:
            detail = _describe_failure(exc)
            if isinstance(exc, SystemExit):
                log.error("[%s] %s: %s", stage, ep.stem, detail)
            else:
                log.exception("[%s] %s failed", stage, ep.stem)
            status(f"[{stage}] FAILED ({ep.stem}): {detail}", quiet=self.quiet)
            self.report.record(
                ep.stem, stage, FAILED, seconds=time.monotonic() - started, detail=detail,
            )
            return False
        self.report.record(
            ep.stem, stage, str(outcome or DONE), seconds=time.monotonic() - started,
        )
        return True

    def episode_state(self, item: dict[str, Any], base_output_dir: Path) -> EpisodeState:
        stem = str(item["stem"])
        output_dir = base_output_dir / stem
        audio = Path(item["audio"])
        transcript_path = output_dir / audio.with_suffix(".json").name
        transcript_srt_path = output_dir / audio.with_suffix(".srt").name
        legacy_audio_json = output_dir / "audio.json"
        if not transcript_path.exists() and legacy_audio_json.exists():
            transcript_path = legacy_audio_json
            transcript_srt_path = legacy_audio_json.with_suffix(".srt")
        return EpisodeState(
            stem=stem,
            source=Path(item.get("source") or item.get("video") or audio),
            video=item.get("video"),
            audio=audio,
            wav=Path(item.get("wav") or audio),
            output_dir=output_dir,
            analysis_folder=output_dir / self.final_model_folder,
            transcript_path=transcript_path,
            transcript_srt_path=transcript_srt_path,
            diar_path=output_dir / "diarization.json",
        )

    # -- phases ---------------------------------------------------------------

    def run_episode(self, ep: EpisodeState) -> None:
        status(f"\n[forge] processing: {ep.stem}", quiet=self.quiet)
        status(
            f"[forge] video: {ep.video.name}" if ep.video is not None
            else "[forge] video: нет (источник только аудио)",
            quiet=self.quiet,
        )
        self.begin_episode(ep)

        self.phase_prepare(ep)
        if not ep.broken:
            if self.llm_needed:
                self.llama.open()
            try:
                self.phase_llm(ep)
            finally:
                self.llama.close()
        self.finish_episode(ep)

    def begin_episode(self, ep: EpisodeState) -> None:
        os.makedirs(ep.output_dir, exist_ok=True)
        self.report.episode(ep.stem)

    def finish_episode(self, ep: EpisodeState) -> None:
        if not ep.broken:
            self.phase_cut(ep)
        self.report.episode(ep.stem).clips = self._count_reels(ep)

    def run_stage_major(self, episodes: list[EpisodeState]) -> None:
        """RU: Очередь по стадиям: все транскрипции, одна сессия LLM, вся нарезка.

        EN: The queue stage by stage: every transcription, one LLM session,
        every cut. Whisper loads once and llama-server (~14 GB of weights)
        starts once for the whole queue instead of once per episode, and the
        two never fight over VRAM.
        """

        status(f"\n[forge] {len(episodes)} episode(s), stage by stage", quiet=self.quiet)
        with whisper_model_session():
            for ep in episodes:
                status(f"\n[forge] prepare: {ep.stem}", quiet=self.quiet)
                self.begin_episode(ep)
                self.phase_prepare(ep)
        ready = [ep for ep in episodes if not ep.broken]
        if ready and self.llm_needed:
            self.llama.open()
            try:
                for ep in ready:
                    status(f"\n[forge] llm: {ep.stem}", quiet=self.quiet)
                    self.phase_llm(ep)
            finally:
                self.llama.close()
        elif ready:
            for ep in ready:
                self.phase_llm(ep)
        for ep in episodes:
            if not ep.broken:
                status(f"\n[forge] cut: {ep.stem}", quiet=self.quiet)
            self.finish_episode(ep)

    def phase_prepare(self, ep: EpisodeState) -> None:
        if not self.guard(ep, "audio", self.stage_audio):
            ep.broken = True
            return
        status(f"[forge] transcribe input: {ep.model_audio.name}", quiet=self.quiet)
        if not self.guard(ep, "transcribe", self.stage_transcribe):
            ep.broken = True
            if self.transcribe_enabled:
                self._tick()
            return
        if self.transcribe_enabled:
            self._tick()
        if self.diar_enabled:
            # Speakers are an enrichment: without them the analysis still works.
            self.guard(ep, "diarize", self.stage_diarize)
            self._tick()

    def phase_llm(self, ep: EpisodeState) -> None:
        if self.proofread_enabled and not self.proofread_clips_only:
            self.guard(ep, "proofread", self.stage_proofread)
            self._tick()
        if not self.proofread_clips_only:
            self._adopt_existing_proofread(ep)
        if self.article_enabled:
            self.guard(ep, "article", self.stage_article)
            self._tick()
        if self.analyze_enabled:
            self.guard(ep, "analyze", self.stage_analyze)
            self._tick()
        else:
            status("[analyze] skip (not selected)", quiet=self.quiet)
        if self.proofread_clips_only:
            # After the analysis: only now is it known which spans matter.
            self.guard(ep, "proofread", self.stage_proofread_clips)
            self._tick()
        if self.delete_wav and ep.wav.exists() and ep.wav != ep.source:
            # Transcription, diarization and audio probing are done with it.
            ep.wav.unlink(missing_ok=True)

    def phase_cut(self, ep: EpisodeState) -> None:
        moments_data = _read_json_if_valid(ep.moments_path)
        if moments_data is None or (isinstance(moments_data, list) and not moments_data):
            # RU: Пустой moments.json резать нечем, и молчать об этом нельзя:
            #     именно так выглядит эпизод, у которого анализ не дал моментов.
            # EN: An empty moments.json leaves nothing to cut, and that must be
            #     said out loud: it is how an episode with no analysed moments looks.
            status(
                f"[cut] skip ({self.final_model_folder}): no moments — "
                "анализ не дал ни одного момента, перезапустите стадию analyze",
                quiet=self.quiet,
            )
            if self.cut_enabled:
                self.report.record(ep.stem, "cut", SKIPPED, detail="no moments")
                self._tick()
            return

        skip_cut = self._cut_should_skip(ep)
        if self.cut_enabled and not skip_cut:
            self.guard(ep, "cut", self.stage_cut)
        elif self.cut_enabled:
            self.report.record(ep.stem, "cut", SKIPPED if ep.video is None else CACHED)

        if isinstance(moments_data, list):
            moments = [m for m in moments_data if isinstance(m, dict)]
            self.guard(ep, "captions", lambda e: self.stage_captions(e, moments, skip_cut))
        if self.cut_enabled:
            self._tick()

    # -- stages ---------------------------------------------------------------

    def stage_audio(self, ep: EpisodeState) -> str:
        transcript_ready = self.skip_existing and _outputs_ready(
            [ep.transcript_path, ep.transcript_srt_path], validate_json=self.validate_json,
        )
        needs_wav = not transcript_ready or self.diar_enabled
        need_mp3 = self.listening_copy and not _file_has_content(ep.audio)
        need_wav = needs_wav and not _file_has_content(ep.wav)
        if not need_mp3 and not need_wav:
            return CACHED
        mp3_path, wav_path = _ensure_audio_companions(ep.source, want_mp3=self.listening_copy)
        ep.audio, ep.wav = mp3_path, wav_path
        return DONE

    def stage_transcribe(self, ep: EpisodeState) -> str:
        if not self.transcribe_enabled:
            status("[transcribe] skip (not selected)", quiet=self.quiet)
            return SKIPPED
        if self.skip_existing and _outputs_ready(
            [ep.transcript_path, ep.transcript_srt_path], validate_json=self.validate_json,
        ):
            status("[transcribe] skip (exists)", quiet=self.quiet)
            return CACHED

        configured_url = str(self.a_conf.get("url", "")).strip()
        local = parse_local_llama_cpp_host_port(configured_url) if configured_url else None
        if local:
            # Whisper large-v3 needs ~10GB VRAM. Kill any llama-server that
            # holds the GPU before we start transcription.
            _kill_llama_server(local[1])

        status("[transcribe] start", quiet=self.quiet)
        t_conf = self.conf.get("transcription", {})
        device_raw = str(t_conf.get("device", "cuda")).strip().lower()
        if device_raw == "auto":
            device_raw = "cuda" if _has_cuda() else "cpu"

        compute_type_raw = t_conf.get("compute_type")
        compute_type = None
        if isinstance(compute_type_raw, str) and compute_type_raw.strip():
            compute_type = compute_type_raw.strip()
            if self.autotune and compute_type.lower() == "auto":
                compute_type = None

        transcribe_config = TranscribeConfig(
            input_path=ep.model_audio,
            outdir=ep.output_dir,
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
            condition_on_previous_text=bool(t_conf.get("condition_on_previous_text", False)),
            mode=str(t_conf.get("mode", "fast")),
            initial_prompt=t_conf.get("initial_prompt") or None,
            quality_beam_size=int(t_conf.get("quality_beam_size", 10)),
            quiet=self.quiet,
            verbose=self.verbose,
        )
        ep.transcript_path = transcribe_file(transcribe_config)
        ep.transcript_srt_path = ep.transcript_path.with_suffix(".srt")
        status("[transcribe] done", quiet=self.quiet)
        return DONE

    def stage_diarize(self, ep: EpisodeState) -> str:
        if self.skip_existing and _outputs_ready([ep.diar_path], validate_json=self.validate_json):
            status("[diarize] skip (exists)", quiet=self.quiet)
            return CACHED
        status("[diarize] start", quiet=self.quiet)
        diarize_args = [
            "--input", str(ep.model_audio),
            "--outdir", str(ep.output_dir),
            "--model", str(self.diar_conf.get("model", "pyannote/speaker-diarization")),
        ]
        num_speakers = self.diar_conf.get("num_speakers")
        if num_speakers:
            diarize_args += ["--num-speakers", str(int(num_speakers))]
        if self.quiet:
            diarize_args.append("--quiet")
        if self.verbose:
            diarize_args.append("--verbose")
        run_module(
            "podcast_reels_forge.scripts.diarize",
            diarize_args,
            quiet=self.quiet,
            verbose=self.verbose,
        )
        status("[diarize] done", quiet=self.quiet)
        return DONE

    def stage_proofread(self, ep: EpisodeState) -> str:
        """Proofread the transcript; on failure the raw one is used."""

        proofread_path = ep.transcript_path.with_name(ep.transcript_path.stem + ".proofread.json")
        proofread_srt_path = proofread_path.with_suffix(".srt")
        if self.skip_existing and _outputs_ready(
            [proofread_path, proofread_srt_path], validate_json=self.validate_json,
        ):
            status("[proofread] skip (exists)", quiet=self.quiet)
            ep.transcript_path = proofread_path
            return CACHED
        status(f"[proofread] start ({self.roles.proofread})", quiet=self.quiet)
        asyncio.run(run_proofread(
            transcript_path=ep.transcript_path,
            output_path=proofread_path,
            url=self.llama.url,
            model=self.roles.proofread,
            proofread_conf=self.proofread_conf,
            prompts_conf=self.prompts_conf,
            quiet=self.quiet,
            verbose=self.verbose,
        ))
        ep.transcript_path = proofread_path
        status("[proofread] done", quiet=self.quiet)
        return DONE

    def _clip_ranges(self, ep: EpisodeState) -> list[tuple[float, float]]:
        moments = _read_json_if_valid(ep.moments_path)
        if not isinstance(moments, list):
            return []
        margin = float(self.p_conf.get("reel_padding", 5)) + 2.0
        ranges: list[tuple[float, float]] = []
        for moment in moments:
            if not isinstance(moment, dict):
                continue
            try:
                start, end = float(moment["start"]), float(moment["end"])
            except (KeyError, TypeError, ValueError):
                continue
            ranges.append((round(max(0.0, start - margin), 3), round(end + margin, 3)))
        ranges.sort()
        merged: list[tuple[float, float]] = []
        for low, high in ranges:
            if merged and low <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], high))
            else:
                merged.append((low, high))
        return merged

    def stage_proofread_clips(self, ep: EpisodeState) -> str:
        """Proofread only what the selected clips show (subtitles, captions)."""

        ranges = self._clip_ranges(ep)
        if not ranges:
            return SKIPPED
        raw = ep.raw_transcript_path
        current = fingerprint(
            "proofread-clips", file_digest(raw), ranges, self.proofread_conf, self.roles.proofread,
        )
        target = ep.proofread_path
        if (
            self.skip_existing
            and _outputs_ready([target, target.with_suffix(".srt")], validate_json=self.validate_json)
            and ep.state.get("proofread") == current
        ):
            ep.subtitle_transcript = target
            status("[proofread] skip (clips unchanged)", quiet=self.quiet)
            return CACHED
        status(f"[proofread] start: {len(ranges)} clip span(s) ({self.roles.proofread})", quiet=self.quiet)
        asyncio.run(run_proofread(
            transcript_path=raw,
            output_path=target,
            url=self.llama.url,
            model=self.roles.proofread,
            proofread_conf=self.proofread_conf,
            prompts_conf=self.prompts_conf,
            quiet=self.quiet,
            verbose=self.verbose,
            time_ranges=ranges,
        ))
        ep.state.set("proofread", current)
        ep.subtitle_transcript = target
        status("[proofread] done", quiet=self.quiet)
        return DONE

    def _adopt_existing_proofread(self, ep: EpisodeState) -> None:
        # RU: Вычитанный транскрипт мог быть сделан прошлым запуском. Даже
        #     если стадия сейчас не запускалась (--only analyze), дальше
        #     должен идти исправленный текст, а не сырой.
        # EN: The proofread transcript may come from an earlier run. Even
        #     when the stage did not execute this time (--only analyze), the
        #     corrected text is what the rest of the pipeline must use.
        existing = ep.transcript_path.with_name(ep.transcript_path.stem + ".proofread.json")
        if existing.exists() and _outputs_ready([existing], validate_json=self.validate_json):
            ep.transcript_path = existing

    def stage_article(self, ep: EpisodeState) -> str:
        """Retell the (proofread) transcript as prose. A side artefact."""

        article_stem = ep.transcript_path.stem.replace(".proofread", "")
        article_md_path = ep.transcript_path.with_name(article_stem + ".article.md")
        article_json_path = article_md_path.with_suffix(".json")
        if self.skip_existing and _outputs_ready(
            [article_md_path, article_json_path], validate_json=self.validate_json,
        ):
            status("[article] skip (exists)", quiet=self.quiet)
            return CACHED
        status(f"[article] start ({self.roles.article})", quiet=self.quiet)
        asyncio.run(run_article(
            transcript_path=ep.transcript_path,
            output_path=article_md_path,
            url=self.llama.url,
            model=self.roles.article,
            article_conf=self.article_conf,
            prompts_conf=self.prompts_conf,
            diarization_path=ep.diar_path if ep.diar_path.exists() else None,
            title=ep.stem,
            quiet=self.quiet,
            verbose=self.verbose,
        ))
        status("[article] done", quiet=self.quiet)
        return DONE

    def _analysis_fingerprint(self, ep: EpisodeState) -> str:
        """Everything the analysis result depends on."""

        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        prompt_files = [
            path
            for pattern in ("chunk_*.txt", "cleanup_*.txt", "judge_*.txt", "context_*.txt")
            for path in prompts_dir.glob(f"*/{pattern}")
        ]
        diar = ep.diar_path if self.diar_enabled and ep.diar_path.exists() else None
        return fingerprint(
            "analyze",
            __version__,
            file_digest(ep.transcript_path),
            file_digest(diar),
            self.p_conf,
            self.prompts_conf,
            self.roles.as_dict(),
            subset(
                self.a_conf,
                ("role_overrides", "model_overrides", "n_predict", "temperature",
                 "chunk_seconds", "max_chars_chunk", "scout_parallelism"),
            ),
            files_digest(prompt_files),
        )

    def stage_analyze(self, ep: EpisodeState) -> str:
        ep.analysis_folder.mkdir(parents=True, exist_ok=True)
        current = self._analysis_fingerprint(ep)
        if self.skip_existing and _analysis_outputs_ready(
            ep.moments_path, ep.reels_md_path, validate_json=self.validate_json,
        ):
            decision = ep.state.decide("analyze", current)
            if decision != "changed":
                status(f"[analyze] skip ({self.final_model_folder})", quiet=self.quiet)
                return CACHED
            status(
                "[analyze] inputs changed (transcript, config or prompts); re-running",
                quiet=self.quiet,
            )
        status(f"[analyze] start ({self.final_model_folder})", quiet=self.quiet)
        try:
            final_moments = asyncio.run(run_staged_analysis(
                transcript_path=ep.transcript_path,
                outdir=ep.analysis_folder,
                provider_name="llama_cpp",
                url=self.llama.url,
                api_key=None,
                roles=self.roles,
                llama_cpp_conf=self.a_conf,
                prompts_conf=self.prompts_conf,
                processing_conf=self.p_conf,
                diarization_path=(
                    ep.diar_path if self.diar_enabled and ep.diar_path.exists() else None
                ),
                quiet=self.quiet,
                verbose=self.verbose,
                progress=self.progress,
            ))
        finally:
            # A failed analysis still leaves valid (empty) outputs behind, so
            # downstream readers never trip over a missing file; an empty
            # moments list is what makes the next run retry it.
            _ensure_placeholder_analyze_outputs(ep.moments_path, ep.reels_md_path)
        ep.state.set("analyze", current)
        status(
            f"[analyze] done ({self.final_model_folder}, moments={len(final_moments)})",
            quiet=self.quiet,
        )
        return DONE

    def _existing_reels(self, ep: EpisodeState) -> list[Path]:
        reels_dir = ep.analysis_folder / "reels"
        if not reels_dir.exists():
            return []
        return [p for p in reels_dir.glob("reel_*.mp4") if re.match(r"^reel_\d+\.mp4$", p.name)]

    def _count_reels(self, ep: EpisodeState) -> int | None:
        reels = self._existing_reels(ep)
        return len(reels) if reels else None

    def _cut_should_skip(self, ep: EpisodeState) -> bool:
        if not self.cut_enabled:
            status("[cut] skip (not selected)", quiet=self.quiet)
            return True
        if ep.video is None:
            # RU: Резать нечего, если видео нет: так выглядит эпизод, скачанный
            #     только аудиодорожкой. Это не ошибка, но сказать надо прямо.
            # EN: Nothing to cut without a video: that is an audio-only fetch.
            #     Not an error, but it must be said out loud.
            status(
                f"[cut] skip ({ep.stem}): источник только аудио — "
                "перекачайте ролик со стадией cut в наборе",
                quiet=self.quiet,
            )
            return True
        if self.skip_existing and self._existing_reels(ep):
            decision = ep.state.decide("cut", self._cut_fingerprint(ep))
            if decision != "changed":
                status(f"[cut] skip ({self.final_model_folder}): exists", quiet=self.quiet)
                return True
            # Moments, subtitles or video settings changed: the old reels no
            # longer match what their captions will describe.
            status(
                f"[cut] inputs changed; re-cutting ({self.final_model_folder})",
                quiet=self.quiet,
            )
            self._discard_reels(ep)
        return False

    def _subtitle_transcript(self, ep: EpisodeState) -> Path:
        if ep.subtitle_transcript is not None:
            return ep.subtitle_transcript
        if self.proofread_clips_only and ep.proofread_path.exists():
            return ep.proofread_path
        return ep.transcript_path

    def _cut_fingerprint(self, ep: EpisodeState) -> str:
        subtitles = self.conf.get("subtitles")
        return fingerprint(
            "cut",
            __version__,
            file_digest(ep.moments_path),
            file_digest(self._subtitle_transcript(ep)) if self.subtitle_settings.enabled else "",
            file_identity(ep.video),
            self.v_conf,
            subtitles if isinstance(subtitles, dict) else {},
            self.exports_conf,
            subset(self.p_conf, ("quality_filters", "reel_padding")) if isinstance(self.p_conf, dict) else {},
        )

    def _discard_reels(self, ep: EpisodeState) -> None:
        reels_dir = ep.analysis_folder / "reels"
        if reels_dir.is_dir():
            shutil.rmtree(reels_dir)
        (ep.analysis_folder / "reels_preview.mp4").unlink(missing_ok=True)

    def stage_cut(self, ep: EpisodeState) -> str:
        status(f"[cut] start ({self.final_model_folder})", quiet=self.quiet)
        assert ep.video is not None
        padding = int(self.p_conf.get("reel_padding", 5))
        v_conf = self.v_conf
        video_args = [
            "--input", str(ep.video),
            "--moments", str(ep.moments_path),
            "--outdir", str(ep.analysis_folder),
            "--threads", str(v_conf.get("threads", 4)),
            "--v-bitrate", str(v_conf.get("video_bitrate", "5M")),
            "--a-bitrate", str(v_conf.get("audio_bitrate", "192k")),
            "--preset", str(v_conf.get("preset", "fast")),
            "--padding", str(padding),
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

        q_conf = self.p_conf.get("quality_filters", {}) if isinstance(self.p_conf, dict) else {}
        if "min_score" in q_conf:
            video_args += ["--filter-min-score", str(q_conf["min_score"])]
        if "min_duration" in q_conf:
            video_args += ["--filter-min-duration", str(q_conf["min_duration"])]
        if "max_duration" in q_conf:
            video_args += ["--filter-max-duration", str(q_conf["max_duration"])]
        if "face_min_ratio" in q_conf:
            video_args += ["--filter-face-ratio", str(q_conf["face_min_ratio"])]

        if self.exports_conf.get("webm", False):
            video_args.append("--export-webm")
        if self.exports_conf.get("gif", False):
            video_args.append("--export-gif")
        if self.exports_conf.get("audio_only", False):
            video_args.append("--export-audio")
        if self.subtitle_settings.enabled:
            video_args.append("--burn-subtitles")
            video_args += ["--transcript-json", str(self._subtitle_transcript(ep))]
            video_args += ["--subtitle-font", str(self.subtitle_settings.font_path)]
            if not self.subtitle_settings.wrap_words:
                video_args.append("--no-subtitle-wrap-words")
            subs_conf = self.conf.get("subtitles")
            if isinstance(subs_conf, dict) and subs_conf.get("keep_nosubs"):
                video_args.append("--keep-nosubs")
        if self.quiet:
            video_args.append("--quiet")
        if self.verbose:
            video_args.append("--verbose")

        q_filters = self.p_conf.get("quality_filters", {}) if isinstance(self.p_conf, dict) else {}
        if isinstance(q_filters, dict) and q_filters.get("render_rejected"):
            video_args.append("--render-rejected")

        run_module(
            "podcast_reels_forge.scripts.video_processor",
            video_args,
            quiet=self.quiet,
            verbose=self.verbose,
        )
        ep.state.set("cut", self._cut_fingerprint(ep))
        status(f"[cut] done ({self.final_model_folder})", quiet=self.quiet)
        return DONE

    def stage_captions(
        self, ep: EpisodeState, moments: list[dict[str, Any]], skip_cut: bool,
    ) -> str:
        """Per-reel markdown, and subtitle sidecars for reels cut earlier."""

        reels_dir = ep.analysis_folder / "reels"
        sync_reel_markdowns(moments, reels_dir)
        if self.subtitle_settings.enabled and skip_cut:
            sync_reel_burned_subtitles(
                moments,
                reels_dir,
                transcript_json_path=self._subtitle_transcript(ep),
                padding=int(self.p_conf.get("reel_padding", 5)),
                settings=self.subtitle_settings,
                verbose=self.verbose and not self.quiet,
            )
        return DONE

    def stages_per_episode(self) -> int:
        return sum(
            1
            for enabled in (
                self.transcribe_enabled,
                self.diar_enabled,
                self.proofread_enabled,
                self.article_enabled,
                self.analyze_enabled,
                self.cut_enabled,
            )
            if enabled
        )


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
    youtube_sources: Sequence[str] | None = None,
    youtube_exclude: Sequence[str] | None = None,
    youtube_overrides: dict[str, Any] | None = None,
    scope_to_youtube: bool = True,
    youtube_list_only: bool = False,
    report: RunReport | None = None,
) -> RunReport:
    """RU: Запускает полный пайплайн на основе config.yaml и файлов на диске.

    Аргументы:
        conf: Словарь конфигурации, загруженный из config.yaml.
        repo_dir: Путь к корню репозитория (для поиска prompts и т.п.).
        quiet: Подавлять статус-сообщения.
        verbose: Включить подробный лог.
        youtube_sources: Ссылки/handle с CLI; дополняют youtube.sources из конфига.
        youtube_exclude: Разовые исключения; складываются с youtube.exclude.
        youtube_overrides: Переопределения блока youtube с CLI.
        scope_to_youtube: Сузить очередь до роликов из youtube_sources.
        youtube_list_only: Показать список роликов и выйти, ничего не скачивая.
        report: Отчёт прогона; создаётся, если не передан.

    Возвращает отчёт: статус каждой стадии каждого эпизода. Ошибка эпизода
    записывается в отчёт и не останавливает очередь.

    EN: Run the full pipeline based on config and filesystem discovery.

    Returns the run report: the status of every stage of every episode. An
    episode's failure is recorded there and never stops the queue.
    """
    report = report if report is not None else RunReport()
    paths = conf.get("paths", {})
    input_dir_path = Path(str(paths.get("input_dir", "input")))
    base_output_dir = Path(str(paths.get("output_dir", "output")))

    cache_conf = conf.get("cache", {}) if isinstance(conf, dict) else {}
    if "enabled" in cache_conf:
        skip_existing = bool(cache_conf.get("enabled", True)) and skip_existing

    # A stage runs when the config enables it AND the caller selected it.
    active = set(stages) if stages is not None else set(PIPELINE_STAGES)

    # 0) YouTube: resolve the sources, download what is missing, and work out
    #    which episodes this run is about. Runs before the queue is built so the
    #    narrowing can spare untouched local files from the ffmpeg pass.
    only_stems = _run_youtube_fetch(
        conf=conf,
        input_dir=input_dir_path,
        active=active,
        sources=youtube_sources,
        cli_exclude=youtube_exclude,
        overrides=youtube_overrides,
        scope=scope_to_youtube,
        list_only=youtube_list_only,
        skip_existing=skip_existing,
        quiet=quiet,
        verbose=verbose,
        progress=progress,
        report=report,
    )
    if youtube_list_only or active == {"fetch"}:
        return report

    queue = find_input_queue(input_dir_path, only_stems=only_stems, ensure_companions=False)
    if not queue:
        if only_stems is not None:
            hint = "" if "fetch" in active else " Запустите стадию fetch, чтобы их скачать."
            status(f"RU: Обрабатывать нечего: подходящих файлов нет.{hint}", quiet=quiet)
            status("EN: Nothing to process: no matching files.", quiet=quiet)
        else:
            status(f"RU: В папке {input_dir_path} не найдено видео-файлов.", quiet=quiet)
            status(f"EN: No video files found in {input_dir_path}", quiet=quiet)
        return report

    run = _PipelineRun(
        conf=conf,
        repo_dir=repo_dir,
        quiet=quiet,
        verbose=verbose,
        skip_existing=skip_existing,
        autotune=autotune,
        progress=progress,
        active=active,
        report=report,
    )
    if not quiet and stages is not None:
        status(
            "[forge] stages: " + ", ".join(s for s in PIPELINE_STAGES if s in active),
            quiet=quiet,
        )

    # A plain progress object, advanced explicitly. Driving a tqdm iterator
    # with next() renders one step behind (tqdm counts an iteration when the
    # following next() arrives), so the bar sat at N-1 and finished at 3/4.
    run.stage_bar = tqdm(
        total=len(queue) * run.stages_per_episode(),
        disable=(not progress) or quiet,
        desc="Podcast Reels Forge",
    )
    episodes = [run.episode_state(item, base_output_dir) for item in queue]
    try:
        if run.scheduling == "episode":
            for ep in episodes:
                run.run_episode(ep)
        else:
            run.run_stage_major(episodes)
    finally:
        run.stage_bar.close()
    status(f"[forge] done: {report.summary_line()}", quiet=quiet)
    return report
