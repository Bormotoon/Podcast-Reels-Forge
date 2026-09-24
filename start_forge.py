#!/usr/bin/env python3
"""RU: Главный оркестратор пайплайна Podcast Reels Forge.

EN: Main orchestrator for the Podcast Reels Forge pipeline.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

log = logging.getLogger("Forge")


def ensure_venv() -> None:
    """RU: Перезапускает процесс в виртуальном окружении, если мы ещё не в нём.

    EN: Re-exec into the virtual environment if not already running there.
    """
    env_flag = "WHISPER_VENV_ACTIVE"
    if os.environ.get(env_flag) == "1":
        return

    # If we're already running inside any virtual environment, keep going.
    # This is more reliable than comparing executable paths, because venv
    # interpreters are often symlinks to the system Python.
    if sys.prefix != sys.base_prefix:
        os.environ[env_flag] = "1"
        return

    script_dir = Path(__file__).resolve().parent
    venv_python = script_dir / "whisper-env" / "bin" / "python"
    if venv_python.exists():
        os.environ[env_flag] = "1"
        os.execv(str(venv_python), [str(venv_python), *sys.argv])  # noqa: S606


ensure_venv()

# RU: Только после re-exec в venv. Эта цепочка тянет requests, aiohttp и torch —
#     импортируй её раньше, и запуск через системный python3 будет зависеть от
#     того, оказались ли они там случайно.
# EN: Only after the re-exec into the venv. This chain pulls in requests, aiohttp
#     and torch — importing it earlier makes a `python3 start_forge.py` launch
#     depend on the system interpreter happening to have them.
from podcast_reels_forge.pipeline import (  # noqa: E402
    PIPELINE_STAGES,
    resolve_stages,
    run_pipeline,
)


def _configure_logging(*, verbose: bool, quiet: bool) -> None:
    """RU: Настраивает уровень логирования по флагам CLI.

    EN: Configure logging level based on CLI flags.
    """
    level = logging.ERROR if quiet else (logging.DEBUG if verbose else logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main() -> None:
    """RU: Точка входа CLI для запуска пайплайна.

    EN: Main entry point for the pipeline.
    """
    ap = argparse.ArgumentParser(
        description="Podcast Reels Forge - Create viral clips from podcasts",
    )
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--quiet", action="store_true", help="Only errors")
    ap.add_argument(
        "--verbose", action="store_true", help="Verbose logs and subcommand output",
    )
    ap.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip stages even if output files already exist",
    )
    ap.add_argument(
        "--autotune",
        action="store_true",
        help="Auto-detect system and pick safer defaults where possible",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress UI (useful for logs/CI)",
    )
    ap.add_argument(
        "--only",
        metavar="STAGES",
        help=(
            "Run only these pipeline stages, comma-separated "
            f"({', '.join(PIPELINE_STAGES)}). Example: --only proofread,article"
        ),
    )
    ap.add_argument(
        "--skip",
        metavar="STAGES",
        help="Run every stage except these, comma-separated",
    )
    ap.add_argument(
        "--list-stages",
        action="store_true",
        help="Print the pipeline stages in order and exit",
    )
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Не проверять окружение перед стартом (llama-server, ffmpeg, токены, диск)",
    )
    ram = ap.add_mutually_exclusive_group()
    ram.add_argument(
        "--free-ram",
        action="store_true",
        help=(
            "Забрать у виртуалок неиспользуемую память на время прогона и вернуть "
            "в конце (переопределяет host_memory.enabled)"
        ),
    )
    ram.add_argument(
        "--no-free-ram",
        action="store_true",
        help="Не трогать память виртуалок, даже если host_memory.enabled: true",
    )

    yt = ap.add_argument_group(
        "YouTube",
        "Забрать материал прямо с YouTube. Значения по умолчанию — в блоке "
        "youtube: файла config.yaml; флаги ниже переопределяют их на один запуск.",
    )
    yt.add_argument(
        "--youtube",
        metavar="URL",
        action="append",
        default=None,
        help=(
            "Ссылка на ролик, плейлист или канал, либо @handle. Можно повторять. "
            "Значение, начинающееся с дефиса, отделяйте знаком =: --youtube=-3LisPanK24"
        ),
    )
    yt.add_argument(
        "--yt-exclude",
        metavar="URL",
        action="append",
        default=None,
        help=(
            "Никогда не брать эти ролики: ссылка на плейлист (удобнее всего), "
            "канал или отдельный ролик. Можно повторять. Складывается со "
            "списком youtube.exclude из конфига, а не заменяет его"
        ),
    )
    yt.add_argument(
        "--yt-list",
        action="store_true",
        help="Показать, какие ролики попадают под отбор, и выйти — ничего не скачивая",
    )
    yt.add_argument("--yt-limit", type=int, default=None, help="Взять только N последних")
    yt.add_argument("--yt-since", default=None, help="Не раньше даты, ГГГГ-ММ-ДД")
    yt.add_argument("--yt-until", default=None, help="Не позже даты, ГГГГ-ММ-ДД")
    yt.add_argument(
        "--yt-min-duration",
        type=int,
        default=None,
        help="Пропускать короче N секунд (60 отсекает Shorts)",
    )
    yt.add_argument(
        "--yt-max-duration", type=int, default=None, help="Пропускать длиннее N секунд",
    )
    yt.add_argument(
        "--yt-audio-only",
        dest="yt_download",
        action="store_const",
        const="audio",
        default=None,
        help="Качать только аудиодорожку (нарезать при этом будет нечего)",
    )
    yt.add_argument(
        "--yt-video",
        dest="yt_download",
        action="store_const",
        const="video",
        help="Качать видео, даже если стадия cut не выбрана",
    )
    yt.add_argument(
        "--yt-max-height", type=int, default=None, help="Потолок высоты видео, напр. 1080",
    )
    yt.add_argument("--yt-cookies", default=None, help="Файл cookies (Netscape)")
    yt.add_argument(
        "--yt-all-inputs",
        action="store_true",
        help=(
            "Не сужать очередь до скачанного: обрабатывать всё, что лежит "
            "во входной папке"
        ),
    )

    args = ap.parse_args()

    if args.list_stages:
        for name in PIPELINE_STAGES:
            print(name)
        return

    # Validate before loading the config so a typo fails immediately.
    stages = (
        resolve_stages(only=args.only, skip=args.skip)
        if (args.only or args.skip)
        else None
    )

    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    # RU: `.env` из корня проекта — источник ключей вроде YOUTUBE_API_KEY и
    #     PYANNOTE_TOKEN. Настоящая переменная окружения всегда важнее файла.
    # EN: The project-root `.env` supplies keys such as YOUTUBE_API_KEY and
    #     PYANNOTE_TOKEN. A real environment variable always beats the file.
    from podcast_reels_forge.utils.env import load_dotenv

    repo_dir = Path(__file__).resolve().parent
    load_dotenv(repo_dir)

    import yaml

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", args.config)
        sys.exit(1)

    with config_path.open(encoding="utf-8") as f:
        conf: dict[str, Any] = yaml.safe_load(f) or {}

    cli_conf = conf.get("cli", {}) if isinstance(conf, dict) else {}
    quiet = bool(args.quiet or cli_conf.get("quiet", False))
    verbose = bool(args.verbose or cli_conf.get("verbose", False))
    _configure_logging(verbose=verbose, quiet=quiet)

    from podcast_reels_forge.autonomy import (
        lock_path,
        notify,
        runs_dir,
        setup_file_logging,
    )
    from podcast_reels_forge.preflight import run_preflight
    from podcast_reels_forge.run_report import EXIT_BUSY, RunReport
    from podcast_reels_forge.utils.host_memory import HostMemoryConfig
    from podcast_reels_forge.utils.run_lock import RunLock, RunLockBusy

    host_memory = HostMemoryConfig.from_conf(conf.get("host_memory"))
    if args.no_free_ram:
        host_memory = replace(host_memory, enabled=False)
    elif args.free_ram:
        host_memory = replace(host_memory, enabled=True)

    youtube_overrides = {
        "limit": args.yt_limit,
        "since": args.yt_since,
        "until": args.yt_until,
        "min_duration": args.yt_min_duration,
        "max_duration": args.yt_max_duration,
        "download": args.yt_download,
        "max_height": args.yt_max_height,
        "cookies_file": args.yt_cookies,
    }

    # RU: --yt-list ничего не обрабатывает: ни блокировки, ни отчёта, ни
    #     освобождения памяти (оно стоит виртуалке полного цикла выключения).
    # EN: --yt-list processes nothing: no lock, no report, no memory freeing
    #     (which costs a VM a full shutdown-and-boot cycle).
    if args.yt_list:
        run_pipeline(
            conf=conf,
            repo_dir=repo_dir,
            quiet=quiet,
            verbose=verbose,
            stages=stages,
            youtube_sources=args.youtube,
            youtube_exclude=args.yt_exclude,
            youtube_overrides=youtube_overrides,
            scope_to_youtube=not args.yt_all_inputs,
            youtube_list_only=True,
        )
        return

    log_file = setup_file_logging(conf, repo_dir)
    if log_file is not None:
        log.info("log file: %s", log_file)

    lock = RunLock(lock_path(conf, repo_dir))
    try:
        lock.acquire()
    except RunLockBusy as exc:
        log.error("%s; этот запуск пропущен", exc)
        print(f"[forge] уже идёт другой прогон: {exc}", file=sys.stderr)
        sys.exit(EXIT_BUSY)

    report = RunReport()
    try:
        youtube_conf = conf.get("youtube") if isinstance(conf.get("youtube"), dict) else {}
        if not args.skip_preflight:
            preflight = run_preflight(
                conf,
                stages=stages if stages is not None else PIPELINE_STAGES,
                repo_dir=repo_dir,
                youtube_requested=bool(args.youtube or (youtube_conf or {}).get("sources")),
            )
            for warning in preflight.warnings:
                log.warning("preflight: %s", warning)
                report.event("warning", warning)
            if not preflight.ok:
                for error in preflight.errors:
                    log.error("preflight: %s", error)
                    print(f"[preflight] {error}", file=sys.stderr)
                report.fatal("preflight: " + "; ".join(preflight.errors))
        if report.fatal_error is None:
            _run_with_memory(
                host_memory=host_memory,
                repo_dir=repo_dir,
                quiet=quiet,
                report=report,
                pipeline_kwargs={
                    "conf": conf,
                    "repo_dir": repo_dir,
                    "quiet": quiet,
                    "verbose": verbose,
                    "skip_existing": not args.no_skip_existing,
                    "autotune": bool(args.autotune),
                    "progress": not args.no_progress,
                    "stages": stages,
                    "youtube_sources": args.youtube,
                    "youtube_exclude": args.yt_exclude,
                    "youtube_overrides": youtube_overrides,
                    "scope_to_youtube": not args.yt_all_inputs,
                    "report": report,
                },
            )
    finally:
        report.finish()
        report_path = None
        try:
            report_path = report.write(runs_dir(conf, repo_dir))
            log.info("run report: %s (%s)", report_path, report.summary_line())
        except OSError as exc:
            log.error("run report not written: %s", exc)
        notify(conf, report, report_path)
        lock.release()
        if not quiet:
            print(f"[forge] {report.summary_line()}", flush=True)
    sys.exit(report.exit_code())


def _run_with_memory(
    *,
    host_memory: Any,
    repo_dir: Path,
    quiet: bool,
    report: Any,
    pipeline_kwargs: dict[str, Any],
) -> None:
    """Free VM memory, run the pipeline, give the memory back — always."""

    from podcast_reels_forge.utils.host_memory import (
        free_host_memory,
        install_restore_on_signals,
        restore_host_memory,
    )

    def give_memory_back() -> None:
        restore_host_memory(host_memory, repo_dir=repo_dir, quiet=quiet)

    # RU: Возврат обязан произойти при любом исходе. finally закрывает обычный
    #     выход и исключения, обработчики — Ctrl+C и `kill`. Чего не закроет
    #     ничто: SIGKILL и OOM — на этот случай размеры лежат на диске, и
    #     следующий запуск (или --restore) вернёт их сам.
    # EN: The return must happen whatever the outcome. `finally` covers a
    #     normal exit and exceptions, the handlers cover Ctrl+C and `kill`.
    #     What nothing can cover is SIGKILL and the OOM killer — for those the
    #     sizes sit on disk, and the next run (or --restore) puts them back.
    install_restore_on_signals(give_memory_back)

    freed_mb = free_host_memory(host_memory, repo_dir=repo_dir, quiet=quiet)
    if freed_mb and not quiet:
        print(f"[ram] всего освобождено {freed_mb} МБ", flush=True)

    try:
        run_pipeline(**pipeline_kwargs)
    except SystemExit as exc:
        # A run-level abort (bad config, unusable input folder): record it.
        report.fatal(f"run aborted: {exc.code}")
        log.error("run aborted: %s", exc.code)
    except Exception as exc:
        report.fatal(f"run crashed: {type(exc).__name__}: {exc}")
        log.exception("run crashed")
    finally:
        give_memory_back()


if __name__ == "__main__":
    main()
