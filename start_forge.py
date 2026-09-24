#!/usr/bin/env python3
"""RU: Главный оркестратор пайплайна Podcast Reels Forge.

EN: Main orchestrator for the Podcast Reels Forge pipeline.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
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

    run_pipeline(
        conf=conf,
        repo_dir=repo_dir,
        quiet=quiet,
        verbose=verbose,
        skip_existing=not args.no_skip_existing,
        autotune=bool(args.autotune),
        progress=not args.no_progress,
        stages=stages,
    )


if __name__ == "__main__":
    main()
