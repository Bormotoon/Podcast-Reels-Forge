#!/usr/bin/env python3
"""RU: Удобный лаунчер для перегенерации рилсов по готовым moments.json.

См. `python3 rerender_videos.py --help`.

EN: Convenience launcher to re-render reels from existing moments.json.

See `python3 rerender_videos.py --help`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


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


def main() -> None:
    ensure_venv()
    from podcast_reels_forge.scripts.rerender_videos import main as _main

    _main()


if __name__ == "__main__":
    main()
