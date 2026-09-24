"""RU: Минимальный загрузчик `.env`.

Репозиторий возит `.env.example` и гитигнорит `.env`, но до сих пор никто этот файл
не читал: `PYANNOTE_TOKEN`, записанный в `.env`, до процесса не доходил. Здесь ровно
столько кода, сколько нужно, чтобы это починить — без зависимости от python-dotenv.

Настоящая переменная окружения всегда важнее файла (`setdefault`): `.env` — это
значения по умолчанию для запуска из папки проекта, а не способ переопределить то,
что пользователь выставил осознанно.

EN: Minimal `.env` loader.

The repo ships `.env.example` and git-ignores `.env`, yet nothing ever read that
file: a `PYANNOTE_TOKEN` written into `.env` never reached the process. This is
just enough code to fix that, without pulling in python-dotenv.

A real environment variable always wins over the file (`setdefault`): `.env` holds
defaults for running from the project directory, not a way to override what the
user set on purpose.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("Forge")


def parse_dotenv(text: str) -> dict[str, str]:
    """RU: Разбирает содержимое `.env` в словарь.

    EN: Parse `.env` contents into a mapping.

    Understands ``KEY=value``, a leading ``export``, ``#`` comments and quoted
    values. Anything malformed is skipped rather than raising: a stray line in a
    config file must not stop the pipeline.
    """

    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        out[key] = value
    return out


def load_dotenv(repo_dir: Path, *, filename: str = ".env") -> dict[str, str]:
    """RU: Подмешивает `.env` из корня проекта в окружение.

    EN: Merge the project-root `.env` into the environment.

    Returns the variables that were actually applied, so callers can log them
    without ever touching the values.
    """

    path = Path(repo_dir) / filename
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    applied: dict[str, str] = {}
    for key, value in parse_dotenv(text).items():
        if key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value

    if applied:
        log.debug("Loaded %d variable(s) from %s: %s",
                  len(applied), path, ", ".join(sorted(applied)))
    return applied
