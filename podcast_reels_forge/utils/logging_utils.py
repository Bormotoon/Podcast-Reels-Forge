"""RU: Утилиты настройки логирования.

EN: Logging setup utilities.
"""

from __future__ import annotations

import logging
from typing import Final

try:
    import coloredlogs
except ImportError:  # pragma: no cover - optional dependency
    coloredlogs = None

DEFAULT_LOGGER_NAME: Final = "forge"


def setup_logging(*, verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """RU: Настраивает логирование c учётом флагов.

    EN: Configure logging according to verbosity flags.
    """
    level = logging.INFO
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG

    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    logger.setLevel(level)

    # RU: Избегаем двойных handlers при повторном вызове.
    # EN: Avoid double handlers if called multiple times.
    if logger.handlers:
        return logger

    fmt = "%(asctime)s %(levelname)s %(message)s"

    if coloredlogs is not None:
        coloredlogs.install(level=level, logger=logger, fmt=fmt)
    else:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    return logger
