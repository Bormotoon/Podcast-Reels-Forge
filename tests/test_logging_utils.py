"""Tests for logging utilities."""

from __future__ import annotations

import logging
from podcast_reels_forge.utils.logging_utils import setup_logging, DEFAULT_LOGGER_NAME

def test_setup_logging_defaults() -> None:
    logger = setup_logging()
    assert logger.name == DEFAULT_LOGGER_NAME
    assert logger.level == logging.INFO

def test_setup_logging_verbose() -> None:
    # Reset logger to ensure clean state
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        
    logger = setup_logging(verbose=True)
    assert logger.level == logging.DEBUG

def test_setup_logging_quiet() -> None:
    # Reset logger
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logger = setup_logging(quiet=True)
    assert logger.level == logging.ERROR

def test_setup_logging_singleton() -> None:
    logger1 = setup_logging()
    logger2 = setup_logging()
    assert logger1 is logger2
    # Second call shouldn't add more handlers
    assert len(logger1.handlers) == len(logger2.handlers)
