"""RU: Общие вспомогательные сущности для стадии обработки видео.

Корневой скрипт `video_processor.py` остаётся CLI-точкой входа и содержит
основную логику вызова FFmpeg. Этот модуль — для небольших утилит,
используемых в тестах и будущих рефакторах.

EN: Shared helpers for the video processing stage.

The root script `video_processor.py` remains the CLI entrypoint and contains the
FFmpeg invocation logic. This module provides small utilities used for testing
and future refactors.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoExportConfig:
    """RU: Флаги экспорта (webm/gif/audio-only).

    EN: Export flags (webm/gif/audio-only).
    """

    webm: bool = False
    gif: bool = False
    audio_only: bool = False
