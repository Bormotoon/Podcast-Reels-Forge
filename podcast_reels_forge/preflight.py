"""RU: Предполётная проверка окружения перед прогоном.

Без неё отсутствие llama-server, модели или токена обнаруживалось посреди
ночи — на первом же эпизоде, после часа транскрибации, а потом повторялось на
каждом следующем (с ожиданием сервера по 300 с). Здесь всё проверяется один
раз и сразу: ошибка — прогон не начинается, предупреждение — начинается, но
в отчёте видно, что чего-то не хватает.

EN: Pre-flight check of the environment before a run.

Without it a missing llama-server, model or token surfaced in the middle of
the night — on the first episode, after an hour of transcription — and then
again on every following one (with a 300 s server wait each). Everything is
checked once, up front: an error stops the run before it starts, a warning
lets it start but shows in the report.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from podcast_reels_forge.utils.ffmpeg import ffmpeg_bin
from podcast_reels_forge.utils.llama_cpp_service import (
    is_tcp_open,
    parse_local_llama_cpp_host_port,
)

LLM_STAGES = frozenset({"proofread", "article", "analyze"})
DEFAULT_MIN_FREE_DISK_GB = 5.0


@dataclass
class PreflightResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _section(conf: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = conf.get(key)
    return value if isinstance(value, Mapping) else {}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _executable(name_or_path: str) -> bool:
    if os.path.isabs(name_or_path):
        return os.path.isfile(name_or_path) and os.access(name_or_path, os.X_OK)
    return shutil.which(name_or_path) is not None


def _free_gb(path: Path) -> float | None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / 1024**3
    except OSError:
        return None


def run_preflight(
    conf: Mapping[str, Any],
    *,
    stages: Collection[str],
    repo_dir: Path,
    youtube_requested: bool = False,
) -> PreflightResult:
    """Check what the selected stages need before any work starts."""

    result = PreflightResult()
    active = set(stages)

    if not _executable(ffmpeg_bin()):
        result.errors.append("ffmpeg не найден (PATH или FORGE_FFMPEG)")

    if "transcribe" in active and not _module_available("faster_whisper"):
        result.errors.append("faster-whisper не установлен: pip install -r requirements.txt")

    diar = _section(conf, "diarization")
    if "diarize" in active and diar.get("enabled"):
        if not os.environ.get("PYANNOTE_TOKEN"):
            result.errors.append("диаризация включена, но PYANNOTE_TOKEN не задан (.env)")
        if not _module_available("pyannote.audio"):
            result.errors.append("диаризация включена, но pyannote.audio не установлен")

    if "fetch" in active and youtube_requested and not _module_available("yt_dlp"):
        result.errors.append("для загрузки с YouTube нужен yt-dlp: pip install -U yt-dlp")

    llm_needed = "analyze" in active or any(
        stage in active and _section(conf, stage).get("enabled") for stage in ("proofread", "article")
    )
    llama = _section(conf, "llama_cpp")
    local = parse_local_llama_cpp_host_port(str(llama.get("url", "")).strip())
    if llm_needed and local is not None and not is_tcp_open(*local):
        service = _section(llama, "service")
        if not service.get("auto_start", True):
            result.errors.append(
                f"llama-server не отвечает на {local[0]}:{local[1]}, а auto_start выключен",
            )
        else:
            if not _executable("llama-server"):
                result.errors.append("llama-server не найден в PATH (нужен для auto_start)")
            model_path = str(service.get("model_path") or "").strip()
            if not model_path:
                result.errors.append("llama_cpp.service.model_path не задан")
            elif not Path(model_path).exists():
                result.errors.append(f"файл модели llama.cpp не найден: {model_path}")

    autonomy = _section(conf, "autonomy")
    try:
        min_free = float(autonomy.get("min_free_disk_gb", DEFAULT_MIN_FREE_DISK_GB))
    except (TypeError, ValueError):
        min_free = DEFAULT_MIN_FREE_DISK_GB
    output_dir = Path(str(_section(conf, "paths").get("output_dir", "output")))
    if not output_dir.is_absolute():
        output_dir = repo_dir / output_dir
    free = _free_gb(output_dir)
    if free is not None and min_free > 0 and free < min_free:
        result.errors.append(
            f"на диске с {output_dir} свободно {free:.1f} ГБ < {min_free:g} ГБ (autonomy.min_free_disk_gb)",
        )

    video = _section(conf, "video")
    if "cut" in active and video.get("smart_crop_face", True):
        from podcast_reels_forge.utils.face_crop import face_detection_available

        if not face_detection_available():
            result.warnings.append(
                "умный кроп по лицу недоступен (нет opencv/mediapipe или модели) — будет центральный кроп",
            )

    return result
