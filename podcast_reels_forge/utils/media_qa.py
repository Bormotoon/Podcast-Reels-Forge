"""RU: Проверка готового ролика перед тем, как считать его готовым.

Без человека в контуре битый ролик (обрезанный, без звука, чёрный) уходит
дальше как обычный. ffprobe проверяет дорожки и длительность — это дёшево;
blackdetect декодирует весь клип и поэтому включается отдельно.

EN: Check a rendered clip before calling it done.

With nobody in the loop a broken clip (truncated, silent, black) ships like
any other. ffprobe checks the streams and duration, which is cheap;
blackdetect decodes the whole clip and is therefore opt-in.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from podcast_reels_forge.utils.ffmpeg import ffmpeg_bin

LOG = logging.getLogger(__name__)

_BLACK_DURATION_RE = re.compile(r"black_duration:\s*(\d+(?:\.\d+)?)")


@functools.lru_cache(maxsize=1)
def ffprobe_bin() -> str | None:
    """ffprobe next to the chosen ffmpeg, else the one on PATH."""

    ffmpeg = ffmpeg_bin()
    if os.path.isabs(ffmpeg):
        sibling = Path(ffmpeg).with_name("ffprobe")
        if sibling.exists():
            return str(sibling)
    return shutil.which("ffprobe")


def probe(path: Path) -> dict[str, Any] | None:
    binary = ffprobe_bin()
    if binary is None:
        return None
    try:
        res = subprocess.run(
            [binary, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return {}
    try:
        data = json.loads(res.stdout or "{}")
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def media_duration(path: Path) -> float | None:
    data = probe(path)
    try:
        return float((data or {}).get("format", {}).get("duration"))
    except (TypeError, ValueError):
        return None


def black_seconds(path: Path) -> float | None:
    try:
        res = subprocess.run(
            [ffmpeg_bin(), "-hide_banner", "-nostats", "-i", str(path),
             "-vf", "blackdetect=d=0.5:pix_th=0.10", "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=300, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return sum(float(v) for v in _BLACK_DURATION_RE.findall(res.stderr or ""))


def check_clip(
    path: Path,
    *,
    expected_duration: float,
    blackdetect: bool = False,
) -> list[str] | None:
    """Problems with a rendered clip; [] when it is fine, None when unchecked."""

    if ffprobe_bin() is None:
        return None
    try:
        if path.stat().st_size <= 0:
            return ["qa: empty file"]
    except OSError:
        return ["qa: file missing"]
    data = probe(path)
    if data is None:
        return None
    if not data:
        return ["qa: ffprobe cannot read the file"]

    problems: list[str] = []
    kinds = {str(stream.get("codec_type")) for stream in data.get("streams", []) if isinstance(stream, dict)}
    if "video" not in kinds:
        problems.append("qa: no video stream")
    if "audio" not in kinds:
        problems.append("qa: no audio stream")
    try:
        duration = float(data.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = 0.0
    tolerance = max(1.5, 0.05 * expected_duration)
    if expected_duration > 0 and abs(duration - expected_duration) > tolerance:
        problems.append(f"qa: duration {duration:.1f}s, expected {expected_duration:.1f}s")
    if blackdetect and duration > 0:
        black = black_seconds(path)
        if black is not None and black > 0.5 * duration:
            problems.append(f"qa: {black:.1f}s of {duration:.1f}s is black")
    return problems
