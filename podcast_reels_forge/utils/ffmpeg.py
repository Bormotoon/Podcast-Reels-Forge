"""RU: Выбор бинарника ffmpeg, умеющего NVENC, и сборка аргументов кодека.

EN: Resolve an NVENC-capable ffmpeg binary and build video-codec arguments.

The repo can have several ffmpeg builds installed at once (e.g. a Homebrew build
with no hardware support plus a system build with NVENC). Calling a bare ``ffmpeg``
silently picks whatever is first on PATH — which here is the Homebrew build with no
NVENC, so every encode fell back to software libx264. This module finds a build that
actually exposes ``h264_nvenc`` so encodes run on the GPU.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from typing import Final

# RU: Кандидаты в порядке приоритета; FORGE_FFMPEG переопределяет всё.
# EN: Candidate binaries in priority order; FORGE_FFMPEG overrides everything.
_CANDIDATES: Final = ("/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg")


def _has_nvenc(ffmpeg: str) -> bool:
    """Return True if this ffmpeg build exposes the h264_nvenc encoder."""
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "h264_nvenc" in (out.stdout or "")


def _has_libass(ffmpeg: str) -> bool:
    """Return True if this ffmpeg build has the libass-based 'ass' subtitle filter."""
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-h", "filter=ass"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    combined = (out.stdout or "") + (out.stderr or "")
    return "Unknown filter" not in combined and "AVOptions" in combined


@functools.lru_cache(maxsize=1)
def resolve_ffmpeg() -> tuple[str, bool]:
    """RU: Возвращает (путь_к_ffmpeg, есть_nvenc), предпочитая NVENC-сборку.

    EN: Return (ffmpeg_path, has_nvenc), preferring an NVENC-capable build.
    """
    override = os.environ.get("FORGE_FFMPEG", "").strip()
    raw = ([override] if override else []) + list(_CANDIDATES)
    on_path = shutil.which("ffmpeg")
    if on_path:
        raw.append(on_path)

    usable: list[str] = []
    seen: set[str] = set()
    for cand in raw:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        resolved = cand if (os.path.isabs(cand) and os.path.exists(cand)) else shutil.which(cand)
        if resolved and resolved not in usable:
            usable.append(resolved)

    for cand in usable:
        if _has_nvenc(cand):
            return (cand, True)
    # RU: NVENC-сборки нет — берём первый рабочий ffmpeg (софтверный кодек).
    # EN: No NVENC build — fall back to the first usable ffmpeg (software codec).
    return (usable[0] if usable else "ffmpeg", False)


def ffmpeg_bin() -> str:
    """Path to the chosen ffmpeg binary."""
    return resolve_ffmpeg()[0]


def ffmpeg_has_nvenc() -> bool:
    """True if the chosen ffmpeg can encode with NVENC."""
    return resolve_ffmpeg()[1]


@functools.lru_cache(maxsize=1)
def resolve_ffmpeg_with_libass() -> str | None:
    """RU: Возвращает ffmpeg-бинарник с фильтром 'ass' (для вжигания субтитров).

    EN: Return an ffmpeg binary that supports the libass 'ass' filter.

    The NVENC-preferred build from ``resolve_ffmpeg()`` may be compiled without
    libass (as observed with the local /usr/local/bin/ffmpeg build), in which case
    the 'ass' filter silently fails to parse for both NVENC and its libx264
    fallback. This is resolved independently so the subtitle-burn pass can pick a
    working (software) build instead. Returns None if no candidate has libass.
    """
    override = os.environ.get("FORGE_FFMPEG_LIBASS", "").strip()
    nvenc_bin, _ = resolve_ffmpeg()
    raw = ([override] if override else []) + [nvenc_bin] + list(_CANDIDATES)
    on_path = shutil.which("ffmpeg")
    if on_path:
        raw.append(on_path)

    seen: set[str] = set()
    for cand in raw:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        resolved = cand if (os.path.isabs(cand) and os.path.exists(cand)) else shutil.which(cand)
        if resolved and _has_libass(resolved):
            return resolved
    return None


def build_video_codec_args(
    *,
    use_nvenc: bool,
    v_bitrate: str,
    preset: str,
    nvenc_cq: int = 21,
    nvenc_preset: str = "p5",
) -> list[str]:
    """RU: Аргументы видео-кодека. NVENC — качество через VBR+CQ, иначе libx264.

    EN: Video-codec args. NVENC uses quality-driven VBR+CQ; otherwise libx264.

    ``use_nvenc`` is treated as "prefer NVENC": it only takes effect when the
    resolved ffmpeg actually has NVENC, so a missing build degrades cleanly to
    software instead of failing the encode.
    """
    if use_nvenc and ffmpeg_has_nvenc():
        # RU: VBR с целевым качеством (cq) и потолком битрейта, чтобы не раздувать файл.
        # EN: VBR targeting a quality level (cq) with a bitrate ceiling to bound size.
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            nvenc_preset,
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(nvenc_cq),
            "-b:v",
            "0",
            "-maxrate",
            v_bitrate,
            "-bufsize",
            v_bitrate,
            "-pix_fmt",
            "yuv420p",
        ]
    return ["-c:v", "libx264", "-preset", preset, "-b:v", v_bitrate, "-pix_fmt", "yuv420p"]
