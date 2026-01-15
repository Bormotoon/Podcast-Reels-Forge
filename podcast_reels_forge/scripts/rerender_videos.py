#!/usr/bin/env python3
"""RU: Перегенерация рилсов по готовым moments.json (без LLM/транскрибации).

Скрипт читает готовые `moments.json` в `output/<model>/moments.json` и
перерендеривает клипы с жёсткими параметрами экспорта:
- 1080x1920 (9:16)
- 30 fps
- H.264 (libx264)
- AAC
- video bitrate 5000k

EN: Re-render reels from existing moments.json only.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(x, **_):
        return x


LOG = logging.getLogger("forge")


@dataclass(frozen=True)
class RenderSettings:
    width: int
    height: int
    fps: int
    v_bitrate_k: int
    a_bitrate_k: int
    padding_s: float


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, check=False)


def _load_moments(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    return []


def _vf(settings: RenderSettings) -> str:
    # Center-crop to 9:16 and force exact size + fps.
    # - force_original_aspect_ratio=increase makes sure we fill the target frame
    # - crop trims the overflow
    # - setsar=1 avoids odd pixel aspect ratios
    return (
        f"scale={settings.width}:{settings.height}:force_original_aspect_ratio=increase,"
        f"crop={settings.width}:{settings.height},"
        f"fps={settings.fps},"
        "format=yuv420p,setsar=1"
    )


def _cut_one(
    *,
    video_in: Path,
    out_path: Path,
    start: float,
    end: float,
    settings: RenderSettings,
) -> tuple[bool, str]:
    start_offset = max(0.0, float(start) - float(settings.padding_s))
    end_offset = float(end) + float(settings.padding_s)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start_offset),
        "-to",
        str(end_offset),
        "-i",
        str(video_in),
        "-vf",
        _vf(settings),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-b:v",
        f"{settings.v_bitrate_k}k",
        "-maxrate",
        f"{settings.v_bitrate_k}k",
        "-bufsize",
        f"{settings.v_bitrate_k * 2}k",
        "-r",
        str(settings.fps),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        f"{settings.a_bitrate_k}k",
        "-movflags",
        "+faststart",
        str(out_path),
    ]

    res = _run(cmd)
    if res.returncode != 0:
        # Keep stderr short but useful.
        err = (res.stderr or "").strip()
        return False, err[-2000:]

    return True, ""


def _find_model_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    dirs = [p for p in output_dir.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.name)
    return dirs


def _with_numeric_suffix(path: Path, n: int) -> Path:
    # Inserts suffix before extension: reel_01.mp4 -> reel_01_2.mp4
    return path.with_name(f"{path.stem}_{n}{path.suffix}")


def _make_unique_path(path: Path) -> Path:
    """Return a non-existing path by appending _N if needed."""
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = _with_numeric_suffix(path, n)
        if not candidate.exists():
            return candidate
        n += 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Re-render reels from existing output/<model>/moments.json only",
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("input/video.mp4"),
        help="Path to source video (default: input/video.mp4)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base output directory containing per-model folders (default: output)",
    )
    ap.add_argument(
        "--model",
        action="append",
        default=None,
        help="Only re-render for this model folder name (repeatable). Example: --model gemma3",
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Replace output/<model>/reels (deletes existing reels). By default writes to reels_rerendered/",
    )
    ap.add_argument("--padding", type=float, default=5.0, help="Padding seconds before/after")

    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--v-bitrate", type=int, default=5000, help="Video bitrate in kbit/s")
    ap.add_argument("--a-bitrate", type=int, default=192, help="Audio bitrate in kbit/s")

    ap.add_argument("--threads", type=int, default=2, help="Parallel ffmpeg jobs")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.input.exists():
        raise SystemExit(f"Input video not found: {args.input}")

    settings = RenderSettings(
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        v_bitrate_k=int(args.v_bitrate),
        a_bitrate_k=int(args.a_bitrate),
        padding_s=float(args.padding),
    )

    model_dirs = _find_model_dirs(args.output_dir)
    if args.model:
        wanted = {m.strip() for m in args.model if m and m.strip()}
        model_dirs = [d for d in model_dirs if d.name in wanted]

    if not model_dirs:
        LOG.warning("No model dirs found under %s", args.output_dir)
        return

    for model_dir in model_dirs:
        moments_path = model_dir / "moments.json"
        if not moments_path.exists():
            LOG.info("skip %s (no moments.json)", model_dir.name)
            continue

        moments = _load_moments(moments_path)
        if not moments:
            LOG.info("skip %s (moments.json empty)", model_dir.name)
            continue

        reels_dir = model_dir / ("reels" if args.replace else "reels_rerendered")
        if args.replace and reels_dir.exists():
            for p in reels_dir.glob("reel_*.mp4"):
                p.unlink(missing_ok=True)
        reels_dir.mkdir(parents=True, exist_ok=True)

        LOG.info("%s: re-render %d reels -> %s", model_dir.name, len(moments), reels_dir)

        def work(item: tuple[int, dict[str, object]]):
            i, m = item
            out_path = reels_dir / f"reel_{i + 1:02d}.mp4"
            if not args.replace:
                out_path = _make_unique_path(out_path)
            try:
                start = float(m.get("start", 0) or 0)
                end = float(m.get("end", 0) or 0)
            except (TypeError, ValueError):
                return False, out_path, "bad start/end"
            if not (0 <= start < end):
                return False, out_path, "invalid time range"
            ok, err = _cut_one(
                video_in=args.input,
                out_path=out_path,
                start=start,
                end=end,
                settings=settings,
            )
            return ok, out_path, err

        with ThreadPoolExecutor(max_workers=max(1, int(args.threads))) as pool:
            it = pool.map(work, list(enumerate(moments)))
            if args.verbose:
                it = tqdm(list(it), total=len(moments))
            results = list(it)

        ok_count = 0
        for ok, out_path, err in results:
            if ok:
                ok_count += 1
            else:
                LOG.warning("%s: failed %s (%s)", model_dir.name, out_path.name, err)

        LOG.info("%s: done (%d/%d)", model_dir.name, ok_count, len(moments))


if __name__ == "__main__":
    main()
