#!/usr/bin/env python3
"""RU: Обработка видео через FFmpeg: нарезка рилсов, вертикальный кроп и превью.

EN: Process video with FFmpeg: cut reels, apply vertical crop, and concat samples.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable: Iterable, **kwargs: object) -> Iterable:
        return iterable


def _status(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(msg, flush=True)


def ffmpeg_cut(
    video_in: str,
    start: float,
    end: float,
    out_path: str,
    vertical_crop: bool = True,
    v_bitrate: str = "5M",
    a_bitrate: str = "192k",
    preset: str = "fast",
    padding: float = 0,
) -> bool:
    """RU: Нарезает сегмент видео (опционально с вертикальным кропом).

    Аргументы:
        video_in: Путь к входному видео.
        start: Время начала (секунды).
        end: Время конца (секунды).
        out_path: Путь к выходному файлу.
        vertical_crop: Кроп до 9:16.
        v_bitrate: Битрейт видео.
        a_bitrate: Битрейт аудио.
        preset: Пресет энкодера.
        padding: Доп. секунды вокруг момента.

    Возвращает:
        True при успехе, иначе False.

    EN: Cut a segment from video with optional vertical crop.

    Args:
        video_in: Input video path.
        start: Start time in seconds.
        end: End time in seconds.
        out_path: Output file path.
        vertical_crop: Whether to crop to 9:16 format.
        v_bitrate: Video bitrate.
        a_bitrate: Audio bitrate.
        preset: Encoding preset.
        padding: Extra seconds around the moment.

    Returns:
        True if successful, False otherwise.

    """
    filters = []
    if vertical_crop:
        # RU: Масштабируем до не меньше 1080x1920, затем кроп по центру.
        # EN: Scale to at least 1080x1920 then crop center.
        filters.append(
            "scale=w=ih*(9/16):h=ih,scale=w=1080:h=1920:force_original_aspect_ratio=increase,crop=1080:1920",
        )

    # RU: Применяем padding.
    # EN: Apply padding.
    start = max(0, start - padding)
    end = end + padding

    cmd = ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", video_in]
    if filters:
        cmd += ["-vf", ",".join(filters)]

    # RU: Пытаемся использовать NVENC, иначе fallback на libx264.
    # EN: Use NVENC if available, else libx264.
    cmd += [
        "-c:v",
        "h264_nvenc",
        "-preset",
        preset,
        "-b:v",
        v_bitrate,
        "-c:a",
        "aac",
        "-b:a",
        a_bitrate,
        out_path,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # RU: Fallback на libx264.
        # EN: Fallback to libx264.
        cmd[cmd.index("h264_nvenc")] = "libx264"
        res = subprocess.run(cmd, capture_output=True, text=True)

    return res.returncode == 0


def create_concat_sample(reels: list[str], out_path: str) -> bool:
    """RU: Склеивает несколько видеофайлов в один превью-ролик.

    EN: Concatenate multiple video files into one preview file.
    """
    if not reels:
        return False
    list_path = out_path + ".txt"
    with open(list_path, "w") as f:
        f.writelines(f"file '{os.path.abspath(r)}'\n" for r in reels)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        out_path,
    ]
    res = subprocess.run(cmd, capture_output=True)
    if os.path.exists(list_path):
        os.remove(list_path)
    return res.returncode == 0


def _export_webm(mp4_path: str, out_path: str) -> bool:
    """RU: Экспортирует видео в WebM.

    EN: Export video as WebM format.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        mp4_path,
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "0",
        "-crf",
        "32",
        "-c:a",
        "libopus",
        out_path,
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def _export_audio(mp4_path: str, out_path: str) -> bool:
    """RU: Экспортирует только аудиодорожку из видео.

    EN: Export audio-only track from video.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        mp4_path,
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        out_path,
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def _export_gif(mp4_path: str, out_path: str) -> bool:
    """RU: Экспортирует видео в GIF с оптимизацией палитры.

    EN: Export video as animated GIF with palette optimization.
    """
    palette = out_path + ".palette.png"
    vf = "fps=12,scale=480:-1:flags=lanczos"
    cmd1 = ["ffmpeg", "-y", "-i", mp4_path, "-vf", f"{vf},palettegen", palette]
    cmd2 = [
        "ffmpeg",
        "-y",
        "-i",
        mp4_path,
        "-i",
        palette,
        "-lavfi",
        f"{vf}[x];[x][1:v]paletteuse",
        out_path,
    ]
    ok = (
        subprocess.run(cmd1, capture_output=True).returncode == 0
        and subprocess.run(cmd2, capture_output=True).returncode == 0
    )
    try:
        if os.path.exists(palette):
            os.remove(palette)
    except OSError:
        pass
    return ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """RU: Парсит аргументы командной строки.

    EN: Parse command line arguments.
    """
    ap = argparse.ArgumentParser(description="Cut video reels based on moments.json")
    ap.add_argument("--input", required=True, help="Input video file")
    ap.add_argument("--moments", required=True, help="Path to moments.json")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument(
        "--threads", type=int, default=4, help="Number of parallel FFmpeg threads",
    )
    ap.add_argument(
        "--vertical", action="store_true", default=False, help="Crop to 9:16 format",
    )
    ap.add_argument("--v-bitrate", default="5M", help="Video bitrate")
    ap.add_argument("--a-bitrate", default="192k", help="Audio bitrate")
    ap.add_argument("--preset", default="fast", help="FFmpeg preset")
    ap.add_argument(
        "--padding", type=float, default=0, help="Extra seconds around moment",
    )
    ap.add_argument("--export-webm", action="store_true", help="Export reels as .webm")
    ap.add_argument("--export-gif", action="store_true", help="Export reels as .gif")
    ap.add_argument(
        "--export-audio", action="store_true", help="Export reels audio-only as .m4a",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress non-error output")
    ap.add_argument(
        "--verbose", action="store_true", help="Verbose output (incl. progress)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """RU: Точка входа для стадии обработки видео.

    EN: Main entry point for video processor.
    """
    args = parse_args(argv)

    if not os.path.exists(args.moments):
        print(f"Error: {args.moments} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.moments, encoding="utf-8") as f:
        moments = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)
    reels_dir = os.path.join(args.outdir, "reels")
    os.makedirs(reels_dir, exist_ok=True)

    def process_moment(i_m: tuple[int, dict]) -> str | None:
        i, m = i_m
        out_file = os.path.join(reels_dir, f"reel_{i + 1:02d}.mp4")
        success = ffmpeg_cut(
            args.input,
            m["start"],
            m["end"],
            out_file,
            vertical_crop=args.vertical,
            v_bitrate=args.v_bitrate,
            a_bitrate=args.a_bitrate,
            preset=args.preset,
            padding=args.padding,
        )
        return out_file if success else None

    _status(f"[cut] {len(moments)} moments", quiet=args.quiet)
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        it: Iterable[str | None] = pool.map(process_moment, enumerate(moments))
        if args.verbose:
            it = tqdm(it, total=len(moments))
        results = list(it)

    final_reels = [r for r in results if r]
    if final_reels:
        sample_path = os.path.join(args.outdir, "reels_preview.mp4")
        if create_concat_sample(final_reels, sample_path):
            if not args.quiet:
                print(sample_path)

        for mp4 in final_reels:
            stem, _ = os.path.splitext(mp4)
            if args.export_webm:
                _export_webm(mp4, stem + ".webm")
            if args.export_audio:
                _export_audio(mp4, stem + ".m4a")
            if args.export_gif:
                _export_gif(mp4, stem + ".gif")

    _status(f"[cut] done ({len(final_reels)} reels)", quiet=args.quiet)


if __name__ == "__main__":
    main()
