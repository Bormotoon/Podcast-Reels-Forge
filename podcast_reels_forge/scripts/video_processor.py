#!/usr/bin/env python3
"""RU: Обработка видео через FFmpeg: нарезка рилсов, вертикальный кроп и превью.

EN: Process video with FFmpeg: cut reels, apply vertical crop, and concat samples.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from podcast_reels_forge.utils.face_crop import (
    FaceCropSettings,
    build_sample_times,
    compute_crop_x_for_scaled_height,
    detect_face_center_ratio,
    face_detection_available,
)

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class FfmpegOptions:
    """Container for FFmpeg tuning options."""

    vertical_crop: bool
    smart_crop_face: bool
    use_nvenc: bool
    v_bitrate: str
    a_bitrate: str
    preset: str
    padding: float
    face_samples: int
    face_min_size: int


def _run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command with safe defaults."""

    normalized_cmd = [str(part) for part in cmd]
    return subprocess.run(normalized_cmd, capture_output=True, text=True, check=False)


def _status(msg: str, *, quiet: bool) -> None:
    if not quiet:
        LOG.info(msg)


def ffmpeg_cut(
    video_in: Path, start: float, end: float, out_path: Path, opts: FfmpegOptions,
) -> bool:
    """Cut a segment from video with optional vertical crop."""

    filters: list[str] = []
    if opts.vertical_crop:
        # Default: center crop to 9:16.
        vf = "scale=w=1080:h=1920:force_original_aspect_ratio=increase,crop=1080:1920"

        # Optional: smart crop around face.
        if opts.smart_crop_face and face_detection_available():
            face_settings = FaceCropSettings(
                samples=int(opts.face_samples),
                min_face_size=int(opts.face_min_size),
            )
            start_offset = max(0, start - opts.padding)
            end_offset = end + opts.padding
            sample_times = build_sample_times(start_offset, end_offset, face_settings.samples)
            center_ratio = detect_face_center_ratio(video_in, sample_times_s=sample_times, settings=face_settings)
            if center_ratio is not None:
                LOG.debug("Face detected at ratio %.2f; applying smart crop", center_ratio)
                try:
                    import cv2  # type: ignore

                    cap = cv2.VideoCapture(str(video_in))
                    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    cap.release()
                except Exception:
                    src_w, src_h = 0, 0

                crop_x = compute_crop_x_for_scaled_height(
                    src_w=src_w,
                    src_h=src_h,
                    target_w=1080,
                    target_h=1920,
                    center_ratio=center_ratio,
                )
                vf = f"scale=-2:1920,crop=1080:1920:{crop_x}:0"

        filters.append(vf)

    start_offset = max(0, start - opts.padding)
    end_offset = end + opts.padding

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_offset),
        "-to",
        str(end_offset),
        "-i",
        str(video_in),
    ]
    if filters:
        cmd += ["-vf", ",".join(filters)]

    vcodec = "h264_nvenc" if opts.use_nvenc else "libx264"
    cmd += [
        "-c:v",
        vcodec,
        "-preset",
        opts.preset,
        "-b:v",
        opts.v_bitrate,
        "-c:a",
        "aac",
        "-b:a",
        opts.a_bitrate,
        str(out_path),
    ]

    res = _run_subprocess(cmd)
    if res.returncode != 0 and opts.use_nvenc:
        # Fallback to software x264 if NVENC is unavailable.
        cmd[cmd.index("h264_nvenc")] = "libx264"
        res = _run_subprocess(cmd)

    return res.returncode == 0


def create_concat_sample(reels: list[Path], out_path: Path) -> bool:
    """Concatenate multiple video files into one preview file."""

    if not reels:
        return False
    list_path = out_path.with_suffix(out_path.suffix + ".txt")
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with list_path.open("w", encoding="utf-8") as f:
        for reel in reels:
            f.write(f"file '{reel.resolve()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(out_path),
    ]
    res = _run_subprocess(cmd)
    if list_path.exists():
        list_path.unlink(missing_ok=True)
    return res.returncode == 0


def _export_webm(mp4_path: Path, out_path: Path) -> bool:
    """Export video as WebM format."""

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4_path),
        "-c:v",
        "libvpx-vp9",
        "-b:v",
        "0",
        "-crf",
        "32",
        "-c:a",
        "libopus",
        str(out_path),
    ]
    return _run_subprocess(cmd).returncode == 0


def _export_audio(mp4_path: Path, out_path: Path) -> bool:
    """Export audio-only track from video."""

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4_path),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(out_path),
    ]
    return _run_subprocess(cmd).returncode == 0


def _export_gif(mp4_path: Path, out_path: Path) -> bool:
    """Export video as animated GIF with palette optimization."""

    palette = out_path.with_suffix(out_path.suffix + ".palette.png")
    vf = "fps=12,scale=480:-1:flags=lanczos"
    cmd1 = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4_path),
        "-vf",
        f"{vf},palettegen",
        str(palette),
    ]
    cmd2 = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4_path),
        "-i",
        str(palette),
        "-lavfi",
        f"{vf}[x];[x][1:v]paletteuse",
        str(out_path),
    ]
    ok = _run_subprocess(cmd1).returncode == 0 and _run_subprocess(cmd2).returncode == 0
    try:
        if palette.exists():
            palette.unlink()
    except OSError:
        pass
    return ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    ap = argparse.ArgumentParser(description="Cut video reels based on moments.json")
    ap.add_argument("--input", type=Path, required=True, help="Input video file")
    ap.add_argument("--moments", type=Path, required=True, help="Path to moments.json")
    ap.add_argument("--outdir", type=Path, default=Path("out"), help="Output directory")
    ap.add_argument(
        "--threads", type=int, default=4, help="Number of parallel FFmpeg threads",
    )
    ap.add_argument(
        "--vertical", action="store_true", default=False, help="Crop to 9:16 format",
    )
    ap.add_argument(
        "--smart-crop-face",
        action="store_true",
        default=False,
        help="When used with --vertical, center crop around detected face (requires opencv)",
    )
    ap.add_argument("--face-samples", type=int, default=7, help="Frames to sample per reel")
    ap.add_argument("--face-min-size", type=int, default=60, help="Min face size in pixels")
    ap.add_argument("--v-bitrate", default="5M", help="Video bitrate")
    ap.add_argument("--a-bitrate", default="192k", help="Audio bitrate")
    ap.add_argument("--preset", default="fast", help="FFmpeg preset")
    ap.add_argument(
        "--nvenc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NVENC if available (default: enabled)",
    )
    ap.add_argument("--padding", type=float, default=0, help="Extra seconds around moment")
    ap.add_argument("--export-webm", action="store_true", help="Export reels as .webm")
    ap.add_argument("--export-gif", action="store_true", help="Export reels as .gif")
    ap.add_argument(
        "--export-audio", action="store_true", help="Export reels audio-only as .m4a",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress non-error output")
    ap.add_argument("--verbose", action="store_true", help="Verbose output (incl. progress)")
    return ap.parse_args(argv)


def _load_moments(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    return []


def main(argv: list[str] | None = None) -> None:
    """Main entry point for video processor."""

    args = parse_args(argv)

    if not args.moments.exists():
        LOG.error("moments file not found: %s", args.moments)
        sys.exit(1)

    moments = _load_moments(args.moments)

    args.outdir.mkdir(parents=True, exist_ok=True)
    reels_dir = args.outdir / "reels"
    reels_dir.mkdir(parents=True, exist_ok=True)

    opts = FfmpegOptions(
        vertical_crop=args.vertical,
        smart_crop_face=bool(args.smart_crop_face),
        use_nvenc=bool(args.nvenc),
        v_bitrate=args.v_bitrate,
        a_bitrate=args.a_bitrate,
        preset=args.preset,
        padding=args.padding,
        face_samples=int(args.face_samples),
        face_min_size=int(args.face_min_size),
    )

    if opts.smart_crop_face and opts.vertical_crop and not face_detection_available():
        LOG.warning("--smart-crop-face enabled but opencv is not available; falling back to center crop")

    def process_moment(i_m: tuple[int, dict[str, object]]) -> Path | None:
        i, m = i_m
        out_file = reels_dir / f"reel_{i + 1:02d}.mp4"
        start_val = m.get("start", 0)
        end_val = m.get("end", 0)
        # Type-safe conversion to float
        try:
            start_f = float(start_val) if start_val is not None else 0.0  # type: ignore[arg-type]
            end_f = float(end_val) if end_val is not None else 0.0  # type: ignore[arg-type]
        except (TypeError, ValueError):
            start_f, end_f = 0.0, 0.0
        success = ffmpeg_cut(
            args.input,
            start_f,
            end_f,
            out_file,
            opts,
        )
        return out_file if success else None

    _status(f"[cut] {len(moments)} moments", quiet=args.quiet)
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        it: Iterable[Path | None] = pool.map(process_moment, enumerate(moments))
        if args.verbose:
            it = tqdm(it, total=len(moments))
        results = list(it)

    final_reels = [r for r in results if r is not None]
    if final_reels:
        sample_path = args.outdir / "reels_preview.mp4"
        if create_concat_sample(final_reels, sample_path) and not args.quiet:
            LOG.info("preview ready: %s", sample_path)

        for mp4 in final_reels:
            stem = mp4.with_suffix("")
            if args.export_webm:
                _export_webm(mp4, stem.with_suffix(".webm"))
            if args.export_audio:
                _export_audio(mp4, stem.with_suffix(".m4a"))
            if args.export_gif:
                _export_gif(mp4, stem.with_suffix(".gif"))

    _status(f"[cut] done ({len(final_reels)} reels)", quiet=args.quiet)


if __name__ == "__main__":
    main()
