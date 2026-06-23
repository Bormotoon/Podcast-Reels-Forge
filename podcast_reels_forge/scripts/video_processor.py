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
from dataclasses import dataclass, replace
from pathlib import Path

from podcast_reels_forge.utils.burned_subtitles import (
    DEFAULT_SUBTITLE_FONT,
    SubtitleRenderSettings,
    load_transcript_segments,
    slice_segments_for_clip,
    write_srt_file,
    _prepare_subtitle_segments,
    _write_ass_file,
    _coerce_float,
)
from podcast_reels_forge.utils.face_crop import (
    FaceCropSettings,
    build_sample_times,
    compute_crop_x_for_scaled_height,
    detect_face_center_ratio,
    face_detection_available,
)
from podcast_reels_forge.utils.ffmpeg import (
    build_video_codec_args,
    ffmpeg_bin,
    ffmpeg_has_nvenc,
)
from podcast_reels_forge.utils.reel_markdown import write_reel_instagram_txt, write_reel_markdown

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
    filter_face_ratio: float = 0.0
    # NVENC quality knobs: cq is the VBR quality target (lower = better), preset is p1..p7.
    nvenc_cq: int = 21
    nvenc_preset: str = "p5"


def _run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command with safe defaults."""

    normalized_cmd = [str(part) for part in cmd]
    return subprocess.run(normalized_cmd, capture_output=True, text=True, check=False)


def _status(msg: str, *, quiet: bool) -> None:
    if not quiet:
        LOG.info(msg)


def ffmpeg_cut(
    video_in: Path,
    start: float,
    end: float,
    out_path: Path,
    opts: FfmpegOptions,
    is_rejected: bool = False,
    rejected_dir: Path | None = None,
    ass_path: Path | None = None,
) -> tuple[bool, Path, str | None]:
    """Cut a segment from video with optional vertical crop."""

    filters: list[str] = []
    face_rejection_reason: str | None = None
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
            center_ratio, face_rate = detect_face_center_ratio(video_in, sample_times_s=sample_times, settings=face_settings)

            if opts.filter_face_ratio > 0 and face_rate < opts.filter_face_ratio:
                LOG.debug("Rejecting clip (face detection rate %.2f < %.2f)", face_rate, opts.filter_face_ratio)
                is_rejected = True
                face_rejection_reason = (
                    f"face ratio {face_rate:.2f} < {opts.filter_face_ratio:.2f}"
                )

            if center_ratio is not None:
                LOG.debug("Face detected at ratio %.2f; applying smart crop", center_ratio)
                try:
                    import cv2

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

    if ass_path and ass_path.exists():
        # Escape path for FFmpeg filter
        safe_ass_path = str(ass_path.resolve()).replace('\\', '/').replace(':', '\\:')
        filters.append(f"ass='{safe_ass_path}'")

    if is_rejected and rejected_dir:
        out_path = rejected_dir / out_path.name

    start_offset = max(0, start - opts.padding)
    end_offset = end + opts.padding

    def _build(use_nvenc: bool) -> list[str]:
        cmd = [
            ffmpeg_bin(),
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
        cmd += build_video_codec_args(
            use_nvenc=use_nvenc,
            v_bitrate=opts.v_bitrate,
            preset=opts.preset,
            nvenc_cq=opts.nvenc_cq,
            nvenc_preset=opts.nvenc_preset,
        )
        # +faststart moves the moov atom to the front for instant playback/upload.
        cmd += ["-c:a", "aac", "-b:a", opts.a_bitrate, "-movflags", "+faststart", str(out_path)]
        return cmd

    res = _run_subprocess(_build(opts.use_nvenc))
    if res.returncode != 0 and opts.use_nvenc and ffmpeg_has_nvenc():
        # NVENC was attempted but failed; rebuild with software libx264.
        LOG.warning("NVENC encode failed for %s; retrying with software libx264", out_path.name)
        res = _run_subprocess(_build(False))

    return res.returncode == 0, out_path, face_rejection_reason


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
        ffmpeg_bin(),
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
        ffmpeg_bin(),
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
        ffmpeg_bin(),
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
        ffmpeg_bin(),
        "-y",
        "-i",
        str(mp4_path),
        "-vf",
        f"{vf},palettegen",
        str(palette),
    ]
    cmd2 = [
        ffmpeg_bin(),
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
    ap.add_argument("--preset", default="fast", help="libx264 preset (software fallback)")
    ap.add_argument(
        "--nvenc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NVENC if available (default: enabled)",
    )
    ap.add_argument("--nvenc-cq", type=int, default=21, help="NVENC VBR quality target, lower=better (default: 21)")
    ap.add_argument("--nvenc-preset", default="p5", help="NVENC preset p1(fast)..p7(quality) (default: p5)")
    ap.add_argument("--padding", type=float, default=0, help="Extra seconds around moment")
    ap.add_argument("--export-webm", action="store_true", help="Export reels as .webm")
    ap.add_argument("--export-gif", action="store_true", help="Export reels as .gif")
    ap.add_argument(
        "--export-audio", action="store_true", help="Export reels audio-only as .m4a",
    )
    ap.add_argument(
        "--burn-subtitles",
        action="store_true",
        help="Burn subtitles into each rendered reel using .ass files",
    )
    ap.add_argument(
        "--transcript-json",
        type=Path,
        help="Path to the full transcript JSON used to derive reel-local subtitles",
    )
    ap.add_argument(
        "--subtitle-font",
        type=Path,
        default=DEFAULT_SUBTITLE_FONT,
        help="Font file for burned subtitles (default: assets/fonts/bignoodletoooblique.ttf)",
    )
    ap.add_argument(
        "--subtitle-wrap-words",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow subtitles to wrap onto multiple lines at spaces (default: enabled)",
    )
    
    # Quality Filters
    ap.add_argument("--filter-min-score", type=float, default=0.0, help="Reject if LLM score is below this")
    ap.add_argument("--filter-min-duration", type=float, default=0.0, help="Reject if duration is below this")
    ap.add_argument("--filter-max-duration", type=float, default=9999.0, help="Reject if duration is above this")
    ap.add_argument("--filter-face-ratio", type=float, default=0.0, help="Reject if face detected ratio is below this")

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

    if args.burn_subtitles and args.transcript_json is None:
        LOG.error("--burn-subtitles requires --transcript-json")
        sys.exit(1)

    subtitle_settings = None
    if args.burn_subtitles:
        subtitle_settings = SubtitleRenderSettings(
            enabled=True,
            font_path=args.subtitle_font.resolve(),
            wrap_words=bool(args.subtitle_wrap_words),
        )
        if not subtitle_settings.font_path.exists():
            subtitle_settings = replace(
                subtitle_settings,
                font_path=(Path.cwd() / DEFAULT_SUBTITLE_FONT).resolve(),
            )

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
        filter_face_ratio=float(args.filter_face_ratio),
        nvenc_cq=int(args.nvenc_cq),
        nvenc_preset=str(args.nvenc_preset),
    )

    if opts.smart_crop_face and opts.vertical_crop and not face_detection_available():
        LOG.warning("--smart-crop-face enabled but opencv is not available; falling back to center crop")

    def process_moment(
        i_m: tuple[int, dict[str, object]],
    ) -> tuple[Path | None, list[str]]:
        """Returns (final_path_or_none, rejection_reasons)."""
        i, m = i_m
        out_file = reels_dir / f"reel_{i + 1:02d}.mp4"
        start_val = m.get("start", 0)
        end_val = m.get("end", 0)
        try:
            start_f = float(start_val) if start_val is not None else 0.0  # type: ignore[arg-type]
            end_f = float(end_val) if end_val is not None else 0.0  # type: ignore[arg-type]
        except (TypeError, ValueError):
            start_f, end_f = 0.0, 0.0

        score_val = m.get("score", 0.0)
        try:
            score = float(score_val) if score_val is not None else 0.0  # type: ignore[arg-type]
        except (TypeError, ValueError):
            score = 0.0

        duration = end_f - start_f

        is_rejected = False
        rejection_reasons: list[str] = []
        if args.filter_min_score > 0 and score < args.filter_min_score:
            is_rejected = True
            rejection_reasons.append(f"score {score:.1f} < {args.filter_min_score:.1f}")
        if args.filter_min_duration > 0 and duration < args.filter_min_duration:
            is_rejected = True
            rejection_reasons.append(f"duration {duration:.0f}s < {args.filter_min_duration:.0f}s")
        if args.filter_max_duration < 9999 and duration > args.filter_max_duration:
            is_rejected = True
            rejection_reasons.append(f"duration {duration:.0f}s > {args.filter_max_duration:.0f}s")

        rejected_dir = reels_dir / "rejected"
        if is_rejected:
            rejected_dir.mkdir(exist_ok=True)

        success, final_path, face_reason = ffmpeg_cut(
            args.input,
            start_f,
            end_f,
            out_file,
            opts,
            is_rejected=is_rejected,
            rejected_dir=rejected_dir,
        )
        if face_reason:
            rejection_reasons.append(face_reason)
        return (final_path if success else None), rejection_reasons

    _status(f"[cut] {len(moments)} moments", quiet=args.quiet)
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        raw_results: Iterable[tuple[Path | None, list[str]]] = pool.map(
            process_moment, enumerate(moments)
        )
        if args.verbose:
            raw_results = tqdm(raw_results, total=len(moments))
        results = list(raw_results)

    # results[i] = (final_path | None, rejection_reasons)
    all_cut_paths = [path for path, _ in results]
    final_reels = [p for p in all_cut_paths if p is not None and "rejected" not in p.parts]

    if any(p is not None for p in all_cut_paths):
        if subtitle_settings is not None:
            try:
                transcript_segments = load_transcript_segments(args.transcript_json)
                for reel_path in final_reels:
                    stem = reel_path.stem
                    moment_match = None
                    for idx_p, rp in enumerate(final_reels):
                        if rp == reel_path and idx_p < len(moments):
                            moment_match = moments[idx_p]
                            break
                    if moment_match is None:
                        continue

                    clip_segments = slice_segments_for_clip(
                        transcript_segments,
                        clip_start=max(0.0, _coerce_float(moment_match.get("start"), default=0.0) - opts.padding),
                        clip_end=_coerce_float(moment_match.get("end"), default=0.0) + opts.padding,
                    )
                    clip_segments = _prepare_subtitle_segments(clip_segments, settings=subtitle_settings)
                    if not clip_segments:
                        continue

                    srt_path = reel_path.with_suffix(".srt")
                    write_srt_file(srt_path, clip_segments)

                    ass_path = reel_path.with_suffix(".ass")
                    _write_ass_file(ass_path, clip_segments, subtitle_settings)

                    subtitled_path = reel_path.with_name(f"{stem}.subtitled.mp4")
                    success, _, _ = ffmpeg_cut(
                        args.input,
                        _coerce_float(moment_match.get("start"), default=0.0),
                        _coerce_float(moment_match.get("end"), default=0.0),
                        subtitled_path,
                        opts,
                        ass_path=ass_path,
                    )
                    if success and subtitled_path.exists():
                        reel_path.unlink(missing_ok=True)
                        subtitled_path.rename(reel_path)
            except Exception as exc:
                LOG.error("Failed to burn subtitles: %s", exc)
                sys.exit(1)

        if final_reels:
            sample_path = args.outdir / "reels_preview.mp4"
            if create_concat_sample(final_reels, sample_path) and not args.quiet:
                LOG.info("preview ready: %s", sample_path)

        for mp4 in final_reels:
            stem_path = mp4.with_suffix("")
            if args.export_webm:
                _export_webm(mp4, stem_path.with_suffix(".webm"))
            if args.export_audio:
                _export_audio(mp4, stem_path.with_suffix(".m4a"))
            if args.export_gif:
                _export_gif(mp4, stem_path.with_suffix(".gif"))

        # Write per-clip .txt (Instagram caption) and .md for every clip, including rejected.
        for i, (maybe_clip_path, rejection_reasons) in enumerate(results):
            if maybe_clip_path is None or i >= len(moments):
                continue
            clip_path: Path = maybe_clip_path
            try:
                write_reel_instagram_txt(
                    moments[i],
                    clip_path,
                    rejection_reasons=rejection_reasons or None,
                )
            except OSError as exc:
                LOG.warning("Failed to write instagram txt for %s: %s", clip_path.name, exc)
            try:
                write_reel_markdown(moments[i], clip_path)
            except OSError as exc:
                LOG.warning("Failed to write reel markdown for %s: %s", clip_path.name, exc)

    _status(f"[cut] done ({len(final_reels)} reels)", quiet=args.quiet)


if __name__ == "__main__":
    main()
