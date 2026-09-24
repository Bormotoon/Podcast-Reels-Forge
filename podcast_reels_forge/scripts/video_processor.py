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
    SubtitleSegment,
    load_transcript_segments,
    slice_segments_for_clip,
    write_srt_file,
    _prepare_subtitle_segments,
    _write_ass_file,
)
from podcast_reels_forge.utils.face_crop import (
    FaceCropSettings,
    analyze_face_layout,
    build_sample_times,
    build_split_filter,
    compute_crop_x_for_scaled_height,
    face_detection_available,
    face_detection_unavailable_reason,
)
from podcast_reels_forge.utils.ffmpeg import (
    build_video_codec_args,
    ffmpeg_bin,
    ffmpeg_has_nvenc,
    resolve_ffmpeg_with_libass,
)
from podcast_reels_forge.utils.media_qa import check_clip, media_duration
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
    # "split" stacks two steadily visible speakers; "single" always crops one.
    two_speaker_layout: str = "split"


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
    encode_rejected: bool = True,
) -> tuple[bool, Path, str | None]:
    """Cut a segment from video with optional vertical crop.

    With ``encode_rejected=False`` a clip the face check rejects is not
    encoded at all: ``(False, out_path, reason)`` comes back and nothing is
    written.
    """

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
            layout = analyze_face_layout(video_in, sample_times_s=sample_times, settings=face_settings)
            face_rate = layout.rate

            if opts.filter_face_ratio > 0 and face_rate < opts.filter_face_ratio:
                LOG.debug("Rejecting clip (face detection rate %.2f < %.2f)", face_rate, opts.filter_face_ratio)
                is_rejected = True
                face_rejection_reason = (
                    f"face ratio {face_rate:.2f} < {opts.filter_face_ratio:.2f}"
                )

            if layout.primary is not None:
                src_w, src_h = _frame_size(video_in)
                if layout.kind == "split" and opts.two_speaker_layout == "split" and src_w and src_h:
                    LOG.debug("Two speakers in frame; stacking them")
                    vf = build_split_filter(src_w=src_w, src_h=src_h, centers=layout.centers)
                else:
                    center_ratio = layout.primary[0]
                    LOG.debug("Face detected at ratio %.2f; applying smart crop", center_ratio)
                    crop_x = compute_crop_x_for_scaled_height(
                        src_w=src_w,
                        src_h=src_h,
                        target_w=1080,
                        target_h=1920,
                        center_ratio=center_ratio,
                    )
                    vf = f"scale=-2:1920,crop=1080:1920:{crop_x}:0"

        filters.append(vf)

    burning_subtitles = ass_path is not None and ass_path.exists()
    libass_ffmpeg: str | None = None
    if ass_path is not None and burning_subtitles:
        # Escape path for FFmpeg filter
        safe_ass_path = str(ass_path.resolve()).replace('\\', '/').replace(':', '\\:')
        filters.append(f"ass='{safe_ass_path}'")
        # The NVENC-preferred ffmpeg build may lack libass, which fails the 'ass'
        # filter identically under NVENC and software libx264. Resolve a build that
        # actually has libass and use it (software-only) for this pass.
        libass_ffmpeg = resolve_ffmpeg_with_libass()
        if libass_ffmpeg is None:
            LOG.error(
                "No ffmpeg build with libass found; cannot burn subtitles into %s",
                out_path.name,
            )
            return False, out_path, face_rejection_reason

    if is_rejected and not encode_rejected:
        return False, out_path, face_rejection_reason

    if is_rejected and rejected_dir:
        out_path = rejected_dir / out_path.name

    # RU: Каталог может ещё не существовать: отбраковка по доле кадров с лицом
    #     решается здесь, уже после mkdir на стороне вызывающего кода. Без этого
    #     ffmpeg молча падает с "No such file or directory", и клип теряется —
    #     ни в reels/, ни в rejected/.
    # EN: The directory may not exist yet: face-ratio rejection is decided here,
    #     after the caller has done its mkdir. Without this ffmpeg fails with
    #     "No such file or directory" and the clip is lost — it lands neither in
    #     reels/ nor in rejected/.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_offset = max(0, start - opts.padding)
    end_offset = end + opts.padding

    def _build(use_nvenc: bool) -> list[str]:
        cmd = [
            libass_ffmpeg if (burning_subtitles and libass_ffmpeg) else ffmpeg_bin(),
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
        if burning_subtitles and libass_ffmpeg and not (use_nvenc and libass_has_nvenc):
            # This libass-capable build has no NVENC (or NVENC failed): libx264.
            cmd += ["-c:v", "libx264", "-preset", opts.preset, "-b:v", opts.v_bitrate, "-pix_fmt", "yuv420p"]
        else:
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

    # A build with both libass and NVENC burns subtitles on the GPU; the old
    # path always fell back to software libx264 when subtitles were on.
    libass_has_nvenc = libass_ffmpeg is not None and _build_has_nvenc(libass_ffmpeg)
    res = _run_subprocess(_build(opts.use_nvenc))
    if res.returncode != 0 and burning_subtitles and opts.use_nvenc and libass_has_nvenc:
        LOG.warning("NVENC subtitle-burn failed for %s; retrying with libx264", out_path.name)
        res = _run_subprocess(_build(False))
    if res.returncode != 0 and burning_subtitles:
        LOG.error(
            "Subtitle-burn encode failed for %s: %s",
            out_path.name,
            (res.stderr or "").strip()[-800:],
        )
    elif res.returncode != 0 and opts.use_nvenc and ffmpeg_has_nvenc():
        # NVENC was attempted but failed; rebuild with software libx264.
        LOG.warning("NVENC encode failed for %s; retrying with software libx264", out_path.name)
        res = _run_subprocess(_build(False))

    if res.returncode != 0 and not burning_subtitles:
        LOG.error(
            "Encode failed for %s: %s",
            out_path.name,
            (res.stderr or "").strip()[-800:],
        )

    return res.returncode == 0, out_path, face_rejection_reason


def _frame_size(video_in: Path) -> tuple[int, int]:
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_in))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
        cap.release()
        return size
    except Exception:
        return 0, 0


def _build_has_nvenc(ffmpeg: str) -> bool:
    from podcast_reels_forge.utils.ffmpeg import build_has_nvenc

    return build_has_nvenc(ffmpeg)


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
    ap.add_argument("--face-min-size", type=int, default=60, help="Min face height in pixels; smaller faces are ignored")
    ap.add_argument(
        "--two-speaker-layout",
        choices=("split", "single"),
        default="split",
        help="Two people steadily in frame: stack them (split) or crop one (single)",
    )
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
    ap.add_argument(
        "--qa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check every rendered clip with ffprobe (streams, duration); broken ones fail",
    )
    ap.add_argument(
        "--qa-blackdetect",
        action="store_true",
        help="Also fail clips that are mostly black (decodes each clip once more)",
    )
    ap.add_argument(
        "--render-rejected",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also encode clips the quality filters reject into reels/rejected/ (default: list them only)",
    )
    ap.add_argument(
        "--keep-nosubs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With --burn-subtitles, also render a clean reel_XX.nosubs.mp4 (an extra encode)",
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
        two_speaker_layout=str(args.two_speaker_layout),
    )

    if (
        opts.smart_crop_face
        and opts.vertical_crop
        and not face_detection_available(download=True)
    ):
        LOG.warning(
            "--smart-crop-face enabled but face detection is unavailable (%s); "
            "falling back to center crop",
            face_detection_unavailable_reason(),
        )

    transcript_segments: list[SubtitleSegment] = []
    if subtitle_settings is not None:
        try:
            transcript_segments = load_transcript_segments(args.transcript_json)
        except Exception as exc:
            LOG.error("Failed to load the transcript for subtitles: %s", exc)
            sys.exit(1)

    subtitle_errors: list[str] = []
    source_duration = media_duration(args.input) if args.qa else None

    def prepare_clip_subtitles(out_file: Path, start: float, end: float) -> Path | None:
        """Write the clip's .srt/.ass for the exact interval ffmpeg will cut.

        Built before the one and only encode, from the same padded interval,
        so subtitles and footage cannot drift apart.
        """

        assert subtitle_settings is not None
        try:
            clip_segments = slice_segments_for_clip(
                transcript_segments,
                clip_start=max(0.0, start - opts.padding),
                clip_end=end + opts.padding,
            )
            clip_segments = _prepare_subtitle_segments(clip_segments, settings=subtitle_settings)
            if not clip_segments:
                return None
            write_srt_file(out_file.with_suffix(".srt"), clip_segments)
            ass_path = out_file.with_suffix(".ass")
            _write_ass_file(ass_path, clip_segments, subtitle_settings)
            return ass_path
        except Exception as exc:
            LOG.error("Failed to prepare subtitles for %s: %s", out_file.name, exc)
            subtitle_errors.append(out_file.name)
            return None

    def process_moment(
        i_m: tuple[int, dict[str, object]],
    ) -> tuple[Path | None, list[str], str]:
        """Returns (final_path_or_none, rejection_reasons, outcome).

        outcome: "ok", "rejected" (encoded into rejected/), "skipped"
        (rejected and not encoded) or "failed".
        """
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
        if is_rejected and not args.render_rejected:
            # Nobody publishes these; encoding them only cost GPU time.
            return None, rejection_reasons, "skipped"
        if is_rejected:
            rejected_dir.mkdir(exist_ok=True)

        # Rejected clips are kept for review only, so they skip subtitles.
        ass_path = (
            prepare_clip_subtitles(out_file, start_f, end_f)
            if subtitle_settings is not None and not is_rejected
            else None
        )
        success, final_path, face_reason = ffmpeg_cut(
            args.input,
            start_f,
            end_f,
            out_file,
            opts,
            is_rejected=is_rejected,
            rejected_dir=rejected_dir,
            ass_path=ass_path,
            encode_rejected=bool(args.render_rejected),
        )
        if not success and face_reason and not args.render_rejected:
            return None, [*rejection_reasons, face_reason], "skipped"
        if not success and ass_path is not None:
            LOG.error(
                "Subtitle burn failed for %s; cutting it without subtitles",
                out_file.name,
            )
            success, final_path, face_reason = ffmpeg_cut(
                args.input,
                start_f,
                end_f,
                out_file,
                opts,
                is_rejected=is_rejected,
                rejected_dir=rejected_dir,
            )
        elif success and ass_path is not None and args.keep_nosubs:
            # Optional clean copy for platforms/edits that want no captions.
            ffmpeg_cut(
                args.input,
                start_f,
                end_f,
                final_path.with_name(f"{final_path.stem}.nosubs.mp4"),
                opts,
            )
        if face_reason:
            rejection_reasons.append(face_reason)
        if not success:
            return None, rejection_reasons, "failed"
        outcome = "rejected" if "rejected" in final_path.parts else "ok"
        if args.qa and outcome == "ok":
            clip_start = max(0.0, start_f - opts.padding)
            clip_end = end_f + opts.padding
            if source_duration:
                clip_end = min(clip_end, source_duration)
            problems = check_clip(
                final_path,
                expected_duration=clip_end - clip_start,
                blackdetect=bool(args.qa_blackdetect),
            )
            if problems:
                LOG.error("%s failed QA: %s", final_path.name, "; ".join(problems))
                rejected_dir.mkdir(exist_ok=True)
                moved = rejected_dir / final_path.name
                if final_path.exists():
                    final_path.replace(moved)
                return moved, [*rejection_reasons, *problems], "failed"
        return final_path, rejection_reasons, outcome

    _status(f"[cut] {len(moments)} moments", quiet=args.quiet)
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        raw_results: Iterable[tuple[Path | None, list[str], str]] = pool.map(
            process_moment, enumerate(moments)
        )
        if args.verbose:
            raw_results = tqdm(raw_results, total=len(moments))
        results = list(raw_results)

    # results[i] = (path | None, rejection_reasons, outcome); indices line up with moments.
    all_cut_paths = [path for path, _, _ in results]
    final_reels = [path for path, _, outcome in results if path is not None and outcome == "ok"]
    if subtitle_errors:
        LOG.error("Failed to burn subtitles for: %s", ", ".join(sorted(subtitle_errors)))
        sys.exit(1)

    # RU: Список отбракованного — всегда, даже если сами клипы не кодировались.
    # EN: The list of what was rejected — always, even when nothing was encoded.
    rejected_rows = [
        {
            "index": i + 1,
            "start": moments[i].get("start"),
            "end": moments[i].get("end"),
            "title": moments[i].get("title"),
            "score": moments[i].get("score"),
            "reasons": reasons,
            "encoded": outcome == "rejected",
        }
        for i, (_path, reasons, outcome) in enumerate(results)
        if outcome in {"rejected", "skipped"} and i < len(moments)
    ]
    rejected_json = reels_dir / "rejected.json"
    if rejected_rows:
        rejected_json.write_text(json.dumps(rejected_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        rejected_json.unlink(missing_ok=True)

    if any(p is not None for p in all_cut_paths):
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

        # Write per-clip .txt (Instagram caption) and .md for every encoded clip.
        for i, (maybe_clip_path, rejection_reasons, _outcome) in enumerate(results):
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

    # RU: Отбраковка и провал кодирования — разные исходы, и оба нужно назвать:
    #     иначе «done (0 reels)» читается как успех.
    # EN: Rejection and a failed encode are different outcomes and both must be
    #     named: otherwise "done (0 reels)" reads as success.
    rejected_count = len(rejected_rows)
    failed_count = sum(1 for _p, _r, outcome in results if outcome == "failed")
    _status(
        f"[cut] done ({len(final_reels)} reels, "
        f"{rejected_count} rejected, {failed_count} failed)",
        quiet=args.quiet,
    )
    if failed_count:
        LOG.error("%d of %d clips failed to encode", failed_count, len(results))
        # Non-zero, so the pipeline's run report shows the cut as failed.
        sys.exit(2)


if __name__ == "__main__":
    main()
