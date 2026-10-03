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
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from podcast_reels_forge.utils import subtitle_sync
from podcast_reels_forge.utils.burned_subtitles import (
    DEFAULT_SUBTITLE_FONT,
    SUBTITLE_SYNC_FILE,
    SubtitleRenderSettings,
    SubtitleSegment,
    WordKey,
    clip_words,
    load_transcript_segments,
    retime_segments,
    slice_segments_for_clip,
    subtitle_settings_from_conf,
    word_key,
    write_srt_file,
    _prepare_subtitle_segments,
    _write_ass_file,
)
from podcast_reels_forge.utils.face_crop import (
    face_detection_available,
    face_detection_unavailable_reason,
)
from podcast_reels_forge.utils.face_track import (
    FramingPlan,
    TrackingSettings,
    analyze_clip,
    build_framing_filter,
    center_plan,
)
from podcast_reels_forge.utils.ffmpeg import (
    build_video_codec_args,
    ffmpeg_bin,
    ffmpeg_has_nvenc,
    resolve_ffmpeg_with_libass,
    resolve_gpu_render_ffmpeg,
)
from podcast_reels_forge.utils.clip_intervals import (
    ClipEdges,
    load_speech_index,
    moment_bounds,
    padded_intervals,
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
    # "speaker": show whoever talks; "split": stack two steadily visible
    # people ("single" is the old name of "speaker").
    two_speaker_layout: str = "speaker"
    # Follow the face within a turn (False: one framing per turn).
    face_follow: bool = True
    # Pick the talking face with Light-ASD when several people are in frame.
    active_speaker: bool = True
    # New speaker: "cut" or "pan".
    speaker_switch: str = "cut"
    # Torch device for face tracking ("cuda": GPU only, no CPU fallback).
    face_device: str = "cuda"
    # Decode and scale reels on the GPU when an ffmpeg build can.
    gpu_decode: bool = True
    # Directory libass loads subtitle fonts from (the subtitle font's folder).
    # Without it a font is only found when it happens to be installed.
    fonts_dir: str = ""


def _run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command with safe defaults."""

    normalized_cmd = [str(part) for part in cmd]
    return subprocess.run(normalized_cmd, capture_output=True, text=True, check=False)


def _status(msg: str, *, quiet: bool) -> None:
    if not quiet:
        LOG.info(msg)


_PLAN_LOCK = threading.Lock()
_PLANS: dict[tuple[str, float, float], FramingPlan | None] = {}


def tracking_settings(opts: FfmpegOptions) -> TrackingSettings:
    layout = "split" if opts.two_speaker_layout == "split" else "speaker"
    return TrackingSettings(
        min_face_size=int(opts.face_min_size),
        follow=bool(opts.face_follow),
        active_speaker=bool(opts.active_speaker),
        layout=layout,
        switch=opts.speaker_switch if opts.speaker_switch in {"cut", "pan"} else "cut",
        device=opts.face_device,
    )


def framing_plan(video_in: Path, start: float, end: float, opts: FfmpegOptions) -> FramingPlan | None:
    """The clip's framing, computed once per interval (a clip is encoded up to
    three times: with subtitles, without them as a fallback, and a clean copy)."""

    key = (str(video_in), round(start, 3), round(end, 3))
    with _PLAN_LOCK:
        if key in _PLANS:
            return _PLANS[key]
    try:
        plan = analyze_clip(video_in, start, end, tracking_settings(opts))
    except Exception as exc:  # noqa: BLE001 - framing must never lose the clip
        LOG.warning("face tracking failed for %.1f-%.1f (%s); centre crop", start, end, exc)
        plan = None
    with _PLAN_LOCK:
        _PLANS[key] = plan
    return plan


def _write_framing_report(out_path: Path, plan: FramingPlan | None, kind: str) -> None:
    try:
        folder = out_path.parent / "framing"
        folder.mkdir(parents=True, exist_ok=True)
        body = {"framing": kind, **(plan.as_dict() if plan is not None else {})}
        (folder / f"{out_path.stem}.json").write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        LOG.debug("could not write the framing report for %s: %s", out_path.name, exc)


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

    The vertical frame follows whoever is talking (see ``face_track``); the
    reel is decoded, scaled and encoded on the GPU when an ffmpeg build can
    (``resolve_gpu_render_ffmpeg``), else by the previous CPU filters.

    With ``encode_rejected=False`` a clip the face check rejects is not
    encoded at all: ``(False, out_path, reason)`` comes back and nothing is
    written.
    """

    face_rejection_reason: str | None = None
    plan: FramingPlan | None = None
    framing = "none"
    start_offset = max(0, start - opts.padding)
    end_offset = end + opts.padding
    if opts.vertical_crop:
        src_w, src_h = _frame_size(video_in)
        if opts.smart_crop_face and face_detection_available():
            plan = framing_plan(video_in, start_offset, end_offset, opts)
            face_rate = plan.rate if plan is not None else 0.0
            if plan is not None and opts.filter_face_ratio > 0 and face_rate < opts.filter_face_ratio:
                LOG.debug("Rejecting clip (face detection rate %.2f < %.2f)", face_rate, opts.filter_face_ratio)
                is_rejected = True
                face_rejection_reason = (
                    f"face ratio {face_rate:.2f} < {opts.filter_face_ratio:.2f}"
                )
            framing = "tracked" if plan is not None else "center"
        if plan is None and src_w and src_h:
            plan = center_plan(src_w, src_h, end_offset - start_offset)
            framing = framing if framing != "none" else "center"

    burning_subtitles = ass_path is not None and ass_path.exists()
    libass_ffmpeg: str | None = None
    ass_filter = ""
    if ass_path is not None and burning_subtitles:
        # Escape path for FFmpeg filter
        safe_ass_path = str(ass_path.resolve()).replace('\\', '/').replace(':', '\\:')
        ass_filter = f"ass='{safe_ass_path}'"
        if opts.fonts_dir:
            safe_fonts_dir = str(Path(opts.fonts_dir).resolve()).replace('\\', '/').replace(':', '\\:')
            ass_filter += f":fontsdir='{safe_fonts_dir}'"
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
    if opts.vertical_crop and opts.smart_crop_face and not out_path.stem.endswith(".nosubs"):
        _write_framing_report(out_path, plan, framing)

    gpu_ffmpeg = (
        resolve_gpu_render_ffmpeg(burning_subtitles)
        if opts.use_nvenc and opts.gpu_decode
        else None
    )

    def _filters(gpu: bool) -> list[str]:
        filters: list[str] = []
        if opts.vertical_crop:
            if plan is not None:
                filters.append(build_framing_filter(plan, gpu=gpu and not plan.split_shots))
            else:
                filters.append("scale=w=1080:h=1920:force_original_aspect_ratio=increase,crop=1080:1920")
        if ass_filter:
            filters.append(ass_filter)
        return filters

    def _build(use_nvenc: bool, gpu: bool = False) -> list[str]:
        if gpu and gpu_ffmpeg:
            binary = gpu_ffmpeg
        elif burning_subtitles and libass_ffmpeg:
            binary = libass_ffmpeg
        else:
            binary = ffmpeg_bin()
        cmd = [binary, "-y"]
        if gpu and gpu_ffmpeg:
            # NVDEC into system memory: the window is cut on the CPU (a pointer
            # move), then uploaded and scaled by scale_cuda.
            cmd += ["-hwaccel", "cuda"]
        cmd += [
            "-ss",
            str(start_offset),
            "-to",
            str(end_offset),
            "-i",
            str(video_in),
        ]
        filters = _filters(gpu and gpu_ffmpeg is not None)
        if filters:
            cmd += ["-vf", ",".join(filters)]
        if gpu and gpu_ffmpeg:
            cmd += build_video_codec_args(
                use_nvenc=True,
                v_bitrate=opts.v_bitrate,
                preset=opts.preset,
                nvenc_cq=opts.nvenc_cq,
                nvenc_preset=opts.nvenc_preset,
            )
        elif burning_subtitles and libass_ffmpeg and not (use_nvenc and libass_has_nvenc):
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
    res = None
    if gpu_ffmpeg:
        res = _run_subprocess(_build(True, gpu=True))
        if res.returncode != 0:
            LOG.warning(
                "GPU render failed for %s (%s); falling back to the CPU filters",
                out_path.name,
                (res.stderr or "").strip()[-300:],
            )
    if res is None or res.returncode != 0:
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
    ap.add_argument("--face-samples", type=int, default=7, help="Legacy, ignored: faces are tracked at 5 fps")
    ap.add_argument("--face-min-size", type=int, default=40, help="Min face height in pixels; smaller faces are ignored")
    ap.add_argument(
        "--two-speaker-layout",
        choices=("speaker", "split", "single"),
        default="speaker",
        help="Several people in frame: show whoever talks (speaker) or stack two "
        "steadily visible people (split); 'single' is the old name of 'speaker'",
    )
    ap.add_argument(
        "--face-follow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Follow the face smoothly within a turn (off: one framing per turn)",
    )
    ap.add_argument(
        "--active-speaker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tell the talking face by lips and sound (Light-ASD) when several people are in frame",
    )
    ap.add_argument(
        "--speaker-switch",
        choices=("cut", "pan"),
        default="cut",
        help="New speaker: hard cut, or a short pan when they sit close",
    )
    ap.add_argument(
        "--face-device",
        default="cuda",
        help="Torch device for face tracking; 'cuda' never falls back to the CPU",
    )
    ap.add_argument(
        "--gpu-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode and scale reels on the GPU (NVDEC + scale_cuda) when an ffmpeg build can",
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
    ap.add_argument(
        "--padding", type=float, default=0,
        help="Extra seconds around a moment the transcript has no word timings for",
    )
    ClipEdges.add_arguments(ap)
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
        "--subtitle-karaoke",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Highlight subtitles word by word (\\kf); off: each cue appears whole",
    )
    ap.add_argument(
        "--subtitle-settings-json",
        default="",
        help="The config's whole `subtitles:` section as JSON (preset, highlight, "
        "layout, timing...); the --subtitle-* flags above override it",
    )
    ap.add_argument(
        "--subtitle-sync-model",
        default="",
        help="Re-check every clip's word timings with this Whisper model before "
        "burning subtitles and fix drift (empty: off)",
    )
    ap.add_argument("--subtitle-sync-min-match", type=float, default=0.5,
                    help="Trust the check only when this share of words is found")
    ap.add_argument("--subtitle-sync-threshold", type=float, default=0.2,
                    help="Retime when the p95 word-start drift reaches this many seconds")
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


def check_subtitle_sync(
    transcript_segments: list[SubtitleSegment],
    clip_intervals: list[tuple[float, float]],
    *,
    transcript_json: Path,
    fallback_audio: Path,
    reels_dir: Path,
    model_name: str,
    min_match_ratio: float,
    apply_threshold_s: float,
    quiet: bool,
) -> list[SubtitleSegment]:
    """Listen to every clip again and fix the subtitle timings that drifted.

    One Whisper model serves all clips. The result (per-clip verdicts and every
    word that was retimed) goes to reels/subtitle_sync.json, which also lets a
    later subtitle re-sync without re-cutting keep the fixed timings.
    """

    try:
        data = json.loads(transcript_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    language = str(data.get("language") or "") or None
    source = Path(str(data.get("source_audio") or data.get("audio") or ""))
    if not source.is_file():
        source = fallback_audio

    try:
        model = subtitle_sync.load_model(model_name)
    except Exception as exc:  # noqa: BLE001 - the check is an extra, not a gate
        LOG.warning("subtitle sync: could not load Whisper %s (%s); skipping the check", model_name, exc)
        return transcript_segments

    reports: list[dict[str, object]] = []
    retimed_rows: list[dict[str, object]] = []
    retimed: dict[WordKey, tuple[float, float]] = {}
    for index, (clip_start, clip_end) in enumerate(clip_intervals, 1):
        reference = clip_words(transcript_segments, clip_start=clip_start, clip_end=clip_end)
        words, report = subtitle_sync.sync_clip(
            model,
            source,
            [subtitle_sync.TimedWord(w.start, w.end, w.text) for w in reference],
            clip_start=clip_start,
            clip_end=clip_end,
            language=language,
            min_match_ratio=min_match_ratio,
            apply_threshold_s=apply_threshold_s,
        )
        row = {"reel": f"reel_{index:02d}", **report.as_dict()}
        reports.append(row)
        _status(
            f"[cut] subtitle sync reel_{index:02d}: {report.verdict} "
            f"(matched {report.matched_words}/{report.reference_words}, "
            f"p95 drift {report.p95_abs_shift_s:.2f}s, max {report.max_abs_shift_s:.2f}s)",
            quiet=quiet,
        )
        if not report.applied:
            continue
        for original, fixed in zip(reference, words):
            if (fixed.start, fixed.end) != (original.start, original.end):
                retimed[word_key(original)] = (fixed.start, fixed.end)
                retimed_rows.append({
                    "orig_start": round(original.start, 3),
                    "text": original.text,
                    "start": fixed.start,
                    "end": fixed.end,
                })
    del model

    reels_dir.mkdir(parents=True, exist_ok=True)
    (reels_dir / SUBTITLE_SYNC_FILE).write_text(
        json.dumps(
            {"model": model_name, "source_audio": str(source), "clips": reports, "retimed_words": retimed_rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return retime_segments(transcript_segments, retimed)


def _load_moments(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    return []


def _subtitle_settings_from_json(raw: str) -> SubtitleRenderSettings:
    """Settings from the pipeline's ``subtitles:`` section (--subtitle-settings-json).

    Only the font, wrapping and karaoke used to reach this process as flags,
    so every other subtitle setting in config.yaml (lines, width, offset,
    fades) was silently ignored on the normal cut path.
    """

    section: object = {}
    if raw:
        try:
            section = json.loads(raw)
        except ValueError:
            LOG.warning("Ignoring malformed --subtitle-settings-json")
    if not isinstance(section, dict):
        section = {}
    return subtitle_settings_from_conf({"subtitles": section}, repo_dir=Path.cwd())


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
        subtitle_settings = replace(
            _subtitle_settings_from_json(args.subtitle_settings_json),
            enabled=True,
            font_path=args.subtitle_font.resolve(),
            wrap_words=bool(args.subtitle_wrap_words),
        )
        if args.subtitle_karaoke:
            subtitle_settings = replace(subtitle_settings, karaoke=True)
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
        face_follow=bool(args.face_follow),
        active_speaker=bool(args.active_speaker),
        speaker_switch=str(args.speaker_switch),
        face_device=str(args.face_device),
        gpu_decode=bool(args.gpu_decode),
        fonts_dir=str(subtitle_settings.font_path.parent) if subtitle_settings is not None else "",
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

    # Edges are resolved per clip up front (see padded_intervals): fitted to
    # the speech when the transcript has word timings, else padded. The cut
    # itself then gets the final interval and no padding of its own.
    clip_intervals = padded_intervals(
        [moment_bounds(m) for m in moments],
        float(opts.padding),
        index=load_speech_index(args.transcript_json),
        edges=ClipEdges.from_args(args),
    )
    cut_opts = replace(opts, padding=0.0)

    if subtitle_settings is not None and args.subtitle_sync_model and transcript_segments:
        transcript_segments = check_subtitle_sync(
            transcript_segments,
            clip_intervals,
            transcript_json=args.transcript_json,
            fallback_audio=args.input,
            reels_dir=reels_dir,
            model_name=str(args.subtitle_sync_model),
            min_match_ratio=float(args.subtitle_sync_min_match),
            apply_threshold_s=float(args.subtitle_sync_threshold),
            quiet=bool(args.quiet),
        )

    def prepare_clip_subtitles(out_file: Path, clip_start: float, clip_end: float) -> Path | None:
        """Write the clip's .srt/.ass for the exact interval ffmpeg will cut.

        Built before the one and only encode, from the same padded interval,
        so subtitles and footage cannot drift apart.
        """

        assert subtitle_settings is not None
        try:
            clip_segments = slice_segments_for_clip(
                transcript_segments,
                clip_start=clip_start,
                clip_end=clip_end,
            )
            clip_segments = _prepare_subtitle_segments(clip_segments, settings=subtitle_settings)
            if not clip_segments:
                LOG.warning(
                    "%s: no transcript words in [%.1f, %.1f]; cutting it without subtitles",
                    out_file.name, clip_start, clip_end,
                )
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
        start_f, end_f = moment_bounds(m)
        clip_start, clip_end = clip_intervals[i]

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
            prepare_clip_subtitles(out_file, clip_start, clip_end)
            if subtitle_settings is not None and not is_rejected
            else None
        )
        success, final_path, face_reason = ffmpeg_cut(
            args.input,
            clip_start,
            clip_end,
            out_file,
            cut_opts,
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
                clip_start,
                clip_end,
                out_file,
                cut_opts,
                is_rejected=is_rejected,
                rejected_dir=rejected_dir,
            )
        elif success and ass_path is not None and args.keep_nosubs:
            # Optional clean copy for platforms/edits that want no captions.
            ffmpeg_cut(
                args.input,
                clip_start,
                clip_end,
                final_path.with_name(f"{final_path.stem}.nosubs.mp4"),
                cut_opts,
            )
        if face_reason:
            rejection_reasons.append(face_reason)
        if not success:
            return None, rejection_reasons, "failed"
        outcome = "rejected" if "rejected" in final_path.parts else "ok"
        if args.qa and outcome == "ok":
            qa_end = min(clip_end, source_duration) if source_duration else clip_end
            problems = check_clip(
                final_path,
                expected_duration=qa_end - clip_start,
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
