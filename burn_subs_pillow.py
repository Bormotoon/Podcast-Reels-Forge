#!/usr/bin/env python3
"""Render viral-style subtitles onto a video using Pillow + ffmpeg overlay.
White text, thick black border, centered bottom — classic TikTok/Reels style."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_clip_words(transcript_path: str, start: float, end: float, padding: float = 0.0):
    data = json.loads(Path(transcript_path).read_text("utf-8"))
    clip_start = max(0.0, start - padding)
    clip_end = end + padding
    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            w_start = w["start"]
            w_end = w["end"]
            w_text = w.get("word", "").strip()
            if not w_text:
                continue
            overlap_start = max(clip_start, w_start)
            overlap_end = min(clip_end, w_end)
            if overlap_end <= overlap_start:
                continue
            shifted_start = max(0.0, overlap_start - clip_start)
            shifted_end = max(0.0, overlap_end - clip_start)
            if shifted_end - shifted_start < 0.01:
                continue
            words.append((shifted_start, shifted_end, w_text))
    return words


def group_words_into_blocks(words, max_words=5, max_chars=30):
    blocks = []
    current = []
    current_len = 0
    for w_start, w_end, w_text in words:
        clean = w_text.rstrip(".,!?…;:-")
        wl = len(clean) + 1
        if current and (len(current) >= max_words or current_len + wl > max_chars):
            blocks.append(current)
            current = []
            current_len = 0
        current.append((w_start, w_end, w_text))
        current_len += wl
    if current:
        blocks.append(current)
    return blocks


def render_subtitle_image(text, font_path, width=1080, height=1920,
                          font_size=36, text_color=(255,255,255),
                          outline_color=(0,0,0), outline_width=4):
    """Render a single subtitle line as a transparent PNG."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x = 16
    pad_y = 8
    img_w = text_w + 2 * pad_x + 2 * outline_width
    img_h = text_h + 2 * pad_y + 2 * outline_width

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x = pad_x + outline_width
    y = pad_y + outline_width

    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx * dx + dy * dy <= outline_width * outline_width:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

    draw.text((x, y), text, font=font, fill=text_color)

    return img


def main():
    if len(sys.argv) < 5:
        print("Usage: burn_subs_pillow.py <transcript.json> <start> <end> <reel.mp4> [output.mp4] [padding]")
        sys.exit(1)

    transcript_path = sys.argv[1]
    clip_start = float(sys.argv[2])
    clip_end = float(sys.argv[3])
    reel_path = sys.argv[4]
    output_path = sys.argv[5] if len(sys.argv) > 5 else str(Path(reel_path).with_name(
        Path(reel_path).stem + ".subtitled.mp4"
    ))
    padding = float(sys.argv[6]) if len(sys.argv) > 6 else 5.0

    font_path = str(Path(__file__).parent / "assets/fonts/bignoodletoooblique.ttf")
    if not Path(font_path).exists():
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not Path(font_path).exists():
        import glob
        fonts = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
        font_path = fonts[0] if fonts else None

    if not font_path or not Path(font_path).exists():
        print("No font found!")
        sys.exit(1)

    print(f"Using font: {font_path}")

    words = load_clip_words(transcript_path, clip_start, clip_end, padding)
    if not words:
        print("No words found in clip range")
        sys.exit(1)

    blocks = group_words_into_blocks(words)
    print(f"Generated {len(blocks)} subtitle blocks from {len(words)} words")

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0", reel_path],
        capture_output=True, text=True
    )
    dims = probe.stdout.strip().split(",")
    vid_w, vid_h = int(dims[0]), int(dims[1])
    print(f"Video: {vid_w}x{vid_h}")

    tmp_dir = tempfile.mkdtemp(prefix="subs_")
    print(f"Temp dir: {tmp_dir}")

    try:
        png_paths = []
        for i, block in enumerate(blocks):
            block_start = block[0][0]
            block_end = block[-1][1]
            text = " ".join(w[2] for w in block)

            img = render_subtitle_image(
                text, font_path,
                width=vid_w, height=vid_h,
                font_size=36,
                text_color=(255, 255, 255),
                outline_color=(0, 0, 0),
                outline_width=4,
            )

            png_path = os.path.join(tmp_dir, f"sub_{i:03d}.png")
            img.save(png_path, "PNG")
            png_paths.append((png_path, block_start, block_end, img.width, img.height))
            print(f"  Block {i}: {block_start:.2f}-{block_end:.2f} '{text[:40]}...'")

        input_args = ["-y", "-i", reel_path]
        for png_path, _, _, _, _ in png_paths:
            input_args.extend(["-i", png_path])

        filter_parts = []
        overlay_ref = "[0:v]"

        for idx, (png_path, bstart, bend, pw, ph) in enumerate(png_paths):
            input_idx = idx + 1
            x_expr = f"(main_w-{pw})/2"
            y_expr = f"main_h*0.85-{ph}"
            out_label = f"[ov{idx}]"

            filter_parts.append(
                f"[{input_idx}:v]format=rgba[img{idx}];"
                f"{overlay_ref}[img{idx}]overlay={x_expr}:{y_expr}:"
                f"enable='between(t,{bstart:.3f},{bend:.3f})'{out_label}"
            )
            overlay_ref = out_label

        filter_complex = ";".join(filter_parts)

        filter_file = os.path.join(tmp_dir, "filter.txt")
        with open(filter_file, "w") as f:
            f.write(filter_complex)

        cmd = [
            "ffmpeg", "-y",
            *input_args,
            "-filter_complex_script", filter_file,
            "-map", f"{overlay_ref}",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-c:a", "copy",
            output_path,
        ]

        print(f"\nRunning ffmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"FFmpeg error:\n{result.stderr[-3000:]}")
            sys.exit(1)

        print(f"\nSuccess! Output: {output_path}")
        if Path(output_path).exists():
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            print(f"File size: {size_mb:.1f} MB")

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
