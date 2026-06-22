#!/usr/bin/env python3
"""Render viral-style subtitles onto a video using ffmpeg drawtext filter.
Works without libass — uses drawtext with per-entry timing."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def load_clip_words(transcript_path: str, start: float, end: float):
    data = json.loads(Path(transcript_path).read_text("utf-8"))
    clip_start = max(0.0, start)
    clip_end = end
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


def escape_drawtext(text):
    """Escape text for ffmpeg drawtext filter."""
    text = text.replace("\\", "\\\\\\\\")
    text = text.replace("'", "'\\\\\\''")
    text = text.replace(":", "\\:")
    text = text.replace("%", "%%")
    return text


def build_drawtext_filters(blocks, font_path, video_height=1920):
    """Build a chain of drawtext filters for subtitle blocks."""
    filters = []
    font_size = 36
    margin_bottom = int(video_height * 0.15)

    for i, block in enumerate(blocks):
        block_start = block[0][0]
        block_end = block[-1][1]
        text = " ".join(w[2] for w in block)
        text_escaped = escape_drawtext(text)

        dt = (
            f"drawtext="
            f"fontfile='{font_path}':"
            f"fontsize={font_size}:"
            f"fontcolor=white:"
            f"borderw=4:"
            f"bordercolor=black:"
            f"shadowcolor=black@0.6:"
            f"shadowx=2:"
            f"shadowy=3:"
            f"text='{text_escaped}':"
            f"x=(w-text_w)/2:"
            f"y=h-{margin_bottom}-text_h:"
            f"enable='between(t,{block_start:.3f},{block_end:.3f})'"
        )
        filters.append(dt)

    return filters


def main():
    if len(sys.argv) < 5:
        print("Usage: burn_drawtext_subs.py <transcript.json> <start> <end> <reel.mp4> [output.mp4]")
        sys.exit(1)

    transcript_path = sys.argv[1]
    clip_start = float(sys.argv[2])
    clip_end = float(sys.argv[3])
    reel_path = sys.argv[4]
    output_path = sys.argv[5] if len(sys.argv) > 5 else str(Path(reel_path).with_name(
        Path(reel_path).stem + ".subtitled.mp4"
    ))

    font_path = str(Path(__file__).parent / "assets/fonts/bignoodletoooblique.ttf")
    if not Path(font_path).exists():
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    words = load_clip_words(transcript_path, clip_start, clip_end)
    if not words:
        print("No words found in clip range")
        sys.exit(1)

    blocks = group_words_into_blocks(words)
    print(f"Generated {len(blocks)} subtitle blocks from {len(words)} words")

    filters = build_drawtext_filters(blocks, font_path)
    filter_chain = ",".join(filters)

    filter_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    filter_file.write(filter_chain)
    filter_file.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", reel_path,
        "-filter_script:v", filter_file.name,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-c:a", "copy",
        output_path,
    ]
    print(f"Filter file: {filter_file.name}")
    print(f"Output: {output_path}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error:\n{result.stderr[-2000:]}")
        sys.exit(1)

    print(f"Success! Output: {output_path}")
    Path(filter_file.name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
