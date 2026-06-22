#!/usr/bin/env python3
"""Generate viral-style ASS subtitles from transcript JSON.
Uses karaoke \kf tags for word-level fill animation."""

import json
import sys
from pathlib import Path


def load_clip_segments(transcript_path: str, start: float, end: float, padding: float = 0.0):
    data = json.loads(Path(transcript_path).read_text("utf-8"))
    clip_start = max(0.0, start - padding)
    clip_end = end + padding

    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            ws, we, wt = w["start"], w["end"], w.get("word", "").strip()
            if not wt:
                continue
            ovs = max(clip_start, ws)
            ove = min(clip_end, we)
            if ove <= ovs:
                continue
            ss = max(0.0, ovs - clip_start)
            se = max(0.0, ove - clip_start)
            if se - ss < 0.01:
                continue
            words.append((ss, se, wt))

    return words


def fmt(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    cs = int((sec % 1) * 100)
    return f"{h}:{m:02d}:{int(sec):02d}.{cs:02d}"


def main():
    if len(sys.argv) < 5:
        print("Usage: render_viral_subs.py <transcript.json> <start> <end> <padding> [output.ass]")
        sys.exit(1)

    transcript_path = sys.argv[1]
    moment_start = float(sys.argv[2])
    moment_end = float(sys.argv[3])
    padding = float(sys.argv[4])
    output_path = sys.argv[5] if len(sys.argv) > 5 else "/tmp/karaoke.ass"

    words = load_clip_segments(transcript_path, moment_start, moment_end, padding)

    MAX_W = 5
    blocks, cur = [], []
    for ws, we, wt in words:
        if cur and len(cur) >= MAX_W:
            blocks.append(cur)
            cur = []
        cur.append((ws, we, wt))
    if cur:
        blocks.append(cur)

    ass = """[Script Info]
Title: Karaoke
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Bignoodletoo Oblique,36,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,4,0,2,40,40,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    for block in blocks:
        bs = block[0][0]
        be = block[-1][1]
        parts = []
        for ws, we, wt in block:
            dur_cs = int((we - ws) * 100)
            parts.append(f"{{\\\\kf{dur_cs}}}{wt}")
        text = " ".join(parts)
        ass += f"Dialogue: 0,{fmt(bs)},{fmt(be)},Default,,0,0,0,,{text}\n"

    Path(output_path).write_text(ass, encoding="utf-8")
    print(f"ASS written: {output_path} ({len(blocks)} blocks, {len(words)} words)")


if __name__ == "__main__":
    main()
