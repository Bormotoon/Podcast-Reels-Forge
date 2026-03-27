"""Tests for per-reel markdown helpers."""

from __future__ import annotations

from podcast_reels_forge.utils.reel_markdown import (
    build_description_text,
    build_hashtags,
    render_reel_markdown,
    sync_reel_markdowns,
)


def test_render_reel_markdown_limits_description_and_hashtags() -> None:
    moment = {
        "title": "Podcast Moment",
        "caption": "This is a long caption " * 80,
        "quote": "Key quote",
        "why": "Why it matters",
        "hashtags": ["#podcast", "#AI", "podcast", "#podcast"],
    }

    description = build_description_text(moment)
    assert len(description) <= 1000

    hashtags = build_hashtags(moment, description_text=description)
    assert len(hashtags) == 5
    assert len(set(hashtags)) == 5
    assert hashtags[:2] == ["#podcast", "#ai"]

    md = render_reel_markdown(moment, reel_label="reel_01")
    lines = md.strip().splitlines()
    assert lines[0] == "# Podcast Moment"
    assert description in md
    assert " ".join(hashtags) == lines[-1]


def test_sync_reel_markdowns_creates_adjacent_files(tmp_path) -> None:
    reels_root = tmp_path / "reels"
    rejected_dir = reels_root / "rejected"
    rejected_dir.mkdir(parents=True)
    (reels_root / "reel_01.mp4").touch()
    (rejected_dir / "reel_02_2.mp4").touch()

    moments = [
        {
            "title": "First reel",
            "caption": "First caption",
            "hashtags": ["#podcast"],
        },
        {
            "title": "Second reel",
            "caption": "Second caption",
            "hashtags": ["#interview"],
        },
    ]

    written = sync_reel_markdowns(moments, reels_root)

    assert (reels_root / "reel_01.md").exists()
    assert (rejected_dir / "reel_02_2.md").exists()
    assert len(written) == 2
