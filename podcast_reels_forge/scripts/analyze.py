#!/usr/bin/env python3
"""Thin CLI wrapper for the staged analysis pipeline."""

from __future__ import annotations

from podcast_reels_forge.stages.analyze_stage import (
    Moment,
    _assign_speakers,
    _normalize_prompt_lang,
    chunk_segments_by_time,
    create_provider,
    fmt_hms,
    find_moments,
    get_llm_json,
    main,
    render_reels_summary_markdown,
    segments_to_compact_text,
)

__all__ = [
    "Moment",
    "_assign_speakers",
    "_normalize_prompt_lang",
    "chunk_segments_by_time",
    "create_provider",
    "fmt_hms",
    "find_moments",
    "get_llm_json",
    "main",
    "render_reels_summary_markdown",
    "segments_to_compact_text",
]


if __name__ == "__main__":
    main()
