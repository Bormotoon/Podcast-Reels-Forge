#!/usr/bin/env python3
"""RU: Тонкий entrypoint для стадии транскрибации.

EN: Thin entrypoint for the transcription stage.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    """RU: CLI-точка входа для стадии транскрибации.

    EN: CLI entrypoint for the transcription stage.
    """
    from podcast_reels_forge.stages.transcribe_stage import main as stage_main

    stage_main(argv)


if __name__ == "__main__":
    main()
