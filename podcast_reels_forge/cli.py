from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    # RU: Entrypoint пакета намеренно тонкий; основной запуск — через start_forge.py.
    # EN: Package entrypoint is intentionally thin; keep start_forge.py as primary.
    parser = argparse.ArgumentParser(description="Podcast Reels Forge")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(__version__)
        return 0

    print("RU: Используйте: python3 start_forge.py\nEN: Use: python3 start_forge.py")
    return 0
