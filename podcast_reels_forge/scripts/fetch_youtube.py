#!/usr/bin/env python3
"""RU: Автономная загрузка роликов с YouTube.

Тот же код, что использует стадия `fetch` пайплайна, но запускается отдельно —
удобно, когда нужно просто набрать материал в папку, ничего не обрабатывая.

Требования:
- `pip install -U yt-dlp` (опциональная зависимость проекта)
- переменная окружения `YOUTUBE_API_KEY` — необязательна: без неё перечисление
  плейлистов и каналов делает сам yt-dlp

Примеры:
    python3 -m podcast_reels_forge.scripts.fetch_youtube --list "@pedobraz"
    python3 -m podcast_reels_forge.scripts.fetch_youtube "https://youtu.be/ID"
    python3 -m podcast_reels_forge.scripts.fetch_youtube "@pedobraz" --limit 5 --audio-only

EN: Standalone YouTube downloader.

The same code the pipeline's `fetch` stage runs, callable on its own — handy for
just filling a folder with material without processing any of it.

Requirements:
- `pip install -U yt-dlp` (an optional project dependency)
- `YOUTUBE_API_KEY` is optional: without it, yt-dlp does the playlist and channel
  listing itself
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from podcast_reels_forge.sources.youtube import (
    VideoFilters,
    YouTubeError,
    api_key_from_env,
    format_duration,
    resolve_sources,
)
from podcast_reels_forge.stages.fetch_stage import (
    DEFAULT_FILENAME_TEMPLATE,
    FetchConfig,
    describe_videos,
    fetch_videos,
)
from podcast_reels_forge.utils.env import load_dotenv
from podcast_reels_forge.utils.logging_utils import setup_logging

LOGGER = setup_logging()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """RU: Парсит аргументы командной строки. EN: Parse command line arguments."""

    ap = argparse.ArgumentParser(
        description="Download YouTube videos, playlists or whole channels.",
    )
    ap.add_argument(
        "sources",
        nargs="+",
        help=(
            "Ссылки на ролик/плейлист/канал, @handle или id. "
            "Id и handle, начинающиеся с дефиса, отделяйте через --"
        ),
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("input/youtube"),
        help="Куда складывать файлы (по умолчанию: input/youtube)",
    )
    ap.add_argument(
        "--exclude",
        metavar="URL",
        action="append",
        default=None,
        help=(
            "Никогда не брать эти ролики: плейлист, канал или отдельный ролик. "
            "Можно повторять"
        ),
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Показать, что будет взято, и выйти — ничего не скачивая",
    )
    ap.add_argument("--limit", type=int, default=0, help="Взять только N последних")
    ap.add_argument("--since", default=None, help="Не раньше даты, ГГГГ-ММ-ДД")
    ap.add_argument("--until", default=None, help="Не позже даты, ГГГГ-ММ-ДД")
    ap.add_argument(
        "--min-duration",
        type=int,
        default=60,
        help="Пропускать короче N секунд (60 отсекает Shorts); 0 — без нижней границы",
    )
    ap.add_argument(
        "--max-duration", type=int, default=0, help="Пропускать длиннее N секунд",
    )
    ap.add_argument(
        "--audio-only",
        action="store_true",
        help="Качать только аудиодорожку — заметно быстрее, но нарезать будет нечего",
    )
    ap.add_argument(
        "--max-height",
        type=int,
        default=1080,
        help="Потолок высоты видео (0 — максимальное доступное)",
    )
    ap.add_argument(
        "--filename-template",
        default=DEFAULT_FILENAME_TEMPLATE,
        help="Шаблон имени файла в синтаксисе yt-dlp",
    )
    ap.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Журнал уже скачанных id (для инкрементального прогона канала)",
    )
    ap.add_argument("--cookies", type=Path, default=None, help="Файл cookies (Netscape)")
    ap.add_argument("--rate-limit", default=None, help="Ограничение скорости, напр. 5M")
    ap.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Скачать заново, даже если файл уже лежит в папке",
    )
    ap.add_argument("--quiet", action="store_true", help="Только ошибки")
    ap.add_argument("--verbose", action="store_true", help="Подробный вывод")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """RU: Точка входа. EN: Entry point."""

    args = parse_args(argv)
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    load_dotenv(Path(__file__).resolve().parents[2])

    filters = VideoFilters(
        limit=max(0, int(args.limit)),
        since=args.since,
        until=args.until,
        min_duration=max(0, int(args.min_duration)),
        max_duration=max(0, int(args.max_duration)),
    )

    try:
        videos = resolve_sources(
            list(args.sources),
            api_key=api_key_from_env(),
            filters=filters,
            exclude=args.exclude,
        )
    except YouTubeError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    if args.list:
        print(describe_videos(videos, args.outdir))
        return 0

    if not videos:
        print("Под условия отбора не попал ни один ролик.")
        return 0

    config = FetchConfig(
        download_dir=args.outdir,
        want_video=not args.audio_only,
        max_height=max(0, int(args.max_height)),
        filename_template=args.filename_template,
        archive=args.archive,
        skip_existing=not args.no_skip_existing,
        cookies_file=args.cookies,
        rate_limit=args.rate_limit,
        quiet=args.quiet,
        verbose=args.verbose,
    )

    def announce(video: object) -> None:
        title = getattr(video, "title", "")
        duration = format_duration(getattr(video, "duration", 0))
        if not args.quiet:
            print(f"[fetch] {title} ({duration})", flush=True)

    fetched = fetch_videos(videos, config, on_progress=announce)

    downloaded = sum(1 for f in fetched if f.downloaded)
    if not args.quiet:
        print(
            f"[fetch] готово: скачано {downloaded}, "
            f"уже было {len(fetched) - downloaded}, "
            f"не получилось {len(videos) - len(fetched)}",
        )
    # A batch where nothing at all became available is a failure worth an exit code.
    return 0 if fetched or not videos else 1


if __name__ == "__main__":
    raise SystemExit(main())
