"""RU: Стадия `fetch` — забирает ролики с YouTube в папку загрузок.

Стадия кладёт файл с предсказуемым именем в `youtube.download_dir`, и на этом её
работа заканчивается: дальше файл ничем не отличается от того, что положили руками,
и его подхватывает обычный обход входной папки.

Скачивает yt-dlp. Зависимость опциональная — без YouTube пайплайн в ней не нуждается,
поэтому импорт ленивый, а отсутствие пакета даёт понятную инструкцию, а не трейсбек.

EN: The `fetch` stage — pull YouTube videos into the download folder.

The stage drops a predictably named file into `youtube.download_dir` and its job
ends there: from that point the file is indistinguishable from one placed by hand,
and the ordinary input-folder scan picks it up.

Downloading is yt-dlp's job. That dependency is optional — the pipeline does not
need it without YouTube — so the import is lazy and a missing package yields an
instruction rather than a traceback.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from podcast_reels_forge.sources.youtube import (
    YouTubeError,
    YouTubeVideo,
    format_duration,
)

log = logging.getLogger("Forge")

#: RU: Имя = дата, заголовок, id. Дата ставит папки эпизодов в хронологический
#:     порядок, id делает имя уникальным и позволяет узнать уже скачанный ролик
#:     даже после переименования на YouTube.
#:     `.150B` — ограничение в БАЙТАХ, а не символах: кириллица весит два байта,
#:     а предел имени файла — 255. Дата (13) + id (16) + расширение оставляют
#:     запас под суффиксы вроде `.proofread.json` в имени выходной папки.
#: EN: Name = date, title, id. The date sorts episode folders chronologically; the
#:     id keeps names unique and lets us recognise an already-downloaded video even
#:     after it was retitled on YouTube.
#:     `.150B` caps BYTES, not characters: Cyrillic costs two bytes each and the
#:     filename limit is 255. Date (13) + id (16) + extension leave room for
#:     suffixes like `.proofread.json` on the output folder name.
DEFAULT_FILENAME_TEMPLATE = "%(upload_date>%Y-%m-%d)s - %(title).150B [%(id)s].%(ext)s"

#: RU: Аудио берём как есть (m4a), не перегоняя в MP3: это было бы второе сжатие
#:     с потерями. MP3 320k и WAV 16 кГц соберёт `_ensure_audio_companions()`.
#: EN: Audio is kept as delivered (m4a) rather than transcoded to MP3 — that would
#:     be a second lossy pass. `_ensure_audio_companions()` builds the 320k MP3 and
#:     the 16 kHz WAV from it.
AUDIO_FORMAT = "ba[ext=m4a]/ba/b"

#: RU: Контейнер аудио-загрузки фиксируем. YouTube почти всегда отдаёт m4a
#:     (формат 140), и тогда постпроцессор ничего не перекодирует — просто
#:     подтверждает формат. Но запасная ветка `/ba` могла принести opus в
#:     контейнере .webm, а .webm числится видео: эпизод попал бы в очередь как
#:     видеофайл, и стадия нарезки честно попыталась бы его резать.
#: EN: Pin the container of an audio download. YouTube almost always serves m4a
#:     (format 140), in which case this postprocessor transcodes nothing and just
#:     confirms the format. The `/ba` fallback, though, could hand back opus in a
#:     .webm container — and .webm counts as video, so the episode would enter the
#:     queue as a video file and the cut stage would dutifully try to cut it.
AUDIO_POSTPROCESSORS = [
    {"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "0"},
]

#: RU: Набор клиентов для повторной попытки. На части старых роликов YouTube
#:     перестаёт отдавать форматы клиентам yt-dlp по умолчанию — ролик выглядит
#:     как «This video is not available», хотя API отдаёт его публичным и без
#:     региональных ограничений. Другие клиенты его при этом видят.
#:     Применяется ТОЛЬКО после неудачи: навязывать этот набор всем подряд —
#:     менять рабочее на неизвестное (у клиента `tv`, например, часть форматов
#:     приходит под DRM).
#: EN: Client set for a second attempt. For some older videos YouTube stops
#:     serving formats to yt-dlp's default clients — the video reads as "This
#:     video is not available" even though the API reports it public with no
#:     region restriction. Other clients still see it.
#:     Used ONLY after a failure: forcing this set on every download would trade
#:     something that works for something unknown (the `tv` client, for one,
#:     returns some formats DRM-protected).
FALLBACK_PLAYER_CLIENTS: tuple[str, ...] = ("web_safari", "tv", "android")

#: Extensions a finished download can carry, for the "already here?" lookup.
MEDIA_SUFFIXES = frozenset({
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
    ".m4a", ".mp3", ".opus", ".aac", ".ogg", ".wav",
})

#: yt-dlp leaves these behind mid-download; they must not count as "already here".
PARTIAL_SUFFIXES = frozenset({".part", ".ytdl", ".temp"})

#: RU: Отдельные дорожки до склейки: `… [id].f137.mp4` (видео без звука) и
#:     `… [id].f140.m4a`. Если загрузку прервали до склейки, они остаются на
#:     диске и выглядят как готовые файлы.
#: EN: Per-format pieces before merging: `… [id].f137.mp4` (video, no audio)
#:     and `… [id].f140.m4a`. An interrupted download leaves them on disk
#:     looking like finished files.
_FRAGMENT_STEM_RE = re.compile(r"\.f\d+(?:-\d+)?$")


def is_download_fragment(path: Path) -> bool:
    """RU: Кусок незавершённой загрузки yt-dlp. EN: A piece of an unfinished yt-dlp download."""

    return path.suffix.lower() in PARTIAL_SUFFIXES or bool(_FRAGMENT_STEM_RE.search(path.stem))


@dataclass(frozen=True)
class FetchConfig:
    """RU: Настройки стадии загрузки. EN: Download-stage settings."""

    download_dir: Path
    #: True downloads video+audio, False only the audio track.
    want_video: bool = True
    #: Height ceiling for the video track. 0 = whatever is best.
    max_height: int = 1080
    filename_template: str = DEFAULT_FILENAME_TEMPLATE
    #: yt-dlp download archive: ids already taken, so a channel re-run is
    #: incremental. None disables it.
    archive: Path | None = None
    #: False re-downloads even when the file is already on disk.
    skip_existing: bool = True
    #: Netscape cookie jar, for age-gated or members-only material.
    cookies_file: Path | None = None
    #: Keep yt-dlp's own metadata sidecar (title, description, chapters).
    write_info_json: bool = True
    retries: int = 3
    #: Bandwidth ceiling, e.g. "5M". None = unlimited.
    rate_limit: str | None = None
    extra_options: dict[str, Any] = field(default_factory=dict)
    quiet: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class FetchedVideo:
    """RU: Результат по одному ролику. EN: The outcome for a single video."""

    video: YouTubeVideo
    path: Path
    #: False when the file was already on disk and nothing was downloaded.
    downloaded: bool

    @property
    def stem(self) -> str:
        """Filename stem — the pipeline's identity key for an episode."""

        return self.path.stem


def _import_yt_dlp() -> Any:
    """RU: Ленивый импорт yt-dlp с инструкцией вместо трейсбека.

    EN: Lazy yt-dlp import, with an instruction instead of a traceback.
    """

    try:
        import yt_dlp
    except ImportError as exc:
        raise SystemExit(
            "Для работы с YouTube нужен yt-dlp. Установите его в то же окружение:\n"
            "    ./whisper-env/bin/pip install -U yt-dlp\n"
            "YouTube часто меняет отдачу видео, поэтому обновлять его стоит регулярно.",
        ) from exc
    return yt_dlp


def find_local_copy(download_dir: Path, video_id: str) -> Path | None:
    """RU: Ищет уже скачанный ролик по id в имени файла.

    EN: Look for an already-downloaded file carrying this id in its name.

    The id is the identity key, not the title: a video retitled on YouTube must
    still be recognised as the one we already have. Undersized and partial files
    do not count — a download interrupted halfway must be retried, not adopted.
    """

    if not download_dir.exists():
        return None
    marker = f"[{video_id}]"
    for path in sorted(download_dir.iterdir()):
        if not path.is_file() or marker not in path.name:
            continue
        if is_download_fragment(path):
            # A merge that never happened: let yt-dlp resume it.
            continue
        if path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        try:
            if path.stat().st_size <= 0:
                continue
        except OSError:
            continue
        return path
    return None


def build_format(config: FetchConfig) -> str:
    """RU: Строка выбора формата для yt-dlp.

    EN: The yt-dlp format selector.

    Video is capped by height because the pipeline outputs 1080-wide 9:16 clips —
    a 4K source only costs disk and bandwidth. Each fallback drops one constraint
    so an episode without the ideal rendition still downloads.
    """

    if not config.want_video:
        return AUDIO_FORMAT
    if config.max_height > 0:
        cap = f"[height<={config.max_height}]"
        return (
            f"bv*{cap}[ext=mp4]+ba[ext=m4a]/"
            f"bv*{cap}+ba/"
            f"b{cap}/"
            f"bv*+ba/b"
        )
    return "bv*[ext=mp4]+ba[ext=m4a]/bv*+ba/b"


def build_ydl_options(config: FetchConfig) -> dict[str, Any]:
    """RU: Опции yt-dlp из настроек стадии. EN: yt-dlp options from the config."""

    options: dict[str, Any] = {
        # The folder lives in the template, and only there: setting `paths.home`
        # as well makes yt-dlp join the two, and files land in
        # input/youtube/input/youtube/.
        "outtmpl": str(config.download_dir / config.filename_template),
        "format": build_format(config),
        "retries": config.retries,
        "ignoreerrors": False,
        "noplaylist": True,
        "quiet": config.quiet or not config.verbose,
        "no_warnings": config.quiet,
        "consoletitle": False,
        "writeinfojson": config.write_info_json,
        # RU: Без этого метаданные пишутся и для «плейлиста из одного ролика».
        # EN: Without this, a one-video "playlist" gets its own metadata file.
        "writedescription": False,
    }
    if config.want_video:
        options["merge_output_format"] = "mp4"
    else:
        options["postprocessors"] = [dict(pp) for pp in AUDIO_POSTPROCESSORS]
    if config.archive is not None:
        options["download_archive"] = str(config.archive)
    if config.cookies_file is not None:
        options["cookiefile"] = str(config.cookies_file)
    if config.rate_limit:
        options["ratelimit"] = _parse_rate_limit(config.rate_limit)
    options.update(config.extra_options)
    return options


def _parse_rate_limit(value: str) -> int | None:
    """RU: ``5M`` → байты в секунду. EN: ``5M`` → bytes per second."""

    text = str(value).strip().upper().rstrip("B")
    if not text:
        return None
    multiplier = 1
    if text[-1] in {"K", "M", "G"}:
        multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3}[text[-1]]
        text = text[:-1]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        log.warning("Не разобрать youtube.rate_limit=%r, ограничение снято", value)
        return None


def _downloaded_path(info: dict[str, Any]) -> Path | None:
    """RU: Достаёт итоговый путь из ответа yt-dlp.

    EN: Pull the final file path out of yt-dlp's info dict.

    ``requested_downloads`` holds the post-processed result (after merging), which
    is the file we want; ``_filename`` is the pre-merge name and only a fallback.
    """

    downloads = info.get("requested_downloads")
    if isinstance(downloads, list) and downloads:
        first = downloads[0]
        if isinstance(first, dict):
            for key in ("filepath", "_filename", "filename"):
                value = first.get(key)
                if value:
                    return Path(str(value))
    for key in ("filepath", "_filename"):
        value = info.get(key)
        if value:
            return Path(str(value))
    return None


def _retry_with_fallback_clients(
    yt_dlp: Any, options: dict[str, Any], video: YouTubeVideo,
) -> dict[str, Any] | None:
    """RU: Вторая попытка тем же запросом, но другими клиентами YouTube.

    EN: A second attempt at the same request through other YouTube clients.

    Returns the info dict on success, ``None`` when this one is genuinely out of
    reach — blocked by a rights holder or region-locked, which no client choice
    can undo.
    """

    retry_options = dict(options)
    retry_options["extractor_args"] = {
        "youtube": {"player_client": list(FALLBACK_PLAYER_CLIENTS)},
    }
    try:
        with yt_dlp.YoutubeDL(retry_options) as ydl:
            info = ydl.extract_info(video.url, download=True)
    except Exception as exc:
        log.error(
            "Ролик недоступен и через запасные клиенты — %s (%s): %s",
            video.title, video.video_id, exc,
        )
        return None
    return info if isinstance(info, dict) else None


def fetch_videos(
    videos: list[YouTubeVideo],
    config: FetchConfig,
    *,
    on_progress: Any = None,
) -> list[FetchedVideo]:
    """RU: Скачивает ролики, пропуская уже лежащие на диске.

    EN: Download the videos, skipping the ones already on disk.

    Returns one entry per video that ended up available locally. A video that
    fails is logged and skipped rather than aborting the batch: a channel run of
    seventy episodes must not die on one age-gated item.
    """

    if not videos:
        return []

    config.download_dir.mkdir(parents=True, exist_ok=True)
    if config.archive is not None:
        config.archive.parent.mkdir(parents=True, exist_ok=True)

    # on_progress fires once per video, skipped ones included: it reports "item
    # N of M handled", so a progress bar over it still reaches its total when
    # everything was already downloaded.
    def announce(video: YouTubeVideo) -> None:
        if callable(on_progress):
            on_progress(video)

    pending: list[YouTubeVideo] = []
    results: list[FetchedVideo] = []
    for video in videos:
        local = (
            find_local_copy(config.download_dir, video.video_id)
            if config.skip_existing
            else None
        )
        if local is not None:
            log.debug("YouTube %s уже скачан: %s", video.video_id, local.name)
            results.append(FetchedVideo(video=video, path=local, downloaded=False))
            announce(video)
        else:
            pending.append(video)

    if not pending:
        return results

    yt_dlp = _import_yt_dlp()
    options = build_ydl_options(config)
    if not config.skip_existing:
        # RU: --no-skip-existing обходит и файлы, и журнал: иначе «перезапустить
        #     всё заново» молча ничего бы не перекачало.
        # EN: --no-skip-existing bypasses both files and the archive; otherwise
        #     "re-run everything" would quietly download nothing.
        options.pop("download_archive", None)
        options["overwrites"] = True

    # A user-supplied client choice is respected as final: retrying behind their
    # back would make the option they set look like it did nothing.
    may_retry = "extractor_args" not in config.extra_options

    with yt_dlp.YoutubeDL(options) as ydl:
        for video in pending:
            announce(video)
            try:
                info = ydl.extract_info(video.url, download=True)
            except Exception as exc:  # yt-dlp raises its own hierarchy
                if not may_retry:
                    log.error(
                        "Не удалось скачать %s (%s): %s",
                        video.title, video.video_id, exc,
                    )
                    continue
                log.warning(
                    "Не удалось скачать %s (%s): %s. Пробую другие клиенты: %s",
                    video.title, video.video_id, exc,
                    ", ".join(FALLBACK_PLAYER_CLIENTS),
                )
                info = _retry_with_fallback_clients(yt_dlp, options, video)
            if not isinstance(info, dict):
                log.error("yt-dlp не вернул метаданные для %s", video.video_id)
                continue

            path = _downloaded_path(info)
            if path is None or not path.exists():
                # The archive can make yt-dlp report success without producing a
                # file (it was taken on an earlier run); fall back to the folder.
                path = find_local_copy(config.download_dir, video.video_id)
            if path is None or not path.exists():
                log.error(
                    "После загрузки %s файл не найден в %s",
                    video.video_id, config.download_dir,
                )
                continue
            results.append(FetchedVideo(video=video, path=path, downloaded=True))

    return results


def locate_existing(
    videos: list[YouTubeVideo], download_dir: Path,
) -> list[FetchedVideo]:
    """RU: Находит уже скачанное, ничего не качая.

    EN: Find what is already downloaded, without downloading anything.

    Used when the `fetch` stage is not selected but the run still has to be
    narrowed to the videos named on the command line.
    """

    out: list[FetchedVideo] = []
    for video in videos:
        path = find_local_copy(download_dir, video.video_id)
        if path is not None:
            out.append(FetchedVideo(video=video, path=path, downloaded=False))
    return out


def describe_videos(videos: list[YouTubeVideo], download_dir: Path) -> str:
    """RU: Таблица «что будет взято» для --yt-list.

    EN: The "what would be taken" table behind --yt-list.
    """

    if not videos:
        return "Под условия отбора не попал ни один ролик."

    rows: list[str] = []
    already = 0
    for index, video in enumerate(videos, start=1):
        local = find_local_copy(download_dir, video.video_id)
        if local is not None:
            already += 1
        rows.append(
            f"{index:>3}. {video.video_id}  "
            f"{video.upload_date or '----------'}  "
            f"{format_duration(video.duration):>8}  "
            f"{'есть' if local is not None else '  — '}  "
            f"{video.title}",
        )

    header = (
        f"Роликов: {len(videos)}"
        f"{f', уже скачано: {already}' if already else ''}\n"
        f"{'  №':>3}  {'id':<11}  {'дата':<10}  {'длит.':>8}  файл  заголовок"
    )
    return header + "\n" + "\n".join(rows)


def resolve_download_dir(conf: dict[str, Any], input_dir: Path) -> Path:
    """RU: Папка загрузок из конфига; по умолчанию — подпапка входной.

    EN: Download folder from config; defaults to a sub-folder of the input dir.
    """

    raw = str((conf or {}).get("download_dir") or "").strip()
    if raw:
        return Path(raw)
    return input_dir / "youtube"


def want_video_for_stages(
    conf: dict[str, Any], active_stages: Collection[str],
) -> bool:
    """RU: Решает, нужна ли видеодорожка. EN: Decide whether video is needed.

    ``auto`` downloads the video only when the cut stage will actually use it —
    a whole-channel run without cutting is then a fraction of the bytes. Under
    ``--only fetch`` the intent is unknown, so ``auto`` keeps the video: guessing
    wrong there costs a re-download of the entire channel.
    """

    mode = str((conf or {}).get("download") or "auto").strip().lower()
    if mode == "video":
        return True
    if mode == "audio":
        return False
    if mode != "auto":
        raise YouTubeError(
            f"youtube.download должен быть auto, video или audio; получено {mode!r}",
        )
    active = set(active_stages)
    return "cut" in active or active == {"fetch"}


def maybe_update_yt_dlp(conf: dict[str, Any], stamp: Path, *, now: float | None = None) -> bool:
    """RU: Раз в N дней обновить yt-dlp (youtube.self_update).

    EN: Update yt-dlp every N days when ``youtube.self_update`` is on.

    YouTube breaks old yt-dlp releases every few weeks, and an unattended
    nightly run has nobody to type ``pip install -U yt-dlp``. The stamp file
    keeps it to one attempt per period; a failure is logged and the run goes
    on with the version it has. Returns True when an update ran successfully.
    """

    if not bool((conf or {}).get("self_update", False)):
        return False
    try:
        days = float((conf or {}).get("self_update_days", 7))
    except (TypeError, ValueError):
        days = 7.0
    current = time.time() if now is None else now
    try:
        if current - stamp.stat().st_mtime < days * 86400:
            return False
    except OSError:
        pass
    log.info("youtube.self_update: updating yt-dlp")
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "--quiet", "yt-dlp"],
            capture_output=True, text=True, timeout=300, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("yt-dlp update failed: %s", exc)
        return False
    # Stamp even on failure: retrying pip every run would only add noise.
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.touch()
    if res.returncode != 0:
        log.warning("yt-dlp update failed: %s", (res.stderr or res.stdout or "").strip()[-300:])
        return False
    return True
