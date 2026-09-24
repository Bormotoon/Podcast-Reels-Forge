"""RU: Разбор ссылок YouTube и перечисление роликов.

Модуль ничего не скачивает — он отвечает на вопрос «какие ролики имеются в виду».
Скачиванием занимается :mod:`podcast_reels_forge.stages.fetch_stage`.

Источником данных служит YouTube Data API v3 (нужен только ключ, без OAuth). Если
ключа нет, перечисление делает сам yt-dlp — фича остаётся рабочей, просто без дат
публикации и точных длительностей до скачивания.

EN: YouTube link parsing and video listing.

This module never downloads anything — it answers "which videos are meant".
Downloading lives in :mod:`podcast_reels_forge.stages.fetch_stage`.

Data comes from the YouTube Data API v3 (an API key is enough, no OAuth). Without
a key, yt-dlp does the listing instead — the feature keeps working, only without
publish dates and exact durations known up front.
"""

from __future__ import annotations

import logging
import os
import re
import time
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from typing import Any, Final, Literal

import requests

log = logging.getLogger("Forge")

API_ROOT: Final = "https://www.googleapis.com/youtube/v3"

#: Page size accepted by playlistItems.list and videos.list.
API_PAGE_SIZE: Final = 50

#: Timeout for a single Data API request, seconds.
API_TIMEOUT: Final = 30

#: RU: Пауза между страницами, чтобы не долбить API очередью запросов подряд.
#: EN: Pause between pages so we do not hammer the API back to back.
API_PAGE_PAUSE: Final = 0.05

#: A video id is 11 characters of the URL-safe base64 alphabet. It may start with
#: a dash (`-3LisPanK24` is a real id on @pedobraz), which is why nothing here may
#: treat a leading `-` as "not an id".
_VIDEO_ID_RE: Final = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PLAYLIST_ID_RE: Final = re.compile(r"^[A-Za-z0-9_-]{2,}$")
_CHANNEL_ID_RE: Final = re.compile(r"^UC[A-Za-z0-9_-]{22}$")

_DURATION_RE: Final = re.compile(
    r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$",
)

SourceKind = Literal["video", "playlist", "channel", "handle", "username"]

_YOUTUBE_HOSTS: Final = frozenset({
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
})
_SHORT_HOSTS: Final = frozenset({"youtu.be", "www.youtu.be"})


@dataclass(frozen=True)
class YouTubeSource:
    """RU: Разобранная ссылка: что именно спросили.

    EN: A parsed link: what exactly was asked for.
    """

    kind: SourceKind
    #: Video id, playlist id, channel id, handle (without @) or legacy username.
    value: str
    #: The string the user actually typed, kept for error messages.
    raw: str

    def describe(self) -> str:
        """Human-readable one-liner for status output."""

        labels = {
            "video": "ролик",
            "playlist": "плейлист",
            "channel": "канал",
            "handle": "канал",
            "username": "канал",
        }
        return f"{labels[self.kind]} {self.value}"


@dataclass(frozen=True)
class YouTubeVideo:
    """RU: Ролик с метаданными, достаточными для отбора и именования.

    EN: A video with the metadata needed to filter and name it.
    """

    video_id: str
    title: str
    #: Publish date as YYYY-MM-DD, or "" when the listing could not provide it.
    upload_date: str
    #: Length in seconds; 0 when unknown (the yt-dlp fallback may not know it).
    duration: int
    channel_title: str = ""
    is_live: bool = False

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


class YouTubeError(RuntimeError):
    """RU: Ошибка обращения к YouTube. EN: A YouTube lookup failure."""


# --------------------------------------------------------------------- parsing


def _looks_like_video_id(text: str) -> bool:
    return bool(_VIDEO_ID_RE.match(text))


def parse_source(raw: str) -> YouTubeSource:
    """RU: Разбирает ссылку/идентификатор в описание источника.

    EN: Turn a link or bare identifier into a source description.

    Understands watch/shorts/live/embed URLs, ``youtu.be`` short links, playlist
    URLs, ``/@handle``, ``/channel/UC…``, legacy ``/c/`` and ``/user/`` paths, a
    bare ``@handle`` and a bare 11-character video id.

    A URL carrying both ``v=`` and ``list=`` resolves to the **video**: that is
    what "the link I copied from the player" means. Pass the ``playlist?list=…``
    URL when the whole playlist is wanted.
    """

    text = (raw or "").strip()
    if not text:
        raise YouTubeError("Пустой источник YouTube")

    # A bare handle is the friendliest thing to type: --youtube @pedobraz
    if text.startswith("@"):
        handle = text[1:].strip("/")
        if not handle:
            raise YouTubeError(f"Не разобрать источник YouTube: {raw!r}")
        return YouTubeSource(kind="handle", value=handle, raw=raw)

    if "://" not in text and "/" not in text:
        if _CHANNEL_ID_RE.match(text):
            return YouTubeSource(kind="channel", value=text, raw=raw)
        if text.upper().startswith(("PL", "UU", "LL", "FL", "OL", "RD")):
            return YouTubeSource(kind="playlist", value=text, raw=raw)
        if _looks_like_video_id(text):
            return YouTubeSource(kind="video", value=text, raw=raw)
        raise YouTubeError(f"Не разобрать источник YouTube: {raw!r}")

    parsed = urllib.parse.urlparse(text if "://" in text else f"https://{text}")
    host = (parsed.netloc or "").lower()
    if host not in _YOUTUBE_HOSTS and host not in _SHORT_HOSTS:
        raise YouTubeError(f"Это не ссылка на YouTube: {raw!r}")

    query = urllib.parse.parse_qs(parsed.query)
    parts = [p for p in (parsed.path or "").split("/") if p]

    # youtu.be/<id>
    if host in _SHORT_HOSTS:
        if parts and _looks_like_video_id(parts[0]):
            return YouTubeSource(kind="video", value=parts[0], raw=raw)
        raise YouTubeError(f"Не разобрать короткую ссылку YouTube: {raw!r}")

    # watch?v=<id> wins over an accompanying list=.
    video_id = (query.get("v") or [""])[0].strip()
    if video_id:
        if not _looks_like_video_id(video_id):
            raise YouTubeError(f"Некорректный id ролика в ссылке: {raw!r}")
        return YouTubeSource(kind="video", value=video_id, raw=raw)

    playlist_id = (query.get("list") or [""])[0].strip()
    if playlist_id and (not parts or parts[0] == "playlist"):
        if not _PLAYLIST_ID_RE.match(playlist_id):
            raise YouTubeError(f"Некорректный id плейлиста в ссылке: {raw!r}")
        return YouTubeSource(kind="playlist", value=playlist_id, raw=raw)

    if not parts:
        raise YouTubeError(f"Не разобрать источник YouTube: {raw!r}")

    head = parts[0]

    # /shorts/<id>, /live/<id>, /embed/<id>, /v/<id>
    if head in {"shorts", "live", "embed", "v"} and len(parts) > 1:
        if _looks_like_video_id(parts[1]):
            return YouTubeSource(kind="video", value=parts[1], raw=raw)
        raise YouTubeError(f"Некорректный id ролика в ссылке: {raw!r}")

    if head.startswith("@"):
        return YouTubeSource(kind="handle", value=head[1:], raw=raw)

    if head == "channel" and len(parts) > 1:
        return YouTubeSource(kind="channel", value=parts[1], raw=raw)

    if head in {"c", "user"} and len(parts) > 1:
        return YouTubeSource(kind="username", value=parts[1], raw=raw)

    # A playlist id can also ride on a channel page URL (/@name/playlists?list=…).
    if playlist_id:
        return YouTubeSource(kind="playlist", value=playlist_id, raw=raw)

    raise YouTubeError(f"Не разобрать источник YouTube: {raw!r}")


def iso8601_duration_seconds(value: str) -> int:
    """RU: ``PT1H2M3S`` → секунды. Мусор и пустая строка → 0.

    EN: Turn an ISO-8601 duration into seconds; unparseable input yields 0.
    """

    match = _DURATION_RE.match((value or "").strip())
    if not match:
        return 0
    days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _parse_date(value: str | date | None) -> date | None:
    """Accept ``YYYY-MM-DD`` (or a date object) and return a date."""

    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").replace(
            tzinfo=timezone.utc,
        ).date()
    except ValueError as exc:
        raise YouTubeError(
            f"Дата должна быть в формате ГГГГ-ММ-ДД, получено: {value!r}",
        ) from exc


# ------------------------------------------------------------------- Data API


def api_key_from_env(env_var: str = "YOUTUBE_API_KEY") -> str | None:
    """RU: Ключ Data API из окружения. EN: Data API key from the environment."""

    key = (os.environ.get(env_var) or "").strip()
    return key or None


def _http_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """RU: GET c разбором JSON и человеческим текстом ошибки.

    EN: GET with JSON parsing and a human-readable error message.
    """

    try:
        response = requests.get(url, headers=headers or {}, timeout=API_TIMEOUT)
    except requests.RequestException as exc:
        raise YouTubeError(f"YouTube API недоступен: {exc}") from exc

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.status_code >= 400:
        error = payload.get("error") if isinstance(payload, dict) else None
        reason = ""
        message = ""
        if isinstance(error, dict):
            message = str(error.get("message") or "")
            errors = error.get("errors")
            if isinstance(errors, list) and errors and isinstance(errors[0], dict):
                reason = str(errors[0].get("reason") or "")
        hint = {
            "quotaExceeded": (
                "Суточная квота YouTube API исчерпана — она обновляется в полночь "
                "по тихоокеанскому времени."
            ),
            "keyInvalid": "Ключ YOUTUBE_API_KEY недействителен.",
            "ipRefererBlocked": (
                "Ключ ограничен по IP/referer в Google Cloud Console; для запуска "
                "с сервера ограничение нужно снять."
            ),
            "accessNotConfigured": (
                "В проекте Google Cloud не включён YouTube Data API v3."
            ),
        }.get(reason, "")
        detail = " ".join(p for p in (message, hint) if p)
        raise YouTubeError(
            f"YouTube API ответил {response.status_code}"
            f"{f' ({reason})' if reason else ''}: {detail or 'без подробностей'}",
        )

    if not isinstance(payload, dict):
        raise YouTubeError("YouTube API вернул неожиданный ответ")
    return payload


def _api(endpoint: str, params: dict[str, Any], key: str) -> dict[str, Any]:
    """RU: Запрос к Data API. Ключ уходит заголовком, а не в URL.

    EN: A Data API request. The key travels as a header, not in the URL.

    ``?key=`` is the documented alternative, but under ``--verbose`` urllib3 logs
    every request line — and the key with it. A header keeps it out of the log,
    and out of any transcript the user pastes into a bug report.
    """

    query = urllib.parse.urlencode(
        {k: v for k, v in params.items() if v is not None},
    )
    return _http_json(
        f"{API_ROOT}/{endpoint}?{query}", headers={"X-goog-api-key": key},
    )


def _uploads_playlist(source: YouTubeSource, key: str) -> str:
    """RU: Возвращает id плейлиста «загрузки» канала.

    EN: Resolve a channel source to its uploads playlist id.

    ``channels.list`` costs 1 quota unit and answers exactly; ``search.list``
    costs 100 and returns nothing for a handle spelled with the leading ``@``, so
    it is only the last resort — and without the ``@``.
    """

    params: dict[str, Any]
    if source.kind == "channel":
        params = {"part": "contentDetails,snippet", "id": source.value}
    elif source.kind == "handle":
        params = {"part": "contentDetails,snippet", "forHandle": source.value}
    else:
        params = {"part": "contentDetails,snippet", "forUsername": source.value}

    data = _api("channels", params, key)
    items = data.get("items") or []

    if not items and source.kind in {"handle", "username"}:
        found = _api(
            "search",
            {"part": "snippet", "type": "channel", "q": source.value, "maxResults": 5},
            key,
        )
        for item in found.get("items") or []:
            channel_id = (item.get("id") or {}).get("channelId")
            if channel_id:
                data = _api(
                    "channels",
                    {"part": "contentDetails,snippet", "id": channel_id},
                    key,
                )
                items = data.get("items") or []
                break

    if not items:
        raise YouTubeError(f"Канал не найден: {source.raw!r}")

    related = (items[0].get("contentDetails") or {}).get("relatedPlaylists") or {}
    uploads = related.get("uploads")
    if not uploads:
        raise YouTubeError(f"У канала нет плейлиста загрузок: {source.raw!r}")
    return str(uploads)


def _playlist_video_ids(playlist_id: str, key: str, *, cap: int = 0) -> list[str]:
    """RU: Все id роликов плейлиста, от новых к старым.

    EN: Every video id in a playlist, newest first.
    """

    ids: list[str] = []
    token: str | None = None
    while True:
        data = _api(
            "playlistItems",
            {
                "part": "contentDetails",
                "playlistId": playlist_id,
                "maxResults": API_PAGE_SIZE,
                "pageToken": token,
            },
            key,
        )
        for item in data.get("items") or []:
            video_id = (item.get("contentDetails") or {}).get("videoId")
            if video_id:
                ids.append(str(video_id))
        token = data.get("nextPageToken")
        if not token or (cap and len(ids) >= cap):
            break
        time.sleep(API_PAGE_PAUSE)
    return ids[:cap] if cap else ids


def _video_details(video_ids: list[str], key: str) -> list[YouTubeVideo]:
    """RU: Метаданные роликов пачками по 50, порядок входа сохраняется.

    EN: Video metadata in batches of 50, preserving the input order.

    ``videos.list`` rather than the playlist item's own snippet: for a hand-made
    playlist ``playlistItems.snippet.publishedAt`` is when the video was *added*
    to it, not when it was published. Durations and the live/private flags only
    exist here anyway.
    """

    by_id: dict[str, YouTubeVideo] = {}
    for start in range(0, len(video_ids), API_PAGE_SIZE):
        chunk = video_ids[start : start + API_PAGE_SIZE]
        data = _api(
            "videos",
            {
                "part": "snippet,contentDetails,status",
                "id": ",".join(chunk),
                "maxResults": API_PAGE_SIZE,
            },
            key,
        )
        for item in data.get("items") or []:
            snippet = item.get("snippet") or {}
            details = item.get("contentDetails") or {}
            status = item.get("status") or {}
            if str(status.get("privacyStatus") or "") == "private":
                continue
            video_id = str(item.get("id") or "")
            if not video_id:
                continue
            published = str(snippet.get("publishedAt") or "")
            by_id[video_id] = YouTubeVideo(
                video_id=video_id,
                title=str(snippet.get("title") or video_id),
                upload_date=published[:10],
                duration=iso8601_duration_seconds(str(details.get("duration") or "")),
                channel_title=str(snippet.get("channelTitle") or ""),
                is_live=str(snippet.get("liveBroadcastContent") or "none") != "none",
            )
        if start + API_PAGE_SIZE < len(video_ids):
            time.sleep(API_PAGE_PAUSE)

    return [by_id[v] for v in video_ids if v in by_id]


# --------------------------------------------------------------- yt-dlp fallback


def _list_with_yt_dlp(source: YouTubeSource, *, cap: int = 0) -> list[YouTubeVideo]:
    """RU: Перечисление без ключа API — силами самого yt-dlp.

    EN: Key-free listing, done by yt-dlp itself.

    Flat extraction is cheap but thin: a channel listing carries ids, titles and
    usually durations, but rarely a publish date. Names then fall back to the
    template's ``NA`` for the date, which yt-dlp fills in during download.
    """

    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise YouTubeError(
            "Нужен либо ключ YOUTUBE_API_KEY, либо установленный yt-dlp "
            "(pip install -U yt-dlp)",
        ) from exc

    if source.kind == "video":
        target = f"https://www.youtube.com/watch?v={source.value}"
    elif source.kind == "playlist":
        target = f"https://www.youtube.com/playlist?list={source.value}"
    elif source.kind == "channel":
        target = f"https://www.youtube.com/channel/{source.value}/videos"
    elif source.kind == "handle":
        target = f"https://www.youtube.com/@{source.value}/videos"
    else:
        target = f"https://www.youtube.com/user/{source.value}/videos"

    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
    }
    if cap:
        options["playlistend"] = cap

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(target, download=False)
    except Exception as exc:  # yt-dlp raises its own hierarchy
        raise YouTubeError(f"yt-dlp не смог прочитать {source.raw!r}: {exc}") from exc

    entries = info.get("entries") if isinstance(info, dict) else None
    raw_items = list(entries) if entries else ([info] if isinstance(info, dict) else [])

    out: list[YouTubeVideo] = []
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        video_id = str(entry.get("id") or "")
        if not _looks_like_video_id(video_id):
            continue
        upload_date = str(entry.get("upload_date") or "")
        if len(upload_date) == 8 and upload_date.isdigit():
            upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
        else:
            upload_date = ""
        duration_raw = entry.get("duration")
        out.append(
            YouTubeVideo(
                video_id=video_id,
                title=str(entry.get("title") or video_id),
                upload_date=upload_date,
                duration=int(duration_raw) if isinstance(duration_raw, (int, float)) else 0,
                channel_title=str(entry.get("channel") or entry.get("uploader") or ""),
                is_live=bool(entry.get("is_live")),
            ),
        )
    return out[:cap] if cap else out


# --------------------------------------------------------------------- filters


@dataclass(frozen=True)
class VideoFilters:
    """RU: Отбор роликов до скачивания. EN: Pre-download video selection."""

    #: Keep at most this many, counting from the newest. 0 = no cap.
    limit: int = 0
    #: Published on or after / on or before, as YYYY-MM-DD.
    since: str | None = None
    until: str | None = None
    #: Length bounds in seconds. min_duration=60 drops Shorts. 0 disables.
    min_duration: int = 0
    max_duration: int = 0
    #: Drop live streams and premiere announcements.
    skip_live: bool = True
    #: Video ids to drop outright, whatever else matches. Built by
    #: :func:`resolve_excluded_ids` from the configured exclusions.
    exclude_ids: frozenset[str] = frozenset()


def apply_filters(
    videos: list[YouTubeVideo], filters: VideoFilters,
) -> list[YouTubeVideo]:
    """RU: Применяет отбор; лимит считается последним, уже по отфильтрованным.

    EN: Apply the filters; the cap is applied last, over what survived.
    """

    since = _parse_date(filters.since)
    until = _parse_date(filters.until)

    kept: list[YouTubeVideo] = []
    for video in videos:
        # RU: Исключения — раньше всех прочих условий и раньше лимита: с
        #     `--yt-limit 5` нужны 5 подходящих роликов, а не «5 новых, из
        #     которых часть выпала».
        # EN: Exclusions come before every other condition and before the cap:
        #     `--yt-limit 5` must yield 5 usable videos, not "the newest 5, some
        #     of which then dropped out".
        if video.video_id in filters.exclude_ids:
            continue
        if filters.skip_live and video.is_live:
            continue
        # A duration of 0 means "unknown", not "zero seconds": the yt-dlp fallback
        # often has no length before download, and dropping those would silently
        # empty a key-free channel run.
        if video.duration:
            if filters.min_duration and video.duration < filters.min_duration:
                continue
            if filters.max_duration and video.duration > filters.max_duration:
                continue
        if video.upload_date:
            published = _parse_date(video.upload_date)
            if published is not None:
                if since is not None and published < since:
                    continue
                if until is not None and published > until:
                    continue
        kept.append(video)

    return kept[: filters.limit] if filters.limit else kept


# ----------------------------------------------------------------- public API


def list_videos(
    source: YouTubeSource,
    *,
    api_key: str | None = None,
    filters: VideoFilters | None = None,
) -> list[YouTubeVideo]:
    """RU: Разворачивает источник в список роликов, от новых к старым.

    EN: Expand a source into its videos, newest first.
    """

    filters = filters or VideoFilters()

    if not api_key:
        videos = _list_with_yt_dlp(source)
        return apply_filters(videos, filters)

    if source.kind == "video":
        video_ids = [source.value]
    else:
        playlist_id = (
            source.value
            if source.kind == "playlist"
            else _uploads_playlist(source, api_key)
        )
        # Fetching more ids than the cap costs nothing extra (a page is 50 either
        # way), but filters can drop items, so we cannot page-limit by `limit`.
        video_ids = _playlist_video_ids(playlist_id, api_key)

    if not video_ids:
        return []
    return apply_filters(_video_details(video_ids, api_key), filters)


def resolve_excluded_ids(
    raw_sources: Sequence[str], *, api_key: str | None = None,
) -> set[str]:
    """RU: Разворачивает исключения в набор id роликов.

    EN: Expand the exclusions into a set of video ids.

    Takes the same source shapes as :func:`resolve_sources` — a playlist is the
    useful one: "everything on the channel except this show" stays correct as
    episodes are added to it, with no list to maintain by hand.

    Only ids are needed, so the per-video ``videos.list`` call is skipped: a
    playlist of any size costs one quota unit per 50 items.
    """

    excluded: set[str] = set()
    for raw in raw_sources:
        source = parse_source(raw)
        log.debug("YouTube exclusion %r -> %s", raw, source.describe())
        if source.kind == "video":
            excluded.add(source.value)
            continue
        if api_key:
            playlist_id = (
                source.value
                if source.kind == "playlist"
                else _uploads_playlist(source, api_key)
            )
            excluded.update(_playlist_video_ids(playlist_id, api_key))
        else:
            excluded.update(v.video_id for v in _list_with_yt_dlp(source))
    return excluded


def resolve_sources(
    raw_sources: list[str],
    *,
    api_key: str | None = None,
    filters: VideoFilters | None = None,
    exclude: Sequence[str] | None = None,
) -> list[YouTubeVideo]:
    """RU: Разбирает и разворачивает несколько источников, убирая дубликаты.

    EN: Parse and expand several sources, de-duplicating by video id.

    Order is preserved: the first source to name a video decides its position,
    so ``--youtube <link> --youtube @channel`` puts the single link first.

    ``exclude`` names sources whose videos must never be taken — typically a
    playlist. It also overrides a directly named link: an excluded video stays
    excluded however it was reached.
    """

    if exclude:
        excluded = resolve_excluded_ids(exclude, api_key=api_key)
        if excluded:
            log.debug("YouTube exclusions cover %d video(s)", len(excluded))
        filters = replace(
            filters or VideoFilters(),
            exclude_ids=frozenset(excluded) | (filters.exclude_ids if filters else frozenset()),
        )

    seen: set[str] = set()
    out: list[YouTubeVideo] = []
    for raw in raw_sources:
        source = parse_source(raw)
        log.debug("YouTube source %r -> %s", raw, source.describe())
        for video in list_videos(source, api_key=api_key, filters=filters):
            if video.video_id in seen:
                continue
            seen.add(video.video_id)
            out.append(video)
    return out


def format_duration(total: int) -> str:
    """RU: Секунды в ``1:02:03`` / ``2:03``. EN: Seconds as ``1:02:03`` / ``2:03``."""

    if total <= 0:
        return "--:--"
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"
