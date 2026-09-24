"""Tests for YouTube link parsing and Data API listing.

Nothing here touches the network: every API response is stubbed, so the tests
pin down our request shape and filtering rather than Google's uptime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from podcast_reels_forge.sources import youtube

if TYPE_CHECKING:
    MonkeyPatch = pytest.MonkeyPatch


# --------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("raw", "kind", "value"),
    [
        ("https://www.youtube.com/watch?v=D6WjXRJt1DA", "video", "D6WjXRJt1DA"),
        ("https://youtu.be/D6WjXRJt1DA", "video", "D6WjXRJt1DA"),
        ("https://m.youtube.com/watch?v=D6WjXRJt1DA", "video", "D6WjXRJt1DA"),
        ("https://www.youtube.com/shorts/D6WjXRJt1DA", "video", "D6WjXRJt1DA"),
        ("https://www.youtube.com/live/D6WjXRJt1DA", "video", "D6WjXRJt1DA"),
        ("https://www.youtube.com/embed/D6WjXRJt1DA", "video", "D6WjXRJt1DA"),
        ("D6WjXRJt1DA", "video", "D6WjXRJt1DA"),
        ("https://www.youtube.com/playlist?list=PLabcdef", "playlist", "PLabcdef"),
        ("PLabcdefghij", "playlist", "PLabcdefghij"),
        ("https://www.youtube.com/@pedobraz", "handle", "pedobraz"),
        ("https://www.youtube.com/@pedobraz/videos", "handle", "pedobraz"),
        ("@pedobraz", "handle", "pedobraz"),
        (
            "https://www.youtube.com/channel/UCNKtGG3Ys_8t8Qcqpjll4CA",
            "channel",
            "UCNKtGG3Ys_8t8Qcqpjll4CA",
        ),
        ("UCNKtGG3Ys_8t8Qcqpjll4CA", "channel", "UCNKtGG3Ys_8t8Qcqpjll4CA"),
        ("https://www.youtube.com/c/PedObraz/videos", "username", "PedObraz"),
        ("https://www.youtube.com/user/PedObraz", "username", "PedObraz"),
    ],
)
def test_parse_source_understands_every_link_shape(
    raw: str, kind: str, value: str,
) -> None:
    """Every way a user might name a video, playlist or channel."""
    source = youtube.parse_source(raw)
    assert source.kind == kind
    assert source.value == value


def test_parse_source_handles_ids_starting_with_a_dash() -> None:
    """A leading dash is a legal id character, not a malformed link.

    `-3LisPanK24` is a real video on the channel this was built for; treating the
    dash as "obviously not an id" would drop it from every channel run.
    """
    assert youtube.parse_source("-3LisPanK24").value == "-3LisPanK24"
    assert youtube.parse_source("https://youtu.be/-3LisPanK24").value == "-3LisPanK24"
    assert youtube.parse_source(
        "https://www.youtube.com/watch?v=-3LisPanK24",
    ).value == "-3LisPanK24"


def test_parse_source_prefers_the_video_over_its_playlist() -> None:
    """A link copied from the player carries both; the video is what was meant."""
    source = youtube.parse_source(
        "https://www.youtube.com/watch?v=D6WjXRJt1DA&list=PLabcdef&index=3",
    )
    assert source.kind == "video"
    assert source.value == "D6WjXRJt1DA"


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "https://vimeo.com/12345", "https://www.youtube.com/", "not a link"],
)
def test_parse_source_rejects_what_it_cannot_resolve(raw: str) -> None:
    """Silence would send an empty queue downstream; these must raise."""
    with pytest.raises(youtube.YouTubeError):
        youtube.parse_source(raw)


@pytest.mark.parametrize(
    ("iso", "seconds"),
    [
        ("PT1H2M3S", 3723),
        ("PT45S", 45),
        ("PT2M", 120),
        ("PT1H", 3600),
        ("P1DT2H", 93600),
        ("", 0),
        ("junk", 0),
        ("PT", 0),
    ],
)
def test_iso8601_duration_seconds(iso: str, seconds: int) -> None:
    assert youtube.iso8601_duration_seconds(iso) == seconds


def test_format_duration() -> None:
    assert youtube.format_duration(3836) == "1:03:56"
    assert youtube.format_duration(125) == "2:05"
    assert youtube.format_duration(0) == "--:--"


# ------------------------------------------------------------------ Data API


def _video_item(
    video_id: str,
    *,
    title: str = "Заголовок",
    published: str = "2026-02-08T10:00:00Z",
    duration: str = "PT1H2M3S",
    live: str = "none",
    privacy: str = "public",
) -> dict[str, Any]:
    return {
        "id": video_id,
        "snippet": {
            "title": title,
            "publishedAt": published,
            "channelTitle": "ПедОбраз",
            "liveBroadcastContent": live,
        },
        "contentDetails": {"duration": duration},
        "status": {"privacyStatus": privacy},
    }


class FakeApi:
    """Records every Data API call and replays canned responses."""

    def __init__(self, responses: dict[str, list[dict[str, Any]]]):
        self.responses = {k: list(v) for k, v in responses.items()}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, endpoint: str, params: dict[str, Any], key: str) -> dict[str, Any]:
        self.calls.append((endpoint, dict(params)))
        queue = self.responses.get(endpoint)
        if not queue:
            raise AssertionError(f"unexpected call to {endpoint}: {params}")
        return queue.pop(0)

    def params_for(self, endpoint: str) -> list[dict[str, Any]]:
        return [p for e, p in self.calls if e == endpoint]


def test_list_videos_for_a_single_video_skips_the_playlist_lookup(
    monkeypatch: MonkeyPatch,
) -> None:
    """One link is one videos.list call — no channel or playlist round-trip."""
    fake = FakeApi({"videos": [{"items": [_video_item("D6WjXRJt1DA")]}]})
    monkeypatch.setattr(youtube, "_api", fake)

    videos = youtube.list_videos(
        youtube.parse_source("https://youtu.be/D6WjXRJt1DA"), api_key="k",
    )

    assert [v.video_id for v in videos] == ["D6WjXRJt1DA"]
    assert videos[0].upload_date == "2026-02-08"
    assert videos[0].duration == 3723
    assert [endpoint for endpoint, _ in fake.calls] == ["videos"]


def test_list_videos_for_a_channel_pages_through_the_uploads_playlist(
    monkeypatch: MonkeyPatch,
) -> None:
    """A handle resolves to uploads, and every page of it is walked."""
    page_one = {
        "items": [{"contentDetails": {"videoId": f"vid{i:08d}00"}} for i in range(3)],
        "nextPageToken": "TOKEN2",
    }
    page_two = {
        "items": [{"contentDetails": {"videoId": "vid0000000300"}}],
    }
    fake = FakeApi({
        "channels": [
            {"items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UUxxx"}}}]},
        ],
        "playlistItems": [page_one, page_two],
        "videos": [
            {"items": [_video_item(f"vid{i:08d}00") for i in range(4)]},
        ],
    })
    monkeypatch.setattr(youtube, "_api", fake)
    monkeypatch.setattr(youtube.time, "sleep", lambda _s: None)

    videos = youtube.list_videos(youtube.parse_source("@pedobraz"), api_key="k")

    assert len(videos) == 4
    # forHandle is the exact, 1-unit lookup; the 100-unit search must stay unused.
    assert fake.params_for("channels")[0]["forHandle"] == "pedobraz"
    assert "search" not in [endpoint for endpoint, _ in fake.calls]
    assert fake.params_for("playlistItems")[1]["pageToken"] == "TOKEN2"


def test_video_details_are_requested_in_batches_of_fifty(
    monkeypatch: MonkeyPatch,
) -> None:
    """120 ids must cost three videos.list calls, not 120."""
    ids = [f"vid{i:08d}" for i in range(120)]
    fake = FakeApi({
        "playlistItems": [{"items": [{"contentDetails": {"videoId": i}} for i in ids]}],
        "videos": [
            {"items": [_video_item(i) for i in ids[0:50]]},
            {"items": [_video_item(i) for i in ids[50:100]]},
            {"items": [_video_item(i) for i in ids[100:120]]},
        ],
    })
    monkeypatch.setattr(youtube, "_api", fake)
    monkeypatch.setattr(youtube.time, "sleep", lambda _s: None)

    videos = youtube.list_videos(
        youtube.parse_source("https://www.youtube.com/playlist?list=PLx"), api_key="k",
    )

    assert len(videos) == 120
    batches = fake.params_for("videos")
    assert len(batches) == 3
    assert len(batches[0]["id"].split(",")) == 50
    assert len(batches[2]["id"].split(",")) == 20
    # Order follows the playlist, not whatever order the API answered in.
    assert [v.video_id for v in videos] == ids


def test_listing_drops_private_videos(monkeypatch: MonkeyPatch) -> None:
    """A private item is not downloadable; it must never reach the queue."""
    fake = FakeApi({
        "playlistItems": [
            {"items": [{"contentDetails": {"videoId": v}} for v in ("aaa", "bbb")]},
        ],
        "videos": [
            {
                "items": [
                    _video_item("aaa"),
                    _video_item("bbb", privacy="private"),
                ],
            },
        ],
    })
    monkeypatch.setattr(youtube, "_api", fake)

    videos = youtube.list_videos(
        youtube.parse_source("https://www.youtube.com/playlist?list=PLx"), api_key="k",
    )

    assert [v.video_id for v in videos] == ["aaa"]


def test_the_api_key_never_appears_in_a_url(monkeypatch: MonkeyPatch) -> None:
    """Under --verbose urllib3 logs every request line, key included.

    Sending it as a header keeps the key out of the log — and out of any
    transcript a user pastes into a bug report.
    """
    seen: list[tuple[str, dict[str, str] | None]] = []

    def fake_http_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        seen.append((url, headers))
        return {"items": [_video_item("D6WjXRJt1DA")]}

    monkeypatch.setattr(youtube, "_http_json", fake_http_json)

    youtube.list_videos(
        youtube.parse_source("https://youtu.be/D6WjXRJt1DA"), api_key="SECRET-KEY",
    )

    url, headers = seen[0]
    assert "SECRET-KEY" not in url
    assert "key=" not in url
    assert headers == {"X-goog-api-key": "SECRET-KEY"}


def test_api_error_becomes_a_readable_message(monkeypatch: MonkeyPatch) -> None:
    """A quota wall must say so, not surface as a bare HTTP status."""

    class Response:
        status_code = 403

        @staticmethod
        def json() -> dict[str, Any]:
            return {
                "error": {
                    "message": "The request cannot be completed.",
                    "errors": [{"reason": "quotaExceeded"}],
                },
            }

    monkeypatch.setattr(youtube.requests, "get", lambda *a, **kw: Response())

    with pytest.raises(youtube.YouTubeError) as excinfo:
        youtube._http_json("https://example.invalid")

    message = str(excinfo.value)
    assert "quotaExceeded" in message
    assert "квота" in message


# --------------------------------------------------------------------- filters


def _video(
    video_id: str, *, date: str = "2026-02-08", duration: int = 3600, live: bool = False,
) -> youtube.YouTubeVideo:
    return youtube.YouTubeVideo(
        video_id=video_id, title=video_id, upload_date=date,
        duration=duration, is_live=live,
    )


def test_filters_drop_shorts_live_and_out_of_range_dates() -> None:
    videos = [
        _video("keep", date="2026-03-01", duration=3600),
        _video("short", duration=30),
        _video("live", live=True),
        _video("old", date="2025-12-31"),
        _video("toolong", duration=99999),
    ]

    kept = youtube.apply_filters(
        videos,
        youtube.VideoFilters(
            since="2026-01-01", min_duration=60, max_duration=7200, skip_live=True,
        ),
    )

    assert [v.video_id for v in kept] == ["keep"]


def test_unknown_duration_survives_the_length_filter() -> None:
    """0 means "not known yet", not "zero seconds".

    The key-free yt-dlp listing often has no duration before download; dropping
    those would silently empty a whole-channel run for anyone without an API key.
    """
    kept = youtube.apply_filters(
        [_video("unknown", duration=0)],
        youtube.VideoFilters(min_duration=60),
    )
    assert [v.video_id for v in kept] == ["unknown"]


def test_limit_applies_after_filtering_not_before() -> None:
    """Otherwise a cap of 2 over [short, short, long] would return nothing."""
    videos = [_video("s1", duration=10), _video("s2", duration=10), _video("long")]

    kept = youtube.apply_filters(
        videos, youtube.VideoFilters(limit=2, min_duration=60),
    )

    assert [v.video_id for v in kept] == ["long"]


def test_resolve_sources_deduplicates_and_keeps_order(monkeypatch: MonkeyPatch) -> None:
    """A link named on its own and again inside a channel appears once, first."""

    def fake_list(
        source: youtube.YouTubeSource, *, api_key: str | None, filters: Any,
    ) -> list[youtube.YouTubeVideo]:
        if source.kind == "video":
            return [_video("bbb")]
        return [_video("aaa"), _video("bbb"), _video("ccc")]

    monkeypatch.setattr(youtube, "list_videos", fake_list)

    videos = youtube.resolve_sources(["https://youtu.be/bbb00000000", "@channel"])

    assert [v.video_id for v in videos] == ["bbb", "aaa", "ccc"]


# ------------------------------------------------------------------ exclusions


def test_excluded_ids_are_dropped_before_the_limit() -> None:
    """`--yt-limit 2` must yield 2 usable videos, not 2 minus the excluded ones."""
    videos = [_video("skip1"), _video("keep1"), _video("skip2"), _video("keep2")]

    kept = youtube.apply_filters(
        videos,
        youtube.VideoFilters(limit=2, exclude_ids=frozenset({"skip1", "skip2"})),
    )

    assert [v.video_id for v in kept] == ["keep1", "keep2"]


def test_resolve_excluded_ids_expands_a_playlist(monkeypatch: MonkeyPatch) -> None:
    """A playlist keeps the rule correct as episodes are added to the show."""
    fake = FakeApi({
        "playlistItems": [
            {"items": [{"contentDetails": {"videoId": v}} for v in ("aaa", "bbb")]},
        ],
    })
    monkeypatch.setattr(youtube, "_api", fake)

    excluded = youtube.resolve_excluded_ids(["PLshow"], api_key="k")

    assert excluded == {"aaa", "bbb"}
    # Only ids are needed, so the per-video details call must not happen.
    assert [endpoint for endpoint, _ in fake.calls] == ["playlistItems"]


def test_resolve_excluded_ids_takes_a_bare_video() -> None:
    assert youtube.resolve_excluded_ids(
        ["https://youtu.be/D6WjXRJt1DA"], api_key="k",
    ) == {"D6WjXRJt1DA"}


def test_exclusion_beats_a_directly_named_link(monkeypatch: MonkeyPatch) -> None:
    """"Never take this one" has to mean never, however it was reached."""

    def fake_list(
        source: youtube.YouTubeSource, *, api_key: str | None, filters: Any,
    ) -> list[youtube.YouTubeVideo]:
        return youtube.apply_filters([_video("banned"), _video("fine")], filters)

    monkeypatch.setattr(youtube, "list_videos", fake_list)
    monkeypatch.setattr(
        youtube, "resolve_excluded_ids", lambda sources, *, api_key=None: {"banned"},
    )

    videos = youtube.resolve_sources(["@channel"], exclude=["PLshow"])

    assert [v.video_id for v in videos] == ["fine"]


def test_exclusions_add_to_the_filters_already_given(monkeypatch: MonkeyPatch) -> None:
    """A caller-supplied exclusion set must not be dropped by the source list."""

    seen: list[frozenset[str]] = []

    def fake_list(
        source: youtube.YouTubeSource, *, api_key: str | None, filters: Any,
    ) -> list[youtube.YouTubeVideo]:
        seen.append(filters.exclude_ids)
        return []

    monkeypatch.setattr(youtube, "list_videos", fake_list)
    monkeypatch.setattr(
        youtube, "resolve_excluded_ids", lambda sources, *, api_key=None: {"fromlist"},
    )

    youtube.resolve_sources(
        ["@channel"],
        filters=youtube.VideoFilters(exclude_ids=frozenset({"preset"})),
        exclude=["PLshow"],
    )

    assert seen == [frozenset({"preset", "fromlist"})]


def test_invalid_date_is_reported_clearly() -> None:
    with pytest.raises(youtube.YouTubeError, match="ГГГГ-ММ-ДД"):
        youtube.apply_filters([_video("x")], youtube.VideoFilters(since="08.02.2026"))
