"""Tests for the YouTube fetch stage.

yt-dlp is replaced by a fake throughout: these tests pin down what we ask it for
and how we decide whether to ask at all, not whether YouTube is reachable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from podcast_reels_forge.sources.youtube import YouTubeVideo
from podcast_reels_forge.stages import fetch_stage

if TYPE_CHECKING:
    MonkeyPatch = pytest.MonkeyPatch


def _video(video_id: str = "D6WjXRJt1DA", title: str = "Эпизод") -> YouTubeVideo:
    return YouTubeVideo(
        video_id=video_id, title=title, upload_date="2026-02-08", duration=3600,
    )


class FakeYoutubeDL:
    """Stands in for yt_dlp.YoutubeDL: records options and writes a fake file."""

    #: Every instance built during a test, so assertions can read the options.
    instances: list[FakeYoutubeDL] = []

    def __init__(self, options: dict[str, Any]):
        self.options = dict(options)
        self.urls: list[str] = []
        FakeYoutubeDL.instances.append(self)

    def __enter__(self) -> FakeYoutubeDL:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
        self.urls.append(url)
        video_id = url.rsplit("=", 1)[-1]
        # Mimic the real naming closely enough for the "already here" lookup.
        outtmpl = Path(str(self.options["outtmpl"]))
        suffix = ".mp4" if "merge_output_format" in self.options else ".m4a"
        path = outtmpl.parent / f"2026-02-08 - Эпизод [{video_id}]{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("media", encoding="utf-8")
        return {"id": video_id, "requested_downloads": [{"filepath": str(path)}]}


@pytest.fixture(autouse=True)
def _fake_yt_dlp(monkeypatch: MonkeyPatch) -> None:
    """Install a fake yt_dlp module so the lazy import finds something."""
    FakeYoutubeDL.instances = []
    module = type(sys)("yt_dlp")
    module.YoutubeDL = FakeYoutubeDL  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yt_dlp", module)


def test_missing_yt_dlp_explains_how_to_install_it(monkeypatch: MonkeyPatch) -> None:
    """An ImportError traceback tells the user nothing actionable."""
    monkeypatch.setitem(sys.modules, "yt_dlp", None)

    with pytest.raises(SystemExit, match="pip install -U yt-dlp"):
        fetch_stage._import_yt_dlp()


def test_already_downloaded_video_is_not_fetched_again(tmp_path: Path) -> None:
    """The file on disk is the answer; no request should be made at all."""
    download_dir = tmp_path / "youtube"
    download_dir.mkdir()
    existing = download_dir / "2026-02-08 - Старое название [D6WjXRJt1DA].mp4"
    existing.write_text("media", encoding="utf-8")

    fetched = fetch_stage.fetch_videos(
        [_video()], fetch_stage.FetchConfig(download_dir=download_dir),
    )

    assert len(fetched) == 1
    assert fetched[0].downloaded is False
    # Recognised by id, even though the title on YouTube has since changed.
    assert fetched[0].path == existing
    assert FakeYoutubeDL.instances == []


def test_partial_downloads_do_not_count_as_present(tmp_path: Path) -> None:
    """A `.part` left by an interrupted run must be retried, not adopted."""
    download_dir = tmp_path / "youtube"
    download_dir.mkdir()
    (download_dir / "2026-02-08 - Эпизод [D6WjXRJt1DA].mp4.part").write_text("half")

    assert fetch_stage.find_local_copy(download_dir, "D6WjXRJt1DA") is None

    fetched = fetch_stage.fetch_videos(
        [_video()], fetch_stage.FetchConfig(download_dir=download_dir),
    )
    assert fetched[0].downloaded is True


def test_empty_file_does_not_count_as_present(tmp_path: Path) -> None:
    download_dir = tmp_path / "youtube"
    download_dir.mkdir()
    (download_dir / "2026-02-08 - Эпизод [D6WjXRJt1DA].mp4").write_text("")

    assert fetch_stage.find_local_copy(download_dir, "D6WjXRJt1DA") is None


def test_no_skip_existing_redownloads_and_bypasses_the_archive(tmp_path: Path) -> None:
    """"Re-run everything" must not be silently absorbed by the archive."""
    download_dir = tmp_path / "youtube"
    download_dir.mkdir()
    (download_dir / "2026-02-08 - Эпизод [D6WjXRJt1DA].mp4").write_text("media")

    fetched = fetch_stage.fetch_videos(
        [_video()],
        fetch_stage.FetchConfig(
            download_dir=download_dir,
            archive=download_dir / ".archive.txt",
            skip_existing=False,
        ),
    )

    assert fetched[0].downloaded is True
    options = FakeYoutubeDL.instances[0].options
    assert "download_archive" not in options
    assert options["overwrites"] is True


def test_archive_is_passed_through_when_skipping_is_on(tmp_path: Path) -> None:
    """A nightly channel run relies on the archive to take only what is new."""
    download_dir = tmp_path / "youtube"
    archive = download_dir / ".archive.txt"

    fetch_stage.fetch_videos(
        [_video()],
        fetch_stage.FetchConfig(download_dir=download_dir, archive=archive),
    )

    options = FakeYoutubeDL.instances[0].options
    assert options["download_archive"] == str(archive)
    assert archive.parent.exists()


def test_options_carry_the_naming_template_and_cookies(tmp_path: Path) -> None:
    download_dir = tmp_path / "youtube"
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")

    fetch_stage.fetch_videos(
        [_video()],
        fetch_stage.FetchConfig(
            download_dir=download_dir, cookies_file=cookies, rate_limit="5M",
        ),
    )

    options = FakeYoutubeDL.instances[0].options
    assert options["outtmpl"] == str(
        download_dir / fetch_stage.DEFAULT_FILENAME_TEMPLATE,
    )
    assert options["cookiefile"] == str(cookies)
    assert options["ratelimit"] == 5 * 1024 * 1024
    # Downloading one link must not drag in the playlist it happens to sit in.
    assert options["noplaylist"] is True


def test_the_folder_is_named_once_not_twice(tmp_path: Path) -> None:
    """yt-dlp joins `paths.home` onto the template, so only one may carry it.

    Setting both put every download under input/youtube/input/youtube/.
    """
    download_dir = tmp_path / "youtube"

    fetch_stage.fetch_videos(
        [_video()], fetch_stage.FetchConfig(download_dir=download_dir),
    )

    options = FakeYoutubeDL.instances[0].options
    assert "paths" not in options
    assert str(options["outtmpl"]).startswith(str(download_dir))


def test_downloads_land_directly_in_the_download_folder(tmp_path: Path) -> None:
    """End of the naming chain: the file is where the next stage will look."""
    download_dir = tmp_path / "input" / "youtube"

    fetched = fetch_stage.fetch_videos(
        [_video()], fetch_stage.FetchConfig(download_dir=download_dir),
    )

    assert fetched[0].path.parent == download_dir


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("5M", 5 * 1024 * 1024), ("500K", 512000), ("1G", 1024**3),
     ("1000", 1000), ("2.5M", int(2.5 * 1024**2)), ("junk", None), ("", None)],
)
def test_rate_limit_parsing(raw: str, expected: int | None) -> None:
    assert fetch_stage._parse_rate_limit(raw) == expected


def test_video_format_is_capped_by_height() -> None:
    """The output is a 1080-wide 9:16 clip, so a 4K source is wasted bytes."""
    selector = fetch_stage.build_format(
        fetch_stage.FetchConfig(download_dir=Path("x"), max_height=1080),
    )
    assert "height<=1080" in selector
    # Each fallback drops one constraint, so an odd rendition still downloads.
    assert selector.endswith("/bv*+ba/b")


def test_audio_only_format_skips_the_video_track() -> None:
    selector = fetch_stage.build_format(
        fetch_stage.FetchConfig(download_dir=Path("x"), want_video=False),
    )
    assert selector == fetch_stage.AUDIO_FORMAT
    assert "bv" not in selector


def test_audio_only_download_produces_no_merge_request(tmp_path: Path) -> None:
    fetch_stage.fetch_videos(
        [_video()],
        fetch_stage.FetchConfig(download_dir=tmp_path / "youtube", want_video=False),
    )
    assert "merge_output_format" not in FakeYoutubeDL.instances[0].options


def test_audio_only_pins_the_container_to_m4a(tmp_path: Path) -> None:
    """The `/ba` fallback can serve opus in a .webm container.

    .webm counts as a video extension, so such a file would enter the queue as a
    video and the cut stage would try to cut an audio-only track. Pinning the
    container closes that off; for YouTube's usual m4a nothing is transcoded.
    """
    fetch_stage.fetch_videos(
        [_video()],
        fetch_stage.FetchConfig(download_dir=tmp_path / "youtube", want_video=False),
    )

    from podcast_reels_forge.pipeline import AUDIO_ONLY_EXTS, VIDEO_EXTS

    postprocessors = FakeYoutubeDL.instances[0].options["postprocessors"]
    assert postprocessors[0]["key"] == "FFmpegExtractAudio"
    codec = postprocessors[0]["preferredcodec"]
    # Keeps the two halves in sync: whatever container the fetch pins must be one
    # the queue reads as audio-only, and must not be one it reads as video.
    assert f".{codec}" in AUDIO_ONLY_EXTS
    assert f".{codec}" not in VIDEO_EXTS


def test_extra_options_are_merged_last(tmp_path: Path) -> None:
    """The raw yt-dlp pass-through must be able to override what we computed.

    It exists so a YouTube-side change can be worked around from config rather
    than by patching the stage; that only holds if it wins over our defaults.
    """
    fetch_stage.fetch_videos(
        [_video()],
        fetch_stage.FetchConfig(
            download_dir=tmp_path / "youtube",
            extra_options={
                "retries": 99,
                "extractor_args": {"youtube": {"player_client": ["web_safari"]}},
            },
        ),
    )

    options = FakeYoutubeDL.instances[0].options
    assert options["retries"] == 99
    assert options["extractor_args"] == {"youtube": {"player_client": ["web_safari"]}}


def test_video_download_declares_no_audio_postprocessor(tmp_path: Path) -> None:
    fetch_stage.fetch_videos(
        [_video()],
        fetch_stage.FetchConfig(download_dir=tmp_path / "youtube", want_video=True),
    )
    assert "postprocessors" not in FakeYoutubeDL.instances[0].options


@pytest.mark.parametrize(
    ("mode", "stages", "expected"),
    [
        ("auto", {"fetch", "transcribe", "cut"}, True),
        ("auto", {"fetch", "transcribe", "article"}, False),
        # Under --only fetch the intent is unknown; guessing wrong would cost a
        # re-download of the whole channel, so the video is kept.
        ("auto", {"fetch"}, True),
        ("video", {"fetch", "transcribe"}, True),
        ("audio", {"fetch", "cut"}, False),
    ],
)
def test_want_video_follows_the_selected_stages(
    mode: str, stages: set[str], expected: bool,
) -> None:
    assert fetch_stage.want_video_for_stages({"download": mode}, stages) is expected


def test_unknown_download_mode_is_rejected() -> None:
    from podcast_reels_forge.sources.youtube import YouTubeError

    with pytest.raises(YouTubeError, match="auto, video или audio"):
        fetch_stage.want_video_for_stages({"download": "mp3"}, {"fetch"})


def test_a_failing_video_does_not_abort_the_batch(
    tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    """Seventy episodes must not die on one blocked item."""

    class Flaky(FakeYoutubeDL):
        def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
            if "bad00000000" in url:
                raise RuntimeError("Video unavailable. Blocked by the rights holder.")
            return super().extract_info(url, download=download)

    module = type(sys)("yt_dlp")
    module.YoutubeDL = Flaky  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yt_dlp", module)

    fetched = fetch_stage.fetch_videos(
        [_video("good0000000"), _video("bad00000000"), _video("also0000000")],
        fetch_stage.FetchConfig(download_dir=tmp_path / "youtube"),
    )

    assert [f.video.video_id for f in fetched] == ["good0000000", "also0000000"]


def test_a_failed_download_is_retried_with_other_clients(
    tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    """YouTube hides formats from the default clients on some older videos.

    The video then reads as "not available" while other clients still see it, so
    one failure earns a second attempt before being written off.
    """

    class DefaultClientBlind(FakeYoutubeDL):
        def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
            clients = (self.options.get("extractor_args") or {}).get("youtube", {})
            if not clients.get("player_client"):
                raise RuntimeError("This video is not available")
            return super().extract_info(url, download=download)

    module = type(sys)("yt_dlp")
    module.YoutubeDL = DefaultClientBlind  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yt_dlp", module)

    fetched = fetch_stage.fetch_videos(
        [_video()], fetch_stage.FetchConfig(download_dir=tmp_path / "youtube"),
    )

    assert len(fetched) == 1
    assert fetched[0].downloaded is True
    retry_clients = FakeYoutubeDL.instances[-1].options["extractor_args"]["youtube"]
    assert retry_clients["player_client"] == list(fetch_stage.FALLBACK_PLAYER_CLIENTS)


def test_a_user_supplied_client_choice_is_not_second_guessed(
    tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    """Retrying behind their back would make the option look like a no-op."""

    class AlwaysFails(FakeYoutubeDL):
        def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
            raise RuntimeError("This video is not available")

    module = type(sys)("yt_dlp")
    module.YoutubeDL = AlwaysFails  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yt_dlp", module)

    fetched = fetch_stage.fetch_videos(
        [_video()],
        fetch_stage.FetchConfig(
            download_dir=tmp_path / "youtube",
            extra_options={"extractor_args": {"youtube": {"player_client": ["mine"]}}},
        ),
    )

    assert fetched == []
    assert len(FakeYoutubeDL.instances) == 1


def test_progress_fires_once_per_video_including_skipped(tmp_path: Path) -> None:
    """A progress bar over this must still reach its total when nothing is new."""
    download_dir = tmp_path / "youtube"
    download_dir.mkdir()
    (download_dir / "2026-02-08 - Эпизод [aaa00000000].mp4").write_text("media")

    seen: list[str] = []
    fetch_stage.fetch_videos(
        [_video("aaa00000000"), _video("bbb00000000")],
        fetch_stage.FetchConfig(download_dir=download_dir),
        on_progress=lambda v: seen.append(v.video_id),
    )

    assert sorted(seen) == ["aaa00000000", "bbb00000000"]


def test_locate_existing_reports_only_what_is_on_disk(tmp_path: Path) -> None:
    """Used when --youtube runs without the fetch stage selected."""
    download_dir = tmp_path / "youtube"
    download_dir.mkdir()
    (download_dir / "2026-02-08 - Эпизод [aaa00000000].mp4").write_text("media")

    found = fetch_stage.locate_existing(
        [_video("aaa00000000"), _video("bbb00000000")], download_dir,
    )

    assert [f.video.video_id for f in found] == ["aaa00000000"]
    assert found[0].stem == "2026-02-08 - Эпизод [aaa00000000]"
    assert FakeYoutubeDL.instances == []


def test_resolve_download_dir_defaults_under_the_input_folder() -> None:
    assert fetch_stage.resolve_download_dir({}, Path("input")) == Path("input/youtube")
    assert fetch_stage.resolve_download_dir(
        {"download_dir": "/data/yt"}, Path("input"),
    ) == Path("/data/yt")


def test_describe_videos_marks_what_is_already_downloaded(tmp_path: Path) -> None:
    download_dir = tmp_path / "youtube"
    download_dir.mkdir()
    (download_dir / "2026-02-08 - Эпизод [aaa00000000].mp4").write_text("media")

    table = fetch_stage.describe_videos(
        [_video("aaa00000000"), _video("bbb00000000")], download_dir,
    )

    assert "Роликов: 2" in table
    assert "уже скачано: 1" in table
    assert "aaa00000000" in table and "bbb00000000" in table
