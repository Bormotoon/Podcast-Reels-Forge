"""RU: Тесты аудио-признаков кандидатов.

EN: Tests for the per-candidate audio features.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from podcast_reels_forge.analysis import audio_features
from podcast_reels_forge.analysis.audio_features import (
    annotate_records_with_audio,
    measure_segment_audio,
    parse_silencedetect,
    parse_volumedetect,
    resolve_source_audio,
)
from podcast_reels_forge.analysis.contracts import MomentRecord, coerce_moment_record
from podcast_reels_forge.analysis.scoring import (
    NEUTRAL_FACTOR,
    audio_quality_score,
    combined_priority_score,
    speech_rate_score,
)

# Captured verbatim from a real ffmpeg run over a 20s span of a podcast.
FFMPEG_STDERR = """\
[Parsed_silencedetect_1 @ 0x7dfb0c082500] silence_start: 1.325458
[Parsed_silencedetect_1 @ 0x7dfb0c082500] silence_end: 2.236479 | silence_duration: 0.911021
[Parsed_silencedetect_1 @ 0x7dfb0c082500] silence_start: 10.457521
[Parsed_silencedetect_1 @ 0x7dfb0c082500] silence_end: 10.878937 | silence_duration: 0.421417
[Parsed_volumedetect_0 @ 0x7dfb0c0022c0] n_samples: 1920000
[Parsed_volumedetect_0 @ 0x7dfb0c0022c0] mean_volume: -17.3 dB
[Parsed_volumedetect_0 @ 0x7dfb0c0022c0] max_volume: -2.4 dB
[Parsed_volumedetect_0 @ 0x7dfb0c0022c0] histogram_2db: 36
"""


def _record(start: float = 0.0, end: float = 45.0, **extra: Any) -> MomentRecord:
    record = coerce_moment_record(
        {
            "start": start,
            "end": end,
            "title": "Момент",
            "quote": "Цитата",
            "why": "Причина",
            "score": 8.0,
            **extra,
        },
    )
    assert record is not None
    return record


# -- parsers -----------------------------------------------------------------


def test_parse_volumedetect_reads_mean_and_max() -> None:
    assert parse_volumedetect(FFMPEG_STDERR) == (-17.3, -2.4)


def test_parse_volumedetect_returns_none_on_junk() -> None:
    assert parse_volumedetect("ffmpeg exploded") == (None, None)


def test_parse_silencedetect_sums_reported_pauses() -> None:
    ratio = parse_silencedetect(FFMPEG_STDERR, span_seconds=20.0)
    assert ratio == pytest.approx((0.911021 + 0.421417) / 20.0, abs=1e-4)


def test_parse_silencedetect_handles_unterminated_trailing_silence() -> None:
    """A pause still running when the clip ends has no duration line."""
    stderr = (
        "[silencedetect] silence_start: 8.0\n"
        "[Parsed_volumedetect_0] mean_volume: -20.0 dB\n"
    )
    assert parse_silencedetect(stderr, span_seconds=10.0) == pytest.approx(0.2)


def test_parse_silencedetect_reports_zero_when_nothing_was_silent() -> None:
    """No markers plus a real volume reading means the span was all speech."""
    stderr = "[Parsed_volumedetect_0] mean_volume: -18.0 dB\n"
    assert parse_silencedetect(stderr, span_seconds=10.0) == 0.0


def test_parse_silencedetect_returns_none_without_a_measurement() -> None:
    assert parse_silencedetect("nothing useful", span_seconds=10.0) is None


def test_parse_silencedetect_is_capped_at_one() -> None:
    stderr = (
        "silence_end: 5.0 | silence_duration: 30.0\n"
        "[Parsed_volumedetect_0] mean_volume: -50.0 dB\n"
    )
    assert parse_silencedetect(stderr, span_seconds=10.0) == 1.0


# -- probing -----------------------------------------------------------------


def test_measure_segment_audio_parses_a_successful_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="", stderr=FFMPEG_STDERR)

    monkeypatch.setattr(audio_features.subprocess, "run", fake_run)

    features = measure_segment_audio(tmp_path / "a.mp3", 60.0, 80.0)
    assert features is not None
    assert features.mean_volume_db == -17.3
    assert features.silence_ratio is not None and features.silence_ratio > 0

    # Audio-only, seeking before the input so ffmpeg does not decode the lead-in.
    assert "-vn" in captured["command"]
    assert captured["command"].index("-ss") < captured["command"].index("-i")


def test_measure_segment_audio_returns_none_when_ffmpeg_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> Any:
        raise OSError("ffmpeg not found")

    monkeypatch.setattr(audio_features.subprocess, "run", fake_run)
    assert measure_segment_audio(tmp_path / "a.mp3", 0.0, 10.0) is None


def test_measure_segment_audio_returns_none_on_a_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(command, 30)

    monkeypatch.setattr(audio_features.subprocess, "run", fake_run)
    assert measure_segment_audio(tmp_path / "a.mp3", 0.0, 10.0) is None


def test_measure_segment_audio_rejects_an_empty_span(tmp_path: Path) -> None:
    assert measure_segment_audio(tmp_path / "a.mp3", 10.0, 10.0) is None


# -- annotation --------------------------------------------------------------


def test_annotation_attaches_measurements(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        audio_features.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="", stderr=FFMPEG_STDERR,
        ),
    )
    source = tmp_path / "a.mp3"
    source.write_bytes(b"")

    annotated = annotate_records_with_audio([_record()], source)
    assert annotated[0].audio_energy_db == -17.3
    assert annotated[0].audio_silence_ratio is not None


def test_annotation_skips_without_source_or_when_disabled(tmp_path: Path) -> None:
    """Missing audio must cost nothing — the whole feature is optional."""
    records = [_record()]
    assert annotate_records_with_audio(records, None) == records
    assert annotate_records_with_audio(records, tmp_path / "a.mp3", enabled=False) == records


def test_resolve_source_audio_prefers_an_existing_file(tmp_path: Path) -> None:
    existing = tmp_path / "episode.mp3"
    existing.write_bytes(b"")
    assert resolve_source_audio({"source_audio": str(existing)}) == existing
    assert resolve_source_audio({"source_audio": str(tmp_path / "missing.mp3")}) is None
    assert resolve_source_audio({}) is None


# -- scoring -----------------------------------------------------------------


def test_audio_scores_are_neutral_when_unmeasured() -> None:
    """Candidates with and without audio data must stay comparable."""
    assert audio_quality_score({}) == NEUTRAL_FACTOR
    assert speech_rate_score({}) == NEUTRAL_FACTOR


def test_loud_speech_outscores_a_quiet_pause_heavy_span() -> None:
    lively = {"audio_energy_db": -16.0, "audio_silence_ratio": 0.15}
    limp = {"audio_energy_db": -32.0, "audio_silence_ratio": 0.8}
    assert audio_quality_score(lively) > audio_quality_score(limp)


def test_speech_rate_favours_a_readable_pace() -> None:
    assert speech_rate_score({"speech_rate_wps": 2.5}) == 1.0
    assert speech_rate_score({"speech_rate_wps": 0.4}) < 0.5
    assert speech_rate_score({"speech_rate_wps": 9.0}) < 0.5


def test_audio_signal_moves_the_combined_priority() -> None:
    base = _record().to_dict()
    lively = {**base, "audio_energy_db": -16.0, "audio_silence_ratio": 0.1}
    limp = {**base, "audio_energy_db": -34.0, "audio_silence_ratio": 0.9}
    assert combined_priority_score(lively, target_min=30, target_max=60) > (
        combined_priority_score(limp, target_min=30, target_max=60)
    )
