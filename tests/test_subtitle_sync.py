"""RU: Синхронизация субтитров: обрезанные фразы, пересечения, караоке и
проверка вторым проходом Whisper.

EN: Subtitle timing: sentences cut by the clip, overlaps, karaoke and the
second Whisper pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from podcast_reels_forge.utils import burned_subtitles as bs
from podcast_reels_forge.utils import subtitle_sync as ss

W = bs._TimedSubtitleWord


def _sentence(words: list[tuple[float, float, str]]) -> bs.SubtitleSegment:
    timed = tuple(W(a, b, t) for a, b, t in words)
    return bs.SubtitleSegment(
        start=timed[0].start, end=timed[-1].end, text=" ".join(w.text for w in timed), words=timed,
    )


# -- slicing a sentence the clip cuts -----------------------------------------


def test_sentence_cut_by_the_clip_keeps_real_timings() -> None:
    """The bug behind seconds of drift: full text, partial word timings."""
    words = [(100.0 + i, 100.5 + i, f"слово{i}") for i in range(20)]
    [clipped] = bs.slice_segments_for_clip([_sentence(words)], clip_start=90.0, clip_end=110.0)

    assert clipped.text.split() == [f"слово{i}" for i in range(10)]
    assert bs._real_timed_words(clipped) is not None
    timed = bs._build_timed_words(clipped)
    # "слово9" is spoken at 109.0 in the episode = 19.0 in the clip, not squeezed earlier.
    assert timed[-1].start == pytest.approx(19.0)


def test_word_straddling_the_boundary_goes_by_its_midpoint() -> None:
    sentence = _sentence([(9.0, 9.4, "до"), (9.8, 10.4, "граница"), (10.6, 11.0, "после")])
    [clipped] = bs.slice_segments_for_clip([sentence], clip_start=10.0, clip_end=20.0)
    assert clipped.text == "граница после"
    assert clipped.start == 0.0


# -- overlaps and karaoke ---------------------------------------------------------


def test_overlap_trims_the_earlier_cue_instead_of_delaying_speech() -> None:
    first = bs.SubtitleSegment(0.0, 1.5, "раз")
    second = bs.SubtitleSegment(1.0, 2.0, "два")
    fixed = bs._remove_overlaps([first, second])
    assert fixed[1].start == 1.0 and fixed[1].end == 2.0
    assert fixed[0].end == pytest.approx(0.95)


def test_plain_cues_by_default(tmp_path: Path) -> None:
    ass = tmp_path / "a.ass"
    settings = bs.subtitle_settings_from_conf(None, repo_dir=tmp_path)
    assert settings.karaoke is False
    bs._write_ass_file(ass, [_sentence([(0.0, 1.0, "раз"), (1.0, 2.0, "два")])], settings)
    text = ass.read_text(encoding="utf-8")
    assert "\\kf" not in text and "раз два" in text


def test_karaoke_waits_for_the_first_word(tmp_path: Path) -> None:
    ass = tmp_path / "a.ass"
    segment = bs.SubtitleSegment(0.0, 3.0, "раз два", words=(W(1.0, 2.0, "раз"), W(2.0, 3.0, "два")))
    settings = bs.subtitle_settings_from_conf({"subtitles": {"karaoke": True}}, repo_dir=tmp_path)
    bs._write_ass_file(ass, [segment], settings)
    assert "{\\k100}{\\kf100}раз" in ass.read_text(encoding="utf-8")


# -- second Whisper pass ----------------------------------------------------------


def _tw(items: list[tuple[float, float, str]]) -> list[ss.TimedWord]:
    return [ss.TimedWord(a, b, t) for a, b, t in items]


REFERENCE = _tw([
    (10.0, 10.4, "То"), (10.4, 10.6, "есть"), (10.6, 11.0, "дать"),
    (11.0, 11.5, "упростится,"), (11.5, 12.0, "успокоит"), (12.0, 12.3, "их"),
])


def test_words_match_exactly_and_by_spelling() -> None:
    heard = _tw([
        (12.0, 12.4, "то"), (12.4, 12.6, "есть"), (12.6, 13.0, "дать"),
        (13.0, 13.5, "простится"), (13.5, 14.0, "успокоит"), (14.0, 14.3, "их"),
    ])
    assert ss.match_words(REFERENCE, heard) == {i: i for i in range(6)}


def test_unmatched_words_move_with_the_speech_around_them() -> None:
    heard = _tw([(12.0, 12.4, "то"), (13.5, 14.0, "успокоит"), (14.0, 14.3, "их")])
    pairs = ss.match_words(REFERENCE, heard)
    fixed = ss.retime(REFERENCE, heard, pairs)
    assert fixed[0].start == 12.0 and fixed[4].start == 13.5
    assert 12.0 < fixed[1].start < fixed[2].start < fixed[3].start < 13.5
    # Unmatched words keep their own durations.
    assert fixed[2].end - fixed[2].start == pytest.approx(0.4)
    assert all(a.start <= b.start for a, b in zip(fixed, fixed[1:]))


def test_an_unheard_word_after_a_pause_stays_after_it() -> None:
    """ "И" was not heard; it must not slide back into the pause before it."""
    reference = _tw([(10.0, 10.5, "сорок"), (10.5, 11.0, "лет."), (12.5, 12.6, "И"), (12.6, 13.0, "там")])
    heard = _tw([(10.0, 10.5, "сорок"), (10.5, 11.0, "лет"), (12.6, 13.0, "там")])
    fixed = ss.retime(reference, heard, ss.match_words(reference, heard))
    assert fixed[2].start == pytest.approx(12.5)


class _FakeModel:
    def __init__(self, words: list[tuple[float, float, str]]) -> None:
        self.words = words

    def transcribe(self, _audio: Any, **_kwargs: Any) -> tuple[Any, Any]:
        segment = SimpleNamespace(words=[SimpleNamespace(start=a, end=b, word=t) for a, b, t in self.words])
        return iter([segment]), None


def _patch_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ss, "extract_audio", lambda *_a, **_k: object())


def test_drift_is_detected_and_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_audio(monkeypatch)
    # Heard 2 s later, relative to a clip starting at 5.0.
    model = _FakeModel([(a - 5.0 + 2.0, b - 5.0 + 2.0, t) for a, b, t in
                        [(w.start, w.end, w.text) for w in REFERENCE]])
    words, report = ss.sync_clip(model, Path("x.wav"), REFERENCE, clip_start=5.0, clip_end=20.0, language="ru")
    assert report.verdict == "retimed" and report.applied
    assert report.median_shift_s == pytest.approx(2.0)
    assert words[0].start == pytest.approx(12.0)


def test_small_drift_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_audio(monkeypatch)
    model = _FakeModel([(w.start - 5.0 + 0.05, w.end - 5.0 + 0.05, w.text) for w in REFERENCE])
    words, report = ss.sync_clip(model, Path("x.wav"), REFERENCE, clip_start=5.0, clip_end=20.0, language="ru")
    assert report.verdict == "in_sync" and not report.applied
    assert words == REFERENCE


def test_unrecognizable_audio_is_not_trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_audio(monkeypatch)
    model = _FakeModel([(1.0, 2.0, "музыка")])
    words, report = ss.sync_clip(model, Path("x.wav"), REFERENCE, clip_start=5.0, clip_end=20.0, language="ru")
    assert report.verdict == "unreliable" and words == REFERENCE


def test_a_failed_check_keeps_the_timings(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("ffmpeg died")

    monkeypatch.setattr(ss, "extract_audio", _boom)
    words, report = ss.sync_clip(_FakeModel([]), Path("x.wav"), REFERENCE, clip_start=5.0, clip_end=20.0, language="ru")
    assert report.verdict == "error" and words == REFERENCE


# -- retimed words reach the subtitles and survive a later re-sync -----------------


def test_retimed_words_reach_the_segments_and_the_saved_file(tmp_path: Path) -> None:
    sentence = _sentence([(10.0, 10.5, "раз"), (10.5, 11.0, "два")])
    retimed = {bs.word_key(sentence.words[1]): (12.0, 12.5)}
    [fixed] = bs.retime_segments([sentence], retimed)
    assert fixed.words[1].start == 12.0 and fixed.end == 12.5

    (tmp_path / bs.SUBTITLE_SYNC_FILE).write_text(
        json.dumps({"retimed_words": [{"orig_start": 10.5, "text": "два", "start": 12.0, "end": 12.5}]}),
        encoding="utf-8",
    )
    assert bs.load_saved_retiming(tmp_path) == retimed
    assert bs.load_saved_retiming(tmp_path / "missing") == {}
