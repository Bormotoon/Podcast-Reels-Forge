"""Tests for the transcript proofreading stage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from podcast_reels_forge.stages.proofread_stage import (
    _extract_corrections,
    _normalize_for_compare,
    build_proofread_batches,
    is_correction_safe,
    run_proofread,
)

if TYPE_CHECKING:
    import pytest

    MonkeyPatch = pytest.MonkeyPatch


class FakeProvider:
    """LLM stub that returns queued responses per call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("No more fake responses")
        return self._responses.pop(0)


def test_normalize_strips_punctuation_case_and_yo() -> None:
    assert _normalize_for_compare("Привет, МИР! Ещё…") == "привет мир еще"
    assert _normalize_for_compare("  ") == ""


def test_correction_safe_accepts_spelling_and_punctuation_fixes() -> None:
    # Punctuation and capitalization only.
    assert is_correction_safe(
        "ну вот мы и пришли к этому вопросу",
        "Ну вот, мы и пришли к этому вопросу.",
    )
    # Small typo / word-agreement fix.
    assert is_correction_safe(
        "они пошел в магазин за хлебом и молоко",
        "Они пошли в магазин за хлебом и молоком.",
    )
    # ё restoration.
    assert is_correction_safe("все еще впереди", "Всё ещё впереди.")


def test_correction_safe_rejects_additions_and_paraphrase() -> None:
    # Model appended a whole new clause.
    assert not is_correction_safe(
        "мы записали подкаст",
        "Мы записали подкаст, и это был самый лучший выпуск в истории.",
    )
    # Model paraphrased the sentence.
    assert not is_correction_safe(
        "короче мы это дело так и не доделали",
        "В итоге работа осталась незавершённой.",
    )
    # Model dropped the content.
    assert not is_correction_safe("тут был длинный содержательный рассказ", "Да.")
    # Empty correction for non-empty text.
    assert not is_correction_safe("текст", "")


def test_build_proofread_batches_respects_char_budget() -> None:
    segments = [
        {"text": "a" * 300},
        {"text": ""},  # empty segments are skipped
        {"text": "b" * 300},
        {"text": "c" * 300},
    ]
    batches = build_proofread_batches(segments, max_chars=500)
    assert batches == [[0], [2], [3]]

    batches_all = build_proofread_batches(segments, max_chars=2000)
    assert batches_all == [[0, 2, 3]]


def test_extract_corrections_handles_wrapped_and_bare_lists() -> None:
    wrapped = {"segments": [{"id": 0, "text": "Раз."}, {"id": "1", "text": "Два."}]}
    assert _extract_corrections(wrapped) == {0: "Раз.", 1: "Два."}

    bare = [{"id": 2, "text": "Три."}, {"id": None, "text": "x"}, "junk"]
    assert _extract_corrections(bare) == {2: "Три."}

    assert _extract_corrections({"segments": [{"id": 3, "text": "  "}]}) == {}


def test_run_proofread_applies_safe_and_keeps_unsafe(tmp_path: Path) -> None:
    transcript = {
        "language": "ru",
        "duration": 30.0,
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "привет как дела", "speaker": None, "words": []},
            {"start": 5.0, "end": 10.0, "text": "все хорошо спасибо", "speaker": None, "words": []},
            {"start": 10.0, "end": 15.0, "text": "мы начинаем подкаст", "speaker": None, "words": []},
        ],
        "sentences": [],
    }
    transcript_path = tmp_path / "audio.json"
    transcript_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")

    response = json.dumps(
        {
            "segments": [
                # Safe fix: punctuation + capitalization.
                {"id": 0, "text": "Привет, как дела?"},
                # Unsafe: the model invented a new clause — must be rejected.
                {"id": 1, "text": "Всё хорошо, спасибо, а у вас как дела на работе?"},
                # id 2 missing: original must be kept.
            ],
        },
        ensure_ascii=False,
    )
    provider = FakeProvider([response])

    out_path = asyncio.run(run_proofread(
        transcript_path=transcript_path,
        provider=provider,
        quiet=True,
    ))

    assert out_path == tmp_path / "audio.proofread.json"
    data = json.loads(out_path.read_text(encoding="utf-8"))
    texts = [seg["text"] for seg in data["segments"]]
    assert texts == [
        "Привет, как дела?",
        "все хорошо спасибо",
        "мы начинаем подкаст",
    ]
    assert data["proofread"]["applied"] == 1
    assert data["proofread"]["rejected"] == 1
    assert data["proofread"]["failed_batches"] == 0
    assert data["proofread"]["segments_total"] == 3

    # Sentences are rebuilt from the corrected segments.
    assert data["sentences"]
    assert "Привет, как дела?" in data["sentences"][0]["text"]

    # The raw transcript stays untouched, and the SRT is written.
    raw = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert raw["segments"][0]["text"] == "привет как дела"
    srt_text = (tmp_path / "audio.proofread.srt").read_text(encoding="utf-8")
    assert "Привет, как дела?" in srt_text

    # The prompt carried the segment payload.
    assert "привет как дела" in provider.prompts[0]


def test_run_proofread_survives_failed_batches(tmp_path: Path) -> None:
    transcript = {
        "language": "ru",
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "a" * 600},
            {"start": 5.0, "end": 10.0, "text": "b" * 600},
        ],
    }
    transcript_path = tmp_path / "audio.json"
    transcript_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")

    class ExplodingProvider:
        async def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
            raise RuntimeError("server down")

    out_path = asyncio.run(run_proofread(
        transcript_path=transcript_path,
        provider=ExplodingProvider(),
        proofread_conf={"max_chars_chunk": 600},
        quiet=True,
    ))

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert [seg["text"] for seg in data["segments"]] == ["a" * 600, "b" * 600]
    assert data["proofread"]["applied"] == 0
    assert data["proofread"]["failed_batches"] == 2
