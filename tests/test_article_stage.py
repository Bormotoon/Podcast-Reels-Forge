"""Tests for the transcript retelling (article) stage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from podcast_reels_forge.stages.article_stage import (
    chunk_plain_text,
    ArticleSection,
    parse_markdown_sections,
    source_coverage,
    check_faithfulness,
    length_ratio,
    merge_adjacent_sections,
    novel_word_ratio,
    render_article_markdown,
    run_article,
)


class FakeProvider:
    """LLM stub returning queued responses, recording the prompts it saw."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.temperatures: list[float] = []

    async def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
        self.prompts.append(prompt)
        self.temperatures.append(temperature)
        if not self._responses:
            raise RuntimeError("No more fake responses")
        return self._responses.pop(0)


def _sections_payload(paragraphs: list[str], title: str = "Тема") -> str:
    """The model answers in markdown, so tests speak markdown too."""
    body = "\n\n".join(paragraphs)
    return f"## {title}\n\n{body}\n"


def test_novel_word_ratio_ignores_russian_inflection() -> None:
    """Case endings must not read as invented vocabulary."""
    source = "Обсуждаем подкаст и фудкорты в торговых центрах"
    # Same content words, different grammatical forms.
    faithful = "В подкасте обсуждаются фудкорты в торговом центре"
    assert novel_word_ratio(source, faithful) == 0.0

    invented = "Ведущий рассказывает про криптовалюту, ипотеку и ремонт квартиры"
    assert novel_word_ratio(source, invented) > 0.5


def test_novel_word_ratio_edge_cases() -> None:
    assert novel_word_ratio("что-то", "") == 0.0
    # Only short function words: nothing to judge, so nothing is flagged.
    assert novel_word_ratio("", "и не да") == 0.0


def test_length_ratio() -> None:
    assert length_ratio("abcd", "ab") == 0.5
    assert length_ratio("", "abc") == 0.0


def test_check_faithfulness_accepts_an_edit_and_rejects_a_rewrite() -> None:
    source = (
        "Ну вот, значит, сегодня мы обсуждаем фудкорты в торговых центрах города, "
        "и родители постоянно спрашивают, почему подростки там сидят целыми днями"
    )

    # Filler removed, wording kept: this is what an edit looks like.
    edited = (
        "Сегодня мы обсуждаем фудкорты в торговых центрах города. Родители "
        "постоянно спрашивают, почему подростки там сидят целыми днями."
    )
    assert check_faithfulness(source, edited).ok

    padded = check_faithfulness(source, source * 3)
    assert padded.padded and not padded.ok

    # Same facts, but rewritten in the model's own words.
    rewritten = check_faithfulness(
        source,
        "Спикер анализирует феномен посещаемости ресторанных зон, разбирая "
        "мотивацию несовершеннолетних посетителей коммерческих помещений",
    )
    assert rewritten.invented_vocabulary and not rewritten.ok

    abridged = check_faithfulness(source, "Обсуждаем фудкорты.")
    assert abridged.abridged and not abridged.ok


def test_parse_markdown_sections() -> None:
    text = """## Первый раздел

Абзац, который
переносится на две строки.

Второй абзац.

## Второй раздел

Текст.
"""

    assert parse_markdown_sections(text) == [
        {
            "title": "Первый раздел",
            "paragraphs": ["Абзац, который переносится на две строки.", "Второй абзац."],
        },
        {"title": "Второй раздел", "paragraphs": ["Текст."]},
    ]

    # Text before any heading still counts as a section.
    assert parse_markdown_sections("Просто абзац.") == [
        {"title": "", "paragraphs": ["Просто абзац."]},
    ]
    assert parse_markdown_sections("") == []


def test_parse_markdown_drops_json_scaffolding() -> None:
    """A model half-answering in JSON must not leak braces into the prose."""
    text = """```markdown
## Раздел

{

paragraphs [

Нормальный абзац.
```"""

    assert parse_markdown_sections(text) == [
        {"title": "Раздел", "paragraphs": ["Нормальный абзац."]},
    ]


def test_source_coverage_catches_abridgement() -> None:
    source = "Обсуждаем фудкорты, торговые центры, школьные проекты и подростков"
    assert source_coverage(source, source) == 1.0
    assert source_coverage(source, "Обсуждаем фудкорты") < 0.5
    assert source_coverage("", "что угодно") == 1.0


def test_merge_adjacent_sections_joins_same_topic_across_chunks() -> None:
    """A topic split by a chunk boundary must read as one section."""
    sections = [
        ArticleSection("Фудкорты", ("абзац 1",), 0.0, 10.0),
        ArticleSection("фудкорты.", ("абзац 2",), 10.0, 20.0),
        ArticleSection("Другое", ("абзац 3",), 20.0, 30.0),
    ]

    merged = merge_adjacent_sections(sections)

    assert len(merged) == 2
    assert merged[0].paragraphs == ("абзац 1", "абзац 2")
    assert merged[0].start == 0.0
    assert merged[0].end == 20.0
    assert merged[1].title == "Другое"


def test_render_article_markdown() -> None:
    md = render_article_markdown(
        "Эпизод",
        [ArticleSection("Раздел", ("Первый абзац.", "Второй абзац."), 0.0, 5.0)],
    )

    assert md.startswith("# Эпизод\n")
    assert "## Раздел" in md
    assert "Первый абзац.\n\nВторой абзац." in md


def test_run_article_writes_markdown_and_metadata(tmp_path: Path) -> None:
    transcript = {
        "language": "ru",
        "segments": [
            {
                "start": 0.0,
                "end": 30.0,
                "text": (
                    "Ну вот, сегодня обсуждаем фудкорты в торговых центрах. "
                    "Родители часто спрашивают, почему дети постоянно там сидят."
                ),
            },
        ],
    }
    transcript_path = tmp_path / "episode.proofread.json"
    transcript_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")

    provider = FakeProvider([
        _sections_payload(
            [
                "Обсуждаются фудкорты в торговых центрах.",
                "Родители спрашивают, почему дети там сидят.",
            ],
            title="Фудкорты в торговых центрах",
        ),
    ])

    md_path = asyncio.run(run_article(
        transcript_path=transcript_path,
        provider=provider,
        quiet=True,
    ))

    # The ".proofread" infix must not leak into the article's name.
    assert md_path == tmp_path / "episode.article.md"
    markdown = md_path.read_text(encoding="utf-8")
    assert markdown.startswith("# episode\n")
    assert "## Фудкорты в торговых центрах" in markdown
    assert "Обсуждаются фудкорты в торговых центрах." in markdown

    meta = json.loads((tmp_path / "episode.article.json").read_text(encoding="utf-8"))
    assert meta["chunks_failed"] == 0
    assert meta["chunks_flagged"] == 0
    assert len(meta["sections"]) == 1
    assert meta["faithfulness"][0]["ok"] is True

    # The transcript itself is untouched.
    assert json.loads(transcript_path.read_text(encoding="utf-8")) == transcript
    # The prompt carried the source text.
    assert "фудкорты" in provider.prompts[0].lower()


def test_run_article_retries_then_flags_an_unfaithful_chunk(tmp_path: Path) -> None:
    """A drifting retelling is retried, and kept only with a flag."""
    transcript = {
        "language": "ru",
        "segments": [
            {"start": 0.0, "end": 30.0, "text": "Обсуждаем фудкорты в торговых центрах."},
        ],
    }
    transcript_path = tmp_path / "episode.json"
    transcript_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")

    invented = _sections_payload([
        "Спикер анализирует криптовалютные инструменты, ипотечное кредитование, "
        "туристические направления Азии и реконструкцию загородной недвижимости.",
    ])
    provider = FakeProvider([invented, invented])

    md_path = asyncio.run(run_article(
        transcript_path=transcript_path,
        provider=provider,
        quiet=True,
    ))

    assert len(provider.prompts) == 2, "an unfaithful chunk must be retried once"
    # The retry drops to temperature 0 and restates the constraint.
    assert provider.temperatures[1] == 0.0
    assert "ВНИМАНИЕ" in provider.prompts[1]

    meta = json.loads(md_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["chunks_flagged"] == 1
    assert meta["faithfulness"][0]["ok"] is False
    assert meta["faithfulness"][0]["reasons"]


def test_run_article_survives_a_failing_chunk(tmp_path: Path) -> None:
    """A dead chunk must not take the whole article down with it."""
    transcript = {
        "language": "ru",
        "segments": [
            {"start": 0.0, "end": 30.0, "text": "Обсуждаем фудкорты в торговых центрах."},
        ],
    }
    transcript_path = tmp_path / "episode.json"
    transcript_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")

    class ExplodingProvider:
        async def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
            raise RuntimeError("server down")

    md_path = asyncio.run(run_article(
        transcript_path=transcript_path,
        provider=ExplodingProvider(),
        quiet=True,
    ))

    meta = json.loads(md_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["chunks_failed"] == 1
    assert meta["sections"] == []
    assert md_path.read_text(encoding="utf-8").startswith("# episode")


def test_chunk_plain_text_strips_scout_markup() -> None:
    """The retelling must see plain speech, not the scout's timecoded markup."""
    from types import SimpleNamespace

    segments = [
        {"text": "Привет, друзья!"},
        {"text": "Сегодня обсуждаем фудкорты."},
        {"text": "Не вошло в этот чанк."},
    ]
    chunk = SimpleNamespace(
        source_segment_ids=(0, 1),
        text="[0-3] (Ведущий) Привет, друзья!\n[3-9] (Ведущий) Сегодня обсуждаем фудкорты.",
    )

    plain = chunk_plain_text(chunk, segments)

    assert plain == "Привет, друзья! Сегодня обсуждаем фудкорты."
    assert "[0-3]" not in plain
    assert "Ведущий" not in plain
    assert "Не вошло" not in plain


def test_chunk_plain_text_falls_back_to_stripping_the_markup() -> None:
    """Without usable segment ids, strip the prefixes off the rendered text."""
    from types import SimpleNamespace

    chunk = SimpleNamespace(
        source_segment_ids=(),
        text="[0-3] (None) Первая реплика.\n[3-9] Вторая реплика.",
    )

    assert chunk_plain_text(chunk, []) == "Первая реплика. Вторая реплика."


def test_chunking_does_not_label_undiarized_speech_as_none() -> None:
    """speaker=None must not become a literal "(None)" tag on every line."""
    from podcast_reels_forge.analysis.chunking import build_analysis_chunks

    segments = [
        {"start": 0.0, "end": 3.0, "text": "Первая реплика.", "speaker": None},
        {"start": 3.0, "end": 6.0, "text": "Вторая реплика.", "speaker": None},
    ]

    chunks = build_analysis_chunks(
        segments, chunk_seconds=600, max_chars=6000, overlap_seconds=0,
    )

    assert chunks
    assert "(None)" not in chunks[0].text
    # A real speaker label still comes through.
    labelled = build_analysis_chunks(
        [{"start": 0.0, "end": 3.0, "text": "Реплика.", "speaker": "SPEAKER_01"}],
        chunk_seconds=600, max_chars=6000, overlap_seconds=0,
    )
    assert "(SPEAKER_01)" in labelled[0].text
