"""RU: Стадия лонгрида: транскрипт → читаемая статья с разделами.

Модель (gemma4 через llama.cpp) получает вычитанный транскрипт по частям и
приводит его в вид статьи: разделы по смыслу с заголовками, абзацы,
исправленные ошибки, без слов-паразитов и оговорок. Это НЕ пересказ: слова,
обороты и лицо автора сохраняются дословно.

Отклонения ловят три проверки — доля новой лексики (текст переписан своими
словами), длина сверху (дописано) и снизу вместе с сохранностью лексики
(сокращено). Нарушение — повтор запроса; если и он не помог, фрагмент
сохраняется, но помечается в метаданных, а не выдаётся за достоверный.

Результат: ``<имя>.article.md`` (для чтения) и ``<имя>.article.json``
(структура + метаданные проверок). Транскрипт не изменяется.

EN: Long-read stage: transcript -> a readable, sectioned article.

The model (gemma4 via llama.cpp) receives the proofread transcript in parts and
edits it into an article: meaning-based sections with headings, paragraphs,
corrected errors, no filler or slips. This is NOT a retelling — the author's
words, phrasing and grammatical person are kept verbatim.

Three guardrails catch drift: the share of new vocabulary (rewritten in the
model's own words), an upper length bound (padded), and a lower one together
with source-vocabulary coverage (abridged). A violation triggers one retry; if
that also fails the fragment is kept but flagged in the metadata rather than
passed off as faithful.

Output: ``<stem>.article.md`` (to read) and ``<stem>.article.json`` (structure
plus guardrail metadata). The transcript itself is left untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from podcast_reels_forge.analysis.chunking import build_analysis_chunks
from podcast_reels_forge.analysis.speaker_turns import (
    SpeakerTurn,
    build_speaker_turns,
    distinct_speakers,
    load_diarization,
    render_turns,
)
from podcast_reels_forge.analysis.serializers import atomic_write_json
from podcast_reels_forge.llm.providers import (
    LlamaCppConfig,
    LlamaCppProvider,
    LLMProvider,
)
from podcast_reels_forge.utils.json_utils import extract_first_json_value
from podcast_reels_forge.utils.llama_cpp_service import (
    ENV_MANAGED_BY_PIPELINE,
    llama_cpp_start,
    llama_cpp_stop,
    parse_local_llama_cpp_host_port,
)
from podcast_reels_forge.utils.logging_utils import setup_logging

LOGGER = setup_logging()

DEFAULT_CHUNK_SECONDS = 600
DEFAULT_MAX_CHARS_CHUNK = 6000
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT = 900
DEFAULT_N_PREDICT = 4096

# Thresholds calibrated against a hand-approved reference edit of a real
# 72-minute episode. On that reference: 3% of the words were new, the text kept
# 64% of the source's vocabulary and came out at 42% of its length (spoken
# filler is what disappears). A third-person paraphrase of the same episode
# scored 24% new words — which is exactly the failure these numbers must catch.
#
# The editor keeps the author's own words, so anything above a sliver of new
# vocabulary means it started writing instead of editing.
DEFAULT_MAX_NOVEL_WORD_RATIO = 0.15
# Padding: the edit removes filler, it never outgrows the transcript.
DEFAULT_MAX_LENGTH_RATIO = 1.15
# Summarizing: dropping this much text means it stopped editing and started
# abridging.
DEFAULT_MIN_LENGTH_RATIO = 0.25
# Share of the source's own vocabulary that must survive the edit.
DEFAULT_MIN_SOURCE_COVERAGE = 0.45

# Words shorter than this are function words and connectives — reused freely by
# any paraphrase, so they carry no signal about invented content.
_MIN_CONTENT_WORD_LEN = 6
# Crude stemming: Russian inflects endings while the stem stays put, so
# comparing prefixes avoids flagging "подкаста" as absent when the source said
# "подкаст".
_STEM_LEN = 5

_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class ArticleSection:
    """One titled section of the finished article."""

    title: str
    paragraphs: tuple[str, ...]
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "paragraphs": list(self.paragraphs),
            "start": round(self.start, 3),
            "end": round(self.end, 3),
        }

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


def _content_stems(text: str) -> set[str]:
    """RU: Основы значимых слов текста — для сравнения лексики.

    EN: Stems of the text's content words, for vocabulary comparison.
    """
    lowered = str(text).lower().replace("ё", "е")
    return {
        word[:_STEM_LEN]
        for word in _WORD_RE.findall(lowered)
        if len(word) >= _MIN_CONTENT_WORD_LEN
    }


def novel_word_ratio(source: str, retelling: str) -> float:
    """RU: Доля основ из пересказа, которых нет в источнике (0..1).

    EN: Share of the retelling's word stems that the source never used (0..1).
    """
    retelling_stems = _content_stems(retelling)
    if not retelling_stems:
        return 0.0
    source_stems = _content_stems(source)
    novel = retelling_stems - source_stems
    return len(novel) / len(retelling_stems)


def length_ratio(source: str, retelling: str) -> float:
    """Retelling length relative to its source, by character count."""

    source_len = len(str(source).strip())
    if source_len <= 0:
        return 0.0
    return len(str(retelling).strip()) / source_len


def source_coverage(source: str, edited: str) -> float:
    """RU: Доля основ источника, уцелевших в отредактированном тексте (0..1).

    EN: Share of the source's word stems that survive into the edited text.

    Catches the opposite failure from ``novel_word_ratio``: an edit that quietly
    summarizes the fragment instead of tidying it up.
    """
    source_stems = _content_stems(source)
    if not source_stems:
        return 1.0
    return len(source_stems & _content_stems(edited)) / len(source_stems)


@dataclass(frozen=True)
class FaithfulnessReport:
    """Outcome of the guardrail checks for one chunk."""

    novel_ratio: float
    length_ratio: float
    coverage: float
    max_novel_ratio: float
    max_length_ratio: float
    min_length_ratio: float
    min_coverage: float

    @property
    def invented_vocabulary(self) -> bool:
        """The editor started writing its own words instead of keeping the author's."""

        return self.novel_ratio > self.max_novel_ratio

    @property
    def padded(self) -> bool:
        return self.length_ratio > self.max_length_ratio

    @property
    def abridged(self) -> bool:
        """It summarized rather than edited."""

        return self.length_ratio < self.min_length_ratio or self.coverage < self.min_coverage

    @property
    def ok(self) -> bool:
        return not (self.invented_vocabulary or self.padded or self.abridged)

    def reasons(self) -> list[str]:
        out: list[str] = []
        if self.invented_vocabulary:
            out.append(
                f"new vocabulary {self.novel_ratio:.0%} > {self.max_novel_ratio:.0%} "
                "(rewritten instead of edited)",
            )
        if self.padded:
            out.append(
                f"length {self.length_ratio:.0%} of source > {self.max_length_ratio:.0%}",
            )
        if self.length_ratio < self.min_length_ratio:
            out.append(
                f"length {self.length_ratio:.0%} of source < {self.min_length_ratio:.0%} "
                "(summarized)",
            )
        if self.coverage < self.min_coverage:
            out.append(
                f"kept only {self.coverage:.0%} of the source vocabulary "
                f"< {self.min_coverage:.0%}",
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "novel_word_ratio": round(self.novel_ratio, 4),
            "length_ratio": round(self.length_ratio, 4),
            "source_coverage": round(self.coverage, 4),
            "ok": self.ok,
            "reasons": self.reasons(),
        }


def check_faithfulness(
    source: str,
    retelling: str,
    *,
    max_novel_ratio: float = DEFAULT_MAX_NOVEL_WORD_RATIO,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
    min_length_ratio: float = DEFAULT_MIN_LENGTH_RATIO,
    min_coverage: float = DEFAULT_MIN_SOURCE_COVERAGE,
) -> FaithfulnessReport:
    """RU: Проверяет, что текст отредактирован, а не переписан или сокращён.

    EN: Check that the text was edited — not rewritten, padded or abridged.
    """
    return FaithfulnessReport(
        novel_ratio=novel_word_ratio(source, retelling),
        length_ratio=length_ratio(source, retelling),
        coverage=source_coverage(source, retelling),
        max_novel_ratio=float(max_novel_ratio),
        max_length_ratio=float(max_length_ratio),
        min_length_ratio=float(min_length_ratio),
        min_coverage=float(min_coverage),
    )


def parse_markdown_sections(text: str) -> list[dict[str, Any]]:
    """RU: Разбирает markdown-ответ модели на разделы и абзацы.

    EN: Parse the model's markdown answer into sections and paragraphs.

    Markdown rather than JSON on purpose: the edited text is long, near-verbatim
    and full of quotes and dashes, and JSON escaping of that reliably broke —
    one run leaked a literal ``paragraphs [`` into the prose. Headings and blank
    lines have nothing to escape.
    """
    sections: list[dict[str, Any]] = []
    current_title = ""
    buffer: list[str] = []

    def flush() -> None:
        paragraphs = [" ".join(block.split()) for block in buffer if block.strip()]
        paragraphs = [p for p in paragraphs if p and not _is_scaffolding(p)]
        if paragraphs:
            sections.append({"title": current_title, "paragraphs": paragraphs})
        buffer.clear()

    block: list[str] = []
    for raw_line in _strip_code_fences(text).splitlines():
        line = raw_line.rstrip()
        heading = _HEADING_RE.match(line)
        if heading:
            if block:
                buffer.append(" ".join(block))
                block = []
            flush()
            current_title = heading.group(1).strip()
            continue
        if not line.strip():
            if block:
                buffer.append(" ".join(block))
                block = []
            continue
        block.append(line.strip())

    if block:
        buffer.append(" ".join(block))
    flush()
    return sections


_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
# Leftovers of a model that started answering in JSON or narrating its own work.
_SCAFFOLDING_RE = re.compile(
    r"^\s*(\{|\}|\[|\]|\"?(sections|paragraphs|title|article)\"?\s*[:\[]|```)",
    flags=re.IGNORECASE,
)


def _strip_code_fences(text: str) -> str:
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned


def _is_scaffolding(paragraph: str) -> bool:
    return bool(_SCAFFOLDING_RE.match(paragraph))


def chunk_plain_text(
    chunk: Any,
    segments: Sequence[Mapping[str, Any]],
) -> str:
    """RU: Чистая речь фрагмента — без таймкодов и меток спикера.

    Готовый ``chunk.text`` размечен для scout-стадии (``[12-34] (Ведущий) …``).
    Пересказу эта разметка не нужна, и она ломает проверку длины: служебные
    префиксы раздувают «источник» примерно на треть, и раздутый пересказ
    прошёл бы порог.

    EN: The fragment's plain speech — no timecodes, no speaker tags.

    The ready-made ``chunk.text`` is marked up for the scout stage
    (``[12-34] (Host) …``). A retelling does not need that markup, and it breaks
    the length guardrail: the prefixes inflate the "source" by roughly a third,
    which would let a padded retelling slip under the threshold.
    """
    parts: list[str] = []
    for index in getattr(chunk, "source_segment_ids", ()) or ():
        if not 0 <= int(index) < len(segments):
            continue
        text = str(segments[int(index)].get("text", "")).strip()
        if text:
            parts.append(text)
    if parts:
        return " ".join(parts)

    # No usable segment ids: fall back to the marked-up text with the
    # per-line prefixes stripped off.
    return " ".join(
        _CHUNK_LINE_PREFIX_RE.sub("", line).strip()
        for line in str(getattr(chunk, "text", "")).splitlines()
    ).strip()


_CHUNK_LINE_PREFIX_RE = re.compile(r"^\s*\[\d+-\d+\]\s*(\([^)]*\)\s*)?")


def chunk_turns(
    turns: Sequence[SpeakerTurn],
    *,
    max_chars: int,
    names: Mapping[str, str] | None = None,
) -> list[list[SpeakerTurn]]:
    """RU: Режет реплики на пачки по бюджету символов, не разрывая реплику.

    EN: Split turns into batches under a character budget, never mid-turn.

    A turn cut in half would strand its speaker label, so an oversized single
    turn simply gets a batch of its own.
    """
    mapping = dict(names or {})
    budget = max(500, int(max_chars))
    batches: list[list[SpeakerTurn]] = []
    current: list[SpeakerTurn] = []
    current_chars = 0

    for turn in turns:
        cost = len(mapping.get(turn.speaker, turn.speaker)) + 2 + len(turn.text) + 1
        if current and current_chars + cost > budget:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(turn)
        current_chars += cost

    if current:
        batches.append(current)
    return batches


def _normalize_prompt_lang(prompt_lang: str | None, transcript_lang: str | None) -> str:
    pl = (prompt_lang or "auto").strip().lower()
    if pl != "auto":
        return pl
    tl = (transcript_lang or "").strip().lower()
    if tl.startswith("en"):
        return "en"
    return "ru"


def _prompt_path(lang: str, *names: str) -> str:
    repo_prompts = Path(__file__).resolve().parent.parent.parent / "prompts"
    for name in names:
        for base in (repo_prompts / lang, repo_prompts / "ru"):
            candidate = base / f"{name}.txt"
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt not found in {repo_prompts}: {names}")


def _load_article_prompt(lang: str, variant: str = "default") -> str:
    return _prompt_path(lang, f"article_{variant}", "article_default")


async def resolve_speaker_names(
    provider: LLMProvider,
    turns: Sequence[SpeakerTurn],
    *,
    lang: str,
    timeout: int,
    max_chars: int = 6000,
) -> dict[str, str]:
    """RU: Выясняет имена спикеров из самого разговора.

    EN: Work out the speakers' names from the conversation itself.

    Diarization only ever produces SPEAKER_00/01. People introduce each other in
    the opening minutes, so the model is shown that opening and asked who is
    who. Labels it cannot name keep their technical id rather than getting a
    made-up "Host".
    """
    ids = distinct_speakers(turns)
    if len(ids) < 2:
        return {}

    # RU: Смотрим оба конца эпизода. Представляются в начале, но прощаются
    #     поимённо в конце — на реальном выпуске имя ведущего звучало ровно
    #     один раз, за 300 символов до финала.
    # EN: Look at both ends of the episode. People introduce themselves at the
    #     start, but the sign-off names everyone — on a real episode the host's
    #     name appeared exactly once, 300 characters before the end.
    whole = render_turns(turns, names=None)
    if len(whole) <= max_chars:
        excerpt = whole
    else:
        head = int(max_chars * 0.6)
        tail = max_chars - head
        excerpt = f"{whole[:head]}\n\n[...]\n\n{whole[-tail:]}"

    prompt = _prompt_path(lang, "speaker_names_default").replace("{transcript}", excerpt)

    try:
        raw = await provider.generate(prompt, temperature=0.0, timeout=timeout)
        payload = extract_first_json_value(raw)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[article] speaker naming failed (%s); keeping the ids", exc)
        return {}

    entries: list[Any] = []
    if isinstance(payload, Mapping):
        raw_entries = payload.get("speakers")
        if isinstance(raw_entries, list):
            entries = raw_entries
    elif isinstance(payload, list):
        entries = payload

    names: dict[str, str] = {}
    used: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        speaker_id = str(entry.get("id", "")).strip()
        name = " ".join(str(entry.get("name", "")).split()).strip()
        if speaker_id not in ids or not name:
            continue
        # Two labels answering to one name would make the turns unreadable.
        if name.casefold() in used:
            continue
        used.add(name.casefold())
        names[speaker_id] = name

    if names:
        LOGGER.info(
            "[article] speakers: %s",
            ", ".join(f"{k}={v}" for k, v in sorted(names.items())),
        )
    return names


def merge_adjacent_sections(sections: Sequence[ArticleSection]) -> list[ArticleSection]:
    """RU: Склеивает соседние разделы с одинаковым заголовком.

    Разговор не обрывается на границе куска, поэтому одна и та же тема может
    прийти из двух запросов подряд — в статье это должен быть один раздел.

    EN: Merge neighbouring sections that share a title.

    A conversation does not stop at a chunk boundary, so the same topic can come
    back from two consecutive requests — in the article it must read as one
    section.
    """
    merged: list[ArticleSection] = []
    for section in sections:
        previous = merged[-1] if merged else None
        if previous is not None and _same_title(previous.title, section.title):
            merged[-1] = ArticleSection(
                title=previous.title,
                paragraphs=previous.paragraphs + section.paragraphs,
                start=previous.start,
                end=section.end,
            )
            continue
        merged.append(section)
    return merged


def _same_title(left: str, right: str) -> bool:
    def norm(value: str) -> str:
        return " ".join(str(value).lower().replace("ё", "е").split()).strip(" .:—-")

    return bool(norm(left)) and norm(left) == norm(right)


def render_article_markdown(title: str, sections: Sequence[ArticleSection]) -> str:
    """RU: Собирает финальный markdown статьи.

    EN: Render the final article markdown.
    """
    lines: list[str] = [f"# {title}".rstrip(), ""]
    for section in sections:
        if section.title:
            lines.append(f"## {section.title}")
            lines.append("")
        for paragraph in section.paragraphs:
            lines.append(paragraph)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def _write_one_chunk(
    provider: LLMProvider,
    *,
    chunk_text: str,
    prompt_template: str,
    temperature: float,
    timeout: int,
    max_novel_ratio: float,
    max_length_ratio: float,
    min_length_ratio: float,
    min_coverage: float,
) -> tuple[list[dict[str, Any]], FaithfulnessReport | None]:
    """RU: Пересказывает один фрагмент, при нарушении — один повтор.

    EN: Retell one chunk, retrying once when a guardrail trips.
    """
    attempts = 2
    best: tuple[list[dict[str, Any]], FaithfulnessReport] | None = None

    for attempt in range(1, attempts + 1):
        prompt = prompt_template.replace("{transcript}", chunk_text)
        if attempt > 1:
            # Second pass: the first drifted, so restate the hard constraint.
            prompt += (
                "\n\nВНИМАНИЕ: предыдущий ответ отклонился от исходного текста — "
                "был переписан своими словами, сокращён или дополнен. Ты редактор, "
                "а не автор: сохраняй слова и обороты говорящего дословно, от того "
                "же лица, ничего не добавляя и не сокращая. Убирать можно только "
                "слова-паразиты, оговорки и дословные повторы.\n"
                "WARNING: the previous answer drifted from the source — it was "
                "rewritten, abridged or padded. You are an editor, not an author: "
                "keep the speaker's words and phrasing verbatim, in the same "
                "person, adding and cutting nothing. Only filler, slips and "
                "verbatim repetitions may go."
            )

        # Temperature 0 on the retry: sampling variety is what let it drift.
        attempt_temperature = temperature if attempt == 1 else 0.0
        raw = await provider.generate(
            prompt,
            temperature=attempt_temperature,
            timeout=timeout,
        )
        sections = parse_markdown_sections(raw if isinstance(raw, str) else str(raw))
        if not sections:
            LOGGER.warning("[article] model returned no usable sections (attempt %d)", attempt)
            continue

        produced = "\n\n".join(
            "\n\n".join(section["paragraphs"]) for section in sections
        )
        report = check_faithfulness(
            chunk_text,
            produced,
            max_novel_ratio=max_novel_ratio,
            max_length_ratio=max_length_ratio,
            min_length_ratio=min_length_ratio,
            min_coverage=min_coverage,
        )
        if report.ok:
            return sections, report

        LOGGER.warning(
            "[article] guardrail tripped on attempt %d: %s",
            attempt,
            "; ".join(report.reasons()),
        )
        if best is None or report.novel_ratio < best[1].novel_ratio:
            best = (sections, report)

    if best is not None:
        LOGGER.warning(
            "[article] keeping the closest attempt but flagging it: %s",
            "; ".join(best[1].reasons()),
        )
        return best[0], best[1]
    return [], None


async def run_article(
    *,
    transcript_path: Path,
    output_path: Path | None = None,
    url: str = "http://127.0.0.1:11440/completion",
    model: str = "gemma4:26b",
    article_conf: Mapping[str, Any] | None = None,
    prompts_conf: Mapping[str, Any] | None = None,
    diarization_path: Path | None = None,
    title: str | None = None,
    quiet: bool = False,
    verbose: bool = False,
    provider: LLMProvider | None = None,
) -> Path:
    """RU: Строит статью-пересказ и пишет `.article.md` + `.article.json`.

    EN: Build the retelling article and write `.article.md` + `.article.json`.
    """
    if not transcript_path.exists():
        raise SystemExit(f"Transcript not found: {transcript_path}")

    try:
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to read transcript: {exc}") from exc

    segments = data.get("segments", [])
    if not isinstance(segments, list):
        raise SystemExit("Transcript JSON is missing a segments list")

    conf = dict(article_conf or {})
    chunk_seconds = int(conf.get("chunk_seconds", DEFAULT_CHUNK_SECONDS))
    max_chars = int(conf.get("max_chars_chunk", DEFAULT_MAX_CHARS_CHUNK))
    temperature = float(conf.get("temperature", DEFAULT_TEMPERATURE))
    timeout = int(conf.get("timeout", DEFAULT_TIMEOUT))
    max_novel_ratio = float(conf.get("max_novel_word_ratio", DEFAULT_MAX_NOVEL_WORD_RATIO))
    max_length_ratio = float(conf.get("max_length_ratio", DEFAULT_MAX_LENGTH_RATIO))
    min_length_ratio = float(conf.get("min_length_ratio", DEFAULT_MIN_LENGTH_RATIO))
    min_coverage = float(conf.get("min_source_coverage", DEFAULT_MIN_SOURCE_COVERAGE))
    n_predict = int(conf.get("n_predict", DEFAULT_N_PREDICT))

    prompt_lang = _normalize_prompt_lang(
        str((prompts_conf or {}).get("language", "auto")),
        str(data.get("language") or ""),
    )
    if provider is None:
        provider = LlamaCppProvider(
            # Prose, not JSON: the article comes back as markdown.
            LlamaCppConfig(url=url, model=model, n_predict=n_predict, json_output=False),
        )

    segment_dicts = [seg for seg in segments if isinstance(seg, dict)]

    # RU: С диаризацией единица текста — реплика, а не сегмент: Whisper режет по
    #     паузам, и один его сегмент запросто содержит реплики троих.
    # EN: With diarization the unit is a turn, not a segment: Whisper splits on
    #     pauses, and one of its segments happily holds three people talking.
    turns = build_speaker_turns(segment_dicts, load_diarization(diarization_path))
    speaker_names: dict[str, str] = {}
    if turns and len(distinct_speakers(turns)) > 1:
        speaker_names = await resolve_speaker_names(
            provider, turns, lang=prompt_lang, timeout=timeout,
        )
        prompt_template = _load_article_prompt(prompt_lang, "speakers")
        chunk_texts = [
            render_turns(batch, names=speaker_names)
            for batch in chunk_turns(turns, max_chars=max_chars, names=speaker_names)
        ]
        chunk_bounds = [
            (batch[0].start, batch[-1].end)
            for batch in chunk_turns(turns, max_chars=max_chars, names=speaker_names)
        ]
    else:
        turns = []
        prompt_template = _load_article_prompt(prompt_lang)
        # No overlap: an overlapping window would edit the same words twice.
        chunks = build_analysis_chunks(
            segment_dicts,
            chunk_seconds=chunk_seconds,
            max_chars=max_chars,
            overlap_seconds=0,
        )
        chunk_texts = [chunk_plain_text(chunk, segment_dicts) for chunk in chunks]
        chunk_bounds = [(float(chunk.start), float(chunk.end)) for chunk in chunks]

    if not quiet:
        LOGGER.info(
            "[article] model=%s segments=%d chunks=%d lang=%s speakers=%s",
            model,
            len(segment_dicts),
            len(chunk_texts),
            prompt_lang,
            ", ".join(speaker_names.values()) if speaker_names
            else (f"{len(distinct_speakers(turns))} unnamed" if turns else "none"),
        )

    sections: list[ArticleSection] = []
    flagged_chunks = 0
    failed_chunks = 0
    reports: list[dict[str, Any]] = []

    for index, chunk_text in enumerate(chunk_texts, 1):
        if not chunk_text:
            continue
        chunk_start, chunk_end = chunk_bounds[index - 1]
        try:
            raw_sections, report = await _write_one_chunk(
                provider,
                chunk_text=chunk_text,
                prompt_template=prompt_template,
                temperature=temperature,
                timeout=timeout,
                max_novel_ratio=max_novel_ratio,
                max_length_ratio=max_length_ratio,
                min_length_ratio=min_length_ratio,
                min_coverage=min_coverage,
            )
        except Exception as exc:
            # RU: Один упавший фрагмент не должен ронять всю статью.
            # EN: One failed chunk must not kill the whole article.
            failed_chunks += 1
            LOGGER.warning(
                "[article] chunk %d/%d failed (%s); it will be missing from the article",
                index,
                len(chunk_texts),
                exc,
            )
            continue

        if not raw_sections:
            failed_chunks += 1
            LOGGER.warning(
                "[article] chunk %d/%d produced nothing usable", index, len(chunk_texts),
            )
            continue

        if report is not None:
            entry = report.to_dict()
            entry["chunk"] = f"chunk_{index:03d}"
            reports.append(entry)
            if not report.ok:
                flagged_chunks += 1

        for raw in raw_sections:
            sections.append(
                ArticleSection(
                    title=raw["title"],
                    paragraphs=tuple(raw["paragraphs"]),
                    start=float(chunk_start),
                    end=float(chunk_end),
                ),
            )

        if verbose and not quiet:
            LOGGER.info(
                "[article] chunk %d/%d: sections=%d", index, len(chunk_texts), len(raw_sections),
            )

    sections = merge_adjacent_sections(sections)

    article_title = title or transcript_path.stem.replace(".proofread", "")
    markdown = render_article_markdown(article_title, sections)

    md_path = output_path or transcript_path.with_name(
        transcript_path.stem.replace(".proofread", "") + ".article.md",
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")

    atomic_write_json(
        md_path.with_suffix(".json"),
        {
            "title": article_title,
            "source_transcript": str(transcript_path.resolve()),
            "model": model,
            "prompt_lang": prompt_lang,
            "chunks_total": len(chunk_texts),
            "chunks_failed": failed_chunks,
            "chunks_flagged": flagged_chunks,
            "speakers": speaker_names or {
                sid: sid for sid in distinct_speakers(turns)
            },
            "sections": [section.to_dict() for section in sections],
            "faithfulness": reports,
        },
    )

    if not quiet:
        LOGGER.info(
            "[article] done: sections=%d flagged=%d failed=%d saved=%s",
            len(sections),
            flagged_chunks,
            failed_chunks,
            md_path,
        )
    return md_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """RU: Парсит аргументы CLI для стадии пересказа.

    EN: Parse CLI args for the retelling stage.
    """
    ap = argparse.ArgumentParser(
        description="Turn a transcript into a readable, sectioned article.",
    )
    ap.add_argument("--transcript", type=Path, required=True, help="Path to transcript JSON")
    ap.add_argument("--output", type=Path, help="Output .md path (default: <stem>.article.md)")
    ap.add_argument("--title", help="Article title (default: the transcript file stem)")
    ap.add_argument(
        "--diarization",
        type=Path,
        help="diarization.json; turns the article into speaker-tagged text",
    )
    ap.add_argument(
        "--url",
        default="http://127.0.0.1:11440/completion",
        help="llama.cpp API URL",
    )
    ap.add_argument("--model", default="gemma4:26b", help="Model id (for logging/metadata)")
    ap.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"Transcript window per request (default: {DEFAULT_CHUNK_SECONDS})",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS_CHUNK,
        help=f"Max source chars per request (default: {DEFAULT_MAX_CHARS_CHUNK})",
    )
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument(
        "--max-novel-word-ratio",
        type=float,
        default=DEFAULT_MAX_NOVEL_WORD_RATIO,
        help="Reject a retelling whose vocabulary strays further than this (0..1)",
    )
    ap.add_argument(
        "--max-length-ratio",
        type=float,
        default=DEFAULT_MAX_LENGTH_RATIO,
        help="Reject an edit longer than this multiple of its source",
    )
    ap.add_argument(
        "--min-length-ratio",
        type=float,
        default=DEFAULT_MIN_LENGTH_RATIO,
        help="Reject an edit shorter than this multiple of its source (summarized)",
    )
    ap.add_argument(
        "--min-source-coverage",
        type=float,
        default=DEFAULT_MIN_SOURCE_COVERAGE,
        help="Reject an edit that keeps less than this share of the source vocabulary",
    )
    ap.add_argument("--prompt-lang", default="auto", help="Prompt language: ru|en|auto")
    ap.add_argument("--quiet", action="store_true", help="Suppress non-error output")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """RU: CLI-точка входа для стадии пересказа.

    EN: CLI entrypoint for the retelling stage.
    """
    args = parse_args(argv)

    global LOGGER
    LOGGER = setup_logging(verbose=bool(args.verbose), quiet=bool(args.quiet))

    proc: subprocess.Popen | None = None
    try:
        managed_by_pipeline = os.environ.get(ENV_MANAGED_BY_PIPELINE) == "1"
        local = parse_local_llama_cpp_host_port(args.url) if args.url else None
        if local and not managed_by_pipeline:
            host, port = local
            proc = llama_cpp_start(host=host, port=port, service_conf={})

        asyncio.run(run_article(
            transcript_path=args.transcript,
            output_path=args.output,
            url=args.url,
            model=args.model,
            title=args.title,
            article_conf={
                "chunk_seconds": args.chunk_seconds,
                "max_chars_chunk": args.max_chars,
                "temperature": args.temperature,
                "timeout": args.timeout,
                "max_novel_word_ratio": args.max_novel_word_ratio,
                "max_length_ratio": args.max_length_ratio,
                "min_length_ratio": args.min_length_ratio,
                "min_source_coverage": args.min_source_coverage,
            },
            prompts_conf={"language": args.prompt_lang},
            diarization_path=args.diarization,
            quiet=bool(args.quiet),
            verbose=bool(args.verbose),
        ))
    finally:
        if proc:
            llama_cpp_stop(proc)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        sys.exit(130)
