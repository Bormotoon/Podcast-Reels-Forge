"""RU: Стадия пересказа: транскрипт → связная статья с разделами.

Модель (gemma4 через llama.cpp) получает вычитанный транскрипт по частям и
превращает прямую речь в читаемый текст: абзацы, разделы по смыслу,
заголовки. Стадия защищена от «творчества» модели двумя проверками — объём
результата не должен превышать источник, а доля незнакомой лексики
(относительно исходного фрагмента) должна оставаться низкой. Нарушение —
повтор запроса; если и он не помог, результат сохраняется, но помечается
в метаданных, а не выдаётся за достоверный.

Результат: ``<имя>.article.md`` (для чтения) и ``<имя>.article.json``
(структура + метаданные проверок). Транскрипт не изменяется.

EN: Retelling stage: transcript -> a readable, sectioned article.

The model (gemma4 via llama.cpp) receives the proofread transcript in parts and
turns spoken dialogue into readable prose: paragraphs, meaning-based sections,
headings. Two guardrails keep it honest — the retelling may not outgrow its
source, and the share of vocabulary absent from that source must stay low. A
violation triggers one retry; if that also fails the result is kept but flagged
in the metadata rather than passed off as faithful.

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
# A faithful retelling compresses; it never meaningfully outgrows its source.
DEFAULT_MAX_LENGTH_RATIO = 1.10
# Share of long words absent from the source chunk. Paraphrasing introduces a
# few, wholesale invention introduces many.
DEFAULT_MAX_NOVEL_WORD_RATIO = 0.30

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


@dataclass(frozen=True)
class FaithfulnessReport:
    """Outcome of the guardrail checks for one chunk."""

    novel_ratio: float
    length_ratio: float
    max_novel_ratio: float
    max_length_ratio: float

    @property
    def invented_vocabulary(self) -> bool:
        return self.novel_ratio > self.max_novel_ratio

    @property
    def padded(self) -> bool:
        return self.length_ratio > self.max_length_ratio

    @property
    def ok(self) -> bool:
        return not (self.invented_vocabulary or self.padded)

    def reasons(self) -> list[str]:
        out: list[str] = []
        if self.invented_vocabulary:
            out.append(
                f"novel vocabulary {self.novel_ratio:.0%} > {self.max_novel_ratio:.0%}",
            )
        if self.padded:
            out.append(
                f"length {self.length_ratio:.0%} of source > {self.max_length_ratio:.0%}",
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "novel_word_ratio": round(self.novel_ratio, 4),
            "length_ratio": round(self.length_ratio, 4),
            "ok": self.ok,
            "reasons": self.reasons(),
        }


def check_faithfulness(
    source: str,
    retelling: str,
    *,
    max_novel_ratio: float = DEFAULT_MAX_NOVEL_WORD_RATIO,
    max_length_ratio: float = DEFAULT_MAX_LENGTH_RATIO,
) -> FaithfulnessReport:
    """RU: Проверяет, что пересказ не выдумывает и не раздувает текст.

    EN: Check that a retelling neither invents content nor pads it out.
    """
    return FaithfulnessReport(
        novel_ratio=novel_word_ratio(source, retelling),
        length_ratio=length_ratio(source, retelling),
        max_novel_ratio=float(max_novel_ratio),
        max_length_ratio=float(max_length_ratio),
    )


def _parse_sections(payload: Any) -> list[dict[str, Any]]:
    """Pull a section list out of whatever shape the model returned."""

    items: list[Any] = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("sections", "article", "parts", "chapters"):
            raw = payload.get(key)
            if isinstance(raw, list):
                items = raw
                break

    sections: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("heading") or "").strip()
        raw_paragraphs = item.get("paragraphs")
        if isinstance(raw_paragraphs, str):
            raw_paragraphs = [raw_paragraphs]
        if not isinstance(raw_paragraphs, list):
            raw_paragraphs = []
        paragraphs = [
            " ".join(str(p).split())
            for p in raw_paragraphs
            if isinstance(p, (str, int, float)) and str(p).strip()
        ]
        if not paragraphs:
            continue
        sections.append({"title": title, "paragraphs": paragraphs})
    return sections


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


def _normalize_prompt_lang(prompt_lang: str | None, transcript_lang: str | None) -> str:
    pl = (prompt_lang or "auto").strip().lower()
    if pl != "auto":
        return pl
    tl = (transcript_lang or "").strip().lower()
    if tl.startswith("en"):
        return "en"
    return "ru"


def _load_article_prompt(lang: str, variant: str = "default") -> str:
    repo_prompts = Path(__file__).resolve().parent.parent.parent / "prompts"
    candidates = [
        repo_prompts / lang / f"article_{variant}.txt",
        repo_prompts / lang / "article_default.txt",
        repo_prompts / "ru" / "article_default.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Article prompt not found in {repo_prompts}")


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
                "\n\nВНИМАНИЕ: предыдущий ответ добавил то, чего нет в исходном "
                "тексте. Пиши строго по фрагменту выше: никаких новых фактов, "
                "имён, чисел, выводов и вступлений от себя.\n"
                "WARNING: the previous answer added material absent from the "
                "source. Follow the fragment above strictly: no new facts, names, "
                "numbers, conclusions or introductions of your own."
            )

        # Temperature 0 on the retry: sampling variety is what let it drift.
        attempt_temperature = temperature if attempt == 1 else 0.0
        raw = await provider.generate(
            prompt,
            temperature=attempt_temperature,
            timeout=timeout,
        )
        try:
            parsed = extract_first_json_value(raw)
        except (ValueError, TypeError):
            preview = raw[:300].replace("\n", "\\n") if isinstance(raw, str) else str(raw)
            LOGGER.warning(
                "[article] failed to parse JSON (attempt %d); preview: %s",
                attempt,
                preview,
            )
            continue

        sections = _parse_sections(parsed)
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
    n_predict = int(conf.get("n_predict", DEFAULT_N_PREDICT))

    prompt_lang = _normalize_prompt_lang(
        str((prompts_conf or {}).get("language", "auto")),
        str(data.get("language") or ""),
    )
    prompt_template = _load_article_prompt(prompt_lang)

    if provider is None:
        provider = LlamaCppProvider(
            LlamaCppConfig(url=url, model=model, n_predict=n_predict),
        )

    segment_dicts = [seg for seg in segments if isinstance(seg, dict)]
    # No overlap: an overlapping window would retell the same words twice.
    chunks = build_analysis_chunks(
        segment_dicts,
        chunk_seconds=chunk_seconds,
        max_chars=max_chars,
        overlap_seconds=0,
    )

    if not quiet:
        LOGGER.info(
            "[article] model=%s segments=%d chunks=%d lang=%s",
            model,
            len(segment_dicts),
            len(chunks),
            prompt_lang,
        )

    sections: list[ArticleSection] = []
    flagged_chunks = 0
    failed_chunks = 0
    reports: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks, 1):
        chunk_text = chunk_plain_text(chunk, segment_dicts)
        if not chunk_text:
            continue
        try:
            raw_sections, report = await _write_one_chunk(
                provider,
                chunk_text=chunk_text,
                prompt_template=prompt_template,
                temperature=temperature,
                timeout=timeout,
                max_novel_ratio=max_novel_ratio,
                max_length_ratio=max_length_ratio,
            )
        except Exception as exc:
            # RU: Один упавший фрагмент не должен ронять всю статью.
            # EN: One failed chunk must not kill the whole article.
            failed_chunks += 1
            LOGGER.warning(
                "[article] chunk %d/%d failed (%s); it will be missing from the article",
                index,
                len(chunks),
                exc,
            )
            continue

        if not raw_sections:
            failed_chunks += 1
            LOGGER.warning(
                "[article] chunk %d/%d produced nothing usable", index, len(chunks),
            )
            continue

        if report is not None:
            entry = report.to_dict()
            entry["chunk"] = chunk.chunk_id
            reports.append(entry)
            if not report.ok:
                flagged_chunks += 1

        for raw in raw_sections:
            sections.append(
                ArticleSection(
                    title=raw["title"],
                    paragraphs=tuple(raw["paragraphs"]),
                    start=float(chunk.start),
                    end=float(chunk.end),
                ),
            )

        if verbose and not quiet:
            LOGGER.info(
                "[article] chunk %d/%d: sections=%d", index, len(chunks), len(raw_sections),
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
            "chunks_total": len(chunks),
            "chunks_failed": failed_chunks,
            "chunks_flagged": flagged_chunks,
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
        help="Reject a retelling longer than this multiple of its source",
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
            },
            prompts_conf={"language": args.prompt_lang},
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
