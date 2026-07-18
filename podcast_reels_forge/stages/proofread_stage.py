"""RU: Стадия вычитки транскрипта: LLM исправляет орфографию и пунктуацию.

Модель (gemma4 через llama.cpp) получает сегменты транскрипта пакетами и
возвращает исправленный текст. Стадия защищена от «творчества» модели:
каждое исправление сравнивается с оригиналом по буквенному составу, и если
модель дописала, удалила или пересказала текст — правка отклоняется и
остаётся оригинал. Результат пишется в отдельный файл
``<имя>.proofread.json`` (+ ``.srt``), исходный транскрипт не изменяется.

EN: Transcript proofreading stage: an LLM fixes spelling and punctuation.

The model (gemma4 via llama.cpp) receives transcript segments in batches and
returns corrected text. A guardrail compares each correction with the
original by letter content: if the model added, removed or paraphrased
anything, the correction is rejected and the original text is kept. Output
goes to a separate ``<stem>.proofread.json`` (+ ``.srt``); the raw
transcript file is left untouched.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

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

DEFAULT_MAX_CHARS_CHUNK = 4000
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 600
DEFAULT_MIN_SIMILARITY = 0.80
DEFAULT_N_PREDICT = 4096

_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _normalize_for_compare(text: str) -> str:
    """RU: Нормализует текст для сравнения: без пунктуации/регистра, ё→е.

    EN: Normalize text for comparison: strip punctuation/case, fold ё→е.
    """
    lowered = str(text).lower().replace("ё", "е")
    return " ".join(_WORD_RE.findall(lowered))


def is_correction_safe(
    original: str,
    corrected: str,
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> bool:
    """RU: Проверяет, что исправление не искажает содержание сегмента.

    Разрешены правки орфографии, пунктуации и регистра (буквенный состав
    почти совпадает). Запрещены дописывание, удаление и пересказ: сильное
    расхождение нормализованных текстов или заметное изменение числа слов
    отклоняет правку.

    EN: Check that a correction does not distort the segment content.

    Spelling/punctuation/case fixes keep the normalized letter content nearly
    identical. Additions, deletions and paraphrases diverge strongly or shift
    the word count — such corrections are rejected.
    """
    orig_norm = _normalize_for_compare(original)
    corr_norm = _normalize_for_compare(corrected)

    if not corr_norm:
        # An empty correction may only "fix" an already empty segment.
        return not orig_norm
    if orig_norm == corr_norm:
        return True

    ratio = difflib.SequenceMatcher(None, orig_norm, corr_norm).ratio()
    if ratio < float(min_similarity):
        return False

    orig_words = orig_norm.split()
    corr_words = corr_norm.split()
    # Merging "по этому"→"поэтому" shifts the count slightly; a whole added
    # or dropped clause shifts it far more.
    max_word_drift = max(2, int(len(orig_words) * 0.2))
    return abs(len(corr_words) - len(orig_words)) <= max_word_drift


def build_proofread_batches(
    segments: list[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_MAX_CHARS_CHUNK,
) -> list[list[int]]:
    """RU: Группирует индексы сегментов в пакеты по бюджету символов.

    EN: Group segment indices into batches under a character budget.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    current_chars = 0
    budget = max(500, int(max_chars))

    for idx, seg in enumerate(segments):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if current and current_chars + len(text) > budget:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(idx)
        current_chars += len(text)

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


def _load_proofread_prompt(lang: str, variant: str = "default") -> str:
    repo_prompts = Path(__file__).resolve().parent.parent.parent / "prompts"
    candidates = [
        repo_prompts / lang / f"proofread_{variant}.txt",
        repo_prompts / lang / "proofread_default.txt",
        repo_prompts / "ru" / "proofread_default.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Proofread prompt not found in {repo_prompts}")


def _extract_corrections(payload: Any) -> dict[int, str]:
    """RU: Достаёт словарь id→текст из ответа модели.

    EN: Extract an id→text mapping from the model response.
    """
    items: list[Any] = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        raw = payload.get("segments")
        if isinstance(raw, list):
            items = raw

    corrections: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        if not isinstance(raw_id, (int, str)):
            continue
        try:
            sid = int(raw_id)
        except ValueError:
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            corrections[sid] = text.strip()
    return corrections


async def _proofread_batch(
    provider: LLMProvider,
    segments: list[dict[str, Any]],
    indices: list[int],
    *,
    prompt_template: str,
    temperature: float,
    timeout: int,
    min_similarity: float,
    verbose: bool = False,
) -> tuple[int, int]:
    """RU: Вычитывает один пакет сегментов, правит их на месте.

    Возвращает (применено, отклонено).

    EN: Proofread one batch of segments, editing them in place.

    Returns (applied, rejected).
    """
    payload = [
        {"id": idx, "text": str(segments[idx].get("text", "")).strip()}
        for idx in indices
    ]
    prompt = prompt_template.replace(
        "{segments_json}",
        json.dumps(payload, ensure_ascii=False),
    )

    raw = await provider.generate(prompt, temperature=temperature, timeout=timeout)
    try:
        parsed = extract_first_json_value(raw)
    except (ValueError, TypeError):
        preview = raw[:300].replace("\n", "\\n") if isinstance(raw, str) else str(raw)
        LOGGER.warning(
            "[proofread] failed to parse JSON from model output; keeping batch "
            "unchanged (preview: %s)",
            preview,
        )
        return 0, 0

    corrections = _extract_corrections(parsed)
    applied = 0
    rejected = 0
    for idx in indices:
        corrected = corrections.get(idx)
        if corrected is None:
            continue
        original = str(segments[idx].get("text", "")).strip()
        if corrected == original:
            continue
        if is_correction_safe(original, corrected, min_similarity=min_similarity):
            segments[idx]["text"] = corrected
            applied += 1
            if verbose:
                LOGGER.info("[proofread] fixed: %r -> %r", original, corrected)
        else:
            rejected += 1
            LOGGER.warning(
                "[proofread] rejected unsafe correction (content changed): %r -> %r",
                original,
                corrected,
            )
    return applied, rejected


def _proofread_output_path(transcript_path: Path) -> Path:
    return transcript_path.with_name(transcript_path.stem + ".proofread.json")


async def run_proofread(
    *,
    transcript_path: Path,
    output_path: Path | None = None,
    url: str = "http://127.0.0.1:11440/completion",
    model: str = "gemma4:26b",
    proofread_conf: Mapping[str, Any] | None = None,
    prompts_conf: Mapping[str, Any] | None = None,
    quiet: bool = False,
    verbose: bool = False,
    provider: LLMProvider | None = None,
) -> Path:
    """RU: Запускает вычитку транскрипта и пишет `.proofread.json` + `.srt`.

    EN: Run transcript proofreading and write `.proofread.json` + `.srt`.
    """
    # RU: Ленивый импорт: transcribe_stage тянет faster_whisper.
    # EN: Lazy import: transcribe_stage pulls in faster_whisper.
    from podcast_reels_forge.stages.transcribe_stage import (
        _build_sentence_groups,
        _dump_srt_output,
    )

    if not transcript_path.exists():
        raise SystemExit(f"Transcript not found: {transcript_path}")

    try:
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to read transcript: {exc}") from exc

    segments = data.get("segments", [])
    if not isinstance(segments, list):
        raise SystemExit("Transcript JSON is missing a segments list")

    conf = dict(proofread_conf or {})
    max_chars = int(conf.get("max_chars_chunk", DEFAULT_MAX_CHARS_CHUNK))
    temperature = float(conf.get("temperature", DEFAULT_TEMPERATURE))
    timeout = int(conf.get("timeout", DEFAULT_TIMEOUT))
    min_similarity = float(conf.get("min_similarity", DEFAULT_MIN_SIMILARITY))
    n_predict = int(conf.get("n_predict", DEFAULT_N_PREDICT))

    prompt_lang = _normalize_prompt_lang(
        str((prompts_conf or {}).get("language", "auto")),
        str(data.get("language") or ""),
    )
    prompt_template = _load_proofread_prompt(prompt_lang)

    if provider is None:
        provider = LlamaCppProvider(
            LlamaCppConfig(
                url=url,
                model=model,
                n_predict=n_predict,
            ),
        )

    segment_dicts = [seg for seg in segments if isinstance(seg, dict)]
    batches = build_proofread_batches(segment_dicts, max_chars=max_chars)

    if not quiet:
        LOGGER.info(
            "[proofread] model=%s segments=%d batches=%d lang=%s",
            model,
            len(segment_dicts),
            len(batches),
            prompt_lang,
        )

    applied_total = 0
    rejected_total = 0
    failed_batches = 0
    for batch_no, indices in enumerate(batches, 1):
        try:
            applied, rejected = await _proofread_batch(
                provider,
                segment_dicts,
                indices,
                prompt_template=prompt_template,
                temperature=temperature,
                timeout=timeout,
                min_similarity=min_similarity,
                verbose=verbose and not quiet,
            )
        except Exception as exc:
            # RU: Одна упавшая пачка не должна ронять стадию — текст остаётся как был.
            # EN: One failed batch must not kill the stage — that text stays as-is.
            failed_batches += 1
            LOGGER.warning(
                "[proofread] batch %d/%d failed (%s); keeping original text",
                batch_no,
                len(batches),
                exc,
            )
            continue
        applied_total += applied
        rejected_total += rejected
        if not quiet and verbose:
            LOGGER.info(
                "[proofread] batch %d/%d: applied=%d rejected=%d",
                batch_no,
                len(batches),
                applied,
                rejected,
            )

    data["sentences"] = _build_sentence_groups(segment_dicts)
    data["proofread"] = {
        "model": model,
        "prompt_lang": prompt_lang,
        "segments_total": len(segment_dicts),
        "applied": applied_total,
        "rejected": rejected_total,
        "failed_batches": failed_batches,
    }

    out_path = output_path or _proofread_output_path(transcript_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, data)
    _dump_srt_output(out_path.with_suffix(".srt"), segment_dicts)

    if not quiet:
        LOGGER.info(
            "[proofread] done: applied=%d rejected=%d failed_batches=%d saved=%s",
            applied_total,
            rejected_total,
            failed_batches,
            out_path,
        )
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """RU: Парсит аргументы CLI для стадии вычитки.

    EN: Parse CLI args for the proofreading stage.
    """
    ap = argparse.ArgumentParser(
        description="Proofread a transcript JSON with a local llama.cpp model.",
    )
    ap.add_argument("--transcript", type=Path, required=True, help="Path to transcript JSON")
    ap.add_argument("--output", type=Path, help="Output path (default: <stem>.proofread.json)")
    ap.add_argument(
        "--url",
        default="http://127.0.0.1:11440/completion",
        help="llama.cpp API URL",
    )
    ap.add_argument("--model", default="gemma4:26b", help="Model id (for logging/metadata)")
    ap.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS_CHUNK,
        help=f"Max source chars per LLM request (default: {DEFAULT_MAX_CHARS_CHUNK})",
    )
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="LLM request timeout, seconds")
    ap.add_argument(
        "--min-similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help="Reject corrections below this normalized-similarity threshold (0..1)",
    )
    ap.add_argument("--prompt-lang", default="auto", help="Prompt language: ru|en|auto")
    ap.add_argument("--quiet", action="store_true", help="Suppress non-error output")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """RU: CLI-точка входа для стадии вычитки.

    EN: CLI entrypoint for the proofreading stage.
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

        asyncio.run(run_proofread(
            transcript_path=args.transcript,
            output_path=args.output,
            url=args.url,
            model=args.model,
            proofread_conf={
                "max_chars_chunk": args.max_chars,
                "temperature": args.temperature,
                "timeout": args.timeout,
                "min_similarity": args.min_similarity,
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
