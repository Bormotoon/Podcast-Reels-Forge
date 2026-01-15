#!/usr/bin/env python3
"""RU: Анализирует transcript.json через LLM, чтобы найти вирусные моменты и метаданные.

EN: Analyze transcript.json with an LLM to find viral moments and generate metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable: object, **_: object) -> object:
        return iterable

from podcast_reels_forge.llm.providers import (
    AnthropicConfig,
    AnthropicProvider,
    GeminiConfig,
    GeminiProvider,
    LLMProvider,
    OllamaConfig,
    OllamaProvider,
    OpenAIConfig,
    OpenAIProvider,
)
from podcast_reels_forge.utils.json_utils import extract_first_json_value
from podcast_reels_forge.utils.logging_utils import setup_logging
from podcast_reels_forge.utils.ollama_service import (
    ENV_MANAGED_BY_PIPELINE,
    ollama_start,
    ollama_stop,
    parse_local_ollama_host_port,
)

LOGGER = setup_logging()


@dataclass
class Moment:
    """RU: Представляет потенциальный «вирусный» клип.

    EN: Represents a potential viral clip.
    """

    start: float
    end: float
    title: str
    quote: str
    why: str
    score: float
    clip_type: str = "reel"
    hook: str = ""
    caption: str = ""
    hashtags: list[str] | None = None
    file: str = ""


def _status(msg: str, *, quiet: bool) -> None:
    if not quiet:
        LOGGER.info(msg)


def fmt_hms(sec: float) -> str:
    """RU: Форматирует секунды как HH:MM:SS или MM:SS.

    EN: Format seconds for display as HH:MM:SS or MM:SS.
    """
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def _parse_local_ollama_host_port(url: str) -> tuple[str, int] | None:
    return parse_local_ollama_host_port(url)


def get_llm_json(
    provider: LLMProvider, prompt: str, temperature: float, timeout: int,
) -> dict[str, Any] | list[Any]:
    """RU: Получает JSON-ответ от LLM (одна попытка, без повторных запросов).

    EN: Get JSON response from LLM (single attempt, no extra repair call).
    """
    raw = provider.generate(prompt, temperature=temperature, timeout=timeout)
    try:
        return extract_first_json_value(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        # RU: Не делаем второй LLM-вызов (пользователь просит без итераций).
        # EN: No second LLM call (user asked to avoid extra iterations).
        preview = raw[:500].replace("\n", "\\n") if isinstance(raw, str) else str(raw)
        LOGGER.warning("Failed to parse JSON from LLM output; returning [] (raw preview: %s)", preview)
        return []


# ----------------------------
# Analysis Logic
# ----------------------------


def segments_to_compact_text(segments: list[dict[str, Any]], max_chars: int) -> str:
    """RU: Преобразует сегменты в плотный текстовый формат для LLM.

    EN: Convert segments to a dense text format for LLM input.
    """
    lines = []
    for seg in segments:
        start = int(seg.get('start', 0))
        end = int(seg.get('end', 0))
        text = seg.get('text', '').strip()
        if text:
            lines.append(f"[{start}-{end}] {text}")
    
    result = "\n".join(lines)
    if len(result) > max_chars:
        # Truncate at word boundary if possible
        result = result[:max_chars]
        last_newline = result.rfind('\n')
        if last_newline > max_chars * 0.8:
            result = result[:last_newline]
    return result


def _render_prompt(template: str, values: dict[str, str]) -> str:
    """Render a prompt template by replacing known placeholders.

    We intentionally avoid `str.format()` because prompt files may contain JSON
    examples with braces, which would be interpreted as format fields.
    """
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out


def chunk_segments_by_time(
    segments: list[dict[str, Any]], chunk_seconds: int,
) -> list[list[dict[str, Any]]]:
    """RU: Делит сегменты на чанки по времени.

    EN: Divide segments into time-based chunks.
    """
    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    chunk_start = segments[0]["start"] if segments else 0
    for s in segments:
        if s["end"] - chunk_start <= chunk_seconds:
            cur.append(s)
        else:
            if cur:
                chunks.append(cur)
            cur = [s]
            chunk_start = s["start"]
    if cur:
        chunks.append(cur)
    return chunks


def find_moments(
    provider: LLMProvider,
    segments: list[dict[str, Any]],
    duration: float,
    r_min: int,
    r_max: int,
    count: int,
    chunk_sec: int,
    max_ch: int,
    timeout: int,
    progress: bool = False,
    *,
    ch_prompt: str,
    select_prompt: str,
    stories_count: int = 0,
    reels_count: int = 0,
    long_reels_count: int = 0,
    highlights_moments: int = 0,
) -> list[Moment]:
    """RU: Оркестрирует анализ по чанкам и финальный выбор вирусных моментов.

    EN: Orchestrate chunk-based analysis and final selection of viral moments.
    """
    candidates: list[dict[str, Any]] = []
    chunks = chunk_segments_by_time(segments, chunk_sec)
    it = enumerate(chunks, 1)
    if progress:
        it = tqdm(it, total=len(chunks), desc="analyze")

    # Build a requirements string for the prompt
    reqs = []
    if stories_count > 0:
        reqs.append(f"Stories (up to 15s): {stories_count}")
    if reels_count > 0:
        reqs.append(f"Reels (up to 60s): {reels_count}")
    if long_reels_count > 0:
        reqs.append(f"Long Reels (up to 180s): {long_reels_count}")
    if highlights_moments > 0:
        reqs.append(f"Hot moments for highlights: {highlights_moments}")

    reqs_str = "\n".join(reqs) if reqs else f"Viral moments ({r_min}-{r_max}s): {count}"

    for _i, ch in it:
        ch_txt = segments_to_compact_text(ch, max_ch)
        prompt = _render_prompt(
            ch_prompt,
            {
                "r_min": str(r_min),
                "r_max": str(r_max),
                "transcript": ch_txt,
                "requirements": reqs_str,
            },
        )
        # LLM might return "moments" key or a direct list
        resp = get_llm_json(provider, prompt, 0.3, timeout)
        chunk_moments: list[Any] = []
        if isinstance(resp, list):
            chunk_moments = resp
        elif isinstance(resp, dict):
            val = resp.get("moments")
            if isinstance(val, list):
                chunk_moments = val
            else:
                m_val = resp.get("moment")
                chunk_moments = [m_val] if m_val else []

        for moment in chunk_moments:
            try:
                if not isinstance(moment, dict):
                    continue
                start = float(moment.get("start", -1))
                end = float(moment.get("end", -1))
            except (TypeError, ValueError, AttributeError):
                continue

            if 0 <= start < end <= duration:
                candidates.append(moment)

    if not candidates:
        LOGGER.info("No candidates found in transcript chunks")
        return []

    # RU: Без финального selection-шага: выбираем топ моментов локально.
    # EN: No final selection step: pick top moments locally.
    out: list[Moment] = []
    seen: set[tuple[float, float, str]] = set()
    for m in candidates:
        if not isinstance(m, dict):
            continue
        try:
            start = float(m.get("start", -1))
            end = float(m.get("end", -1))
        except (TypeError, ValueError):
            continue
        if not (0 <= start < end <= duration):
            continue

        clip_type = str(m.get("clip_type", "reel") or "reel").strip()
        key = (round(start, 1), round(end, 1), clip_type.lower())
        if key in seen:
            continue
        seen.add(key)

        hashtags_raw = m.get("hashtags")
        hashtags = (
            [str(x) for x in hashtags_raw if str(x).strip()]
            if isinstance(hashtags_raw, list)
            else None
        )

        try:
            score = float(m.get("score", 0) or 0)
        except (TypeError, ValueError):
            score = 0.0

        out.append(
            Moment(
                start=start,
                end=end,
                title=str(m.get("title", "")).strip(),
                quote=str(m.get("quote", "")).strip(),
                why=str(m.get("why", "")).strip(),
                score=score,
                clip_type=clip_type,
                hook=str(m.get("hook", "")).strip(),
                caption=str(m.get("caption", "")).strip(),
                hashtags=hashtags,
            ),
        )

    out.sort(key=lambda x: x.score, reverse=True)

    requested_total = sum(
        int(x)
        for x in (stories_count, reels_count, long_reels_count, highlights_moments)
        if int(x) > 0
    )
    limit = requested_total if requested_total > 0 else int(count)
    if limit <= 0:
        return []

    # If per-type counts are specified, do a lightweight per-type selection.
    if requested_total > 0:
        def _cat(ct: str) -> str:
            s = (ct or "").lower()
            if "story" in s:
                return "story"
            if "highlight" in s or "hot" in s:
                return "highlight"
            if "long" in s:
                return "long_reel"
            return "reel"

        quotas = {
            "story": int(stories_count),
            "reel": int(reels_count),
            "long_reel": int(long_reels_count),
            "highlight": int(highlights_moments),
        }
        selected: list[Moment] = []
        for bucket in ("story", "reel", "long_reel", "highlight"):
            q = max(0, int(quotas.get(bucket, 0)))
            if q <= 0:
                continue
            for moment_obj in out:
                if len([x for x in selected if _cat(x.clip_type) == bucket]) >= q:
                    break
                if _cat(moment_obj.clip_type) == bucket and moment_obj not in selected:
                    selected.append(moment_obj)
        # Fill any remaining slots with best overall.
        if len(selected) < requested_total:
            for moment_obj in out:
                if len(selected) >= requested_total:
                    break
                if moment_obj not in selected:
                    selected.append(moment_obj)
        return selected[:requested_total]

    return out[:limit]


def _get_prompts_dir() -> Path:
    """RU: Возвращает путь к директории с промптами.

    EN: Get the path to the prompts directory.
    """
    # RU: Сначала пробуем директорию пакета, затем корень репозитория.
    # EN: Try package directory first, then repo root.
    repo_prompts = Path(__file__).resolve().parent.parent.parent / "prompts"
    if repo_prompts.exists():
        return repo_prompts
    return Path.cwd() / "prompts"


def _load_prompt(*, lang: str, variant: str, name: str) -> str:
    base = _get_prompts_dir() / lang
    p = base / f"{name}_{variant}.txt"
    if not p.exists():
        # Fall back to default variant.
        p = base / f"{name}_default.txt"
    return p.read_text(encoding="utf-8")


def _normalize_prompt_lang(prompt_lang: str, transcript_lang: str | None) -> str:
    pl = (prompt_lang or "auto").strip().lower()
    if pl != "auto":
        return pl
    tl = (transcript_lang or "").strip().lower()
    if tl.startswith("ru"):
        return "ru"
    if tl.startswith("en"):
        return "en"
    return "ru"


def _load_diarization(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []

    diar_path = Path(path)
    if not diar_path.exists():
        return []

    try:
        data = json.loads(diar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return []

    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []


def _assign_speakers(
    segments: list[dict[str, Any]], diar: list[dict[str, Any]], *, prefix: bool = True,
) -> None:
    if not diar:
        return

    def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    for segment in segments:
        try:
            s0 = float(segment.get("start", 0))
            s1 = float(segment.get("end", 0))
        except (TypeError, ValueError):
            continue

        best_spk = None
        best_ov = 0.0
        for diar_entry in diar:
            try:
                d0 = float(diar_entry.get("start", 0))
                d1 = float(diar_entry.get("end", 0))
                spk = str(diar_entry.get("speaker", ""))
            except (TypeError, ValueError):
                continue
            ov = overlap(s0, s1, d0, d1)
            if ov > best_ov and spk:
                best_ov = ov
                best_spk = spk

        if best_spk:
            segment["speaker"] = best_spk
            if (
                prefix
                and isinstance(segment.get("text"), str)
                and not segment["text"].lstrip().startswith("(")
            ):
                segment["text"] = f"({best_spk}) {segment['text']}"


def create_provider(
    provider_name: str,
    *,
    model: str,
    url: str | None = None,
    api_key: str | None = None,
    ollama_fallback_models: list[str] | None = None,
    ollama_watchdog: bool = True,
    ollama_first_token_timeout_s: int = 120,
    ollama_stall_timeout_s: int = 120,
    ollama_log_interval_s: int = 10,
    ollama_max_retries: int = 2,
) -> LLMProvider:
    """RU: Создаёт инстанс LLM-провайдера по имени провайдера.

    EN: Create an LLM provider instance based on the provider name.
    """
    if provider_name == "ollama":
        return OllamaProvider(
            OllamaConfig(
                url=url or "http://127.0.0.1:11434/api/generate",
                model=model,
                watchdog_enabled=bool(ollama_watchdog),
                first_token_timeout_s=int(ollama_first_token_timeout_s),
                stall_timeout_s=int(ollama_stall_timeout_s),
                log_interval_s=int(ollama_log_interval_s),
                max_retries=int(ollama_max_retries),
                fallback_models=tuple(ollama_fallback_models or []),
            ),
        )
    if provider_name == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Missing OpenAI key. Set OPENAI_API_KEY or pass --api-key")
        return OpenAIProvider(OpenAIConfig(api_key=key, model=model))
    if provider_name == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(
                "Missing Anthropic key. Set ANTHROPIC_API_KEY or pass --api-key",
            )
        return AnthropicProvider(AnthropicConfig(api_key=key, model=model))
    if provider_name == "gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("Missing Gemini key. Set GEMINI_API_KEY or pass --api-key")
        return GeminiProvider(GeminiConfig(api_key=key, model=model))
    raise SystemExit(f"Unsupported provider: {provider_name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """RU: Парсит аргументы командной строки.

    EN: Parse command line arguments.
    """
    ap = argparse.ArgumentParser(
        description="Analyze transcript with LLM to find viral moments.",
    )
    ap.add_argument(
        "--transcript",
        type=Path,
        required=True,
        help="Path to transcript JSON file",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("out"),
        help="Output directory",
    )
    ap.add_argument(
        "--provider",
        choices=("ollama", "openai", "anthropic", "gemini"),
        default="ollama",
        help="LLM provider to use",
    )
    ap.add_argument(
        "--api-key", help="Optional override for cloud providers; prefer env vars",
    )
    ap.add_argument(
        "--url",
        default="http://127.0.0.1:11434/api/generate",
        help="Ollama API URL",
    )
    ap.add_argument("--model", default="gemma2:9b", help="LLM model name")
    ap.add_argument("--temperature", type=float, default=0.3, help="LLM temperature")
    ap.add_argument("--reels", type=int, default=4, help="Number of reels to generate")
    ap.add_argument("--stories-count", type=int, default=0, help="Number of stories (up to 15s)")
    ap.add_argument("--reels-count", type=int, default=0, help="Number of reels (up to 60s)")
    ap.add_argument("--long-reels-count", type=int, default=0, help="Number of long reels (up to 180s)")
    ap.add_argument("--highlights-moments", type=int, default=0, help="Number of hot moments for highlights")
    ap.add_argument(
        "--reel-min", type=int, default=30, help="Minimum reel duration (seconds)",
    )
    ap.add_argument(
        "--reel-max", type=int, default=60, help="Maximum reel duration (seconds)",
    )
    ap.add_argument(
        "--chunk-seconds", type=int, default=600, help="Chunk size for analysis",
    )
    ap.add_argument(
        "--max_chars_chunk", type=int, default=12000, help="Max chars per chunk",
    )
    ap.add_argument("--timeout", type=int, default=900, help="LLM request timeout")
    ap.add_argument("--prompt-lang", default="auto", help="Prompt language: ru|en|auto")
    ap.add_argument(
        "--prompt-variant", default="default", help="Prompt variant: default|a|b",
    )
    ap.add_argument(
        "--diarization",
        type=Path,
        help="Optional diarization.json for speaker tags",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress non-error output")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")

    # Ollama watchdog / fallback
    ap.add_argument(
        "--ollama-watchdog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable Ollama stall watchdog (default: enabled)",
    )
    ap.add_argument(
        "--ollama-first-token-timeout",
        type=int,
        default=120,
        help="Abort if Ollama produces no output for N seconds (before first token)",
    )
    ap.add_argument(
        "--ollama-stall-timeout",
        type=int,
        default=120,
        help="Abort if Ollama produces no output for N seconds (read timeout)",
    )
    ap.add_argument(
        "--ollama-log-interval",
        type=int,
        default=10,
        help="Log progress every N seconds while waiting for Ollama",
    )
    ap.add_argument(
        "--ollama-max-retries",
        type=int,
        default=2,
        help="How many times to retry Ollama on stall/timeout",
    )
    ap.add_argument(
        "--ollama-fallback-model",
        action="append",
        default=[],
        help="Fallback model to try on stall/timeout (can be specified multiple times)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """RU: Точка входа для стадии анализа.

    EN: Main entry point for analyze script.
    """
    args = parse_args(argv)

    # RU: Применяем флаги логирования (иначе module-level setup_logging() уже всё зафиксировал).
    # EN: Apply verbosity flags (module-level setup_logging() would otherwise lock defaults).
    global LOGGER
    LOGGER = setup_logging(verbose=bool(args.verbose), quiet=bool(args.quiet))

    if not args.transcript.exists():
        message = f"Transcript not found: {args.transcript}"
        raise SystemExit(message)

    try:
        data = json.loads(args.transcript.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        message = f"Failed to read transcript: {exc}"
        raise SystemExit(message) from exc

    segments = data.get("segments", [])
    duration = data.get("duration", 0.0)
    if not duration and isinstance(segments, list) and segments:
        try:
            duration = float(segments[-1]["end"])
        except (KeyError, TypeError, ValueError):
            duration = 0.0

    diar = _load_diarization(args.diarization)
    if isinstance(segments, list) and diar:
        _assign_speakers(segments, diar)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    provider = create_provider(
        args.provider,
        model=args.model,
        url=args.url if args.provider == "ollama" else None,
        api_key=args.api_key,
        ollama_fallback_models=list(args.ollama_fallback_model or []),
        ollama_watchdog=bool(args.ollama_watchdog),
        ollama_first_token_timeout_s=int(args.ollama_first_token_timeout),
        ollama_stall_timeout_s=int(args.ollama_stall_timeout),
        ollama_log_interval_s=int(args.ollama_log_interval),
        ollama_max_retries=int(args.ollama_max_retries),
    )

    proc: subprocess.Popen | None = None
    try:
        managed_by_pipeline = os.environ.get(ENV_MANAGED_BY_PIPELINE) == "1"
        local = _parse_local_ollama_host_port(args.url) if args.url else None
        if args.provider == "ollama" and local and not managed_by_pipeline:
            host, port = local
            proc = ollama_start(host=host, port=port)

        prompt_lang = _normalize_prompt_lang(args.prompt_lang, data.get("language"))
        variant = str(args.prompt_variant).strip().lower()
        ch_prompt = _load_prompt(lang=prompt_lang, variant=variant, name="chunk")
        select_prompt = _load_prompt(lang=prompt_lang, variant=variant, name="select")

        moments = find_moments(
            provider,
            segments,
            float(duration),
            args.reel_min,
            args.reel_max,
            args.reels,
            args.chunk_seconds,
            args.max_chars_chunk,
            args.timeout,
            progress=bool(args.verbose and not args.quiet),
            ch_prompt=ch_prompt,
            select_prompt=select_prompt,
            stories_count=args.stories_count,
            reels_count=args.reels_count,
            long_reels_count=args.long_reels_count,
            highlights_moments=args.highlights_moments,
        )

        out_json = outdir / "moments.json"
        out_json.write_text(
            json.dumps([asdict(m) for m in moments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        reels_md = outdir / "reels.md"
        with reels_md.open("w", encoding="utf-8") as f:
            f.write("# Reels Suggestions\n\n")
            for i, m in enumerate(moments, 1):
                f.write(
                    f"## {i}. {m.title} [{m.clip_type}]\n- Time: {fmt_hms(m.start)}-{fmt_hms(m.end)}\n- Why: {m.why}\n",
                )
                if m.hook:
                    f.write(f"- Hook: {m.hook}\n")
                if m.caption:
                    f.write(f"- Caption: {m.caption}\n")
                if m.hashtags:
                    f.write(f"- Hashtags: {' '.join(m.hashtags)}\n")
                f.write("\n")

        if args.verbose:
            _status(f"[analyze] moments={len(moments)}", quiet=args.quiet)
        if not args.quiet:
            LOGGER.info("%s", out_json)
    finally:
        if proc:
            ollama_stop(proc)


if __name__ == "__main__":
    main()
