#!/usr/bin/env python3
"""RU: Анализирует transcript.json через LLM, чтобы найти вирусные моменты и метаданные.

EN: Analyze transcript.json with an LLM to find viral moments and generate metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
from podcast_reels_forge.utils.json_utils import extract_first_json_object


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
    hook: str = ""
    caption: str = ""
    hashtags: list[str] | None = None
    file: str = ""


def _status(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(msg, flush=True)


def fmt_hms(sec: float) -> str:
    """RU: Форматирует секунды как HH:MM:SS или MM:SS.

    EN: Format seconds for display as HH:MM:SS or MM:SS.
    """
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def wait_tcp(host: str, port: int, timeout_s: int = 20) -> bool:
    """RU: Ждёт, пока TCP-порт начнёт принимать соединения.

    EN: Wait for a TCP port to open.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def ollama_start() -> subprocess.Popen | None:
    """RU: Стартует Ollama в фоне, если бинарник доступен.

    EN: Start Ollama server in background if available.
    """
    try:
        p = subprocess.Popen(
            ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if wait_tcp("127.0.0.1", 11434, timeout_s=30):
            return p
        p.terminate()
    except FileNotFoundError:
        pass
    return None


def ollama_stop(p: subprocess.Popen) -> None:
    """RU: Останавливает процесс Ollama.

    EN: Terminate the Ollama process.
    """
    try:
        p.terminate()
        p.wait(timeout=10)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def get_llm_json(
    provider: LLMProvider, prompt: str, temperature: float, timeout: int,
) -> dict[str, Any]:
    """RU: Получает JSON-ответ от LLM с простой попыткой «починки».

    EN: Get JSON response from LLM with simple repair logic.
    """
    raw = provider.generate(prompt, temperature=temperature, timeout=timeout)
    try:
        return extract_first_json_object(raw)
    except Exception:
        repair = f"Fix the following JSON and return only the valid JSON:\n\n{raw}"
        raw2 = provider.generate(repair, temperature=0.2, timeout=timeout)
        return extract_first_json_object(raw2)


# ----------------------------
# Analysis Logic
# ----------------------------


def segments_to_compact_text(segments: list[dict[str, Any]], max_chars: int) -> str:
    """RU: Преобразует сегменты в плотный текстовый формат для LLM.

    EN: Convert segments to a dense text format for LLM input.
    """
    return "\n".join(
        [f"[{int(s['start'])}-{int(s['end'])}] {s['text']}" for s in segments],
    )[:max_chars]


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
    *,
    ch_prompt: str,
    select_prompt: str,
) -> list[Moment]:
    """RU: Оркестрирует анализ по чанкам и финальный выбор вирусных моментов.

    EN: Orchestrate chunk-based analysis and final selection of viral moments.
    """
    candidates: list[dict[str, Any]] = []
    for ch in chunk_segments_by_time(segments, chunk_sec):
        ch_txt = segments_to_compact_text(ch, max_ch)
        prompt = ch_prompt.format(r_min=r_min, r_max=r_max, transcript=ch_txt)
        try:
            m = get_llm_json(provider, prompt, 0.3, timeout).get("moment", {})
            start = float(m.get("start", -1))
            end = float(m.get("end", -1))
            if 0 <= start < end <= duration and r_min <= (end - start) <= r_max:
                candidates.append(m)
        except Exception:
            continue
    if not candidates:
        return []
    prompt2 = select_prompt.format(
        count=count, candidates_json=json.dumps(candidates, ensure_ascii=False),
    )
    obj2 = get_llm_json(provider, prompt2, 0.4, timeout)
    out: list[Moment] = []
    for m in obj2.get("moments", []):
        try:
            hashtags = m.get("hashtags")
            if isinstance(hashtags, list):
                hashtags = [str(x) for x in hashtags if str(x).strip()]
            else:
                hashtags = None
            out.append(
                Moment(
                    start=float(m["start"]),
                    end=float(m["end"]),
                    title=str(m.get("title", "")).strip(),
                    quote=str(m.get("quote", "")).strip(),
                    why=str(m.get("why", "")).strip(),
                    score=float(m.get("score", 0)),
                    hook=str(m.get("hook", "")).strip(),
                    caption=str(m.get("caption", "")).strip(),
                    hashtags=hashtags,
                ),
            )
        except Exception:
            continue
    out = [
        m
        for m in out
        if 0 <= m.start < m.end <= duration and r_min <= (m.end - m.start) <= r_max
    ]
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:count]


def _get_prompts_dir() -> Path:
    """RU: Возвращает путь к директории с промптами.

    EN: Get the path to the prompts directory.
    """
    # RU: Сначала пробуем директорию пакета, затем корень репозитория.
    # EN: Try package directory first, then repo root.
    pkg_prompts = Path(__file__).resolve().parent.parent.parent / "prompts"
    if pkg_prompts.exists():
        return pkg_prompts
    # RU: Fallback относительно скрипта.
    # EN: Fallback to relative to script.
    return Path(__file__).resolve().parent.parent.parent / "prompts"


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


def _load_diarization(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except Exception:
        return []
    return []


def _assign_speakers(
    segments: list[dict[str, Any]], diar: list[dict[str, Any]], *, prefix: bool = True,
) -> None:
    if not diar:
        return

    def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    for s in segments:
        try:
            s0 = float(s.get("start", 0))
            s1 = float(s.get("end", 0))
        except Exception:
            continue
        best_spk = None
        best_ov = 0.0
        for d in diar:
            try:
                d0 = float(d.get("start", 0))
                d1 = float(d.get("end", 0))
                spk = str(d.get("speaker", ""))
            except Exception:
                continue
            ov = overlap(s0, s1, d0, d1)
            if ov > best_ov and spk:
                best_ov = ov
                best_spk = spk
        if best_spk:
            s["speaker"] = best_spk
            if (
                prefix
                and isinstance(s.get("text"), str)
                and not s["text"].lstrip().startswith("(")
            ):
                s["text"] = f"({best_spk}) {s['text']}"


def create_provider(
    provider_name: str,
    *,
    model: str,
    url: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """RU: Создаёт инстанс LLM-провайдера по имени провайдера.

    EN: Create an LLM provider instance based on the provider name.
    """
    if provider_name == "ollama":
        return OllamaProvider(
            OllamaConfig(url=url or "http://127.0.0.1:11434/api/generate", model=model),
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
    ap.add_argument("--transcript", required=True, help="Path to transcript JSON file")
    ap.add_argument("--outdir", default="out", help="Output directory")
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
        "--diarization", help="Optional diarization.json for speaker tags",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress non-error output")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """RU: Точка входа для стадии анализа.

    EN: Main entry point for analyze script.
    """
    args = parse_args(argv)

    with open(args.transcript, encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    duration = data.get("duration", segments[-1]["end"] if segments else 0)

    diar = _load_diarization(args.diarization)
    if isinstance(segments, list) and diar:
        _assign_speakers(segments, diar)

    os.makedirs(args.outdir, exist_ok=True)

    # RU: Инициализируем провайдер.
    # EN: Initialize provider.
    provider = create_provider(
        args.provider,
        model=args.model,
        url=args.url if args.provider == "ollama" else None,
        api_key=args.api_key,
    )

    proc = None
    try:
        if args.provider == "ollama" and "127.0.0.1" in args.url:
            proc = ollama_start()

        prompt_lang = _normalize_prompt_lang(args.prompt_lang, data.get("language"))
        variant = str(args.prompt_variant).strip().lower()
        ch_prompt = _load_prompt(lang=prompt_lang, variant=variant, name="chunk")
        select_prompt = _load_prompt(lang=prompt_lang, variant=variant, name="select")

        moments = find_moments(
            provider,
            segments,
            duration,
            args.reel_min,
            args.reel_max,
            args.reels,
            args.chunk_seconds,
            args.max_chars_chunk,
            args.timeout,
            ch_prompt=ch_prompt,
            select_prompt=select_prompt,
        )

        # RU: Сохраняем результаты.
        # EN: Save results.
        out_json = os.path.join(args.outdir, "moments.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in moments], f, ensure_ascii=False, indent=2)

        # RU: Генерируем markdown-описания.
        # EN: Generate markdown descriptions.
        with open(os.path.join(args.outdir, "reels.md"), "w", encoding="utf-8") as f:
            f.write("# Reels Suggestions\n\n")
            for i, m in enumerate(moments, 1):
                f.write(
                    f"## {i}. {m.title}\n- Time: {fmt_hms(m.start)}–{fmt_hms(m.end)}\n- Why: {m.why}\n",
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
            print(out_json)
    finally:
        if proc:
            ollama_stop(proc)


if __name__ == "__main__":
    main()
