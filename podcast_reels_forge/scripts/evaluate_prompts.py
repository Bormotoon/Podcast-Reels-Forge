#!/usr/bin/env python3
"""RU: Оценка вариантов промптов (A/B/default) на одном транскрипте.

Скрипт генерирует простой отчёт метрик, чтобы помочь выбрать лучший вариант.

Пример использования:
    python -m podcast_reels_forge.scripts.evaluate_prompts \
        --transcript output/my.json --outdir output --variants default,a,b

Примечания:
- Для каждого варианта запускается analyze в отдельную подпапку, чтобы не
    перезаписывать moments.json.
- Метрики эвристические (без ground truth). Они помогают обнаруживать неверные
    длительности, низкие оценки и нестабильность между вариантами.

EN: Evaluate prompt variants (A/B/default) on the same transcript.

Produces a simple metrics report to help pick the best prompt variant.

Usage example:
    python -m podcast_reels_forge.scripts.evaluate_prompts \
        --transcript output/my.json --outdir output --variants default,a,b

Notes:
- This script runs analyze for each variant into a separate subfolder to
    avoid overwriting moments.json.
- Metrics are heuristic (no ground truth). They help detect invalid durations,
    low scores, and instability across variants.

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from podcast_reels_forge.scripts import analyze as analyze_script
from podcast_reels_forge.utils.logging_utils import setup_logging

LOGGER = setup_logging()


@dataclass(frozen=True)
class VariantMetrics:
    """RU: Метрики для одного варианта промпта.

    EN: Metrics for a single prompt variant.
    """

    variant: str
    moments: int
    avg_score: float
    avg_duration: float
    violations: int


def _load_moments(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _dur(m: dict[str, Any]) -> float:
    try:
        return float(m.get("end", 0)) - float(m.get("start", 0))
    except (TypeError, ValueError):
        return 0.0


def _score(m: dict[str, Any]) -> float:
    try:
        return float(m.get("score", 0))
    except (TypeError, ValueError):
        return 0.0


def _interval_key(m: dict[str, Any], *, bucket_s: int = 3) -> tuple[int, int]:
    """RU: Бакетирует (округляет) start/end для сравнения стабильности.

    EN: Bucket start/end for stability comparisons.
    """
    try:
        s = float(m.get("start", 0))
        e = float(m.get("end", 0))
    except (TypeError, ValueError):
        return (0, 0)
    return (int(s // bucket_s), int(e // bucket_s))


def _jaccard(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _run_analyze(
    *, args: argparse.Namespace, base_outdir: Path, variant: str, diarization: Path | None,
) -> Path:
    outdir = base_outdir / variant
    outdir.mkdir(parents=True, exist_ok=True)
    argv = [
        "--transcript",
        str(args.transcript),
        "--outdir",
        str(outdir),
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--reels",
        str(args.reels),
        "--reel-min",
        str(args.reel_min),
        "--reel-max",
        str(args.reel_max),
        "--chunk-seconds",
        str(args.chunk_seconds),
        "--max_chars_chunk",
        str(args.max_chars_chunk),
        "--timeout",
        str(args.timeout),
        "--prompt-lang",
        args.prompt_lang,
        "--prompt-variant",
        variant,
        "--quiet",
    ]
    if args.url:
        argv += ["--url", args.url]
    if diarization:
        argv += ["--diarization", str(diarization)]

    analyze_script.main(argv)

    return outdir / "moments.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """RU: Парсит аргументы командной строки.

    EN: Parse command line arguments.
    """
    ap = argparse.ArgumentParser(description="Evaluate prompt variants for analysis")
    ap.add_argument("--transcript", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("output"))
    ap.add_argument(
        "--variants", default="default,a,b", help="Comma-separated: default,a,b",
    )

    ap.add_argument(
        "--provider",
        default="ollama",
        choices=("ollama", "openai", "anthropic", "gemini"),
    )
    ap.add_argument("--model", default="gemma2:9b")
    ap.add_argument("--url", default=None)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--reels", type=int, default=4)
    ap.add_argument("--reel-min", type=int, default=30)
    ap.add_argument("--reel-max", type=int, default=60)
    ap.add_argument("--chunk-seconds", type=int, default=600)
    ap.add_argument("--max-chars-chunk", type=int, default=12000)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--prompt-lang", default="auto")
    ap.add_argument("--diarization", type=Path)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """RU: Точка входа для оценки вариантов промптов.

    EN: Main entry point for prompt evaluation.
    """
    args = parse_args(argv)

    if not args.transcript.exists():
        message = f"Transcript not found: {args.transcript}"
        raise SystemExit(message)

    variants = [v.strip().lower() for v in str(args.variants).split(",") if v.strip()]
    eval_dir = args.outdir / "prompt_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    per_variant: list[VariantMetrics] = []
    buckets: dict[str, set[tuple[int, int]]] = {}

    for v in variants:
        moments_path = _run_analyze(
            args=args,
            base_outdir=eval_dir,
            variant=v,
            diarization=args.diarization,
        )
        moments = _load_moments(moments_path)

        durs = [_dur(m) for m in moments]
        scores = [_score(m) for m in moments]
        violations = sum(1 for d in durs if d < args.reel_min or d > args.reel_max)
        avg_duration = sum(durs) / len(durs) if durs else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        per_variant.append(
            VariantMetrics(
                variant=v,
                moments=len(moments),
                avg_score=round(avg_score, 3),
                avg_duration=round(avg_duration, 3),
                violations=violations,
            ),
        )
        buckets[v] = {_interval_key(m) for m in moments}

    stability: dict[str, dict[str, float]] = {}
    for a in variants:
        stability[a] = {}
        for b in variants:
            stability[a][b] = round(
                _jaccard(buckets.get(a, set()), buckets.get(b, set())), 3,
            )

    def rank_score(m: VariantMetrics) -> float:
        # RU: Эвристика: повышаем вес avg_score, штрафуем violations и недобор моментов.
        # EN: Heuristic: prioritize avg_score, penalize violations and missing moments.
        return (
            float(m.avg_score)
            - 0.75 * float(m.violations)
            - 0.25 * max(0, args.reels - m.moments)
        )

    best = None
    best_s = -math.inf
    for m in per_variant:
        s = rank_score(m)
        if s > best_s:
            best_s = s
            best = m.variant

    report = {
        "transcript": str(args.transcript),
        "provider": args.provider,
        "model": args.model,
        "variants": [asdict(m) for m in per_variant],
        "stability_jaccard": stability,
        "best_variant": best,
    }

    out_report = args.outdir / "prompt_eval.json"
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    LOGGER.info("%s", out_report)


if __name__ == "__main__":
    main()
