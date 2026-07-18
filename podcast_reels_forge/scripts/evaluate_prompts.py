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
from typing import Any, Sequence

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
    # Only populated when a golden set is available for this transcript.
    recall_must: float | None = None
    recall_all: float | None = None
    precision: float | None = None


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


def load_golden(path: Path) -> list[dict[str, Any]]:
    """RU: Загружает эталонную разметку эпизода.

    EN: Load the hand-labelled reference moments for an episode.

    Format (golden/<transcript-stem>.json)::

        {"episode": "<stem>",
         "moments": [{"start": 120.5, "end": 165.0, "label": "must",
                      "topics": ["..."], "note": "..."}]}

    ``label`` is must | good | ok — "must" marks moments a run really should
    not miss, and is reported separately from overall recall.
    """

    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("could not read the golden set at %s", path)
        return []

    raw = data.get("moments") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def default_golden_path(transcript: Path) -> Path:
    """Where a golden set for this transcript is looked for by default."""

    repo_root = Path(__file__).resolve().parent.parent.parent
    stem = transcript.stem
    # Analysis usually runs on <name>.proofread.json; the golden set belongs
    # to the episode, not to that particular transcript revision.
    if stem.endswith(".proofread"):
        stem = stem[: -len(".proofread")]
    return repo_root / "golden" / f"{stem}.json"


def intervals_match(
    predicted: dict[str, Any],
    golden: dict[str, Any],
    *,
    min_overlap_ratio: float = 0.5,
) -> bool:
    """Whether a predicted moment covers a golden one.

    Measured against the shorter of the two, so a long clip containing a
    short golden moment counts as finding it.
    """

    try:
        p_start, p_end = float(predicted["start"]), float(predicted["end"])
        g_start, g_end = float(golden["start"]), float(golden["end"])
    except (KeyError, TypeError, ValueError):
        return False

    overlap = min(p_end, g_end) - max(p_start, g_start)
    if overlap <= 0:
        return False
    shortest = min(p_end - p_start, g_end - g_start)
    if shortest <= 0:
        return False
    return overlap / shortest >= min_overlap_ratio


def score_against_golden(
    predicted: Sequence[dict[str, Any]],
    golden: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Recall (overall and for "must" moments) and precision."""

    if not golden:
        return {}

    matched_golden = [
        item
        for item in golden
        if any(intervals_match(p, item) for p in predicted)
    ]
    matched_predicted = [
        p
        for p in predicted
        if any(intervals_match(p, item) for item in golden)
    ]

    must = [item for item in golden if str(item.get("label", "")).lower() == "must"]
    matched_must = [item for item in matched_golden if item in must]

    return {
        "recall_all": round(len(matched_golden) / len(golden), 3),
        "recall_must": round(len(matched_must) / len(must), 3) if must else 1.0,
        "precision": round(len(matched_predicted) / len(predicted), 3) if predicted else 0.0,
    }


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
        "--golden",
        type=Path,
        help="Hand-labelled reference moments; defaults to golden/<episode>.json",
    )

    ap.add_argument(
        "--provider",
        default="llama_cpp",
        choices=("llama_cpp",),
    )
    ap.add_argument("--model", default="gemma4")
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

    golden_path = args.golden or default_golden_path(args.transcript)
    golden = load_golden(golden_path)
    if golden:
        LOGGER.info("scoring against %d golden moment(s) from %s", len(golden), golden_path)
    else:
        LOGGER.info(
            "no golden set at %s; reporting heuristic metrics only", golden_path,
        )

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

        against_golden = score_against_golden(moments, golden)
        per_variant.append(
            VariantMetrics(
                variant=v,
                moments=len(moments),
                avg_score=round(avg_score, 3),
                avg_duration=round(avg_duration, 3),
                violations=violations,
                recall_must=against_golden.get("recall_must"),
                recall_all=against_golden.get("recall_all"),
                precision=against_golden.get("precision"),
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
        # RU: С golden-разметкой считаем реальное качество; без неё — эвристика.
        # EN: With a golden set this is measured quality; without one it falls
        # back to the old heuristic (avg_score, penalizing violations and a
        # short moment count).
        if m.recall_must is not None and m.recall_all is not None:
            precision = m.precision or 0.0
            return (
                6.0 * float(m.recall_must)
                + 3.0 * float(m.recall_all)
                + 2.0 * precision
                - 0.75 * float(m.violations)
            )
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
        "golden": str(golden_path) if golden else None,
        "variants": [asdict(m) for m in per_variant],
        "stability_jaccard": stability,
        "best_variant": best,
    }

    out_report = args.outdir / "prompt_eval.json"
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    LOGGER.info("%s", out_report)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        if LOGGER:
            LOGGER.warning("Interrupted by user.")
        import sys
        sys.exit(130)
    except Exception as exc:
        import sys
        if LOGGER:
            LOGGER.error("Evaluation failed: %s", exc)
        else:
            print(f"Evaluation failed: {exc}", file=sys.stderr)
        import os
        if os.environ.get("DEBUG_FORGE") == "1":
            import traceback
            traceback.print_exc()
        sys.exit(1)
