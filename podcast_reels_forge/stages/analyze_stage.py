"""RU: Общие вспомогательные сущности для стадии анализа.

Основная логика по-прежнему находится в `analyze.py` (для standalone запуска и
перезапуска в venv). Этот модуль нужен, чтобы сделать код более модульным,
тестируемым и соответствующим плану рефакторинга.

EN: Shared helpers for the analysis stage.

The heavy lifting still lives in the root script `analyze.py` (kept for
standalone usage and venv re-exec). This module exists to make the codebase more
modular and testable and to satisfy the refactor roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnalyzeConfig:
    """RU: Базовый конфиг параметров анализа.

    EN: Basic analysis configuration.
    """

    provider: str
    model: str
    temperature: float
    reels: int
    reel_min: int
    reel_max: int


def basic_quality_metrics(
    moments: list[dict[str, Any]], *, reel_min: int, reel_max: int,
) -> dict[str, Any]:
    """RU: Считает простые эвристические метрики качества выбранных моментов.

    EN: Compute small heuristic metrics for moment selection quality.
    """
    durations = []
    scores = []
    violations = 0
    for m in moments:
        try:
            d = float(m.get("end", 0)) - float(m.get("start", 0))
            s = float(m.get("score", 0))
        except (TypeError, ValueError):
            continue
        durations.append(d)
        scores.append(s)
        if d < reel_min or d > reel_max:
            violations += 1

    return {
        "moments": len(moments),
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "avg_duration": sum(durations) / len(durations) if durations else 0.0,
        "violations": violations,
    }
