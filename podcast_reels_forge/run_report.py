"""RU: Отчёт о прогоне пайплайна: что случилось с каждым эпизодом.

Автономный прогон никто не смотрит в реальном времени, поэтому итог должен
быть машиночитаемым: по отчёту (и коду возврата) планировщик и уведомление
понимают, прошло всё, прошло частично или не прошло вовсе.

EN: The pipeline run report: what happened to every episode.

Nobody watches an unattended run live, so the outcome has to be machine
readable: from the report (and the exit code) a scheduler or a notification
can tell whether everything passed, some of it did, or nothing did.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Stage finished and produced output.
DONE = "done"
#: Output already existed (cache hit).
CACHED = "cached"
#: Not selected, disabled, or nothing to do.
SKIPPED = "skipped"
#: Raised or produced nothing usable.
FAILED = "failed"

#: Exit codes of ``start_forge.py``.
EXIT_OK = 0
EXIT_FATAL = 1
EXIT_PARTIAL = 3
#: Another run holds the lock (BSD ``EX_TEMPFAIL``: try again later).
EXIT_BUSY = 75


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class StageResult:
    status: str
    seconds: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"status": self.status, "seconds": round(self.seconds, 1)}
        if self.detail:
            data["detail"] = self.detail
        return data


@dataclass
class EpisodeReport:
    stem: str
    stages: dict[str, StageResult] = field(default_factory=dict)
    clips: int | None = None

    @property
    def failed(self) -> bool:
        return any(result.status == FAILED for result in self.stages.values())

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": FAILED if self.failed else "ok",
            "stages": {name: result.to_dict() for name, result in self.stages.items()},
        }
        if self.clips is not None:
            data["clips"] = self.clips
        return data


class RunReport:
    """Collects per-episode stage outcomes and run-level events."""

    def __init__(self) -> None:
        self.started_at = _now_iso()
        self.finished_at: str | None = None
        self._started = time.monotonic()
        self.episodes: dict[str, EpisodeReport] = {}
        self.events: list[dict[str, str]] = []
        self.fatal_error: str | None = None

    def episode(self, stem: str) -> EpisodeReport:
        if stem not in self.episodes:
            self.episodes[stem] = EpisodeReport(stem=stem)
        return self.episodes[stem]

    def record(
        self,
        stem: str,
        stage: str,
        status: str,
        *,
        seconds: float = 0.0,
        detail: str = "",
    ) -> None:
        self.episode(stem).stages[stage] = StageResult(status, seconds, detail)

    def event(self, level: str, message: str) -> None:
        """A run-level note that belongs to no single episode."""

        self.events.append({"level": level, "message": message, "at": _now_iso()})

    def fatal(self, message: str) -> None:
        self.fatal_error = message
        self.event("error", message)

    @property
    def outcome(self) -> str:
        if self.fatal_error is not None:
            return "failed"
        if any(ep.failed for ep in self.episodes.values()):
            return "partial"
        if any(event["level"] == "error" for event in self.events):
            return "partial"
        return "ok"

    def exit_code(self) -> int:
        return {"ok": EXIT_OK, "partial": EXIT_PARTIAL}.get(self.outcome, EXIT_FATAL)

    def finish(self) -> None:
        self.finished_at = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        failed = [stem for stem, ep in self.episodes.items() if ep.failed]
        return {
            "outcome": self.outcome,
            "exit_code": self.exit_code(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": round(time.monotonic() - self._started, 1),
            "episodes_total": len(self.episodes),
            "episodes_failed": failed,
            "clips_total": sum(ep.clips or 0 for ep in self.episodes.values()),
            "fatal_error": self.fatal_error,
            "events": self.events,
            "episodes": {stem: ep.to_dict() for stem, ep in self.episodes.items()},
        }

    def summary_line(self) -> str:
        data = self.to_dict()
        return (
            f"outcome={data['outcome']} episodes={data['episodes_total']} "
            f"failed={len(data['episodes_failed'])} clips={data['clips_total']} "
            f"elapsed={data['elapsed_s']}s"
        )

    def write(self, runs_dir: Path) -> Path:
        """Write ``<runs_dir>/<timestamp>.json`` and refresh ``latest.json``."""

        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        path = runs_dir / f"{stamp}.json"
        for target in (path, runs_dir / "latest.json"):
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, target)
        return path
