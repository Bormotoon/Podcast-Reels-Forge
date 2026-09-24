"""RU: Отпечатки входов стадий: пересчитывать то, что изменилось, и только это.

Кэш «по наличию файла» не замечал смены промптов, квот или модели: анализ не
пересчитывался, а нарезка пропускалась при любом существующем ролике, хотя
подписи к нему уже собирались от новых моментов. Отпечаток — хэш всего, от
чего зависит результат стадии; он хранится в `.forge_state.json` папки
эпизода.

EN: Stage input fingerprints: redo what changed, and only that.

A "file exists" cache missed changed prompts, quotas or models: the analysis
was never redone, and the cut was skipped whenever any reel existed, while the
captions next to it were already rebuilt from the new moments. A fingerprint
is a hash of everything a stage's result depends on; it lives in the
episode folder's `.forge_state.json`.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

STATE_FILE = ".forge_state.json"

_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


def file_digest(path: Path | None) -> str:
    """Content hash of a file ("" when missing), memoised by size and mtime."""

    if path is None:
        return ""
    try:
        stat = path.stat()
    except OSError:
        return ""
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    cached = _DIGEST_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    value = digest.hexdigest()
    _DIGEST_CACHE[key] = value
    return value


def file_identity(path: Path | None) -> str:
    """Cheap identity for large media: name, size and mtime, no hashing."""

    if path is None:
        return ""
    try:
        stat = path.stat()
    except OSError:
        return ""
    return f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"


def files_digest(paths: Iterable[Path]) -> str:
    return fingerprint({str(p.name): file_digest(p) for p in sorted(paths)})


def fingerprint(*parts: Any) -> str:
    """Stable hash of JSON-able parts (dict key order does not matter)."""

    material = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


class StageState:
    """Per-episode record of the fingerprint each stage last completed with."""

    def __init__(self, folder: Path) -> None:
        self.path = folder / STATE_FILE

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, stage: str) -> str | None:
        value = self._load().get(stage)
        return value if isinstance(value, str) else None

    def set(self, stage: str, value: str) -> None:
        data = self._load()
        data[stage] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def decide(self, stage: str, current: str) -> str:
        """Compare with the stored fingerprint.

        Returns ``"same"``, ``"changed"`` or ``"adopted"``: outputs made
        before fingerprints existed are trusted once and stamped with the
        current value, so upgrading does not redo a whole channel.
        """

        stored = self.get(stage)
        if stored is None:
            self.set(stage, current)
            return "adopted"
        return "same" if stored == current else "changed"


def subset(conf: Mapping[str, Any] | None, keys: Iterable[str]) -> dict[str, Any]:
    if not isinstance(conf, Mapping):
        return {}
    return {key: conf.get(key) for key in keys if key in conf}
