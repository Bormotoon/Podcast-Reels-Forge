"""Safe serialization helpers for analysis artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=str(path.parent),
        encoding=encoding,
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    # NamedTemporaryFile creates the file with mode 0600, and replace() keeps
    # it — leaving artifacts unreadable to the owning group (e.g. anyone
    # downloading via a group-based Samba share). Re-apply the umask-derived
    # mode a normal open() would have produced so these files match their
    # siblings (typically 0664).
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp_path, 0o666 & ~umask)
    tmp_path.replace(path)
    return path


def atomic_write_json(path: Path, data: Any) -> Path:
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    return atomic_write_text(path, payload + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
