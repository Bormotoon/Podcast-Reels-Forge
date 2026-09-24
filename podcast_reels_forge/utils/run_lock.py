"""RU: Блокировка «один прогон за раз».

Два одновременных прогона (таймер и ручной запуск) делят GPU и порт
llama-server и убивают сервер друг друга. Блокировка держится на файле через
``flock``: ядро снимает её само, если процесс умер, поэтому «залипших»
блокировок после падения не бывает.

EN: A "one run at a time" lock.

Two concurrent runs (a timer and a manual start) share the GPU and the
llama-server port and kill each other's server. The lock is an ``flock`` on a
file: the kernel drops it when the process dies, so a crash never leaves a
stale lock behind.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


class RunLockBusy(RuntimeError):
    """Another process holds the lock."""


class RunLock:
    """Exclusive, non-blocking process lock on ``path``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                holder = _read_pid(fd)
                os.close(fd)
                raise RunLockBusy(
                    f"another run holds {self.path}" + (f" (pid {holder})" if holder else ""),
                ) from exc
        else:  # pragma: no cover - best effort without flock
            holder = _read_pid(fd)
            if holder and _pid_alive(holder):
                os.close(fd)
                raise RunLockBusy(f"another run holds {self.path} (pid {holder})")
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            os.ftruncate(fd, 0)
        except OSError:
            pass
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def _read_pid(fd: int) -> int | None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 32).decode().strip()
        return int(raw) if raw else None
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:  # pragma: no cover - Windows fallback only
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
