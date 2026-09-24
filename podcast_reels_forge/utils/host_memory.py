"""RU: Освобождение оперативной памяти хоста на время прогона.

Пайплайн держит на хосте несколько гигабайт (llama-server плюс питон с torch), и
на машине, где рядом живут виртуалки, этого хватает, чтобы упереться в OOM. Здесь
память у ВМ забирается на живую через virtio-balloon и возвращается в конце.

Почему баллон, а не пауза или сохранение:
- `virsh suspend` не освобождает ни байта: он останавливает vCPU, но qemu
  продолжает держать всю память гостя;
- `managedsave` невозможен для домена с проброшенным PCI-устройством, а именно
  так собраны ВМ с видеокартой;
- баллон работает на живой ВМ: гость отдаёт неиспользуемые страницы, сервисы
  внутри продолжают отвечать, на диск ничего не пишется, возврат мгновенный.

Но у баллона есть предел, проверенный измерением: **на ВМ с проброшенным PCI он
не освобождает ничего**. Вся память такого гостя залочена в RAM (`VmLck` равен её
размеру), потому что IOMMU нужен постоянный маппинг для DMA. Баллон при этом
честно сдвигается и гость сообщает о свободных страницах, а RSS у qemu не падает
ни на байт — то есть память отнимается у гостя и не достаётся никому. Такие
домены пропускаются (см. `locked_memory_kib`), а забрать у них память может
только полная остановка — для этого есть отдельный список `stop_domains`.

Возврат — главное свойство модуля. Исходный размер записывается на диск ДО
первого изменения, поэтому память можно вернуть даже после падения процесса,
перезагрузки или OOM: см. `restore_host_memory`.

EN: Freeing host RAM for the duration of a run.

The pipeline holds several gigabytes on the host (llama-server plus Python with
torch), and on a machine that also runs VMs that is enough to hit the OOM killer.
This module reclaims VM memory live through virtio-balloon and gives it back at
the end.

Why ballooning rather than pausing or saving:
- `virsh suspend` frees nothing: it stops the vCPUs while qemu keeps the whole
  guest memory mapped;
- `managedsave` is impossible for a domain with an assigned PCI device, which is
  exactly how a GPU-passthrough VM is built;
- ballooning works on a live VM: the guest hands back unused pages, services
  inside keep answering, nothing is written to disk, and the return is instant.

Ballooning has one measured limit, though: **on a VM with a passed-through PCI
device it frees nothing at all**. Such a guest has its whole memory locked into
RAM (`VmLck` equals its size) because the IOMMU needs a permanent DMA mapping.
The balloon does move and the guest does report free pages, yet qemu's RSS does
not drop by a byte — the memory is taken from the guest and handed to nobody.
Those domains are skipped (see `locked_memory_kib`); the only way to reclaim
their memory is to stop them, which is what `stop_domains` is for.

Restoring is this module's main property. The original size is written to disk
BEFORE the first change, so memory can be returned even after a crash, a reboot
or an OOM kill — see `restore_host_memory`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import time
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

log = logging.getLogger("Forge")

#: libvirt connection URI. VMs with GPU passthrough live in the system session.
DEFAULT_CONNECT: Final = "qemu:///system"

#: RU: Сколько оставить гостю сверх того, что он занимает. Ужать до самого
#:     занятого — верный способ получить OOM уже внутри ВМ: гостю нужен запас
#:     на страничный кэш и всплески.
#: EN: How much to leave the guest above what it occupies. Squeezing to exactly
#:     the used figure is a reliable way to OOM inside the VM instead: a guest
#:     needs headroom for page cache and spikes.
DEFAULT_HEADROOM_MB: Final = 1536

#: Never balloon a guest below this, whatever it reports.
DEFAULT_MIN_MB: Final = 1024

#: Anything smaller than this is not worth a balloon round-trip.
MIN_WORTHWHILE_MB: Final = 512

DEFAULT_STATE_FILE: Final = ".forge-host-memory.json"

VIRSH_TIMEOUT: Final = 30

#: RU: Как часто повторять ACPI-запрос на выключение, пока гость не отзовётся.
#: EN: How often to repeat the ACPI shutdown request until the guest answers.
SHUTDOWN_RETRY_S: Final = 30


@dataclass(frozen=True)
class DomainMemory:
    """RU: Память домена по данным гостя. EN: Domain memory as the guest sees it."""

    name: str
    #: Current balloon target in KiB — what qemu actually holds.
    actual_kib: int
    #: Free inside the guest, in KiB. 0 when the guest agent reports nothing.
    unused_kib: int

    @property
    def used_kib(self) -> int:
        """RU: Занято гостем. EN: In use by the guest.

        Clamped at zero: a guest whose balloon sits below its total RAM can
        report ``unused`` against the larger figure, which would otherwise come
        out negative and be read as "needs nothing".
        """

        return max(0, self.actual_kib - self.unused_kib)


def virsh_available() -> bool:
    """RU: Есть ли virsh в PATH. EN: Is virsh on PATH."""

    return shutil.which("virsh") is not None


def _virsh(
    args: list[str], *, connect: str = DEFAULT_CONNECT, timeout_s: int = VIRSH_TIMEOUT,
) -> str | None:
    """RU: Запускает virsh, возвращает stdout или None при ошибке.

    EN: Run virsh, returning stdout, or None on failure.

    Never raises: freeing memory is an optimisation, and a libvirt hiccup must
    not take down a run that would otherwise have succeeded.

    The default timeout suits the read-only queries. Anything that changes a
    domain's state needs its own, longer one — see `start_domain`.
    """

    cmd = ["virsh", "-c", connect, *args]
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("virsh %s failed: %s", " ".join(args), exc)
        return None
    if res.returncode != 0:
        detail = (res.stderr or res.stdout or "").strip()
        log.warning("virsh %s failed: %s", " ".join(args), detail[-300:])
        return None
    return res.stdout


def list_running_domains(*, connect: str = DEFAULT_CONNECT) -> list[str]:
    """RU: Имена запущенных доменов. EN: Names of the running domains."""

    out = _virsh(["list", "--name", "--state-running"], connect=connect)
    if out is None:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def domain_memory(name: str, *, connect: str = DEFAULT_CONNECT) -> DomainMemory | None:
    """RU: Читает actual/unused домена. EN: Read a domain's actual/unused figures."""

    out = _virsh(["dommemstat", name], connect=connect)
    if out is None:
        return None
    stats: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            stats[parts[0]] = int(parts[1])
    actual = stats.get("actual")
    if not actual:
        log.warning("Домен %s не сообщает actual — баллон недоступен", name)
        return None
    return DomainMemory(name=name, actual_kib=actual, unused_kib=stats.get("unused", 0))


def locked_memory_kib(name: str) -> int:
    """RU: Сколько памяти домена заблокировано в RAM (mlock).

    EN: How much of a domain's memory is locked into RAM (mlock).

    A domain with a passed-through PCI device has its **entire** guest memory
    locked: the IOMMU needs a permanent mapping for DMA. The balloon still moves
    — the guest dutifully reports free pages — but the host reclaims nothing,
    because a locked page cannot be dropped. Ballooning such a domain is worse
    than useless: it takes memory away from the guest and gives none to the host.

    Returns 0 when the process cannot be found or read, which reads as "not
    locked" — the conservative direction, since the balloon is then merely
    ineffective rather than harmful.
    """

    try:
        found = subprocess.run(
            ["pgrep", "-f", f"guest={name},"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    pids = [p for p in (found.stdout or "").split() if p.isdigit()]
    if not pids:
        return 0
    try:
        status = Path(f"/proc/{pids[0]}/status").read_text(encoding="utf-8")
    except OSError:
        return 0
    for line in status.splitlines():
        if line.startswith("VmLck:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return 0


#: RU: Доля заблокированной памяти, при которой баллон бессмысленен.
#: EN: The share of locked memory at which ballooning is pointless.
LOCKED_SHARE_THRESHOLD: Final = 0.5


def plan_target_kib(
    memory: DomainMemory, *, headroom_mb: int = DEFAULT_HEADROOM_MB,
    min_mb: int = DEFAULT_MIN_MB,
) -> int:
    """RU: До какого размера сжимать домен.

    EN: The size to squeeze a domain down to.

    Derived from what the guest itself reports as used, so it adapts to whatever
    the VM is doing rather than trusting a number typed months ago. Never returns
    more than the current size — this function only ever shrinks.
    """

    target = memory.used_kib + headroom_mb * 1024
    target = max(target, min_mb * 1024)
    return min(target, memory.actual_kib)


def _write_state(path: Path, payload: dict[str, Any]) -> bool:
    """RU: Пишет файл состояния и сбрасывает его на диск.

    EN: Write the state file and flush it to disk.

    Written before the first balloon change and fsynced, because this file is the
    only way back if the process dies between shrinking and restoring.
    """

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".hostmem-")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except OSError as exc:
        log.error("Не удалось записать %s: %s. Память ВМ трогать не буду.", path, exc)
        return False
    return True


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def set_balloon(name: str, kib: int, *, connect: str = DEFAULT_CONNECT) -> bool:
    """RU: Ставит целевой размер баллона. EN: Set the balloon target."""

    return _virsh(["setmem", name, f"{kib}KiB", "--live"], connect=connect) is not None


def domain_state(name: str, *, connect: str = DEFAULT_CONNECT) -> str:
    """RU: Состояние домена одним словом. EN: A domain's state in one word."""

    out = _virsh(["domstate", name], connect=connect)
    return (out or "").strip().lower()


def stop_domain(
    name: str, *, connect: str = DEFAULT_CONNECT, timeout_s: int = 180,
) -> bool:
    """RU: Корректно выключает домен и ждёт остановки.

    EN: Shut a domain down cleanly and wait for it to stop.

    A clean shutdown, never `destroy`: the guest gets to flush its filesystems.
    If it has not stopped within the timeout we give up and say so rather than
    pulling the plug — a corrupted guest is a far worse outcome than a run that
    had less RAM than hoped.
    """

    if domain_state(name, connect=connect) != "running":
        return True
    if _virsh(["shutdown", name], connect=connect) is None:
        return False

    # RU: Запрос повторяется, пока идёт ожидание. `virsh shutdown` — это нажатие
    #     ACPI-кнопки питания, и гость, который ещё не догрузился, её попросту не
    #     слышит: обработчик поднимается позже самого qemu. Один запрос в начале
    #     означает, что ВМ, поднятая минуту назад, не выключится никогда.
    # EN: The request is repeated throughout the wait. `virsh shutdown` is an ACPI
    #     power-button press, and a guest still booting simply does not hear it —
    #     its handler comes up later than qemu does. Asking only once means a VM
    #     started a minute ago never shuts down at all.
    deadline = time.monotonic() + timeout_s
    next_nudge = time.monotonic() + SHUTDOWN_RETRY_S
    while time.monotonic() < deadline:
        if domain_state(name, connect=connect) in {"shut off", "shutoff", ""}:
            return True
        if time.monotonic() >= next_nudge:
            log.debug("Повторяю запрос на выключение %s", name)
            _virsh(["shutdown", name], connect=connect)
            next_nudge = time.monotonic() + SHUTDOWN_RETRY_S
        time.sleep(2)
    log.error(
        "Домен %s не выключился за %d с. Принудительно гасить не буду — "
        "верните его вручную, если нужно.", name, timeout_s,
    )
    return False


def start_domain(
    name: str, *, connect: str = DEFAULT_CONNECT, timeout_s: int = 180,
) -> bool:
    """RU: Поднимает домен и убеждается, что он действительно запустился.

    EN: Start a domain and confirm it actually came up.

    Success is judged by polling the domain's state, not by what `virsh start`
    returned. On a domain with a passed-through PCI device the command can take
    well over a minute — device reset and IOMMU setup — and a client-side timeout
    would otherwise be read as failure while the VM is busy booting perfectly
    fine. That misreading is worse than it sounds: it leaves a record claiming
    the VM is still down.
    """

    if domain_state(name, connect=connect) == "running":
        return True

    _virsh(["start", name], connect=connect, timeout_s=timeout_s)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if domain_state(name, connect=connect) == "running":
            return True
        time.sleep(2)
    return False


@dataclass(frozen=True)
class HostMemoryConfig:
    """RU: Настройки освобождения памяти. EN: Memory-freeing settings."""

    enabled: bool = False
    #: Domains to squeeze. Empty means every running domain.
    domains: tuple[str, ...] = ()
    #: RU: Домены, которые на время прогона выключаются целиком. Единственный
    #:     способ забрать память у ВМ с проброшенным PCI-устройством: её память
    #:     заблокирована в RAM, и баллон там не работает. Возврат — обычный
    #:     запуск домена, то есть чистая загрузка гостя, а не продолжение с того
    #:     же места; для сервисной ВМ это обычно одно и то же, но сказать надо.
    #: EN: Domains shut down entirely for the duration of a run. The only way to
    #:     reclaim memory from a VM with a passed-through PCI device: its memory
    #:     is locked into RAM and ballooning does nothing there. Restoring means
    #:     starting the domain again — a clean guest boot, not a resume from the
    #:     same point; for a service VM that is usually the same thing, but it
    #:     needs saying.
    stop_domains: tuple[str, ...] = ()
    shutdown_timeout_s: int = 180
    headroom_mb: int = DEFAULT_HEADROOM_MB
    min_mb: int = DEFAULT_MIN_MB
    state_file: str = DEFAULT_STATE_FILE
    connect: str = DEFAULT_CONNECT

    @classmethod
    def from_conf(cls, conf: dict[str, Any] | None) -> HostMemoryConfig:
        raw = conf if isinstance(conf, dict) else {}
        domains = raw.get("domains")
        names = (
            tuple(str(d).strip() for d in domains if str(d).strip())
            if isinstance(domains, list)
            else ()
        )
        stop = raw.get("stop_domains")
        stop_names = (
            tuple(str(d).strip() for d in stop if str(d).strip())
            if isinstance(stop, list)
            else ()
        )
        return cls(
            enabled=bool(raw.get("enabled", False)),
            domains=names,
            stop_domains=stop_names,
            shutdown_timeout_s=max(30, int(raw.get("shutdown_timeout_s", 180))),
            headroom_mb=max(0, int(raw.get("headroom_mb", DEFAULT_HEADROOM_MB))),
            min_mb=max(64, int(raw.get("min_mb", DEFAULT_MIN_MB))),
            state_file=str(raw.get("state_file") or DEFAULT_STATE_FILE),
            connect=str(raw.get("connect") or DEFAULT_CONNECT),
        )


def free_host_memory(
    config: HostMemoryConfig, *, repo_dir: Path, quiet: bool = False,
) -> int:
    """RU: Забирает у ВМ неиспользуемую память. Возвращает освобождённые МБ.

    EN: Reclaim unused memory from the VMs. Returns the megabytes freed.

    The original sizes reach disk before the first change, so a crash between the
    two halves still leaves a way back.
    """

    if not config.enabled:
        return 0
    if not virsh_available():
        log.warning("virsh не найден — память ВМ освободить не могу")
        return 0

    state_path = repo_dir / config.state_file
    # A leftover file means an earlier run never restored; put that right first,
    # or the originals recorded now would be the already-shrunken sizes.
    restore_host_memory(config, repo_dir=repo_dir, quiet=quiet)

    running = list_running_domains(connect=config.connect)
    # A domain that is going to be shut down must not also be ballooned: it would
    # only slow the shutdown down and muddle what the state file has to undo.
    to_stop = [n for n in config.stop_domains if n in running]
    names = [
        n for n in (list(config.domains) or running) if n not in config.stop_domains
    ]

    plans: dict[str, dict[str, int]] = {}
    for name in names:
        memory = domain_memory(name, connect=config.connect)
        if memory is None:
            continue
        locked = locked_memory_kib(name)
        if locked >= memory.actual_kib * LOCKED_SHARE_THRESHOLD:
            log.warning(
                "Домен %s держит %d МБ заблокированными в RAM (проброшенное "
                "PCI-устройство): баллон отнимет память у гостя, но хосту ничего "
                "не вернёт — пропускаю. Освободить её может только остановка ВМ.",
                name, locked // 1024,
            )
            if not quiet:
                print(
                    f"[ram] {name}: пропущен — {locked // 1024} МБ заблокировано "
                    "под проброшенное устройство, баллон бесполезен",
                    flush=True,
                )
            continue
        target = plan_target_kib(
            memory, headroom_mb=config.headroom_mb, min_mb=config.min_mb,
        )
        freed_mb = (memory.actual_kib - target) // 1024
        if freed_mb < MIN_WORTHWHILE_MB:
            log.debug("Домену %s сжиматься незачем (%d МБ)", name, freed_mb)
            continue
        plans[name] = {"original_kib": memory.actual_kib, "target_kib": target}

    if not plans and not to_stop:
        return 0

    if not _write_state(state_path, {"domains": plans, "stopped": to_stop}):
        return 0

    freed_mb = 0
    for name in to_stop:
        memory = domain_memory(name, connect=config.connect)
        size_mb = (memory.actual_kib // 1024) if memory else 0
        if not quiet:
            print(f"[ram] {name}: выключаю на время прогона…", flush=True)
        if stop_domain(
            name, connect=config.connect, timeout_s=config.shutdown_timeout_s,
        ):
            freed_mb += size_mb
            if not quiet:
                print(f"[ram] {name}: выключен, освобождено ~{size_mb} МБ", flush=True)

    for name, plan in plans.items():
        if not set_balloon(name, plan["target_kib"], connect=config.connect):
            continue
        gained = (plan["original_kib"] - plan["target_kib"]) // 1024
        freed_mb += gained
        if not quiet:
            print(
                f"[ram] {name}: {plan['original_kib'] // 1024} МБ → "
                f"{plan['target_kib'] // 1024} МБ (освобождено {gained} МБ)",
                flush=True,
            )
    return freed_mb


def restore_host_memory(
    config: HostMemoryConfig, *, repo_dir: Path, quiet: bool = False,
) -> int:
    """RU: Возвращает ВМ исходный размер. Безопасна к повторному вызову.

    EN: Give the VMs their original size back. Safe to call any number of times.

    Runs from the state file rather than from memory, so it works from a fresh
    process — which is the whole point after a crash or an OOM kill.
    """

    state_path = repo_dir / config.state_file
    if not state_path.exists():
        return 0
    if not virsh_available():
        log.warning("virsh не найден — не могу вернуть память ВМ (%s)", state_path)
        return 0

    state = _read_state(state_path)
    domains = (state or {}).get("domains")
    stopped = (state or {}).get("stopped")
    if not isinstance(domains, dict):
        log.warning("Не разобрать %s; удаляю файл", state_path)
        state_path.unlink(missing_ok=True)
        return 0

    restored_mb = 0
    failed: dict[str, Any] = {}
    still_stopped: list[str] = []

    for name in stopped if isinstance(stopped, list) else []:
        if start_domain(
            str(name), connect=config.connect, timeout_s=config.shutdown_timeout_s,
        ):
            if not quiet:
                print(f"[ram] {name}: запущен обратно", flush=True)
        else:
            still_stopped.append(str(name))
            log.error("Не удалось поднять домен %s — запустите вручную", name)

    for name, plan in domains.items():
        try:
            original = int(plan["original_kib"])
        except (KeyError, TypeError, ValueError):
            continue
        current = domain_memory(str(name), connect=config.connect)
        # A host reboot brings the domain up at its configured size, so there is
        # nothing to undo — and growing it further would be wrong.
        if current is not None and current.actual_kib >= original:
            continue
        if not set_balloon(str(name), original, connect=config.connect):
            failed[name] = plan
            continue
        restored_mb += original // 1024
        if not quiet:
            print(f"[ram] {name}: возвращено {original // 1024} МБ", flush=True)

    if failed or still_stopped:
        # Keep what could not be restored, so the next run tries again instead of
        # leaving a VM squeezed — or switched off — forever.
        _write_state(state_path, {"domains": failed, "stopped": still_stopped})
        if failed:
            log.error("Не удалось вернуть память доменам: %s", ", ".join(failed))
    else:
        state_path.unlink(missing_ok=True)
    return restored_mb


def install_restore_on_signals(restore: Callable[[], None]) -> None:
    """RU: Вешает возврат памяти на сигналы завершения.

    EN: Hook the memory restore onto the termination signals.

    Две детали здесь отделяют «работает» от «тихо не работает».

    Первая: в обработчике нельзя звать `sys.exit()`. Он поднимает SystemExit, а
    стадия анализа ловит именно его, чтобы один сорвавшийся вызов не ронял весь
    прогон, — и сигнал молча съедался: память уходила обратно виртуалкам, а
    обработка продолжалась уже без неё. Возврат диспозиции в SIG_DFL и повторная
    отправка сигнала себе не обходятся ничем.

    Вторая: сигнал, который уже игнорируется, не перехватывается. Под `nohup`
    SIGHUP выставлен в SIG_IGN, и свой обработчик превратил бы закрытие
    терминала обратно в убийство фонового прогона.

    Not `sys.exit()` in the handler: it raises SystemExit, which the analyze
    stage catches so one failed call cannot bring down a whole run — so the
    signal was swallowed, memory went back to the VMs, and processing carried on
    without it. Resetting the disposition to SIG_DFL and re-raising the signal at
    ourselves cannot be worked around. And a signal that is already ignored is
    left alone: under `nohup` SIGHUP is SIG_IGN, and taking it over would turn
    closing the terminal back into killing the background run.
    """

    def handler(signum: int, _frame: Any) -> None:
        restore()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            if signal.getsignal(sig) is signal.SIG_IGN:
                continue
            signal.signal(sig, handler)
        except (ValueError, OSError, AttributeError):
            # Not the main thread, or the platform lacks the signal. Freeing
            # memory is an optimisation: failing to arm a handler must not stop
            # the run, and the state file still covers the crash case.
            log.debug("Не удалось повесить обработчик на сигнал %s", sig)
