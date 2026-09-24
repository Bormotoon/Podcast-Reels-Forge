"""Tests for reclaiming and returning VM memory around a run.

virsh is faked throughout. The property that matters most here is that memory
always finds its way back, so most of these are about the restore path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from podcast_reels_forge.utils import host_memory as hm

if TYPE_CHECKING:
    MonkeyPatch = pytest.MonkeyPatch


class FakeVirsh:
    """Stands in for the virsh binary, holding a balloon size per domain."""

    def __init__(self, domains: dict[str, tuple[int, int]]):
        #: name -> (actual_kib, unused_kib)
        self.domains = dict(domains)
        self.calls: list[list[str]] = []
        self.fail_setmem_for: set[str] = set()

    def __call__(self, args: list[str], *, connect: str = "") -> str | None:
        self.calls.append(list(args))
        if args[0] == "list":
            return "\n".join(self.domains) + "\n"
        if args[0] == "dommemstat":
            name = args[1]
            if name not in self.domains:
                return None
            actual, unused = self.domains[name]
            return f"actual {actual}\nunused {unused}\nswap_in 0\n"
        if args[0] == "setmem":
            name = args[1]
            if name in self.fail_setmem_for:
                return None
            kib = int(args[2].replace("KiB", ""))
            actual, unused = self.domains[name]
            # A balloon change moves the target; free memory moves with it.
            self.domains[name] = (kib, max(0, unused - (actual - kib)))
            return ""
        raise AssertionError(f"unexpected virsh call: {args}")

    def setmem_targets(self) -> list[tuple[str, int]]:
        return [
            (c[1], int(c[2].replace("KiB", "")))
            for c in self.calls
            if c[0] == "setmem"
        ]


@pytest.fixture(autouse=True)
def _no_locked_memory(monkeypatch: MonkeyPatch) -> None:
    """Default to "nothing is pinned"; the passthrough tests override this."""
    monkeypatch.setattr(hm, "locked_memory_kib", lambda name: 0)


@pytest.fixture
def virsh(monkeypatch: MonkeyPatch) -> FakeVirsh:
    fake = FakeVirsh({
        # 12 GiB balloon, 11.5 GiB unused inside — the real shape of the VM this
        # was built for.
        "gpu-vm": (12582912, 12103740),
        # 4 GiB balloon, mostly idle — has enough to give to be worth squeezing.
        "small-vm": (4194304, 3500000),
    })
    monkeypatch.setattr(hm, "_virsh", fake)
    monkeypatch.setattr(hm, "virsh_available", lambda: True)
    return fake


def _config(**kwargs: Any) -> hm.HostMemoryConfig:
    base = {"enabled": True, "domains": ["gpu-vm"], "state_file": "state.json"}
    base.update(kwargs)
    return hm.HostMemoryConfig.from_conf(base)


# ------------------------------------------------------------------- planning


def test_target_follows_what_the_guest_actually_uses() -> None:
    """A number typed months ago goes stale; the guest's own figure does not."""
    memory = hm.DomainMemory(name="gpu-vm", actual_kib=12582912, unused_kib=12103740)

    assert memory.used_kib == 479172  # ~468 MB really in use
    target = hm.plan_target_kib(memory, headroom_mb=1536, min_mb=1024)

    assert target == 479172 + 1536 * 1024
    assert target < memory.actual_kib


def test_unused_larger_than_actual_does_not_read_as_zero_need() -> None:
    """A ballooned guest can report free memory against its full size.

    Subtracting naively yields a negative "used", which would look like the guest
    needs nothing at all and invite squeezing it into an OOM.
    """
    memory = hm.DomainMemory(name="vm", actual_kib=2097152, unused_kib=3894592)

    assert memory.used_kib == 0
    assert hm.plan_target_kib(memory, headroom_mb=1536, min_mb=1024) == 1536 * 1024


def test_the_floor_is_never_crossed() -> None:
    memory = hm.DomainMemory(name="vm", actual_kib=8 * 1024 * 1024, unused_kib=8 * 1024 * 1024)

    assert hm.plan_target_kib(memory, headroom_mb=0, min_mb=1024) == 1024 * 1024


def test_planning_only_ever_shrinks() -> None:
    """A guest using nearly everything must not be handed more than it had."""
    memory = hm.DomainMemory(name="vm", actual_kib=2 * 1024 * 1024, unused_kib=0)

    assert hm.plan_target_kib(memory, headroom_mb=4096, min_mb=1024) == 2 * 1024 * 1024


# -------------------------------------------------------------------- freeing


def test_free_records_the_original_before_touching_anything(
    virsh: FakeVirsh, tmp_path: Path,
) -> None:
    """The state file is the only way back if the process dies mid-run."""
    freed = hm.free_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["domains"]["gpu-vm"]["original_kib"] == 12582912
    assert freed > 9000  # roughly 10 GB back


def test_state_is_written_before_the_balloon_moves(
    virsh: FakeVirsh, tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    """Ordering is the whole guarantee: shrink first, record later loses the VM."""
    order: list[str] = []

    real_write = hm._write_state

    def spy_write(path: Path, payload: dict[str, Any]) -> bool:
        order.append("write_state")
        return real_write(path, payload)

    def spy_setmem(name: str, kib: int, *, connect: str = "") -> bool:
        order.append("setmem")
        return True

    monkeypatch.setattr(hm, "_write_state", spy_write)
    monkeypatch.setattr(hm, "set_balloon", spy_setmem)

    hm.free_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    assert order[0] == "write_state"
    assert "setmem" in order


def test_an_unwritable_state_file_stops_the_squeeze(
    virsh: FakeVirsh, tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    """Without a way back, we do not go in."""
    monkeypatch.setattr(hm, "_write_state", lambda path, payload: False)

    freed = hm.free_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    assert freed == 0
    assert virsh.setmem_targets() == []


def test_disabled_config_does_nothing(virsh: FakeVirsh, tmp_path: Path) -> None:
    assert hm.free_host_memory(
        _config(enabled=False), repo_dir=tmp_path, quiet=True,
    ) == 0
    assert virsh.calls == []


def test_a_domain_with_little_to_give_is_left_alone(
    virsh: FakeVirsh, tmp_path: Path,
) -> None:
    """A balloon round-trip for a couple of hundred megabytes is not worth it."""
    virsh.domains["tight-vm"] = (2 * 1024 * 1024, 100 * 1024)

    hm.free_host_memory(
        _config(domains=["tight-vm"]), repo_dir=tmp_path, quiet=True,
    )

    assert virsh.setmem_targets() == []
    assert not (tmp_path / "state.json").exists()


def test_a_passthrough_domain_is_left_alone(
    virsh: FakeVirsh, tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    """Ballooning a VFIO domain is worse than useless, and was measured to be.

    Its whole guest memory is mlock'ed for IOMMU DMA, so the balloon moves and
    the guest reports free pages, but qemu's RSS does not drop by a byte. The
    guest loses memory and the host gains none.
    """
    monkeypatch.setattr(
        hm, "locked_memory_kib", lambda name: 12582912 if name == "gpu-vm" else 0,
    )

    freed = hm.free_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    assert freed == 0
    assert virsh.setmem_targets() == []
    assert not (tmp_path / "state.json").exists()


def test_a_partly_locked_domain_is_still_squeezed(
    virsh: FakeVirsh, tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    """A little locked memory (vhost buffers and such) is not passthrough."""
    monkeypatch.setattr(hm, "locked_memory_kib", lambda name: 64 * 1024)

    freed = hm.free_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    assert freed > 9000


def test_missing_virsh_is_not_fatal(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Freeing memory is an optimisation, not a precondition for running."""
    monkeypatch.setattr(hm, "virsh_available", lambda: False)

    assert hm.free_host_memory(_config(), repo_dir=tmp_path, quiet=True) == 0


# ------------------------------------------------------------------- restoring


def test_restore_gives_back_exactly_the_original(
    virsh: FakeVirsh, tmp_path: Path,
) -> None:
    hm.free_host_memory(_config(), repo_dir=tmp_path, quiet=True)
    assert virsh.domains["gpu-vm"][0] < 12582912

    hm.restore_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    assert virsh.domains["gpu-vm"][0] == 12582912
    assert not (tmp_path / "state.json").exists()


def test_restore_works_from_a_fresh_process(virsh: FakeVirsh, tmp_path: Path) -> None:
    """After an OOM kill nothing is left in memory — only the file on disk."""
    (tmp_path / "state.json").write_text(
        json.dumps({"domains": {"gpu-vm": {"original_kib": 12582912, "target_kib": 2000000}}}),
        encoding="utf-8",
    )
    virsh.domains["gpu-vm"] = (2000000, 1500000)

    restored = hm.restore_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    assert virsh.domains["gpu-vm"][0] == 12582912
    assert restored > 0


def test_restore_is_safe_to_repeat(virsh: FakeVirsh, tmp_path: Path) -> None:
    """It runs from `finally`, from a signal handler and from the next startup."""
    hm.free_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    hm.restore_host_memory(_config(), repo_dir=tmp_path, quiet=True)
    hm.restore_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    assert virsh.domains["gpu-vm"][0] == 12582912


def test_restore_after_a_reboot_does_not_inflate_the_vm(
    virsh: FakeVirsh, tmp_path: Path,
) -> None:
    """A host reboot brings the domain back at its configured size already."""
    (tmp_path / "state.json").write_text(
        json.dumps({"domains": {"gpu-vm": {"original_kib": 8000000}}}), encoding="utf-8",
    )

    hm.restore_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    # Already at 12 GB, above the recorded original — nothing was set.
    assert virsh.setmem_targets() == []
    assert not (tmp_path / "state.json").exists()


def test_a_failed_restore_keeps_the_state_for_the_next_attempt(
    virsh: FakeVirsh, tmp_path: Path,
) -> None:
    """Dropping the file would leave the VM squeezed with no record of its size."""
    hm.free_host_memory(_config(), repo_dir=tmp_path, quiet=True)
    virsh.fail_setmem_for = {"gpu-vm"}

    hm.restore_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["domains"]["gpu-vm"]["original_kib"] == 12582912


def test_a_stale_state_file_is_restored_before_the_next_squeeze(
    virsh: FakeVirsh, tmp_path: Path,
) -> None:
    """Otherwise the "original" recorded now is the already-shrunken size.

    That is how a VM loses memory permanently: squeeze, crash, squeeze again from
    the smaller figure, and the way back points at the wrong number.
    """
    virsh.domains["gpu-vm"] = (2000000, 1500000)
    (tmp_path / "state.json").write_text(
        json.dumps({"domains": {"gpu-vm": {"original_kib": 12582912}}}), encoding="utf-8",
    )

    hm.free_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["domains"]["gpu-vm"]["original_kib"] == 12582912


def test_corrupt_state_is_discarded_not_obeyed(
    virsh: FakeVirsh, tmp_path: Path,
) -> None:
    (tmp_path / "state.json").write_text("{ not json", encoding="utf-8")

    hm.restore_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    assert virsh.setmem_targets() == []
    assert not (tmp_path / "state.json").exists()


def test_restore_without_a_state_file_is_a_no_op(
    virsh: FakeVirsh, tmp_path: Path,
) -> None:
    assert hm.restore_host_memory(_config(), repo_dir=tmp_path, quiet=True) == 0
    assert virsh.calls == []


def test_empty_domain_list_covers_every_running_domain(
    virsh: FakeVirsh, tmp_path: Path,
) -> None:
    hm.free_host_memory(_config(domains=[]), repo_dir=tmp_path, quiet=True)

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert set(state["domains"]) == {"gpu-vm", "small-vm"}


# ------------------------------------------------ stopping passthrough domains


def test_a_stop_domain_is_shut_down_and_recorded(
    virsh: FakeVirsh, tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    """The only way to reclaim memory from a PCI-passthrough VM."""
    stopped: list[str] = []
    monkeypatch.setattr(
        hm, "stop_domain",
        lambda name, *, connect="", timeout_s=0: (stopped.append(name) or True),
    )

    hm.free_host_memory(
        _config(domains=[], stop_domains=["gpu-vm"]), repo_dir=tmp_path, quiet=True,
    )

    assert stopped == ["gpu-vm"]
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["stopped"] == ["gpu-vm"]
    # It must not also be ballooned — that would only slow the shutdown down.
    assert "gpu-vm" not in state["domains"]


def test_a_stopped_domain_is_started_again_on_restore(
    virsh: FakeVirsh, tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    started: list[str] = []
    monkeypatch.setattr(
        hm, "start_domain",
        lambda name, *, connect="", timeout_s=0: (started.append(name) or True),
    )
    (tmp_path / "state.json").write_text(
        json.dumps({"domains": {}, "stopped": ["gpu-vm"]}), encoding="utf-8",
    )

    hm.restore_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    assert started == ["gpu-vm"]
    assert not (tmp_path / "state.json").exists()


def test_a_domain_that_will_not_start_stays_on_the_books(
    virsh: FakeVirsh, tmp_path: Path, monkeypatch: MonkeyPatch,
) -> None:
    """Dropping the record would leave a VM switched off with nobody tracking it."""
    monkeypatch.setattr(
        hm, "start_domain", lambda name, *, connect="", timeout_s=0: False,
    )
    (tmp_path / "state.json").write_text(
        json.dumps({"domains": {}, "stopped": ["gpu-vm"]}), encoding="utf-8",
    )

    hm.restore_host_memory(_config(), repo_dir=tmp_path, quiet=True)

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["stopped"] == ["gpu-vm"]


def test_a_guest_that_ignores_shutdown_is_never_force_killed(
    monkeypatch: MonkeyPatch,
) -> None:
    """A corrupted guest is a far worse outcome than a run with less RAM."""
    calls: list[list[str]] = []

    def fake_virsh(args: list[str], *, connect: str = "") -> str | None:
        calls.append(args)
        if args[0] == "domstate":
            return "running\n"
        return ""

    monkeypatch.setattr(hm, "_virsh", fake_virsh)
    monkeypatch.setattr(hm.time, "sleep", lambda _s: None)
    monkeypatch.setattr(hm.time, "monotonic", _ticking_clock())

    assert hm.stop_domain("stubborn", timeout_s=10) is False
    assert not any(a[0] == "destroy" for a in calls)


def _ticking_clock():
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += 3.0
        return state["t"]

    return clock


def test_a_slow_start_is_not_mistaken_for_a_failure(monkeypatch: MonkeyPatch) -> None:
    """`virsh start` on a passthrough domain can outlast a client-side timeout.

    Judging by the return code then records the VM as still down while it is
    busy booting perfectly well — which is how a restore came to leave a stale
    "stopped" entry behind. Success is the domain's state, not the exit code.
    """
    states = iter(["shut off", "shut off", "shut off", "running"])
    calls: list[list[str]] = []

    def fake_virsh(args: list[str], *, connect: str = "", timeout_s: int = 0) -> str | None:
        calls.append(args)
        if args[0] == "domstate":
            return next(states, "running")
        if args[0] == "start":
            return None  # the command timed out client-side
        return ""

    monkeypatch.setattr(hm, "_virsh", fake_virsh)
    monkeypatch.setattr(hm.time, "sleep", lambda _s: None)

    assert hm.start_domain("gpu-vm", timeout_s=60) is True
    assert ["start"] == [a[0] for a in calls if a[0] == "start"]


def test_start_gives_up_only_after_the_deadline(monkeypatch: MonkeyPatch) -> None:
    def fake_virsh(args: list[str], *, connect: str = "", timeout_s: int = 0) -> str | None:
        return "shut off\n" if args[0] == "domstate" else ""

    monkeypatch.setattr(hm, "_virsh", fake_virsh)
    monkeypatch.setattr(hm.time, "sleep", lambda _s: None)
    monkeypatch.setattr(hm.time, "monotonic", _ticking_clock())

    assert hm.start_domain("gpu-vm", timeout_s=10) is False


def test_shutdown_is_repeated_while_the_guest_boots(monkeypatch: MonkeyPatch) -> None:
    """A guest that is still booting does not hear the ACPI power button.

    Its handler comes up later than qemu itself, so a single request at the start
    means a VM started a minute ago never shuts down — which is exactly what
    happened on the first real run.
    """
    calls: list[str] = []
    # Stays up through several polls, then finally answers a later request.
    states = iter(["running"] * 40 + ["shut off"])

    def fake_virsh(args: list[str], *, connect: str = "", timeout_s: int = 0) -> str | None:
        calls.append(args[0])
        return next(states, "shut off") if args[0] == "domstate" else ""

    monkeypatch.setattr(hm, "_virsh", fake_virsh)
    monkeypatch.setattr(hm.time, "sleep", lambda _s: None)
    monkeypatch.setattr(hm.time, "monotonic", _ticking_clock())

    assert hm.stop_domain("slow-guest", timeout_s=600) is True
    assert calls.count("shutdown") > 1


# ------------------------------------------------------------------- signals


def test_the_signal_handler_cannot_be_swallowed(monkeypatch: MonkeyPatch) -> None:
    """`sys.exit()` in the handler was silently ignored by the pipeline.

    The analyze stage catches SystemExit so one failed call cannot bring down a
    whole run — so a SIGTERM handed the memory back to the VMs and then let
    processing carry on without it, which is the worst of both. Resetting the
    disposition and re-raising at ourselves cannot be caught by anything.
    """
    restored: list[int] = []
    dispositions: list[object] = []
    killed: list[int] = []
    handlers: dict[int, Any] = {}

    def fake_signal(sig: int, handler: Any) -> Any:
        if handler is hm.signal.SIG_DFL:
            dispositions.append(sig)
        else:
            handlers[sig] = handler
        return hm.signal.SIG_DFL

    monkeypatch.setattr(hm.signal, "signal", fake_signal)
    monkeypatch.setattr(hm.signal, "getsignal", lambda sig: hm.signal.SIG_DFL)
    monkeypatch.setattr(hm.os, "kill", lambda pid, sig: killed.append(sig))

    hm.install_restore_on_signals(lambda: restored.append(1))
    handlers[hm.signal.SIGTERM](hm.signal.SIGTERM, None)

    assert restored == [1]
    # Order matters: memory back first, then the default disposition, then die.
    assert dispositions == [hm.signal.SIGTERM]
    assert killed == [hm.signal.SIGTERM]


def test_an_ignored_signal_is_left_alone(monkeypatch: MonkeyPatch) -> None:
    """Under `nohup` SIGHUP is SIG_IGN.

    Taking it over would turn closing the terminal back into killing the very
    background run that nohup was used to protect.
    """
    installed: list[int] = []

    monkeypatch.setattr(
        hm.signal, "getsignal",
        lambda sig: hm.signal.SIG_IGN if sig == hm.signal.SIGHUP else hm.signal.SIG_DFL,
    )
    monkeypatch.setattr(
        hm.signal, "signal",
        lambda sig, handler: installed.append(sig) or hm.signal.SIG_DFL,
    )

    hm.install_restore_on_signals(lambda: None)

    assert hm.signal.SIGHUP not in installed
    assert hm.signal.SIGTERM in installed


def test_failing_to_arm_a_handler_is_not_fatal(monkeypatch: MonkeyPatch) -> None:
    """Off the main thread signal.signal raises; the run must still proceed."""

    def boom(sig: int, handler: Any) -> Any:
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(hm.signal, "getsignal", lambda sig: hm.signal.SIG_DFL)
    monkeypatch.setattr(hm.signal, "signal", boom)

    hm.install_restore_on_signals(lambda: None)  # must not raise
