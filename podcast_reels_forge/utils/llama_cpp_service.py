"""Utilities for managing a local llama.cpp server process.

These helpers are intentionally small and dependency-free so they can be used
from both the pipeline orchestrator and stage scripts.
"""

from __future__ import annotations

import functools
import logging
import os
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Final
from urllib.parse import urlparse

LOGGER = logging.getLogger("Forge")

# When set to "1", stage scripts should not start/stop llama-server themselves.
ENV_MANAGED_BY_PIPELINE: Final[str] = "FORGE_MANAGED_LLAMA_CPP"


def wait_tcp(host: str, port: int, timeout_s: int = 20) -> bool:
    """Wait until a TCP port starts accepting connections."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def is_tcp_open(host: str, port: int) -> bool:
    """Return True if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def wait_for_server_ready(host: str, port: int, *, timeout_s: int = 300) -> bool:
    """Poll /health until the server reports status=ok, or timeout expires.

    Returns True when the server is ready, False if timeout is reached.
    Uses stdlib only (no third-party deps) so it works before venv activation.
    """
    import json
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout_s
    logged_waiting = False
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    return True
        except urllib.error.HTTPError:
            # 503 "Loading model" → still loading, keep waiting
            if not logged_waiting:
                LOGGER.info("llama.cpp server loading model; waiting (max %ds)...", timeout_s)
                logged_waiting = True
        except Exception:
            pass
        time.sleep(3)
    LOGGER.warning("llama.cpp server not ready after %ds; proceeding anyway", timeout_s)
    return False


def parse_local_llama_cpp_host_port(url: str) -> tuple[str, int] | None:
    """Return (host, port) if URL points to local llama.cpp server, else None."""
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return None
    if host not in {"127.0.0.1", "localhost"}:
        return None
    port = parsed.port or 8080
    return (host, port)


def _build_llama_server_cmd(
    *,
    model_path: str,
    host: str,
    port: int,
    threads: int,
    ctx_size: int,
    n_gpu_layers: int,
    batch_size: int,
    ubatch_size: int,
    main_gpu: int,
    parallel: int,
    extra_args: list[str] | None,
    cache_type_k: str | None = "q8_0",
    cache_type_v: str | None = "q8_0",
    cache_ram_mb: int | None = None,
    ctx_checkpoints: int | None = None,
) -> list[str]:
    cmd = [
        "llama-server",
        "-m",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "-t",
        str(threads),
        "-c",
        str(ctx_size),
        "--batch-size",
        str(batch_size),
        "--ubatch-size",
        str(ubatch_size),
        "--main-gpu",
        str(main_gpu),
        # RU: Явное значение: голый --flash-attn в новых сборках принимает [on|off|auto]
        #     и может «съесть» следующий аргумент как значение.
        # EN: Pass an explicit value: bare --flash-attn in recent builds takes [on|off|auto]
        #     and could swallow the next argument as its value.
        "--flash-attn",
        "on",
    ]
    # RU: n_gpu_layers=0 → авто-fit решает сколько слоёв на GPU (максимально возможно).
    #     Любое ненулевое значение передаётся явно и отключает auto-fit.
    # EN: n_gpu_layers=0 → auto-fit decides layer count (fills VRAM as much as possible).
    #     Any non-zero value is passed explicitly and disables auto-fit.
    if n_gpu_layers != 0:
        cmd += ["--n-gpu-layers", str(n_gpu_layers)]
    # RU: Квантование KV-кэша (требует flash-attn) вдвое снижает VRAM под кэш —
    #     это даёт запас на 16GB для большего контекста/parallel.
    # EN: KV-cache quantization (needs flash-attn) halves KV VRAM — frees headroom
    #     on 16GB for larger context / more parallel slots.
    if cache_type_k:
        cmd += ["--cache-type-k", str(cache_type_k)]
    if cache_type_v:
        cmd += ["--cache-type-v", str(cache_type_v)]
    # RU: --parallel всегда передаём явно — иначе llama-server ставит auto (=4),
    #     что занимает лишние ~3-4 GB VRAM под KV-cache дополнительных слотов.
    # EN: Always pass --parallel explicitly — without it llama-server defaults to
    #     auto (=4), wasting ~3-4 GB of VRAM on extra KV-cache slots.
    cmd += ["--parallel", str(parallel)]
    # RU: Кэш промптов живёт в ОПЕРАТИВНОЙ памяти хоста, и по умолчанию ему
    #     разрешено 8192 МиБ. Именно он, а не веса, набрал 6 ГБ и стал нашим
    #     вкладом в OOM: модель при n_gpu_layers=999 целиком на GPU, а хост
    #     держал 32 контекстных чекпоинта по ~160 МБ. Флаг есть не во всех
    #     сборках, поэтому передаём только если он поддерживается — иначе
    #     llama-server не запустится вовсе.
    # EN: The prompt cache lives in host RAM and is allowed 8192 MiB by default.
    #     That, not the weights, grew to 6 GB and became our share of the OOM:
    #     with n_gpu_layers=999 the model is entirely on the GPU while the host
    #     held 32 context checkpoints of ~160 MB each. The flag is missing from
    #     older builds, so it is passed only when supported — otherwise
    #     llama-server would refuse to start at all.
    if cache_ram_mb is not None and _server_supports("--cache-ram"):
        cmd += ["--cache-ram", str(cache_ram_mb)]
    if ctx_checkpoints is not None and _server_supports("--ctx-checkpoints"):
        cmd += ["--ctx-checkpoints", str(ctx_checkpoints)]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


#: RU: Куда писать вывод llama-server. Раньше он уходил в /dev/null, и это
#:     скрывало ровно тот отчёт, по которому видно, сколько слоёв реально уехало
#:     на GPU и сколько буферов осталось в RAM. Диагностировать «сервер занял
#:     6 ГБ хостовой памяти при полном оффлоаде» было нечем.
#: EN: Where llama-server's output goes. It used to be discarded, which hid the
#:     one report showing how many layers actually reached the GPU and how much
#:     buffer stayed in RAM. There was no way to diagnose "the server took 6 GB
#:     of host memory under a supposedly full offload".
DEFAULT_SERVER_LOG: Final = "llama-server.log"


def _open_server_log(raw: object, cmd: list[str]) -> IO[bytes] | None:
    """RU: Открывает лог llama-server на дозапись; None — писать в /dev/null.

    EN: Open the llama-server log for appending; None means "discard".

    A log that cannot be opened must not stop the pipeline — the server itself is
    what matters, so we fall back to discarding as before.
    """

    path_text = str(raw or "").strip()
    if not path_text:
        return None
    try:
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("ab")
    except OSError as exc:
        LOGGER.warning("Cannot write llama-server log to %s: %s", path_text, exc)
        return None
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    handle.write(f"\n===== {stamp} {' '.join(cmd)} =====\n".encode())
    handle.flush()
    return handle


def available_ram_mb() -> int | None:
    """RU: Сколько памяти реально доступно сейчас, в МиБ.

    EN: How much memory is genuinely available right now, in MiB.

    Reads ``MemAvailable``, not ``MemFree``: the kernel's own estimate of what a
    new allocation can get, page cache it is willing to drop included. Returns
    None where /proc is unavailable.
    """

    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) // 1024
    return None


def resolve_cache_ram_mb(
    raw: object,
    *,
    reserve_mb: int,
    min_mb: int,
    max_mb: int,
) -> int | None:
    """RU: Размер кэша промптов: число, None или расчёт от свободной памяти.

    EN: The prompt-cache size: a number, None, or derived from free memory.

    ``auto`` takes what is free right now minus a reserve, which is the only way
    to be both generous and safe: the cache is worth real time on repeated
    prefixes, but a number fixed in advance cannot know whether a VM was shut
    down for this run or a browser has since eaten the difference.

    The reserve is what everything else on the machine is allowed to grow into
    without pushing the host into an OOM. Note this is decided once, when the
    server starts — it is a budget, not a live guard.
    """

    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"", "null", "none"}:
        return None
    if text != "auto":
        try:
            return max(0, int(float(text)))
        except ValueError:
            LOGGER.warning("Не разобрать cache_ram_mb=%r; кэш не ограничиваю", raw)
            return None

    available = available_ram_mb()
    if available is None:
        LOGGER.warning("Не прочитать MemAvailable; беру нижнюю границу кэша")
        return min_mb
    budget = available - reserve_mb
    resolved = max(min_mb, min(budget, max_mb))
    LOGGER.info(
        "Кэш промптов: %d МиБ (доступно %d, резерв %d, потолок %d)",
        resolved, available, reserve_mb, max_mb,
    )
    return resolved


@functools.lru_cache(maxsize=8)
def _server_supports(flag: str) -> bool:
    """RU: Понимает ли установленный llama-server такой флаг.

    EN: Does the installed llama-server understand this flag.

    Checked rather than assumed: llama.cpp adds and renames options often, and an
    unknown one makes the server exit instead of start. Cached, so it costs one
    `--help` per flag per process.
    """

    try:
        res = subprocess.run(
            ["llama-server", "--help"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return flag in (res.stdout or "") + (res.stderr or "")


def llama_cpp_start(
    *,
    host: str,
    port: int,
    service_conf: dict[str, Any] | None = None,
) -> subprocess.Popen | None:
    """Start local llama-server in background, returning the started process.

    Returns:
        subprocess.Popen if a new instance was started by this call.
        None if server is already running, unavailable, or failed to start.
    """
    if is_tcp_open(host, port):
        LOGGER.info(
            "llama.cpp server already running on %s:%s; not starting a new instance",
            host,
            port,
        )
        return None

    conf = dict(service_conf or {})
    model_path = str(conf.get("model_path", "")).strip()
    if not model_path:
        LOGGER.warning("llama_cpp.service.model_path is not set; cannot auto-start llama-server")
        return None
    if not Path(model_path).exists():
        LOGGER.warning("llama.cpp model file not found: %s", model_path)
        return None

    cpu_default_threads = max(4, (os.cpu_count() or 8) - 2)

    cmd = _build_llama_server_cmd(
        model_path=model_path,
        host=host,
        port=port,
        threads=int(conf.get("threads", cpu_default_threads)),
        ctx_size=int(conf.get("ctx_size", 8192)),
        n_gpu_layers=int(conf.get("n_gpu_layers", 99)),
        batch_size=int(conf.get("batch_size", 1024)),
        ubatch_size=int(conf.get("ubatch_size", 512)),
        main_gpu=int(conf.get("main_gpu", 0)),
        parallel=max(1, int(conf.get("parallel", 1))),
        extra_args=[str(x) for x in conf.get("extra_args", []) if str(x).strip()],
        cache_type_k=str(conf.get("cache_type_k", "q8_0")) or None,
        cache_type_v=str(conf.get("cache_type_v", "q8_0")) or None,
        cache_ram_mb=resolve_cache_ram_mb(
            conf.get("cache_ram_mb", "auto"),
            reserve_mb=int(conf.get("cache_ram_reserve_mb", 6144)),
            min_mb=int(conf.get("cache_ram_min_mb", 512)),
            max_mb=int(conf.get("cache_ram_max_mb", 8192)),
        ),
        ctx_checkpoints=(
            int(conf["ctx_checkpoints"])
            if conf.get("ctx_checkpoints") is not None
            else None
        ),
    )

    log_handle = _open_server_log(conf.get("log_file", DEFAULT_SERVER_LOG), cmd)
    try:
        target = log_handle if log_handle is not None else subprocess.DEVNULL
        p = subprocess.Popen(cmd, stdout=target, stderr=subprocess.STDOUT)
        if wait_tcp(host, port, timeout_s=int(conf.get("startup_timeout", 60))):
            if p.poll() is None:
                return p
            LOGGER.warning("llama-server exited early while port became available")
            return None

        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    except FileNotFoundError:
        LOGGER.warning("llama-server binary not found in PATH")
        return None
    except OSError as exc:
        LOGGER.warning("Failed to start llama-server: %s", exc)
        return None
    return None


def llama_cpp_stop(p: subprocess.Popen) -> None:
    """Terminate a llama-server process started by this app."""
    try:
        p.terminate()
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        LOGGER.warning("llama-server did not terminate in time; killing")
        try:
            p.kill()
        except OSError:
            LOGGER.exception("Failed to kill llama-server process")
    except OSError:
        LOGGER.exception("Failed to terminate llama-server process")
