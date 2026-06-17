"""Utilities for managing a local llama.cpp server process.

These helpers are intentionally small and dependency-free so they can be used
from both the pipeline orchestrator and stage scripts.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Final
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
    if extra_args:
        cmd.extend(extra_args)
    return cmd


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
    )

    try:
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
