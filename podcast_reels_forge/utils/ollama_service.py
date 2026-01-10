"""Utilities for managing a local Ollama server process.

These helpers are intentionally small and dependency-free so they can be used
from both the pipeline orchestrator and stage scripts.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import time
from typing import Final
from urllib.parse import urlparse

LOGGER = logging.getLogger("Forge")

# When set to "1", stage scripts should not start/stop Ollama themselves.
ENV_MANAGED_BY_PIPELINE: Final[str] = "FORGE_MANAGED_OLLAMA"


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


def parse_local_ollama_host_port(url: str) -> tuple[str, int] | None:
    """Return (host, port) if URL points to local Ollama, else None."""
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return None
    if host not in {"127.0.0.1", "localhost"}:
        return None
    port = parsed.port or 11434
    return (host, port)


def ollama_start(*, host: str, port: int) -> subprocess.Popen | None:
    """Start Ollama server in background, returning the process we started.

    Returns:
        subprocess.Popen if a new instance was started by this call.
        None if Ollama is already running, unavailable, or failed to start.
    """
    if is_tcp_open(host, port):
        LOGGER.info(
            "Ollama already running on %s:%s; not starting a new instance",
            host,
            port,
        )
        return None

    try:
        p = subprocess.Popen(
            ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if wait_tcp(host, port, timeout_s=30):
            if p.poll() is None:
                return p
            LOGGER.warning("Ollama process exited early while port became available")
            return None

        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    except FileNotFoundError:
        return None
    except OSError as exc:
        LOGGER.warning("Failed to start Ollama: %s", exc)
        return None
    return None


def ollama_stop(p: subprocess.Popen) -> None:
    """Terminate an Ollama process started by this app."""
    try:
        p.terminate()
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        LOGGER.warning("Ollama did not terminate in time; killing")
        try:
            p.kill()
        except OSError:
            LOGGER.exception("Failed to kill Ollama process")
    except OSError:
        LOGGER.exception("Failed to terminate Ollama process")
