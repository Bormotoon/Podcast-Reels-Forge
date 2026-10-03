"""Load config.yaml with host-local overlays.

RU: Загрузка config.yaml с локальными наложениями.

config.yaml holds the shared defaults and is committed (the GUI also rewrites
it on export). Settings that belong to one machine — a llama-server cache cap
that keeps systemd-oomd away, a different input folder — used to be edited
into it and either leaked into commits or sat uncommitted forever. Two
overlays keep them out:

* ``extends: config.yaml`` at the top of a config file makes it a partial
  overlay of that base (``config.pos.yaml`` = ``extends`` + ``paths.input_dir``);
* ``config.local.yaml`` next to the loaded file, if present, is merged last.
  It is gitignored and survives GUI exports.

Mappings merge key by key; any other value (lists included) replaces the base.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

LOCAL_OVERLAY_NAME = "config.local.yaml"
_MAX_EXTENDS_DEPTH = 8


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml  # noqa: PLC0415 - keep import time low for --help

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the top level must be a mapping")
    return data


def _load_with_extends(path: Path, depth: int, sources: list[Path]) -> dict[str, Any]:
    if depth > _MAX_EXTENDS_DEPTH:
        raise ValueError(f"{path}: `extends` chain is too deep (a cycle?)")
    data = _read_yaml(path)
    parent = data.pop("extends", None)
    if parent:
        parent_path = Path(str(parent)).expanduser()
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        base = _load_with_extends(parent_path, depth + 1, sources)
        data = deep_merge(base, data)
    sources.append(path)
    return data


def load_config_with_sources(path: Path | str) -> tuple[dict[str, Any], list[Path]]:
    """The merged config and the files it came from, base first."""

    config_path = Path(path)
    sources: list[Path] = []
    conf = _load_with_extends(config_path, 0, sources)
    # Next to the loaded file, else next to the root of its `extends` chain:
    # a config kept elsewhere that extends config.yaml keeps the host tweaks.
    loaded = {s.resolve() for s in sources}
    for folder in (config_path.parent, sources[0].parent):
        local = folder / LOCAL_OVERLAY_NAME
        if local.exists() and local.resolve() not in loaded:
            conf = deep_merge(conf, _read_yaml(local))
            sources.append(local)
            break
    return conf, sources


def load_config(path: Path | str) -> dict[str, Any]:
    conf, sources = load_config_with_sources(path)
    if len(sources) > 1:
        LOG.info("Config: %s", " + ".join(str(s) for s in sources))
    return conf
