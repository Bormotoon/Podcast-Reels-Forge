#!/usr/bin/env python3
"""RU: Ручное управление памятью виртуалок на время прогона.

Тот же код, что пайплайн вызывает сам. Отдельная команда нужна прежде всего для
`--restore`: если процесс убили по OOM или машину перезагрузили посреди прогона,
ВМ останутся сжатыми, и вернуть их надо будет вручную.

    python3 -m podcast_reels_forge.scripts.host_memory --status
    python3 -m podcast_reels_forge.scripts.host_memory --restore

EN: Manual control of VM memory for the duration of a run.

The same code the pipeline calls on its own. The standalone command exists mainly
for `--restore`: if the process was OOM-killed or the machine rebooted mid-run,
the VMs stay squeezed and need putting back by hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from podcast_reels_forge.utils.host_memory import (
    LOCKED_SHARE_THRESHOLD,
    HostMemoryConfig,
    domain_memory,
    free_host_memory,
    list_running_domains,
    locked_memory_kib,
    plan_target_kib,
    restore_host_memory,
    virsh_available,
)

REPO_DIR = Path(__file__).resolve().parents[2]


def _load_config(config_path: Path) -> HostMemoryConfig:
    try:
        with config_path.open(encoding="utf-8") as handle:
            conf = yaml.safe_load(handle) or {}
    except OSError:
        conf = {}
    return HostMemoryConfig.from_conf(
        conf.get("host_memory") if isinstance(conf, dict) else None,
    )


def _print_status(config: HostMemoryConfig) -> int:
    if not virsh_available():
        print("virsh не найден — управлять памятью ВМ нельзя", file=sys.stderr)
        return 1

    names = list(config.domains) or list_running_domains(connect=config.connect)
    if not names:
        print("Запущенных доменов нет")
        return 0

    state_path = REPO_DIR / config.state_file
    print(f"Файл состояния: {state_path}" + ("" if state_path.exists() else " (нет)"))
    print(f"{'домен':<22} {'сейчас':>10} {'занято':>10} {'итог':>28}")
    for name in names:
        memory = domain_memory(name, connect=config.connect)
        if memory is None:
            print(f"{name:<22} {'—':>10} {'—':>10} {'нет баллона':>28}")
            continue
        locked = locked_memory_kib(name)
        head = f"{name:<22} {memory.actual_kib // 1024:>8} МБ {memory.used_kib // 1024:>8} МБ"
        if locked >= memory.actual_kib * LOCKED_SHARE_THRESHOLD:
            print(f"{head} {'заблокировано под PCI —':>20} пропуск")
            continue
        target = plan_target_kib(
            memory, headroom_mb=config.headroom_mb, min_mb=config.min_mb,
        )
        print(f"{head} {'сожмётся до ' + str(target // 1024) + ' МБ':>28}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """RU: Точка входа. EN: Entry point."""

    ap = argparse.ArgumentParser(
        description="Reclaim and restore VM memory around a pipeline run.",
    )
    ap.add_argument("--config", default="config.yaml", help="Путь к config.yaml")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true", help="Показать раскладку и выйти")
    group.add_argument("--free", action="store_true", help="Забрать у ВМ лишнюю память")
    group.add_argument(
        "--restore", action="store_true", help="Вернуть ВМ исходный размер",
    )
    ap.add_argument("--quiet", action="store_true", help="Только ошибки")
    args = ap.parse_args(argv)

    config = _load_config(Path(args.config))

    if args.status:
        return _print_status(config)

    if args.restore:
        # Restoring is always allowed: after a crash the config flag says nothing
        # about whether memory is currently owed back.
        restored = restore_host_memory(config, repo_dir=REPO_DIR, quiet=args.quiet)
        if not args.quiet and not restored:
            print("Возвращать нечего — файла состояния нет")
        return 0

    # --free
    forced = HostMemoryConfig(
        enabled=True,
        domains=config.domains,
        headroom_mb=config.headroom_mb,
        min_mb=config.min_mb,
        state_file=config.state_file,
        connect=config.connect,
    )
    freed = free_host_memory(forced, repo_dir=REPO_DIR, quiet=args.quiet)
    if not args.quiet:
        print(f"Освобождено {freed} МБ" if freed else "Освобождать нечего")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
