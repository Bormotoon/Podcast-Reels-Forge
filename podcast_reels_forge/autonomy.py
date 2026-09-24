"""RU: Всё, что нужно прогону без присмотра: лог-файл, отчёт, уведомление.

EN: What an unattended run needs: a log file, the report, a notification.

Configured by the optional ``autonomy`` block of config.yaml::

    autonomy:
      lock_file: .forge.lock        # one run at a time
      log_dir: logs                 # daily-rotated forge.log
      log_keep_days: 14
      runs_dir: ""                  # default: <output_dir>/_runs
      min_free_disk_gb: 5           # preflight refuses to start below this
      notify:
        when: failure               # always | failure | never
        command: ""                 # shell command; gets FORGE_* env vars
        webhook: ""                 # POST of the report summary as JSON
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from podcast_reels_forge.run_report import RunReport

log = logging.getLogger("Forge")

_FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def autonomy_conf(conf: Mapping[str, Any]) -> Mapping[str, Any]:
    section = conf.get("autonomy") if isinstance(conf, Mapping) else None
    return section if isinstance(section, Mapping) else {}


def _resolve(repo_dir: Path, raw: object, default: str) -> Path:
    path = Path(str(raw or default)).expanduser()
    return path if path.is_absolute() else repo_dir / path


def lock_path(conf: Mapping[str, Any], repo_dir: Path) -> Path:
    return _resolve(repo_dir, autonomy_conf(conf).get("lock_file"), ".forge.lock")


def runs_dir(conf: Mapping[str, Any], repo_dir: Path) -> Path:
    raw = autonomy_conf(conf).get("runs_dir")
    if raw:
        return _resolve(repo_dir, raw, "")
    paths = conf.get("paths") if isinstance(conf, Mapping) else None
    output_dir = (paths or {}).get("output_dir", "output") if isinstance(paths, Mapping) else "output"
    return _resolve(repo_dir, output_dir, "output") / "_runs"


def setup_file_logging(conf: Mapping[str, Any], repo_dir: Path) -> Path | None:
    """Attach a daily-rotated ``forge.log`` to the root logger.

    Always at INFO, whatever the console verbosity: the console is for a
    person watching, the file is for the morning after. Returns the log path,
    or None when ``log_dir`` is set to an empty string.
    """

    section = autonomy_conf(conf)
    if "log_dir" in section and not section.get("log_dir"):
        return None
    log_dir = _resolve(repo_dir, section.get("log_dir"), "logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        keep = int(section.get("log_keep_days", 14))
        handler = logging.handlers.TimedRotatingFileHandler(
            log_dir / "forge.log", when="midnight", backupCount=max(1, keep), encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        log.warning("log file disabled: %s", exc)
        return None
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    root = logging.getLogger()
    console_level = root.level or logging.WARNING
    # Keep the console as quiet as it was; open only the root gate to INFO.
    for existing in root.handlers:
        if existing.level == logging.NOTSET:
            existing.setLevel(console_level)
    root.addHandler(handler)
    root.setLevel(min(console_level, logging.INFO))
    # "forge" is raised to ERROR under --quiet, which would starve the file;
    # its own console handler carries its own level, so lowering the logger
    # changes only what reaches the file.
    for name in ("Forge", "forge"):
        named = logging.getLogger(name)
        if named.level > logging.INFO:
            named.setLevel(logging.INFO)
    return log_dir / "forge.log"


def should_notify(conf: Mapping[str, Any], report: RunReport) -> bool:
    notify = autonomy_conf(conf).get("notify")
    if not isinstance(notify, Mapping):
        return False
    # `on:` is what people type, but YAML 1.1 reads a bare `on` as True —
    # hence `when`, with the parsed-away key still honoured.
    raw = notify.get("when", notify.get("on", notify.get(True, "failure")))
    when = str(raw).strip().lower()
    if when == "never" or not (notify.get("command") or notify.get("webhook")):
        return False
    return when == "always" or report.outcome != "ok"


def notify(conf: Mapping[str, Any], report: RunReport, report_path: Path | None) -> None:
    """Run the configured notification hooks. Never raises."""

    if not should_notify(conf, report):
        return
    section = autonomy_conf(conf).get("notify")
    assert isinstance(section, Mapping)
    summary = report.summary_line()

    command = str(section.get("command") or "").strip()
    if command:
        env = {
            **os.environ,
            "FORGE_OUTCOME": report.outcome,
            "FORGE_EXIT_CODE": str(report.exit_code()),
            "FORGE_SUMMARY": summary,
            "FORGE_REPORT": str(report_path or ""),
        }
        try:
            subprocess.run(command, shell=True, env=env, timeout=60, check=False)  # noqa: S602
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("notify command failed: %s", exc)

    webhook = str(section.get("webhook") or "").strip()
    if webhook:
        data = report.to_dict()
        payload = {
            "text": f"Podcast Reels Forge: {summary}",
            "outcome": data["outcome"],
            "episodes_failed": data["episodes_failed"],
            "clips_total": data["clips_total"],
            "fatal_error": data["fatal_error"],
            "report": str(report_path or ""),
        }
        try:
            import requests

            requests.post(webhook, data=json.dumps(payload), timeout=20,
                          headers={"Content-Type": "application/json"})
        except Exception as exc:  # network trouble must not fail the run
            log.warning("notify webhook failed: %s", exc)
