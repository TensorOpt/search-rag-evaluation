"""Console + file logging setup, shared by the harness and the eval scripts.

Use logging everywhere instead of ``print()`` (see CLAUDE.md). Call
:func:`setup_logging` once at program start to attach a console handler (stderr)
and a per-run file handler at ``logs/run_{timestamp}.log``; obtain module
loggers with :func:`get_logger`.

This is a cross-cutting leaf utility: it imports only the standard library and
may be imported by any module (it is not a dataset/backend adapter, so it does
not affect the generality invariant in docs/experiment.md §11).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

_configured = False
_log_path: Path | None = None


def setup_logging(
    level: str | int | None = None,
    *,
    log_dir: str | Path = "logs",
    timestamp: str | None = None,
) -> Path | None:
    """Idempotently configure root logging: console (stderr) plus a per-run file
    ``{log_dir}/run_{timestamp}.log``.

    - ``level`` defaults to the ``LOG_LEVEL`` env var or ``INFO``.
    - ``timestamp`` defaults to the current UTC time as ``YYYYMMDDTHHMMSSZ``
      (the §9 artifact convention); pass the run's timestamp to align the log
      file name with that run's CSV artifacts.
    - Set ``LOG_TO_FILE=0`` to disable the file handler (console only).

    Safe to call more than once; only the first call configures handlers.
    Returns the log file path, or ``None`` when file logging is disabled.
    """
    global _configured, _log_path
    if _configured:
        return _log_path

    root = logging.getLogger()
    root.setLevel(level or os.environ.get("LOG_LEVEL", "INFO").upper())
    fmt = logging.Formatter(DEFAULT_FORMAT)

    console = logging.StreamHandler()  # stderr
    console.setFormatter(fmt)
    root.addHandler(console)

    if os.environ.get("LOG_TO_FILE", "1") != "0":
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        _log_path = directory / f"run_{ts}.log"
        file_handler = logging.FileHandler(_log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    _configured = True
    return _log_path


def get_logger(name: str) -> logging.Logger:
    """Return a module logger; pair with :func:`setup_logging` at program start."""
    return logging.getLogger(name)
