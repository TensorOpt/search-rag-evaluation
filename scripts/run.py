"""eval:run — placeholder entry point, wired in Phase 11.

Non-destructive: imports cleanly, logs a notice, exits 0. The real run path
(ExperimentRunner.run over config.yaml, §8.0) is delivered in Phase 11.
"""

from __future__ import annotations

from benchmark.logging_setup import get_logger, setup_logging

log = get_logger(__name__)


def main() -> int:
    setup_logging()
    log.info("eval:run is a placeholder; wired in Phase 11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
