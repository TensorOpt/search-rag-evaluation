"""eval:run — run the whole benchmark end to end (docs/experiment.md §8.0). Phase 11.

Loads the explicit §10 config, then drives :class:`benchmark.runner.ExperimentRunner` over it:
index build → one ``run_one`` per pipeline (baseline first) → the family-wide comparator pass →
run-config JSON (§8.0). ``--dry-run`` prints the pipeline list (baseline first) and writes NOTHING.
"""

from __future__ import annotations

import argparse

from benchmark.config import load_config
from benchmark.common.logging_setup import get_logger, setup_logging
from benchmark.runner import ExperimentRunner

log = get_logger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eval:run", description="Run the search-relevance benchmark.")
    parser.add_argument("--config", default="config.yaml", help="path to the config file")
    parser.add_argument(
        "--output-dir", default="results", help="directory for the CSV/JSON artifacts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the pipeline list (baseline first) and exit without running or writing anything",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(timestamp=cfg.timestamp)

    if args.dry_run:
        # Baseline first, then the named variants in config order (§8.0/§10). Write nothing.
        for position, pcfg in enumerate(cfg.pipelines()):
            marker = "baseline" if pcfg.id == cfg.baseline_id else f"variant {position}"
            log.info("pipeline %d: %s (%s)", position, pcfg.id, marker)
        log.info("--dry-run: %d pipeline(s) listed; nothing was run or written", len(cfg.pipelines()))
        return 0

    ExperimentRunner().run(cfg, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
