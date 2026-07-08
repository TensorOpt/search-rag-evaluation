"""eval:run — run the whole benchmark end to end (docs/architecture.md §6). Phase 11.

Loads the explicit §10 config, then drives :class:`benchmark.runner.ExperimentRunner` over it:
verify the index (built by ``eval:index``) is fully populated → one ``run_one`` per pipeline
(baseline first) → the family-wide comparator pass → run-config JSON (§6). ``eval:run`` does NOT
index; if the index is missing or partial it exits non-zero with a clear message (build it with
``eval:index`` first). ``--dry-run`` prints the pipeline list (baseline first) and writes NOTHING.
"""

from __future__ import annotations

import argparse

from benchmark.config import load_config
from benchmark.common.logging_setup import get_logger, setup_logging
from benchmark.runner import ExperimentRunner, IndexNotReadyError

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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="measure per-system cost + stage latency (P1-3): emit a cost_latency_{ts}.csv and a "
        "diagnostics.cost_latency manifest block. Off by default (a standard run stays byte-identical). "
        "Profile a COLD-cache run for a full read — a warm cache reports ~0 marginal API cost",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(timestamp=cfg.timestamp)

    if args.dry_run:
        # Baseline first, then the named variants in config order (§6/§10). Write nothing.
        for position, pcfg in enumerate(cfg.pipelines()):
            marker = "baseline" if pcfg.id == cfg.baseline_id else f"variant {position}"
            log.info("pipeline %d: %s (%s)", position, pcfg.id, marker)
        log.info("--dry-run: %d pipeline(s) listed; nothing was run or written", len(cfg.pipelines()))
        return 0

    try:
        ExperimentRunner().run(cfg, output_dir=args.output_dir, profile=args.profile)
    except IndexNotReadyError as exc:
        # The index isn't built/complete — a config/workflow error, not a crash. Clear message, exit 1.
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
