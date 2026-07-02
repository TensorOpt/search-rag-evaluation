"""eval:index — build + populate the ES index (docs/experiment.md §3.5). Phase 11.

Drives the ONE shared index-build path (:meth:`benchmark.runner.ExperimentRunner.build_index`):
register each embedder endpoint → ``ensure_index`` (``search_text`` ``copy_to`` one ``semantic_text``
field per embedder) → streamed ``bulk_index``. ``eval:index`` builds/populates the index; ``eval:run``
consumes it. Logs the index name + a non-zero doc count so a populated index is verifiable.
"""

from __future__ import annotations

import argparse

from benchmark.config import load_config
from benchmark.logging_setup import get_logger, setup_logging
from benchmark.runner import ExperimentRunner

log = get_logger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eval:index", description="Build + populate the ES index.")
    parser.add_argument("--config", default="config.yaml", help="path to the config file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(timestamp=cfg.timestamp)

    _dataset, backend, mapping = ExperimentRunner().build_index(cfg)
    doc_count = backend.client.count(index=mapping.index_name)["count"]
    log.info("index %r built and populated: %d docs", mapping.index_name, doc_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
