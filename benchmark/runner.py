"""The single execution path — ``ExperimentRunner`` (docs/experiment.md §8.0). Phase 11.

One runner, config-only differences: every pipeline — the baseline included — traverses the
IDENTICAL ``run_one`` code path; only the :class:`~benchmark.config.PipelineCfg` differs (the DRY
guarantee, §8.0). The runner is a flat loop over the explicit config pipelines (baseline first) —
there is NO matrix expansion, NO sweep, and NO selection phase.

Setup prelude (§8.0), before any per-pipeline work: load the dataset, build the index (register
embedder endpoints → ensure_index → bulk_index via :class:`ESIndexer`), build the searcher factory,
freeze the shared query set, and build the :class:`QrelIndex` + :class:`Evaluator` once. The
index-build sequence lives in ONE place — :meth:`ExperimentRunner.build_index` — reused by both
:meth:`ExperimentRunner.run`'s setup and the ``eval:index`` entry point (``scripts/index.py``), so
there is a single register/ensure/bulk path (DRY).

Imports the pure consumers (``config``/``metrics``/``stats``/``io_csv``) + the ES ``ESIndexer``
adapter (the concrete indexer §8.0 names as ``Indexer().build(...)``). The lazy ``config`` factories
(``load_dataset``/``make_indexer``/``make_searcher_factory``) are referenced through the ``config``
module so tests can monkeypatch them with in-memory fakes (no ES, no network).
"""

from __future__ import annotations

from typing import Any

import benchmark.config as config
from benchmark.backends.elasticsearch import ESIndexer
from benchmark.config import PipelineCfg, ResolvedConfig
from benchmark.io_csv import (
    DEFAULT_OUTPUT_DIR,
    write_comparison_csv,
    write_metrics_csv,
    write_result_csv,
    write_run_config,
)
from benchmark.logging_setup import get_logger
from benchmark.metrics import Evaluator, Metrics, QrelIndex
from benchmark.models import RankedResult
from benchmark.stats import Comparator

logger = get_logger(__name__)


class ExperimentRunner:
    """Runs the whole benchmark for a resolved config, the single execution path (§8.0)."""

    def build_index(self, cfg: ResolvedConfig) -> tuple[Any, Any, Any]:
        """Build the index (§3.5 register→ensure_index→bulk_index); return (dataset, backend, mapping).

        The ONE index-build path (DRY): reused by :meth:`run`'s setup prelude AND the ``eval:index``
        entry point. Loads the dataset, builds the ingest backend, and drives ``ESIndexer.build``
        over EVERY configured embedder (§8.0 passes ``cfg.services.embedders.values()`` — every
        embedder gets a ``semantic_text`` field). Returns the dataset (reused by :meth:`run` for the
        shared query set + qrels), the backend (so the caller can register a reranker endpoint at
        R0), and the resulting :class:`IndexMapping`.
        """
        dataset = config.load_dataset(cfg.dataset)
        backend = config.make_indexer(cfg.indexer)  # ingest seam (register/ensure_index/bulk_index)
        embedders = list(cfg.services.embedders.values())
        logger.info(
            "building index %r over %d embedder(s): %s",
            cfg.indexer.get("index"),
            len(embedders),
            [e.name for e in embedders],
        )
        mapping = ESIndexer().build(dataset, backend, embedders)
        return dataset, backend, mapping

    def run(self, cfg: ResolvedConfig, *, output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
        """Run every pipeline baseline-first, then the family-wide comparator pass (§8.0)."""
        dataset, backend, mapping = self.build_index(cfg)
        factory = config.make_searcher_factory(cfg.indexer)  # ES: Lexical/Vector/ESReranker
        queries = list(dataset.queries())  # frozen, shared query set
        query_texts = [q.text for q in queries]
        qrel_index = QrelIndex(dataset.qrels())
        evaluator = Evaluator(qrel_index, cutoff=cfg.cutoff)

        # Per-pipeline per-query Metrics, keyed by pipeline id. The pipelines are exactly what the
        # config declares — baseline first, then the named variants in config order (§10). No
        # expansion, no sweep, no selection phase.
        per_query: dict[str, dict[str, Metrics]] = {}

        def run_one(pcfg: PipelineCfg) -> None:
            # R0 — lazy reranker endpoint registration + the W <= top_n cap (§5.3/§8.0).
            if pcfg.reranker is not None:
                endpoint = cfg.services.reranker(pcfg.reranker).as_endpoint()
                backend.register_inference(endpoint)  # emits service_settings + task_settings
                top_n = endpoint.task_settings["top_n"]
                if pcfg.rerank_window_size is None or pcfg.rerank_window_size > top_n:
                    raise ValueError(
                        f"pipeline {pcfg.id!r}: rerank_window_size {pcfg.rerank_window_size} "
                        f"exceeds reranker {pcfg.reranker!r} top_n {top_n} (§5.3 W <= top_n)"
                    )

            pipeline = config.build_pipeline(pcfg, cfg.services, mapping, factory)
            # ONE bulk_search over the frozen shared query set — retrieval leaves batch via _msearch
            # (§5.3) rather than one round trip per query; result[i] aligns to queries[i].
            ranked = pipeline.bulk_search(query_texts, top_k=cfg.top_k)
            results = [
                RankedResult(query.query_id, docs) for query, docs in zip(queries, ranked)
            ]
            write_result_csv(pcfg, results, cfg.timestamp, output_dir=output_dir)
            metrics = evaluator.score_run(results)  # per-query vectors
            write_metrics_csv(pcfg, metrics, cfg.timestamp, output_dir=output_dir)
            per_query[pcfg.id] = metrics
            logger.info("pipeline %r: scored %d queries", pcfg.id, len(metrics))

        for pcfg in cfg.pipelines():  # baseline first, then variants
            run_one(pcfg)

        # Comparator pass — ONE family-wide call so the FDR correction (§8.3) is applied across the
        # whole (variant × metric) family. Adapt each Metrics -> {metric: value} via as_dict(): the
        # baseline maps come from the baseline pipeline, the variant maps from every non-baseline
        # pipeline (the baseline is NEVER compared to itself, §8.0).
        baseline_maps = {
            query_id: m.as_dict() for query_id, m in per_query[cfg.baseline_id].items()
        }
        variant_maps = {
            vid: {query_id: m.as_dict() for query_id, m in metrics.items()}
            for vid, metrics in per_query.items()
            if vid != cfg.baseline_id
        }
        rows = Comparator(cfg.stats).compare(baseline_maps, variant_maps)  # family-wide FDR inside
        for vid in variant_maps:
            variant_rows = [row for row in rows if row.variant == vid]
            write_comparison_csv(
                cfg.baseline_id, vid, variant_rows, cfg.timestamp, output_dir=output_dir
            )
        write_run_config(cfg, output_dir=output_dir)
        logger.info(
            "run complete: %d pipeline(s), %d variant comparison(s) written to %r",
            len(per_query),
            len(variant_maps),
            output_dir,
        )
