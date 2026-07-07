"""The single execution path — ``ExperimentRunner`` (docs/experiment.md §8.0). Phase 11.

One runner, config-only differences: every pipeline — the baseline included — traverses the
IDENTICAL ``run_one`` code path; only the :class:`~benchmark.config.PipelineCfg` differs (the DRY
guarantee, §8.0). The runner is a flat loop over the explicit config pipelines (baseline first) —
there is NO matrix expansion, NO sweep, and NO selection phase.

Two entry points, two paths. ``eval:index`` (``scripts/index.py``) BUILDS the index via
:meth:`ExperimentRunner.build_index` (instantiate the embedder connectors → the domain
``indexing.Indexer`` discovers dims → ensure_index → embed the corpus → bulk_index, delegating
backend bits to the injected ``IndexWriter``). ``eval:run`` (:meth:`ExperimentRunner.run`) does NOT
index — its setup prelude VERIFIES a pre-built index (doc count == the dataset's, else
:class:`IndexNotReadyError`, §8.0), builds the full leaf ``Searcher`` / ``Reranker`` maps (wired to
the embedder/reranker connectors) over that index's field names, freezes the shared query set, and
builds the :class:`QrelIndex` + :class:`Evaluator` once.

Imports the pure consumers (``config``/``evaluation``/``io_csv``/``common``) + the domain
``indexing.Indexer`` — NEVER an adapter at import time (§11). The concrete ``IndexWriter`` + leaf
``Searcher`` / ``Reranker`` maps arrive through the lazy ``config.make_index_writer`` /
``config.make_searchers`` / ``config.make_rerankers_bound`` factories (all referenced through the
``config`` module so tests can monkeypatch them with in-memory fakes — no ES, no network). Because the
runner names no backend, swapping the backend is a config-only edit (the §1.4(3) Generality criterion).
"""

from __future__ import annotations

from typing import Any

import benchmark.config as config
from benchmark.common.logging_setup import get_logger
from benchmark.common.models import RankedResult
from benchmark.config import PipelineCfg, ResolvedConfig
from benchmark.evaluation.metrics import Evaluator, Metrics, QrelIndex
from benchmark.evaluation.stats import Comparator
from benchmark.indexing import Indexer
from benchmark.io_csv import (
    DEFAULT_OUTPUT_DIR,
    write_comparison_csv,
    write_metrics_csv,
    write_results_csv,
    write_run_config,
)

logger = get_logger(__name__)


class IndexNotReadyError(RuntimeError):
    """``eval:run`` was invoked but the index is missing or not fully built (§8.0).

    ``eval:run`` does NOT (re)index — it requires an index already populated by ``eval:index``. This
    is raised when the target index does not exist, or when its doc count does not equal the
    dataset's (a partial/stale index), so a run never silently scores against incomplete data.
    """


class ExperimentRunner:
    """Runs the whole benchmark for a resolved config, the single execution path (§8.0)."""

    def build_index(self, cfg: ResolvedConfig) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Build the index (§3.5 ensure_index→embed→bulk_index); return (dataset, writer, mapping, embedders).

        The index-build path driven by the ``eval:index`` entry point (``scripts/index.py``).
        ``eval:run`` does NOT call this — it verifies a pre-built index instead (see :meth:`run`).
        Loads the dataset, builds the ``IndexWriter`` (``config.make_index_writer``),
        instantiates every configured embedder connector (``config.make_embedders`` — §3.4), and
        drives the domain :class:`~benchmark.indexing.Indexer` over them (each embedder gets a
        ``dense_vector`` field; the harness embeds the corpus at ingest, §3.5). Returns the dataset
        (reused by :meth:`run` for the shared query set + qrels), the writer (its ``.client`` reused by
        ``eval:index`` for the doc-count check), the resulting :class:`IndexMapping`, and the embedder
        registry (reused by :meth:`run` for the vector searchers' query embedding).
        """
        dataset = config.load_dataset(cfg.dataset)
        writer = config.make_index_writer(cfg.indexer)  # IndexWriter (mapping/ensure_index/bulk_index)
        embedders = config.make_embedders(cfg.services)  # name -> Embedder connector (§3.4)
        logger.info(
            "building index %r over %d embedder(s): %s",
            cfg.indexer.get("index"),
            len(embedders),
            sorted(embedders),
        )
        mapping = Indexer(writer, list(embedders.values())).build(dataset)
        return dataset, writer, mapping, embedders

    def run(self, cfg: ResolvedConfig, *, output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
        """Run every pipeline baseline-first, then the family-wide comparator pass (§8.0).

        ``eval:run`` does NOT (re)index — it REQUIRES an index already built by ``eval:index``. Before
        any pipeline runs, it verifies the index exists and holds the whole corpus (its doc count
        equals the dataset's document count), raising :class:`IndexNotReadyError` otherwise so a
        missing/partial/stale index never silently skews the metrics.
        """
        dataset = config.load_dataset(cfg.dataset)
        writer = config.make_index_writer(cfg.indexer)
        embedders = config.make_embedders(cfg.services)  # name -> Embedder connector (§3.4)
        # Query-only mapping: field names for the leaf searchers; NO dim probe, NO (re)indexing.
        mapping = Indexer(writer, list(embedders.values())).mapping(dataset)

        # Require a fully-built index (built by eval:index). Compare the index doc count to the
        # dataset's (§8.0) — fail fast on a missing or partial index rather than re-indexing here.
        indexed = writer.doc_count()
        expected = sum(1 for _ in dataset.documents())
        if indexed is None:
            raise IndexNotReadyError(
                f"index {mapping.index_name!r} does not exist — build it first with `eval:index`"
            )
        if indexed != expected:
            raise IndexNotReadyError(
                f"index {mapping.index_name!r} has {indexed} docs but the dataset has {expected}; "
                "it is not fully indexed — (re)build it with `eval:index` before `eval:run`"
            )
        logger.info(
            "index %r ready: %d docs match the dataset; running eval (no re-indexing)",
            mapping.index_name, indexed,
        )

        rerank_clients = config.make_rerankers(cfg.services)  # name -> RerankClient connector (§3.4/§5.4)
        # Build the FULL configured leaf-searcher + reranker sets ONCE, shared across all pipelines
        # (each pipeline selects its leaves/reranker by name in build_pipeline). ES: Lexical/Vector/
        # ESReranker over one shared client, wired to the provider connectors.
        searchers = config.make_searchers(cfg.indexer, mapping, cfg.services, embedders=embedders)
        rerankers = config.make_rerankers_bound(
            cfg.indexer, mapping, cfg.services, rerank_clients=rerank_clients
        )
        queries = list(dataset.queries())  # frozen, shared query set
        query_texts = [q.text for q in queries]
        qrel_index = QrelIndex(dataset.qrels())
        evaluator = Evaluator(qrel_index, cutoff=cfg.cutoff)

        # Per-pipeline ranked results + per-query Metrics, keyed by pipeline id. The pipelines are
        # exactly what the config declares — baseline first, then the named variants in config order
        # (§10). No expansion, no sweep, no selection phase. Both dicts stay baseline-first (insertion
        # order) so the single result/metrics files list the baseline before the variants (§9).
        results_by_variant: dict[str, list[RankedResult]] = {}
        per_query: dict[str, dict[str, Metrics]] = {}

        def run_one(pcfg: PipelineCfg) -> None:
            # R0 — the W <= top_n cap (§5.4/§8.0). No endpoint registration: the reranker is a
            # provider connector (already built in `rerankers`); `top_n` is a plain settings key
            # capping how many candidates the provider scores per request.
            if pcfg.reranker is not None:
                top_n = cfg.services.reranker(pcfg.reranker).settings.get("top_n")
                if top_n is None:
                    raise ValueError(
                        f"pipeline {pcfg.id!r}: reranker {pcfg.reranker!r} has no settings.top_n "
                        "(required as the W <= top_n cap, §5.4)"
                    )
                if pcfg.rerank_window_size is None or pcfg.rerank_window_size > int(top_n):
                    raise ValueError(
                        f"pipeline {pcfg.id!r}: rerank_window_size {pcfg.rerank_window_size} "
                        f"exceeds reranker {pcfg.reranker!r} top_n {top_n} (§5.4 W <= top_n)"
                    )

            pipeline = config.build_pipeline(pcfg, searchers, rerankers)
            # ONE bulk_search over the frozen shared query set — retrieval leaves batch via _msearch
            # (§5.3) rather than one round trip per query; result[i] aligns to queries[i].
            ranked = pipeline.bulk_search(query_texts, top_k=cfg.top_k)
            results = [
                RankedResult(query.query_id, docs) for query, docs in zip(queries, ranked)
            ]
            results_by_variant[pcfg.id] = results
            metrics = evaluator.score_run(results)  # per-query vectors
            per_query[pcfg.id] = metrics
            logger.info("pipeline %r: scored %d queries", pcfg.id, len(metrics))

        for pcfg in cfg.pipelines():  # baseline first, then variants
            run_one(pcfg)

        # One result + one metrics file for the whole run (all pipelines, baseline-first, §9).
        write_results_csv(results_by_variant, cfg.timestamp, output_dir=output_dir)
        write_metrics_csv(per_query, cfg.timestamp, output_dir=output_dir)

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
        write_comparison_csv(cfg.baseline_id, rows, cfg.timestamp, output_dir=output_dir)
        write_run_config(cfg, output_dir=output_dir)
        logger.info(
            "run complete: %d pipeline(s), %d variant comparison(s) written to %r",
            len(per_query),
            len(variant_maps),
            output_dir,
        )
