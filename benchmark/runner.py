"""The single execution path — ``ExperimentRunner`` (docs/architecture.md §6). Phase 11.

One runner, config-only differences: every pipeline — the baseline included — traverses the
IDENTICAL ``run_one`` code path; only the :class:`~benchmark.config.PipelineCfg` differs (the DRY
guarantee, §6). The runner is a flat loop over the explicit config pipelines (baseline first) —
there is NO matrix expansion, NO sweep, and NO selection phase.

Two entry points, two paths. ``eval:index`` (``scripts/index.py``) BUILDS the index via
:meth:`ExperimentRunner.build_index` (instantiate the embedder connectors → the domain
``indexing.Indexer`` discovers dims → ensure_index → embed the corpus → bulk_index, delegating
backend bits to the injected ``IndexWriter``). ``eval:run`` (:meth:`ExperimentRunner.run`) does NOT
index — its setup prelude VERIFIES a pre-built index (doc count == the dataset's, else
:class:`IndexNotReadyError`, §6), builds the full leaf ``Searcher`` / ``Reranker`` maps (wired to
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

import statistics
from typing import Any, Sequence

import benchmark.config as config
from benchmark.common.logging_setup import get_logger
from benchmark.common.models import RankedResult
from benchmark.config import PipelineCfg, ResolvedConfig
from benchmark.evaluation.metrics import (
    RECALL_CUTOFFS,
    Evaluator,
    Metrics,
    QrelIndex,
    qrels_digest,
)
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


#: recall@k is flagged low-information when k / median(|R|) falls below this (P2-3). On WANDS
#: (median |R| ≈ 146) recall@10 (0.068) warns; recall@50 (0.34) / recall@100 (0.68) do not.
_RECALL_LOW_INFO_RATIO = 0.2


def _recall_information(relevant_counts: Sequence[int]) -> dict[str, float]:
    """``recall@k -> k/median(|R|)``, warning where it is below the low-information floor (P2-3).

    ``|R|`` = per-query relevant-set size under the resolved threshold; the median is over queries
    WITH relevant docs (``R > 0``). A cutoff whose ratio falls below :data:`_RECALL_LOW_INFO_RATIO`
    is logged as low-information (``recall@k`` is capped near ``k/|R|`` and cannot move). Returns the
    per-cutoff ratios for the manifest; empty when no query has a relevant doc.
    """
    positive = [r for r in relevant_counts if r > 0]
    if not positive:
        return {}
    median_r = statistics.median(positive)
    info: dict[str, float] = {}
    for k in RECALL_CUTOFFS:
        ratio = k / median_r
        info[f"recall@{k}"] = ratio
        if ratio < _RECALL_LOW_INFO_RATIO:
            logger.warning(
                "recall@%d is low-information on this dataset: k/median(|R|)=%.3f < %.2f "
                "(median |R|=%.1f) — it cannot move and should not be put in front of an audience",
                k, ratio, _RECALL_LOW_INFO_RATIO, median_r,
            )
    return info


def _reranker_only_window(a: PipelineCfg, b: PipelineCfg) -> int | None:
    """The reranked side's ``rerank_window_size`` iff ``(a, b)`` differ ONLY by a reranker (MF-1).

    Two systems differ only by a reranker when their retrieval graph is identical (same retrievers,
    same fuser) and exactly one of them applies a reranker. Returns that reranker's window ``W`` (the
    quantity the ``recall@k`` identification rule ``W == k`` tests), else ``None``. Structural, from
    config — the runner computes it so ``stats.py`` stays adapter/pipeline-free (§11 import rule).
    """
    if a.retrievers != b.retrievers or a.fuser != b.fuser:
        return None
    if a.reranker is not None and b.reranker is None:
        return a.rerank_window_size
    if b.reranker is not None and a.reranker is None:
        return b.rerank_window_size
    return None


def _structural_exclusions(cfg: ResolvedConfig) -> dict[tuple[str, str, str], str]:
    """``(a, b, "recall@k") -> reason`` for every reranker-only contrast where ``W == k`` (MF-1).

    ``recall@k`` is not identified for a contrast between two systems differing only by a reranker
    when ``rerank_window_size == k`` (top-k-set identity): reranking permutes the top-k set without
    changing its membership. At the shipped ``W = 100`` only ``recall@100`` is excluded;
    ``recall@10``/``recall@50`` ARE identified (reranking moves docs across those boundaries).
    """
    pipe_by_id = {pcfg.id: pcfg for pcfg in cfg.pipelines()}
    exclusions: dict[tuple[str, str, str], str] = {}
    for contrast in cfg.stats.contrasts:
        a = pipe_by_id.get(contrast.a)
        b = pipe_by_id.get(contrast.b)
        if a is None or b is None:
            continue
        window = _reranker_only_window(a, b)
        if window is None:
            continue
        for k in RECALL_CUTOFFS:
            if window == k:
                metric = f"recall@{k}"
                exclusions[(contrast.a, contrast.b, metric)] = (
                    f"{metric} not identified: {contrast.a} vs {contrast.b} differ only by a "
                    f"reranker with rerank_window_size == {k} (top-k set identity, MF-1)"
                )
    return exclusions


class IndexNotReadyError(RuntimeError):
    """``eval:run`` was invoked but the index is missing or not fully built (§6).

    ``eval:run`` does NOT (re)index — it requires an index already populated by ``eval:index``. This
    is raised when the target index does not exist, or when its doc count does not equal the
    dataset's (a partial/stale index), so a run never silently scores against incomplete data.
    """


class ExperimentRunner:
    """Runs the whole benchmark for a resolved config, the single execution path (§6)."""

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
        # Open the cache (None when disabled/corrupt, §5/§7) so re-indexing an unchanged corpus reuses
        # its document embeddings; close it in `finally`. build_index builds NO searchers, so only the
        # embedders are wrapped here (no index-fingerprint fetch).
        cache = config.open_cache(cfg.cache)
        try:
            embedders = config.make_embedders(cfg.services, cache=cache)  # name -> Embedder connector (§3.4)
            logger.info(
                "building index %r over %d embedder(s): %s",
                cfg.indexer.get("index"),
                len(embedders),
                sorted(embedders),
            )
            mapping = Indexer(writer, list(embedders.values())).build(dataset)
            return dataset, writer, mapping, embedders
        finally:
            if cache is not None:
                cache.close()

    def run(self, cfg: ResolvedConfig, *, output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
        """Run every pipeline baseline-first, then the family-wide comparator pass (§6).

        ``eval:run`` does NOT (re)index — it REQUIRES an index already built by ``eval:index``. Before
        any pipeline runs, it verifies the index exists and holds the whole corpus (its doc count
        equals the dataset's document count), raising :class:`IndexNotReadyError` otherwise so a
        missing/partial/stale index never silently skews the metrics.
        """
        dataset = config.load_dataset(cfg.dataset)
        writer = config.make_index_writer(cfg.indexer)
        # The composition layer owns the live cache resource: open it here (None when disabled or on
        # a corrupt DB, docs/architecture.md §5.5), thread it into the factories that wrap the
        # embedder/rerank connectors + searcher leaves, and CLOSE it in `finally`.
        cache = config.open_cache(cfg.cache)
        try:
            embedders = config.make_embedders(cfg.services, cache=cache)  # name -> Embedder connector (§3.4)
            # Query-only mapping: field names for the leaf searchers; NO dim probe, NO (re)indexing.
            mapping = Indexer(writer, list(embedders.values())).mapping(dataset)

            # Require a fully-built index (built by eval:index). Compare the index doc count to the
            # dataset's (§6) — fail fast on a missing or partial index rather than re-indexing here.
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
            # P1-2: the BM25 similarity + analysis chain, RESOLVED from the live index (never assumed).
            index_profile = writer.resolved_index_profile()

            rerank_clients = config.make_rerankers(cfg.services, cache=cache)  # name -> RerankClient (§3.4/§5.4)
            # Build the FULL configured leaf-searcher + reranker sets ONCE, shared across all pipelines
            # (each pipeline selects its leaves/reranker by name in build_pipeline). ES: Lexical/Vector/
            # ESReranker over one shared client, wired to the provider connectors.
            searchers = config.make_searchers(
                cfg.indexer, mapping, cfg.services, embedders=embedders, cache=cache
            )
            rerankers = config.make_rerankers_bound(
                cfg.indexer, mapping, cfg.services, rerank_clients=rerank_clients
            )
            queries = list(dataset.queries())  # frozen, shared query set
            query_texts = [q.text for q in queries]
            # ONE relevance policy over every metric (§7, P0-2): the configured threshold feeds BOTH
            # the QrelIndex (R / digest, N-3) AND the Evaluator; the unjudged policy feeds the
            # Evaluator. qrels are materialized once (reused for the P0-3 digest).
            threshold = cfg.metrics.relevance_threshold
            qrels_list = list(dataset.qrels())
            qrel_index = QrelIndex(qrels_list, relevance_threshold=threshold)
            evaluator = Evaluator(
                qrel_index,
                cutoff=cfg.cutoff,
                unjudged=cfg.metrics.unjudged,
                relevance_threshold=threshold,
            )

            # Per-pipeline ranked results + per-query Metrics, keyed by pipeline id. The pipelines are
            # exactly what the config declares — baseline first, then the named variants in config order
            # (§10). No expansion, no sweep, no selection phase. Both dicts stay baseline-first (insertion
            # order) so the single result/metrics files list the baseline before the variants (§9).
            results_by_variant: dict[str, list[RankedResult]] = {}
            per_query: dict[str, dict[str, Metrics]] = {}

            def run_one(pcfg: PipelineCfg) -> None:
                # R0 — the W <= top_n cap (§5.4/§6). No endpoint registration: the reranker is a
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

            # Comparator pass — ONE call over the config-declared contrasts. Every pipeline (baseline
            # included, no longer split out) becomes a system map via Metrics.as_dict(); the baseline
            # is just another system and "variant vs bm25" is one contrast among many (§8.1/§8.3).
            systems = {
                vid: {query_id: m.as_dict() for query_id, m in metrics.items()}
                for vid, metrics in per_query.items()
            }
            # Structural (config-derived) recall@k exclusions for reranker-only contrasts (MF-1);
            # emitted with a reason string, never a silent NaN, and never counted in the FDR family.
            structural_exclusions = _structural_exclusions(cfg)
            rows = Comparator(cfg.stats).compare(
                systems, cfg.stats.contrasts, structural_exclusions=structural_exclusions
            )  # family FDR inside
            write_comparison_csv(rows, cfg.timestamp, output_dir=output_dir)

            # Reproducibility diagnostics (§9.1, Fix 6): per-metric common-subset sizes (all rows of a
            # metric share n_common) and per-system retrieval-failure counts (queries with 0 results).
            n_queries = len(queries)
            common_subset = {
                row.metric: {"n_common": row.n_common, "n_excluded": n_queries - row.n_common}
                for row in rows
            }
            retrieval_failures = {
                vid: sum(1 for m in metrics.values() if m.n_results == 0)
                for vid, metrics in per_query.items()
            }
            # P0-3: the qrels provenance — digest (over the gain-mapped triples + threshold), the
            # human-readable gain mapping, the resolved threshold, and n_qrels. Two runs with
            # differing digests are not comparable (report guidance, §7 provenance note).
            dataset_diag = {
                "qrels_digest": qrels_digest(qrels_list, relevance_threshold=threshold),
                "relevance_threshold": threshold,
                "gain_mapping": dict(dataset.gain_mapping()),
                "n_qrels": len(qrels_list),
            }
            # P1-1: the FDR family — size m (number of in-family real tests), full membership, and
            # the structurally-excluded contrasts with their reason strings. Recorded so a reader can
            # audit the multiple-comparison regime (8 contrasts × 1 metric = 8 tests on WANDS).
            family_members = [
                {"a": row.system_a, "b": row.system_b, "metric": row.metric}
                for row in rows
                if row.in_family
            ]
            excluded = [
                {"a": row.system_a, "b": row.system_b, "metric": row.metric, "reason": row.note}
                for row in rows
                if (row.system_a, row.system_b, row.metric) in structural_exclusions
            ]
            stats_diag = {
                "family_size": len(family_members),
                "family_members": family_members,
                "excluded": excluded,
            }
            # P2-3: flag recall@k that is structurally uninformative on this dataset (k/median(|R|)
            # below the floor) and record the ratios. |R| is dataset-level (from qrels), one per query.
            recall_information = _recall_information(
                [qrel_index.relevant_count(q.query_id) for q in queries]
            )
            diagnostics = {
                "common_subset": common_subset,
                "retrieval_failures": retrieval_failures,
                "dataset": dataset_diag,
                "stats": stats_diag,
                "index": index_profile,
                "recall_information": recall_information,
            }
            write_run_config(cfg, diagnostics=diagnostics, output_dir=output_dir)
            logger.info(
                "run complete: %d pipeline(s), %d comparison(s) written to %r",
                len(per_query),
                len(rows),
                output_dir,
            )
        finally:
            if cache is not None:
                cache.close()
