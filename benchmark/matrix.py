"""Deterministic matrix expansion, best_per_model selection, and ``spec_for`` (docs/experiment.md §4, §8.0a, §10). Phase 6.

``expand_matrix`` turns a :class:`ResolvedConfig` into the ordered list of :class:`VariantCfg`s
for one run (baseline ``bm25`` first, §10). ``resolve_hybrid_rerank_best_per_model`` runs only in
the opt-in two-pass mode (§8.0a): it picks, per embedding model, the ``hybrid`` ``rank_constant``
with the highest mean nDCG@10 and emits the deferred ``hybrid_rerank`` rows at that k. ``spec_for``
maps one ``VariantCfg`` to a ``SearchPipeline`` object graph (§4) via a ``SearcherFactory`` seam.

Imports only ``benchmark.models`` / ``benchmark.protocols`` / ``benchmark.pipeline`` + stdlib —
never adapters or numpy (§11). The ``matrix`` -> ``pipeline`` import is the one allowed backward
edge (§4): ``spec_for`` builds the composers, so it must know them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from benchmark.models import IndexMapping
from benchmark.pipeline import HybridSearch, RRFFuser, SearchPipeline
from benchmark.protocols import Searcher, SearcherFactory

if TYPE_CHECKING:  # stats.py is a pure sibling; the annotation avoids a runtime matrix->stats edge (§11).
    from benchmark.stats import StatsCfg

#: The selection metric for best_per_model (§8.0a): mean over queries of this per-query metric.
SELECTION_METRIC = "ndcg@10"


@dataclass(frozen=True)
class EmbeddingModelCfg:
    """One resolved ``embedding_models`` entry (§10) — its logical id joins the semantic axis."""

    inference_id: str


@dataclass(frozen=True)
class RerankerCfg:
    """One resolved ``rerankers`` entry (§10) — its logical id joins the ``*_rerank`` axis."""

    inference_id: str


@dataclass(frozen=True)
class VariantCfg:
    """A fully-expanded pipeline configuration — one row of the experiment matrix (§4, §10).

    ``family`` is the base variant name (``bm25``/``semantic``/``hybrid``/``bm25_rerank``/
    ``semantic_rerank``/``hybrid_rerank``); ``id`` is the §9 expanded id
    (e.g. ``hybrid__e5-small__k60``). The behavioral flags are what ``spec_for`` reads: the
    ``bm25`` baseline has ``use_bm25=True`` and everything else off. ``rrf_k`` is a concrete int
    only when ``fuse`` is set (``spec_for`` never selects it); ``reranker_id`` set means a rerank
    pass at ``window``.
    """

    id: str
    family: str
    use_bm25: bool
    embedding_model_id: str | None
    fuse: bool
    rrf_k: int | None
    window: int
    reranker_id: str | None


@dataclass(frozen=True)
class ResolvedConfig:
    """The fully-resolved run configuration (§10 axes + §9.1 run metadata).

    ``dataset``/``backend`` are the raw resolved config sections (``config.py`` dispatches the
    live adapter from ``dataset["name"]`` / ``backend["kind"]``, deferred to Phase 11).
    ``hybrid_rerank_k`` is either a concrete int (static expansion) or the literal
    ``"best_per_model"`` (two-pass, §8.0a). ``top_k`` is the retrieval depth; ``rank_window_size``
    is the fusion/rerank candidate window W; ``hybrid_rerank_k`` mirrors §10.
    """

    dataset: Mapping[str, object]
    backend: Mapping[str, object]
    embedding_models: Sequence[EmbeddingModelCfg]
    rerankers: Sequence[RerankerCfg]
    rrf_k_sweep: Sequence[int]
    variants: Sequence[str]
    reranker_endpoints: Mapping[str, Any]  # inference_id -> InferenceEndpoint (for R0, §8.0)
    stats: "StatsCfg"
    cutoff: int
    top_k: int
    rank_window_size: int
    hybrid_rerank_k: int | str
    baseline_id: str
    timestamp: str
    seed: int


def _bm25(window: int) -> VariantCfg:
    return VariantCfg(
        id="bm25",
        family="bm25",
        use_bm25=True,
        embedding_model_id=None,
        fuse=False,
        rrf_k=None,
        window=window,
        reranker_id=None,
    )


def _semantic(model_id: str, window: int) -> VariantCfg:
    return VariantCfg(
        id=f"semantic__{model_id}",
        family="semantic",
        use_bm25=False,
        embedding_model_id=model_id,
        fuse=False,
        rrf_k=None,
        window=window,
        reranker_id=None,
    )


def _hybrid(model_id: str, rrf_k: int, window: int) -> VariantCfg:
    return VariantCfg(
        id=f"hybrid__{model_id}__k{rrf_k}",
        family="hybrid",
        use_bm25=True,
        embedding_model_id=model_id,
        fuse=True,
        rrf_k=rrf_k,
        window=window,
        reranker_id=None,
    )


def _bm25_rerank(reranker_id: str, window: int) -> VariantCfg:
    return VariantCfg(
        id=f"bm25_rerank__{reranker_id}",
        family="bm25_rerank",
        use_bm25=True,
        embedding_model_id=None,
        fuse=False,
        rrf_k=None,
        window=window,
        reranker_id=reranker_id,
    )


def _semantic_rerank(model_id: str, reranker_id: str, window: int) -> VariantCfg:
    return VariantCfg(
        id=f"semantic_rerank__{model_id}__{reranker_id}",
        family="semantic_rerank",
        use_bm25=False,
        embedding_model_id=model_id,
        fuse=False,
        rrf_k=None,
        window=window,
        reranker_id=reranker_id,
    )


def _hybrid_rerank(model_id: str, reranker_id: str, rrf_k: int, window: int) -> VariantCfg:
    return VariantCfg(
        id=f"hybrid_rerank__{model_id}__{reranker_id}__k{rrf_k}",
        family="hybrid_rerank",
        use_bm25=True,
        embedding_model_id=model_id,
        fuse=True,
        rrf_k=rrf_k,
        window=window,
        reranker_id=reranker_id,
    )


def expand_matrix(cfg: ResolvedConfig) -> list[VariantCfg]:
    """Deterministically expand ``cfg`` into the ordered variant list, ``bm25`` first (§10).

    Order: ``bm25``(1) -> ``semantic``(per model) -> ``hybrid``(models x k-sweep) ->
    ``bm25_rerank``(per reranker) -> ``semantic_rerank``(models x rerankers) -> ``hybrid_rerank``.
    ``hybrid_rerank`` is emitted (models x rerankers at the fixed k) ONLY when ``hybrid_rerank_k``
    is an int; when it is ``"best_per_model"`` no ``hybrid_rerank`` rows are emitted here — they
    are deferred to :func:`resolve_hybrid_rerank_best_per_model` (§8.0a). Pure and deterministic.
    """
    window = cfg.rank_window_size
    model_ids = [m.inference_id for m in cfg.embedding_models]
    reranker_ids = [r.inference_id for r in cfg.rerankers]

    variants: list[VariantCfg] = [_bm25(window)]
    variants.extend(_semantic(m, window) for m in model_ids)
    variants.extend(
        _hybrid(m, k, window) for m in model_ids for k in cfg.rrf_k_sweep
    )
    variants.extend(_bm25_rerank(r, window) for r in reranker_ids)
    variants.extend(
        _semantic_rerank(m, r, window) for m in model_ids for r in reranker_ids
    )

    hybrid_rerank_k = cfg.hybrid_rerank_k
    if isinstance(hybrid_rerank_k, bool):  # bool is an int subclass; a mode flag is never a k
        raise ValueError(f"hybrid_rerank_k must be an int or 'best_per_model', got {hybrid_rerank_k!r}")
    if isinstance(hybrid_rerank_k, int):
        variants.extend(
            _hybrid_rerank(m, r, hybrid_rerank_k, window)
            for m in model_ids
            for r in reranker_ids
        )
    elif hybrid_rerank_k == "best_per_model":
        pass  # deferred to the §8.0a selection phase — no static rows
    else:
        raise ValueError(
            f"hybrid_rerank_k must be an int or 'best_per_model', got {hybrid_rerank_k!r}"
        )

    return variants


def _mean_selection_metric(
    per_query: Mapping[str, Mapping[str, Mapping[str, float]]], variant_id: str
) -> float:
    """Mean of the selection metric over that variant's non-NaN queries; NaN if none (§8.0a)."""
    values = [
        metrics[SELECTION_METRIC]
        for metrics in per_query[variant_id].values()
        if not math.isnan(metrics[SELECTION_METRIC])
    ]
    if not values:
        return math.nan
    return sum(values) / len(values)


def resolve_hybrid_rerank_best_per_model(
    cfg: ResolvedConfig,
    per_query: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> list[VariantCfg]:
    """Emit the deferred ``hybrid_rerank`` rows at the per-model best k (§8.0a).

    ``per_query`` maps ``variant_id -> query_id -> {metric_name: value}`` — the in-memory
    ``Metrics.as_dict()`` output the runner already holds, so this stays pure (no metrics.py / no
    adapter import). For each embedding model, pick the ``rank_constant`` k whose ``hybrid__m__k*``
    row has the highest MEAN nDCG@10; tie-break on the SMALLEST k, then the lexicographically
    smallest variant id. Deterministic and seed-independent (§1.4(2)). Emits one ``hybrid_rerank``
    row per ``(model, reranker)`` at the chosen k. Never re-reads CSV.
    """
    window = cfg.rank_window_size
    reranker_ids = [r.inference_id for r in cfg.rerankers]
    sweep = list(cfg.rrf_k_sweep)

    rows: list[VariantCfg] = []
    for model in cfg.embedding_models:
        model_id = model.inference_id
        # Candidate hybrid rows for this model, in a fixed order so the tie-break is deterministic.
        candidates = [
            candidate
            for k in sweep
            if (candidate := _hybrid(model_id, k, window)).id in per_query
        ]
        if not candidates:
            raise ValueError(
                f"no hybrid metrics for embedding model {model_id!r}; "
                "best_per_model selection needs the hybrid rows scored first (§8.0a)"
            )

        # argmax mean nDCG@10; tie-break smallest k, then lexicographically smallest id.
        # min over (-score, k, id): highest score wins, then smallest k, then smallest id.
        # NaN sorts last (its -score is +inf).
        def sort_key(variant: VariantCfg) -> tuple[float, int, str]:
            mean_ndcg = _mean_selection_metric(per_query, variant.id)
            neg_score = math.inf if math.isnan(mean_ndcg) else -mean_ndcg
            assert variant.rrf_k is not None  # hybrid rows always carry a concrete k
            return (neg_score, variant.rrf_k, variant.id)

        best = min(candidates, key=sort_key)
        chosen_k = best.rrf_k
        assert chosen_k is not None
        rows.extend(_hybrid_rerank(model_id, r, chosen_k, window) for r in reranker_ids)

    return rows


def spec_for(
    variant: VariantCfg, mapping: IndexMapping, factory: SearcherFactory
) -> SearchPipeline:
    """Build a ``VariantCfg``'s ``SearchPipeline`` object graph via the ``SearcherFactory`` (§4).

    ``use_bm25`` -> a lexical leaf on ``mapping.search_text_field``; ``embedding_model_id`` -> a
    vector leaf on ``mapping.sem_field(id)``. With ``fuse`` the leaves are wrapped in a
    ``HybridSearch`` (RRF at the concrete ``rrf_k``, window ``variant.window``); otherwise exactly
    one leaf is expected (else ``ValueError``). ``reranker_id`` wraps the retriever in a
    ``SearchPipeline`` with a reranker at ``rerank_window_size=variant.window``; else a bare
    pass-through pipeline. NEVER performs k-selection — ``variant.rrf_k`` is already concrete (§4).

    The reranker's doc-text field is ``mapping.search_text_field``: ``IndexMapping`` carries only
    that canonical text field, and §5.3 fixes the ES rerank field to that same ``search_text``.
    """
    retrievers: list[Searcher] = []
    if variant.use_bm25:
        retrievers.append(factory.lexical(fields=[mapping.search_text_field]))
    if variant.embedding_model_id is not None:
        retrievers.append(
            factory.vector(field=mapping.sem_field(variant.embedding_model_id))
        )

    if variant.fuse:
        if variant.rrf_k is None:
            raise ValueError(f"variant {variant.id!r} sets fuse but has no rrf_k")
        retriever: Searcher = HybridSearch(
            retrievers=retrievers,
            fuser=RRFFuser(rank_constant=variant.rrf_k),
            retrieval_window_size=variant.window,
        )
    else:
        if len(retrievers) != 1:
            raise ValueError(
                f"variant {variant.id!r} does not fuse but built {len(retrievers)} leaf "
                "retrievers; expected exactly one"
            )
        (retriever,) = retrievers

    if variant.reranker_id is not None:
        return SearchPipeline(
            retriever=retriever,
            reranker=factory.reranker(variant.reranker_id, mapping.search_text_field),
            rerank_window_size=variant.window,
        )
    return SearchPipeline(retriever=retriever)
