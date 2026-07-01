"""Resolved config dataclasses + pipeline assembly: ``build_pipeline`` (docs/experiment.md §4, §10). Phase 6.

Historical name (this was the matrix-expansion module); with the explicit-config refactor there is
NO matrix expansion and NO sweep. Pipelines are named and explicit in the config (§10): the config
declares one ``baseline`` pipeline plus a map of named ``variants``, and each is resolved into a
:class:`PipelineCfg`. This module holds the pure resolved-config value types — the :class:`Services`
registry (embedders/rerankers/searchers by name, typed), :class:`PipelineCfg`, and
:class:`ResolvedConfig` — and :func:`build_pipeline`, which assembles one ``PipelineCfg`` into a
``SearchPipeline`` object graph (§4) via a ``SearcherFactory`` seam, reading the concrete field names
from ``IndexMapping``. ``config.py`` parses the YAML into these types.

Imports only ``benchmark.models`` / ``benchmark.protocols`` / ``benchmark.pipeline`` + stdlib —
never adapters or numpy (§11). The ``matrix`` -> ``pipeline`` import is the one allowed backward
edge (§4): ``build_pipeline`` builds the composers, so it must know them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from benchmark.models import IndexMapping, InferenceEndpoint, InferenceTaskType
from benchmark.pipeline import HybridSearch, RRFFuser, SearchPipeline
from benchmark.protocols import Searcher, SearcherFactory

if TYPE_CHECKING:  # stats.py is a pure sibling; the annotation avoids a runtime matrix->stats edge (§11).
    from benchmark.stats import StatsCfg


@dataclass(frozen=True)
class EmbedderCfg:
    """A named embedder service (§10 ``services`` block).

    ``name`` is the config-local reference; ``provider`` selects the backend adapter; ``settings``
    carries provider-specific knobs (``model_id``, ``api_key`` …). It flattens to an embedding
    ``InferenceEndpoint`` (``inference_id = name``) the indexer registers at ingest (§3.4/§3.5).
    """

    name: str
    provider: str
    task_type: InferenceTaskType
    settings: Mapping[str, Any]

    def as_endpoint(self) -> InferenceEndpoint:
        return InferenceEndpoint(
            inference_id=self.name,
            task_type=self.task_type,
            service=self.provider,
            service_settings=dict(self.settings),
        )


@dataclass(frozen=True)
class RerankerCfg:
    """A named reranker service (§10 ``services`` block).

    Flattens to a ``rerank`` ``InferenceEndpoint`` (``top_n`` is a ``task_settings`` key, §3.4/§5.3)
    the runner registers lazily at R0 and hands to the ES ``searcher_factory`` to build an
    ``ESReranker`` (``Reranker`` is a behavioral ABC realized by the backend, not a descriptor).
    """

    name: str
    provider: str
    settings: Mapping[str, Any]

    def as_endpoint(self) -> InferenceEndpoint:
        settings = dict(self.settings)
        top_n = settings.pop("top_n", None)
        return InferenceEndpoint(
            inference_id=self.name,
            task_type=InferenceTaskType.RERANK,
            service=self.provider,
            service_settings=settings,
            task_settings={} if top_n is None else {"top_n": top_n},
        )


@dataclass(frozen=True)
class SearcherCfg:
    """A named searcher service (§10 ``services`` block).

    ``kind`` is ``"lexical"`` or ``"vector"``; a vector searcher references an ``embedder`` service
    name (``None`` for lexical). ``build_pipeline`` resolves these into leaf ``Searcher``s.
    """

    name: str
    provider: str
    kind: str
    embedder: str | None


@dataclass(frozen=True)
class Services:
    """The typed registry of named services (§10 ``services`` block).

    Embedders/rerankers/searchers keyed by their config name. The accessors raise a clear
    ``KeyError``-style ``ValueError`` on a missing/mistyped reference so ``build_pipeline`` and
    config validation fail loudly rather than silently.
    """

    embedders: Mapping[str, EmbedderCfg]
    rerankers: Mapping[str, RerankerCfg]
    searchers: Mapping[str, SearcherCfg]

    def embedder(self, name: str) -> EmbedderCfg:
        if name not in self.embedders:
            raise ValueError(f"unknown embedder service {name!r}; known: {sorted(self.embedders)}")
        return self.embedders[name]

    def reranker(self, name: str) -> RerankerCfg:
        if name not in self.rerankers:
            raise ValueError(f"unknown reranker service {name!r}; known: {sorted(self.rerankers)}")
        return self.rerankers[name]

    def searcher(self, name: str) -> SearcherCfg:
        if name not in self.searchers:
            raise ValueError(f"unknown searcher service {name!r}; known: {sorted(self.searchers)}")
        return self.searchers[name]


@dataclass(frozen=True)
class FuserCfg:
    """RRF fusion parameters for a multi-retriever pipeline (§10 ``fuser`` block).

    Only ``rrf`` fusion exists today; ``build_pipeline`` raises on any other ``type`` (exhaustive).
    ``rank_constant`` is the RRF k; ``window`` is the retrieval/fusion candidate depth W.
    """

    type: str
    rank_constant: int
    window: int


@dataclass(frozen=True)
class PipelineCfg:
    """One explicit, named pipeline from the config (§4, §10).

    ``id`` is the pipeline's config name (the map key; the baseline's is ``"baseline"`` by default).
    ``retrievers`` is a tuple of searcher service names (exactly one leaf when ``fuser`` is None; 2+
    when fusing). ``fuser`` is present iff there are multiple retrievers. ``reranker`` is a reranker
    service name (paired with ``rerank_window_size``); both are set together or both None.
    """

    id: str
    retrievers: tuple[str, ...]
    fuser: FuserCfg | None
    reranker: str | None
    rerank_window_size: int | None


@dataclass(frozen=True)
class ResolvedConfig:
    """The fully-resolved run configuration (§10, §9.1 run metadata).

    ``dataset``/``indexer`` are the raw resolved config sections (``config.py`` dispatches the live
    adapter from ``dataset["name"]`` / ``indexer["provider"]``, deferred to Phase 11). ``services``
    is the typed registry. ``baseline`` is the reference pipeline; ``variants`` is the ORDERED list
    of explicit variant pipelines (iterate ``baseline`` first, then ``variants``, via
    :meth:`pipelines`). ``cutoff`` is the metric depth k; ``top_k`` is the retrieval depth.
    """

    dataset: Mapping[str, object]
    indexer: Mapping[str, object]
    services: Services
    baseline: PipelineCfg
    variants: Sequence[PipelineCfg]
    stats: "StatsCfg"
    cutoff: int
    top_k: int
    baseline_id: str
    timestamp: str
    seed: int

    def pipelines(self) -> list[PipelineCfg]:
        """The run's pipelines, baseline first (§8.0)."""
        return [self.baseline, *self.variants]


def build_pipeline(
    pcfg: PipelineCfg,
    services: Services,
    mapping: IndexMapping,
    factory: SearcherFactory,
) -> SearchPipeline:
    """Assemble ``pcfg``'s ``SearchPipeline`` object graph via the ``SearcherFactory`` (§4).

    Each retriever name resolves to its :class:`SearcherCfg` in ``services``; a lexical searcher
    becomes ``factory.lexical(fields=[mapping.search_text_field])`` and a vector searcher becomes
    ``factory.vector(field=mapping.sem_field(embedder_name))``. With a ``fuser`` the leaves are
    wrapped in a ``HybridSearch`` (RRF at ``fuser.rank_constant``, window ``fuser.window``); without
    one, exactly one leaf is expected (else ``ValueError``). A ``reranker`` wraps the retriever in a
    ``SearchPipeline`` with a rerank pass at ``rerank_window_size``; else a bare pass-through pipeline.

    The reranker's doc-text field is ``mapping.search_text_field`` — ``IndexMapping`` carries only
    that canonical text field, and §5.3 fixes the ES rerank field to that same ``search_text``.
    """
    leaves: list[Searcher] = [
        _build_leaf(services.searcher(name), services, mapping, factory)
        for name in pcfg.retrievers
    ]

    if pcfg.fuser is not None:
        if pcfg.fuser.type != "rrf":
            raise ValueError(
                f"pipeline {pcfg.id!r}: unknown fuser type {pcfg.fuser.type!r}; only 'rrf' is supported"
            )
        retriever: Searcher = HybridSearch(
            retrievers=leaves,
            fuser=RRFFuser(rank_constant=pcfg.fuser.rank_constant),
            retrieval_window_size=pcfg.fuser.window,
        )
    else:
        if len(leaves) != 1:
            raise ValueError(
                f"pipeline {pcfg.id!r} has no fuser but built {len(leaves)} leaf retrievers; "
                "expected exactly one"
            )
        (retriever,) = leaves

    if pcfg.reranker is not None:
        return SearchPipeline(
            retriever=retriever,
            reranker=factory.reranker(pcfg.reranker, mapping.search_text_field),
            rerank_window_size=pcfg.rerank_window_size,
        )
    return SearchPipeline(retriever=retriever)


def _build_leaf(
    searcher: SearcherCfg,
    services: Services,
    mapping: IndexMapping,
    factory: SearcherFactory,
) -> Searcher:
    """Resolve one searcher service to a leaf ``Searcher`` (§4). Exhaustive on ``kind``."""
    if searcher.kind == "lexical":
        return factory.lexical(fields=[mapping.search_text_field])
    if searcher.kind == "vector":
        if searcher.embedder is None:
            raise ValueError(f"vector searcher {searcher.name!r} has no embedder reference")
        embedder: EmbedderCfg = services.embedder(searcher.embedder)
        return factory.vector(field=mapping.sem_field(embedder.name))
    raise ValueError(
        f"searcher {searcher.name!r} has unknown kind {searcher.kind!r}; expected 'lexical' or 'vector'"
    )
