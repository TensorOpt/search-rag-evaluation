"""Backend-agnostic composers of the composite retrieval model (docs/architecture.md §3.6/§3.7).

Everything that produces a ranked list is a ``Searcher`` (§3.3). The three concrete composers
here wire leaf ``Searcher``s (the ES-specific ``LexicalSearcher``/``VectorSearch``, Phase 9/10)
into the six retrieval shapes as object graphs — no declarative spec layer, no per-variant branching:

- ``RRFFuser``     — a ``Fuser`` that reciprocal-rank-fuses result lists client-side (§3.7).
- ``HybridSearch`` — a ``Searcher`` that runs N retrievers and fuses their lists client-side.
- ``SearchPipeline`` — the top-level ``Searcher``: an optional rerank pass over a retriever.

Imports only ``benchmark.common.models`` / ``benchmark.common.protocols`` / ``benchmark.common.ranking``
+ stdlib — never adapters, ``config``, or numpy (§11). ``build_pipeline`` (``PipelineCfg`` ->
``SearchPipeline`` object graph) lives in ``config.py``, not here, to avoid a search->config forward
dependency (§4); ``config`` importing ``search`` for ``build_pipeline`` is the one-way wiring edge.
"""

from __future__ import annotations

from typing import Sequence

from benchmark.common.models import ScoredDoc
from benchmark.common.protocols import Fuser, Reranker, Searcher
from benchmark.common.ranking import fuse_rrf_local


class RRFFuser(Fuser):
    """Client-side reciprocal-rank fusion (§3.7): wraps ``fuse_rrf_local``."""

    def __init__(self, *, rank_constant: int) -> None:
        self.rank_constant = rank_constant

    def fuse(
        self, result_lists: Sequence[Sequence[ScoredDoc]], *, rank_window_size: int
    ) -> list[ScoredDoc]:
        return fuse_rrf_local(
            result_lists,
            rank_constant=self.rank_constant,
            rank_window_size=rank_window_size,
        )


class HybridSearch(Searcher):
    """Runs several retrievers and fuses their result lists client-side (§3.6).

    Each retriever is queried at ``retrieval_window_size`` (the fusion candidate depth), the
    lists are fused, and the fused list is truncated to ``top_k`` by ``search``.
    """

    def __init__(
        self,
        *,
        retrievers: Sequence[Searcher],
        fuser: Fuser,
        retrieval_window_size: int,
    ) -> None:
        self.retrievers = retrievers
        self.fuser = fuser
        self.retrieval_window_size = retrieval_window_size

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        result_lists = [
            retriever.search(query, top_k=self.retrieval_window_size)
            for retriever in self.retrievers
        ]
        fused = self.fuser.fuse(result_lists, rank_window_size=self.retrieval_window_size)
        return fused[:top_k]

    def bulk_search(self, queries: Sequence[str], *, top_k: int) -> list[list[ScoredDoc]]:
        """Batch each retriever's whole query set, then fuse+truncate per query (aligned, §3.6).

        Calls ``retriever.bulk_search`` ONCE per retriever (so ES leaves batch via ``_msearch``
        instead of one round trip per query), then for query ``i`` fuses that query's list from each
        retriever and truncates to ``top_k``. Results are aligned to ``queries`` by index.
        """
        per_retriever = [
            retriever.bulk_search(queries, top_k=self.retrieval_window_size)
            for retriever in self.retrievers
        ]
        results: list[list[ScoredDoc]] = []
        for query_index in range(len(queries)):
            result_lists = [lists[query_index] for lists in per_retriever]
            fused = self.fuser.fuse(result_lists, rank_window_size=self.retrieval_window_size)
            results.append(fused[:top_k])
        return results


class SearchPipeline(Searcher):
    """The top-level variant graph: a retriever with an optional rerank pass (§3.6).

    With a reranker, retrieval fetches ``rerank_window_size`` candidates which the reranker
    rescores/reorders before truncation to ``top_k``. Without a reranker, it is a pass-through
    over the retriever. ``rerank_window_size`` is REQUIRED iff a reranker is present; supplying
    it without a reranker (or omitting it with one) is a misconfiguration (ValueError).
    """

    def __init__(
        self,
        *,
        retriever: Searcher,
        reranker: Reranker | None = None,
        rerank_window_size: int | None = None,
    ) -> None:
        if reranker is not None and rerank_window_size is None:
            raise ValueError("rerank_window_size is required when a reranker is set")
        if reranker is None and rerank_window_size is not None:
            raise ValueError("rerank_window_size must be None when no reranker is set")
        self.retriever = retriever
        self.reranker = reranker
        self.rerank_window_size = rerank_window_size

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        if self.reranker is None:
            return self.retriever.search(query, top_k=top_k)
        assert self.rerank_window_size is not None  # __init__ invariant
        candidates = self.retriever.search(query, top_k=self.rerank_window_size)
        return self.reranker.rerank(query, candidates)[:top_k]

    def bulk_search(self, queries: Sequence[str], *, top_k: int) -> list[list[ScoredDoc]]:
        """Batch retrieval over the whole query set; rerank per query (aligned, §3.6/§6).

        Retrieval is batched via ``retriever.bulk_search`` (so ES leaves batch via ``_msearch``).
        Reranking stays PER QUERY: the provider rerank call is per-query, so batching it is a future
        optimization (§5.4). Without a reranker this is a pass-through to ``retriever.bulk_search``.
        Results are aligned to ``queries`` by index.
        """
        if self.reranker is None:
            return self.retriever.bulk_search(queries, top_k=top_k)
        assert self.rerank_window_size is not None  # __init__ invariant
        candidates_all = self.retriever.bulk_search(queries, top_k=self.rerank_window_size)
        return [
            self.reranker.rerank(query, candidates)[:top_k]
            for query, candidates in zip(queries, candidates_all)
        ]
