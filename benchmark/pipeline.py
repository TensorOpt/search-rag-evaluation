"""Backend-agnostic composers of the composite retrieval model (docs/experiment.md §3.6/§3.7).

Everything that produces a ranked list is a ``Searcher`` (§3.3). The three concrete composers
here wire leaf ``Searcher``s (the ES-specific ``LexicalSearcher``/``VectorSearch``, Phase 9/10)
into the six retrieval shapes as object graphs — no declarative spec layer, no per-variant branching:

- ``RRFFuser``     — a ``Fuser`` that reciprocal-rank-fuses result lists client-side (§3.7).
- ``HybridSearch`` — a ``Searcher`` that runs N retrievers and fuses their lists client-side.
- ``SearchPipeline`` — the top-level ``Searcher``: an optional rerank pass over a retriever.

Imports only ``benchmark.models`` / ``benchmark.protocols`` / ``benchmark.fusion`` + stdlib —
never adapters, ``matrix``, or numpy (§11). ``build_pipeline`` (``PipelineCfg`` -> ``SearchPipeline``
object graph) lives in ``matrix.py`` (Phase 6), not here, to avoid a pipeline->matrix forward
dependency (§4).
"""

from __future__ import annotations

from typing import Sequence

from benchmark.fusion import fuse_rrf_local
from benchmark.models import ScoredDoc
from benchmark.protocols import Fuser, Reranker, Searcher


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
