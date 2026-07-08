"""build_pipeline object-graph tests (docs/architecture.md §4, §10, plan Phase 6).

Builds each explicit pipeline shape (single lexical/vector leaf; HybridSearch+RRFFuser; +reranker
with rerank_window_size) from pre-built fake ``{name: Searcher}`` / ``{name: Reranker}`` maps passed
straight to :func:`build_pipeline` (the ``_ESSearcherFactory`` seam is gone — the leaves are minted by
the ES adapter's ``build_searchers``/``build_rerankers`` and selected by name here). Asserts the
resulting ``SearchPipeline`` object graph. There is NO expansion/sweep — pipelines are explicit.
"""

from __future__ import annotations

from typing import Sequence

import pytest

from benchmark.common.models import ScoredDoc
from benchmark.common.protocols import Reranker, Searcher
from benchmark.config import FuserCfg, PipelineCfg, build_pipeline
from benchmark.search import HybridSearch, SearchPipeline

WINDOW = 100


class _FakeSearcher(Searcher):
    def __init__(self, kind: str, target: object, embedder_id: str | None = None) -> None:
        self.kind = kind
        self.target = target
        self.embedder_id = embedder_id

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        return []


class _FakeReranker(Reranker):
    def __init__(self, name: str, field: str) -> None:
        self.name = name
        self.field = field

    def rerank(self, query: str, candidates: Sequence[ScoredDoc]) -> list[ScoredDoc]:
        return list(candidates)


# Pre-built leaf maps keyed by service name (what build_searchers/build_rerankers would mint). The
# leaf attributes below mirror what the ES builders attach (lexical field, vector sem-field + query
# embedder id, reranker field/name) so the graph-shape assertions are identical to before.
SEARCHERS: dict[str, Searcher] = {
    "bm25": _FakeSearcher("lexical", ["search_text"]),
    "semantic_e5": _FakeSearcher("vector", "sem__e5", "e5"),
}
RERANKERS: dict[str, Reranker] = {
    "co-rr": _FakeReranker("co-rr", "search_text"),
}


def _build(pcfg: PipelineCfg) -> SearchPipeline:
    return build_pipeline(pcfg, SEARCHERS, RERANKERS)


def _pcfg(**kw: object) -> PipelineCfg:
    base: dict[str, object] = {
        "retrievers": ("bm25",),
        "fuser": None,
        "reranker": None,
        "rerank_window_size": None,
    }
    base.update(kw)
    return PipelineCfg(id="p", **base)  # type: ignore[arg-type]


def test_single_lexical_leaf() -> None:
    pipeline = _build(_pcfg(retrievers=("bm25",)))
    assert isinstance(pipeline, SearchPipeline)
    assert isinstance(pipeline.retriever, _FakeSearcher)
    assert pipeline.retriever.kind == "lexical"
    assert pipeline.retriever.target == ["search_text"]
    assert pipeline.reranker is None
    assert pipeline.rerank_window_size is None


def test_single_vector_leaf() -> None:
    pipeline = _build(_pcfg(retrievers=("semantic_e5",)))
    assert isinstance(pipeline.retriever, _FakeSearcher)
    assert pipeline.retriever.kind == "vector"
    assert pipeline.retriever.target == "sem__e5"  # mapping.sem_field(embedder.name)
    assert pipeline.retriever.embedder_id == "e5"  # the query embedder to attach (§4)
    assert pipeline.reranker is None


def test_hybrid_is_hybridsearch_with_rrf() -> None:
    pipeline = _build(
        _pcfg(retrievers=("bm25", "semantic_e5"), fuser=FuserCfg("rrf", 60, WINDOW))
    )
    assert isinstance(pipeline.retriever, HybridSearch)
    assert len(pipeline.retriever.retrievers) == 2  # lexical + vector
    assert pipeline.retriever.fuser.rank_constant == 60  # concrete k from config
    assert pipeline.retriever.retrieval_window_size == WINDOW
    assert pipeline.reranker is None


def test_reranker_wraps_leaf_with_window() -> None:
    pipeline = _build(_pcfg(retrievers=("bm25",), reranker="co-rr", rerank_window_size=WINDOW))
    assert isinstance(pipeline.retriever, _FakeSearcher)
    assert pipeline.retriever.kind == "lexical"
    assert isinstance(pipeline.reranker, _FakeReranker)
    assert pipeline.reranker.field == "search_text"  # mapping.search_text_field
    assert pipeline.reranker.name == "co-rr"
    assert pipeline.rerank_window_size == WINDOW


def test_hybrid_rerank_has_hybrid_and_reranker() -> None:
    pipeline = _build(
        _pcfg(
            retrievers=("bm25", "semantic_e5"),
            fuser=FuserCfg("rrf", 60, WINDOW),
            reranker="co-rr",
            rerank_window_size=WINDOW,
        )
    )
    assert isinstance(pipeline.retriever, HybridSearch)
    assert isinstance(pipeline.reranker, _FakeReranker)
    assert pipeline.rerank_window_size == WINDOW


def test_multi_leaf_without_fuser_raises() -> None:
    with pytest.raises(ValueError):
        _build(_pcfg(retrievers=("bm25", "semantic_e5"), fuser=None))


def test_unknown_fuser_type_raises() -> None:
    with pytest.raises(ValueError):
        _build(_pcfg(retrievers=("bm25", "semantic_e5"), fuser=FuserCfg("magic", 60, WINDOW)))
