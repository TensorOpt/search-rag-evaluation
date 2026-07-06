"""Composite-model composer tests (docs/experiment.md §3.6/§3.7, plan Phase 5).

Exercises the three backend-agnostic composers against ``FakeSearcher``/``FakeReranker``:
``RRFFuser`` fuses, ``HybridSearch`` retrieves-at-window then fuses then truncates,
``SearchPipeline`` reranks (or passes through) then truncates, and ``SearchPipeline.__init__``
rejects misconfiguration.
"""

from __future__ import annotations

import pytest

from benchmark.common.ranking import fuse_rrf_local
from benchmark.search import HybridSearch, RRFFuser, SearchPipeline
from tests.conftest import (
    _BM25_DOCS,
    _SEMANTIC_DOCS,
    FakeReranker,
    FakeSearcher,
    RecordingBulkReranker,
    RecordingBulkSearcher,
)

RANK_CONSTANT = 10
WINDOW = 50


# --- RRFFuser ---------------------------------------------------------------------------------


def test_rrf_fuser_equals_fuse_rrf_local() -> None:
    fuser = RRFFuser(rank_constant=RANK_CONSTANT)
    fused = fuser.fuse([_BM25_DOCS, _SEMANTIC_DOCS], rank_window_size=WINDOW)
    expected = fuse_rrf_local(
        [_BM25_DOCS, _SEMANTIC_DOCS], rank_constant=RANK_CONSTANT, rank_window_size=WINDOW
    )
    assert fused == expected


# --- HybridSearch -----------------------------------------------------------------------------


def test_hybrid_search_retrieves_at_window_fuses_and_truncates() -> None:
    bm25 = FakeSearcher(_BM25_DOCS)
    semantic = FakeSearcher(_SEMANTIC_DOCS)
    hybrid = HybridSearch(
        retrievers=[bm25, semantic],
        fuser=RRFFuser(rank_constant=RANK_CONSTANT),
        retrieval_window_size=WINDOW,
    )

    result = hybrid.search("sofa", top_k=2)

    # each retriever was queried at retrieval_window_size, not top_k
    assert bm25.top_k_calls == [WINDOW]
    assert semantic.top_k_calls == [WINDOW]

    expected = fuse_rrf_local(
        [_BM25_DOCS, _SEMANTIC_DOCS], rank_constant=RANK_CONSTANT, rank_window_size=WINDOW
    )
    assert result == expected[:2]


# --- SearchPipeline ---------------------------------------------------------------------------


def test_pipeline_without_reranker_is_pass_through() -> None:
    retriever = FakeSearcher(_BM25_DOCS)
    pipeline = SearchPipeline(retriever=retriever)

    assert pipeline.search("sofa", top_k=2) == _BM25_DOCS[:2]
    assert retriever.top_k_calls == [2]  # retrieved directly at top_k


def test_pipeline_with_reranker_retrieves_window_then_reranks_then_truncates() -> None:
    retriever = FakeSearcher(_BM25_DOCS)
    reranker = FakeReranker()
    pipeline = SearchPipeline(retriever=retriever, reranker=reranker, rerank_window_size=WINDOW)

    result = pipeline.search("sofa", top_k=2)

    assert retriever.top_k_calls == [WINDOW]  # retrieved at rerank_window_size
    assert reranker.rerank_calls == [("sofa", tuple(d.doc_id for d in _BM25_DOCS))]
    # FakeReranker reverses the candidates, then truncated to top_k
    assert result == list(reversed(_BM25_DOCS))[:2]


# --- SearchPipeline __init__ misconfig --------------------------------------------------------


def test_pipeline_reranker_without_window_raises() -> None:
    with pytest.raises(ValueError, match="rerank_window_size is required"):
        SearchPipeline(retriever=FakeSearcher(_BM25_DOCS), reranker=FakeReranker())


def test_pipeline_window_without_reranker_raises() -> None:
    with pytest.raises(ValueError, match="must be None"):
        SearchPipeline(retriever=FakeSearcher(_BM25_DOCS), rerank_window_size=WINDOW)


# --- bulk_search --------------------------------------------------------------------------------


def test_searcher_default_bulk_search_loops_search_aligned() -> None:
    # FakeSearcher implements only `search`; the Searcher ABC default bulk_search loops it,
    # returning one aligned result list per query.
    searcher = FakeSearcher(_BM25_DOCS)
    results = searcher.bulk_search(["a", "b", "c"], top_k=2)

    assert results == [_BM25_DOCS[:2], _BM25_DOCS[:2], _BM25_DOCS[:2]]
    assert searcher.top_k_calls == [2, 2, 2]  # one search per query, at top_k


def test_hybrid_bulk_search_calls_each_retriever_bulk_once_and_fuses_per_query() -> None:
    bm25 = RecordingBulkSearcher(_BM25_DOCS)
    semantic = RecordingBulkSearcher(_SEMANTIC_DOCS)
    hybrid = HybridSearch(
        retrievers=[bm25, semantic],
        fuser=RRFFuser(rank_constant=RANK_CONSTANT),
        retrieval_window_size=WINDOW,
    )
    queries = ["sofa", "table"]

    results = hybrid.bulk_search(queries, top_k=2)

    # each retriever.bulk_search called ONCE (batched), at retrieval_window_size — NOT per-query search
    assert bm25.bulk_calls == [(tuple(queries), WINDOW)]
    assert semantic.bulk_calls == [(tuple(queries), WINDOW)]
    assert bm25.search_calls == []
    assert semantic.search_calls == []

    # aligned + fused per query + truncated to top_k (same result the fuser gives on the two leaves)
    expected = fuse_rrf_local(
        [_BM25_DOCS, _SEMANTIC_DOCS], rank_constant=RANK_CONSTANT, rank_window_size=WINDOW
    )[:2]
    assert results == [expected, expected]


def test_pipeline_bulk_search_without_reranker_delegates_to_retriever_bulk() -> None:
    retriever = RecordingBulkSearcher(_BM25_DOCS)
    pipeline = SearchPipeline(retriever=retriever)

    results = pipeline.bulk_search(["a", "b"], top_k=2)

    assert retriever.bulk_calls == [(("a", "b"), 2)]  # delegated to bulk_search at top_k
    assert retriever.search_calls == []
    assert results == [_BM25_DOCS[:2], _BM25_DOCS[:2]]


def test_pipeline_bulk_search_with_reranker_batches_retrieval_reranks_per_query() -> None:
    retriever = RecordingBulkSearcher(_BM25_DOCS)
    reranker = RecordingBulkReranker()
    pipeline = SearchPipeline(retriever=retriever, reranker=reranker, rerank_window_size=WINDOW)
    queries = ["a", "b"]

    results = pipeline.bulk_search(queries, top_k=2)

    # retrieval batched ONCE at rerank_window_size; rerank stays per query
    assert retriever.bulk_calls == [(tuple(queries), WINDOW)]
    assert reranker.rerank_calls == [
        ("a", tuple(d.doc_id for d in _BM25_DOCS)),
        ("b", tuple(d.doc_id for d in _BM25_DOCS)),
    ]
    # FakeReranker rule reverses candidates, then truncated to top_k; aligned per query
    expected = list(reversed(_BM25_DOCS))[:2]
    assert results == [expected, expected]
