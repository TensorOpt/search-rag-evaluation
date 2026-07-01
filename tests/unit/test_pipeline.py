"""Composite-model composer tests (docs/experiment.md §3.6/§3.7, plan Phase 5).

Exercises the three backend-agnostic composers against ``FakeSearcher``/``FakeReranker``:
``RRFFuser`` fuses, ``HybridSearch`` retrieves-at-window then fuses then truncates,
``SearchPipeline`` reranks (or passes through) then truncates, and ``SearchPipeline.__init__``
rejects misconfiguration.
"""

from __future__ import annotations

import pytest

from benchmark.fusion import fuse_rrf_local
from benchmark.pipeline import HybridSearch, RRFFuser, SearchPipeline
from tests.conftest import _BM25_DOCS, _SEMANTIC_DOCS, FakeReranker, FakeSearcher

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
