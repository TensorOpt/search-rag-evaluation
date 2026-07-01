"""build_pipeline object-graph tests (docs/experiment.md §4, §10, plan Phase 6).

Builds each explicit pipeline shape (single lexical/vector leaf; HybridSearch+RRFFuser; +reranker
with rerank_window_size) via a fake SearcherFactory + a fake Services registry, asserting the
resulting SearchPipeline object graph. There is NO expansion/sweep — pipelines are explicit.
"""

from __future__ import annotations

from typing import Sequence

import pytest

from benchmark.matrix import (
    EmbedderCfg,
    FuserCfg,
    PipelineCfg,
    RerankerCfg,
    SearcherCfg,
    Services,
    build_pipeline,
)
from benchmark.models import IndexMapping, InferenceTaskType, ScoredDoc
from benchmark.pipeline import HybridSearch, SearchPipeline
from benchmark.protocols import Reranker, Searcher

WINDOW = 100


class _FakeSearcher(Searcher):
    def __init__(self, kind: str, target: object) -> None:
        self.kind = kind
        self.target = target

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        return []


class _FakeReranker(Reranker):
    def __init__(self, inference_id: str, field: str) -> None:
        self.inference_id = inference_id
        self.field = field

    def rerank(self, query: str, candidates: Sequence[ScoredDoc]) -> list[ScoredDoc]:
        return list(candidates)


class _FakeFactory:
    def lexical(self, *, fields: Sequence[str]) -> Searcher:
        return _FakeSearcher("lexical", list(fields))

    def vector(self, *, field: str) -> Searcher:
        return _FakeSearcher("vector", field)

    def reranker(self, inference_id: str, field: str) -> Reranker:
        return _FakeReranker(inference_id, field)


SERVICES = Services(
    embedders={
        "e5": EmbedderCfg("e5", "elasticsearch", InferenceTaskType.TEXT_EMBEDDING, {}),
    },
    rerankers={
        "co-rr": RerankerCfg("co-rr", "cohere", {"top_n": 100}),
    },
    searchers={
        "bm25": SearcherCfg("bm25", "elasticsearch", "lexical", None),
        "semantic_e5": SearcherCfg("semantic_e5", "elasticsearch", "vector", "e5"),
    },
)

MAPPING = IndexMapping(
    index_name="wands_bench",
    search_text_field="search_text",
    sem_fields={"e5": "sem__e5"},
    backend_mapping={},
)


def _build(pcfg: PipelineCfg) -> SearchPipeline:
    return build_pipeline(pcfg, SERVICES, MAPPING, _FakeFactory())


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
    assert pipeline.reranker.inference_id == "co-rr"
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
