"""Phase 1 unit tests for benchmark.protocols (docs/experiment.md §3.2-§3.5).

Two seam kinds:
- structural ingest Protocols (``Dataset``, ``EmbeddingModel``, ``Indexer``, ``SearchBackend``):
  a trivial in-test class satisfies each (mypy checks data attributes; ``@runtime_checkable``
  ``isinstance`` here only verifies method presence).
- behavioral ABCs (``Searcher``, ``Fuser``, ``Reranker``): a trivial subclass implementing the
  abstract method instantiates and works; a subclass that does not is uninstantiable.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pytest

from benchmark.models import (
    Document,
    FieldSchema,
    IndexMapping,
    InferenceEndpoint,
    InferenceTaskType,
    Qrel,
    Query,
    ScoredDoc,
)
from benchmark.protocols import (
    Dataset,
    EmbeddingModel,
    Fuser,
    Indexer,
    Reranker,
    Searcher,
    SearchBackend,
)


class _Dataset:
    name = "fake"
    version = "1.0"

    def queries(self) -> Iterable[Query]:
        return []

    def documents(self) -> Iterable[Document]:
        return []

    def qrels(self) -> Iterable[Qrel]:
        return []

    def field_schema(self) -> FieldSchema:
        return FieldSchema(fields=[])


class _EmbeddingModel:
    inference_id = "e5-small"
    task_type = InferenceTaskType.TEXT_EMBEDDING

    def as_endpoint(self) -> InferenceEndpoint:
        return InferenceEndpoint(self.inference_id, self.task_type, "elasticsearch")


class _Backend:
    """A trivial ingest seam: register_inference / ensure_index / bulk_index only."""

    def register_inference(self, ep: InferenceEndpoint) -> str:
        return ep.inference_id

    def ensure_index(self, mapping: IndexMapping) -> None:
        return None

    def bulk_index(self, docs: Iterable[Document], *, mapping: IndexMapping) -> None:
        return None


class _Indexer:
    def build(
        self,
        dataset: Dataset,
        backend: SearchBackend,
        embeddings: Sequence[EmbeddingModel],
    ) -> IndexMapping:
        return IndexMapping("i", "search_text", {}, {})


# --- trivial ABC subclasses -------------------------------------------------------------------


class _Searcher(Searcher):
    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        return [ScoredDoc("d1", 1.0)][:top_k]


class _Fuser(Fuser):
    def fuse(
        self, result_lists: Sequence[Sequence[ScoredDoc]], *, rank_window_size: int
    ) -> list[ScoredDoc]:
        return [doc for lst in result_lists for doc in lst]


class _Reranker(Reranker):
    def rerank(self, query: str, candidates: Sequence[ScoredDoc]) -> list[ScoredDoc]:
        return list(reversed(candidates))


# --- static (mypy) structural conformance: assigning to the Protocol type ---------------------


def test_static_structural_conformance() -> None:
    dataset: Dataset = _Dataset()
    backend: SearchBackend = _Backend()
    embedding: EmbeddingModel = _EmbeddingModel()
    indexer: Indexer = _Indexer()
    assert dataset.name == "fake"
    assert backend.register_inference(embedding.as_endpoint()) == "e5-small"
    assert embedding.as_endpoint().inference_id == "e5-small"
    assert isinstance(indexer.build(dataset, backend, [embedding]), IndexMapping)


# --- runtime isinstance checks against @runtime_checkable ingest Protocols ---------------------


def test_runtime_isinstance_method_protocols() -> None:
    assert isinstance(_Dataset(), Dataset)
    assert isinstance(_Backend(), SearchBackend)
    assert isinstance(_EmbeddingModel(), EmbeddingModel)
    assert isinstance(_Indexer(), Indexer)


def test_runtime_isinstance_negative() -> None:
    class _NotABackend:
        pass

    assert not isinstance(_NotABackend(), SearchBackend)
    assert not isinstance(_NotABackend(), Dataset)


# --- behavioral ABCs: trivial subclasses satisfy them -----------------------------------------


def test_searcher_subclass_works() -> None:
    searcher: Searcher = _Searcher()
    assert searcher.search("q", top_k=10) == [ScoredDoc("d1", 1.0)]


def test_fuser_subclass_works() -> None:
    fuser: Fuser = _Fuser()
    fused = fuser.fuse([[ScoredDoc("d1", 1.0)], [ScoredDoc("d2", 2.0)]], rank_window_size=10)
    assert [d.doc_id for d in fused] == ["d1", "d2"]


def test_reranker_subclass_works() -> None:
    reranker: Reranker = _Reranker()
    reordered = reranker.rerank("q", [ScoredDoc("d1", 1.0), ScoredDoc("d2", 2.0)])
    assert [d.doc_id for d in reordered] == ["d2", "d1"]


def test_abc_missing_method_is_uninstantiable() -> None:
    class _Incomplete(Searcher):
        pass

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]
