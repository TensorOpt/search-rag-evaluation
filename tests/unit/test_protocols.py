"""Phase 1 unit tests for benchmark.protocols (docs/experiment.md §3.2-§3.5).

These verify that trivial in-test classes structurally satisfy each Protocol. mypy provides
full structural conformance checking (incl. data attributes); the runtime ``isinstance`` checks
here only verify method presence, since ``@runtime_checkable`` Protocols cannot check attributes.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from benchmark.models import (
    BackendCapabilities,
    Document,
    FieldSchema,
    IndexMapping,
    InferenceEndpoint,
    InferenceTaskType,
    Qrel,
    Query,
    RankedResult,
)
from benchmark.protocols import (
    Dataset,
    EmbeddingModel,
    Indexer,
    Reranker,
    RetrieverSpec,
    SearchBackend,
)


class _Spec:
    """A trivial opaque backend-native plan."""


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


class _Reranker:
    inference_id = "cohere-rerank-v3"
    task_type = InferenceTaskType.RERANK

    def as_endpoint(self) -> InferenceEndpoint:
        return InferenceEndpoint(self.inference_id, self.task_type, "cohere")


class _Backend:
    def register_inference(self, ep: InferenceEndpoint) -> str:
        return ep.inference_id

    def ensure_index(self, mapping: IndexMapping) -> None:
        return None

    def bulk_index(self, docs: Iterable[Document], *, mapping: IndexMapping) -> None:
        return None

    def bm25(self, *, fields: Sequence[str]) -> RetrieverSpec:
        return _Spec()

    def semantic(self, *, field: str) -> RetrieverSpec:
        return _Spec()

    def fuse_rrf(
        self,
        children: Sequence[RetrieverSpec],
        *,
        rank_constant: int,
        rank_window_size: int,
    ) -> RetrieverSpec:
        return _Spec()

    def rerank(
        self,
        child: RetrieverSpec,
        *,
        inference_id: str,
        field: str,
        rank_window_size: int,
    ) -> RetrieverSpec:
        return _Spec()

    def execute(self, spec: RetrieverSpec, query: Query, *, top_k: int) -> RankedResult:
        return RankedResult(query_id=query.query_id, docs=[])

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(True, True, True)


class _Indexer:
    def build(
        self,
        dataset: Dataset,
        backend: SearchBackend,
        embeddings: Sequence[EmbeddingModel],
    ) -> IndexMapping:
        return IndexMapping("i", "search_text", {}, {})


# --- static (mypy) structural conformance: assigning to the Protocol type -----


def test_static_structural_conformance() -> None:
    dataset: Dataset = _Dataset()
    backend: SearchBackend = _Backend()
    embedding: EmbeddingModel = _EmbeddingModel()
    reranker: Reranker = _Reranker()
    indexer: Indexer = _Indexer()
    spec: RetrieverSpec = _Spec()
    # touch them so the assignments are not flagged as unused.
    assert dataset.name == "fake"
    assert backend.capabilities().semantic_query is True
    assert embedding.as_endpoint().inference_id == "e5-small"
    assert reranker.task_type is InferenceTaskType.RERANK
    assert isinstance(indexer.build(dataset, backend, [embedding]), IndexMapping)
    assert spec is not None


# --- runtime isinstance checks against @runtime_checkable Protocols -----------
# Note: isinstance only verifies method presence, not data attributes.


def test_runtime_isinstance_method_protocols() -> None:
    assert isinstance(_Dataset(), Dataset)
    assert isinstance(_Backend(), SearchBackend)
    assert isinstance(_EmbeddingModel(), EmbeddingModel)
    assert isinstance(_Reranker(), Reranker)
    assert isinstance(_Indexer(), Indexer)


def test_runtime_isinstance_negative() -> None:
    class _NotABackend:
        pass

    assert not isinstance(_NotABackend(), SearchBackend)
    assert not isinstance(_NotABackend(), Dataset)


def test_backend_execute_returns_ranked_result() -> None:
    backend: SearchBackend = _Backend()
    result: Any = backend.execute(_Spec(), Query("q1", "text"), top_k=10)
    assert isinstance(result, RankedResult)
    assert result.query_id == "q1"
