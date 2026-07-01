"""Structural Protocol seams for datasets, backends, and models (docs/experiment.md §3.2-§3.5). Phase 1.

``typing.Protocol`` definitions ONLY — structural, no implementation. All data models are
imported from ``benchmark.models``. The Protocols are ``@runtime_checkable`` so tests can do
structural ``isinstance`` checks (note: ``isinstance`` only verifies method presence, not data
attributes — rely on mypy for full structural conformance).
"""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence, runtime_checkable

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


@runtime_checkable
class RetrieverSpec(Protocol):
    """Opaque, backend-native retrieval plan (built but not yet executed) (§3.3).

    The pipeline only composes these via the backend's combinators; it never inspects
    the internals.
    """


@runtime_checkable
class Dataset(Protocol):
    """A dataset adapter (§3.2)."""

    name: str
    version: str

    def queries(self) -> Iterable[Query]: ...
    def documents(self) -> Iterable[Document]: ...
    def qrels(self) -> Iterable[Qrel]: ...
    def field_schema(self) -> FieldSchema: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """A pluggable embedding model descriptor that flattens to an InferenceEndpoint (§3.4)."""

    inference_id: str
    task_type: InferenceTaskType

    def as_endpoint(self) -> InferenceEndpoint: ...


@runtime_checkable
class Reranker(Protocol):
    """A pluggable reranker descriptor that flattens to an InferenceEndpoint (§3.4)."""

    inference_id: str
    task_type: InferenceTaskType

    def as_endpoint(self) -> InferenceEndpoint: ...


@runtime_checkable
class Indexer(Protocol):
    """Builds a backend index from a dataset + embedding models (§3.5)."""

    def build(
        self,
        dataset: Dataset,
        backend: SearchBackend,
        embeddings: Sequence[EmbeddingModel],
    ) -> IndexMapping: ...


@runtime_checkable
class SearchBackend(Protocol):
    """Storage + retrieval primitives; the only place that knows a wire format (§3.3)."""

    # ---- lifecycle ----
    def register_inference(self, ep: InferenceEndpoint) -> str: ...
    def ensure_index(self, mapping: IndexMapping) -> None: ...
    def bulk_index(self, docs: Iterable[Document], *, mapping: IndexMapping) -> None: ...

    # ---- retrieval primitives: build plans, do not execute ----
    def bm25(self, *, fields: Sequence[str]) -> RetrieverSpec: ...
    def semantic(self, *, field: str) -> RetrieverSpec: ...
    def fuse_rrf(
        self,
        children: Sequence[RetrieverSpec],
        *,
        rank_constant: int,
        rank_window_size: int,
    ) -> RetrieverSpec: ...
    def rerank(
        self,
        child: RetrieverSpec,
        *,
        inference_id: str,
        field: str,
        rank_window_size: int,
    ) -> RetrieverSpec: ...

    # ---- execution: bind the query, run, return a ranked list ----
    def execute(self, spec: RetrieverSpec, query: Query, *, top_k: int) -> RankedResult: ...

    def capabilities(self) -> BackendCapabilities: ...
