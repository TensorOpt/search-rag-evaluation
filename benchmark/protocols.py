"""Core abstraction seams (docs/experiment.md §3.2-§3.5).

Two kinds of seam live here:

- **Behavioral ABCs** (``abc.ABC`` + ``@abstractmethod``): ``Searcher``, ``Fuser``, ``Reranker``.
  These are the composite retrieval model (§3.3/§3.6): everything that produces a ranked list is
  a ``Searcher``; composition (client-side fusion, reranking) mirrors a real search pipeline.
- **Structural Protocols** for the ingest side: ``Dataset``, ``EmbeddingModel`` (a descriptor),
  ``Indexer``, and ``SearchBackend`` (the index-writer/ingest seam used by ``Indexer.build``).

Data models are imported from ``benchmark.models``. Structural Protocols are ``@runtime_checkable``
so tests can do ``isinstance`` checks (note: that only verifies method presence, not data
attributes — rely on mypy for full structural conformance).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Protocol, Sequence, runtime_checkable

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


class Searcher(ABC):
    """Anything that turns a query into a ranked list of docs (§3.3).

    The single composite seam: leaf retrievers (ES lexical/vector queries), client-side
    ``HybridSearch`` fusion, and the top-level ``SearchPipeline`` are all ``Searcher``s, so
    they compose uniformly.
    """

    @abstractmethod
    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        """Return up to ``top_k`` docs ranked best-first (score desc, tie-break doc_id, §9.1)."""
        ...


class Fuser(ABC):
    """Combines several ranked lists into one, client-side (§3.7)."""

    @abstractmethod
    def fuse(
        self, result_lists: Sequence[Sequence[ScoredDoc]], *, rank_window_size: int
    ) -> list[ScoredDoc]:
        """Fuse ``result_lists`` over a fixed window into one ranked list."""
        ...


class Reranker(ABC):
    """Rescores + reorders a candidate list for a query, client-side (§3.4/§3.7).

    Behavioral now (was a descriptor Protocol): a concrete reranker calls its backend's rerank
    inference over the candidate doc-text and returns the reordered list.
    """

    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[ScoredDoc]) -> list[ScoredDoc]:
        """Return ``candidates`` reordered best-first by the model's relevance scores."""
        ...


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
    """A pluggable embedding model descriptor that flattens to an InferenceEndpoint (§3.4).

    Stays a descriptor (not behavioral): embeddings are registered once at ingest via
    ``SearchBackend.register_inference`` and then produced by the backend at index time.
    """

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
    """The index-writer / ingest seam used by ``Indexer.build`` (§3.3/§3.5).

    This is the only place that knows a wire format for WRITING: it registers inference
    endpoints, creates the index mapping, and bulk-indexes documents (ES embeds each
    ``semantic_text`` field at ingest). RETRIEVAL is no longer here — it moved to the
    ``Searcher`` / ``Fuser`` / ``Reranker`` composite model (§3.3/§3.6); a backend realizes
    those as concrete ``Searcher``/``Reranker`` implementations (e.g. ES ``LexicalSearcher`` /
    ``VectorSearch`` / ``ESReranker`` in Phase 9/10).
    """

    def register_inference(self, ep: InferenceEndpoint) -> str: ...
    def ensure_index(self, mapping: IndexMapping) -> None: ...
    def bulk_index(self, docs: Iterable[Document], *, mapping: IndexMapping) -> None: ...


@runtime_checkable
class SearcherFactory(Protocol):
    """Backend-agnostic builder of leaf ``Searcher``s / the ``Reranker`` (§4).

    ``build_pipeline`` (``config.py``) uses this seam to assemble a pipeline's ``SearchPipeline`` object
    graph without importing any adapter: the backend supplies a factory that binds these to its
    client + ``IndexMapping`` (ES ``LexicalSearcher``/``VectorSearch``/``ESReranker``, Phase 9/10).
    """

    def lexical(self, *, fields: Sequence[str]) -> Searcher: ...
    def vector(self, *, field: str) -> Searcher: ...
    def reranker(self, inference_id: str, field: str) -> Reranker: ...
