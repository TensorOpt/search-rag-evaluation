"""Core abstraction seams (docs/experiment.md §3.2-§3.5).

Two kinds of seam live here:

- **ABCs** (``abc.ABC`` + ``@abstractmethod``): the behavioral retrieval seams ``Searcher``,
  ``Fuser``, ``Reranker`` (§3.3/§3.6 — everything that produces a ranked list is a ``Searcher``;
  composition mirrors a real search pipeline), plus the ``Dataset`` base every dataset adapter
  derives from (§3.2). ``Dataset`` is an ABC (not a Protocol) so it can carry the two shared
  concrete helpers (``build_search_text``, ``map_label``) every adapter reuses.
- **Structural Protocols** for the rest of the ingest/inference side: ``Embedder`` and
  ``RerankClient`` (the provider connectors, realized in ``benchmark.providers``), ``Indexer``, and
  ``SearchBackend`` (the index-writer/ingest seam used by ``Indexer.build``).

Data models are imported from ``benchmark.models``. Structural Protocols are ``@runtime_checkable``
so tests can do ``isinstance`` checks (note: that only verifies method presence, not data
attributes — rely on mypy for full structural conformance).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from benchmark.models import (
    Document,
    FieldRole,
    FieldSchema,
    IndexMapping,
    Qrel,
    Query,
    ScoredDoc,
)

#: Text roles whose values are concatenated (in schema order) into the canonical search_text (§5.1).
_SEARCH_TEXT_ROLES = frozenset({FieldRole.BM25, FieldRole.SEMANTIC_SOURCE})


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

    def bulk_search(self, queries: Sequence[str], *, top_k: int) -> list[list[ScoredDoc]]:
        """Search several queries at once; results are ALIGNED to ``queries`` by index (§3.3).

        The default implementation loops :meth:`search` (correct — one round trip per query) so any
        ``Searcher`` (e.g. a fake) works without extra code. Efficient backends OVERRIDE this to
        batch the round trips: the ES leaf searchers (``LexicalSearcher``/``VectorSearch``) override
        it via the Multi-Search API (``_msearch``), and the composers (``HybridSearch``/
        ``SearchPipeline``) override it to propagate batching to their leaves. This default is the
        correctness fallback.
        """
        return [self.search(query, top_k=top_k) for query in queries]


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


class Dataset(ABC):
    """The base every dataset adapter derives from (§3.2).

    The single, format-agnostic dataset seam: a concrete adapter (``WandsDataset``; future
    Amazon ESCI, BEIR) implements the four abstract methods below and owns its own file parsing,
    label->gain mapping, and field roles. ``queries``/``documents``/``qrels``/``field_schema`` +
    ``Qrel(gain: float)`` are enough to describe any graded-relevance IR dataset regardless of
    on-disk format (TSV, parquet, JSONL, …). Nothing dataset-specific lives on this base.

    Subclasses MUST set two attributes in ``__init__``:

    - ``name``: the dataset's config-dispatch name (e.g. ``"wands"``).
    - ``version``: the dataset version string (e.g. ``"2022.0"``).

    Two concrete helpers are shared by all adapters (the reason this is an ABC, not a Protocol):
    :meth:`build_search_text` (the §5.1 search_text concatenation) and :meth:`map_label` (a
    string-label -> gain mapper for datasets whose qrels use string labels).
    """

    #: Config-dispatch name; a subclass sets this in ``__init__`` (e.g. ``"wands"``).
    name: str
    #: Dataset version string; a subclass sets this in ``__init__`` (e.g. ``"2022.0"``).
    version: str

    @abstractmethod
    def queries(self) -> Iterable[Query]:
        """Yield every :class:`Query` in the dataset."""
        ...

    @abstractmethod
    def documents(self) -> Iterable[Document]:
        """Yield every :class:`Document` (streamed for large corpora)."""
        ...

    @abstractmethod
    def qrels(self) -> Iterable[Qrel]:
        """Yield every graded relevance judgement as a :class:`Qrel` (``gain`` is a float)."""
        ...

    @abstractmethod
    def field_schema(self) -> FieldSchema:
        """Declare this dataset's field roles + canonical text fields (§3.2/§5.1)."""
        ...

    @staticmethod
    def build_search_text(field_values: Mapping[str, Any], schema: FieldSchema) -> str:
        """Concatenate the BM25- and SEMANTIC_SOURCE-role field values into search_text (§5.1).

        Joins the values of every ``BM25``- and ``SEMANTIC_SOURCE``-role field, in ``schema``
        (``FieldSpec``) order, by ``"\\n"`` — the single canonical text used as BOTH the BM25
        target and the semantic source, so every variant ranks the same input. A search-text
        field missing from ``field_values`` raises ``KeyError`` (never silently emits empty).
        """
        return "\n".join(
            str(field_values[spec.name])
            for spec in schema.fields
            if spec.role in _SEARCH_TEXT_ROLES
        )

    @staticmethod
    def map_label(label: str, mapping: Mapping[str, float]) -> float:
        """Map a string relevance label to a float gain via ``mapping`` (§7). Exhaustive.

        Convenience for string-labeled datasets (WANDS Exact/Partial/Irrelevant, ESCI E/S/C/I).
        A label not in ``mapping`` raises ``ValueError`` — no silent default. Numeric-qrel
        datasets (BEIR) skip this and set ``gain = float(rel)`` directly.
        """
        if label not in mapping:
            raise ValueError(f"unknown label {label!r}; expected one of {sorted(mapping)}")
        return mapping[label]


@runtime_checkable
class Embedder(Protocol):
    """A dense-embedding provider connector (§3.4), realized in ``benchmark.providers``.

    The harness embeds the corpus at ingest (``embed_documents``) into ``dense_vector`` fields and
    embeds each query at search time (``embed_queries``) to run ES ``knn``. ``id`` is the config
    service name (== the sem-field naming key, §3.5); ``dim`` is the output dimensionality (probed
    once or taken from ``settings.dims``) the ``dense_vector`` mapping needs before ingest.
    """

    id: str

    @property
    def dim(self) -> int: ...
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class RerankClient(Protocol):
    """A rerank provider connector (§3.4/§5.4), realized in ``benchmark.providers``.

    Returns one relevance score per document, ALIGNED 1:1 to ``documents`` (higher = more relevant).
    A backend ``Reranker`` (ES ``ESReranker``) fetches candidate doc-text and calls this over it.
    """

    def rerank_scores(self, query: str, documents: Sequence[str]) -> list[float]: ...


@runtime_checkable
class Indexer(Protocol):
    """Builds a backend index from a dataset + embedder connectors (§3.5)."""

    def build(
        self,
        dataset: Dataset,
        backend: SearchBackend,
        embeddings: Sequence[Embedder],
    ) -> IndexMapping: ...


@runtime_checkable
class SearchBackend(Protocol):
    """The index-writer / ingest seam used by ``Indexer.build`` (§3.3/§3.5).

    This is the only place that knows a wire format for WRITING: it creates the index mapping and
    bulk-indexes documents. ES is a plain index writer now — the harness embeds the corpus
    client-side (``Embedder``) and stores the vectors in ``dense_vector`` fields; no inference runs
    server-side. RETRIEVAL is no longer here — it moved to the ``Searcher`` / ``Fuser`` /
    ``Reranker`` composite model (§3.3/§3.6); a backend realizes those as concrete
    ``Searcher``/``Reranker`` implementations (e.g. ES ``LexicalSearcher`` / ``VectorSearch`` /
    ``ESReranker`` in Phase 9/10).
    """

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
    def vector(self, *, field: str, embedder_id: str) -> Searcher: ...
    def reranker(self, name: str, field: str) -> Reranker: ...
