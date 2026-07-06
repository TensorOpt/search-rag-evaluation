"""Domain indexing seam (a): the backend-agnostic ``Indexer`` + embed-at-ingest streaming (docs/experiment.md §3.5).

The orchestration that used to live in the ES adapter's ``ESIndexer.build`` is now a clean,
backend-agnostic domain object. ``Indexer`` composes an injected :class:`~benchmark.common.protocols.IndexWriter`
(the ES-specific mapping/field-naming/ingest, a ``provider``) with the injected
:class:`~benchmark.common.protocols.Embedder` connectors (built by the ``embedding`` factory):

    discover each embedder's ``dim`` -> ask the writer for the ``IndexMapping`` -> ensure the index ->
    stream the corpus through the embedders -> ``bulk_index``.

Imports only ``common`` abstractions at import time (§11); the concrete ``IndexWriter`` +
``Embedder``s are INJECTED by the composition layer (``ExperimentRunner.build_index``), so this
domain module names no adapter. ``_embed_documents`` / ``_embed_batch`` only touch ``Document`` +
``Embedder``, so they are domain (not backend) code and live here.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Mapping, Sequence

from benchmark.common.logging_setup import get_logger
from benchmark.common.models import Document, IndexMapping
from benchmark.common.protocols import Dataset, Embedder, IndexWriter

logger = get_logger(__name__)


class Indexer:
    """Builds + populates an index from a dataset + embedder connectors, backend-agnostic (§3.5).

    Lifecycle: discover each embedder's output ``dim`` (probe / ``settings.dims``), ask the
    ``IndexWriter`` to translate the dataset ``FieldSchema`` into the §5.2 ``IndexMapping`` (a plain
    ``text`` ``search_text`` field + one ``dense_vector`` field per embedder), then stream the corpus
    THROUGH the embedders (:func:`_embed_documents`) so each document lands with its vectors. No
    inference endpoints are registered (ES is a plain index, §1.1).
    """

    def __init__(self, writer: IndexWriter, embedders: Sequence[Embedder]) -> None:
        self.writer = writer
        self.embedders = list(embedders)

    def build(self, dataset: Dataset) -> IndexMapping:
        # 1. sem_fields: embedder id -> dense_vector field name (backend-safe, from the writer).
        #    Discover each dim (probes the provider or reads settings.dims) — needed to map the
        #    dense_vector field before ingest.
        sem_fields: dict[str, str] = {
            embedder.id: self.writer.sem_field_name(embedder.id) for embedder in self.embedders
        }
        vector_dims: dict[str, int] = {
            sem_fields[embedder.id]: embedder.dim for embedder in self.embedders
        }

        # 2. Translate the dataset field schema into the backend index mapping (§5.2), owned by the writer.
        schema = dataset.field_schema()
        mapping = self.writer.create_mapping(schema, sem_fields, vector_dims)
        logger.info(
            "index %r: %d embedder(s) -> dense_vector fields %s",
            mapping.index_name, len(self.embedders), vector_dims,
        )

        # 3. Create the index, then stream the corpus through the embedders so each doc lands with
        #    its dense_vector values. embed_batch_size is the ingest buffering granularity (§3.5).
        self.writer.ensure_index(mapping)
        enriched = _embed_documents(
            dataset.documents(),
            self.embedders,
            sem_fields,
            schema.search_text_field,
            self.writer.embed_batch_size,
        )
        self.writer.bulk_index(enriched, mapping=mapping)

        # 4. Return the mapping so leaf Searchers can name fields without re-deriving them.
        return mapping


def _embed_documents(
    docs: Iterable[Document],
    embedders: Sequence[Embedder],
    sem_fields: Mapping[str, str],
    search_text_field: str,
    batch_size: int,
) -> Iterator[Document]:
    """Stream ``docs``, attaching each embedder's document vector under its ``dense_vector`` field (§3.5).

    Buffers ``batch_size`` docs, embeds their ``search_text`` with EACH embedder in one provider call
    per batch (the connector sub-chunks to its own limit), and yields COPIES of the docs enriched with
    the vectors — STAYS LAZY (bounded buffer; the corpus is never fully materialized). A doc missing
    the ``search_text`` field RAISES (never silently embeds empty).
    """
    batch: list[Document] = []
    for doc in docs:
        batch.append(doc)
        if len(batch) >= batch_size:
            yield from _embed_batch(batch, embedders, sem_fields, search_text_field)
            batch = []
    if batch:
        yield from _embed_batch(batch, embedders, sem_fields, search_text_field)


def _embed_batch(
    batch: Sequence[Document],
    embedders: Sequence[Embedder],
    sem_fields: Mapping[str, str],
    search_text_field: str,
) -> Iterator[Document]:
    """Embed one buffered batch with every embedder; yield the docs enriched with their vectors."""
    texts: list[str] = []
    for doc in batch:
        if search_text_field not in doc.fields:
            raise KeyError(
                f"document {doc.doc_id!r} has no {search_text_field!r} field to embed"
            )
        texts.append(doc.fields[search_text_field])
    vectors_by_embedder = {embedder.id: embedder.embed_documents(texts) for embedder in embedders}
    for offset, doc in enumerate(batch):
        fields = dict(doc.fields)
        for embedder in embedders:
            fields[sem_fields[embedder.id]] = vectors_by_embedder[embedder.id][offset]
        yield Document(doc_id=doc.doc_id, fields=fields)
