"""Offline unit tests for the domain ``benchmark.indexing.Indexer`` (docs/experiment.md §3.5).

The backend-agnostic ``Indexer`` orchestration was split out of the old ES ``ESIndexer.build``: it
discovers each embedder's dim, asks the injected ``IndexWriter`` for the ``IndexMapping``, ensures the
index, streams the corpus through the embedders, and bulk-indexes. These tests drive ``Indexer.build``
against a RECORDING fake ``IndexWriter`` (no ES) so the ensure→index call order and the embed-at-ingest
enrichment (each doc lands carrying its ``dense_vector`` field) are observable — the behavioral
assertions preserved verbatim from the former ``test_es_backend`` ``ESIndexer.build`` cases. The
backend-specific mapping body / dot-free field naming is covered in
``tests/unit/providers/test_elasticsearch.py`` (``ESIndexWriter.create_mapping``/``sem_field_name``).
"""

from __future__ import annotations

import re
from typing import Any

from benchmark.common.models import Document, FieldRole, FieldSchema, FieldSpec, IndexMapping
from benchmark.common.protocols import Dataset, Embedder, IndexWriter
from benchmark.indexing import Indexer


class _FakeEmbedder(Embedder):
    """A fake ``Embedder``: fixed-``dim`` canned per-index document vectors (no network)."""

    def __init__(self, embedder_id: str, dim: int = 3) -> None:
        self.id = embedder_id
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Any) -> list[list[float]]:
        return [[float(i)] * self._dim for i, _ in enumerate(texts)]

    def embed_queries(self, texts: Any) -> list[list[float]]:
        return [[0.0] * self._dim for _ in texts]


class _FakeDataset(Dataset):
    """A tiny in-memory dataset with one text + one numeric + one id + one stored field."""

    name = "fake"
    version = "0"

    def queries(self) -> Any:
        return []

    def documents(self) -> Any:
        return iter([Document(doc_id="p1", fields={"search_text": "sofa"})])

    def qrels(self) -> Any:
        return []

    def field_schema(self) -> FieldSchema:
        return FieldSchema(
            fields=[
                FieldSpec("product_id", FieldRole.ID),
                FieldSpec("product_name", FieldRole.BM25),
                FieldSpec("product_description", FieldRole.SEMANTIC_SOURCE),
                FieldSpec("rating", FieldRole.NUMERIC),
                FieldSpec("product_class", FieldRole.STORED),
            ]
        )


class _RecordingWriter(IndexWriter):
    """A fake ``IndexWriter`` recording call order + the mapping/docs it was handed (no ES register).

    ``sem_field_name`` mirrors the backend-safe dot-free naming; ``create_mapping`` returns a real
    ``IndexMapping`` so the ``Indexer`` can name fields; ``ensure_index``/``bulk_index`` record order.
    """

    embed_batch_size = 96

    def __init__(self) -> None:
        self.index = "wands_bench"
        self.calls: list[str] = []
        self.ensured_mapping: Any = None
        self.indexed_docs: list[Document] = []

    def sem_field_name(self, embedder_id: str) -> str:
        return "sem__" + re.sub(r"[^0-9a-zA-Z]+", "_", embedder_id)

    def create_mapping(self, schema: FieldSchema, sem_fields: Any, vector_dims: Any) -> IndexMapping:
        properties: dict[str, Any] = {schema.search_text_field: {"type": "text"}}
        for field_name, dims in vector_dims.items():
            properties[field_name] = {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
            }
        return IndexMapping(
            index_name=self.index,
            search_text_field=schema.search_text_field,
            sem_fields=dict(sem_fields),
            backend_mapping={"properties": properties},
        )

    def ensure_index(self, mapping: Any) -> None:
        self.calls.append("ensure")
        self.ensured_mapping = mapping

    def bulk_index(self, docs: Any, *, mapping: Any) -> None:
        self.calls.append("index")
        self.indexed_docs = list(docs)  # drive the streamed generator


def test_indexer_builds_dense_vector_mapping_and_embeds_at_ingest() -> None:
    writer = _RecordingWriter()
    # embedder id carries dots -> the dense_vector field name must be dot-free.
    embedders = [_FakeEmbedder("e5.small.v1", dim=4)]

    mapping = Indexer(writer, embedders).build(_FakeDataset())

    # ensure BEFORE index (no register step — ES is a plain index)
    assert writer.calls == ["ensure", "index"]

    props = mapping.backend_mapping["properties"]
    sem_field = "sem__e5_small_v1"  # dots -> "_", prefixed
    # search_text is a plain text field — NO copy_to, NO semantic_text (§5.2)
    assert props["search_text"] == {"type": "text"}
    # one dense_vector field per embedder (dims from embedder.dim, cosine, indexed)
    assert props[sem_field] == {
        "type": "dense_vector",
        "dims": 4,
        "index": True,
        "similarity": "cosine",
    }
    assert "." not in sem_field

    # sem_fields resolves via IndexMapping.sem_field(embedder_id)
    assert mapping.sem_field("e5.small.v1") == sem_field
    assert mapping.search_text_field == "search_text"
    assert mapping.index_name == "wands_bench"

    # the corpus was embedded at ingest: each indexed doc carries its dense_vector under the sem field
    assert [d.doc_id for d in writer.indexed_docs] == ["p1"]
    stored = writer.indexed_docs[0].fields[sem_field]
    assert stored == [0.0, 0.0, 0.0, 0.0]  # _FakeEmbedder's index-0 vector, dim 4
    assert writer.indexed_docs[0].fields["search_text"] == "sofa"  # original field preserved
    assert writer.ensured_mapping is mapping


def test_indexer_multiple_embedders_one_dense_vector_each() -> None:
    writer = _RecordingWriter()
    embedders = [_FakeEmbedder("e5-small", dim=3), _FakeEmbedder("elser", dim=5)]

    mapping = Indexer(writer, embedders).build(_FakeDataset())

    props = mapping.backend_mapping["properties"]
    assert props["sem__e5_small"]["type"] == "dense_vector" and props["sem__e5_small"]["dims"] == 3
    assert props["sem__elser"]["type"] == "dense_vector" and props["sem__elser"]["dims"] == 5
    assert mapping.sem_field("elser") == "sem__elser"
    # each indexed doc carries BOTH embedders' vectors
    doc_fields = writer.indexed_docs[0].fields
    assert len(doc_fields["sem__e5_small"]) == 3
    assert len(doc_fields["sem__elser"]) == 5
