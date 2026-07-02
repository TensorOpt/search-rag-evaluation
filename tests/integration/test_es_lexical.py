"""Live-ES integration test for the Phase 9 lexical path (marked; skips when ES is unreachable).

Runs against ``ES_URL`` (default ``http://localhost:9200``) on a uniquely-named throwaway index
that is created and deleted per test. Skipped automatically when the cluster is unreachable, so the
offline unit suite is unaffected. Does NOT depend on ``WandsDataset``. See docs/experiment.md §5.3.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from benchmark.backends.elasticsearch import (
    ElasticsearchBackend,
    LexicalSearcher,
)
from benchmark.models import Document, IndexMapping

pytestmark = pytest.mark.integration

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")


@pytest.fixture
def backend() -> ElasticsearchBackend:
    """An ``ElasticsearchBackend`` bound to a unique throwaway index; skip if ES is unreachable."""
    indexer_cfg = {"index": f"phase9_it_{uuid.uuid4().hex}", "settings": {"url": ES_URL}}
    backend = ElasticsearchBackend(indexer_cfg)
    try:
        if not backend.client.ping():
            pytest.skip(f"ES not reachable at {ES_URL}")
    except Exception as exc:  # noqa: BLE001 - any transport failure -> skip, not fail
        pytest.skip(f"ES not reachable at {ES_URL}: {exc}")
    return backend


@pytest.fixture
def mapping(backend: ElasticsearchBackend) -> Iterator[IndexMapping]:
    """Create the throwaway index, yield its mapping, and delete it on teardown."""
    mapping = IndexMapping(
        index_name=backend.index,
        search_text_field="search_text",
        sem_fields={},
        backend_mapping={"properties": {"search_text": {"type": "text"}}},
    )
    backend.ensure_index(mapping)
    try:
        yield mapping
    finally:
        backend.client.indices.delete(index=backend.index, ignore_unavailable=True)


def test_lexical_round_trip_and_distinctive_token(
    backend: ElasticsearchBackend, mapping: IndexMapping
) -> None:
    docs = [
        Document(doc_id="p1", fields={"search_text": "blue velvet sofa"}),
        Document(doc_id="p2", fields={"search_text": "wooden dining table"}),
        Document(doc_id="p3", fields={"search_text": "kitchen stool"}),
    ]
    backend.bulk_index(docs, mapping=mapping)

    searcher = LexicalSearcher(backend.client, backend.index, ["search_text"])
    result = searcher.search("velvet", top_k=10)

    assert result, "distinctive token should match at least one doc"
    assert result[0].doc_id == "p1"
    assert len(result) <= 10


def test_lexical_score_tie_breaks_on_doc_id_asc(
    backend: ElasticsearchBackend, mapping: IndexMapping
) -> None:
    # Identical text -> identical BM25 score -> a constructed tie; doc_ids chosen so asc order
    # differs from insertion order, proving the client-side (score desc, doc_id asc) tie-break.
    docs = [
        Document(doc_id="z9", fields={"search_text": "identical widget"}),
        Document(doc_id="a1", fields={"search_text": "identical widget"}),
        Document(doc_id="m5", fields={"search_text": "identical widget"}),
    ]
    backend.bulk_index(docs, mapping=mapping)

    searcher = LexicalSearcher(backend.client, backend.index, ["search_text"])
    result = searcher.search("identical widget", top_k=10)

    assert [d.doc_id for d in result] == ["a1", "m5", "z9"]


def test_lexical_respects_top_k(
    backend: ElasticsearchBackend, mapping: IndexMapping
) -> None:
    docs = [Document(doc_id=f"p{i}", fields={"search_text": "common term"}) for i in range(5)]
    backend.bulk_index(docs, mapping=mapping)

    searcher = LexicalSearcher(backend.client, backend.index, ["search_text"])
    result = searcher.search("common term", top_k=2)

    assert len(result) <= 2


def test_lexical_bulk_search_aligned_over_several_queries(
    backend: ElasticsearchBackend, mapping: IndexMapping
) -> None:
    # Distinctive tokens so each query has exactly one obvious match; a small msearch chunk size so
    # the query set spans more than one _msearch round trip.
    docs = [
        Document(doc_id="p1", fields={"search_text": "blue velvet sofa"}),
        Document(doc_id="p2", fields={"search_text": "wooden dining table"}),
        Document(doc_id="p3", fields={"search_text": "kitchen stool"}),
    ]
    backend.bulk_index(docs, mapping=mapping)

    searcher = LexicalSearcher(backend.client, backend.index, ["search_text"], msearch_chunk_size=2)
    results = searcher.bulk_search(["velvet", "dining", "stool"], top_k=10)

    assert len(results) == 3  # aligned to the query list by index
    assert results[0][0].doc_id == "p1"
    assert results[1][0].doc_id == "p2"
    assert results[2][0].doc_id == "p3"


def test_bulk_index_more_than_one_chunk_indexes_all_docs(
    backend: ElasticsearchBackend, mapping: IndexMapping
) -> None:
    # Small chunk size + more docs than one chunk -> multiple streaming_bulk chunks; assert ALL land.
    backend.bulk_chunk_size = 3
    n = 10
    docs = [Document(doc_id=f"p{i}", fields={"search_text": f"widget {i}"}) for i in range(n)]
    backend.bulk_index(docs, mapping=mapping)

    count = backend.client.count(index=backend.index)["count"]
    assert count == n
