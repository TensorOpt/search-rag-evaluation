"""Live integration tests for the ES adapter (marked; skip-not-fail, never touches the unit suite).

ES is a plain vector/BM25 index (§1.1). Runs against ``ES_URL`` (default ``http://localhost:9200``)
on uniquely-named throwaway indices, created and deleted per test. SKIPS (never fails) when the
cluster is unreachable, when a required provider API key is absent, or on a provider ``ProviderError``
(an environment constraint — auth/rate limit — not a code defect). Covers the lexical path
(``LexicalSearcher`` + ``bulk_index`` chunking), the semantic path (client-side embed → ES ``knn``
over a live ``dense_vector`` field, via a real embedding connector), and ``ESReranker`` over a real
provider ``RerankClient``. The semantic + rerank tests need a Cohere API key
(``COHERE_KEY``). Does NOT depend on ``WandsDataset``. See docs/architecture.md §5.2, §5.3, §5.4.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from benchmark.common.models import Document, IndexMapping, ScoredDoc
from benchmark.providers.elasticsearch import (
    ESIndexWriter,
    ESReranker,
    LexicalSearcher,
    VectorSearch,
)
from benchmark.providers.inference import CohereEmbedder, CohereReranker, ProviderError

pytestmark = pytest.mark.integration

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
COHERE_KEY = os.environ.get("COHERE_KEY")


def _require_cohere() -> str:
    """Skip (not fail) when no Cohere API key is configured — the connector tests need one."""
    if not COHERE_KEY:
        pytest.skip("COHERE_KEY not set — provider connector tests need a live key")
    return COHERE_KEY


@pytest.fixture
def backend() -> ESIndexWriter:
    """An ``ESIndexWriter`` bound to a unique throwaway index; skip if ES is unreachable."""
    indexer_cfg = {"index": f"es_it_{uuid.uuid4().hex}", "settings": {"url": ES_URL}}
    backend = ESIndexWriter(indexer_cfg)
    try:
        if not backend.client.ping():
            pytest.skip(f"ES not reachable at {ES_URL}")
    except Exception as exc:  # noqa: BLE001 - any transport failure -> skip, not fail
        pytest.skip(f"ES not reachable at {ES_URL}: {exc}")
    return backend


@pytest.fixture
def mapping(backend: ESIndexWriter) -> Iterator[IndexMapping]:
    """A plain text index (BM25 only); deleted on teardown.

    Shared by the lexical searchers and by the reranker's ``mget`` doc-text lookup — neither needs a
    vector field, so this fixture is fully local (no provider call).
    """
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


# --- lexical path -----------------------------------------------------------------------------


def test_lexical_round_trip_and_distinctive_token(
    backend: ESIndexWriter, mapping: IndexMapping
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
    backend: ESIndexWriter, mapping: IndexMapping
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
    backend: ESIndexWriter, mapping: IndexMapping
) -> None:
    docs = [Document(doc_id=f"p{i}", fields={"search_text": "common term"}) for i in range(5)]
    backend.bulk_index(docs, mapping=mapping)

    searcher = LexicalSearcher(backend.client, backend.index, ["search_text"])
    result = searcher.search("common term", top_k=2)

    assert len(result) <= 2


def test_lexical_bulk_search_aligned_over_several_queries(
    backend: ESIndexWriter, mapping: IndexMapping
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
    backend: ESIndexWriter, mapping: IndexMapping
) -> None:
    # Small chunk size + more docs than one chunk -> multiple streaming_bulk chunks; assert ALL land.
    backend.bulk_chunk_size = 3
    n = 10
    docs = [Document(doc_id=f"p{i}", fields={"search_text": f"widget {i}"}) for i in range(n)]
    backend.bulk_index(docs, mapping=mapping)

    count = backend.client.count(index=backend.index)["count"]
    assert count == n


# --- semantic path (client-side embed via connector -> ES knn) --------------------------------


def test_vector_search_knn_returns_scored_docs(backend: ESIndexWriter) -> None:
    api_key = _require_cohere()
    embedder = CohereEmbedder("cohere", {"api_key": api_key, "model_id": "embed-english-v3.0"})
    sem_field = "sem__cohere"

    try:
        dim = embedder.dim  # probes the provider once for the output dimensionality
        docs = [
            Document(doc_id="p1", fields={"search_text": "a comfortable blue velvet sofa"}),
            Document(doc_id="p2", fields={"search_text": "a wooden dining table"}),
        ]
        vectors = embedder.embed_documents([d.fields["search_text"] for d in docs])
    except ProviderError as exc:
        pytest.skip(f"cohere embedding unavailable (env constraint): {exc}")

    mapping = IndexMapping(
        index_name=backend.index,
        search_text_field="search_text",
        sem_fields={"cohere": sem_field},
        backend_mapping={
            "properties": {
                "search_text": {"type": "text"},
                sem_field: {"type": "dense_vector", "dims": dim, "index": True, "similarity": "cosine"},
            }
        },
    )
    backend.ensure_index(mapping)
    try:
        enriched = [
            Document(doc_id=doc.doc_id, fields={**doc.fields, sem_field: vector})
            for doc, vector in zip(docs, vectors)
        ]
        backend.bulk_index(enriched, mapping=mapping)

        searcher = VectorSearch(backend.client, backend.index, sem_field, embedder)
        try:
            result = searcher.search("couch for the living room", top_k=10)
        except ProviderError as exc:
            pytest.skip(f"cohere query embedding unavailable (env constraint): {exc}")

        assert result, "knn query should return at least one scored doc"
        assert all(isinstance(doc.score, float) for doc in result)
        assert {doc.doc_id for doc in result} <= {"p1", "p2"}
    finally:
        backend.client.indices.delete(index=backend.index, ignore_unavailable=True)


# --- reranker (ESReranker over a live provider RerankClient) ----------------------------------


def test_reranker_reorders_candidates_by_model_score(
    backend: ESIndexWriter, mapping: IndexMapping
) -> None:
    api_key = _require_cohere()
    docs = [
        Document(doc_id="c1", fields={"search_text": "a wooden dining table"}),
        Document(doc_id="c2", fields={"search_text": "a comfortable blue velvet sofa"}),
        Document(doc_id="c3", fields={"search_text": "a stainless steel kitchen knife"}),
    ]
    backend.bulk_index(docs, mapping=mapping)

    rerank_client = CohereReranker("co-rr", {"api_key": api_key, "model_id": "rerank-v3.5"})
    reranker = ESReranker(backend.client, backend.index, "search_text", rerank_client)
    # Retrieval order deliberately NOT the relevance order for the query "couch to sit on".
    candidates = [ScoredDoc("c1", 1.0), ScoredDoc("c3", 1.0), ScoredDoc("c2", 1.0)]
    try:
        result = reranker.rerank("a couch to sit on in the living room", candidates)
    except ProviderError as exc:
        pytest.skip(f"cohere rerank unavailable (env constraint): {exc}")

    assert {doc.doc_id for doc in result} == {"c1", "c2", "c3"}
    # the sofa/couch doc should rank first after reranking
    assert result[0].doc_id == "c2"
    # scores are model scores, sorted DESC
    scores = [doc.score for doc in result]
    assert scores == sorted(scores, reverse=True)
