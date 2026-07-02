"""Live-ES integration tests for the ES adapter (marked; skip-not-fail, never touches the unit suite).

Runs against ``ES_URL`` (default ``http://localhost:9200``) on uniquely-named throwaway indices,
created and deleted per test. SKIPS (never fails) when the cluster is unreachable OR when a required
ML model cannot be deployed (HTTP 429 ``status_exception`` — insufficient memory / could-not-start-
deployment); see ``_skip_if_deploy_error``. Covers the lexical path (``LexicalSearcher`` +
``bulk_index`` chunking), the semantic path (``VectorSearch`` over a live ``semantic_text`` field,
E5-embedded at ingest), and ``ESReranker`` (``.rerank-v1``). The semantic and rerank tests are
INDEPENDENT — the ``VectorSearch`` test needs only E5, the reranker test needs only ``.rerank-v1`` —
so neither requires both models resident at once. Does NOT depend on ``WandsDataset``.
See docs/experiment.md §5.2, §5.3, §13.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from elasticsearch import ApiError
from elasticsearch.helpers import BulkIndexError

from benchmark.backends.elasticsearch import (
    ElasticsearchBackend,
    ESReranker,
    LexicalSearcher,
    VectorSearch,
)
from benchmark.models import Document, IndexMapping, ScoredDoc

pytestmark = pytest.mark.integration

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")

# Preconfigured endpoints in the validation container (§ prompt): E5 (dense, 384-dim) + rerank v1.
E5_INFERENCE_ID = ".multilingual-e5-small-elasticsearch"
RERANK_INFERENCE_ID = ".rerank-v1-elasticsearch"


def _skip_if_deploy_error(exc: ApiError | BulkIndexError) -> None:
    """Skip (not fail) on a 429 model-deployment/memory error; re-raise anything else.

    When ML memory is tight a model deploy returns HTTP 429 ``status_exception`` ("insufficient ...
    memory" / "Could not start deployment"). That surfaces two ways: a direct ``ApiError`` (a
    search/rerank call), OR — at ingest — a ``BulkIndexError`` whose per-item ``status`` is 429 (ES
    embeds each ``semantic_text`` at ingest, so a failed E5 deploy fails the ingest item). Either is
    an environment constraint, not a code defect — skip it; anything else re-raises.
    """
    if isinstance(exc, BulkIndexError):
        if any(
            next(iter(item.values())).get("status") == 429 for item in exc.errors
        ):
            pytest.skip(f"model could not be deployed (429 on ingest): {exc.errors[:1]}")
        raise exc
    if exc.status_code == 429:
        pytest.skip(f"model could not be deployed (429, insufficient ML memory): {exc}")
    raise exc


@pytest.fixture
def backend() -> ElasticsearchBackend:
    """An ``ElasticsearchBackend`` bound to a unique throwaway index; skip if ES is unreachable."""
    indexer_cfg = {"index": f"es_it_{uuid.uuid4().hex}", "settings": {"url": ES_URL}}
    backend = ElasticsearchBackend(indexer_cfg)
    try:
        if not backend.client.ping():
            pytest.skip(f"ES not reachable at {ES_URL}")
    except Exception as exc:  # noqa: BLE001 - any transport failure -> skip, not fail
        pytest.skip(f"ES not reachable at {ES_URL}: {exc}")
    return backend


@pytest.fixture
def mapping(backend: ElasticsearchBackend) -> Iterator[IndexMapping]:
    """A plain text index (no semantic field, so no model deploy); deleted on teardown.

    Shared by the lexical searchers and by the reranker's ``mget`` doc-text lookup — neither needs
    an embedding model, so this fixture never triggers a deploy.
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


# --- semantic path (VectorSearch, live E5) ----------------------------------------------------


def test_vector_search_semantic_query_returns_scored_docs(backend: ElasticsearchBackend) -> None:
    sem_field = "sem__e5"
    mapping = IndexMapping(
        index_name=backend.index,
        search_text_field="search_text",
        sem_fields={E5_INFERENCE_ID: sem_field},
        backend_mapping={
            "properties": {
                "search_text": {"type": "text", "copy_to": [sem_field]},
                sem_field: {"type": "semantic_text", "inference_id": E5_INFERENCE_ID},
            }
        },
    )
    try:
        backend.ensure_index(mapping)  # binding the semantic_text field deploys E5
    except ApiError as exc:
        _skip_if_deploy_error(exc)
    try:
        try:
            # ES embeds each semantic_text at ingest, so a failed E5 deploy fails here as a
            # BulkIndexError with a per-item 429 (not an ApiError) — skip on that too.
            backend.bulk_index(
                [
                    Document(doc_id="p1", fields={"search_text": "a comfortable blue velvet sofa"}),
                    Document(doc_id="p2", fields={"search_text": "a wooden dining table"}),
                ],
                mapping=mapping,
            )
        except (ApiError, BulkIndexError) as exc:
            _skip_if_deploy_error(exc)

        searcher = VectorSearch(backend.client, backend.index, sem_field)
        try:
            result = searcher.search("couch for the living room", top_k=10)
        except ApiError as exc:
            _skip_if_deploy_error(exc)

        assert result, "semantic query should return at least one scored doc"
        assert all(isinstance(doc.score, float) for doc in result)
        assert {doc.doc_id for doc in result} <= {"p1", "p2"}
    finally:
        backend.client.indices.delete(index=backend.index, ignore_unavailable=True)


# --- reranker (ESReranker, live .rerank-v1) ---------------------------------------------------


def test_reranker_reorders_candidates_by_model_score(
    backend: ElasticsearchBackend, mapping: IndexMapping
) -> None:
    docs = [
        Document(doc_id="c1", fields={"search_text": "a wooden dining table"}),
        Document(doc_id="c2", fields={"search_text": "a comfortable blue velvet sofa"}),
        Document(doc_id="c3", fields={"search_text": "a stainless steel kitchen knife"}),
    ]
    backend.bulk_index(docs, mapping=mapping)

    reranker = ESReranker(backend.client, backend.index, RERANK_INFERENCE_ID, "search_text")
    # Retrieval order deliberately NOT the relevance order for the query "couch to sit on".
    candidates = [ScoredDoc("c1", 1.0), ScoredDoc("c3", 1.0), ScoredDoc("c2", 1.0)]
    try:
        result = reranker.rerank("a couch to sit on in the living room", candidates)
    except ApiError as exc:
        _skip_if_deploy_error(exc)

    assert {doc.doc_id for doc in result} == {"c1", "c2", "c3"}
    # the sofa/couch doc should rank first after reranking
    assert result[0].doc_id == "c2"
    # scores are model scores (may be negative), sorted DESC
    scores = [doc.score for doc in result]
    assert scores == sorted(scores, reverse=True)
