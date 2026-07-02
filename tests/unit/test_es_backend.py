"""Offline unit tests for the ES adapter (Phase 9) — the ES client is MOCKED (no network).

Covers the ingest seam (``register_inference``/``ensure_index``/``bulk_index``), the shared
``_search`` helper's client-side score-desc/doc_id-asc tie-break, ``LexicalSearcher``'s match body,
and the factory's Phase-10 ``vector``/``reranker`` guards. See docs/experiment.md §3.3, §3.4, §5.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from elasticsearch import NotFoundError

from benchmark.backends import elasticsearch as es
from benchmark.models import (
    Document,
    IndexMapping,
    InferenceEndpoint,
    InferenceTaskType,
)

INDEXER_CFG = {"index": "wands_bench", "settings": {"url": "http://localhost:9200"}}


def _not_found() -> NotFoundError:
    """A minimal ``NotFoundError`` for the ``inference.get`` idempotency probe."""
    return NotFoundError("not found", meta=MagicMock(status=404), body={})


def _fake_client() -> MagicMock:
    """A MagicMock ES client with the ``inference`` / ``indices`` sub-clients tests touch."""
    client = MagicMock()
    client.inference = MagicMock()
    client.indices = MagicMock()
    return client


def _backend_with(client: MagicMock) -> es.ElasticsearchBackend:
    """Build an ``ElasticsearchBackend`` without constructing a real client."""
    backend = es.ElasticsearchBackend.__new__(es.ElasticsearchBackend)
    backend.index = INDEXER_CFG["index"]
    backend.client = client
    return backend


# --- register_inference -----------------------------------------------------------------------


def test_register_inference_emits_separate_service_and_task_settings() -> None:
    client = _fake_client()
    client.inference.get.side_effect = _not_found()  # endpoint absent -> create it
    backend = _backend_with(client)

    ep = InferenceEndpoint(
        inference_id="cohere-rerank",
        task_type=InferenceTaskType.RERANK,
        service="cohere",
        service_settings={"api_key": "k", "model_id": "rerank-v3"},
        task_settings={"top_n": 100},
    )
    returned = backend.register_inference(ep)

    assert returned == "cohere-rerank"
    client.inference.put.assert_called_once()
    kwargs = client.inference.put.call_args.kwargs
    assert kwargs["task_type"] == "rerank"
    assert kwargs["inference_id"] == "cohere-rerank"
    body = kwargs["body"]
    # service / service_settings / task_settings stay SEPARATE maps (§3.4).
    assert body["service"] == "cohere"
    assert body["service_settings"] == {"api_key": "k", "model_id": "rerank-v3"}
    assert body["task_settings"] == {"top_n": 100}


def test_register_inference_omits_empty_task_settings() -> None:
    client = _fake_client()
    client.inference.get.side_effect = _not_found()
    backend = _backend_with(client)

    ep = InferenceEndpoint(
        inference_id="e5-small",
        task_type=InferenceTaskType.TEXT_EMBEDDING,
        service="elasticsearch",
        service_settings={"model_id": ".multilingual-e5-small"},
    )
    backend.register_inference(ep)

    body = client.inference.put.call_args.kwargs["body"]
    assert "task_settings" not in body  # empty task_settings omitted
    assert body["service_settings"] == {"model_id": ".multilingual-e5-small"}


def test_register_inference_idempotent_when_endpoint_exists() -> None:
    client = _fake_client()
    client.inference.get.return_value = {"endpoints": [{"inference_id": "e5-small"}]}
    backend = _backend_with(client)

    ep = InferenceEndpoint(
        inference_id="e5-small",
        task_type=InferenceTaskType.TEXT_EMBEDDING,
        service="elasticsearch",
        service_settings={"model_id": ".multilingual-e5-small"},
    )
    returned = backend.register_inference(ep)

    assert returned == "e5-small"
    client.inference.put.assert_not_called()  # existing endpoint is not recreated


# --- ensure_index -----------------------------------------------------------------------------


def _mapping() -> IndexMapping:
    return IndexMapping(
        index_name="wands_bench",
        search_text_field="search_text",
        sem_fields={},
        backend_mapping={"properties": {"search_text": {"type": "text"}}},
    )


def test_ensure_index_creates_with_backend_mapping() -> None:
    client = _fake_client()
    client.indices.exists.return_value = False
    backend = _backend_with(client)

    backend.ensure_index(_mapping())

    client.indices.create.assert_called_once()
    kwargs = client.indices.create.call_args.kwargs
    assert kwargs["index"] == "wands_bench"
    assert kwargs["mappings"] == {"properties": {"search_text": {"type": "text"}}}


def test_ensure_index_idempotent_when_index_exists() -> None:
    client = _fake_client()
    client.indices.exists.return_value = True
    backend = _backend_with(client)

    backend.ensure_index(_mapping())

    client.indices.create.assert_not_called()  # existing index -> skip, no raise


# --- bulk_index -------------------------------------------------------------------------------


def test_bulk_index_uses_doc_id_and_fields_then_refreshes() -> None:
    client = _fake_client()
    backend = _backend_with(client)

    docs = [
        Document(doc_id="p1", fields={"search_text": "sofa"}),
        Document(doc_id="p2", fields={"search_text": "table"}),
    ]
    backend.bulk_index(docs, mapping=_mapping())

    kwargs = client.bulk.call_args.kwargs
    assert kwargs["index"] == "wands_bench"
    ops = kwargs["operations"]
    assert ops[0] == {"index": {"_id": "p1"}}  # _id == doc.doc_id
    assert ops[1] == {"search_text": "sofa"}  # _source == doc.fields
    assert ops[2] == {"index": {"_id": "p2"}}
    assert ops[3] == {"search_text": "table"}
    client.indices.refresh.assert_called_once_with(index="wands_bench")


def test_bulk_index_no_docs_is_noop() -> None:
    client = _fake_client()
    backend = _backend_with(client)

    backend.bulk_index([], mapping=_mapping())

    client.bulk.assert_not_called()
    client.indices.refresh.assert_not_called()


# --- _search + LexicalSearcher ----------------------------------------------------------------


def _hits(*pairs: tuple[str, float]) -> dict[str, Any]:
    """A canned ES search response from ``(doc_id, score)`` pairs."""
    return {"hits": {"hits": [{"_id": doc_id, "_score": score} for doc_id, score in pairs]}}


def test_search_sorts_score_desc_then_doc_id_asc_client_side() -> None:
    client = _fake_client()
    # Deliberately unsorted + a score TIE (d_b and d_a both 2.0) to prove the doc_id tie-break.
    client.search.return_value = _hits(("d_b", 2.0), ("d_c", 3.0), ("d_a", 2.0))

    result = es._search(client, "wands_bench", {"query": {"match": {"search_text": "x"}}})

    assert [(d.doc_id, d.score) for d in result] == [
        ("d_c", 3.0),  # highest score first
        ("d_a", 2.0),  # tie broken by doc_id asc: d_a before d_b
        ("d_b", 2.0),
    ]


def test_lexical_searcher_builds_match_body_with_size_and_no_id_sort() -> None:
    client = _fake_client()
    client.search.return_value = _hits(("p1", 5.0), ("p2", 4.0))

    searcher = es.LexicalSearcher(client, "wands_bench", ["search_text"])
    docs = searcher.search("distinctive-token", top_k=2)

    kwargs = client.search.call_args.kwargs
    assert kwargs["index"] == "wands_bench"
    assert kwargs["query"] == {"match": {"search_text": "distinctive-token"}}
    assert kwargs["size"] == 2
    # No server-side _id sort emitted (§9.1 — ES 8.x fielddata on _id would error).
    assert "sort" not in kwargs
    assert [d.doc_id for d in docs] == ["p1", "p2"]


def test_lexical_searcher_rejects_multiple_fields() -> None:
    client = _fake_client()
    with pytest.raises(ValueError):
        es.LexicalSearcher(client, "wands_bench", ["search_text", "other"])


# --- make_searcher_factory --------------------------------------------------------------------


def test_factory_lexical_builds_lexical_searcher(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_client()
    monkeypatch.setattr(es, "_make_client", lambda cfg: fake)

    factory = es.make_searcher_factory(INDEXER_CFG)
    searcher = factory.lexical(fields=["search_text"])

    assert isinstance(searcher, es.LexicalSearcher)
    assert searcher.index == "wands_bench"
    assert searcher.field == "search_text"


def test_factory_vector_and_reranker_raise_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_client()
    monkeypatch.setattr(es, "_make_client", lambda cfg: fake)
    factory = es.make_searcher_factory(INDEXER_CFG)

    with pytest.raises(NotImplementedError, match="Phase 10"):
        factory.vector(field="sem__e5")
    with pytest.raises(NotImplementedError, match="Phase 10"):
        factory.reranker("cohere-rerank", "search_text")
