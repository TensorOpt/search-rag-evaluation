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
    backend.bulk_chunk_size = es._BULK_CHUNK_SIZE
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


def _capture_bulk_actions(
    monkeypatch: pytest.MonkeyPatch, *, fail_on: str | None = None
) -> list[dict[str, Any]]:
    """Patch ``streaming_bulk`` to record the actions it is fed and simulate its (ok, info) yield.

    Consumes the LAZY actions iterable, appending each into ``captured`` so a test can assert the
    action shape. If ``fail_on`` names a ``_id``, that item's ``streaming_bulk`` raise is simulated
    (``raise_on_error=True`` surfaces a failed item) so the caller does not swallow it.
    """
    captured: list[dict[str, Any]] = []

    def fake_streaming_bulk(client: Any, actions: Any, **kwargs: Any) -> Any:
        assert kwargs["chunk_size"] == es._BULK_CHUNK_SIZE  # module-constant default used
        for action in actions:  # drive the LAZY generator
            captured.append(action)
            if fail_on is not None and action["_id"] == fail_on:
                raise RuntimeError(f"simulated failed item {fail_on}")
            yield (True, {"index": {"_id": action["_id"], "status": 201}})

    monkeypatch.setattr(es, "streaming_bulk", fake_streaming_bulk)
    return captured


def test_bulk_index_streams_chunked_actions_then_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_bulk_actions(monkeypatch)
    client = _fake_client()
    backend = _backend_with(client)

    docs = [
        Document(doc_id="p1", fields={"search_text": "sofa"}),
        Document(doc_id="p2", fields={"search_text": "table"}),
    ]
    backend.bulk_index(docs, mapping=_mapping())

    # one action per doc: {"_op_type": "index", "_index": idx, "_id": doc_id, "_source": fields}
    assert captured == [
        {"_op_type": "index", "_index": "wands_bench", "_id": "p1", "_source": {"search_text": "sofa"}},
        {"_op_type": "index", "_index": "wands_bench", "_id": "p2", "_source": {"search_text": "table"}},
    ]
    client.indices.refresh.assert_called_once_with(index="wands_bench")


def test_bulk_index_actions_iterable_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    # streaming_bulk receives a generator, not a materialized list (corpus must stream, not
    # accumulate — 43K/1M docs). Assert the actions arg is an iterator, not a sequence.
    seen_type: dict[str, bool] = {}

    def fake_streaming_bulk(client: Any, actions: Any, **kwargs: Any) -> Any:
        import collections.abc

        seen_type["is_iterator"] = isinstance(actions, collections.abc.Iterator)
        seen_type["is_list"] = isinstance(actions, list)
        for action in actions:
            yield (True, {"index": {"_id": action["_id"]}})

    monkeypatch.setattr(es, "streaming_bulk", fake_streaming_bulk)
    backend = _backend_with(_fake_client())
    backend.bulk_index([Document(doc_id="p1", fields={"t": "x"})], mapping=_mapping())

    assert seen_type["is_iterator"] is True
    assert seen_type["is_list"] is False


def test_bulk_index_failed_item_raises_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_bulk_actions(monkeypatch, fail_on="p2")
    client = _fake_client()
    backend = _backend_with(client)

    docs = [
        Document(doc_id="p1", fields={"search_text": "sofa"}),
        Document(doc_id="p2", fields={"search_text": "table"}),
    ]
    with pytest.raises(RuntimeError, match="simulated failed item p2"):
        backend.bulk_index(docs, mapping=_mapping())
    # a failed item aborts before refresh (the error surfaces, exception convention)
    client.indices.refresh.assert_not_called()


def test_bulk_index_no_docs_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_bulk_actions(monkeypatch)
    client = _fake_client()
    backend = _backend_with(client)

    backend.bulk_index([], mapping=_mapping())

    client.indices.refresh.assert_not_called()  # nothing indexed -> no refresh


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


# --- _msearch + LexicalSearcher.bulk_search ---------------------------------------------------


def _msearch_response(*per_search_hits: dict[str, Any]) -> dict[str, Any]:
    """A canned ``_msearch`` response wrapping one hits payload per sub-search, in order."""
    return {"responses": list(per_search_hits)}


def test_lexical_bulk_search_chunks_and_aligns(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _fake_client()
    # chunk size 2, three queries -> TWO msearch calls (chunks of [q0,q1] then [q2]).
    searcher = es.LexicalSearcher(client, "wands_bench", ["search_text"], msearch_chunk_size=2)

    # per-query canned hits; q1 is deliberately unsorted + a score TIE to prove client-side sort.
    client.msearch.side_effect = [
        _msearch_response(_hits(("a", 1.0)), _hits(("d_b", 2.0), ("d_c", 3.0), ("d_a", 2.0))),
        _msearch_response(_hits(("z", 9.0))),
    ]

    result = searcher.bulk_search(["q0", "q1", "q2"], top_k=5)

    # TWO chunked msearch round trips (not one per query, not a single call)
    assert client.msearch.call_count == 2

    # per-chunk payload: alternating {} header then the match body with size=top_k
    first_searches = client.msearch.call_args_list[0].kwargs["searches"]
    assert client.msearch.call_args_list[0].kwargs["index"] == "wands_bench"
    assert first_searches[0] == {}  # header
    assert first_searches[1] == {"query": {"match": {"search_text": "q0"}}, "size": 5}
    assert first_searches[2] == {}
    assert first_searches[3] == {"query": {"match": {"search_text": "q1"}}, "size": 5}

    # aligned list[list[ScoredDoc]] with client-side (score desc, doc_id asc) tie-break on q1
    assert [(d.doc_id, d.score) for d in result[0]] == [("a", 1.0)]
    assert [(d.doc_id, d.score) for d in result[1]] == [
        ("d_c", 3.0),
        ("d_a", 2.0),  # tie broken by doc_id asc: d_a before d_b
        ("d_b", 2.0),
    ]
    assert [(d.doc_id, d.score) for d in result[2]] == [("z", 9.0)]


def test_msearch_per_response_error_raises() -> None:
    client = _fake_client()
    client.msearch.return_value = _msearch_response(
        _hits(("a", 1.0)),
        {"error": {"type": "search_phase_execution_exception", "reason": "boom"}},
    )
    bodies = [
        {"query": {"match": {"search_text": "q0"}}, "size": 5},
        {"query": {"match": {"search_text": "q1"}}, "size": 5},
    ]
    with pytest.raises(RuntimeError, match="sub-request 1 failed"):
        es._msearch(client, "wands_bench", bodies, chunk_size=10)


def test_lexical_bulk_search_empty_queries_no_round_trip() -> None:
    client = _fake_client()
    searcher = es.LexicalSearcher(client, "wands_bench", ["search_text"])
    assert searcher.bulk_search([], top_k=5) == []
    client.msearch.assert_not_called()


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
