"""Offline unit tests for the ES adapter — the ES client is MOCKED (no network).

Covers the whole ES adapter surface: the ingest seam (``register_inference``/``ensure_index``/
``bulk_index``); the searchers — ``LexicalSearcher``'s match body and ``VectorSearch``'s semantic
body, both over the shared ``_search``/``_msearch`` helpers' client-side score-desc/doc_id-asc
tie-break; ``ESReranker`` (``mget`` doc-text + ``_inference`` rerank parsed BY ``index`` +
``rerank_local`` reorder); ``ESIndexer.build`` (register→ensure→index order, ``copy_to`` + dot-free
sem fields); the ``make_searcher_factory`` builders; and a client-side ``HybridSearch`` ==
``fuse_rrf_local`` cross-check. See docs/experiment.md §3.3-§3.7, §5.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from elasticsearch import NotFoundError

from benchmark.backends import elasticsearch as es
from benchmark.fusion import fuse_rrf_local
from benchmark.models import (
    Document,
    FieldRole,
    FieldSchema,
    FieldSpec,
    IndexMapping,
    InferenceEndpoint,
    InferenceTaskType,
    ScoredDoc,
)
from benchmark.pipeline import HybridSearch, RRFFuser
from benchmark.protocols import Dataset, Searcher

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


def test_factory_vector_builds_vector_search(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_client()
    monkeypatch.setattr(es, "_make_client", lambda cfg: fake)
    factory = es.make_searcher_factory(INDEXER_CFG)

    searcher = factory.vector(field="sem__e5")

    assert isinstance(searcher, es.VectorSearch)
    assert searcher.index == "wands_bench"
    assert searcher.field == "sem__e5"


def test_factory_reranker_builds_es_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_client()
    monkeypatch.setattr(es, "_make_client", lambda cfg: fake)
    factory = es.make_searcher_factory(INDEXER_CFG)

    reranker = factory.reranker("cohere-rerank", "search_text")

    assert isinstance(reranker, es.ESReranker)
    assert reranker.index == "wands_bench"
    assert reranker.inference_id == "cohere-rerank"
    assert reranker.field == "search_text"


# --- VectorSearch -----------------------------------------------------------------------------


def test_vector_search_builds_explicit_semantic_body() -> None:
    client = _fake_client()
    client.search.return_value = _hits(("p1", 5.0), ("p2", 4.0))

    searcher = es.VectorSearch(client, "wands_bench", "sem__e5")
    docs = searcher.search("comfy sofa", top_k=2)

    kwargs = client.search.call_args.kwargs
    assert kwargs["index"] == "wands_bench"
    # explicit semantic query (version-robust ES >= 8.15) — NOT the implicit match form
    assert kwargs["query"] == {"semantic": {"field": "sem__e5", "query": "comfy sofa"}}
    assert kwargs["size"] == 2
    assert "sort" not in kwargs  # no server-side _id sort (§9.1)
    assert [d.doc_id for d in docs] == ["p1", "p2"]


def test_vector_search_sorts_score_desc_then_doc_id_asc() -> None:
    client = _fake_client()
    client.search.return_value = _hits(("d_b", 2.0), ("d_c", 3.0), ("d_a", 2.0))

    searcher = es.VectorSearch(client, "wands_bench", "sem__e5")
    result = searcher.search("x", top_k=10)

    assert [(d.doc_id, d.score) for d in result] == [
        ("d_c", 3.0),
        ("d_a", 2.0),  # tie broken by doc_id asc
        ("d_b", 2.0),
    ]


def test_vector_bulk_search_chunks_and_aligns_via_msearch() -> None:
    client = _fake_client()
    searcher = es.VectorSearch(client, "wands_bench", "sem__e5", msearch_chunk_size=2)
    client.msearch.side_effect = [
        _msearch_response(_hits(("a", 1.0)), _hits(("d_b", 2.0), ("d_c", 3.0), ("d_a", 2.0))),
        _msearch_response(_hits(("z", 9.0))),
    ]

    result = searcher.bulk_search(["q0", "q1", "q2"], top_k=5)

    # two chunked _msearch round trips (chunk size 2 over three queries)
    assert client.msearch.call_count == 2
    first_searches = client.msearch.call_args_list[0].kwargs["searches"]
    assert client.msearch.call_args_list[0].kwargs["index"] == "wands_bench"
    assert first_searches[0] == {}  # per-search header
    assert first_searches[1] == {
        "query": {"semantic": {"field": "sem__e5", "query": "q0"}},
        "size": 5,
    }
    assert first_searches[3] == {
        "query": {"semantic": {"field": "sem__e5", "query": "q1"}},
        "size": 5,
    }
    # aligned + client-side tie-break identical to LexicalSearcher
    assert [(d.doc_id, d.score) for d in result[1]] == [
        ("d_c", 3.0),
        ("d_a", 2.0),
        ("d_b", 2.0),
    ]
    assert [d.doc_id for d in result[2]] == ["z"]


# --- ESReranker -------------------------------------------------------------------------------


def _mget_response(field: str, *pairs: tuple[str, str]) -> dict[str, Any]:
    """A canned ``mget`` response: one found doc per (id, text) pair carrying ``field``."""
    return {
        "docs": [
            {"_id": doc_id, "found": True, "_source": {field: text}} for doc_id, text in pairs
        ]
    }


def test_reranker_maps_scores_by_index_and_reorders_desc() -> None:
    client = _fake_client()
    # candidate INPUT order c1,c2,c3 (retrieval order) — different from the model's score order.
    candidates = [ScoredDoc("c1", 5.0), ScoredDoc("c2", 4.0), ScoredDoc("c3", 3.0)]
    client.mget.return_value = _mget_response(
        "search_text", ("c1", "text one"), ("c2", "text two"), ("c3", "text three")
    )
    # response NOT in input order + a NEGATIVE score (cross-encoder logit); map BY "index".
    client.inference.rerank.return_value = {
        "rerank": [
            {"index": 1, "relevance_score": 2.95},  # c2 highest
            {"index": 0, "relevance_score": -6.25},  # c1 lowest (negative)
            {"index": 2, "relevance_score": 0.5},  # c3 middle
        ]
    }

    reranker = es.ESReranker(client, "wands_bench", "cohere-rerank", "search_text")
    result = reranker.rerank("a query", candidates)

    # reordered by model score DESC: c2(2.95) > c3(0.5) > c1(-6.25)
    assert [d.doc_id for d in result] == ["c2", "c3", "c1"]
    assert [d.score for d in result] == [2.95, 0.5, -6.25]

    # mget fetched the candidate ids + only the rerank field
    mget_kwargs = client.mget.call_args.kwargs
    assert mget_kwargs["index"] == "wands_bench"
    assert mget_kwargs["ids"] == ["c1", "c2", "c3"]
    assert mget_kwargs["source"] == ["search_text"]

    # the inference call received the query + the candidate doc-texts IN INPUT ORDER
    infer_kwargs = client.inference.rerank.call_args.kwargs
    assert infer_kwargs["inference_id"] == "cohere-rerank"
    assert infer_kwargs["query"] == "a query"
    assert infer_kwargs["input"] == ["text one", "text two", "text three"]


def test_reranker_missing_candidate_raises() -> None:
    client = _fake_client()
    candidates = [ScoredDoc("c1", 5.0), ScoredDoc("c2", 4.0)]
    # c2 not found in the index -> raise, not silently drop.
    client.mget.return_value = {
        "docs": [
            {"_id": "c1", "found": True, "_source": {"search_text": "text one"}},
            {"_id": "c2", "found": False},
        ]
    }

    reranker = es.ESReranker(client, "wands_bench", "cohere-rerank", "search_text")
    with pytest.raises(KeyError, match="c2"):
        reranker.rerank("a query", candidates)
    client.inference.rerank.assert_not_called()  # aborted before scoring


def test_reranker_empty_candidates_no_round_trip() -> None:
    client = _fake_client()
    reranker = es.ESReranker(client, "wands_bench", "cohere-rerank", "search_text")

    # A query with no retrieval hits -> nothing to rerank; return [] without any ES call
    # (ES mget/_inference reject an empty ids/input list with a 400).
    assert reranker.rerank("a query", []) == []
    client.mget.assert_not_called()
    client.inference.rerank.assert_not_called()


# --- ESIndexer.build --------------------------------------------------------------------------


class _FakeEmbedder:
    """Minimal ``EmbeddingModel`` descriptor for the indexer test."""

    def __init__(self, inference_id: str, model_id: str) -> None:
        self.inference_id = inference_id
        self.task_type = InferenceTaskType.TEXT_EMBEDDING
        self._model_id = model_id

    def as_endpoint(self) -> InferenceEndpoint:
        return InferenceEndpoint(
            inference_id=self.inference_id,
            task_type=self.task_type,
            service="elasticsearch",
            service_settings={"model_id": self._model_id},
        )


class _FakeDataset(Dataset):
    """A tiny in-memory ``Dataset`` with one text + one numeric + one id + one stored field."""

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


class _RecordingBackend:
    """A fake ``SearchBackend`` recording call order + the mapping it was handed."""

    def __init__(self) -> None:
        self.index = "wands_bench"
        self.calls: list[str] = []
        self.registered: list[str] = []
        self.ensured_mapping: Any = None
        self.indexed_docs: list[Document] = []

    def register_inference(self, ep: InferenceEndpoint) -> str:
        self.calls.append("register")
        self.registered.append(ep.inference_id)
        return ep.inference_id

    def ensure_index(self, mapping: Any) -> None:
        self.calls.append("ensure")
        self.ensured_mapping = mapping

    def bulk_index(self, docs: Any, *, mapping: Any) -> None:
        self.calls.append("index")
        self.indexed_docs = list(docs)  # drive the streamed generator


def test_indexer_registers_before_ensure_and_builds_mapping() -> None:
    backend = _RecordingBackend()
    # embedder id carries dots -> the sem field name must be dot-free.
    embedders = [_FakeEmbedder("e5.small.v1", ".multilingual-e5-small")]

    mapping = es.ESIndexer().build(_FakeDataset(), backend, embedders)

    # register happens for every embedder BEFORE ensure_index (a semantic field can't map first)
    assert backend.calls == ["register", "ensure", "index"]
    assert backend.registered == ["e5.small.v1"]

    props = mapping.backend_mapping["properties"]
    sem_field = "sem__e5_small_v1"  # dots -> "_", prefixed
    # copy_to lives on the SOURCE search_text field and points at the sem field (§5.2)
    assert props["search_text"]["type"] == "text"
    assert props["search_text"]["copy_to"] == [sem_field]
    assert "copy_to_source" not in props["search_text"]
    # the semantic_text field carries its inference_id explicitly and its name is dot-free
    assert props[sem_field] == {"type": "semantic_text", "inference_id": "e5.small.v1"}
    assert "." not in sem_field
    # non-text roles mapped; id/bm25/semantic_source are NOT own mapped fields
    assert props["rating"] == {"type": "float"}
    assert props["product_class"] == {"type": "keyword"}
    assert "product_id" not in props
    assert "product_name" not in props
    assert "product_description" not in props

    # sem_fields resolves via IndexMapping.sem_field(embedder_id)
    assert mapping.sem_field("e5.small.v1") == sem_field
    assert mapping.search_text_field == "search_text"
    assert mapping.index_name == "wands_bench"

    # documents() were streamed to bulk_index; the SAME mapping was handed to ensure + index
    assert [d.doc_id for d in backend.indexed_docs] == ["p1"]
    assert backend.ensured_mapping is mapping


def test_indexer_multiple_embedders_copy_to_all_sem_fields() -> None:
    backend = _RecordingBackend()
    embedders = [
        _FakeEmbedder("e5-small", ".e5"),
        _FakeEmbedder("elser", ".elser"),
    ]

    mapping = es.ESIndexer().build(_FakeDataset(), backend, embedders)

    props = mapping.backend_mapping["properties"]
    assert props["search_text"]["copy_to"] == ["sem__e5_small", "sem__elser"]
    assert props["sem__e5_small"]["inference_id"] == "e5-small"
    assert props["sem__elser"]["inference_id"] == "elser"
    assert mapping.sem_field("elser") == "sem__elser"


# --- client-side hybrid cross-check (fakes, no live models) -----------------------------------


class _FixedSearcher(Searcher):
    def __init__(self, docs: list[ScoredDoc]) -> None:
        self._docs = docs

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        return self._docs[:top_k]


def test_hybrid_over_fakes_equals_fuse_rrf_local() -> None:
    lexical_list = [ScoredDoc("d1", 5.0), ScoredDoc("d2", 4.0), ScoredDoc("d3", 3.0)]
    vector_list = [ScoredDoc("d2", 0.9), ScoredDoc("d3", 0.8), ScoredDoc("d4", 0.7)]
    window = 3
    hybrid = HybridSearch(
        retrievers=[_FixedSearcher(lexical_list), _FixedSearcher(vector_list)],
        fuser=RRFFuser(rank_constant=60),
        retrieval_window_size=window,
    )

    fused = hybrid.search("q", top_k=10)
    expected = fuse_rrf_local(
        [lexical_list, vector_list], rank_constant=60, rank_window_size=window
    )
    assert [(d.doc_id, d.score) for d in fused] == [(d.doc_id, d.score) for d in expected]
