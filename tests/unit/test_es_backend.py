"""Offline unit tests for the ES adapter — the ES client + provider connectors are FAKES (no network).

ES is a plain vector/BM25 index (§1.1): no ``_inference``/``semantic_text``. Covers the ingest seam
(``ensure_index`` / streamed ``bulk_index``); the searchers — ``LexicalSearcher``'s ``match`` body and
``VectorSearch``'s embed-query + ``knn`` body, both over the shared ``_search``/``_msearch`` helpers'
client-side score-desc/doc_id-asc tie-break; ``ESReranker`` (``mget`` doc-text + a provider
``RerankClient`` + ``rerank_local`` reorder); ``ESIndexer.build`` (dense_vector mapping, dot-free sem
fields, embed-at-ingest); the ``make_searcher_factory`` builders; and a client-side ``HybridSearch``
== ``fuse_rrf_local`` cross-check. See docs/experiment.md §3.3-§3.7, §5.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from elasticsearch.helpers import BulkIndexError

from benchmark.backends import elasticsearch as es
from benchmark.fusion import fuse_rrf_local
from benchmark.models import (
    Document,
    FieldRole,
    FieldSchema,
    FieldSpec,
    IndexMapping,
    ScoredDoc,
)
from benchmark.pipeline import HybridSearch, RRFFuser
from benchmark.protocols import Dataset, Searcher

INDEXER_CFG = {"index": "wands_bench", "settings": {"url": "http://localhost:9200"}}


def _fake_client() -> MagicMock:
    """A MagicMock ES client with the ``indices`` sub-client tests touch."""
    client = MagicMock()
    client.indices = MagicMock()
    return client


def _backend_with(client: MagicMock) -> es.ElasticsearchBackend:
    """Build an ``ElasticsearchBackend`` without constructing a real client."""
    backend = es.ElasticsearchBackend.__new__(es.ElasticsearchBackend)
    backend.index = INDEXER_CFG["index"]
    backend.client = client
    backend.bulk_chunk_size = es._BULK_CHUNK_SIZE
    backend.embed_batch_size = es._EMBED_BATCH_SIZE
    return backend


# --- fake provider connectors -----------------------------------------------------------------


class _FakeEmbedder:
    """A fake ``Embedder``: fixed-``dim`` canned vectors; records the texts it embedded (no network).

    ``embed_queries`` returns a constant vector per input so the ``knn`` body is assertable;
    ``embed_documents`` returns a per-index vector so a doc's stored vector is identifiable.
    """

    def __init__(self, embedder_id: str, dim: int = 3, query_vector: list[float] | None = None) -> None:
        self.id = embedder_id
        self._dim = dim
        self._query_vector = query_vector if query_vector is not None else [1.0, 2.0, 3.0]
        self.doc_batches: list[list[str]] = []
        self.query_batches: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Any) -> list[list[float]]:
        self.doc_batches.append(list(texts))
        return [[float(i)] * self._dim for i, _ in enumerate(texts)]

    def embed_queries(self, texts: Any) -> list[list[float]]:
        self.query_batches.append(list(texts))
        return [list(self._query_vector) for _ in texts]


class _FakeRerankClient:
    """A fake ``RerankClient`` returning canned scores aligned to input; records its calls."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def rerank_scores(self, query: str, documents: Any) -> list[float]:
        self.calls.append((query, list(documents)))
        return list(self._scores)


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


def test_bulk_index_logs_per_item_reasons_then_reraises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # An ES write error (e.g. a dense_vector dim mismatch) surfaces as a BulkIndexError whose per-item
    # reasons live on .errors — assert they are LOGGED (not just the opaque count) and the error still
    # propagates (never swallowed).
    def fake_streaming_bulk(client: Any, actions: Any, **kwargs: Any) -> Any:
        list(actions)  # drive the lazy generator
        raise BulkIndexError(
            "1 document(s) failed to index.",
            [{"index": {"_id": "p2", "status": 400,
                        "error": {"type": "mapper_parsing_exception", "reason": "wrong vector dims"}}}],
        )
        yield  # pragma: no cover - generator marker

    monkeypatch.setattr(es, "streaming_bulk", fake_streaming_bulk)
    backend = _backend_with(_fake_client())

    with caplog.at_level("ERROR"):
        with pytest.raises(BulkIndexError):
            backend.bulk_index([Document(doc_id="p2", fields={"search_text": "x"})], mapping=_mapping())

    assert "wrong vector dims" in caplog.text  # the real reason is surfaced
    assert "p2" in caplog.text
    backend.client.indices.refresh.assert_not_called()  # re-raised before refresh


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


def test_lexical_bulk_search_chunks_and_aligns() -> None:
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


def _factory(client: MagicMock, monkeypatch: pytest.MonkeyPatch, **kw: Any) -> Any:
    monkeypatch.setattr(es, "_make_client", lambda cfg: client)
    embedders = kw.pop("embedders", {})
    rerankers = kw.pop("rerankers", {})
    return es.make_searcher_factory(INDEXER_CFG, embedders=embedders, rerankers=rerankers)


def test_factory_lexical_builds_lexical_searcher(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _factory(_fake_client(), monkeypatch)
    searcher = factory.lexical(fields=["search_text"])

    assert isinstance(searcher, es.LexicalSearcher)
    assert searcher.index == "wands_bench"
    assert searcher.field == "search_text"


def test_factory_vector_builds_vector_search_with_query_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder = _FakeEmbedder("e5")
    factory = _factory(_fake_client(), monkeypatch, embedders={"e5": embedder})

    searcher = factory.vector(field="sem__e5", embedder_id="e5")

    assert isinstance(searcher, es.VectorSearch)
    assert searcher.index == "wands_bench"
    assert searcher.field == "sem__e5"
    assert searcher.query_embedder is embedder  # the referenced connector is attached


def test_factory_reranker_builds_es_reranker_with_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rerank_client = _FakeRerankClient([1.0])
    factory = _factory(_fake_client(), monkeypatch, rerankers={"co-rr": rerank_client})

    reranker = factory.reranker("co-rr", "search_text")

    assert isinstance(reranker, es.ESReranker)
    assert reranker.index == "wands_bench"
    assert reranker.field == "search_text"
    assert reranker.rerank_client is rerank_client  # the referenced connector is attached


# --- VectorSearch (embed-query + knn) ---------------------------------------------------------


def test_vector_search_builds_knn_body() -> None:
    client = _fake_client()
    client.search.return_value = _hits(("p1", 5.0), ("p2", 4.0))
    embedder = _FakeEmbedder("e5", dim=3, query_vector=[0.1, 0.2, 0.3])

    searcher = es.VectorSearch(client, "wands_bench", "sem__e5", embedder)
    docs = searcher.search("comfy sofa", top_k=2)

    # the query was embedded CLIENT-SIDE, then a knn query was issued (not the old semantic query)
    assert embedder.query_batches == [["comfy sofa"]]
    kwargs = client.search.call_args.kwargs
    assert kwargs["index"] == "wands_bench"
    assert kwargs["knn"] == {
        "field": "sem__e5",
        "query_vector": [0.1, 0.2, 0.3],
        "k": 2,
        "num_candidates": es._KNN_NUM_CANDIDATES,  # max(top_k, default) = 100
    }
    assert kwargs["size"] == 2
    assert "sort" not in kwargs  # no server-side _id sort (§9.1)
    assert [d.doc_id for d in docs] == ["p1", "p2"]


def test_vector_search_num_candidates_floored_at_top_k() -> None:
    client = _fake_client()
    client.search.return_value = _hits(("p1", 1.0))
    embedder = _FakeEmbedder("e5")
    # top_k above the default -> num_candidates is floored at top_k.
    searcher = es.VectorSearch(client, "wands_bench", "sem__e5", embedder, num_candidates=50)
    searcher.search("q", top_k=200)
    assert client.search.call_args.kwargs["knn"]["num_candidates"] == 200


def test_vector_search_sorts_score_desc_then_doc_id_asc() -> None:
    client = _fake_client()
    client.search.return_value = _hits(("d_b", 2.0), ("d_c", 3.0), ("d_a", 2.0))

    searcher = es.VectorSearch(client, "wands_bench", "sem__e5", _FakeEmbedder("e5"))
    result = searcher.search("x", top_k=10)

    assert [(d.doc_id, d.score) for d in result] == [
        ("d_c", 3.0),
        ("d_a", 2.0),  # tie broken by doc_id asc
        ("d_b", 2.0),
    ]


def test_vector_bulk_search_embeds_all_then_msearch() -> None:
    client = _fake_client()
    embedder = _FakeEmbedder("e5", dim=3, query_vector=[0.5, 0.5, 0.5])
    searcher = es.VectorSearch(client, "wands_bench", "sem__e5", embedder, msearch_chunk_size=2)
    client.msearch.side_effect = [
        _msearch_response(_hits(("a", 1.0)), _hits(("d_b", 2.0), ("d_c", 3.0), ("d_a", 2.0))),
        _msearch_response(_hits(("z", 9.0))),
    ]

    result = searcher.bulk_search(["q0", "q1", "q2"], top_k=5)

    # all queries embedded in ONE batch (the connector batches), then chunked _msearch round trips
    assert embedder.query_batches == [["q0", "q1", "q2"]]
    assert client.msearch.call_count == 2
    first_searches = client.msearch.call_args_list[0].kwargs["searches"]
    assert client.msearch.call_args_list[0].kwargs["index"] == "wands_bench"
    assert first_searches[0] == {}  # per-search header
    assert first_searches[1] == {
        "knn": {"field": "sem__e5", "query_vector": [0.5, 0.5, 0.5], "k": 5, "num_candidates": es._KNN_NUM_CANDIDATES},
        "size": 5,
    }
    # aligned + client-side tie-break identical to LexicalSearcher
    assert [(d.doc_id, d.score) for d in result[1]] == [
        ("d_c", 3.0),
        ("d_a", 2.0),
        ("d_b", 2.0),
    ]
    assert [d.doc_id for d in result[2]] == ["z"]


# --- ESReranker (provider RerankClient) -------------------------------------------------------


def _mget_response(field: str, *pairs: tuple[str, str]) -> dict[str, Any]:
    """A canned ``mget`` response: one found doc per (id, text) pair carrying ``field``."""
    return {
        "docs": [
            {"_id": doc_id, "found": True, "_source": {field: text}} for doc_id, text in pairs
        ]
    }


def test_reranker_scores_via_connector_and_reorders_desc() -> None:
    client = _fake_client()
    # candidate INPUT order c1,c2,c3 (retrieval order) — different from the model's score order.
    candidates = [ScoredDoc("c1", 5.0), ScoredDoc("c2", 4.0), ScoredDoc("c3", 3.0)]
    client.mget.return_value = _mget_response(
        "search_text", ("c1", "text one"), ("c2", "text two"), ("c3", "text three")
    )
    # scores ALIGNED to input (c1,c2,c3); a NEGATIVE score is allowed (cross-encoder logit).
    rerank_client = _FakeRerankClient([-6.25, 2.95, 0.5])

    reranker = es.ESReranker(client, "wands_bench", "search_text", rerank_client)
    result = reranker.rerank("a query", candidates)

    # reordered by model score DESC: c2(2.95) > c3(0.5) > c1(-6.25)
    assert [d.doc_id for d in result] == ["c2", "c3", "c1"]
    assert [d.score for d in result] == [2.95, 0.5, -6.25]

    # mget fetched the candidate ids + only the rerank field
    mget_kwargs = client.mget.call_args.kwargs
    assert mget_kwargs["index"] == "wands_bench"
    assert mget_kwargs["ids"] == ["c1", "c2", "c3"]
    assert mget_kwargs["source"] == ["search_text"]

    # the connector received the query + the candidate doc-texts IN INPUT ORDER
    assert rerank_client.calls == [("a query", ["text one", "text two", "text three"])]


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
    rerank_client = _FakeRerankClient([1.0, 2.0])

    reranker = es.ESReranker(client, "wands_bench", "search_text", rerank_client)
    with pytest.raises(KeyError, match="c2"):
        reranker.rerank("a query", candidates)
    assert rerank_client.calls == []  # aborted before scoring


def test_reranker_empty_candidates_no_round_trip() -> None:
    client = _fake_client()
    rerank_client = _FakeRerankClient([])
    reranker = es.ESReranker(client, "wands_bench", "search_text", rerank_client)

    # A query with no retrieval hits -> nothing to rerank; return [] without any ES/provider call.
    assert reranker.rerank("a query", []) == []
    client.mget.assert_not_called()
    assert rerank_client.calls == []


# --- ESIndexer.build (dense_vector mapping + embed-at-ingest) ----------------------------------


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
    """A fake ``SearchBackend`` recording call order + the mapping/docs it was handed (no register)."""

    def __init__(self) -> None:
        self.index = "wands_bench"
        self.embed_batch_size = es._EMBED_BATCH_SIZE
        self.calls: list[str] = []
        self.ensured_mapping: Any = None
        self.indexed_docs: list[Document] = []

    def ensure_index(self, mapping: Any) -> None:
        self.calls.append("ensure")
        self.ensured_mapping = mapping

    def bulk_index(self, docs: Any, *, mapping: Any) -> None:
        self.calls.append("index")
        self.indexed_docs = list(docs)  # drive the streamed generator


def test_indexer_builds_dense_vector_mapping_and_embeds_at_ingest() -> None:
    backend = _RecordingBackend()
    # embedder id carries dots -> the dense_vector field name must be dot-free.
    embedders = [_FakeEmbedder("e5.small.v1", dim=4)]

    mapping = es.ESIndexer().build(_FakeDataset(), backend, embedders)

    # ensure BEFORE index (no register step — ES is a plain index)
    assert backend.calls == ["ensure", "index"]

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

    # the corpus was embedded at ingest: each indexed doc carries its dense_vector under the sem field
    assert [d.doc_id for d in backend.indexed_docs] == ["p1"]
    stored = backend.indexed_docs[0].fields[sem_field]
    assert stored == [0.0, 0.0, 0.0, 0.0]  # _FakeEmbedder's index-0 vector, dim 4
    assert backend.indexed_docs[0].fields["search_text"] == "sofa"  # original field preserved
    assert backend.ensured_mapping is mapping


def test_indexer_multiple_embedders_one_dense_vector_each() -> None:
    backend = _RecordingBackend()
    embedders = [_FakeEmbedder("e5-small", dim=3), _FakeEmbedder("elser", dim=5)]

    mapping = es.ESIndexer().build(_FakeDataset(), backend, embedders)

    props = mapping.backend_mapping["properties"]
    assert props["sem__e5_small"]["type"] == "dense_vector" and props["sem__e5_small"]["dims"] == 3
    assert props["sem__elser"]["type"] == "dense_vector" and props["sem__elser"]["dims"] == 5
    assert mapping.sem_field("elser") == "sem__elser"
    # each indexed doc carries BOTH embedders' vectors
    doc_fields = backend.indexed_docs[0].fields
    assert len(doc_fields["sem__e5_small"]) == 3
    assert len(doc_fields["sem__elser"]) == 5


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
