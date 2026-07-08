"""Offline unit tests for the ES adapter — the ES client + provider connectors are FAKES (no network).

ES is a plain vector/BM25 index (§1.1): no ``_inference``/``semantic_text``. Covers the ingest seam
(``ESIndexWriter``: ``ensure_index`` / streamed ``bulk_index`` + ``create_mapping``/``sem_field_name``);
the searchers — ``LexicalSearcher``'s ``match`` body and ``VectorSearch``'s embed-query + ``knn`` body,
both over the shared ``_search``/``_msearch`` helpers' client-side score-desc/doc_id-asc tie-break;
``ESReranker`` (``mget`` doc-text + a provider ``RerankClient`` + ``rerank_local`` reorder); the
``build_searchers``/``build_rerankers`` leaf builders (replacing the deleted ``_ESSearcherFactory``); and
a client-side ``HybridSearch`` == ``fuse_rrf_local`` cross-check. The domain ``Indexer`` orchestration
(embed-at-ingest, ensure→index call order) is covered in ``tests/unit/test_indexing.py``. See
docs/architecture.md §3.3-§3.7, §5.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from elasticsearch.helpers import BulkIndexError

from benchmark.common.cache import CachingEmbedder, CachingSearcher, DiskCache
from benchmark.common.models import (
    Document,
    FieldRole,
    FieldSchema,
    FieldSpec,
    IndexMapping,
    ScoredDoc,
)
from benchmark.common.protocols import Dataset, Embedder, RerankClient, Searcher
from benchmark.common.ranking import fuse_rrf_local
from benchmark.providers import elasticsearch as es
from benchmark.search import HybridSearch, RRFFuser

INDEXER_CFG = {"index": "wands_bench", "settings": {"url": "http://localhost:9200"}}


def _fake_client() -> MagicMock:
    """A MagicMock ES client with the ``indices`` sub-client tests touch."""
    client = MagicMock()
    client.indices = MagicMock()
    return client


def _writer_with(client: MagicMock) -> es.ESIndexWriter:
    """Build an ``ESIndexWriter`` without constructing a real client."""
    writer = es.ESIndexWriter.__new__(es.ESIndexWriter)
    writer.index = INDEXER_CFG["index"]
    writer.client = client
    writer.bulk_chunk_size = es._BULK_CHUNK_SIZE
    writer.embed_batch_size = es._EMBED_BATCH_SIZE
    # P1-2: BM25/analysis defaults (bypassing __init__ here) so ensure_index/create_mapping work.
    writer.bm25_k1 = es._DEFAULT_BM25_K1
    writer.bm25_b = es._DEFAULT_BM25_B
    writer.analyzer = es._DEFAULT_ANALYZER
    return writer


# --- fake provider connectors -----------------------------------------------------------------


class _FakeEmbedder(Embedder):
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


class _FakeRerankClient(RerankClient):
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
    writer = _writer_with(client)

    writer.ensure_index(_mapping())

    client.indices.create.assert_called_once()
    kwargs = client.indices.create.call_args.kwargs
    assert kwargs["index"] == "wands_bench"
    assert kwargs["mappings"] == {"properties": {"search_text": {"type": "text"}}}


def test_bm25_similarity_written_to_index() -> None:
    # P1-2: ensure_index bakes the explicit tuned BM25 similarity (k1/b) into the index settings so
    # the baseline is recorded, not ES defaults applied silently.
    client = _fake_client()
    client.indices.exists.return_value = False
    writer = _writer_with(client)  # default k1=1.2, b=0.75

    writer.ensure_index(_mapping())

    settings = client.indices.create.call_args.kwargs["settings"]
    assert settings["similarity"]["bm25_tuned"] == {"type": "BM25", "k1": 1.2, "b": 0.75}


def test_resolved_index_profile_reads_back_from_es() -> None:
    # P1-2: resolved_index_profile reads the BM25 params + analyzer BACK from _mapping/_settings
    # (never assumed). k1/b arrive as strings from ES settings -> coerced to float.
    client = _fake_client()
    client.indices.get_mapping.return_value = {
        "wands_bench": {
            "mappings": {
                "properties": {
                    "search_text": {
                        "type": "text",
                        "similarity": "bm25_tuned",
                        "analyzer": "standard",
                    },
                    "sem__e5": {"type": "dense_vector"},
                }
            }
        }
    }
    client.indices.get_settings.return_value = {
        "wands_bench": {
            "settings": {
                "index": {"similarity": {"bm25_tuned": {"type": "BM25", "k1": "1.5", "b": "0.3"}}}
            },
            "defaults": {"index": {"analysis": {}}},
        }
    }
    writer = _writer_with(client)

    profile = writer.resolved_index_profile()

    assert profile["bm25"] == {"similarity": "bm25_tuned", "k1": 1.5, "b": 0.3}
    assert profile["analysis"]["analyzer"] == "standard"
    client.indices.get_settings.assert_called_once_with(
        index="wands_bench", include_defaults=True
    )


def test_resolved_index_profile_deep_merges_index_settings() -> None:
    # SF: the k1×b sweep writes BM25 params under settings.index.similarity, while a custom analyzer's
    # tokenizer/filters live under defaults.index.analysis. A shallow {**defaults, **settings} merge
    # would drop the analysis sub-dict wholesale; the deep per-sub-key merge keeps BOTH.
    client = _fake_client()
    client.indices.get_mapping.return_value = {
        "wands_bench": {
            "mappings": {
                "properties": {
                    "search_text": {
                        "type": "text",
                        "similarity": "bm25_tuned",
                        "analyzer": "wands_analyzer",
                    }
                }
            }
        }
    }
    client.indices.get_settings.return_value = {
        "wands_bench": {
            # settings.index carries ONLY the tuned similarity (what the k1×b sweep sets)...
            "settings": {
                "index": {"similarity": {"bm25_tuned": {"type": "BM25", "k1": "2.0", "b": "0.9"}}}
            },
            # ...defaults.index carries ONLY the analysis chain — it must NOT be dropped by the merge.
            "defaults": {
                "index": {
                    "analysis": {
                        "analyzer": {
                            "wands_analyzer": {
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop"],
                            }
                        }
                    }
                }
            },
        }
    }
    writer = _writer_with(client)

    profile = writer.resolved_index_profile()

    # BM25 params come from settings.index; the analyzer's tokenizer/filters from defaults.index — the
    # deep merge preserves both (the shallow merge dropped analysis once similarity was present).
    assert profile["bm25"] == {"similarity": "bm25_tuned", "k1": 2.0, "b": 0.9}
    assert profile["analysis"] == {
        "analyzer": "wands_analyzer",
        "tokenizer": "standard",
        "filters": ["lowercase", "stop"],
    }


def test_ensure_index_idempotent_when_index_exists() -> None:
    client = _fake_client()
    client.indices.exists.return_value = True
    writer = _writer_with(client)

    writer.ensure_index(_mapping())

    client.indices.create.assert_not_called()  # existing index -> skip, no raise


def test_doc_count_returns_count_when_index_exists() -> None:
    client = _fake_client()
    client.indices.exists.return_value = True
    client.count.return_value = {"count": 42994}
    writer = _writer_with(client)

    assert writer.doc_count() == 42994
    client.count.assert_called_once_with(index="wands_bench")


def test_doc_count_none_when_index_absent() -> None:
    client = _fake_client()
    client.indices.exists.return_value = False
    writer = _writer_with(client)

    assert writer.doc_count() is None  # absent -> None (the runner treats this as "run eval:index")
    client.count.assert_not_called()


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
    writer = _writer_with(client)

    docs = [
        Document(doc_id="p1", fields={"search_text": "sofa"}),
        Document(doc_id="p2", fields={"search_text": "table"}),
    ]
    writer.bulk_index(docs, mapping=_mapping())

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
    writer = _writer_with(_fake_client())
    writer.bulk_index([Document(doc_id="p1", fields={"t": "x"})], mapping=_mapping())

    assert seen_type["is_iterator"] is True
    assert seen_type["is_list"] is False


def test_bulk_index_failed_item_raises_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_bulk_actions(monkeypatch, fail_on="p2")
    client = _fake_client()
    writer = _writer_with(client)

    docs = [
        Document(doc_id="p1", fields={"search_text": "sofa"}),
        Document(doc_id="p2", fields={"search_text": "table"}),
    ]
    with pytest.raises(RuntimeError, match="simulated failed item p2"):
        writer.bulk_index(docs, mapping=_mapping())
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
    writer = _writer_with(_fake_client())

    with caplog.at_level("ERROR"):
        with pytest.raises(BulkIndexError):
            writer.bulk_index([Document(doc_id="p2", fields={"search_text": "x"})], mapping=_mapping())

    assert "wrong vector dims" in caplog.text  # the real reason is surfaced
    assert "p2" in caplog.text
    writer.client.indices.refresh.assert_not_called()  # re-raised before refresh


def test_bulk_index_no_docs_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _capture_bulk_actions(monkeypatch)
    client = _fake_client()
    writer = _writer_with(client)

    writer.bulk_index([], mapping=_mapping())

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


# --- build_searchers / build_rerankers (replacing _ESSearcherFactory) -------------------------

#: A search-side IndexMapping the builders name fields from (search_text + one dense_vector field).
_SEARCH_MAPPING = IndexMapping(
    index_name="wands_bench",
    search_text_field="search_text",
    sem_fields={"e5": "sem__e5"},
    backend_mapping={},
)


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: MagicMock) -> None:
    """Route the builders' shared-client ``_open`` at the fake client (clear the process cache)."""
    monkeypatch.setattr(es, "_make_client", lambda cfg: client)
    es._SEARCH_CLIENTS.clear()


def test_build_searchers_lexical(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, _fake_client())
    out = es.build_searchers(INDEXER_CFG, _SEARCH_MAPPING, [("bm25", "lexical", None)], embedders={})

    searcher = out["bm25"]
    assert isinstance(searcher, es.LexicalSearcher)
    assert searcher.index == "wands_bench"
    assert searcher.field == "search_text"


def test_build_searchers_vector_attaches_query_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = _FakeEmbedder("e5")
    _patch_client(monkeypatch, _fake_client())
    out = es.build_searchers(
        INDEXER_CFG, _SEARCH_MAPPING, [("semantic_e5", "vector", "e5")], embedders={"e5": embedder}
    )

    searcher = out["semantic_e5"]
    assert isinstance(searcher, es.VectorSearch)
    assert searcher.index == "wands_bench"
    assert searcher.field == "sem__e5"  # mapping.sem_field("e5")
    assert searcher.query_embedder is embedder  # the referenced connector is attached


def test_build_searchers_unknown_kind_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, _fake_client())
    with pytest.raises(ValueError, match="unknown kind"):
        es.build_searchers(INDEXER_CFG, _SEARCH_MAPPING, [("bad", "bogus", None)], embedders={})


def test_build_searchers_cache_active_wraps_and_fingerprints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # The cache-ACTIVE path (all other build_searchers tests pass cache=None): a real DiskCache + a
    # CachingEmbedder-wrapped embedder -> both leaves wrapped in CachingSearcher, the vector leaf's
    # identity carries the embedder's cache_identity, and the index fingerprint is fetched ONCE.
    client = _fake_client()
    client.indices.get.return_value = {"wands_bench": {"settings": {"index": {"uuid": "abc123"}}}}
    client.count.return_value = {"count": 5}
    _patch_client(monkeypatch, client)

    cache = DiskCache(str(tmp_path))
    embedder = CachingEmbedder(
        _FakeEmbedder("e5"), cache,
        provider="cohere", model_id="embed-english-v3.0", endpoint=None, dims=None,
    )
    out = es.build_searchers(
        INDEXER_CFG,
        _SEARCH_MAPPING,
        [("bm25", "lexical", None), ("semantic_e5", "vector", "e5")],
        embedders={"e5": embedder},
        cache=cache,
    )

    # (a) both leaves are CachingSearcher instances
    assert isinstance(out["bm25"], CachingSearcher)
    assert isinstance(out["semantic_e5"], CachingSearcher)

    # (b) vector-leaf identity: knn:{field}:num_candidates={n}:emb={embedder.cache_identity}
    assert (
        out["semantic_e5"]._identity
        == f"knn:sem__e5:num_candidates={es._KNN_NUM_CANDIDATES}:emb={embedder.cache_identity}"
    )
    assert embedder.cache_identity in out["semantic_e5"]._identity  # folds in the embedder identity
    assert out["bm25"]._identity == "match:search_text"

    # (c) index_version == "{uuid}:{doc_count}", fetched ONCE (settings + count) — not per leaf
    assert out["bm25"]._index_version == "abc123:5"
    assert out["semantic_e5"]._index_version == "abc123:5"
    assert client.indices.get.call_count == 1
    assert client.count.call_count == 1
    cache.close()


def test_build_rerankers_binds_connector(monkeypatch: pytest.MonkeyPatch) -> None:
    rerank_client = _FakeRerankClient([1.0])
    _patch_client(monkeypatch, _fake_client())
    out = es.build_rerankers(
        INDEXER_CFG, _SEARCH_MAPPING, ["co-rr"], rerank_clients={"co-rr": rerank_client}
    )

    reranker = out["co-rr"]
    assert isinstance(reranker, es.ESReranker)
    assert reranker.index == "wands_bench"
    assert reranker.field == "search_text"  # mapping.search_text_field
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


# --- ESIndexWriter.create_mapping + sem_field_name (dense_vector mapping) ----------------------


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


def test_writer_sem_field_name_is_dot_free() -> None:
    writer = _writer_with(_fake_client())
    # embedder id carries dots -> the dense_vector field name must be dot-free.
    assert writer.sem_field_name("e5.small.v1") == "sem__e5_small_v1"
    assert "." not in writer.sem_field_name("e5.small.v1")


def test_writer_create_mapping_dense_vector_and_roles() -> None:
    writer = _writer_with(_fake_client())
    schema = _FakeDataset().field_schema()
    sem_field = writer.sem_field_name("e5.small.v1")
    sem_fields = {"e5.small.v1": sem_field}
    vector_dims = {sem_field: 4}

    mapping = writer.create_mapping(schema, sem_fields, vector_dims)

    props = mapping.backend_mapping["properties"]
    # search_text is a text field carrying the EXPLICIT tuned BM25 similarity + analyzer (P1-2) so the
    # resolved profile reads back from _mapping — NO copy_to, NO semantic_text (§5.2).
    assert props["search_text"] == {
        "type": "text",
        "similarity": "bm25_tuned",
        "analyzer": "standard",
    }
    # one dense_vector field per embedder (dims from vector_dims, cosine, indexed)
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


def test_writer_create_mapping_multiple_embedders_one_dense_vector_each() -> None:
    writer = _writer_with(_fake_client())
    schema = _FakeDataset().field_schema()
    sem_fields = {
        "e5-small": writer.sem_field_name("e5-small"),
        "elser": writer.sem_field_name("elser"),
    }
    vector_dims = {sem_fields["e5-small"]: 3, sem_fields["elser"]: 5}

    mapping = writer.create_mapping(schema, sem_fields, vector_dims)

    props = mapping.backend_mapping["properties"]
    assert props["sem__e5_small"]["type"] == "dense_vector" and props["sem__e5_small"]["dims"] == 3
    assert props["sem__elser"]["type"] == "dense_vector" and props["sem__elser"]["dims"] == 5
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
