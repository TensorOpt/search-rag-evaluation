"""Persistent inference/result cache tests (docs/architecture.md §5.5).

The DIRECT wrapper unit tests here are the real key-correctness guards (§11): they use COUNTING
FAKES whose output is DERIVED from the keyed fields — an embedder vector from
``sha256(provider|model_id|endpoint|mode|dims|text)``, a rerank score from
``sha256(provider|model_id|endpoint|query|doc)``, a searcher list from ``query`` + ``top_k`` — so a
dropped key field changes a returned number and the key-sensitivity tests actually BITE. We do NOT
reuse the position-dependent ``test_runner.FakeEmbedder`` / input-ignoring conftest ``FakeSearcher``
(they violate the pure-function-of-input premise the cache relies on).

Float assertions use ``np.isclose``/``np.allclose`` with a NAMED tolerance (CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np
import pytest

from benchmark.common.cache import (
    CachingEmbedder,
    CachingRerankClient,
    CachingSearcher,
    DiskCache,
    embedding_key,
)
from benchmark.common.models import ScoredDoc
from benchmark.common.protocols import Embedder, RerankClient, Searcher

#: Named tolerance for exact-round-trip float assertions (CLAUDE.md — never test float ==).
ZERO_ABS_TOL = 1e-6


# --- counting fakes (output derived from the keyed fields) ------------------------------------


class CountingEmbedder(Embedder):
    """Vectors derived from provider|model_id|endpoint|mode|dims|text; ``calls`` per computed item."""

    def __init__(
        self, *, provider: str, model_id: str, endpoint: str | None, dims: int | None, dim: int = 4
    ) -> None:
        self.id = "fake"
        self.calls = 0
        self._provider, self._model_id = provider, model_id
        self._endpoint, self._dims, self._dim = endpoint, dims, dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "query")

    def _embed(self, texts: Sequence[str], mode: str) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            self.calls += 1
            material = "|".join(
                [self._provider, self._model_id, str(self._endpoint), mode, str(self._dims), text]
            )
            digest = hashlib.sha256(material.encode("utf-8")).digest()
            out.append([digest[i] / 255.0 for i in range(self._dim)])
        return out


def _make_embedder(
    cache: DiskCache,
    *,
    provider: str = "cohere",
    model_id: str = "m",
    endpoint: str | None = None,
    dims: int | None = None,
    dim: int = 4,
) -> tuple[CachingEmbedder, CountingEmbedder]:
    """A (CachingEmbedder, inner CountingEmbedder) pair sharing the SAME keyed fields."""
    inner = CountingEmbedder(provider=provider, model_id=model_id, endpoint=endpoint, dims=dims, dim=dim)
    wrapped = CachingEmbedder(
        inner, cache, provider=provider, model_id=model_id, endpoint=endpoint, dims=dims
    )
    return wrapped, inner


class CountingReranker(RerankClient):
    """Scores derived from provider|model_id|endpoint|query|doc; records the docs each call saw."""

    def __init__(self, *, provider: str, model_id: str, endpoint: str | None) -> None:
        self.seen: list[list[str]] = []
        self._provider, self._model_id, self._endpoint = provider, model_id, endpoint

    def rerank_scores(self, query: str, documents: Sequence[str]) -> list[float]:
        self.seen.append(list(documents))
        return [self.score(query, doc) for doc in documents]

    def score(self, query: str, doc: str) -> float:
        material = "|".join([self._provider, self._model_id, str(self._endpoint), query, doc])
        return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big") / 2**32


class CountingSearcher(Searcher):
    """Ranked list derived from query + top_k; records each query searched."""

    def __init__(self) -> None:
        self.searched: list[str] = []

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        return self.bulk_search([query], top_k=top_k)[0]

    def bulk_search(self, queries: Sequence[str], *, top_k: int) -> list[list[ScoredDoc]]:
        out: list[list[ScoredDoc]] = []
        for query in queries:
            self.searched.append(query)
            out.append([ScoredDoc(f"{query}#{i}", 1.0 / (i + 1)) for i in range(top_k)])
        return out


# --- DiskCache store tests ----------------------------------------------------------


def test_store_roundtrip(tmp_path) -> None:
    cache = DiskCache(str(tmp_path))
    cache.set_many({"a": [1.0, 2.0], "b": 3.5, "c": [["d1", 0.5]]})
    got = cache.get_many(["a", "b", "c"])
    assert np.allclose(got["a"], [1.0, 2.0], rtol=0.0, atol=ZERO_ABS_TOL)
    assert np.isclose(got["b"], 3.5, rtol=0.0, atol=ZERO_ABS_TOL)
    assert got["c"][0][0] == "d1"
    assert np.isclose(got["c"][0][1], 0.5, rtol=0.0, atol=ZERO_ABS_TOL)
    cache.close()


def test_store_get_many_empty_is_noop(tmp_path) -> None:
    cache = DiskCache(str(tmp_path))
    assert cache.get_many([]) == {}  # never a `WHERE k IN ()`
    cache.close()


def test_store_chunks_over_999_keys(tmp_path) -> None:
    cache = DiskCache(str(tmp_path))
    items = {f"k{i}": [float(i)] for i in range(1500)}  # > the 999 SQLite var cap and > _CHUNK_SIZE
    cache.set_many(items)
    got = cache.get_many(list(items))
    assert len(got) == 1500
    assert np.allclose(got["k1234"], [1234.0], rtol=0.0, atol=ZERO_ABS_TOL)
    cache.close()


def test_stored_value_is_byte_exact_json(tmp_path) -> None:
    """The load-bearing guarantee is byte-identical persistence: the string stored for a computed
    vector is byte-for-byte ``json.dumps(vector)`` (a str ==, NOT a float ==) — so a hit re-serves
    the exact value a miss computed (the cache-on ≡ cache-off byte-identity foundation, §2/§10)."""
    cache = DiskCache(str(tmp_path))
    wrapped, _ = _make_embedder(cache)
    vector = wrapped.embed_queries(["byte-exact"])[0]
    key = embedding_key("cohere", "m", None, "query", None, "byte-exact")
    (raw,) = cache._conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
    assert raw == json.dumps(vector)  # exact round-trip: persisted bytes == the value's JSON
    cache.close()


# --- #1 no recompute on repeat ----------------------------------------------------------------


def test_no_recompute_on_repeat(tmp_path) -> None:
    cache = DiskCache(str(tmp_path))
    wrapped, inner = _make_embedder(cache)
    first = wrapped.embed_queries(["a", "b"])
    second = wrapped.embed_queries(["a", "b"])
    assert inner.calls == 2  # each text computed ONCE, not 4
    for f, s in zip(first, second):
        assert np.allclose(f, s, rtol=0.0, atol=ZERO_ABS_TOL)
    cache.close()


# --- #2 partial-hit rerank --------------------------------------------------------------------


def test_partial_hit_rerank(tmp_path) -> None:
    cache = DiskCache(str(tmp_path))
    inner = CountingReranker(provider="cohere", model_id="rr", endpoint=None)
    wrapped = CachingRerankClient(inner, cache, provider="cohere", model_id="rr", endpoint=None)
    query = "sofa"
    pre = wrapped.rerank_scores(query, ["d1", "d3"])  # pre-populate d1, d3
    inner.seen.clear()

    scores = wrapped.rerank_scores(query, ["d1", "d2", "d3"])
    assert inner.seen == [["d2"]]  # ONLY the miss reached the inner
    assert len(scores) == 3  # aligned 1:1 to input
    assert np.isclose(scores[0], pre[0], rtol=0.0, atol=ZERO_ABS_TOL)  # d1 from cache
    assert np.isclose(scores[2], pre[1], rtol=0.0, atol=ZERO_ABS_TOL)  # d3 from cache
    assert np.isclose(scores[1], inner.score(query, "d2"), rtol=0.0, atol=ZERO_ABS_TOL)  # d2 fresh
    cache.close()


# --- #3 key sensitivity — every field ---------------------------------------------------------


def test_key_sensitivity_every_field(tmp_path) -> None:
    cache = DiskCache(str(tmp_path))
    text = "hello"
    base = dict(provider="cohere", model_id="m", endpoint=None, dims=None)

    w0, i0 = _make_embedder(cache, **base)
    v0 = w0.embed_queries([text])[0]
    assert i0.calls == 1  # cold miss

    # Vary one field at a time -> distinct key -> miss (both computed) -> different vector.
    for field, value in [
        ("provider", "voyage"),
        ("model_id", "other"),
        ("endpoint", "https://proxy"),
        ("dims", 256),
    ]:
        w, inner = _make_embedder(cache, **{**base, field: value})
        v = w.embed_queries([text])[0]
        assert inner.calls == 1, f"expected a miss when {field!r} changed"
        assert not np.allclose(v, v0, rtol=0.0, atol=ZERO_ABS_TOL), f"key ignored {field!r}"

    # mode: the SAME text embeds differently as document vs query (input_type is captured).
    w_doc, i_doc = _make_embedder(cache, **base)
    v_doc = w_doc.embed_documents([text])[0]
    assert i_doc.calls == 1  # query entry does not serve a document request
    assert not np.allclose(v_doc, v0, rtol=0.0, atol=ZERO_ABS_TOL)

    # dims 256 vs 512 are distinct keys (OpenAI truncation param).
    w256, i256 = _make_embedder(cache, **{**base, "dims": 256})
    w512, i512 = _make_embedder(cache, **{**base, "dims": 512})
    v256 = w256.embed_queries(["dimtext"])[0]
    v512 = w512.embed_queries(["dimtext"])[0]
    assert i256.calls == 1 and i512.calls == 1
    assert not np.allclose(v256, v512, rtol=0.0, atol=ZERO_ABS_TOL)
    cache.close()


# --- #4 alignment with duplicates -------------------------------------------------------------


def test_alignment_with_duplicates(tmp_path) -> None:
    cache = DiskCache(str(tmp_path))
    wrapped, inner = _make_embedder(cache)
    result = wrapped.embed_documents(["a", "a", "b"])
    assert inner.calls == 2  # inner sees a and b ONCE each (dedup)
    assert len(result) == 3
    assert np.allclose(result[0], result[1], rtol=0.0, atol=ZERO_ABS_TOL)  # positions 0,1 identical
    assert not np.allclose(result[0], result[2], rtol=0.0, atol=ZERO_ABS_TOL)
    cache.close()


# --- #5 persistence across a fresh instance ---------------------------------------------------


def test_persistence_across_instances(tmp_path) -> None:
    cache = DiskCache(str(tmp_path))
    wrapped, inner = _make_embedder(cache)
    v1 = wrapped.embed_queries(["persist"])[0]
    assert inner.calls == 1
    cache.close()

    cache2 = DiskCache(str(tmp_path))  # new instance, SAME dir (simulates a new run)
    wrapped2, inner2 = _make_embedder(cache2)
    v2 = wrapped2.embed_queries(["persist"])[0]
    assert inner2.calls == 0  # served from disk — no inner call
    assert np.allclose(v1, v2, rtol=0.0, atol=ZERO_ABS_TOL)
    cache2.close()


# --- #7 corrupt entry -> FAIL FAST (raises, never a silent miss) ------------------------------


def test_corrupt_entry_raises(tmp_path) -> None:
    cache = DiskCache(str(tmp_path))
    key = embedding_key("cohere", "m", None, "query", None, "corrupt-me")
    cache._conn.execute("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (key, "not json{{"))
    cache._conn.commit()

    # The store only ever writes valid JSON, so a corrupt value is an integrity failure -> fail fast
    # (§2/§7), never a silent miss + recompute that would hide a compromised cache behind a slowdown.
    with pytest.raises(RuntimeError, match="corrupt value"):
        cache.get_many([key])
    cache.close()


# --- #8 RateLimiter untouched on a hit (zero inner calls => zero acquire) ----------------------


def test_no_inner_call_on_hit(tmp_path) -> None:
    cache = DiskCache(str(tmp_path))
    wrapped, inner = _make_embedder(cache)
    wrapped.embed_queries(["warm"])  # populate
    inner.calls = 0
    wrapped.embed_queries(["warm"])  # a full hit
    assert inner.calls == 0  # inner (and thus RateLimiter.acquire) is never called (§8)
    cache.close()


# --- shared-leaf recompute-once ---------------------------------------------------------------


def test_s1_shared_leaf_recompute_once(tmp_path) -> None:
    cache = DiskCache(str(tmp_path))
    inner = CountingSearcher()
    searcher = CachingSearcher(inner, cache, index_version="uuid:4", identity="match:search_text")
    first = searcher.bulk_search(["chair", "table"], top_k=3)
    second = searcher.bulk_search(["chair", "table"], top_k=3)  # mimics a second variant sharing the leaf
    assert inner.searched == ["chair", "table"]  # each query searched ONCE
    for lf, ls in zip(first, second):
        assert [d.doc_id for d in lf] == [d.doc_id for d in ls]
        for a, b in zip(lf, ls):
            assert np.isclose(a.score, b.score, rtol=0.0, atol=ZERO_ABS_TOL)
    cache.close()


# --- bulk_search partial miss -----------------------------------------------------------------


def test_s2_partial_miss(tmp_path) -> None:
    cache = DiskCache(str(tmp_path))
    inner = CountingSearcher()
    searcher = CachingSearcher(inner, cache, index_version="uuid:4", identity="id")
    searcher.bulk_search(["q1", "q3"], top_k=2)  # pre-populate q1, q3
    inner.searched.clear()

    out = searcher.bulk_search(["q1", "q2", "q3"], top_k=2)
    assert inner.searched == ["q2"]  # ONLY the miss reached the inner
    assert [[d.doc_id for d in lst] for lst in out] == [
        ["q1#0", "q1#1"],
        ["q2#0", "q2#1"],
        ["q3#0", "q3#1"],
    ]  # aligned to input
    cache.close()


# --- key sensitivity — every field (the staleness guard) --------------------------------------


def test_s3_key_sensitivity(tmp_path) -> None:
    cache = DiskCache(str(tmp_path))
    query = "chair"

    def run(index_version: str, identity: str, top_k: int) -> CountingSearcher:
        inner = CountingSearcher()
        CachingSearcher(inner, cache, index_version=index_version, identity=identity).bulk_search(
            [query], top_k=top_k
        )
        return inner

    assert run("uuid:4", "match:st", 3).searched == [query]  # cold miss (baseline)
    assert run("uuid:5", "match:st", 3).searched == [query]  # new doc_count -> miss
    assert run("other:4", "match:st", 3).searched == [query]  # new uuid -> miss
    assert run("uuid:4", "knn:field", 3).searched == [query]  # new identity -> miss
    assert run("uuid:4", "match:st", 5).searched == [query]  # new top_k -> miss
    assert run("uuid:4", "match:st", 3).searched == []  # identical tuple -> HIT (no search)
    cache.close()


# --- ScoredDoc round-trip ---------------------------------------------------------------------


def test_s4_scoreddoc_roundtrip(tmp_path) -> None:
    cache = DiskCache(str(tmp_path))
    first = CachingSearcher(
        CountingSearcher(), cache, index_version="uuid:4", identity="id"
    ).bulk_search(["sofa"], top_k=4)[0]

    inner2 = CountingSearcher()
    second = CachingSearcher(
        inner2, cache, index_version="uuid:4", identity="id"
    ).bulk_search(["sofa"], top_k=4)[0]

    assert inner2.searched == []  # served from cache
    assert [d.doc_id for d in first] == [d.doc_id for d in second]
    for rank, doc in enumerate(second):  # scores survive the ScoredDoc -> [id, score] -> ScoredDoc trip
        assert doc.doc_id == f"sofa#{rank}"
        assert np.isclose(doc.score, 1.0 / (rank + 1), rtol=0.0, atol=ZERO_ABS_TOL)
    cache.close()


# --- #6 disable bypasses entirely (no wrap; embed + searcher factories) ------------------------


def test_disable_bypasses_embedder_factory() -> None:
    from benchmark.config import EmbedderCfg, Services, make_embedders

    services = Services(
        embedders={"co": EmbedderCfg("co", "cohere", {"api_key": "k", "model_id": "embed-english-v3.0"})},
        rerankers={},
        searchers={},
    )
    out = make_embedders(services, cache=None)  # no cache -> bare connector, no wrapping (no store opened)
    assert not isinstance(out["co"], CachingEmbedder)


def test_disable_bypasses_searcher_factory() -> None:
    from benchmark.common.models import IndexMapping
    from benchmark.providers.elasticsearch import LexicalSearcher, build_searchers

    indexer_cfg = {"provider": "elasticsearch", "index": "x", "settings": {"url": "http://localhost:9200"}}
    mapping = IndexMapping("x", "search_text", {}, {"properties": {}})
    # cache=None -> bare LexicalSearcher, no index-fingerprint fetch (offline: the ES client is lazy).
    out = build_searchers(indexer_cfg, mapping, [("bm25", "lexical", None)], embedders={}, cache=None)
    assert isinstance(out["bm25"], LexicalSearcher)
    assert not isinstance(out["bm25"], CachingSearcher)


# --- #9 runner plumbing / no-crash (NOT a key-correctness guard, §11) --------------------------


def test_runner_plumbing_cache_enabled(monkeypatch, tmp_path) -> None:
    """open_cache -> make_*(cache=...) -> close wires up and writes artifacts without crashing.

    Under the runner fakes the wrappers are OFF the hot path (§11 caveat) — this only proves the
    plumbing: a real sqlite cache is opened, threaded through the factories, and closed.
    """
    import dataclasses

    from benchmark.config import CacheCfg, PipelineCfg
    from benchmark.runner import ExperimentRunner
    from tests.unit.test_runner import _config, patch_runner_factories

    patch_runner_factories(monkeypatch)
    ts = "20260707T120000Z"
    cache_dir = tmp_path / ".cache"
    cfg = _config(
        variants=[PipelineCfg("semantic_e5", ("semantic_e5",), None, None, None)], timestamp=ts
    )
    cfg = dataclasses.replace(cfg, cache=CacheCfg(enabled=True, dir=str(cache_dir)))

    ExperimentRunner().run(cfg, output_dir=str(tmp_path))

    assert (tmp_path / f"metrics_{ts}.csv").exists()  # completed + wrote artifacts
    assert (cache_dir / "inference.sqlite").exists()  # cache opened (and closed) without error


# --- config surface: resolve_config parsing + open_cache (docs/architecture.md §5.5) --------

_MINIMAL_RAW = {
    "dataset": {"name": "wands", "path": "./x"},
    "indexer": {"provider": "elasticsearch", "index": "x", "settings": {"url": "http://localhost:9200"}},
    "services": [{"searcher": {"name": "bm25", "provider": "elasticsearch", "kind": "lexical"}}],
    "pipelines": {"baseline": {"retriever": "bm25"}, "variants": {}},
    "stats": {"seed": 1},
    "cutoff": 10,
    "top_k": 50,
}


def test_resolve_config_parses_cache_block() -> None:
    from benchmark.config import resolve_config

    cfg = resolve_config({**_MINIMAL_RAW, "cache": {"enabled": True, "dir": "/tmp/xyz"}})
    assert cfg.cache.enabled is True
    assert cfg.cache.dir == "/tmp/xyz"


def test_resolve_config_cache_default_off_when_absent() -> None:
    from benchmark.config import resolve_config

    cfg = resolve_config(dict(_MINIMAL_RAW))  # no `cache` block -> disabled cold run
    assert cfg.cache.enabled is False
    assert cfg.cache.dir == ".cache"


def test_open_cache_disabled_returns_none() -> None:
    from benchmark.config import CacheCfg, open_cache

    assert open_cache(CacheCfg(enabled=False)) is None  # true bypass — no file opened


def test_open_cache_enabled_opens(tmp_path) -> None:
    from benchmark.config import CacheCfg, open_cache

    cache = open_cache(CacheCfg(enabled=True, dir=str(tmp_path / ".cache")))
    assert cache is not None
    cache.close()


def test_open_cache_corrupt_db_raises(tmp_path) -> None:
    from benchmark.config import CacheCfg, open_cache

    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    # A non-sqlite file where inference.sqlite is expected -> sqlite3.DatabaseError on open. An
    # ENABLED cache that cannot open FAILS FAST (§7) — never a silent cacheless degrade.
    (cache_dir / "inference.sqlite").write_bytes(b"this is not a sqlite database file at all")
    with pytest.raises(RuntimeError, match="unusable"):
        open_cache(CacheCfg(enabled=True, dir=str(cache_dir)))
