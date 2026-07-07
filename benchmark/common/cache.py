"""Persistent inference/result cache — sqlite KV store + airtight key builders + Decorators.

docs/caching_design.md. A **pure-function cache** (`key -> value`, where the key captures EVERY
value-determining input, §2/§6) shared by three Decorators, each explicitly subclassing the
``common.protocols`` seam it fulfills (CLAUDE.md "declare the interface"):

- :class:`CachingEmbedder` (:class:`~benchmark.common.protocols.Embedder`) — memoize query/doc vectors.
- :class:`CachingRerankClient` (:class:`~benchmark.common.protocols.RerankClient`) — memoize per
  ``(query, doc-text)`` rerank scores (pointwise, §5/§10).
- :class:`CachingSearcher` (:class:`~benchmark.common.protocols.Searcher`) — memoize ranked lists per
  ``(index_version, identity, top_k, query)``; backend-agnostic (the two strings are opaque, §4).

All three share one :class:`DiskCache` (a single sqlite file, JSON values). A miss recomputes; a hit
never calls the inner connector (so the connector's ``RateLimiter`` budget is untouched, §8). A
corrupt/undecodable entry is a fail-safe MISS (recomputed, never served, never crashes, §2/§7).

Cross-cutting infra like :mod:`benchmark.common.logging_setup`: imports only **stdlib**
(:mod:`sqlite3`/:mod:`json`/:mod:`hashlib`/:mod:`pathlib`) + the ``common`` seams/models — never a
provider/backend/dataset — so it stays in ``common`` (the bottom layer) and adds no forbidden
import edge (§5/§11, enforced by ``tests/unit/test_import_graph.py``).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar

from benchmark.common.logging_setup import get_logger
from benchmark.common.models import ScoredDoc
from benchmark.common.protocols import Embedder, RerankClient, Searcher

logger = get_logger(__name__)

#: Unit-separator byte — cannot appear in text, so hashed fields cannot bleed into each other (§6).
_SEP = "\x1f"

#: SQLite params-per-statement cap is 999; 500 leaves headroom.
#: ponytail: 500 < SQLite var limit (999); raise only if profiled.
_CHUNK_SIZE = 500


# --- cache-key derivation (airtight, §6) -------------------------------------------------------


def _digest(parts: Sequence[str]) -> str:
    """SHA-256 hex of ``parts`` joined by the unit separator (64-hex, bounded, §6)."""
    return hashlib.sha256(_SEP.join(parts).encode("utf-8")).hexdigest()


def embedding_key(
    provider: str, model_id: str, endpoint: str | None, mode: str, dims: int | None, text: str
) -> str:
    """Key for one text's embedding (§6).

    ``endpoint`` is the resolved ``base_url`` (proxy / Azure deployment / region); ``None`` -> the
    sentinel ``"default"`` (unambiguous given ``provider`` is in the key). ``mode`` is
    ``"document"``/``"query"`` (Cohere/Voyage ``input_type`` differs, so the SAME text embeds
    differently as doc vs query). ``dims`` is OpenAI's truncation param (``None`` when unset).
    """
    return _digest(
        ["embed", provider, model_id, endpoint or "default", mode,
         "none" if dims is None else str(dims), text]
    )


def rerank_key(
    provider: str, model_id: str, endpoint: str | None, query: str, doc_text: str
) -> str:
    """Key for one ``(query, doc-text)`` rerank score — one entry per pair so partial hits across
    overlapping candidate sets are reused (§6)."""
    return _digest(["rerank", provider, model_id, endpoint or "default", query, doc_text])


def search_key(index_version: str, identity: str, top_k: int, query: str) -> str:
    """Key for one query's ranked list (§6).

    ``index_version`` is the index fingerprint ``"uuid:doc_count"`` (what makes a result list valid);
    ``identity`` is the opaque per-leaf id (``"match:field"`` | ``"knn:field:num_candidates=N:emb=..."``);
    ``top_k`` is in the key because a list retrieved at one depth cannot serve another.
    """
    return _digest(["search", index_version, identity, str(top_k), query])


# --- disk store — sqlite (stdlib), JSON values (§7) --------------------------------------------


def _chunks(seq: Sequence[str], n: int) -> Iterator[Sequence[str]]:
    """Yield ``seq`` in slices of ``n``; range-based so ``_chunks([], n)`` yields nothing (no ``IN ()``)."""
    for start in range(0, len(seq), n):
        yield seq[start : start + n]


class DiskCache:  # NOT a Protocol — exactly one implementation (YAGNI, §7)
    """A sqlite key/value store: ``sha256-hex -> json``, WAL, one file (embed/rerank/search share it).

    Single-threaded serial harness -> one connection, a transaction per ``set_many`` batch (§7).
    """

    def __init__(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)  # missing dir -> create it (§7, not an error)
        db_path = path / "inference.sqlite"
        if not db_path.exists():
            logger.info("cache: created %s", db_path)
        # sqlite3.connect succeeds even on a non-DB file; the DatabaseError surfaces on the FIRST
        # statement below (WITHOUT ROWID create / WAL pragma). Set up on a LOCAL handle and adopt it
        # only after success, so a corrupt DB closes the fd here (not leaked out to open_cache, §7).
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL) WITHOUT ROWID"
            )
            conn.execute("PRAGMA journal_mode=WAL")  # atomic commit, crash-safe across a long run
            conn.execute("PRAGMA synchronous=NORMAL")  # fast + safe under WAL
        except sqlite3.DatabaseError:
            conn.close()  # no fd leak on the corrupt-DB path (config.open_cache re-catches + returns None)
            raise
        self._conn = conn

    def get_many(self, keys: Sequence[str]) -> dict[str, Any]:
        """Return ``{key: value}`` for the keys present + decodable; absent/corrupt keys are misses.

        ``get_many([])`` -> ``{}`` (no chunks, no query — never a ``WHERE k IN ()``). A single
        undecodable value is a fail-safe miss (debug-logged, recomputed by the caller), §2/§7.
        """
        out: dict[str, Any] = {}
        for chunk in _chunks(keys, _CHUNK_SIZE):
            placeholders = ",".join("?" * len(chunk))
            for k, v in self._conn.execute(
                f"SELECT k, v FROM kv WHERE k IN ({placeholders})", chunk
            ):
                try:
                    out[k] = json.loads(v)
                except json.JSONDecodeError:
                    logger.debug("cache: corrupt value for key %s; treating as miss", k)
        return out

    def set_many(self, items: Mapping[str, Any]) -> None:
        """Store ``items`` (json-encoded) in ONE transaction; ``INSERT OR REPLACE`` overwrites a
        prior (incl. a corrupt) row for the same key (§7)."""
        with self._conn:  # atomic under WAL
            self._conn.executemany(
                "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
                [(k, json.dumps(v)) for k, v in items.items()],
            )

    def close(self) -> None:
        self._conn.close()


# --- shared serve-cached helper (§7) -----------------------------------------------------------

_Input = TypeVar("_Input")
_Value = TypeVar("_Value")


def _serve_cached(
    cache: DiskCache,
    keys: Sequence[str],
    inputs: Sequence[_Input],
    compute: Callable[[list[_Input]], list[_Value]],
) -> list[_Value]:
    """``keys[i]`` is the cache key for ``inputs[i]``. Compute ONLY unique misses, store them, and
    return the values ALIGNED to ``keys`` (the one correctness-critical path, shared by every wrapper).
    """
    if not keys:
        return []  # safe no-op (also short-circuited by the callers' empty-input guards)
    cached = cache.get_many(keys)  # {key: value} for hits only
    miss_input_by_key = {k: x for k, x in zip(keys, inputs) if k not in cached}  # dedup dup inputs
    if miss_input_by_key:
        miss_keys = list(miss_input_by_key)
        values = compute([miss_input_by_key[k] for k in miss_keys])  # ONLY misses hit the provider
        fresh = dict(zip(miss_keys, values))
        cache.set_many(fresh)  # one transaction
        cached = {**cached, **fresh}
    return [cached[k] for k in keys]  # ALIGNED to input order


# --- the three Decorators (§4/§5) --------------------------------------------------------------


class CachingEmbedder(Embedder):
    """Decorator: serve cached vectors, batch ONLY the misses to the inner connector (§5)."""

    def __init__(
        self,
        inner: Embedder,
        cache: DiskCache,
        *,
        provider: str,
        model_id: str,
        endpoint: str | None,
        dims: int | None,
    ) -> None:
        self._inner, self._cache = inner, cache
        self._provider, self._model_id = provider, model_id
        self._endpoint, self._dims = endpoint, dims
        self.id = inner.id  # sem-field naming key (§3.5) — delegate
        #: Public leaf-identity fragment the ES build_searchers folds into a knn searcher's identity
        #: (§4/§5); includes dims (safe over-keying) so an OpenAI `dimensions` change can't serve a
        #: stale searcher list.
        self.cache_identity = (
            f"{provider}/{model_id}@{endpoint or 'default'}:"
            f"dims={dims if dims is not None else 'none'}"
        )

    @property
    def dim(self) -> int:
        return self._inner.dim  # delegate (probe/settings.dims live in the connector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, mode="document")

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, mode="query")

    def _embed(self, texts: Sequence[str], *, mode: str) -> list[list[float]]:
        if not texts:
            return []  # symmetry with the rerank empty-guard
        keys = [
            embedding_key(self._provider, self._model_id, self._endpoint, mode, self._dims, text)
            for text in texts
        ]
        compute = self._inner.embed_documents if mode == "document" else self._inner.embed_queries
        return _serve_cached(self._cache, keys, texts, compute)


class CachingRerankClient(RerankClient):
    """Decorator: cache per ``(query, doc-text)``; send ONLY missing docs to the inner (§5)."""

    def __init__(
        self, inner: RerankClient, cache: DiskCache, *, provider: str, model_id: str, endpoint: str | None
    ) -> None:
        self._inner, self._cache = inner, cache
        self._provider, self._model_id, self._endpoint = provider, model_id, endpoint

    def rerank_scores(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        keys = [
            rerank_key(self._provider, self._model_id, self._endpoint, query, doc)
            for doc in documents
        ]
        # ponytail: pointwise-scoring assumption (Cohere/Voyage cross-encoders) — a future LISTWISE
        # reranker would break per-(query, doc) caching and must gate this off (§5/§10/§13).
        return _serve_cached(
            self._cache, keys, documents, lambda docs: self._inner.rerank_scores(query, docs)
        )  # scores ALIGNED 1:1 to documents (§3.4 contract)


class CachingSearcher(Searcher):
    """Decorator over a leaf ``Searcher``: memoize ranked lists per ``(index_version, identity,
    top_k, query)``. Backend-agnostic — ``index_version`` (index fingerprint) and ``identity``
    (leaf identity) are opaque strings the backend supplies (§4)."""

    def __init__(
        self, inner: Searcher, cache: DiskCache, *, index_version: str, identity: str
    ) -> None:
        self._inner, self._cache = inner, cache
        self._index_version, self._identity = index_version, identity

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        return self.bulk_search([query], top_k=top_k)[0]  # single-query case (DRY)

    def bulk_search(self, queries: Sequence[str], *, top_k: int) -> list[list[ScoredDoc]]:
        if not queries:
            return []
        keys = [search_key(self._index_version, self._identity, top_k, q) for q in queries]

        def compute(miss_queries: list[str]) -> list[list[list[Any]]]:  # ONLY missing queries hit ES
            lists = self._inner.bulk_search(miss_queries, top_k=top_k)
            return [[[d.doc_id, d.score] for d in lst] for lst in lists]  # -> JSON-native

        encoded = _serve_cached(self._cache, keys, queries, compute)  # aligned JSON-native
        return [[ScoredDoc(doc_id, score) for doc_id, score in lst] for lst in encoded]
