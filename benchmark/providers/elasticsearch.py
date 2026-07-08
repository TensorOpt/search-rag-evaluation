"""ES adapter: LexicalSearcher, VectorSearch, ESReranker, ESIndexWriter + build_searchers/build_rerankers.

docs/architecture.md §3.3, §3.4, §3.5, §5. ES is a **plain vector/BM25 index** (§1.1) — it is NOT an
inference gateway. The harness computes embeddings via the provider connectors
(``benchmark.providers.inference``, §3.4) and stores them in ``dense_vector`` fields; there is no
``_inference`` endpoint, no ``semantic_text`` field, and no ``register_inference``.

- Ingest (:class:`ESIndexWriter`, the ``IndexWriter`` seam the domain ``Indexer`` delegates to): names
  the ``dense_vector`` field per embedder, builds the §5.2 mapping, creates the index, and streams
  already-embedded documents via ``helpers.streaming_bulk``.
- Lexical retrieval (:class:`LexicalSearcher`): a ``match`` query on ``search_text``.
- Vector retrieval (:class:`VectorSearch`): embed the query with the embedder, then an ES ``knn``
  query over that embedder's ``dense_vector`` field — batched via the shared ``_msearch``.
- Rerank (:class:`ESReranker`): fetch candidate doc-text (``mget``) and score it with a provider
  :class:`~benchmark.common.protocols.RerankClient`, reordering client-side via ``rerank_local``.
- Leaf builders (:func:`build_searchers` / :func:`build_rerankers`): mint the full configured set of
  leaf ``Searcher``s / ``Reranker``s over ONE shared ES client (§4), replacing the deleted
  ``_ESSearcherFactory``.

Fusion stays client-side (``RRFFuser``) — no server-side ``rrf``. The ES client
(``elasticsearch>=8.15,<9``) is pinned, so its API is called DIRECTLY — no ``getattr``/``hasattr``
feature probing (CLAUDE.md move-with-certainty).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator, Mapping, Sequence, cast

from elasticsearch import Elasticsearch
from elasticsearch.helpers import BulkIndexError, streaming_bulk

from benchmark.common.cache import CachingEmbedder, CachingSearcher, DiskCache
from benchmark.common.logging_setup import get_logger
from benchmark.common.models import (
    Document,
    FieldRole,
    FieldSchema,
    IndexMapping,
    ScoredDoc,
)
from benchmark.common.protocols import Embedder, IndexWriter, Reranker, RerankClient, Searcher
from benchmark.common.ranking import rerank_local

logger = get_logger(__name__)

#: Default per-request timeout (seconds) for the ES client. Ingest bulk calls can be slow.
_DEFAULT_REQUEST_TIMEOUT = 60

#: Default docs per ``streaming_bulk`` chunk (the ES helpers default). Overridable via
#: ``indexer.settings.bulk_chunk_size`` for very large corpora (WANDS ~43K, ESCI ~1M docs).
_BULK_CHUNK_SIZE = 500

#: Default docs embedded per provider call at ingest (safe for Cohere's 96-text cap). Overridable via
#: ``indexer.settings.embed_batch_size``. The connectors sub-chunk to their own per-provider limit,
#: so this is only the ingest buffering granularity (kept modest so ingest stays lazy, §3.5).
_EMBED_BATCH_SIZE = 96

#: Default per-search count per Multi-Search (``_msearch``) request. The query set is chunked
#: into groups of this size so ~480 (WANDS) / ~48K (ESCI) queries take few round trips, not one
#: per query. Overridable via ``indexer.settings.msearch_chunk_size``.
_MSEARCH_CHUNK_SIZE = 100

#: Default ``num_candidates`` for a ``knn`` query — the per-shard candidate pool ANN explores before
#: returning ``k``. Larger = more accurate/slower. Floored at ``top_k``. Overridable via
#: ``indexer.settings.knn_num_candidates``.
_KNN_NUM_CANDIDATES = 100

#: Log an ingest progress line every this many successfully-indexed docs.
_BULK_PROGRESS_EVERY = 10_000

#: Name of the named BM25 similarity the ``search_text`` field carries (§5, P1-2). Set EXPLICITLY on
#: the field + defined in index settings so the resolved ``k1``/``b`` read back from ``_settings``.
_BM25_SIMILARITY = "bm25_tuned"

#: ES BM25 defaults (``k1``/``b``) — the STANDARD run keeps these (no tuning, decision 3); recorded
#: resolved from the index either way. The default analyzer is the built-in ``standard``.
_DEFAULT_BM25_K1 = 1.2
_DEFAULT_BM25_B = 0.75
_DEFAULT_ANALYZER = "standard"


def _make_client(indexer_cfg: Mapping[str, Any]) -> Elasticsearch:
    """Build an :class:`Elasticsearch` client from ``indexer.settings.url`` (§10).

    ``request_timeout`` (``indexer.settings.request_timeout``) bounds each request so an
    unresponsive ES fails fast with a ``ConnectionTimeout`` rather than hanging silently — keep it
    modest (a healthy localhost bulk is sub-second); a very large value just turns a real problem
    (e.g. a broken index) into a long silent stall.
    """
    settings = indexer_cfg["settings"]
    url = settings["url"]
    request_timeout = int(settings.get("request_timeout", _DEFAULT_REQUEST_TIMEOUT))
    return Elasticsearch(url, request_timeout=request_timeout)


def _hits_to_scored(hits: Sequence[Mapping[str, Any]]) -> list[ScoredDoc]:
    """Map ES ``hits`` -> ``ScoredDoc`` sorted CLIENT-SIDE by (score desc, doc_id asc) (§9.1).

    ES 8.x disallows fielddata on ``_id`` so a server-side ``_id`` sort errors; the deterministic
    tie-break lives here instead. Shared by :func:`_search` and :func:`_msearch` (lexical + knn).
    """
    scored = [ScoredDoc(doc_id=hit["_id"], score=float(hit["_score"])) for hit in hits]
    scored.sort(key=lambda doc: (-doc.score, doc.doc_id))
    return scored


def _search(client: Elasticsearch, index: str, body: Mapping[str, Any]) -> list[ScoredDoc]:
    """Run a single search ``body`` against ``index`` and map hits -> ``ScoredDoc`` (§3.3, §9.1).

    Shared by ``LexicalSearcher`` (``match``) and ``VectorSearch`` (``knn``). See
    :func:`_hits_to_scored` for the tie-break.
    """
    response = client.search(index=index, **body)
    return _hits_to_scored(response["hits"]["hits"])


def _msearch(
    client: Elasticsearch,
    index: str,
    bodies: Sequence[Mapping[str, Any]],
    *,
    chunk_size: int,
) -> list[list[ScoredDoc]]:
    """Run ``bodies`` via the ES Multi-Search API, ALIGNED to ``bodies`` by index (§5.3).

    Chunks ``bodies`` into groups of ``chunk_size`` and issues one ``_msearch`` per chunk (far
    fewer round trips than one search per query). Each per-search entry is an empty header ``{}``
    followed by its body (the alternating Multi-Search payload). Responses are parsed IN ORDER;
    a per-search response carrying an ``"error"`` key RAISES (never silently returns empty — the
    exception convention). Shared by ``LexicalSearcher`` (``match``) and ``VectorSearch`` (``knn``).
    """
    results: list[list[ScoredDoc]] = []
    for start in range(0, len(bodies), chunk_size):
        chunk = bodies[start : start + chunk_size]
        searches: list[Mapping[str, Any]] = []
        for body in chunk:
            searches.append({})  # per-search header (defaults; index is passed to msearch)
            searches.append(body)
        response = client.msearch(index=index, searches=searches)
        responses = response["responses"]
        for offset, per_search in enumerate(responses):
            if "error" in per_search:
                raise RuntimeError(
                    f"_msearch sub-request {start + offset} failed: {per_search['error']}"
                )
            results.append(_hits_to_scored(per_search["hits"]["hits"]))
    return results


class ESIndexWriter(IndexWriter):
    """The ES ``IndexWriter`` seam the domain ``Indexer`` delegates to (§3.3/§3.5).

    Names the ``dense_vector`` field per embedder, builds the §5.2 ``IndexMapping``, creates the
    index, and bulk-indexes documents. ES is a plain index writer — the harness embeds the corpus
    client-side (:class:`~benchmark.common.protocols.Embedder`) and hands documents whose field bag
    already carries the ``dense_vector`` values; no inference runs server-side. Both ingest methods
    are idempotent so a re-run over an existing index is a no-op.
    """

    def __init__(self, indexer_cfg: Mapping[str, Any]) -> None:
        self.index: str = indexer_cfg["index"]
        self.client: Elasticsearch = _make_client(indexer_cfg)
        settings = indexer_cfg["settings"]
        self.bulk_chunk_size: int = int(settings.get("bulk_chunk_size", _BULK_CHUNK_SIZE))
        self.embed_batch_size: int = int(settings.get("embed_batch_size", _EMBED_BATCH_SIZE))
        # P1-2: explicit BM25 similarity (k1/b) + analyzer, baked into the index at build so the
        # baseline is recorded, not ES defaults applied silently. Absent -> the ES defaults, set
        # EXPLICITLY so they read back from _settings/_mapping (SF-2).
        bm25 = settings.get("bm25") or {}
        self.bm25_k1: float = float(bm25.get("k1", _DEFAULT_BM25_K1))
        self.bm25_b: float = float(bm25.get("b", _DEFAULT_BM25_B))
        analysis = settings.get("analysis") or {}
        self.analyzer: str = str(analysis.get("analyzer", _DEFAULT_ANALYZER))

    def sem_field_name(self, embedder_id: str) -> str:
        """The backend-safe ``dense_vector`` field name for ``embedder_id`` (§5.2)."""
        return _sem_field_name(embedder_id)

    def create_mapping(
        self,
        schema: FieldSchema,
        sem_fields: Mapping[str, str],
        vector_dims: Mapping[str, int],
    ) -> IndexMapping:
        """Translate ``schema`` (+ per-field vector dims) into the §5.2 ``IndexMapping``.

        ``sem_fields`` maps embedder id -> its ``dense_vector`` field name; ``vector_dims`` maps that
        field name -> the embedder's output dim. Produces the same ``{"properties": {...}}`` body the
        former free ``_schema_to_mapping`` built, wrapped in an ``IndexMapping`` naming this index.
        """
        backend_mapping = _schema_to_mapping(
            schema, vector_dims, similarity=_BM25_SIMILARITY, analyzer=self.analyzer
        )
        return IndexMapping(
            index_name=self.index,
            search_text_field=schema.search_text_field,
            sem_fields=dict(sem_fields),
            backend_mapping=backend_mapping,
        )

    def _index_settings(self) -> dict[str, Any]:
        """Index settings baking the explicit BM25 similarity (k1/b) into the index (§5, P1-2).

        The tuned similarity is referenced by name on the ``search_text`` field (``_schema_to_mapping``)
        and defined here, so its ``k1``/``b`` read back from ``_settings``. The analyzer is the
        built-in referenced on the field (no custom ``analysis`` block needed for a built-in).
        """
        return {
            "similarity": {
                _BM25_SIMILARITY: {"type": "BM25", "k1": self.bm25_k1, "b": self.bm25_b}
            }
        }

    def ensure_index(self, mapping: IndexMapping) -> None:
        """Create ``mapping.index_name`` with the §5.2 mappings + BM25 settings; idempotent skip."""
        if self.client.indices.exists(index=mapping.index_name):
            logger.info("index %r already exists; skipping create", mapping.index_name)
            return
        logger.info("creating index %r", mapping.index_name)
        self.client.indices.create(
            index=mapping.index_name,
            mappings=mapping.backend_mapping,
            settings=self._index_settings(),
        )

    def doc_count(self) -> int | None:
        """Number of docs currently in the index, or ``None`` if the index does not exist (§6).

        The runner calls this to verify a fully-built index before an eval (``eval:run`` does not
        index). Uses the searchable count (post-refresh) — ``bulk_index`` refreshes at the end, so a
        completed ``eval:index`` reports the full corpus; an interrupted one reports fewer.
        """
        if not self.client.indices.exists(index=self.index):
            return None
        return int(self.client.count(index=self.index)["count"])

    def resolved_index_profile(self) -> dict[str, Any]:
        """The BM25 similarity + analysis chain RESOLVED FROM the live index (§5, P1-2).

        Reads ``_mapping`` (the ``search_text`` field's ``similarity``/``analyzer`` names) and
        ``_settings?include_defaults=true`` (the named similarity's ``k1``/``b`` — falling back to the
        ES BM25 defaults when the field uses the built-in ``BM25`` similarity; and any custom analyzer
        definition). Never assumed — read back so the manifest records what the index actually scores
        with. The single ``text`` field is ``search_text`` (our schema has exactly one).
        """
        mapping = self.client.indices.get_mapping(index=self.index)[self.index]["mappings"]
        props = mapping.get("properties", {})
        text_field = next(
            (name for name, spec in props.items() if spec.get("type") == "text"), None
        )
        field_spec = props.get(text_field, {}) if text_field is not None else {}
        similarity_name = field_spec.get("similarity", "BM25")
        analyzer_name = field_spec.get("analyzer", _DEFAULT_ANALYZER)

        raw = self.client.indices.get_settings(index=self.index, include_defaults=True)[self.index]
        index_settings = {**raw.get("defaults", {}), **raw.get("settings", {})}.get("index", {})
        sim_def = index_settings.get("similarity", {}).get(similarity_name, {})
        analysis = index_settings.get("analysis", {})
        analyzer_def = analysis.get("analyzer", {}).get(analyzer_name, {})
        return {
            "bm25": {
                "similarity": similarity_name,
                "k1": float(sim_def.get("k1", _DEFAULT_BM25_K1)),
                "b": float(sim_def.get("b", _DEFAULT_BM25_B)),
            },
            "analysis": {
                "analyzer": analyzer_name,
                "tokenizer": analyzer_def.get("tokenizer"),
                "filters": list(analyzer_def.get("filter", [])),
            },
        }

    def bulk_index(self, docs: Iterable[Document], *, mapping: IndexMapping) -> None:
        """Stream ``docs`` into ``mapping.index_name`` (``_id = doc.doc_id``) in chunks, then refresh (§3.5).

        Uses :func:`elasticsearch.helpers.streaming_bulk` over a LAZY generator so the corpus never
        materializes in memory — required for 43K (WANDS) / 1M (ESCI) doc corpora. Each doc's
        ``fields`` bag is the ``_source`` (a real ``_source`` key so a field named like a bulk meta
        key cannot collide); the index op upserts by ``_id`` so a re-run is idempotent.
        ``raise_on_error=True`` surfaces any failed item as a ``BulkIndexError`` (never swallowed —
        the exception convention). The index is refreshed ONCE at the end so docs are searchable.

        Embedding happens UPSTREAM of this call (the caller streams already-embedded documents, §3.5),
        so a provider failure surfaces as a ``ProviderError`` while this generator is consumed; a
        ``BulkIndexError`` here is an ES write error (e.g. a vector dim not matching the mapping).
        """
        index_name = mapping.index_name

        def actions() -> Iterator[dict[str, Any]]:
            for doc in docs:
                yield {
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": doc.doc_id,
                    "_source": dict(doc.fields),
                }

        indexed = 0
        try:
            for _ok, _info in streaming_bulk(
                self.client,
                actions(),
                chunk_size=self.bulk_chunk_size,
                raise_on_error=True,
            ):
                indexed += 1
                if indexed % _BULK_PROGRESS_EVERY == 0:
                    logger.info("bulk_index: %d docs into %r so far", indexed, index_name)
        except BulkIndexError as exc:
            # The per-item failure reasons live on ``exc.errors``, NOT in the message (which is just a
            # count). Log the first few with their status so the cause (e.g. a dense_vector dim
            # mismatch, a mapping conflict) is visible, then re-raise (never swallowed).
            for item in exc.errors[:3]:
                op = next(iter(item.values()))  # {"_id", "status", "error": {...}} under the op key
                logger.error(
                    "bulk_index: doc %r failed (status %s): %s",
                    op.get("_id"), op.get("status"), op.get("error"),
                )
            logger.error(
                "bulk_index: %d docs failed indexing into %r (see reasons above)",
                len(exc.errors), index_name,
            )
            raise

        if indexed == 0:
            logger.info("bulk_index: no documents to index into %r", index_name)
            return

        logger.info("bulk indexed %d docs into %r; refreshing", indexed, index_name)
        self.client.indices.refresh(index=index_name)


class LexicalSearcher(Searcher):
    """BM25 leaf ``Searcher``: a single ``match`` query on ``search_text`` (§3.3, §5.3)."""

    def __init__(
        self,
        client: Elasticsearch,
        index: str,
        fields: Sequence[str],
        *,
        msearch_chunk_size: int = _MSEARCH_CHUNK_SIZE,
    ) -> None:
        if len(fields) != 1:
            raise ValueError(
                f"LexicalSearcher expects exactly one search field, got {list(fields)}"
            )
        self.client = client
        self.index = index
        self.field = fields[0]
        self._msearch_chunk_size = msearch_chunk_size

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        """Run ``{"query": {"match": {field: query}}, "size": top_k}``; ≤ ``top_k`` docs (§5.3)."""
        body = {"query": {"match": {self.field: query}}, "size": top_k}
        return _search(self.client, self.index, body)

    def bulk_search(self, queries: Sequence[str], *, top_k: int) -> list[list[ScoredDoc]]:
        """Batch all ``queries`` via one chunked ES Multi-Search (``_msearch``); ALIGNED by index (§5.3).

        Overrides the ``Searcher`` default (per-query loop) so the whole QRel query set costs a few
        ``_msearch`` round trips instead of one per query. Each body is the same ``match`` query as
        :meth:`search`. See :func:`_msearch` for chunking + the client-side tie-break.
        """
        bodies = [{"query": {"match": {self.field: query}}, "size": top_k} for query in queries]
        return _msearch(self.client, self.index, bodies, chunk_size=self._msearch_chunk_size)


class VectorSearch(Searcher):
    """Semantic leaf ``Searcher``: embed the query, then an ES ``knn`` over a ``dense_vector`` field (§5.3).

    ``field`` is the ``dense_vector`` field name (``IndexMapping.sem_field(embedder_id)``);
    ``query_embedder`` is that embedder's provider connector (§3.4). The query text is embedded
    CLIENT-SIDE (ES no longer embeds it) and the resulting vector drives a ``knn`` query. Shares
    ``_search``/``_msearch`` with ``LexicalSearcher`` so the client-side (score desc, doc_id asc)
    tie-break is identical.
    """

    def __init__(
        self,
        client: Elasticsearch,
        index: str,
        field: str,
        query_embedder: Embedder,
        *,
        msearch_chunk_size: int = _MSEARCH_CHUNK_SIZE,
        num_candidates: int = _KNN_NUM_CANDIDATES,
    ) -> None:
        self.client = client
        self.index = index
        self.field = field
        self.query_embedder = query_embedder
        self._msearch_chunk_size = msearch_chunk_size
        self.num_candidates = num_candidates

    def _body(self, vector: Sequence[float], top_k: int) -> dict[str, Any]:
        return {
            "knn": {
                "field": self.field,
                "query_vector": list(vector),
                "k": top_k,
                "num_candidates": max(top_k, self.num_candidates),
            },
            "size": top_k,
        }

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        """Embed ``query`` then run a ``knn`` query; ≤ ``top_k`` docs (score desc, doc_id asc, §5.3)."""
        vector = self.query_embedder.embed_queries([query])[0]
        return _search(self.client, self.index, self._body(vector, top_k))

    def bulk_search(self, queries: Sequence[str], *, top_k: int) -> list[list[ScoredDoc]]:
        """Embed all ``queries`` (batched by the connector), then batch ``knn`` bodies via ``_msearch`` (§5.3).

        The embedder batches the query set into few provider calls; ES round trips go through the
        shared chunked ``_msearch`` (chunking + per-response error-raise + ``_hits_to_scored``
        tie-break). Result ``i`` aligns to ``queries[i]``.
        """
        vectors = self.query_embedder.embed_queries(list(queries))
        bodies = [self._body(vector, top_k) for vector in vectors]
        return _msearch(self.client, self.index, bodies, chunk_size=self._msearch_chunk_size)


class ESReranker(Reranker):
    """Client-side ``Reranker`` over a provider :class:`~benchmark.common.protocols.RerankClient` (§3.7, §5.4).

    ``field`` is the doc-text field (``search_text``) whose value is fed to the reranker.
    ``rerank_client`` is the provider connector (Cohere/Voyage, §3.4). ``rerank`` fetches each
    candidate's text by id (one ``mget``), then delegates the windowed reorder to
    :func:`benchmark.common.ranking.rerank_local`: ``score_fn`` calls ``rerank_client.rerank_scores`` over the
    candidate doc-text (scores returned ALIGNED to input, higher = more relevant).
    """

    def __init__(
        self, client: Elasticsearch, index: str, field: str, rerank_client: RerankClient
    ) -> None:
        self.client = client
        self.index = index
        self.field = field
        self.rerank_client = rerank_client

    def _doc_texts_by_id(self, doc_ids: Sequence[str]) -> dict[str, str]:
        """``mget`` the ``field`` value for each id; a missing/not-found doc RAISES (§3.7)."""
        response = self.client.mget(index=self.index, ids=list(doc_ids), source=[self.field])
        texts: dict[str, str] = {}
        for doc in response["docs"]:
            if not doc.get("found", False):
                raise KeyError(
                    f"rerank candidate {doc['_id']!r} not found in index {self.index!r}"
                )
            source = doc["_source"]
            if self.field not in source:
                raise KeyError(
                    f"rerank candidate {doc['_id']!r} has no {self.field!r} field to rerank"
                )
            texts[doc["_id"]] = source[self.field]
        return texts

    def _score(self, query: str, doc_texts: Sequence[str]) -> list[float]:
        """Call the provider rerank connector; one relevance score per doc text, ALIGNED to input (§5.4)."""
        return list(self.rerank_client.rerank_scores(query, doc_texts))

    def rerank(self, query: str, candidates: Sequence[ScoredDoc]) -> list[ScoredDoc]:
        """Reorder ``candidates`` best-first by the provider's relevance scores (§3.7).

        The whole candidate list is the rerank window (``rank_window_size=len(candidates)``) — the
        ``SearchPipeline`` already retrieved exactly ``rerank_window_size`` candidates (§3.6).
        """
        if not candidates:
            # Nothing to rerank (a query with no retrieval hits) — return empty without a round
            # trip. ES `mget` and the provider rerank API both reject an empty ids/documents list.
            return []
        text_by_id = self._doc_texts_by_id([candidate.doc_id for candidate in candidates])
        return rerank_local(
            query,
            candidates,
            rank_window_size=len(candidates),
            doc_text=lambda doc_id: text_by_id[doc_id],
            score_fn=self._score,
        )


#: ES field-name sanitizer: ES field names cannot contain ``.`` (dots denote subfields), so a
#: dense_vector field name is ``"sem__"`` + the embedder id with every non-alphanumeric run -> ``"_"``.
_SEM_FIELD_PREFIX = "sem__"


def _sem_field_name(embedder_id: str) -> str:
    """Build a dot-free ``dense_vector`` field name from an embedder id (§5.2)."""
    return _SEM_FIELD_PREFIX + re.sub(r"[^0-9a-zA-Z]+", "_", embedder_id)


def _schema_to_mapping(
    schema: FieldSchema,
    vector_field_dims: Mapping[str, int],
    *,
    similarity: str = _BM25_SIMILARITY,
    analyzer: str = _DEFAULT_ANALYZER,
) -> dict[str, Any]:
    """Translate a ``FieldSchema`` into the ES ``{"properties": {...}}`` mapping body (§5.2).

    The canonical ``search_text`` field is a ``text`` field (BM25 target) carrying an EXPLICIT
    ``similarity`` (the tuned BM25, defined in index settings) and ``analyzer`` (P1-2) so the resolved
    scoring profile reads back verbatim from ``_mapping``. Each embedder gets
    one ``dense_vector`` field (``dims`` = the embedder's output dim, ``index: true``,
    ``similarity: cosine`` — cosine suits the normalized embeddings these providers emit). NUMERIC
    roles map to ``float`` (a superset of integer that never loses precision), STORED roles to a
    ``keyword`` stored field, ID roles become the doc ``_id`` (not a mapped field). BM25/
    SEMANTIC_SOURCE role fields are the source columns concatenated INTO ``search_text`` (§5.1), so
    they need no own mapping. Branching is exhaustive over ``FieldRole``.
    """
    properties: dict[str, Any] = {
        schema.search_text_field: {
            "type": "text",
            "similarity": similarity,
            "analyzer": analyzer,
        }
    }
    for field_name, dims in vector_field_dims.items():
        properties[field_name] = {
            "type": "dense_vector",
            "dims": dims,
            "index": True,
            "similarity": "cosine",
        }

    for spec in schema.fields:
        role = spec.role
        if role in (FieldRole.BM25, FieldRole.SEMANTIC_SOURCE):
            continue  # concatenated INTO search_text (§5.1); not mapped on its own
        if role is FieldRole.ID:
            continue  # becomes the doc _id, not a mapped field
        if role is FieldRole.NUMERIC:
            properties[spec.name] = {"type": "float"}
        elif role is FieldRole.STORED:
            properties[spec.name] = {"type": "keyword"}
        else:
            raise ValueError(f"unhandled field role {role!r} for field {spec.name!r}")

    return {"properties": properties}


#: (name, kind, embedder_id-or-None) — the flat searcher spec ``config`` translates ``Services`` into
#: so this adapter never imports ``config`` (§11). ``embedder_id`` is set only for ``kind == "vector"``.
SearcherSpec = tuple[str, str, str | None]

#: One ES client per ``url`` per process, SHARED by :func:`build_searchers` + :func:`build_rerankers`
#: so a run's search phase opens exactly one client (as the deleted ``_ESSearcherFactory`` did, §4).
#: ponytail: process-global cache; a single-run CLI reuses the client — swap for an explicit per-run
#: connection object only if the harness ever runs multiple configs concurrently in one process.
_SEARCH_CLIENTS: dict[str, Elasticsearch] = {}


def _open(indexer_cfg: Mapping[str, Any]) -> tuple[Elasticsearch, int, int]:
    """Open (or reuse) the shared search-side ES client + read the §10 search tuning knobs.

    Returns ``(client, msearch_chunk_size, knn_num_candidates)``. The client is memoized by
    ``settings.url`` so :func:`build_searchers` and :func:`build_rerankers` share ONE client per run.
    """
    settings = indexer_cfg["settings"]
    url = settings["url"]
    client = _SEARCH_CLIENTS.get(url)
    if client is None:
        client = _make_client(indexer_cfg)
        _SEARCH_CLIENTS[url] = client
    msearch_chunk_size = int(settings.get("msearch_chunk_size", _MSEARCH_CHUNK_SIZE))
    num_candidates = int(settings.get("knn_num_candidates", _KNN_NUM_CANDIDATES))
    return client, msearch_chunk_size, num_candidates


def build_searchers(
    indexer_cfg: Mapping[str, Any],
    mapping: IndexMapping,
    specs: Sequence[SearcherSpec],
    *,
    embedders: Mapping[str, Embedder],
    cache: DiskCache | None = None,
) -> dict[str, Searcher]:
    """Mint the configured leaf ``Searcher``s over the shared ES client (§4), keyed by service name.

    Each spec is ``(name, kind, embedder_id)``: a ``lexical`` spec -> ``LexicalSearcher`` on
    ``mapping.search_text_field``; a ``vector`` spec -> ``VectorSearch`` on that embedder's
    ``dense_vector`` field (``mapping.sem_field(embedder_id)``) with the referenced query embedder.
    Exhaustive on ``kind`` — an unknown kind raises (never a silent default).

    When ``cache`` is set, each leaf is wrapped in a :class:`~benchmark.common.cache.CachingSearcher`
    keyed by the index fingerprint (:func:`_index_version`, fetched ONCE — cheap, and the index is
    already known to exist, §6) and a per-leaf ``identity`` (``match:field`` for lexical;
    ``knn:field:num_candidates=N:emb=<embedder cache_identity>`` for vector — the embedder identity
    makes a re-embed-into-the-same-index model swap a guaranteed miss, docs/architecture.md §5.5).
    ``cache is None`` returns bare leaves (full bypass; no fingerprint fetch, no identity built).
    """
    client, msearch_chunk_size, num_candidates = _open(indexer_cfg)
    index = mapping.index_name
    index_version = _index_version(client, index) if cache is not None else None  # fetched ONCE
    out: dict[str, Searcher] = {}
    for name, kind, embedder_id in specs:
        if kind == "lexical":
            leaf: Searcher = LexicalSearcher(
                client, index, [mapping.search_text_field], msearch_chunk_size=msearch_chunk_size
            )
            identity = f"match:{mapping.search_text_field}"
        elif kind == "vector":
            if embedder_id is None:
                raise ValueError(f"vector searcher {name!r} has no embedder reference")
            field = mapping.sem_field(embedder_id)
            embedder = embedders[embedder_id]
            leaf = VectorSearch(
                client, index, field, embedder,
                msearch_chunk_size=msearch_chunk_size, num_candidates=num_candidates,
            )
            # cache_identity only exists on the CachingEmbedder that make_embedders(...) mints under
            # the SAME cache handle (§5); a bare embedder (cache is None) is never asked for it.
            identity = (
                ""
                if cache is None
                else f"knn:{field}:num_candidates={num_candidates}:"
                f"emb={cast(CachingEmbedder, embedder).cache_identity}"
            )
        else:
            raise ValueError(
                f"searcher {name!r}: unknown kind {kind!r}; expected 'lexical' or 'vector'"
            )
        if cache is None:
            out[name] = leaf
        else:
            assert index_version is not None  # always set when cache is active (fetched above)
            out[name] = CachingSearcher(leaf, cache, index_version=index_version, identity=identity)
    return out


def _index_version(client: Elasticsearch, index: str) -> str:
    """The index fingerprint ``"uuid:doc_count"`` — what makes a cached result list valid (§6).

    The UUID changes on any delete+recreate (a rebuilt index cannot serve stale lists); the doc
    count catches incremental growth INTO the same index (same UUID, new docs, changed rankings).
    Both are one cheap call each, fetched once per build.
    """
    uuid = client.indices.get(index=index)[index]["settings"]["index"]["uuid"]
    count = client.count(index=index)["count"]
    return f"{uuid}:{count}"


def build_rerankers(
    indexer_cfg: Mapping[str, Any],
    mapping: IndexMapping,
    names: Sequence[str],
    *,
    rerank_clients: Mapping[str, RerankClient],
) -> dict[str, Reranker]:
    """Mint the configured ``ESReranker``s over the shared ES client (§4), keyed by service name.

    Each name -> an ``ESReranker`` on ``mapping.search_text_field`` (the fixed §5.3 rerank field)
    wired to that name's provider ``RerankClient`` connector (built by ``config.make_rerankers``).
    """
    client, _, _ = _open(indexer_cfg)
    index = mapping.index_name
    return {
        name: ESReranker(client, index, mapping.search_text_field, rerank_clients[name])
        for name in names
    }
