"""ES adapter: LexicalSearcher, VectorSearch, ESReranker, ESIndexer, ElasticsearchBackend (ingest).

docs/experiment.md §3.3, §3.4, §3.5, §5. ES is a **plain vector/BM25 index** (§1.1) — it is NOT an
inference gateway. The harness computes embeddings via the provider connectors
(``benchmark.providers``, §3.4) and stores them in ``dense_vector`` fields; there is no
``_inference`` endpoint, no ``semantic_text`` field, and no ``register_inference``.

- Ingest (:class:`ElasticsearchBackend` + :class:`ESIndexer`): the indexer embeds the corpus with
  each :class:`~benchmark.protocols.Embedder` (batched, §3.5) and writes one ``dense_vector`` field
  per embedder alongside the BM25 ``search_text`` field; ``bulk_index`` streams via
  ``helpers.streaming_bulk``.
- Lexical retrieval (:class:`LexicalSearcher`): a ``match`` query on ``search_text``.
- Vector retrieval (:class:`VectorSearch`): embed the query with the embedder, then an ES ``knn``
  query over that embedder's ``dense_vector`` field — batched via the shared ``_msearch``.
- Rerank (:class:`ESReranker`): fetch candidate doc-text (``mget``) and score it with a provider
  :class:`~benchmark.protocols.RerankClient`, reordering client-side via ``rerank_local``.

Fusion stays client-side (``RRFFuser``) — no server-side ``rrf``. The ES client
(``elasticsearch>=8.15,<9``) is pinned, so its API is called DIRECTLY — no ``getattr``/``hasattr``
feature probing (CLAUDE.md move-with-certainty).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

from elasticsearch import Elasticsearch
from elasticsearch.helpers import BulkIndexError, streaming_bulk

from benchmark.logging_setup import get_logger
from benchmark.models import (
    Document,
    FieldRole,
    FieldSchema,
    IndexMapping,
    ScoredDoc,
)
from benchmark.protocols import Dataset, Embedder, Reranker, RerankClient, Searcher
from benchmark.rerank import rerank_local

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


def _make_client(indexer_cfg: Mapping[str, Any]) -> Elasticsearch:
    """Build an :class:`Elasticsearch` client from ``indexer.settings.url`` (§10)."""
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


class ElasticsearchBackend:
    """The ES ingest seam used by ``Indexer.build`` (``SearchBackend``, §3.3/§3.5).

    Creates the index mapping and bulk-indexes documents. ES is a plain index writer — the harness
    embeds the corpus client-side (:class:`~benchmark.protocols.Embedder`) and hands documents whose
    field bag already carries the ``dense_vector`` values; no inference runs server-side. Both ingest
    methods are idempotent so a re-run over an existing index is a no-op.
    """

    def __init__(self, indexer_cfg: Mapping[str, Any]) -> None:
        self.index: str = indexer_cfg["index"]
        self.client: Elasticsearch = _make_client(indexer_cfg)
        settings = indexer_cfg["settings"]
        self.bulk_chunk_size: int = int(settings.get("bulk_chunk_size", _BULK_CHUNK_SIZE))
        self.embed_batch_size: int = int(settings.get("embed_batch_size", _EMBED_BATCH_SIZE))

    def ensure_index(self, mapping: IndexMapping) -> None:
        """Create ``mapping.index_name`` with the §5.2 mappings body; skip if it exists (idempotent)."""
        if self.client.indices.exists(index=mapping.index_name):
            logger.info("index %r already exists; skipping create", mapping.index_name)
            return
        logger.info("creating index %r", mapping.index_name)
        self.client.indices.create(index=mapping.index_name, mappings=mapping.backend_mapping)

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
    """Client-side ``Reranker`` over a provider :class:`~benchmark.protocols.RerankClient` (§3.7, §5.4).

    ``field`` is the doc-text field (``search_text``) whose value is fed to the reranker.
    ``rerank_client`` is the provider connector (Cohere/Voyage, §3.4). ``rerank`` fetches each
    candidate's text by id (one ``mget``), then delegates the windowed reorder to
    :func:`benchmark.rerank.rerank_local`: ``score_fn`` calls ``rerank_client.rerank_scores`` over the
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


def _embed_documents(
    docs: Iterable[Document],
    embedders: Sequence[Embedder],
    sem_fields: Mapping[str, str],
    search_text_field: str,
    batch_size: int,
) -> Iterator[Document]:
    """Stream ``docs``, attaching each embedder's document vector under its ``dense_vector`` field (§3.5).

    Buffers ``batch_size`` docs, embeds their ``search_text`` with EACH embedder in one provider call
    per batch (the connector sub-chunks to its own limit), and yields COPIES of the docs enriched with
    the vectors — STAYS LAZY (bounded buffer; the corpus is never fully materialized). A doc missing
    the ``search_text`` field RAISES (never silently embeds empty).
    """
    batch: list[Document] = []
    for doc in docs:
        batch.append(doc)
        if len(batch) >= batch_size:
            yield from _embed_batch(batch, embedders, sem_fields, search_text_field)
            batch = []
    if batch:
        yield from _embed_batch(batch, embedders, sem_fields, search_text_field)


def _embed_batch(
    batch: Sequence[Document],
    embedders: Sequence[Embedder],
    sem_fields: Mapping[str, str],
    search_text_field: str,
) -> Iterator[Document]:
    """Embed one buffered batch with every embedder; yield the docs enriched with their vectors."""
    texts: list[str] = []
    for doc in batch:
        if search_text_field not in doc.fields:
            raise KeyError(
                f"document {doc.doc_id!r} has no {search_text_field!r} field to embed"
            )
        texts.append(doc.fields[search_text_field])
    vectors_by_embedder = {embedder.id: embedder.embed_documents(texts) for embedder in embedders}
    for offset, doc in enumerate(batch):
        fields = dict(doc.fields)
        for embedder in embedders:
            fields[sem_fields[embedder.id]] = vectors_by_embedder[embedder.id][offset]
        yield Document(doc_id=doc.doc_id, fields=fields)


class ESIndexer:
    """Builds the single ES index from a dataset + embedder connectors (``Indexer``, §3.5).

    Lifecycle: discover each embedder's output ``dim`` (probe / ``settings.dims``), translate the
    dataset ``FieldSchema`` into the §5.2 mapping (a plain ``text`` ``search_text`` field + one
    ``dense_vector`` field per embedder), then stream the corpus THROUGH the embedders
    (:func:`_embed_documents`) so each document lands with its vectors. No inference endpoints are
    registered (ES is a plain index, §1.1).
    """

    def build(
        self,
        dataset: Dataset,
        backend: Any,
        embeddings: Sequence[Embedder],
    ) -> IndexMapping:
        embedders = list(embeddings)

        # 1. sem_fields: embedder id -> dense_vector field name. Discover each dim (probes the
        #    provider or reads settings.dims) — needed to map the dense_vector field before ingest.
        sem_fields: dict[str, str] = {embedder.id: _sem_field_name(embedder.id) for embedder in embedders}
        vector_field_dims: dict[str, int] = {
            sem_fields[embedder.id]: embedder.dim for embedder in embedders
        }
        logger.info(
            "index %r: %d embedder(s) -> dense_vector fields %s",
            backend.index, len(embedders), vector_field_dims,
        )

        # 2. Translate the dataset field schema into the ES "properties" body (§5.2).
        schema = dataset.field_schema()
        backend_mapping = _schema_to_mapping(schema, vector_field_dims)
        mapping = IndexMapping(
            index_name=backend.index,
            search_text_field=schema.search_text_field,
            sem_fields=sem_fields,
            backend_mapping=backend_mapping,
        )

        # 3. Create the index, then stream the corpus through the embedders so each doc lands with
        #    its dense_vector values. embed_batch_size is the ingest buffering granularity (§3.5).
        backend.ensure_index(mapping)
        embed_batch_size = getattr(backend, "embed_batch_size", _EMBED_BATCH_SIZE)
        enriched = _embed_documents(
            dataset.documents(), embedders, sem_fields, schema.search_text_field, embed_batch_size
        )
        backend.bulk_index(enriched, mapping=mapping)

        # 4. Return the mapping so leaf Searchers can name fields without re-deriving them.
        return mapping


def _schema_to_mapping(
    schema: FieldSchema, vector_field_dims: Mapping[str, int]
) -> dict[str, Any]:
    """Translate a ``FieldSchema`` into the ES ``{"properties": {...}}`` mapping body (§5.2).

    The canonical ``search_text`` field is a plain ``text`` field (BM25 target). Each embedder gets
    one ``dense_vector`` field (``dims`` = the embedder's output dim, ``index: true``,
    ``similarity: cosine`` — cosine suits the normalized embeddings these providers emit). NUMERIC
    roles map to ``float`` (a superset of integer that never loses precision), STORED roles to a
    ``keyword`` stored field, ID roles become the doc ``_id`` (not a mapped field). BM25/
    SEMANTIC_SOURCE role fields are the source columns concatenated INTO ``search_text`` (§5.1), so
    they need no own mapping. Branching is exhaustive over ``FieldRole``.
    """
    properties: dict[str, Any] = {schema.search_text_field: {"type": "text"}}
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


class _ESSearcherFactory:
    """Backend-bound ``SearcherFactory`` (§4): builds leaf ``Searcher``s + the ``Reranker`` on the
    ES client + index, wiring in the provider connectors.

    Holds the resolved ``embedders`` (id -> ``Embedder``) and ``rerankers`` (name -> ``RerankClient``)
    so ``vector`` can attach the right query embedder and ``reranker`` the right rerank connector.
    """

    def __init__(
        self,
        client: Elasticsearch,
        index: str,
        *,
        embedders: Mapping[str, Embedder],
        rerankers: Mapping[str, RerankClient],
        msearch_chunk_size: int = _MSEARCH_CHUNK_SIZE,
        num_candidates: int = _KNN_NUM_CANDIDATES,
    ) -> None:
        self.client = client
        self.index = index
        self.embedders = embedders
        self.rerankers = rerankers
        self.msearch_chunk_size = msearch_chunk_size
        self.num_candidates = num_candidates

    def lexical(self, *, fields: Sequence[str]) -> Searcher:
        return LexicalSearcher(
            self.client, self.index, fields, msearch_chunk_size=self.msearch_chunk_size
        )

    def vector(self, *, field: str, embedder_id: str) -> Searcher:
        return VectorSearch(
            self.client,
            self.index,
            field,
            self.embedders[embedder_id],
            msearch_chunk_size=self.msearch_chunk_size,
            num_candidates=self.num_candidates,
        )

    def reranker(self, name: str, field: str) -> Reranker:
        return ESReranker(self.client, self.index, field, self.rerankers[name])


def make_searcher_factory(
    indexer_cfg: Mapping[str, Any],
    *,
    embedders: Mapping[str, Embedder],
    rerankers: Mapping[str, RerankClient],
) -> _ESSearcherFactory:
    """Build the ES ``SearcherFactory`` bound to a client + index + the provider connectors (§4, §11).

    ``build_pipeline`` (``config.py``) uses this seam to assemble a pipeline's leaf ``Searcher``s /
    ``Reranker`` without importing this adapter: ``factory.lexical`` -> ``LexicalSearcher``,
    ``factory.vector`` -> ``VectorSearch`` (with the referenced query embedder), ``factory.reranker``
    -> ``ESReranker`` (with the referenced rerank connector). ``embedders``/``rerankers`` are the
    connector registries the runner builds via ``config.make_embedders``/``make_rerankers``.
    """
    backend = ElasticsearchBackend(indexer_cfg)
    settings = indexer_cfg["settings"]
    msearch_chunk_size = int(settings.get("msearch_chunk_size", _MSEARCH_CHUNK_SIZE))
    num_candidates = int(settings.get("knn_num_candidates", _KNN_NUM_CANDIDATES))
    return _ESSearcherFactory(
        backend.client,
        backend.index,
        embedders=embedders,
        rerankers=rerankers,
        msearch_chunk_size=msearch_chunk_size,
        num_candidates=num_candidates,
    )
