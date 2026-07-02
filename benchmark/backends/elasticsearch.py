"""ES adapter: LexicalSearcher, VectorSearch, ESReranker, ESIndexer, ElasticsearchBackend (ingest).

docs/experiment.md §3.3, §3.4, §3.5, §5. Phase 9 lands the ingest seam
(:class:`ElasticsearchBackend`, incl. batched streaming ``bulk_index`` via
``helpers.streaming_bulk``), the first leaf ``Searcher`` (:class:`LexicalSearcher`) with per-query
``search`` and batched ``bulk_search`` (ES Multi-Search), the shared query helpers (:func:`_search`,
:func:`_msearch`, :func:`_hits_to_scored`), and :func:`make_searcher_factory`.
Phase 10 completes the adapter: :class:`VectorSearch` (the explicit ``semantic`` query, batched via
the shared ``_msearch``), :class:`ESReranker` (client-side ``rerank_local`` over the ``_inference``
rerank endpoint + an ``mget`` doc-text lookup), and :class:`ESIndexer` (§3.5 register→ensure→index,
``search_text`` ``copy_to`` one ``semantic_text`` field per embedder). Fusion stays client-side
(``RRFFuser``, Phase 5) — no server-side ``rrf`` / ``text_similarity_reranker``.

The ES client (``elasticsearch>=8.15,<9``) is pinned, so its API is called DIRECTLY — no
``getattr``/``hasattr`` feature probing (CLAUDE.md move-with-certainty).
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import streaming_bulk

from benchmark.logging_setup import get_logger
from benchmark.models import (
    Document,
    FieldRole,
    FieldSchema,
    IndexMapping,
    InferenceEndpoint,
    ScoredDoc,
)
from benchmark.protocols import Dataset, EmbeddingModel, Reranker, Searcher
from benchmark.rerank import rerank_local

logger = get_logger(__name__)

#: Default per-request timeout (seconds) for the ES client. Ingest bulk calls can be slow.
_DEFAULT_REQUEST_TIMEOUT = 60

#: Default docs per ``streaming_bulk`` chunk (the ES helpers default). Overridable via
#: ``indexer.settings.bulk_chunk_size`` for very large corpora (WANDS ~43K, ESCI ~1M docs).
_BULK_CHUNK_SIZE = 500

#: Default per-search count per Multi-Search (``_msearch``) request. The query set is chunked
#: into groups of this size so ~480 (WANDS) / ~48K (ESCI) queries take few round trips, not one
#: per query. Overridable via ``indexer.settings.msearch_chunk_size``.
_MSEARCH_CHUNK_SIZE = 100

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
    tie-break lives here instead. Shared by :func:`_search` and :func:`_msearch` (and thus
    ``VectorSearch`` in Phase 10).
    """
    scored = [ScoredDoc(doc_id=hit["_id"], score=float(hit["_score"])) for hit in hits]
    scored.sort(key=lambda doc: (-doc.score, doc.doc_id))
    return scored


def _search(client: Elasticsearch, index: str, body: Mapping[str, Any]) -> list[ScoredDoc]:
    """Run a single search ``body`` against ``index`` and map hits -> ``ScoredDoc`` (§3.3, §9.1).

    Shared with ``VectorSearch`` in Phase 10. See :func:`_hits_to_scored` for the tie-break.
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
    exception convention). Reused by ``VectorSearch`` in Phase 10.
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

    Registers ``_inference`` endpoints, creates the index mapping, and bulk-indexes documents. All
    three ingest methods are idempotent so a re-run over an existing index/endpoint is a no-op.
    """

    def __init__(self, indexer_cfg: Mapping[str, Any]) -> None:
        self.index: str = indexer_cfg["index"]
        self.client: Elasticsearch = _make_client(indexer_cfg)
        settings = indexer_cfg["settings"]
        self.bulk_chunk_size: int = int(settings.get("bulk_chunk_size", _BULK_CHUNK_SIZE))

    def register_inference(self, ep: InferenceEndpoint) -> str:
        """Idempotent create-or-get of ``PUT _inference/{task_type}/{inference_id}`` (§3.4).

        The body keeps ``service`` / ``service_settings`` / ``task_settings`` as SEPARATE maps
        (``task_settings`` included only when non-empty). If the endpoint already exists we do not
        recreate it — the id is returned as-is. Returns ``ep.inference_id``.
        """
        task_type = ep.task_type.value
        try:
            self.client.inference.get(task_type=task_type, inference_id=ep.inference_id)
        except NotFoundError:
            # Expected on first registration: the endpoint does not exist yet, so create it.
            # Only NotFoundError is caught — any other error (auth, connection, bad request)
            # propagates instead of being silently swallowed.
            body: dict[str, Any] = {
                "service": ep.service,
                "service_settings": dict(ep.service_settings),
            }
            if ep.task_settings:
                body["task_settings"] = dict(ep.task_settings)
            logger.info("inference endpoint %r not found; registering (task=%s)", ep.inference_id, task_type)
            self.client.inference.put(task_type=task_type, inference_id=ep.inference_id, body=body)
        else:
            logger.info("inference endpoint %r already exists; reusing", ep.inference_id)
        return ep.inference_id

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
        for _ok, _info in streaming_bulk(
            self.client,
            actions(),
            chunk_size=self.bulk_chunk_size,
            raise_on_error=True,
        ):
            indexed += 1
            if indexed % _BULK_PROGRESS_EVERY == 0:
                logger.info("bulk_index: %d docs into %r so far", indexed, index_name)

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
    """Semantic leaf ``Searcher``: the explicit ``semantic`` query on a ``semantic_text`` field (§5.3).

    ``field`` is the ``semantic_text`` field name (``IndexMapping.sem_field(embedder)``). Uses the
    explicit ``{"semantic": {"field", "query"}}`` form — version-robust across ES >= 8.15 (§5.3),
    not the implicit match-on-``semantic_text`` form (ES >= 8.18 only). Shares ``_search``/``_msearch``
    with ``LexicalSearcher`` so the client-side (score desc, doc_id asc) tie-break is identical.
    """

    def __init__(
        self,
        client: Elasticsearch,
        index: str,
        field: str,
        *,
        msearch_chunk_size: int = _MSEARCH_CHUNK_SIZE,
    ) -> None:
        self.client = client
        self.index = index
        self.field = field
        self._msearch_chunk_size = msearch_chunk_size

    def _body(self, query: str, top_k: int) -> dict[str, Any]:
        return {"query": {"semantic": {"field": self.field, "query": query}}, "size": top_k}

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        """Run the explicit ``semantic`` query; ≤ ``top_k`` docs (score desc, doc_id asc, §5.3)."""
        return _search(self.client, self.index, self._body(query, top_k))

    def bulk_search(self, queries: Sequence[str], *, top_k: int) -> list[list[ScoredDoc]]:
        """Batch the ``semantic`` bodies through the shared chunked ``_msearch``; ALIGNED (§5.3).

        Mirrors ``LexicalSearcher.bulk_search`` — reuses ``_msearch`` (chunking + per-response
        error-raise + ``_hits_to_scored`` tie-break); no new batching code.
        """
        bodies = [self._body(query, top_k) for query in queries]
        return _msearch(self.client, self.index, bodies, chunk_size=self._msearch_chunk_size)


class ESReranker(Reranker):
    """Client-side ``Reranker`` over the ES ``_inference`` rerank endpoint (§3.7, §5.3).

    ``field`` is the doc-text field (``search_text``) whose value is fed to the reranker. ``rerank``
    fetches each candidate's text by id (one ``mget``), then delegates the windowed reorder to
    :func:`benchmark.rerank.rerank_local`: ``score_fn`` wraps the ``_inference`` rerank call, mapping
    each returned score BACK to input order by the response's ``"index"`` field. No server-side
    ``text_similarity_reranker`` retriever.
    """

    def __init__(
        self, client: Elasticsearch, index: str, inference_id: str, field: str
    ) -> None:
        self.client = client
        self.index = index
        self.inference_id = inference_id
        self.field = field

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
        """Call ``_inference/rerank/{id}`` with ``query`` + ``doc_texts``; scores ALIGNED to input.

        The raw response is ``{"rerank": [{"index": i, "relevance_score": s}, ...]}`` — ``index`` is
        the position in the input list (order NOT guaranteed) and ``relevance_score`` MAY be negative
        (a cross-encoder logit; higher = more relevant). Scores are placed back at ``item["index"]``
        so the returned list aligns with ``doc_texts`` for ``rerank_local``.
        """
        response = self.client.inference.rerank(
            inference_id=self.inference_id, query=query, input=list(doc_texts)
        )
        scores: list[float] = [0.0] * len(doc_texts)
        for item in response["rerank"]:
            scores[item["index"]] = float(item["relevance_score"])
        return scores

    def rerank(self, query: str, candidates: Sequence[ScoredDoc]) -> list[ScoredDoc]:
        """Reorder ``candidates`` best-first by the model's relevance scores (§3.7).

        The whole candidate list is the rerank window (``rank_window_size=len(candidates)``) — the
        ``SearchPipeline`` already retrieved exactly ``rerank_window_size`` candidates (§3.6).
        """
        if not candidates:
            # Nothing to rerank (a query with no retrieval hits) — return empty without a round
            # trip. ES `mget`/`_inference` reject an empty ids/input list ("no documents to get").
            return []
        text_by_id = self._doc_texts_by_id([candidate.doc_id for candidate in candidates])
        return rerank_local(
            query,
            candidates,
            rank_window_size=len(candidates),
            doc_text=lambda doc_id: text_by_id[doc_id],
            score_fn=self._score,
        )


#: ES field-name sanitizer: ES field names cannot contain ``.`` (dots denote subfields), so a sem
#: field name is ``"sem__"`` + the embedder id with every non-alphanumeric run replaced by ``"_"``.
_SEM_FIELD_PREFIX = "sem__"


def _sem_field_name(embedder_id: str) -> str:
    """Build a dot-free ``semantic_text`` field name from an embedder inference id (§5.2)."""
    return _SEM_FIELD_PREFIX + re.sub(r"[^0-9a-zA-Z]+", "_", embedder_id)


class ESIndexer:
    """Builds the single ES index from a dataset + embedding models (``Indexer``, §3.5).

    Strict lifecycle: register every embedder endpoint FIRST (a ``semantic_text`` field cannot map
    before its ``inference_id`` exists), translate the dataset ``FieldSchema`` into the §5.2 mapping
    (``search_text`` ``copy_to`` one ``semantic_text`` field per embedder), then stream the corpus.
    Rerankers are NOT registered here — they touch no mapping (lazy at run, §8 R0).
    """

    def build(
        self,
        dataset: Dataset,
        backend: Any,
        embeddings: Sequence[EmbeddingModel],
    ) -> IndexMapping:
        # 1. Register each embedder endpoint BEFORE the mapping (idempotent — a preconfigured
        #    endpoint is reused). sem_fields maps embedder inference_id -> its sem field name.
        sem_fields: dict[str, str] = {}
        for model in embeddings:
            inference_id = backend.register_inference(model.as_endpoint())
            sem_fields[inference_id] = _sem_field_name(inference_id)

        # 2. Translate the dataset field schema into the ES "properties" body. copy_to lives on the
        #    SOURCE search_text field and points at every sem field (§5.2), NOT copy_to_source.
        schema = dataset.field_schema()
        backend_mapping = _schema_to_mapping(schema, sem_fields)
        mapping = IndexMapping(
            index_name=backend.index,
            search_text_field=schema.search_text_field,
            sem_fields=sem_fields,
            backend_mapping=backend_mapping,
        )

        # 3. Create the index then stream the corpus (ES embeds each semantic_text at ingest).
        backend.ensure_index(mapping)
        backend.bulk_index(dataset.documents(), mapping=mapping)

        # 4. Return the mapping so leaf Searchers can name fields without re-deriving them.
        return mapping


def _schema_to_mapping(
    schema: FieldSchema, sem_fields: Mapping[str, str]
) -> dict[str, Any]:
    """Translate a ``FieldSchema`` into the ES ``{"properties": {...}}`` mapping body (§5.2).

    The canonical ``search_text`` field is a ``text`` field carrying ``copy_to`` -> every
    ``semantic_text`` field (one per embedder); each ``semantic_text`` field sets its ``inference_id``
    explicitly. NUMERIC roles map to ``float`` (a superset of integer that never loses precision),
    STORED roles to a ``keyword`` stored field, ID roles become the doc ``_id`` (not a mapped field).
    BM25/SEMANTIC_SOURCE role fields are the source columns concatenated INTO ``search_text`` (§5.1),
    so they need no own mapping. Branching is exhaustive over ``FieldRole``.
    """
    properties: dict[str, Any] = {
        schema.search_text_field: {
            "type": "text",
            "copy_to": list(sem_fields.values()),
        }
    }
    for embedder_id, sem_field in sem_fields.items():
        properties[sem_field] = {"type": "semantic_text", "inference_id": embedder_id}

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
    ES client + index. Builds ``LexicalSearcher`` / ``VectorSearch`` / ``ESReranker``.
    """

    def __init__(
        self, client: Elasticsearch, index: str, *, msearch_chunk_size: int = _MSEARCH_CHUNK_SIZE
    ) -> None:
        self.client = client
        self.index = index
        self.msearch_chunk_size = msearch_chunk_size

    def lexical(self, *, fields: Sequence[str]) -> Searcher:
        return LexicalSearcher(
            self.client, self.index, fields, msearch_chunk_size=self.msearch_chunk_size
        )

    def vector(self, *, field: str) -> Searcher:
        return VectorSearch(
            self.client, self.index, field, msearch_chunk_size=self.msearch_chunk_size
        )

    def reranker(self, inference_id: str, field: str) -> Reranker:
        return ESReranker(self.client, self.index, inference_id, field)


def make_searcher_factory(
    indexer_cfg: Mapping[str, Any], *args: Any, **kwargs: Any
) -> _ESSearcherFactory:
    """Build the ES ``SearcherFactory`` bound to a client + index (§4, §11).

    ``build_pipeline`` (``config.py``) uses this seam to assemble a pipeline's leaf ``Searcher``s /
    ``Reranker`` without importing this adapter: ``factory.lexical`` -> ``LexicalSearcher``,
    ``factory.vector`` -> ``VectorSearch``, ``factory.reranker`` -> ``ESReranker``.
    """
    backend = ElasticsearchBackend(indexer_cfg)
    msearch_chunk_size = int(indexer_cfg["settings"].get("msearch_chunk_size", _MSEARCH_CHUNK_SIZE))
    return _ESSearcherFactory(
        backend.client, backend.index, msearch_chunk_size=msearch_chunk_size
    )
