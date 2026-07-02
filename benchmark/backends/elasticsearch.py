"""ES adapter: LexicalSearcher, VectorSearch, ESReranker, ESIndexer, ElasticsearchBackend (ingest).

docs/experiment.md §3.3, §3.4, §3.5, §5. Phase 9 lands the ingest seam
(:class:`ElasticsearchBackend`, incl. batched streaming ``bulk_index`` via
``helpers.streaming_bulk``), the first leaf ``Searcher`` (:class:`LexicalSearcher`) with per-query
``search`` and batched ``bulk_search`` (ES Multi-Search), the shared query helpers (:func:`_search`,
:func:`_msearch`, :func:`_hits_to_scored`), and the lexical-only :func:`make_searcher_factory`.
``VectorSearch`` / ``ESReranker`` / ``ESIndexer`` are Phase 10 — the factory's ``vector`` /
``reranker`` seams raise ``NotImplementedError`` until then.

The ES client (``elasticsearch>=8.15,<9``) is pinned, so its API is called DIRECTLY — no
``getattr``/``hasattr`` feature probing (CLAUDE.md move-with-certainty).
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, Sequence

from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import streaming_bulk

from benchmark.logging_setup import get_logger
from benchmark.models import Document, IndexMapping, InferenceEndpoint, ScoredDoc
from benchmark.protocols import Reranker, Searcher

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


class _ESSearcherFactory:
    """Backend-bound ``SearcherFactory`` (§4): builds leaf ``Searcher``s on the ES client + index.

    Phase 9 supports only ``lexical``. ``vector`` / ``reranker`` land in Phase 10 and raise until
    then (exhaustive — no silent ``None`` stub).
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
        raise NotImplementedError("VectorSearch/ESReranker land in Phase 10")

    def reranker(self, inference_id: str, field: str) -> Reranker:
        raise NotImplementedError("VectorSearch/ESReranker land in Phase 10")


def make_searcher_factory(
    indexer_cfg: Mapping[str, Any], *args: Any, **kwargs: Any
) -> _ESSearcherFactory:
    """Build the ES ``SearcherFactory`` bound to a client + index (§4, §11).

    ``build_pipeline`` (``config.py``) uses this seam to assemble a pipeline's leaf ``Searcher``s
    without importing this adapter. Phase 9 wires ``factory.lexical``; ``vector``/``reranker`` raise
    until Phase 10.
    """
    backend = ElasticsearchBackend(indexer_cfg)
    msearch_chunk_size = int(indexer_cfg["settings"].get("msearch_chunk_size", _MSEARCH_CHUNK_SIZE))
    return _ESSearcherFactory(
        backend.client, backend.index, msearch_chunk_size=msearch_chunk_size
    )
