"""ES adapter: LexicalSearcher, VectorSearch, ESReranker, ESIndexer, ElasticsearchBackend (ingest).

docs/experiment.md §3.3, §3.4, §3.5, §5. Phase 9 lands the ingest seam
(:class:`ElasticsearchBackend`), the first leaf ``Searcher`` (:class:`LexicalSearcher`), the
shared query helper (:func:`_search`), and the lexical-only :func:`make_searcher_factory`.
``VectorSearch`` / ``ESReranker`` / ``ESIndexer`` are Phase 10 — the factory's ``vector`` /
``reranker`` seams raise ``NotImplementedError`` until then.

The ES client (``elasticsearch>=8.15,<9``) is pinned, so its API is called DIRECTLY — no
``getattr``/``hasattr`` feature probing (CLAUDE.md move-with-certainty).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from elasticsearch import Elasticsearch, NotFoundError

from benchmark.logging_setup import get_logger
from benchmark.models import Document, IndexMapping, InferenceEndpoint, ScoredDoc
from benchmark.protocols import Reranker, Searcher

logger = get_logger(__name__)

#: Default per-request timeout (seconds) for the ES client. Ingest bulk calls can be slow.
_DEFAULT_REQUEST_TIMEOUT = 60


def _make_client(indexer_cfg: Mapping[str, Any]) -> Elasticsearch:
    """Build an :class:`Elasticsearch` client from ``indexer.settings.url`` (§10)."""
    settings = indexer_cfg["settings"]
    url = settings["url"]
    request_timeout = int(settings.get("request_timeout", _DEFAULT_REQUEST_TIMEOUT))
    return Elasticsearch(url, request_timeout=request_timeout)


def _search(client: Elasticsearch, index: str, body: Mapping[str, Any]) -> list[ScoredDoc]:
    """Run a search ``body`` against ``index`` and map hits -> ``ScoredDoc`` (§3.3, §9.1).

    Sorts CLIENT-SIDE by (score desc, doc_id asc) — ES 8.x disallows fielddata on ``_id`` so a
    server-side ``_id`` sort errors; the deterministic tie-break lives here instead (§9.1). Shared
    with ``VectorSearch`` in Phase 10.
    """
    response = client.search(index=index, **body)
    hits = response["hits"]["hits"]
    scored = [ScoredDoc(doc_id=hit["_id"], score=float(hit["_score"])) for hit in hits]
    scored.sort(key=lambda doc: (-doc.score, doc.doc_id))
    return scored


class ElasticsearchBackend:
    """The ES ingest seam used by ``Indexer.build`` (``SearchBackend``, §3.3/§3.5).

    Registers ``_inference`` endpoints, creates the index mapping, and bulk-indexes documents. All
    three ingest methods are idempotent so a re-run over an existing index/endpoint is a no-op.
    """

    def __init__(self, indexer_cfg: Mapping[str, Any]) -> None:
        self.index: str = indexer_cfg["index"]
        self.client: Elasticsearch = _make_client(indexer_cfg)

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
        """Bulk index ``docs`` into ``mapping.index_name`` (``_id = doc.doc_id``), then refresh (§3.5).

        Each doc's ``fields`` bag is the ``_source``. The index op upserts by ``_id`` so a re-run is
        idempotent. The index is refreshed afterward so docs are immediately searchable.
        """
        operations: list[Mapping[str, Any]] = []
        for doc in docs:
            operations.append({"index": {"_id": doc.doc_id}})
            operations.append(dict(doc.fields))

        if not operations:
            logger.info("bulk_index: no documents to index into %r", mapping.index_name)
            return

        logger.info("bulk indexing %d docs into %r", len(operations) // 2, mapping.index_name)
        self.client.bulk(index=mapping.index_name, operations=operations)
        self.client.indices.refresh(index=mapping.index_name)


class LexicalSearcher(Searcher):
    """BM25 leaf ``Searcher``: a single ``match`` query on ``search_text`` (§3.3, §5.3)."""

    def __init__(self, client: Elasticsearch, index: str, fields: Sequence[str]) -> None:
        if len(fields) != 1:
            raise ValueError(
                f"LexicalSearcher expects exactly one search field, got {list(fields)}"
            )
        self.client = client
        self.index = index
        self.field = fields[0]

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        """Run ``{"query": {"match": {field: query}}, "size": top_k}``; ≤ ``top_k`` docs (§5.3)."""
        body = {"query": {"match": {self.field: query}}, "size": top_k}
        return _search(self.client, self.index, body)


class _ESSearcherFactory:
    """Backend-bound ``SearcherFactory`` (§4): builds leaf ``Searcher``s on the ES client + index.

    Phase 9 supports only ``lexical``. ``vector`` / ``reranker`` land in Phase 10 and raise until
    then (exhaustive — no silent ``None`` stub).
    """

    def __init__(self, client: Elasticsearch, index: str) -> None:
        self.client = client
        self.index = index

    def lexical(self, *, fields: Sequence[str]) -> Searcher:
        return LexicalSearcher(self.client, self.index, fields)

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
    return _ESSearcherFactory(backend.client, backend.index)
