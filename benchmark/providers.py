"""Direct inference-provider connectors — embeddings + rerank (docs/experiment.md §3.4, §5.4).

The harness calls Cohere / Voyage / OpenAI **directly** (ES is no longer an inference gateway,
§1.1) so we have fine-grained control over batching, rate limiting, retries, and — crucially —
**failure inspection**: every non-recoverable error surfaces as a :class:`ProviderError` carrying
the provider, HTTP status, URL, and the raw response body (never swallowed).

Two seams (§3.4), realized here as concrete connectors:

- :class:`Embedder` (``embed_documents`` / ``embed_queries`` → dense vectors; a ``dim`` property):
  :class:`OpenAIEmbedder`, :class:`CohereEmbedder`, :class:`VoyageEmbedder`. The indexer embeds the
  corpus and writes ``dense_vector`` fields; ``VectorSearch`` embeds the query and runs ES ``knn``.
- :class:`RerankClient` (``rerank_scores`` → one relevance score per document, aligned to input):
  :class:`CohereReranker`, :class:`VoyageReranker`. A backend ``Reranker`` (ES ``ESReranker``) fetches
  candidate doc-text and calls this over it. **OpenAI has no reranker** — a reranker configured with
  ``provider: openai`` is rejected (:func:`make_reranker`).

Zero new dependencies (CLAUDE.md — favor stdlib): HTTP is stdlib :mod:`urllib.request` + :mod:`json`.
The three providers share nearly identical REST shapes, so one :func:`_post_json` (retry/backoff on
429/5xx, honoring ``Retry-After``) + a :class:`RateLimiter` (``requests_per_minute``) back every
connector. Imports only ``benchmark.protocols`` (for structural conformance typing, not required at
runtime) + the cross-cutting ``logging_setup`` + stdlib — never a backend/dataset (§11).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence

from benchmark.logging_setup import get_logger

logger = get_logger(__name__)

#: Default per-request socket timeout (seconds). Embedding a full batch can be slow; generous.
_DEFAULT_TIMEOUT = 60.0
#: Default retry budget for a single request on a retryable status / connection error.
_DEFAULT_MAX_RETRIES = 5
#: Base seconds for exponential backoff (``base * 2**attempt``), capped by :data:`_BACKOFF_CAP`.
_BACKOFF_BASE = 1.0
_BACKOFF_CAP = 30.0
#: HTTP statuses worth retrying: 429 (rate limit) + transient 5xx. A 4xx (auth, bad request) is a
#: hard error — surfaced immediately with its body so the cause is visible.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

#: Per-provider default max texts per embedding request (provider batch limits, verified):
#: OpenAI 2048 inputs/req, Cohere 96 texts/req, Voyage 128 inputs/req. Overridable via
#: ``settings.batch_size``. The embedder sub-chunks any longer input to this size.
_DEFAULT_EMBED_BATCH: Mapping[str, int] = {"openai": 2048, "cohere": 96, "voyage": 128}

#: Probe text used to discover an embedder's output dimensionality once, when ``settings.dims`` is
#: absent (§3.5): the dense_vector mapping needs ``dims`` before ingest, and the authoritative source
#: is the provider itself.
_DIM_PROBE_TEXT = "dimension probe"


class ProviderError(RuntimeError):
    """A provider request failed unrecoverably — carries the raw context for inspection (§3.4).

    ``status`` is the HTTP code (``None`` for a connection-level failure with no response).
    ``body`` is the raw response body (or the connection error reason), truncated in the message but
    kept whole on the attribute so a caller can log/inspect the provider's own error payload.
    """

    def __init__(self, provider: str, status: int | None, url: str, body: str) -> None:
        self.provider = provider
        self.status = status
        self.url = url
        self.body = body
        status_str = "no HTTP response" if status is None else f"HTTP {status}"
        super().__init__(f"{provider} request to {url} failed ({status_str}): {body[:500]}")


class RateLimiter:
    """A minimal single-thread request-spacing limiter from ``requests_per_minute`` (§3.4).

    Enforces a minimum interval of ``60 / requests_per_minute`` seconds between successive
    :meth:`acquire` calls (a monotonic-clock spacing, not a burst bucket — the harness issues
    requests serially). ``None``/``0`` requests_per_minute disables limiting (a no-op).

    ponytail: serial spacing, not a token bucket — the harness embeds/reranks one request at a time.
    Swap for a concurrency-aware limiter only if requests ever go parallel.
    """

    def __init__(self, requests_per_minute: int | None) -> None:
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0.0:
            return
        now = time.monotonic()
        if now < self._next_allowed:
            time.sleep(self._next_allowed - now)
            now = self._next_allowed
        self._next_allowed = now + self._min_interval


def _retry_delay(exc: urllib.error.HTTPError, attempt: int) -> float:
    """Seconds to wait before the next retry: honor ``Retry-After`` if present, else exp backoff."""
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after is not None:
        try:
            return min(float(retry_after), _BACKOFF_CAP)
        except ValueError:
            # A non-numeric Retry-After (HTTP-date form) — fall back to backoff rather than parse it.
            logger.debug("non-numeric Retry-After %r; using backoff", retry_after)
    return min(_BACKOFF_BASE * (2**attempt), _BACKOFF_CAP)


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    api_key: str,
    provider: str,
    rate_limiter: RateLimiter,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    """POST ``payload`` as JSON with bearer auth; parse the JSON response (§3.4).

    Retries on a retryable HTTP status (:data:`_RETRYABLE_STATUS`) or a connection-level error, with
    exponential backoff honoring ``Retry-After`` (:func:`_retry_delay`), up to ``max_retries``. A
    non-retryable status (e.g. 401/403/400) raises :class:`ProviderError` immediately with the raw
    body; an exhausted retry budget raises with the last body. Errors are NEVER swallowed.
    """
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    attempt = 0
    while True:
        rate_limiter.acquire()
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in _RETRYABLE_STATUS and attempt < max_retries:
                delay = _retry_delay(exc, attempt)
                logger.warning(
                    "%s HTTP %s (attempt %d/%d); retrying in %.1fs: %s",
                    provider, exc.code, attempt + 1, max_retries, delay, body[:200],
                )
                time.sleep(delay)
                attempt += 1
                continue
            raise ProviderError(provider, exc.code, url, body) from exc
        except urllib.error.URLError as exc:
            # Connection-level failure (DNS, refused, timeout) — no HTTP status. Retryable.
            reason = str(exc.reason)
            if attempt < max_retries:
                delay = min(_BACKOFF_BASE * (2**attempt), _BACKOFF_CAP)
                logger.warning(
                    "%s connection error (attempt %d/%d); retrying in %.1fs: %s",
                    provider, attempt + 1, max_retries, delay, reason,
                )
                time.sleep(delay)
                attempt += 1
                continue
            raise ProviderError(provider, None, url, reason) from exc


# --- shared settings parsing -------------------------------------------------------------------


class _Connector:
    """Shared connector plumbing: auth, model, endpoint URL, rate limiter, timeout, retries (§3.4).

    ``settings`` (an ``embedder``/``reranker`` service's ``settings`` block, §10) supplies
    ``api_key`` (required), ``model_id`` (required), optional ``rate_limit.requests_per_minute``,
    ``batch_size``, ``dims``, ``timeout``, ``max_retries``, ``base_url``.
    """

    provider: str
    default_url: str

    def __init__(self, name: str, settings: Mapping[str, Any]) -> None:
        self.id = name  # the config service name; == the embedder id used for sem-field naming (§3.5)
        self.api_key = _require_setting(settings, "api_key", self.provider, name)
        self.model = _require_setting(settings, "model_id", self.provider, name)
        rate_limit = settings.get("rate_limit") or {}
        self.rate_limiter = RateLimiter(rate_limit.get("requests_per_minute"))
        self.url = str(settings.get("base_url", self.default_url))
        self.timeout = float(settings.get("timeout", _DEFAULT_TIMEOUT))
        self.max_retries = int(settings.get("max_retries", _DEFAULT_MAX_RETRIES))

    def _post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return _post_json(
            self.url,
            payload,
            api_key=self.api_key,
            provider=self.provider,
            rate_limiter=self.rate_limiter,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )


def _require_setting(settings: Mapping[str, Any], key: str, provider: str, name: str) -> str:
    value = settings.get(key)
    if value is None:
        raise ValueError(f"{provider} service {name!r} requires settings.{key}")
    return str(value)


# --- embedders ---------------------------------------------------------------------------------


class _BaseEmbedder(_Connector):
    """Common embedder machinery: batching + one-shot ``dim`` discovery (§3.4/§3.5).

    Subclasses implement :meth:`_embed` (one provider call for a within-limit batch, ``is_query``
    selecting the provider's document/query input mode). ``embed_documents`` / ``embed_queries``
    sub-chunk arbitrary input to ``batch_size`` so a 43K-doc corpus never exceeds a provider's
    per-request cap. ``dim`` is taken from ``settings.dims`` if given (move-with-certainty), else
    discovered once by embedding a probe text (the authoritative source is the provider).
    """

    def __init__(self, name: str, settings: Mapping[str, Any]) -> None:
        super().__init__(name, settings)
        self.batch_size = int(settings.get("batch_size", _DEFAULT_EMBED_BATCH[self.provider]))
        dims = settings.get("dims")
        self._dims: int | None = int(dims) if dims is not None else None

    @property
    def dim(self) -> int:
        if self._dims is None:
            probe = self._embed([_DIM_PROBE_TEXT], is_query=False)
            self._dims = len(probe[0])
            logger.info(
                "%s embedder %r discovered dim=%d (model=%s)",
                self.provider, self.id, self._dims, self.model,
            )
        return self._dims

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed_batched(texts, is_query=False)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed_batched(texts, is_query=True)

    def _embed_batched(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed(list(texts[start : start + self.batch_size]), is_query=is_query))
        _check_count(vectors, texts, self.provider, self.id)
        return vectors

    def _embed(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        raise NotImplementedError


def _check_count(got: Sequence[Any], texts: Sequence[str], provider: str, name: str) -> None:
    """A provider must return exactly one vector/score per input — surface a mismatch, never pad."""
    if len(got) != len(texts):
        raise ProviderError(
            provider, None, name,
            f"expected {len(texts)} results, got {len(got)} (provider returned a truncated batch)",
        )


class OpenAIEmbedder(_BaseEmbedder):
    """OpenAI ``/v1/embeddings`` (``text-embedding-3-*``). No document/query input mode.

    When ``settings.dims`` is set it is sent as the ``dimensions`` request param (OpenAI -3 models
    support output-dim truncation) AND used as the dense_vector ``dims`` — so no probe is needed.
    """

    provider = "openai"
    default_url = "https://api.openai.com/v1/embeddings"

    def _embed(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model, "input": list(texts), "encoding_format": "float"}
        if self._dims is not None:
            payload["dimensions"] = self._dims
        response = self._post(payload)
        ordered = sorted(response["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]


class CohereEmbedder(_BaseEmbedder):
    """Cohere ``/v2/embed`` (``embed-*-v3.0``). ``input_type`` distinguishes document vs query."""

    provider = "cohere"
    default_url = "https://api.cohere.com/v2/embed"

    def _embed(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        payload = {
            "model": self.model,
            "texts": list(texts),
            "input_type": "search_query" if is_query else "search_document",
            "embedding_types": ["float"],
        }
        response = self._post(payload)
        return response["embeddings"]["float"]


class VoyageEmbedder(_BaseEmbedder):
    """Voyage ``/v1/embeddings`` (``voyage-3*``). ``input_type`` distinguishes document vs query."""

    provider = "voyage"
    default_url = "https://api.voyageai.com/v1/embeddings"

    def _embed(self, texts: Sequence[str], *, is_query: bool) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": list(texts),
            "input_type": "query" if is_query else "document",
        }
        response = self._post(payload)
        ordered = sorted(response["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]


# --- rerankers ---------------------------------------------------------------------------------


class _BaseReranker(_Connector):
    """Common reranker machinery: request all documents scored, realign to input order (§5.4).

    Subclasses implement :meth:`_rerank_response` (one provider call) + :attr:`_results_key` /
    :attr:`_top_param`. The provider returns results in RELEVANCE order carrying each doc's input
    ``index`` and ``relevance_score``; :meth:`rerank_scores` places each score back at its input
    index and returns a list ALIGNED to ``documents`` (so ``rerank_local`` reorders correctly). We
    request a score for EVERY document (``top_n``/``top_k`` = ``len(documents)``); a missing index
    (a truncated response) raises rather than defaulting to 0.
    """

    _results_key: str  # response key holding the per-doc results list
    _top_param: str  # request param capping returned results ("top_n" cohere / "top_k" voyage)

    def rerank_scores(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        response = self._post(
            {
                "model": self.model,
                "query": query,
                "documents": list(documents),
                self._top_param: len(documents),
            }
        )
        scores: list[float | None] = [None] * len(documents)
        for item in response[self._results_key]:
            scores[item["index"]] = float(item["relevance_score"])
        missing = [idx for idx, score in enumerate(scores) if score is None]
        if missing:
            raise ProviderError(
                self.provider, None, self.url,
                f"reranker returned no score for input indices {missing[:10]} "
                f"({len(missing)} of {len(documents)} documents unscored)",
            )
        return [score for score in scores if score is not None]


class CohereReranker(_BaseReranker):
    """Cohere ``/v2/rerank`` (``rerank-v3.5``). Results under ``results``; cap param ``top_n``."""

    provider = "cohere"
    default_url = "https://api.cohere.com/v2/rerank"
    _results_key = "results"
    _top_param = "top_n"


class VoyageReranker(_BaseReranker):
    """Voyage ``/v1/rerank`` (``rerank-2``/``rerank-2.5``). Results under ``data``; cap param ``top_k``."""

    provider = "voyage"
    default_url = "https://api.voyageai.com/v1/rerank"
    _results_key = "data"
    _top_param = "top_k"


# --- factories (dispatch on provider) ----------------------------------------------------------

#: Embedder providers. Source of truth for config-time validation (config.py mirrors these names).
_EMBEDDER_CLASSES: Mapping[str, type[_BaseEmbedder]] = {
    "openai": OpenAIEmbedder,
    "cohere": CohereEmbedder,
    "voyage": VoyageEmbedder,
}
#: Reranker providers — OpenAI is DELIBERATELY absent (it has no reranker, §3.4).
_RERANKER_CLASSES: Mapping[str, type[_BaseReranker]] = {
    "cohere": CohereReranker,
    "voyage": VoyageReranker,
}

EMBEDDER_PROVIDERS = frozenset(_EMBEDDER_CLASSES)
RERANKER_PROVIDERS = frozenset(_RERANKER_CLASSES)


def make_embedder(name: str, provider: str, settings: Mapping[str, Any]) -> _BaseEmbedder:
    """Build the embedding connector for ``provider`` (§3.4). Unknown provider raises (exhaustive)."""
    cls = _EMBEDDER_CLASSES.get(provider)
    if cls is None:
        raise ValueError(
            f"unknown embedder provider {provider!r}; expected one of {sorted(_EMBEDDER_CLASSES)}"
        )
    return cls(name, settings)


def make_reranker(name: str, provider: str, settings: Mapping[str, Any]) -> _BaseReranker:
    """Build the rerank connector for ``provider`` (§5.4). Unknown / openai provider raises."""
    cls = _RERANKER_CLASSES.get(provider)
    if cls is None:
        raise ValueError(
            f"unknown reranker provider {provider!r}; expected one of {sorted(_RERANKER_CLASSES)} "
            "(openai has no reranker)"
        )
    return cls(name, settings)
