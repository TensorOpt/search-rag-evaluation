"""Offline unit tests for the provider connectors (docs/experiment.md §3.4, §5.4).

``urllib.request.urlopen`` is MOCKED — NO network. Covers, per provider: request URL + method +
bearer auth + JSON body shape; response parsing/index-alignment; document-vs-query input mode;
batching (sub-chunking beyond ``batch_size``); ``dim`` probe vs explicit ``settings.dims``; the
one-score-per-input guard; rerank index-realignment + the missing-index guard; the ``RateLimiter``
spacing; retry-on-429-then-success (+ ``Retry-After``); non-retryable + exhausted-retry surfacing as
``ProviderError``; and the ``make_embedder``/``make_reranker`` provider dispatch (openai has no
reranker).
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from benchmark import providers
from benchmark.providers import (
    CohereEmbedder,
    CohereReranker,
    OpenAIEmbedder,
    ProviderError,
    RateLimiter,
    VoyageEmbedder,
    VoyageReranker,
    make_embedder,
    make_reranker,
)


class _Resp:
    """A fake urlopen return: a context manager whose ``read()`` yields the JSON body bytes."""

    def __init__(self, payload: Any) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


def _http_error(code: int, body: str = '{"error":"boom"}', headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.example/x", code, "msg", headers or {}, io.BytesIO(body.encode()))


def _install(monkeypatch: pytest.MonkeyPatch, responses: list[Any]) -> list[urllib.request.Request]:
    """Patch ``urlopen`` to return/raise ``responses`` in order; return the captured Request list.

    Also no-ops ``time.sleep`` so retry/backoff tests don't actually wait.
    """
    calls: list[urllib.request.Request] = []
    queue = list(responses)

    def fake_urlopen(request: urllib.request.Request, timeout: Any = None) -> Any:
        calls.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Resp(item)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(providers.time, "sleep", lambda _seconds: None)
    return calls


def _body(request: urllib.request.Request) -> dict[str, Any]:
    return json.loads(request.data)  # type: ignore[arg-type]


# --- OpenAI embedder --------------------------------------------------------------------------


def test_openai_embed_request_shape_and_index_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    # response returned OUT of index order -> must be realigned by "index".
    calls = _install(monkeypatch, [{"data": [
        {"index": 1, "embedding": [0.3, 0.4]},
        {"index": 0, "embedding": [0.1, 0.2]},
    ]}])
    embedder = OpenAIEmbedder("oai", {"api_key": "sk-test", "model_id": "text-embedding-3-small"})

    vectors = embedder.embed_documents(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]  # realigned by index
    request = calls[0]
    assert request.full_url == "https://api.openai.com/v1/embeddings"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer sk-test"
    assert request.get_header("Content-type") == "application/json"
    payload = _body(request)
    assert payload == {"model": "text-embedding-3-small", "input": ["a", "b"], "encoding_format": "float"}


def test_openai_dims_setting_sent_as_dimensions_and_skips_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install(monkeypatch, [{"data": [{"index": 0, "embedding": [0.0, 0.0, 0.0]}]}])
    embedder = OpenAIEmbedder("oai", {"api_key": "k", "model_id": "text-embedding-3-small", "dims": 3})

    assert embedder.dim == 3  # from settings.dims — no probe call
    assert calls == []  # dim did NOT hit the network

    embedder.embed_queries(["q"])
    assert _body(calls[0])["dimensions"] == 3  # dims forwarded to OpenAI as the output-dim param


# --- Cohere embedder --------------------------------------------------------------------------


def test_cohere_embed_input_type_document_vs_query(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install(monkeypatch, [
        {"embeddings": {"float": [[1.0, 2.0]]}},
        {"embeddings": {"float": [[3.0, 4.0]]}},
    ])
    embedder = CohereEmbedder("co", {"api_key": "k", "model_id": "embed-english-v3.0"})

    assert embedder.embed_documents(["d"]) == [[1.0, 2.0]]
    assert embedder.embed_queries(["q"]) == [[3.0, 4.0]]

    assert calls[0].full_url == "https://api.cohere.com/v2/embed"
    doc_body = _body(calls[0])
    assert doc_body["texts"] == ["d"]
    assert doc_body["input_type"] == "search_document"
    assert doc_body["embedding_types"] == ["float"]
    assert _body(calls[1])["input_type"] == "search_query"


# --- Voyage embedder --------------------------------------------------------------------------


def test_voyage_embed_input_type_and_index_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install(monkeypatch, [{"data": [
        {"index": 0, "embedding": [1.0]},
        {"index": 1, "embedding": [2.0]},
    ]}])
    embedder = VoyageEmbedder("vo", {"api_key": "k", "model_id": "voyage-3.5"})

    assert embedder.embed_queries(["q0", "q1"]) == [[1.0], [2.0]]
    request = calls[0]
    assert request.full_url == "https://api.voyageai.com/v1/embeddings"
    payload = _body(request)
    assert payload["input"] == ["q0", "q1"]
    assert payload["input_type"] == "query"  # embed_queries -> query mode


# --- batching + dim probe + count guard -------------------------------------------------------


def test_embed_batches_sub_chunk_and_concatenate_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    # batch_size 2 over 3 texts -> TWO provider calls; vectors concatenated in input order.
    calls = _install(monkeypatch, [
        {"data": [{"index": 0, "embedding": [0.0]}, {"index": 1, "embedding": [1.0]}]},
        {"data": [{"index": 0, "embedding": [2.0]}]},
    ])
    embedder = OpenAIEmbedder("oai", {"api_key": "k", "model_id": "m", "batch_size": 2})

    vectors = embedder.embed_documents(["a", "b", "c"])

    assert vectors == [[0.0], [1.0], [2.0]]
    assert len(calls) == 2
    assert _body(calls[0])["input"] == ["a", "b"]
    assert _body(calls[1])["input"] == ["c"]


def test_dim_probe_when_no_dims_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install(monkeypatch, [{"data": [{"index": 0, "embedding": [0.0, 0.0, 0.0, 0.0]}]}])
    embedder = OpenAIEmbedder("oai", {"api_key": "k", "model_id": "m"})

    assert embedder.dim == 4  # probed once
    assert len(calls) == 1  # one probe request
    assert _body(calls[0])["input"] == [providers._DIM_PROBE_TEXT]
    assert embedder.dim == 4  # cached — no second probe
    assert len(calls) == 1


def test_embed_count_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # provider returns fewer vectors than inputs (a truncated batch) -> surface, never pad.
    _install(monkeypatch, [{"data": [{"index": 0, "embedding": [1.0]}]}])
    embedder = OpenAIEmbedder("oai", {"api_key": "k", "model_id": "m"})
    with pytest.raises(ProviderError, match="expected 2 results, got 1"):
        embedder.embed_documents(["a", "b"])


# --- rerankers --------------------------------------------------------------------------------


def test_cohere_rerank_realigns_by_index_and_requests_all(monkeypatch: pytest.MonkeyPatch) -> None:
    # results in RELEVANCE order (not input order) -> realign to input by "index".
    calls = _install(monkeypatch, [{"results": [
        {"index": 2, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.1},
        {"index": 1, "relevance_score": 0.5},
    ]}])
    reranker = CohereReranker("co", {"api_key": "k", "model_id": "rerank-v3.5"})

    scores = reranker.rerank_scores("q", ["d0", "d1", "d2"])

    assert scores == [0.1, 0.5, 0.9]  # aligned to input order
    request = calls[0]
    assert request.full_url == "https://api.cohere.com/v2/rerank"
    payload = _body(request)
    assert payload["query"] == "q"
    assert payload["documents"] == ["d0", "d1", "d2"]
    assert payload["top_n"] == 3  # request a score for EVERY document


def test_voyage_rerank_uses_data_and_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install(monkeypatch, [{"data": [
        {"index": 0, "relevance_score": 0.2},
        {"index": 1, "relevance_score": 0.8},
    ]}])
    reranker = VoyageReranker("vo", {"api_key": "k", "model_id": "rerank-2.5"})

    assert reranker.rerank_scores("q", ["a", "b"]) == [0.2, 0.8]
    assert calls[0].full_url == "https://api.voyageai.com/v1/rerank"
    assert _body(calls[0])["top_k"] == 2


def test_rerank_missing_index_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # provider scored only 2 of 3 documents -> a missing input index must surface, not default to 0.
    _install(monkeypatch, [{"results": [
        {"index": 0, "relevance_score": 0.1},
        {"index": 2, "relevance_score": 0.9},
    ]}])
    reranker = CohereReranker("co", {"api_key": "k", "model_id": "m"})
    with pytest.raises(ProviderError, match="no score for input indices"):
        reranker.rerank_scores("q", ["d0", "d1", "d2"])


def test_rerank_empty_documents_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install(monkeypatch, [])
    reranker = CohereReranker("co", {"api_key": "k", "model_id": "m"})
    assert reranker.rerank_scores("q", []) == []
    assert calls == []  # no round trip for an empty candidate list


# --- rate limiter -----------------------------------------------------------------------------


def test_rate_limiter_spaces_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(providers.time, "monotonic", lambda: 0.0)  # clock frozen at 0
    monkeypatch.setattr(providers.time, "sleep", lambda seconds: slept.append(seconds))

    limiter = RateLimiter(requests_per_minute=60)  # min interval 1.0s
    limiter.acquire()  # first: no wait (now >= next_allowed)
    limiter.acquire()  # second: must wait the full interval
    assert slept == [pytest.approx(1.0)]


def test_rate_limiter_none_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(providers.time, "sleep", lambda seconds: slept.append(seconds))
    limiter = RateLimiter(requests_per_minute=None)
    limiter.acquire()
    limiter.acquire()
    assert slept == []  # disabled


# --- retry + error surfacing ------------------------------------------------------------------


def test_retry_on_429_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(providers.time, "sleep", lambda seconds: slept.append(seconds))
    # first attempt 429 (with Retry-After), then success.
    calls: list[urllib.request.Request] = []
    queue: list[Any] = [_http_error(429, headers={"Retry-After": "2"}), {"data": [{"index": 0, "embedding": [1.0]}]}]

    def fake_urlopen(request: urllib.request.Request, timeout: Any = None) -> Any:
        calls.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Resp(item)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    embedder = OpenAIEmbedder("oai", {"api_key": "k", "model_id": "m"})

    assert embedder.embed_documents(["a"]) == [[1.0]]
    assert len(calls) == 2  # retried once
    assert slept == [pytest.approx(2.0)]  # honored Retry-After


def test_non_retryable_status_surfaces_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, [_http_error(401, body='{"error":"bad key"}')])
    embedder = CohereEmbedder("co", {"api_key": "wrong", "model_id": "m"})
    with pytest.raises(ProviderError) as excinfo:
        embedder.embed_documents(["a"])
    assert excinfo.value.status == 401
    assert "bad key" in excinfo.value.body  # the raw provider body is preserved for inspection


def test_retries_exhausted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # every attempt 503 -> after max_retries the last error surfaces as a ProviderError.
    _install(monkeypatch, [_http_error(503) for _ in range(10)])
    embedder = CohereEmbedder("co", {"api_key": "k", "model_id": "m", "max_retries": 2})
    with pytest.raises(ProviderError) as excinfo:
        embedder.embed_documents(["a"])
    assert excinfo.value.status == 503


# --- factory dispatch -------------------------------------------------------------------------


def test_make_embedder_dispatches_by_provider() -> None:
    assert isinstance(make_embedder("e", "openai", {"api_key": "k", "model_id": "m"}), OpenAIEmbedder)
    assert isinstance(make_embedder("e", "cohere", {"api_key": "k", "model_id": "m"}), CohereEmbedder)
    assert isinstance(make_embedder("e", "voyage", {"api_key": "k", "model_id": "m"}), VoyageEmbedder)


def test_make_embedder_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="unknown embedder provider"):
        make_embedder("e", "elasticsearch", {"api_key": "k", "model_id": "m"})


def test_make_reranker_dispatches_and_rejects_openai() -> None:
    assert isinstance(make_reranker("r", "cohere", {"api_key": "k", "model_id": "m"}), CohereReranker)
    assert isinstance(make_reranker("r", "voyage", {"api_key": "k", "model_id": "m"}), VoyageReranker)
    with pytest.raises(ValueError, match="openai has no reranker"):
        make_reranker("r", "openai", {"api_key": "k", "model_id": "m"})


def test_connector_requires_api_key_and_model_id() -> None:
    with pytest.raises(ValueError, match="requires settings.api_key"):
        OpenAIEmbedder("oai", {"model_id": "m"})
    with pytest.raises(ValueError, match="requires settings.model_id"):
        CohereReranker("co", {"api_key": "k"})
