"""Opt-in stage-latency profiling Decorators (docs/architecture.md §5.5, §9.1; P1-3).

A cross-cutting infra layer like caching (§5.5) and logging (§11): imports only stdlib + the
``common`` seams — never a provider/backend/dataset — so it adds no forbidden import edge (§11,
enforced by ``tests/unit/test_import_graph.py``). Two Decorators, each explicitly subclassing the
seam it fulfills (CLAUDE.md "declare the interface"), time the stage they wrap when the runner turns
profiling on (``eval:run --profile``):

- :class:`TimingSearcher` (:class:`~benchmark.common.protocols.Searcher`) — records the wall-clock of
  each ``search`` / ``bulk_search`` call. Retrieval batches the WHOLE query set in one ``_msearch``
  (§5.3), so each sample is a **batch-amortized** total, NOT a per-query figure — the runner reports it
  as a total / per-query-average, never as retrieval p50/p95 (SF-3).
- :class:`TimingReranker` (:class:`~benchmark.common.protocols.Reranker`) — records the wall-clock of
  each PER-QUERY ``rerank`` call. Rerank is the cost driver and IS per-query, so its samples yield the
  meaningful p50/p95 (§5.4).

Profiling is **off by default** and never feeds cached values or metrics — the wrappers only observe
timing, so a standard run stays byte-identical (§9.1). Wall-clock is contaminated by the connector
``RateLimiter`` (``requests_per_minute``): the API call/doc/token counters (``inference._Connector``)
are the PRIMARY cost figure, latency is indicative (P1-3).
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Sequence

from benchmark.common.models import ScoredDoc
from benchmark.common.protocols import Reranker, Searcher


class TimingSearcher(Searcher):
    """Decorator over a leaf ``Searcher`` recording each call's wall-clock (batch-amortized, §5.3)."""

    def __init__(self, inner: Searcher) -> None:
        self._inner = inner
        #: Seconds per ``search`` / ``bulk_search`` call. One ``bulk_search`` covers the whole query
        #: set (one ``_msearch``), so a sample is a BATCH total, not a per-query time (SF-3).
        self.samples: list[float] = []

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        start = perf_counter()
        try:
            return self._inner.search(query, top_k=top_k)
        finally:
            self.samples.append(perf_counter() - start)

    def bulk_search(self, queries: Sequence[str], *, top_k: int) -> list[list[ScoredDoc]]:
        start = perf_counter()
        try:
            return self._inner.bulk_search(queries, top_k=top_k)
        finally:
            self.samples.append(perf_counter() - start)


class TimingReranker(Reranker):
    """Decorator over a ``Reranker`` recording each PER-QUERY ``rerank`` call's wall-clock (§5.4)."""

    def __init__(self, inner: Reranker) -> None:
        self._inner = inner
        #: Seconds per per-query ``rerank`` call — the cost driver, so p50/p95 are meaningful here.
        self.samples: list[float] = []

    def rerank(self, query: str, candidates: Sequence[ScoredDoc]) -> list[ScoredDoc]:
        start = perf_counter()
        try:
            return self._inner.rerank(query, candidates)
        finally:
            self.samples.append(perf_counter() - start)


def _percentile(sorted_ms: Sequence[float], q: float) -> float:
    """The ``q`` quantile (0..1) of an ASCENDING-sorted sample list, linear-interpolated.

    ponytail: linear-interpolated percentile over an in-memory sample list — fine for a per-run
    diagnostic. Swap for a streaming/quantile-sketch only if a run ever holds millions of samples.
    """
    if not sorted_ms:
        return 0.0
    if len(sorted_ms) == 1:
        return sorted_ms[0]
    pos = (len(sorted_ms) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_ms[lo]
    return sorted_ms[lo] * (hi - pos) + sorted_ms[hi] * (pos - lo)


def summarize(samples_s: Sequence[float]) -> dict[str, float]:
    """Summarize per-call durations (seconds) into ms stats: ``{n, total_ms, p50_ms, p95_ms}``.

    ``n`` is the sample count, ``total_ms`` the summed wall-clock. ``p50_ms``/``p95_ms`` are the
    percentiles over the per-call samples — meaningful for rerank (per query); for retrieval the
    runner uses ``total_ms`` / a per-query average instead (batch-amortized, SF-3).
    """
    ms = sorted(sample * 1000.0 for sample in samples_s)
    return {
        "n": len(ms),
        "total_ms": sum(ms),
        "p50_ms": _percentile(ms, 0.5),
        "p95_ms": _percentile(ms, 0.95),
    }
