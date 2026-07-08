"""Offline tests for the P1-3 stage-latency profiling Decorators (docs/architecture.md §5.5).

The timing Decorators wrap the ``Searcher`` / ``Reranker`` seams and record per-call wall-clock
WITHOUT changing the result — pass-through fakes prove the wrapped output equals the inner's and that
one sample is recorded per call (retrieval batch-amortized, rerank per query, SF-3). ``summarize``
converts seconds -> ms stats; ``time.perf_counter`` is patched so the assertions are deterministic.
"""

from __future__ import annotations

from typing import Sequence

import pytest

from benchmark.common import profiling
from benchmark.common.models import ScoredDoc
from benchmark.common.profiling import TimingReranker, TimingSearcher, summarize
from benchmark.common.protocols import Reranker, Searcher

_DOCS = [ScoredDoc("d1", 3.0), ScoredDoc("d2", 2.0), ScoredDoc("d3", 1.0)]


class _Leaf(Searcher):
    """A pass-through leaf returning a canned list truncated to top_k (records nothing)."""

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        return _DOCS[:top_k]


class _Rev(Reranker):
    """A reranker whose rule is 'reverse' (so a rerank pass is observable through the timer)."""

    def rerank(self, query: str, candidates: Sequence[ScoredDoc]) -> list[ScoredDoc]:
        return list(reversed(candidates))


def _fake_clock(monkeypatch: pytest.MonkeyPatch, ticks: list[float]) -> None:
    """Patch ``perf_counter`` to return ``ticks`` in order (start/stop pairs -> exact durations)."""
    seq = iter(ticks)
    monkeypatch.setattr(profiling, "perf_counter", lambda: next(seq))


def test_timing_searcher_passes_through_and_records_one_sample_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_clock(monkeypatch, [0.0, 0.5, 1.0, 1.2])  # search: 0.5s; bulk_search: 0.2s
    timed = TimingSearcher(_Leaf())

    assert timed.search("q", top_k=2) == _DOCS[:2]  # result unchanged
    # The inner's bulk_search (ABC default) loops its OWN search internally; the timer wraps the whole
    # batch in ONE sample (batch-amortized, SF-3), not one per query.
    assert timed.bulk_search(["q1", "q2"], top_k=3) == [_DOCS, _DOCS]
    assert timed.samples == pytest.approx([0.5, 0.2])  # one sample per timer call


def test_timing_reranker_records_one_sample_per_query(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_clock(monkeypatch, [0.0, 0.3, 10.0, 10.1])  # two rerank calls: 0.3s, 0.1s
    timed = TimingReranker(_Rev())

    assert timed.rerank("q1", _DOCS) == list(reversed(_DOCS))  # result unchanged
    timed.rerank("q2", _DOCS)
    assert timed.samples == pytest.approx([0.3, 0.1])  # PER-QUERY samples (the cost driver)


def test_summarize_reports_ms_and_percentiles() -> None:
    # samples in seconds -> ms; p50 = median, p95 near the top (linear-interpolated).
    result = summarize([0.010, 0.020, 0.030, 0.040])
    assert result["n"] == 4
    assert result["total_ms"] == pytest.approx(100.0)
    assert result["p50_ms"] == pytest.approx(25.0)  # median of 10,20,30,40
    assert result["p95_ms"] == pytest.approx(38.5)  # 0.95*(3) = 2.85 -> between 30 and 40


def test_summarize_empty_is_zeros() -> None:
    assert summarize([]) == {"n": 0, "total_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
