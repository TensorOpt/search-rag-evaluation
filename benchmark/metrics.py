"""Evaluator, MetricVector, and QrelIndex over graded relevance (docs/experiment.md §7). Phase 2.

Per-query metrics at cutoff k=10, computed once and returned in memory keyed by ``query_id``
so §8 statistics and the §8.0a ``best_per_model`` selection reuse them without re-parsing CSV.

All four metrics follow §7 EXACTLY:

- ``avg_relevance`` = (1/10)·Σ_{i=1..10} gain(d_i); short lists are zero-padded at the gain
  level, denominator stays 10.
- ``ndcg@10``: DCG@10 = Σ_{i=1..10} (2^{gain(d_i)}−1)/log2(i+1); IDCG@10 is the DCG of the
  top-10 of the ideal (judged-gains sorted descending, truncated to 10 — so a query with more
  than 10 relevant docs is NOT deflated). nDCG@10 = DCG@10/IDCG@10, defined 0.0 when IDCG@10 == 0.
- ``recall@10`` = |relevant ∩ top-10| / R, where relevant iff gain >= 0.5 and R = number of
  relevant judged docs for the query over ALL qrels. When R == 0, recall is ``math.nan`` (the
  comparator excludes NaN queries, §8.1 — that exclusion is the comparator's job, not here).
- ``precision@10`` = |relevant ∩ top-10| / 10 (denominator fixed at 10).

Unjudged docs count as gain 0.0 (§7 TREC condensed-list assumption).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from benchmark.models import Qrel, RankedResult

#: Evaluation cutoff k (§7).
DEFAULT_CUTOFF = 10

#: Binary-relevance threshold: a doc is relevant iff gain >= this (§7, Partial or Exact).
RELEVANCE_THRESHOLD = 0.5


def _dcg(gains: Iterable[float]) -> float:
    """Discounted cumulative gain: Σ (2^{gain}−1)/log2(i+1), i 1-based (§7)."""
    return sum((2.0**g - 1.0) / math.log2(i + 1) for i, g in enumerate(gains, start=1))


class QrelIndex:
    """Judged gains indexed as ``dict[query_id, dict[doc_id, gain]]`` (§6, §7).

    Built once from an iterable of ``Qrel``. Provides gain lookup (0.0 for unjudged, the §7
    condensed-list assumption), the per-query count of relevant judged docs (gain >= 0.5, for
    recall's R), and the per-query judged gains sorted descending (for IDCG).
    """

    def __init__(self, qrels: Iterable[Qrel]) -> None:
        index: dict[str, dict[str, float]] = {}
        for qr in qrels:
            index.setdefault(qr.query_id, {})[qr.doc_id] = qr.gain
        self._index = index

    def gain(self, query_id: str, doc_id: str) -> float:
        """Judged gain for ``(query_id, doc_id)``; 0.0 if unjudged (§7)."""
        return self._index.get(query_id, {}).get(doc_id, 0.0)

    def relevant_count(self, query_id: str) -> int:
        """R: number of relevant judged docs (gain >= 0.5) for the query over all qrels (§7)."""
        return sum(
            1 for g in self._index.get(query_id, {}).values() if g >= RELEVANCE_THRESHOLD
        )

    def sorted_judged_gains(self, query_id: str) -> list[float]:
        """This query's judged gains sorted descending — the ideal ordering (§7, for IDCG)."""
        return sorted(self._index.get(query_id, {}).values(), reverse=True)


@dataclass(frozen=True)
class MetricVector:
    """The four per-query metrics as floats (§7).

    Field names are Python identifiers; ``as_dict()`` exposes the canonical §9 metric names
    (``"avg_relevance"``, ``"ndcg@10"``, ``"recall@10"``, ``"precision@10"``) that the
    comparator and CSV writers key on. ``recall_at_10`` may be ``math.nan`` when R == 0 (§7).
    """

    avg_relevance: float
    ndcg_at_10: float
    recall_at_10: float
    precision_at_10: float

    def as_dict(self) -> Mapping[str, float]:
        """Map to the canonical §9 metric-name keys (CSV columns / §8 metric keys)."""
        return {
            "avg_relevance": self.avg_relevance,
            "ndcg@10": self.ndcg_at_10,
            "recall@10": self.recall_at_10,
            "precision@10": self.precision_at_10,
        }


class Evaluator:
    """Scores runs against qrels, producing per-query ``MetricVector``s (§7, §6 step 5)."""

    def __init__(self, qrels: QrelIndex, *, cutoff: int = DEFAULT_CUTOFF) -> None:
        self._qrels = qrels
        self._cutoff = cutoff

    def score_run(self, results: Iterable[RankedResult]) -> dict[str, MetricVector]:
        """Score each ``RankedResult`` (joined to qrels by ``query_id``), keyed by query_id (§6)."""
        return {rr.query_id: self._score_one(rr) for rr in results}

    def _score_one(self, result: RankedResult) -> MetricVector:
        k = self._cutoff
        qid = result.query_id
        # Gains of the top-k returned docs (unjudged -> 0.0, §7).
        top_gains = [self._qrels.gain(qid, d.doc_id) for d in result.docs[:k]]

        # avg_relevance: mean over top-k, denominator fixed at k (short lists zero-padded, §7).
        avg_relevance = sum(top_gains) / k

        # DCG@10 over the (possibly short) returned list; missing positions contribute 0.
        dcg = _dcg(top_gains)
        # IDCG@10: DCG of the top-k ideal ordering (judged gains desc, truncated to k, §7).
        idcg = _dcg(self._qrels.sorted_judged_gains(qid)[:k])
        ndcg = dcg / idcg if idcg != 0.0 else 0.0

        # relevant iff gain >= 0.5 (§7).
        hits = sum(1 for g in top_gains if g >= RELEVANCE_THRESHOLD)
        precision = hits / k  # denominator fixed at k (§7).
        r = self._qrels.relevant_count(qid)
        recall = hits / r if r > 0 else math.nan  # R == 0 -> NaN (§7, excluded by comparator).

        return MetricVector(
            avg_relevance=avg_relevance,
            ndcg_at_10=ndcg,
            recall_at_10=recall,
            precision_at_10=precision,
        )
