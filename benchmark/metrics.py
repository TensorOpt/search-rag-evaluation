"""Evaluator, Metrics, and QrelIndex over graded relevance (docs/experiment.md §7). Phase 2.

Per-query metrics at cutoff k=10, computed once and returned in memory keyed by ``query_id``
so §8 statistics and the §8.0a ``best_per_model`` selection reuse them without re-parsing CSV.

**Missing-judgement policy — condensed-list evaluation (§7, Sakai).** A returned doc that has a
qrel entry uses its float gain (``0.0``/``0.5``/``1.0``); a returned doc with NO qrel entry has
gain ``math.nan`` (MISSING). A MISSING judgement is NOT "irrelevant": it is *skipped*. The
CONDENSED list is the query's ranked returned docs with the MISSING ones dropped, judged docs
kept in original rank order. Metrics are computed over the **condensed top-10** — the first
``min(10, #judged-in-list)`` judged docs — which MAY reach past original rank 10 to fill up to
10 judged docs. A JUDGED-irrelevant doc (gain ``0.0``, present in qrels) is KEPT in the condensed
list: it counts toward ``n_scored`` and contributes ``2^0-1 = 0`` to DCG; only docs with no qrel
entry are skipped.

Per query we also record two counts:

- ``n_scored`` = size of the condensed top-10 (the judged docs the metrics were computed over,
  ``<= 10``) — "total number this was calculated from".
- ``n_missing`` = number of MISSING docs skipped while scanning the ranked list to collect that
  condensed top-10 (count of NaN-gain docs in the scanned prefix: rank 1 up to and including the
  10th judged doc, or the whole returned list if fewer than 10 judged docs exist) — "number where
  the judgement was missing".

All four metrics follow §7, computed over the condensed top-10 gains ``g_1..g_m`` (condensed rank
order, ``m = n_scored``, all judged):

- ``avg_relevance`` = ``(1/m)·Σ_{i=1..m} g_i``; ``math.nan`` if ``m == 0``.
- ``ndcg@10``: ``DCG = Σ_{i=1..m} (2^{g_i}−1)/log2(i+1)`` using CONDENSED positions ``1..m``;
  ``IDCG`` = DCG of the query's judged gains sorted descending, truncated to the top-10 (over ALL
  judged gains for the query, unaffected by skipping); ``nDCG = DCG/IDCG``, ``0.0`` when
  ``IDCG == 0``; ``math.nan`` if ``m == 0``.
- ``precision@10`` = ``(#relevant in condensed top-10) / m`` (denominator is ``m = n_scored``, NOT
  10); relevant iff gain ``>= 0.5``; ``math.nan`` if ``m == 0``.
- ``recall@10`` = ``(#relevant in condensed top-10) / R``, where ``R`` = #relevant judged docs for
  the query over ALL qrels; ``math.nan`` if ``R == 0``.

Each metric may independently be ``math.nan``: ``avg_relevance``/``ndcg@10``/``precision@10`` when
``n_scored == 0``; ``recall@10`` when ``R == 0``. The comparator excludes NaN queries per metric
(§8.1). ``math.nan`` is the identical float value to ``np.nan``; this module stays stdlib-only.
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

#: Absolute tolerance for treating a computed float as zero (never use `== 0.0` on floats).
ZERO_ABS_TOL = 1e-6


def _dcg(gains: Iterable[float]) -> float:
    """Discounted cumulative gain: Σ (2^{gain}−1)/log2(i+1), i 1-based (§7)."""
    return sum((2.0**g - 1.0) / math.log2(i + 1) for i, g in enumerate(gains, start=1))


class QrelIndex:
    """Judged gains indexed as ``dict[query_id, dict[doc_id, gain]]`` (§6, §7).

    Built once from an iterable of ``Qrel``. Provides gain lookup (``math.nan`` when there is NO
    qrel entry — a MISSING judgement, condensed-list skipped, §7), the per-query count of relevant
    judged docs (gain >= 0.5, for recall's R), and the per-query judged gains sorted descending
    (for IDCG). A judged 0.0 is a real judgement (not relevant, contributes 0 to IDCG) — only the
    absence of an entry is MISSING.
    """

    def __init__(self, qrels: Iterable[Qrel]) -> None:
        index: dict[str, dict[str, float]] = {}
        for qr in qrels:
            index.setdefault(qr.query_id, {})[qr.doc_id] = qr.gain
        self._index = index

    def gain(self, query_id: str, doc_id: str) -> float:
        """Judged gain for ``(query_id, doc_id)``; ``math.nan`` if there is NO qrel entry (§7)."""
        return self._index.get(query_id, {}).get(doc_id, math.nan)

    def relevant_count(self, query_id: str) -> int:
        """R: number of relevant judged docs (gain >= 0.5) for the query over all qrels (§7)."""
        return sum(
            1 for g in self._index.get(query_id, {}).values() if g >= RELEVANCE_THRESHOLD
        )

    def sorted_judged_gains(self, query_id: str) -> list[float]:
        """This query's judged gains sorted descending — the ideal ordering (§7, for IDCG)."""
        return sorted(self._index.get(query_id, {}).values(), reverse=True)


@dataclass(frozen=True)
class Metrics:
    """The four per-query metrics plus the condensed-list counts (§7).

    The four metric fields are floats and may each independently be ``math.nan`` (see the module
    docstring for the per-metric NaN conditions). ``as_dict()`` exposes only the four metrics under
    the canonical §9 metric-name keys that the comparator and CSV writers key on; the two counts
    (``n_scored``, ``n_missing``) are non-negative ints exposed as fields, not in ``as_dict()``.
    """

    avg_relevance: float
    ndcg_at_10: float
    recall_at_10: float
    precision_at_10: float
    n_scored: int
    n_missing: int

    def as_dict(self) -> Mapping[str, float]:
        """Map the four metrics to the canonical §9 metric-name keys (CSV columns / §8 keys)."""
        return {
            "avg_relevance": self.avg_relevance,
            "ndcg@10": self.ndcg_at_10,
            "recall@10": self.recall_at_10,
            "precision@10": self.precision_at_10,
        }


class Evaluator:
    """Scores runs against qrels, producing per-query ``Metrics`` (§7, §6 step 5)."""

    def __init__(self, qrels: QrelIndex, *, cutoff: int = DEFAULT_CUTOFF) -> None:
        self._qrels = qrels
        self._cutoff = cutoff

    def score_run(self, results: Iterable[RankedResult]) -> dict[str, Metrics]:
        """Score each ``RankedResult`` (joined to qrels by ``query_id``), keyed by query_id (§6)."""
        return {rr.query_id: self._score_one(rr) for rr in results}

    def _score_one(self, result: RankedResult) -> Metrics:
        k = self._cutoff
        qid = result.query_id

        # Scan the ranked list, collecting the condensed top-k: keep JUDGED docs (finite gain) in
        # rank order, SKIP MISSING docs (NaN gain, no qrel entry). Stop once k judged docs are
        # collected — this may reach past original rank k. n_missing counts the MISSING docs seen
        # in the scanned prefix (up to and including the k-th judged doc, or the whole list).
        condensed: list[float] = []
        n_missing = 0
        for d in result.docs:
            g = self._qrels.gain(qid, d.doc_id)
            if math.isnan(g):
                n_missing += 1
            else:
                condensed.append(g)
                if len(condensed) >= k:
                    break

        m = len(condensed)  # n_scored

        if m == 0:
            avg_relevance = math.nan
            ndcg = math.nan
            precision = math.nan
        else:
            avg_relevance = sum(condensed) / m
            # DCG over the condensed positions 1..m; judged-irrelevant (0.0) contributes 0.
            dcg = _dcg(condensed)
            # IDCG: DCG of the top-k ideal ordering (ALL judged gains desc, truncated to k, §7);
            # unaffected by skipping missing docs.
            idcg = _dcg(self._qrels.sorted_judged_gains(qid)[:k])
            # isclose (not `!= 0.0`) — never test float equality; IDCG==0 means no positive gains.
            ndcg = 0.0 if math.isclose(idcg, 0.0, abs_tol=ZERO_ABS_TOL) else dcg / idcg
            hits = sum(1 for g in condensed if g >= RELEVANCE_THRESHOLD)
            precision = hits / m  # denominator = n_scored (§7).

        # recall@10 uses R (relevant judged docs over ALL qrels) and the condensed-top-k hits.
        r = self._qrels.relevant_count(qid)
        hits = sum(1 for g in condensed if g >= RELEVANCE_THRESHOLD)
        recall = hits / r if r > 0 else math.nan  # R == 0 -> NaN (§7).

        return Metrics(
            avg_relevance=avg_relevance,
            ndcg_at_10=ndcg,
            recall_at_10=recall,
            precision_at_10=precision,
            n_scored=m,
            n_missing=n_missing,
        )
