"""Evaluator, Metrics, QrelIndex, qrels_digest over graded relevance (docs/methodology.md §7). Phase 2.

Per-query metrics computed once and returned in memory keyed by ``query_id`` so §8 statistics
reuse them without re-parsing CSV. The point/quality metrics (``avg_relevance``/``ndcg@10``/
``precision@10``) are cut at ``cutoff`` (=10); recall is reported at cutoffs ``{10, 50, 100}``.

**ONE uniform relevance policy governs ALL SIX metrics (§7, P0-2).** A returned doc with a qrel
entry uses its float gain (``0.0``/``0.5``/``1.0``); a returned doc with NO qrel entry is MISSING
(``math.nan``). ``metrics.unjudged`` (injected as ``unjudged``) selects how MISSING is handled,
UNIFORMLY across every metric — there is no per-metric carve-out:

- ``"condensed"`` (Sakai deletion, the shipped default): the eval list is the query's ranked
  returned docs with the MISSING ones DROPPED, judged docs kept in original rank order. A
  JUDGED-irrelevant doc (gain ``0.0``, present in qrels) is KEPT (it contributes ``2^0-1 = 0`` to
  DCG and is not a relevant hit); only docs with no qrel entry are dropped. recall reverts to the
  condensed list under this policy.
- ``"irrelevant"`` (trec_eval): the eval list is the raw retrieved positions with each MISSING doc
  scored as gain ``0.0`` in place (kept, not dropped). ``precision@k`` denom is then ``k`` (the
  raw top-k includes unjudged docs); recall is the standard position recall.

The Evaluator builds ONE full eval list from the policy, scanned to ``max(RECALL_CUTOFFS)`` judged
(condensed) / positions (irrelevant), then each metric SLICES it: the point metrics take
``eval_list[:cutoff]`` (cutoff=10), recall@k takes ``eval_list[:k]``. Let ``g_1..g_m`` be the point
slice (``m = len(eval_list[:cutoff])``):

- ``avg_relevance`` = ``(1/m)·Σ g_i``; ``math.nan`` if ``m == 0``.
- ``ndcg@10``: ``DCG = Σ (2^{g_i}−1)/log2(i+1)`` over positions ``1..m``; ``IDCG`` = DCG of the
  query's JUDGED gains sorted descending, truncated to the top-10; ``nDCG = DCG/IDCG``, ``0.0`` when
  ``IDCG == 0``; ``math.nan`` if ``m == 0``.
- ``precision@10`` = ``(#{g_i >= threshold}) / m``. Under ``condensed`` ``m == n_scored``; under
  ``irrelevant`` ``m == min(cutoff, n_results)``.
- ``recall@k`` = ``|{g_i >= threshold in eval_list[:k]}| / R``, ``R`` = #relevant judged docs for
  the query over ALL qrels; ``math.nan`` iff ``R == 0``, else defined (0 for an empty/failed result).

The binary-relevance ``threshold`` (default 0.5; the runner injects ``metrics.relevance_threshold``)
is applied to every binary metric AND to ``QrelIndex.relevant_count`` / :func:`qrels_digest` (N-3).

Two per-query DIAGNOSTIC counts are recorded (ALWAYS condensed semantics, cutoff depth, in BOTH
policies):

- ``n_scored`` = size of the condensed top-cutoff (judged docs, ``<= cutoff``).
- ``n_missing`` = number of MISSING docs skipped while scanning to the cutoff-th judged doc.

``n_relevant`` = ``|R|`` (P2-3), the relevant-set size under the resolved threshold.

Each metric may independently be ``math.nan``: ``avg_relevance``/``ndcg@10``/``precision@10`` when
``m == 0``; every ``recall@k`` when ``R == 0``. The comparator excludes NaN queries per metric
(§8.1). ``math.nan`` is the identical float value to ``np.nan``; this module stays stdlib-only.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from benchmark.common.models import Qrel, RankedResult

#: Evaluation cutoff k for the point/quality metrics (§7).
DEFAULT_CUTOFF = 10

#: Recall cutoffs (§7). recall@k slices the SAME policy eval list at each k, so the eval list is
#: scanned to a depth of ``max(RECALL_CUTOFFS)`` (deep enough for every metric's slice).
RECALL_CUTOFFS = (10, 50, 100)

#: Default binary-relevance threshold: a doc is relevant iff gain >= this (§7, Partial or Exact).
#: The runner injects the configured ``metrics.relevance_threshold`` into BOTH the Evaluator and the
#: QrelIndex; this constant is only the standalone default for tests / bare construction.
RELEVANCE_THRESHOLD = 0.5

#: Default unjudged policy (§7, P0-2): ``"condensed"`` (Sakai) or ``"irrelevant"`` (trec_eval).
DEFAULT_UNJUDGED = "condensed"

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

    def __init__(
        self, qrels: Iterable[Qrel], *, relevance_threshold: float = RELEVANCE_THRESHOLD
    ) -> None:
        index: dict[str, dict[str, float]] = {}
        for qr in qrels:
            index.setdefault(qr.query_id, {})[qr.doc_id] = qr.gain
        self._index = index
        # N-3: the injected threshold feeds BOTH R (recall denom) AND the P0-3 digest, so it must not
        # silently keep 0.5 while the metrics use the config threshold.
        self._threshold = relevance_threshold

    def gain(self, query_id: str, doc_id: str) -> float:
        """Judged gain for ``(query_id, doc_id)``; ``math.nan`` if there is NO qrel entry (§7)."""
        return self._index.get(query_id, {}).get(doc_id, math.nan)

    def relevant_count(self, query_id: str) -> int:
        """R: number of relevant judged docs (gain >= threshold) for the query over all qrels (§7)."""
        return sum(1 for g in self._index.get(query_id, {}).values() if g >= self._threshold)

    def sorted_judged_gains(self, query_id: str) -> list[float]:
        """This query's judged gains sorted descending — the ideal ordering (§7, for IDCG)."""
        return sorted(self._index.get(query_id, {}).values(), reverse=True)


@dataclass(frozen=True)
class Metrics:
    """The six per-query metrics plus the condensed-list counts (§7).

    The six metric fields are floats and may each independently be ``math.nan`` (see the module
    docstring for the per-metric NaN conditions). ``as_dict()`` exposes only the six metrics under
    the canonical §9 metric-name keys that the comparator and CSV writers key on; the four counts
    (``n_results``, ``n_scored``, ``n_missing``, ``n_relevant``) are non-negative ints exposed as
    fields, not in ``as_dict()``. ``n_relevant`` (P2-3) is ``|R|``, the per-query relevant-set size
    under the resolved threshold. Field order matches the §9 metrics-CSV column order.
    """

    avg_relevance: float
    ndcg_at_10: float
    recall_at_10: float
    recall_at_50: float
    recall_at_100: float
    precision_at_10: float
    n_results: int
    n_scored: int
    n_missing: int
    n_relevant: int

    def as_dict(self) -> Mapping[str, float]:
        """Map the six metrics to the canonical §9 metric-name keys (CSV columns / §8 keys)."""
        return {
            "avg_relevance": self.avg_relevance,
            "ndcg@10": self.ndcg_at_10,
            "recall@10": self.recall_at_10,
            "recall@50": self.recall_at_50,
            "recall@100": self.recall_at_100,
            "precision@10": self.precision_at_10,
        }


class Evaluator:
    """Scores runs against qrels, producing per-query ``Metrics`` (§7, §6 step 5)."""

    def __init__(
        self,
        qrels: QrelIndex,
        *,
        cutoff: int = DEFAULT_CUTOFF,
        unjudged: str = DEFAULT_UNJUDGED,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
    ) -> None:
        if unjudged not in ("condensed", "irrelevant"):
            # Exhaustive on the enumerated policy — no silent default for an invalid value.
            raise ValueError(
                f"unknown unjudged policy {unjudged!r}; expected 'condensed' or 'irrelevant'"
            )
        self._qrels = qrels
        self._cutoff = cutoff
        self._unjudged = unjudged
        self._threshold = relevance_threshold

    def score_run(self, results: Iterable[RankedResult]) -> dict[str, Metrics]:
        """Score each ``RankedResult`` (joined to qrels by ``query_id``), keyed by query_id (§6)."""
        return {rr.query_id: self._score_one(rr) for rr in results}

    def _score_one(self, result: RankedResult) -> Metrics:
        cutoff = self._cutoff
        qid = result.query_id
        n_results = len(result.docs)
        threshold = self._threshold

        gains = [self._qrels.gain(qid, d.doc_id) for d in result.docs]

        # Diagnostic condensed-scan counts (n_scored, n_missing) — ALWAYS condensed semantics at
        # ``cutoff`` depth, in BOTH policies (§7): keep JUDGED docs, SKIP MISSING, stop at the
        # cutoff-th judged doc (may reach past rank cutoff). n_missing counts the MISSING docs seen
        # in that scanned prefix.
        n_scored = 0
        n_missing = 0
        for gain in gains:
            if math.isnan(gain):
                n_missing += 1
            else:
                n_scored += 1
                if n_scored >= cutoff:
                    break

        # ONE full policy eval list (§7, P0-2), scanned to max(RECALL_CUTOFFS) so every metric's
        # slice is deep enough. condensed = judged docs in rank order (MISSING dropped); irrelevant =
        # raw positions with MISSING scored as gain 0.0 (kept, not dropped). Each metric then slices.
        max_depth = max(RECALL_CUTOFFS)
        if self._unjudged == "condensed":
            eval_list = [gain for gain in gains if not math.isnan(gain)][:max_depth]
        else:  # "irrelevant" — validated in __init__; only these two policies exist.
            eval_list = [0.0 if math.isnan(gain) else gain for gain in gains][:max_depth]

        # Point/quality metrics over eval_list[:cutoff]; m = its length (n_scored under condensed,
        # min(cutoff, n_results) under irrelevant, §7). NaN iff m == 0 (empty/all-MISSING top-k).
        point = eval_list[:cutoff]
        m = len(point)
        if m == 0:
            avg_relevance = math.nan
            ndcg = math.nan
            precision = math.nan
        else:
            avg_relevance = sum(point) / m
            dcg = _dcg(point)  # DCG over positions 1..m; a 0.0 gain contributes 2^0-1 = 0.
            # IDCG: ideal over ALL judged gains desc, truncated to cutoff (judged-only, both policies).
            idcg = _dcg(self._qrels.sorted_judged_gains(qid)[:cutoff])
            # isclose (not `!= 0.0`) — never test float equality; IDCG==0 means no positive gains.
            ndcg = 0.0 if math.isclose(idcg, 0.0, abs_tol=ZERO_ABS_TOL) else dcg / idcg
            hits = sum(1 for gain in point if gain >= threshold)
            precision = hits / m

        # recall@k slices the SAME policy eval list at each k / R (§7). Under condensed this reverts
        # recall to the condensed list; under irrelevant it is the standard (position) recall. A
        # MISSING doc is never a relevant hit in either policy (condensed drops it; irrelevant scores
        # it 0.0 < threshold). R == 0 -> NaN; else defined for every cutoff (0 for an empty result).
        r = self._qrels.relevant_count(qid)
        recall_by_cutoff: dict[int, float] = {}
        for cutoff_k in RECALL_CUTOFFS:
            if r > 0:
                hits_k = sum(1 for gain in eval_list[:cutoff_k] if gain >= threshold)
                recall_by_cutoff[cutoff_k] = hits_k / r
            else:
                recall_by_cutoff[cutoff_k] = math.nan

        return Metrics(
            avg_relevance=avg_relevance,
            ndcg_at_10=ndcg,
            recall_at_10=recall_by_cutoff[10],
            recall_at_50=recall_by_cutoff[50],
            recall_at_100=recall_by_cutoff[100],
            precision_at_10=precision,
            n_results=n_results,
            n_scored=n_scored,
            n_missing=n_missing,
            n_relevant=r,
        )


def qrels_digest(qrels: Iterable[Qrel], *, relevance_threshold: float) -> str:
    """SHA-256 over the sorted ``(query_id, doc_id, gain)`` triples + the threshold (P0-3).

    A stable fingerprint of the loaded, gain-mapped qrels: two runs with the SAME judgements and
    threshold get the SAME digest; flipping any grade, adding/removing a triple, or changing the
    threshold changes it. Because the dataset adapter has already applied its label->gain map when
    emitting ``Qrel.gain``, hashing the RESOLVED float gains captures the gain mapping without the
    runner knowing dataset specifics (Generality). Runs with differing digests are not comparable.
    """
    hasher = hashlib.sha256()
    for query_id, doc_id, gain in sorted(
        (qr.query_id, qr.doc_id, qr.gain) for qr in qrels
    ):
        hasher.update(f"{query_id}\t{doc_id}\t{gain!r}\n".encode("utf-8"))
    hasher.update(repr(relevance_threshold).encode("utf-8"))
    return hasher.hexdigest()
