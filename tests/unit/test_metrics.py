"""Phase 2 unit tests for benchmark.metrics (docs/experiment.md §7).

Every expected value is hand-computed; the arithmetic is written out in the test body so a
reviewer can recompute independently. Recall 2^0.5 - 1 == 0.41421356 (≈).
"""

from __future__ import annotations

import math

import pytest

from benchmark.metrics import Evaluator, MetricVector, QrelIndex
from benchmark.models import Qrel, RankedResult, ScoredDoc


def _rr(query_id: str, doc_ids: list[str]) -> RankedResult:
    """A RankedResult with descending placeholder scores (only the ORDER matters for metrics)."""
    n = len(doc_ids)
    return RankedResult(query_id=query_id, docs=[ScoredDoc(d, float(n - i)) for i, d in enumerate(doc_ids)])


# --- QrelIndex -----------------------------------------------------------------


def test_qrelindex_gain_unjudged_is_zero():
    idx = QrelIndex([Qrel("q1", "p1", 1.0)])
    assert idx.gain("q1", "p1") == 1.0
    assert idx.gain("q1", "p_missing") == 0.0  # unjudged doc, judged query
    assert idx.gain("q_missing", "p1") == 0.0  # unjudged query


def test_qrelindex_relevant_count_thresholds_at_half():
    idx = QrelIndex(
        [
            Qrel("q1", "p1", 1.0),  # relevant (Exact)
            Qrel("q1", "p2", 0.5),  # relevant (Partial)
            Qrel("q1", "p3", 0.0),  # not relevant (Irrelevant)
        ]
    )
    assert idx.relevant_count("q1") == 2  # only gain >= 0.5 counts
    assert idx.relevant_count("q_missing") == 0


def test_qrelindex_sorted_judged_gains_descending():
    idx = QrelIndex([Qrel("q1", "p1", 0.5), Qrel("q1", "p2", 1.0), Qrel("q1", "p3", 0.0)])
    assert idx.sorted_judged_gains("q1") == [1.0, 0.5, 0.0]
    assert idx.sorted_judged_gains("q_missing") == []


# --- MetricVector.as_dict canonical keys ---------------------------------------


def test_metricvector_as_dict_exact_canonical_keys():
    mv = MetricVector(avg_relevance=0.1, ndcg_at_10=0.2, recall_at_10=0.3, precision_at_10=0.4)
    d = mv.as_dict()
    assert set(d.keys()) == {"avg_relevance", "ndcg@10", "recall@10", "precision@10"}
    assert d["avg_relevance"] == 0.1
    assert d["ndcg@10"] == 0.2
    assert d["recall@10"] == 0.3
    assert d["precision@10"] == 0.4


# --- perfect ranking -> ndcg@10 == 1.0 -----------------------------------------


def test_perfect_ranking_ndcg_is_one():
    # Judged: p1=1.0, p2=0.5. Ideal order [1.0, 0.5]; the ranking returns exactly that order.
    qrels = QrelIndex([Qrel("q1", "p1", 1.0), Qrel("q1", "p2", 0.5)])
    mv = Evaluator(qrels).score_run([_rr("q1", ["p1", "p2"])])["q1"]
    assert math.isclose(mv.ndcg_at_10, 1.0, rel_tol=0.0, abs_tol=1e-12)


# --- mixed graded case with hand-computed DCG/IDCG -----------------------------


def test_graded_mixed_case_hand_computed():
    # Ranked list (positions 1,2,3): gains 1.0, 0.0, 0.5.
    #   DCG@10  = (2^1-1)/log2(2) + (2^0-1)/log2(3) + (2^0.5-1)/log2(4)
    #           = 1/1 + 0 + 0.41421356/2 = 1.2071067811865475
    # Judged gains for the query: {1.0, 1.0, 0.5, 0.0}; ideal desc [1.0, 1.0, 0.5, 0.0].
    #   IDCG@10 = 1/log2(2) + 1/log2(3) + (2^0.5-1)/log2(4) + 0/log2(5)
    #           = 1 + 0.6309297535714575 + 0.20710678 = 1.8380365347580052
    #   nDCG@10 = 1.2071067811865475 / 1.8380365347580052 = 0.6567370987244682
    #   avg_relevance = (1.0 + 0.0 + 0.5)/10 = 0.15
    #   precision@10  = 2 relevant (p_a=1.0, p_c=0.5) in top-10 / 10 = 0.2
    #   R = 3 relevant judged (two 1.0 + one 0.5); recall@10 = 2/3
    qrels = QrelIndex(
        [
            Qrel("q1", "p_a", 1.0),
            Qrel("q1", "p_b", 0.0),
            Qrel("q1", "p_c", 0.5),
            Qrel("q1", "p_d", 1.0),  # relevant but not returned
        ]
    )
    mv = Evaluator(qrels).score_run([_rr("q1", ["p_a", "p_b", "p_c"])])["q1"]

    assert math.isclose(mv.avg_relevance, 0.15, abs_tol=1e-12)
    assert math.isclose(mv.ndcg_at_10, 0.6567370987244682, abs_tol=1e-12)
    assert mv.precision_at_10 == pytest.approx(0.2, abs=1e-12)
    assert mv.recall_at_10 == pytest.approx(2.0 / 3.0, abs=1e-12)


# --- short list: zero-padding + fixed-10 denominators --------------------------


def test_short_list_zero_padded_fixed_ten_denominators():
    # Two returned docs, both Exact (gain 1.0). Judged: only these two are relevant, R=2.
    #   avg_relevance = (1.0 + 1.0)/10 = 0.2   (denominator stays 10, not 2)
    #   precision@10  = 2/10 = 0.2             (denominator fixed at 10)
    #   recall@10     = 2/2 = 1.0
    qrels = QrelIndex([Qrel("q1", "p1", 1.0), Qrel("q1", "p2", 1.0)])
    mv = Evaluator(qrels).score_run([_rr("q1", ["p1", "p2"])])["q1"]
    assert mv.avg_relevance == pytest.approx(0.2, abs=1e-12)
    assert mv.precision_at_10 == pytest.approx(0.2, abs=1e-12)
    assert mv.recall_at_10 == pytest.approx(1.0, abs=1e-12)


# --- IDCG truncation: > 10 relevant docs must NOT deflate a strong ranking -----


def test_idcg_truncated_to_top_ten_not_deflated():
    # 15 judged docs, all Exact (gain 1.0) -> R = 15, ideal has 15 relevant.
    # A ranking returning 10 of them in the top 10 is a "perfect top-10": with IDCG TRUNCATED
    # to the top-10 ideal, DCG@10 == IDCG@10, so nDCG@10 == 1.0 (NOT deflated by the 5 extras).
    # If IDCG summed over all 15 gains it would exceed DCG@10 and nDCG would fall below 1.0.
    judged = [Qrel("q1", f"p{i}", 1.0) for i in range(15)]
    qrels = QrelIndex(judged)
    returned = [f"p{i}" for i in range(10)]  # top-10 are all relevant
    mv = Evaluator(qrels).score_run([_rr("q1", returned)])["q1"]

    # IDCG@10 = Σ_{i=1..10} 1/log2(i+1); a full-15-gain IDCG would be strictly larger.
    idcg_top10 = sum(1.0 / math.log2(i + 1) for i in range(1, 11))
    idcg_all15 = sum(1.0 / math.log2(i + 1) for i in range(1, 16))
    assert idcg_all15 > idcg_top10  # sanity: the deflating (wrong) denominator is bigger

    assert math.isclose(mv.ndcg_at_10, 1.0, abs_tol=1e-12)
    assert mv.precision_at_10 == pytest.approx(1.0, abs=1e-12)  # 10 relevant / 10
    assert mv.recall_at_10 == pytest.approx(10.0 / 15.0, abs=1e-12)  # R = 15


# --- IDCG@10 == 0 -> ndcg@10 == 0.0 (no division error) ------------------------


def test_idcg_zero_yields_ndcg_zero():
    # No judged docs for the query at all -> IDCG == 0 -> nDCG defined as 0.0.
    qrels = QrelIndex([Qrel("other", "p1", 1.0)])
    mv = Evaluator(qrels).score_run([_rr("q1", ["p1", "p2"])])["q1"]
    assert mv.ndcg_at_10 == 0.0
    assert mv.avg_relevance == 0.0  # all unjudged -> gain 0


def test_idcg_zero_all_irrelevant_yields_ndcg_zero():
    # Judged docs exist but all gain 0.0 -> ideal DCG == 0 -> nDCG == 0.0, no ZeroDivisionError.
    qrels = QrelIndex([Qrel("q1", "p1", 0.0), Qrel("q1", "p2", 0.0)])
    mv = Evaluator(qrels).score_run([_rr("q1", ["p1", "p2"])])["q1"]
    assert mv.ndcg_at_10 == 0.0


# --- R == 0 -> recall is NaN; precision still uses denom 10 ---------------------


def test_r_zero_recall_is_nan_precision_uses_ten():
    # Judged docs exist but none relevant (all gain 0.0) -> R == 0.
    qrels = QrelIndex([Qrel("q1", "p1", 0.0)])
    mv = Evaluator(qrels).score_run([_rr("q1", ["p1", "p2"])])["q1"]
    assert math.isnan(mv.recall_at_10)  # R == 0 -> NaN, NOT 0.0
    assert mv.precision_at_10 == pytest.approx(0.0, abs=1e-12)  # 0 hits / 10


def test_r_zero_recall_nan_even_with_relevant_hits_impossible_but_precision_ten():
    # No qrels at all for the query -> R == 0 -> recall NaN; precision denom stays 10.
    qrels = QrelIndex([Qrel("other", "p9", 1.0)])
    mv = Evaluator(qrels).score_run([_rr("q1", ["p1"])])["q1"]
    assert math.isnan(mv.recall_at_10)
    assert mv.precision_at_10 == pytest.approx(0.0, abs=1e-12)


# --- unjudged doc in the ranked list contributes gain 0.0 ----------------------


def test_unjudged_doc_contributes_zero_gain():
    # p1 judged 1.0 at rank 1; p_unjudged at rank 2 contributes gain 0 to DCG/avg_relevance.
    #   DCG@10 = (2^1-1)/log2(2) + 0 = 1.0 ; ideal = [1.0] -> IDCG@10 = 1.0 -> nDCG = 1.0
    #   avg_relevance = (1.0 + 0.0)/10 = 0.1 ; precision@10 = 1/10 ; R = 1 -> recall = 1.0
    qrels = QrelIndex([Qrel("q1", "p1", 1.0)])
    mv = Evaluator(qrels).score_run([_rr("q1", ["p1", "p_unjudged"])])["q1"]
    assert math.isclose(mv.ndcg_at_10, 1.0, abs_tol=1e-12)
    assert mv.avg_relevance == pytest.approx(0.1, abs=1e-12)
    assert mv.precision_at_10 == pytest.approx(0.1, abs=1e-12)
    assert mv.recall_at_10 == pytest.approx(1.0, abs=1e-12)


# --- score_run keying + multiple queries ---------------------------------------


def test_score_run_keyed_by_query_id():
    qrels = QrelIndex([Qrel("q1", "p1", 1.0), Qrel("q2", "p2", 1.0)])
    out = Evaluator(qrels).score_run([_rr("q1", ["p1"]), _rr("q2", ["p2"])])
    assert set(out.keys()) == {"q1", "q2"}
    assert all(isinstance(v, MetricVector) for v in out.values())


def test_custom_cutoff_denominator():
    # cutoff=2: avg_relevance and precision use denominator 2, top-2 only.
    qrels = QrelIndex([Qrel("q1", "p1", 1.0), Qrel("q1", "p2", 1.0), Qrel("q1", "p3", 1.0)])
    mv = Evaluator(qrels, cutoff=2).score_run([_rr("q1", ["p1", "p2", "p3"])])["q1"]
    assert mv.avg_relevance == pytest.approx(2.0 / 2.0, abs=1e-12)  # (1+1)/2
    assert mv.precision_at_10 == pytest.approx(2.0 / 2.0, abs=1e-12)  # 2 hits / 2
    assert mv.recall_at_10 == pytest.approx(2.0 / 3.0, abs=1e-12)  # R = 3
