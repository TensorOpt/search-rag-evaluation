"""Phase 2 unit tests for benchmark.metrics (docs/experiment.md §7).

Condensed-list evaluation: a MISSING judgement (no qrel entry) is SKIPPED, NOT scored as 0.0; a
JUDGED-irrelevant doc (gain 0.0, present in qrels) is KEPT and contributes 0 to DCG. Every expected
value is hand-computed; the arithmetic is written out in the test body so a reviewer can recompute
independently. Recall 2^0.5 - 1 == 0.41421356 (≈).
"""

from __future__ import annotations

import math

import pytest

from benchmark.common.models import Qrel, RankedResult, ScoredDoc
from benchmark.evaluation.metrics import Evaluator, Metrics, QrelIndex


def _rr(query_id: str, doc_ids: list[str]) -> RankedResult:
    """A RankedResult with descending placeholder scores (only the ORDER matters for metrics)."""
    n = len(doc_ids)
    return RankedResult(query_id=query_id, docs=[ScoredDoc(d, float(n - i)) for i, d in enumerate(doc_ids)])


# --- QrelIndex -----------------------------------------------------------------


def test_qrelindex_gain_missing_is_nan():
    idx = QrelIndex([Qrel("q1", "p1", 1.0), Qrel("q1", "p0", 0.0)])
    assert idx.gain("q1", "p1") == 1.0
    assert idx.gain("q1", "p0") == 0.0  # judged-irrelevant: a real judgement, NOT missing
    assert math.isnan(idx.gain("q1", "p_missing"))  # no qrel entry -> MISSING (NaN)
    assert math.isnan(idx.gain("q_missing", "p1"))  # unjudged query -> MISSING (NaN)


def test_qrelindex_relevant_count_thresholds_at_half():
    idx = QrelIndex(
        [
            Qrel("q1", "p1", 1.0),  # relevant (Exact)
            Qrel("q1", "p2", 0.5),  # relevant (Partial)
            Qrel("q1", "p3", 0.0),  # judged, not relevant (Irrelevant)
        ]
    )
    assert idx.relevant_count("q1") == 2  # only gain >= 0.5 counts
    assert idx.relevant_count("q_missing") == 0


def test_qrelindex_sorted_judged_gains_descending():
    idx = QrelIndex([Qrel("q1", "p1", 0.5), Qrel("q1", "p2", 1.0), Qrel("q1", "p3", 0.0)])
    assert idx.sorted_judged_gains("q1") == [1.0, 0.5, 0.0]  # judged 0.0 is included
    assert idx.sorted_judged_gains("q_missing") == []


# --- Metrics.as_dict canonical keys + int count fields -------------------------


def test_metrics_as_dict_exact_canonical_keys():
    m = Metrics(
        avg_relevance=0.1, ndcg_at_10=0.2, recall_at_10=0.3, recall_at_50=0.35,
        recall_at_100=0.45, precision_at_10=0.4,
        n_results=7, n_scored=5, n_missing=2,
    )
    d = m.as_dict()
    assert set(d.keys()) == {
        "avg_relevance", "ndcg@10", "recall@10", "recall@50", "recall@100", "precision@10"
    }
    assert d["avg_relevance"] == 0.1
    assert d["ndcg@10"] == 0.2
    assert d["recall@10"] == 0.3
    assert d["recall@50"] == 0.35
    assert d["recall@100"] == 0.45
    assert d["precision@10"] == 0.4
    # counts are int fields, NOT in as_dict()
    assert m.n_scored == 5
    assert m.n_missing == 2
    assert isinstance(m.n_scored, int)
    assert isinstance(m.n_missing, int)
    assert "n_scored" not in d
    assert "n_missing" not in d


# --- perfect ranking -> ndcg@10 == 1.0 -----------------------------------------


def test_perfect_ranking_ndcg_is_one():
    # Judged: p1=1.0, p2=0.5. Ideal order [1.0, 0.5]; the ranking returns exactly that order.
    qrels = QrelIndex([Qrel("q1", "p1", 1.0), Qrel("q1", "p2", 0.5)])
    m = Evaluator(qrels).score_run([_rr("q1", ["p1", "p2"])])["q1"]
    assert math.isclose(m.ndcg_at_10, 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert m.n_scored == 2
    assert m.n_missing == 0


# --- MISSING doc is SKIPPED (condensed), NOT scored 0.0; judged-irrelevant KEPT --


def test_missing_doc_skipped_judged_irrelevant_kept():
    # Ranked list (rank 1..4): p_a=1.0 (judged), p_miss=MISSING (no qrel), p_b=0.0 (judged
    # irrelevant), p_c=0.5 (judged).
    #   MISSING p_miss is SKIPPED and NOT scored; p_b (judged 0.0) is KEPT.
    #   CONDENSED gains in rank order: [1.0, 0.0, 0.5]  -> positions 1,2,3
    #   n_scored = 3 (three judged docs), n_missing = 1 (one skipped)
    #   DCG = (2^1-1)/log2(2) + (2^0-1)/log2(3) + (2^0.5-1)/log2(4)
    #       = 1/1 + 0 + 0.41421356/2 = 1.2071067811865475
    #   Judged gains for query: {1.0, 0.0, 0.5, 1.0(p_d not returned)}; ideal desc [1.0,1.0,0.5,0.0]
    #   IDCG = 1/log2(2) + 1/log2(3) + (2^0.5-1)/log2(4) + 0/log2(5)
    #        = 1 + 0.6309297535714575 + 0.20710678118654752 = 1.8380365347580052
    #   nDCG = 1.2071067811865475 / 1.8380365347580052 = 0.6567370987244682
    #   avg_relevance = (1.0 + 0.0 + 0.5)/3 = 0.5
    #   precision@10 = 2 relevant (p_a=1.0, p_c=0.5) / n_scored(3) = 2/3   (denom = n_scored)
    #   R = 3 relevant judged (two 1.0 + one 0.5); recall = 2/3
    qrels = QrelIndex(
        [
            Qrel("q1", "p_a", 1.0),
            Qrel("q1", "p_b", 0.0),  # judged-irrelevant, KEPT
            Qrel("q1", "p_c", 0.5),
            Qrel("q1", "p_d", 1.0),  # relevant but not returned
        ]
    )
    m = Evaluator(qrels).score_run([_rr("q1", ["p_a", "p_miss", "p_b", "p_c"])])["q1"]

    assert m.n_scored == 3
    assert m.n_missing == 1
    assert math.isclose(m.avg_relevance, 0.5, abs_tol=1e-12)
    assert math.isclose(m.ndcg_at_10, 0.6567370987244682, abs_tol=1e-12)
    assert m.precision_at_10 == pytest.approx(2.0 / 3.0, abs=1e-12)  # denom = n_scored, NOT 10
    assert m.recall_at_10 == pytest.approx(2.0 / 3.0, abs=1e-12)


# --- condensed list reaches PAST original rank 10 to fill 10 judged docs --------


def test_condensed_reaches_past_rank_ten():
    # 12 returned docs. The FIRST TWO are MISSING (no qrel), then p0..p9 are judged Exact (1.0).
    # The condensed top-10 (avg/ndcg/precision) must collect p0..p9 — reaching original ranks
    # 3..12 (past rank 10).
    #   n_scored = 10, n_missing = 2 (the two missing docs in the scanned prefix, ranks 1-2).
    #   All 10 condensed gains are 1.0 -> a perfect top-10. R = 10 judged relevant.
    #   IDCG@10 = Σ_{i=1..10} 1/log2(i+1); DCG over condensed == IDCG -> nDCG == 1.0.
    # STANDARD recall@10 (Fix 4) is over the ACTUAL top-10 positions, NOT the condensed list:
    #   result.docs[:10] = [m1, m2, p0..p7]; relevant hits among them = p0..p7 = 8 -> recall@10 =
    #   8/10 = 0.8 (the two MISSING docs occupy top-10 slots and are not hits). recall@50/@100 see
    #   all 12 docs -> all 10 relevant -> 1.0.
    judged = [Qrel("q1", f"p{i}", 1.0) for i in range(10)]
    qrels = QrelIndex(judged)
    returned = ["m1", "m2"] + [f"p{i}" for i in range(10)]  # 2 missing, then 10 judged
    m = Evaluator(qrels).score_run([_rr("q1", returned)])["q1"]

    assert m.n_scored == 10
    assert m.n_missing == 2  # only the two missing docs seen before the 10th judged doc
    assert m.avg_relevance == pytest.approx(1.0, abs=1e-12)
    assert math.isclose(m.ndcg_at_10, 1.0, abs_tol=1e-12)
    assert m.precision_at_10 == pytest.approx(1.0, abs=1e-12)  # 10/10, denom = n_scored
    assert m.recall_at_10 == pytest.approx(8.0 / 10.0, abs=1e-12)  # standard: 8 hits in top-10 / R=10
    assert m.recall_at_50 == pytest.approx(1.0, abs=1e-12)  # all 10 relevant in top-50 / R=10
    assert m.recall_at_100 == pytest.approx(1.0, abs=1e-12)


def test_missing_docs_after_ten_judged_not_counted():
    # 10 judged docs (ranks 1..10) then 3 MISSING docs (ranks 11..13). The condensed top-10 is
    # filled by rank 10, so the scan STOPS there and the 3 trailing missing docs are NOT counted.
    judged = [Qrel("q1", f"p{i}", 1.0) for i in range(10)]
    qrels = QrelIndex(judged)
    returned = [f"p{i}" for i in range(10)] + ["m1", "m2", "m3"]
    m = Evaluator(qrels).score_run([_rr("q1", returned)])["q1"]
    assert m.n_scored == 10
    assert m.n_missing == 0  # scan stopped at the 10th judged doc, before any missing doc


# --- graded mixed {0.0, 0.5, 1.0} + missing: hand-computed DCG/IDCG/ndcg --------


def test_graded_mixed_with_missing_hand_computed():
    # Ranked (rank 1..5): p_a=1.0, p_miss1=MISSING, p_b=0.5, p_c=0.0, p_miss2=MISSING.
    #   CONDENSED gains (rank order): [1.0, 0.5, 0.0]. n_scored=3, n_missing=2.
    #   DCG = (2^1-1)/log2(2) + (2^0.5-1)/log2(3) + (2^0-1)/log2(4)
    #       = 1.0 + 0.41421356237309515/1.5849625007211562 + 0
    #       = 1.0 + 0.26133966083401244 = 1.2613396608340124
    #   Judged gains for query: p_a=1.0, p_b=0.5, p_c=0.0, p_d=1.0(not returned).
    #     ideal desc = [1.0, 1.0, 0.5, 0.0]; IDCG@10 =
    #       1/log2(2) + 1/log2(3) + (2^0.5-1)/log2(4) + 0/log2(5)
    #       = 1 + 0.6309297535714575 + 0.20710678118654752 = 1.8380365347580052
    #   nDCG = 1.2613396608340124 / 1.8380365347580052 = 0.686242975578328
    #   avg_relevance = (1.0 + 0.5 + 0.0)/3 = 0.5
    #   precision@10 = 2 relevant (1.0, 0.5) / 3 = 2/3
    #   R = 3 (p_a, p_b, p_d); recall = 2/3
    qrels = QrelIndex(
        [
            Qrel("q1", "p_a", 1.0),
            Qrel("q1", "p_b", 0.5),
            Qrel("q1", "p_c", 0.0),
            Qrel("q1", "p_d", 1.0),
        ]
    )
    m = Evaluator(qrels).score_run(
        [_rr("q1", ["p_a", "p_miss1", "p_b", "p_c", "p_miss2"])]
    )["q1"]

    assert m.n_scored == 3
    assert m.n_missing == 2
    dcg = 1.0 + (2.0**0.5 - 1.0) / math.log2(3)
    idcg = 1.0 + 1.0 / math.log2(3) + (2.0**0.5 - 1.0) / math.log2(4)
    assert math.isclose(dcg, 1.2613396608340124, abs_tol=1e-12)
    assert math.isclose(idcg, 1.8380365347580052, abs_tol=1e-12)
    assert math.isclose(m.avg_relevance, 0.5, abs_tol=1e-12)
    assert math.isclose(m.ndcg_at_10, dcg / idcg, abs_tol=1e-12)
    assert math.isclose(m.ndcg_at_10, 0.686242975578328, abs_tol=1e-12)
    assert m.precision_at_10 == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert m.recall_at_10 == pytest.approx(2.0 / 3.0, abs=1e-12)


# --- precision & avg_relevance denominators are n_scored (NOT 10), n_scored < 10 -


def test_denominators_are_n_scored_not_ten():
    # Two judged docs returned (both Exact 1.0); one extra returned doc is MISSING and skipped.
    #   condensed = [1.0, 1.0], n_scored = 2 (< 10), n_missing = 1.
    #   avg_relevance = (1.0 + 1.0)/2 = 1.0   (denominator is 2, NOT 10)
    #   precision@10  = 2/2 = 1.0             (denominator is n_scored=2, NOT 10)
    #   R = 2; recall@10 = 2/2 = 1.0
    qrels = QrelIndex([Qrel("q1", "p1", 1.0), Qrel("q1", "p2", 1.0)])
    m = Evaluator(qrels).score_run([_rr("q1", ["p1", "p_missing", "p2"])])["q1"]
    assert m.n_scored == 2
    assert m.n_missing == 1
    assert m.avg_relevance == pytest.approx(1.0, abs=1e-12)  # NOT 0.2 (would be /10)
    assert m.precision_at_10 == pytest.approx(1.0, abs=1e-12)  # NOT 0.2 (would be /10)
    assert m.recall_at_10 == pytest.approx(1.0, abs=1e-12)


# --- IDCG truncation: > 10 relevant docs must NOT deflate a strong ranking -----


def test_idcg_truncated_to_top_ten_not_deflated():
    # 15 judged docs, all Exact (gain 1.0) -> R = 15, ideal has 15 relevant.
    # A ranking returning 10 of them in the top 10 is a "perfect top-10": with IDCG TRUNCATED
    # to the top-10 ideal (over ALL judged gains), DCG == IDCG, so nDCG == 1.0.
    judged = [Qrel("q1", f"p{i}", 1.0) for i in range(15)]
    qrels = QrelIndex(judged)
    returned = [f"p{i}" for i in range(10)]  # top-10 are all relevant judged
    m = Evaluator(qrels).score_run([_rr("q1", returned)])["q1"]

    idcg_top10 = sum(1.0 / math.log2(i + 1) for i in range(1, 11))
    idcg_all15 = sum(1.0 / math.log2(i + 1) for i in range(1, 16))
    assert idcg_all15 > idcg_top10  # sanity: the deflating (wrong) denominator is bigger

    assert m.n_scored == 10
    assert m.n_missing == 0
    assert math.isclose(m.ndcg_at_10, 1.0, abs_tol=1e-12)
    assert m.precision_at_10 == pytest.approx(1.0, abs=1e-12)  # 10 relevant / n_scored 10
    assert m.recall_at_10 == pytest.approx(10.0 / 15.0, abs=1e-12)  # R = 15


# --- IDCG@10 == 0 (with judged docs present) -> ndcg@10 == 0.0 (no div error) ---


def test_idcg_zero_all_irrelevant_yields_ndcg_zero():
    # Judged docs exist but all gain 0.0 -> ideal DCG == 0 -> nDCG == 0.0, no ZeroDivisionError.
    #   condensed = [0.0, 0.0], n_scored = 2. avg_relevance = 0.0. R = 0 -> recall NaN.
    qrels = QrelIndex([Qrel("q1", "p1", 0.0), Qrel("q1", "p2", 0.0)])
    m = Evaluator(qrels).score_run([_rr("q1", ["p1", "p2"])])["q1"]
    assert m.n_scored == 2
    assert m.ndcg_at_10 == pytest.approx(0.0, abs=1e-12)
    assert m.avg_relevance == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(m.recall_at_10)  # R == 0


# --- n_scored == 0: all returned docs unjudged (MISSING) -----------------------


def test_all_missing_n_scored_zero_metrics_nan_recall_zero_when_r_positive():
    # Query q1 has judged docs (R>0) but NONE of them are returned; every returned doc is MISSING.
    #   n_scored = 0 -> avg_relevance, ndcg@10, precision@10 are NaN.
    #   R = 1 (> 0) and 0 condensed hits -> recall@10 = 0/1 = 0.0 (NOT NaN).
    qrels = QrelIndex([Qrel("q1", "p_judged", 1.0)])
    m = Evaluator(qrels).score_run([_rr("q1", ["x1", "x2", "x3"])])["q1"]
    assert m.n_scored == 0
    assert m.n_missing == 3
    assert math.isnan(m.avg_relevance)
    assert math.isnan(m.ndcg_at_10)
    assert math.isnan(m.precision_at_10)
    assert m.recall_at_10 == pytest.approx(0.0, abs=1e-12)  # R>0 -> 0.0, not NaN


def test_all_missing_and_r_zero_recall_is_nan():
    # No qrels for the query at all -> every returned doc MISSING, n_scored == 0 AND R == 0.
    #   avg_relevance/ndcg/precision NaN (n_scored==0); recall NaN (R==0).
    qrels = QrelIndex([Qrel("other", "p9", 1.0)])
    m = Evaluator(qrels).score_run([_rr("q1", ["p1", "p2"])])["q1"]
    assert m.n_scored == 0
    assert m.n_missing == 2
    assert math.isnan(m.avg_relevance)
    assert math.isnan(m.ndcg_at_10)
    assert math.isnan(m.precision_at_10)
    assert math.isnan(m.recall_at_10)  # R == 0


# --- R == 0 -> recall is NaN even when judged docs are scored -------------------


def test_r_zero_recall_is_nan_with_scored_judged_docs():
    # Judged docs exist and are returned but none relevant (all gain 0.0) -> R == 0.
    #   condensed = [0.0], n_scored = 1. precision@10 = 0 hits / 1 = 0.0. recall NaN.
    qrels = QrelIndex([Qrel("q1", "p1", 0.0)])
    m = Evaluator(qrels).score_run([_rr("q1", ["p1", "p_missing"])])["q1"]
    assert m.n_scored == 1
    assert m.n_missing == 1
    assert math.isnan(m.recall_at_10)  # R == 0 -> NaN
    assert m.precision_at_10 == pytest.approx(0.0, abs=1e-12)  # 0 hits / n_scored 1
    assert m.avg_relevance == pytest.approx(0.0, abs=1e-12)


# --- score_run keying + multiple queries ---------------------------------------


def test_score_run_keyed_by_query_id():
    qrels = QrelIndex([Qrel("q1", "p1", 1.0), Qrel("q2", "p2", 1.0)])
    out = Evaluator(qrels).score_run([_rr("q1", ["p1"]), _rr("q2", ["p2"])])
    assert set(out.keys()) == {"q1", "q2"}
    assert all(isinstance(v, Metrics) for v in out.values())


def test_custom_cutoff_condensed_denominator():
    # cutoff=2 governs ONLY the condensed top-2 (avg/ndcg/precision). Returned p1(1.0), p_miss(MISSING),
    # p2(1.0), p3(1.0).
    #   scan: p1 judged (1), p_miss skipped (n_missing=1), p2 judged (2) -> stop at k=2.
    #   condensed = [1.0, 1.0], n_scored = 2. avg = 2/2 = 1.0. precision = 2/2 = 1.0.
    # STANDARD recall@10 (Fix 4) is INDEPENDENT of the condensed cutoff: it scans result.docs[:10] =
    #   all 4 returned; relevant hits = p1, p2, p3 = 3; R = 3 -> recall@10 = 3/3 = 1.0.
    qrels = QrelIndex(
        [Qrel("q1", "p1", 1.0), Qrel("q1", "p2", 1.0), Qrel("q1", "p3", 1.0)]
    )
    m = Evaluator(qrels, cutoff=2).score_run([_rr("q1", ["p1", "p_miss", "p2", "p3"])])["q1"]
    assert m.n_scored == 2
    assert m.n_missing == 1
    assert m.avg_relevance == pytest.approx(1.0, abs=1e-12)  # (1+1)/2
    assert m.precision_at_10 == pytest.approx(1.0, abs=1e-12)  # 2 hits / 2
    assert m.recall_at_10 == pytest.approx(1.0, abs=1e-12)  # standard: 3 hits in top-10 / R=3
