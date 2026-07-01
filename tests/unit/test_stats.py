"""Unit tests for the Comparator (docs/experiment.md §8). Phase 3.

Covers: seeded bootstrap-CI determinism and B honoring; the two degenerate short-circuits
(empty paired set / all-zero deltas) with a hard assertion that scipy/bootstrap/RNG are NEVER
called and that degenerate rows are excluded from the FDR family size m; the generalized
per-metric NaN exclusion (avg/ndcg/precision via n_scored==0, recall via R==0); family-wide FDR
(Benjamini-Hochberg step-up rejection set + adjusted q-values, and BY as a more-conservative
option); raw-vs-FDR significance (significant_raw is the uncorrected per-test decision, may differ
from the FDR significant flag); CI-vs-significant disagreement (both reported, no exception); the
Wilcoxon zero_method/correction/two-sided path and the seeded reproducible permutation path; and the
exact §8.1-table ComparisonResult fields/notes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmark import stats as stats_mod
from benchmark.stats import (
    CANONICAL_METRICS,
    Comparator,
    ComparisonResult,
    StatsCfg,
    _fdr_adjust,
)

# --------------------------------------------------------------------------------------------------
# Helpers: build baseline / variants maps in the plain `Metrics.as_dict()` shape.
# --------------------------------------------------------------------------------------------------


def _all_metrics(value: float) -> dict[str, float]:
    """A per-query metric map with the same value for all four canonical metrics."""
    return {m: value for m in CANONICAL_METRICS}


def _baseline(values: dict[str, float]) -> dict[str, dict[str, float]]:
    """query_id -> {metric: value}, same value across metrics, for the given per-query values."""
    return {qid: _all_metrics(v) for qid, v in values.items()}


def _variant(values: dict[str, float]) -> dict[str, dict[str, float]]:
    return {qid: _all_metrics(v) for qid, v in values.items()}


def _row(rows: list[ComparisonResult], variant: str, metric: str) -> ComparisonResult:
    (r,) = [x for x in rows if x.variant == variant and x.metric == metric]
    return r


# --------------------------------------------------------------------------------------------------
# Seeded determinism + B honored (§8.2).
# --------------------------------------------------------------------------------------------------


def test_seeded_ci_is_deterministic_across_repeated_compare() -> None:
    base = _baseline({f"q{i}": float(i % 3) for i in range(30)})
    var = _variant({f"q{i}": float(i % 3) + 0.3 for i in range(30)})
    cfg = StatsCfg(bootstrap_B=2000, seed=1234)
    rows1 = Comparator(cfg).compare(base, {"v": var})
    rows2 = Comparator(cfg).compare(base, {"v": var})
    for m in CANONICAL_METRICS:
        r1, r2 = _row(rows1, "v", m), _row(rows2, "v", m)
        assert r1.delta_ci_lo == r2.delta_ci_lo
        assert r1.delta_ci_high == r2.delta_ci_high


def test_different_seed_generally_changes_ci() -> None:
    base = _baseline({f"q{i}": float(i % 4) for i in range(40)})
    var = _variant({f"q{i}": float(i % 4) + (0.5 if i % 2 else -0.2) for i in range(40)})
    r_a = _row(Comparator(StatsCfg(bootstrap_B=2000, seed=1)).compare(base, {"v": var}), "v", "ndcg@10")
    r_b = _row(Comparator(StatsCfg(bootstrap_B=2000, seed=999)).compare(base, {"v": var}), "v", "ndcg@10")
    # Same point estimate, but the bootstrap interval differs with a different seed.
    assert r_a.delta == r_b.delta
    assert (r_a.delta_ci_lo, r_a.delta_ci_high) != (r_b.delta_ci_lo, r_b.delta_ci_high)


def test_bootstrap_B_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _baseline({f"q{i}": float(i % 3) for i in range(20)})
    var = _variant({f"q{i}": float(i % 3) + 0.4 for i in range(20)})
    seen_sizes: list[tuple[int, ...]] = []

    real_default_rng = np.random.default_rng

    class _SpyRng:
        def __init__(self, inner: np.random.Generator) -> None:
            self._inner = inner

        def integers(self, low: int, high: int, size: tuple[int, ...]) -> np.ndarray:
            seen_sizes.append(size)
            return self._inner.integers(low, high, size=size)

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    def _spy_default_rng(seed: int) -> _SpyRng:
        return _SpyRng(real_default_rng(seed))

    monkeypatch.setattr(np.random, "default_rng", _spy_default_rng)
    Comparator(StatsCfg(bootstrap_B=1234, seed=7)).compare(base, {"v": var})
    # Every bootstrap draws a (B, n) index matrix: B == bootstrap_B.
    assert seen_sizes, "bootstrap RNG.integers was never called"
    assert all(sz[0] == 1234 for sz in seen_sizes)


# --------------------------------------------------------------------------------------------------
# Degenerate short-circuits (§8.1 table): no scipy/bootstrap/RNG; excluded from FDR family.
# --------------------------------------------------------------------------------------------------


def _forbid_scipy_and_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom_wilcoxon(*_a: object, **_k: object) -> object:
        raise AssertionError("scipy.stats.wilcoxon must NOT be called for degenerate rows")

    def _boom_rng(*_a: object, **_k: object) -> object:
        raise AssertionError("bootstrap RNG (default_rng) must NOT be called for degenerate rows")

    monkeypatch.setattr(stats_mod.scipy_stats, "wilcoxon", _boom_wilcoxon)
    monkeypatch.setattr(np.random, "default_rng", _boom_rng)


def test_empty_paired_set_row_and_no_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_scipy_and_rng(monkeypatch)
    # recall@10 NaN on every query for both runs (R==0) -> empty paired set for recall.
    # Make ALL metrics NaN so every row is degenerate-empty and scipy/rng are never touched.
    base = {"q1": _all_metrics(math.nan), "q2": _all_metrics(math.nan)}
    var = {"q1": _all_metrics(math.nan), "q2": _all_metrics(math.nan)}
    rows = Comparator(StatsCfg()).compare(base, {"v": var})
    for m in CANONICAL_METRICS:
        r = _row(rows, "v", m)
        assert r.delta is None
        assert r.delta_ci_lo is None
        assert r.delta_ci_high is None
        assert r.p_value == 1.0
        assert r.significant_raw is False
        assert r.p_value_adjusted == 1.0
        assert r.significant is False
        assert r.note == "empty_paired_set"


def test_all_zero_delta_row_and_no_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_scipy_and_rng(monkeypatch)
    base = _baseline({f"q{i}": float(i % 2) for i in range(10)})
    var = _variant({f"q{i}": float(i % 2) for i in range(10)})  # identical -> all-zero deltas
    rows = Comparator(StatsCfg()).compare(base, {"v": var})
    for m in CANONICAL_METRICS:
        r = _row(rows, "v", m)
        assert r.delta == 0.0
        assert r.delta_ci_lo == 0.0
        assert r.delta_ci_high == 0.0
        assert r.p_value == 1.0
        assert r.significant_raw is False
        assert r.p_value_adjusted == 1.0
        assert r.significant is False
        assert r.note == "all_zero_delta"


def test_degenerate_rows_excluded_from_fdr_family() -> None:
    # One real test on avg_relevance (tiny p so it would reject at any sane m), and a degenerate
    # all-zero row for the other metrics. Degenerate rows must NOT enter the FDR family: with m == 1
    # (the family only contains the real test), the BH q-value equals the raw p and the test rejects.
    n = 30
    base = _baseline({f"q{i}": 0.0 for i in range(n)})
    # avg_relevance differs strongly (real test); other metrics identical (all-zero degenerate).
    var: dict[str, dict[str, float]] = {}
    for i in range(n):
        d = _all_metrics(0.0)
        d["avg_relevance"] = 1.0  # constant +1 delta -> very significant
        var[f"q{i}"] = d
    rows = Comparator(StatsCfg(alpha=0.05, bootstrap_B=500)).compare(base, {"v": var})
    real = _row(rows, "v", "avg_relevance")
    assert real.note is None
    assert real.significant is True  # only member of the family; q == raw p tiny <= 0.05
    assert real.p_value_adjusted == pytest.approx(real.p_value)  # m == 1 -> q == p
    for m in ("ndcg@10", "recall@10", "precision@10"):
        r = _row(rows, "v", m)
        assert r.note == "all_zero_delta"
        assert r.significant is False
        assert r.significant_raw is False


# --------------------------------------------------------------------------------------------------
# Generalized per-metric NaN exclusion (§8.1) — avg/ndcg/precision (n_scored==0) AND recall (R==0).
# --------------------------------------------------------------------------------------------------


def test_per_metric_nan_exclusion_pairs_only_finite_both() -> None:
    # q0..q4 finite everywhere; q5 has NaN ndcg in the variant (n_scored==0) -> ndcg pairs 5, not 6.
    base = _baseline({f"q{i}": 0.1 for i in range(6)})
    var: dict[str, dict[str, float]] = {}
    for i in range(6):
        var[f"q{i}"] = _all_metrics(0.5)
    var["q5"]["ndcg@10"] = math.nan  # excluded for ndcg only
    var["q3"]["recall@10"] = math.nan  # excluded for recall only

    cfg = StatsCfg(bootstrap_B=200, seed=3)
    # Capture the paired delta vector size per metric by spying on _paired_deltas.
    sizes: dict[str, int] = {}
    orig = stats_mod._paired_deltas

    def _spy(b: object, v: object, metric: str) -> np.ndarray:
        arr = orig(b, v, metric)  # type: ignore[arg-type]
        sizes[metric] = arr.size
        return arr

    stats_mod._paired_deltas = _spy  # type: ignore[assignment]
    try:
        Comparator(cfg).compare(base, {"v": var})
    finally:
        stats_mod._paired_deltas = orig  # type: ignore[assignment]

    assert sizes["avg_relevance"] == 6
    assert sizes["precision@10"] == 6
    assert sizes["ndcg@10"] == 5  # q5 excluded
    assert sizes["recall@10"] == 5  # q3 excluded


def test_all_excluded_becomes_empty_paired_set() -> None:
    # recall NaN (R==0) for the ONLY query in the baseline -> empty paired set for recall.
    base = {"q0": {**_all_metrics(0.4), "recall@10": math.nan}}
    var = {"q0": {**_all_metrics(0.6), "recall@10": 0.6}}
    rows = Comparator(StatsCfg(bootstrap_B=100)).compare(base, {"v": var})
    assert _row(rows, "v", "recall@10").note == "empty_paired_set"
    # A non-recall metric with a single finite pair is a real 1-sample test (not degenerate-empty).
    assert _row(rows, "v", "avg_relevance").note is None


# --------------------------------------------------------------------------------------------------
# FDR family-wide (§8.3) — direct unit test of the _fdr_adjust helper (BH step-up + q-values, BY).
# --------------------------------------------------------------------------------------------------


def _bh_reject_set(ps: list[float], alpha: float) -> set[int]:
    """Classic BH step-up rejection set: largest k with p_(k) <= (k/m)*alpha; reject all <= that."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    k_star = 0
    for rank in range(1, m + 1):
        if ps[order[rank - 1]] <= (rank / m) * alpha:
            k_star = rank
    return {order[r - 1] for r in range(1, k_star + 1)}


def test_bh_step_up_rejection_set_and_qvalues() -> None:
    # Family of 5 raw p-values. alpha = 0.05, m = 5.
    # Sorted ascending: 0.001, 0.008, 0.02, 0.04, 0.7
    # BH thresholds (k/m)*alpha: 0.01, 0.02, 0.03, 0.04, 0.05
    #   0.001 <= 0.01 ; 0.008 <= 0.02 ; 0.02 <= 0.03 ; 0.04 <= 0.04 ; 0.7 <= 0.05? no
    #   largest k with p_(k) <= (k/m)*alpha is k=4 -> reject the four smallest.
    ps = [0.02, 0.7, 0.001, 0.04, 0.008]
    alpha = 0.05
    q = _fdr_adjust(ps, "bh")
    reject_expected = _bh_reject_set(ps, alpha)
    assert reject_expected == {0, 2, 3, 4}  # everything except the p=0.7 test (index 1)

    reject_from_q = {i for i, qi in enumerate(q) if qi <= alpha}
    assert reject_from_q == reject_expected  # significant == (q <= alpha) reproduces the step-up set

    # q-values are monotone non-decreasing in rank and clamped <= 1.
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    q_sorted = [q[i] for i in order]
    assert q_sorted == sorted(q_sorted)  # monotone in rank
    assert all(0.0 <= qi <= 1.0 for qi in q)


def test_bh_qvalues_match_hand_computed() -> None:
    # BH q_(k) = min over j>=k of p_(j)*m/j, monotone non-decreasing by rank, clamped <=1.
    # ps sorted: 0.001(r1),0.008(r2),0.02(r3),0.04(r4),0.5(r5),0.7(r6), m=6 -> raw q by rank:
    #   0.006, 0.024, 0.04, 0.06, 0.6, 0.7  (already monotone), mapped back to input order below.
    ps = [0.001, 0.008, 0.02, 0.04, 0.7, 0.5]
    expected = [0.006, 0.024, 0.04, 0.06, 0.7, 0.6]
    assert _fdr_adjust(ps, "bh") == pytest.approx(expected)


def test_by_is_more_conservative_than_bh() -> None:
    # On the same family, BY rejections are a subset-or-equal of BH rejections (BY costs a log-factor).
    ps = [0.001, 0.008, 0.02, 0.04, 0.7, 0.5]
    alpha = 0.05
    bh_q = _fdr_adjust(ps, "bh")
    by_q = _fdr_adjust(ps, "by")
    bh_reject = {i for i, q in enumerate(bh_q) if q <= alpha}
    by_reject = {i for i, q in enumerate(by_q) if q <= alpha}
    assert by_reject <= bh_reject
    assert by_reject != bh_reject  # strictly more conservative on this family
    # BY q-values are >= BH q-values everywhere (same ordering, larger scaling).
    for b, y in zip(bh_q, by_q):
        assert y >= b - 1e-12


def test_raw_significant_can_exceed_fdr_significant() -> None:
    # A family where a test is raw-significant (p <= alpha) but NOT FDR-significant (q > alpha):
    # a marginal p=0.04 alongside many large p's -> BH q inflates above alpha.
    ps = [0.04, 0.6, 0.7, 0.8, 0.9]
    alpha = 0.05
    q = _fdr_adjust(ps, "bh")
    assert ps[0] <= alpha  # raw-significant
    assert q[0] > alpha  # but NOT FDR-significant after correction


def test_bh_more_powerful_than_holm_but_still_corrects() -> None:
    # Two tests both raw-significant and both FDR-significant (BH keeps power a strict Holm/FWER
    # step-down would have thrown away), yet a third large-p test is correctly NOT rejected.
    ps = [0.001, 0.02, 0.9]
    alpha = 0.05
    q = _fdr_adjust(ps, "bh")
    assert all(p <= alpha for p in ps[:2])  # both raw-significant
    assert q[0] <= alpha and q[1] <= alpha  # both survive BH -> both FDR-significant
    assert q[2] > alpha  # the null test is still corrected out


# --------------------------------------------------------------------------------------------------
# Raw vs FDR significance inside compare() (§8.3).
# --------------------------------------------------------------------------------------------------


def test_raw_and_fdr_significance_in_compare() -> None:
    # Two variants: strong effect on one metric (raw AND FDR significant), weak on another
    # (neither). Both flags are exposed independently on each row.
    n = 40
    base = _baseline({f"q{i}": 0.0 for i in range(n)})
    strong: dict[str, dict[str, float]] = {}
    weak: dict[str, dict[str, float]] = {}
    for i in range(n):
        s = _all_metrics(0.0)
        s["avg_relevance"] = 1.0  # huge, constant +1 -> tiny p
        strong[f"q{i}"] = s
        w = _all_metrics(0.0)
        # tiny alternating signal, near-zero mean -> large p
        w["avg_relevance"] = 0.001 if i % 2 else -0.001
        weak[f"q{i}"] = w
    rows = Comparator(StatsCfg(alpha=0.05, bootstrap_B=300)).compare(
        base, {"strong": strong, "weak": weak}
    )
    r_strong = _row(rows, "strong", "avg_relevance")
    r_weak = _row(rows, "weak", "avg_relevance")
    assert r_strong.significant_raw is True
    assert r_strong.significant is True
    assert r_weak.significant_raw is False
    assert r_weak.significant is False
    # significant_raw is exactly the per-test p_value <= alpha, independent of the family.
    for r in rows:
        if r.note is None:
            assert r.significant_raw == (r.p_value <= 0.05)


# --------------------------------------------------------------------------------------------------
# CI vs significant may disagree (§8.3) — both reported, no reconciliation, no exception.
# --------------------------------------------------------------------------------------------------


def test_ci_excludes_zero_but_fdr_may_retain() -> None:
    # A consistent moderate effect on ndcg (CI excludes 0). Whatever the FDR decision, both the CI
    # and the significance flags coexist and no exception is raised.
    n = 25
    base = _baseline({f"q{i}": 0.0 for i in range(n)})
    var: dict[str, dict[str, float]] = {}
    for i in range(n):
        d = _all_metrics(0.0)
        d["ndcg@10"] = 0.2  # constant positive -> CI well above 0
        var[f"q{i}"] = d
    variants: dict[str, dict[str, dict[str, float]]] = {"target": var}
    rows = Comparator(StatsCfg(alpha=0.05, bootstrap_B=500, seed=5)).compare(base, variants)
    r = _row(rows, "target", "ndcg@10")
    # Constant +0.2 delta: bootstrap CI is a degenerate point at 0.2 (all resamples mean 0.2) -> excludes 0.
    assert r.delta is not None and r.delta_ci_lo is not None and r.delta_ci_high is not None
    assert r.delta_ci_lo > 0.0  # CI excludes 0
    # Both flags well-formed regardless of whether they agree with the CI.
    assert isinstance(r.significant, bool)
    assert isinstance(r.significant_raw, bool)


def test_ci_includes_zero_no_exception() -> None:
    # Wide, near-zero-mean noise: CI likely straddles 0 while we do not force any reconciliation.
    rng = np.random.default_rng(0)
    n = 60
    base = _baseline({f"q{i}": 0.0 for i in range(n)})
    var: dict[str, dict[str, float]] = {}
    draws = rng.normal(0.0, 1.0, size=n)
    for i in range(n):
        d = _all_metrics(0.0)
        d["ndcg@10"] = float(draws[i])
        var[f"q{i}"] = d
    rows = Comparator(StatsCfg(bootstrap_B=400, seed=2)).compare(base, {"v": var})
    r = _row(rows, "v", "ndcg@10")
    assert r.delta_ci_lo is not None and r.delta_ci_high is not None
    # No exception; CI, significant, significant_raw, and adjusted p all present as independent fields.
    assert isinstance(r.significant, bool)
    assert isinstance(r.significant_raw, bool)
    assert isinstance(r.p_value, float)
    assert r.p_value_adjusted is not None


# --------------------------------------------------------------------------------------------------
# Wilcoxon path parameters + permutation path reproducibility (§8.2).
# --------------------------------------------------------------------------------------------------


def test_wilcoxon_receives_zero_method_correction_two_sided(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Res:
        pvalue = 0.5

    def _fake_wilcoxon(deltas: object, **kwargs: object) -> _Res:
        captured.update(kwargs)
        return _Res()

    monkeypatch.setattr(stats_mod.scipy_stats, "wilcoxon", _fake_wilcoxon)
    base = _baseline({f"q{i}": 0.0 for i in range(10)})
    var = _variant({f"q{i}": 0.3 for i in range(10)})
    cfg = StatsCfg(test="wilcoxon", wilcoxon_zero_method="pratt", wilcoxon_correction=False, bootstrap_B=50)
    Comparator(cfg).compare(base, {"v": var})
    assert captured["zero_method"] == "pratt"
    assert captured["correction"] is False
    assert captured["alternative"] == "two-sided"


def test_permutation_path_is_seeded_and_reproducible() -> None:
    # n large enough that 2**n > bootstrap_B -> Monte-Carlo branch (seeded).
    n = 40
    base = _baseline({f"q{i}": 0.0 for i in range(n)})
    var: dict[str, dict[str, float]] = {}
    r = np.random.default_rng(11)
    vals = r.normal(0.1, 0.5, size=n)
    for i in range(n):
        d = _all_metrics(0.0)
        d["ndcg@10"] = float(vals[i])
        var[f"q{i}"] = d
    cfg = StatsCfg(test="permutation", bootstrap_B=1000, seed=42)
    p1 = _row(Comparator(cfg).compare(base, {"v": var}), "v", "ndcg@10").p_value
    p2 = _row(Comparator(cfg).compare(base, {"v": var}), "v", "ndcg@10").p_value
    assert p1 == p2  # seeded reproducibility
    assert 0.0 < p1 <= 1.0


def test_permutation_exact_enumeration_for_small_n() -> None:
    # n=3 -> 2**3=8 <= bootstrap_B: exact enumeration. Known deltas -> hand-checkable two-sided p.
    # deltas = [1, 1, 1]; obs |mean| = 1. Sign vectors give means in {-1,-1/3,1/3,1}. |mean|>=1 only
    # for all-+ and all-- => 2 of 8. p = (1 + 2)/(8 + 1) = 3/9 = 1/3.
    base = {"q0": _all_metrics(0.0), "q1": _all_metrics(0.0), "q2": _all_metrics(0.0)}
    var = {"q0": _all_metrics(1.0), "q1": _all_metrics(1.0), "q2": _all_metrics(1.0)}
    cfg = StatsCfg(test="permutation", bootstrap_B=1000, seed=1)
    p = _row(Comparator(cfg).compare(base, {"v": var}), "v", "avg_relevance").p_value
    assert p == pytest.approx(3.0 / 9.0)


# --------------------------------------------------------------------------------------------------
# Point estimate + row shape (§8.1 table).
# --------------------------------------------------------------------------------------------------


def test_delta_is_mean_of_paired_deltas() -> None:
    base = _baseline({"q0": 0.0, "q1": 0.0, "q2": 0.0})
    var = _variant({"q0": 0.1, "q1": 0.3, "q2": 0.5})  # deltas 0.1,0.3,0.5 -> mean 0.3
    rows = Comparator(StatsCfg(bootstrap_B=100, seed=1)).compare(base, {"v": var})
    assert _row(rows, "v", "avg_relevance").delta == pytest.approx(0.3)


def test_rows_are_ordered_by_variant_then_canonical_metric() -> None:
    base = _baseline({f"q{i}": 0.0 for i in range(5)})
    a = _variant({f"q{i}": 0.2 for i in range(5)})
    b = _variant({f"q{i}": 0.4 for i in range(5)})
    rows = Comparator(StatsCfg(bootstrap_B=50)).compare(base, {"zeta": b, "alpha": a})
    variants_seen = [r.variant for r in rows]
    # variants in sorted order
    assert variants_seen[:4] == ["alpha"] * 4
    assert variants_seen[4:] == ["zeta"] * 4
    # metrics in canonical order within each variant
    assert [r.metric for r in rows[:4]] == list(CANONICAL_METRICS)


# --------------------------------------------------------------------------------------------------
# Config guards.
# --------------------------------------------------------------------------------------------------


def test_unknown_correction_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        Comparator(StatsCfg(correction="holm"))
    with pytest.raises(NotImplementedError):
        Comparator(StatsCfg(correction="max_stat"))


def test_bh_and_by_corrections_are_accepted() -> None:
    base = _baseline({f"q{i}": 0.0 for i in range(20)})
    var = _variant({f"q{i}": 0.3 for i in range(20)})
    for correction in ("bh", "by"):
        rows = Comparator(StatsCfg(correction=correction, bootstrap_B=100)).compare(base, {"v": var})
        assert rows  # constructs and runs without error


def test_unknown_test_raises_value_error() -> None:
    with pytest.raises(ValueError):
        Comparator(StatsCfg(test="ttest"))
