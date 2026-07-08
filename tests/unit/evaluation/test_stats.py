"""Unit tests for the Comparator (docs/methodology.md §8). Phase 3 + methodology_fixes.md Fix 3/6/7.

Covers: seeded bootstrap-CI determinism and B honoring; the two degenerate short-circuits
(empty paired set / all-zero deltas) with a hard assertion that scipy/bootstrap/RNG are NEVER
called and that per M3 they carry ``in_family=false`` + empty ``p_value_adjusted``/``significant``;
the family-wide common subset (one value per system, baseline included) and the per-metric NaN
exclusion; arbitrary contrasts (variant-vs-variant, not just vs baseline); FDR family gating by
``contrast.family`` × ``fdr_metrics`` (Fix 7); family-wide FDR (Benjamini-Hochberg step-up + q-values,
BY as more-conservative); raw-vs-FDR significance; CI-vs-significant disagreement; the Wilcoxon
zero_method/correction/two-sided path and the seeded reproducible permutation path.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from benchmark.evaluation import stats as stats_mod
from benchmark.evaluation.stats import (
    CANONICAL_METRICS,
    Comparator,
    ComparisonResult,
    Contrast,
    StatsCfg,
    _common_qids,
    _fdr_adjust,
)

# --------------------------------------------------------------------------------------------------
# Helpers: build system metric maps in the plain `Metrics.as_dict()` shape + drive the new API.
# --------------------------------------------------------------------------------------------------


def _all_metrics(value: float) -> dict[str, float]:
    """A per-query metric map with the same value for all canonical metrics."""
    return {m: value for m in CANONICAL_METRICS}


def _qmap(values: dict[str, float]) -> dict[str, dict[str, float]]:
    """query_id -> {metric: value}, same value across metrics, for the given per-query values."""
    return {qid: _all_metrics(v) for qid, v in values.items()}


# Backward-compatible aliases used throughout the (adapted) tests.
_baseline = _qmap
_variant = _qmap


def _compare(
    cfg: StatsCfg,
    base: dict[str, dict[str, float]],
    variants: dict[str, dict[str, dict[str, float]]],
    *,
    family: bool = True,
    fdr_metrics: tuple[str, ...] = CANONICAL_METRICS,
) -> list[ComparisonResult]:
    """Reproduce the old all-vs-``bm25`` comparison via the new (systems, contrasts) API.

    ``fdr_metrics`` defaults to ALL canonical metrics so every non-degenerate row is in the family —
    matching the old "family = all non-degenerate (variant, metric)" behavior. Individual tests
    narrow it to exercise Fix-7 gating.
    """
    cfg = replace(cfg, fdr_metrics=tuple(fdr_metrics))
    systems = {"bm25": base, **variants}
    contrasts = [Contrast(a=vid, b="bm25", family=family) for vid in variants]
    return Comparator(cfg).compare(systems, contrasts)


def _row(rows: list[ComparisonResult], system_a: str, metric: str) -> ComparisonResult:
    (r,) = [x for x in rows if x.system_a == system_a and x.metric == metric]
    return r


# --------------------------------------------------------------------------------------------------
# Seeded determinism + B honored (§8.2).
# --------------------------------------------------------------------------------------------------


def test_seeded_ci_is_deterministic_across_repeated_compare() -> None:
    base = _baseline({f"q{i}": float(i % 3) for i in range(30)})
    var = _variant({f"q{i}": float(i % 3) + 0.3 for i in range(30)})
    cfg = StatsCfg(bootstrap_B=2000, seed=1234)
    rows1 = _compare(cfg, base, {"v": var})
    rows2 = _compare(cfg, base, {"v": var})
    for m in CANONICAL_METRICS:
        r1, r2 = _row(rows1, "v", m), _row(rows2, "v", m)
        assert r1.delta_ci_lo == r2.delta_ci_lo
        assert r1.delta_ci_high == r2.delta_ci_high


def test_different_seed_generally_changes_ci() -> None:
    base = _baseline({f"q{i}": float(i % 4) for i in range(40)})
    var = _variant({f"q{i}": float(i % 4) + (0.5 if i % 2 else -0.2) for i in range(40)})
    r_a = _row(_compare(StatsCfg(bootstrap_B=2000, seed=1), base, {"v": var}), "v", "ndcg@10")
    r_b = _row(_compare(StatsCfg(bootstrap_B=2000, seed=999), base, {"v": var}), "v", "ndcg@10")
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
    _compare(StatsCfg(bootstrap_B=1234, seed=7), base, {"v": var})
    # Every bootstrap draws a (B, n) index matrix: B == bootstrap_B.
    assert seen_sizes, "bootstrap RNG.integers was never called"
    assert all(sz[0] == 1234 for sz in seen_sizes)


# --------------------------------------------------------------------------------------------------
# Degenerate short-circuits (§8.1 table): no scipy/bootstrap/RNG; NOT in family; empty adjusted (M3).
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
    # Every metric NaN on every query for both runs -> empty paired set for every metric; scipy/rng
    # are never touched.
    base = {"q1": _all_metrics(math.nan), "q2": _all_metrics(math.nan)}
    var = {"q1": _all_metrics(math.nan), "q2": _all_metrics(math.nan)}
    rows = _compare(StatsCfg(), base, {"v": var})
    for m in CANONICAL_METRICS:
        r = _row(rows, "v", m)
        assert r.value_a is None
        assert r.value_b is None
        assert r.delta is None
        assert r.delta_ci_lo is None
        assert r.delta_ci_high is None
        assert r.p_value == 1.0
        assert r.significant_raw is False
        assert r.in_family is False
        assert r.p_value_adjusted is None  # M3: in_family=false -> empty
        assert r.significant is None  # M3: in_family=false -> empty
        assert r.n_common == 0
        assert r.note == "empty_paired_set"


def test_all_zero_delta_row_and_no_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_scipy_and_rng(monkeypatch)
    base = _baseline({f"q{i}": float(i % 2) for i in range(10)})
    var = _variant({f"q{i}": float(i % 2) for i in range(10)})  # identical -> all-zero deltas
    rows = _compare(StatsCfg(), base, {"v": var})
    # Mean of {0,1,0,1,...} over 10 queries is 0.5; a==b per query so both means equal.
    for m in CANONICAL_METRICS:
        r = _row(rows, "v", m)
        assert math.isclose(r.value_a, 0.5, abs_tol=1e-6)
        assert math.isclose(r.value_b, 0.5, abs_tol=1e-6)
        assert r.value_a == r.value_b
        assert r.delta == 0.0
        assert r.delta_ci_lo == 0.0
        assert r.delta_ci_high == 0.0
        assert r.p_value == 1.0
        assert r.significant_raw is False
        assert r.in_family is False
        assert r.p_value_adjusted is None  # M3
        assert r.significant is None  # M3
        assert r.n_common == 10
        assert r.note == "all_zero_delta"


def test_degenerate_rows_excluded_from_fdr_family() -> None:
    # One real test on avg_relevance (tiny p so it would reject at any sane m), and a degenerate
    # all-zero row for the other metrics. Degenerate rows must NOT enter the FDR family: with
    # fdr_metrics={avg_relevance} the family only contains the real test (m == 1), so its BH q-value
    # equals the raw p and the test rejects.
    n = 30
    base = _baseline({f"q{i}": 0.0 for i in range(n)})
    var: dict[str, dict[str, float]] = {}
    for i in range(n):
        d = _all_metrics(0.0)
        d["avg_relevance"] = 1.0  # constant +1 delta -> very significant
        var[f"q{i}"] = d
    rows = _compare(
        StatsCfg(alpha=0.05, bootstrap_B=500), base, {"v": var}, fdr_metrics=("avg_relevance",)
    )
    real = _row(rows, "v", "avg_relevance")
    assert real.note is None
    assert real.in_family is True
    assert real.significant is True  # only member of the family; q == raw p tiny <= 0.05
    assert real.p_value_adjusted == pytest.approx(real.p_value)  # m == 1 -> q == p
    for m in ("ndcg@10", "recall@10", "recall@50", "recall@100", "precision@10"):
        r = _row(rows, "v", m)
        assert r.note == "all_zero_delta"
        assert r.in_family is False
        assert r.p_value_adjusted is None
        assert r.significant is None
        assert r.significant_raw is False


# --------------------------------------------------------------------------------------------------
# Family-wide common subset (§8.1, Fix 6) — one value per system, per-metric NaN exclusion.
# --------------------------------------------------------------------------------------------------


def test_per_metric_nan_exclusion_uses_shared_common_subset() -> None:
    # q0..q5 finite everywhere; q5 has NaN ndcg in the variant (n_scored==0) -> ndcg pairs 5, not 6.
    base = _baseline({f"q{i}": 0.1 for i in range(6)})
    var: dict[str, dict[str, float]] = {}
    for i in range(6):
        var[f"q{i}"] = _all_metrics(0.5)
    var["q5"]["ndcg@10"] = math.nan  # excluded for ndcg only
    var["q3"]["recall@10"] = math.nan  # excluded for recall@10 only

    cfg = StatsCfg(bootstrap_B=200, seed=3)
    sizes: dict[str, int] = {}
    orig = stats_mod._paired_values

    def _spy(map_a: object, map_b: object, metric: str, common_qids: object) -> tuple[np.ndarray, np.ndarray]:
        a_arr, b_arr = orig(map_a, map_b, metric, common_qids)  # type: ignore[arg-type]
        assert a_arr.size == b_arr.size  # the two arrays share one common mask
        sizes[metric] = a_arr.size
        return a_arr, b_arr

    stats_mod._paired_values = _spy  # type: ignore[assignment]
    try:
        _compare(cfg, base, {"v": var})
    finally:
        stats_mod._paired_values = orig  # type: ignore[assignment]

    assert sizes["avg_relevance"] == 6
    assert sizes["precision@10"] == 6
    assert sizes["recall@50"] == 6
    assert sizes["recall@100"] == 6
    assert sizes["ndcg@10"] == 5  # q5 excluded
    assert sizes["recall@10"] == 5  # q3 excluded


def test_common_subset_gives_baseline_one_value_across_contrasts() -> None:
    # THE 3-distinct-baseline bug fix: v1 is NaN on q2, v2 on q3. The family-wide subset for
    # avg_relevance is the UNION of per-system NaN queries over {bm25, v1, v2} = exclude {q2, q3}, so
    # the baseline reports ONE value across BOTH contrasts (not a per-pair number).
    base = _qmap({"q0": 0.1, "q1": 0.2, "q2": 0.3, "q3": 0.4, "q4": 0.5})
    v1 = _qmap({"q0": 0.6, "q1": 0.7, "q2": 0.8, "q3": 0.9, "q4": 1.0})
    v2 = _qmap({"q0": 0.2, "q1": 0.3, "q2": 0.4, "q3": 0.5, "q4": 0.6})
    v1["q2"]["avg_relevance"] = math.nan
    v2["q3"]["avg_relevance"] = math.nan

    systems = {"bm25": base, "v1": v1, "v2": v2}
    contrasts = [Contrast("v1", "bm25", True), Contrast("v2", "bm25", True)]
    rows = Comparator(StatsCfg(bootstrap_B=100, seed=1)).compare(systems, contrasts)

    r1 = _row(rows, "v1", "avg_relevance")
    r2 = _row(rows, "v2", "avg_relevance")
    # Common subset for avg_relevance = {q0, q1, q4}; baseline mean over it, identical on both rows.
    expected_baseline = (0.1 + 0.2 + 0.5) / 3
    assert r1.n_common == 3
    assert r2.n_common == 3
    assert math.isclose(r1.value_b, expected_baseline, abs_tol=1e-9)
    assert r1.value_b == r2.value_b  # ONE baseline value across contrasts


def test_common_qids_scoped_to_referenced_systems() -> None:
    # A system in NO contrast must not shrink the common subset (S2). `spoiler` is all-NaN but is not
    # referenced by any contrast, so the (v, bm25) subset is unaffected.
    base = _qmap({"q0": 0.1, "q1": 0.2})
    var = _qmap({"q0": 0.5, "q1": 0.6})
    spoiler = {"q0": _all_metrics(math.nan), "q1": _all_metrics(math.nan)}
    systems = {"bm25": base, "v": var, "spoiler": spoiler}
    contrasts = [Contrast("v", "bm25", True)]
    assert _common_qids(systems, contrasts, "avg_relevance") == ["q0", "q1"]
    rows = Comparator(StatsCfg(bootstrap_B=50)).compare(systems, contrasts)
    assert _row(rows, "v", "avg_relevance").n_common == 2


def test_empty_common_subset_is_empty_paired_set() -> None:
    # recall@10 NaN (R==0) for the ONLY query in the baseline -> empty subset for recall@10 (S3).
    base = {"q0": {**_all_metrics(0.4), "recall@10": math.nan}}
    var = {"q0": {**_all_metrics(0.6), "recall@10": 0.6}}
    rows = _compare(StatsCfg(bootstrap_B=100), base, {"v": var})
    empty = _row(rows, "v", "recall@10")
    assert empty.note == "empty_paired_set"
    assert empty.n_common == 0
    assert empty.in_family is False
    assert empty.p_value_adjusted is None
    assert empty.significant is None
    # A non-recall metric with a single finite pair is a real 1-sample test (not degenerate-empty).
    assert _row(rows, "v", "avg_relevance").note is None


# --------------------------------------------------------------------------------------------------
# Arbitrary contrasts (Fix 3) — variant-vs-variant, not just vs baseline.
# --------------------------------------------------------------------------------------------------


def test_variant_vs_variant_contrast() -> None:
    # delta = value(a) - value(b) for a contrast that never touches the baseline.
    base = _qmap({f"q{i}": 0.0 for i in range(10)})
    v1 = _qmap({f"q{i}": 0.7 for i in range(10)})
    v2 = _qmap({f"q{i}": 0.2 for i in range(10)})
    systems = {"bm25": base, "v1": v1, "v2": v2}
    contrasts = [Contrast("v1", "v2", True)]  # v1 vs v2, baseline absent
    rows = Comparator(StatsCfg(bootstrap_B=100, seed=1)).compare(systems, contrasts)
    r = _row(rows, "v1", "avg_relevance")
    assert r.system_a == "v1"
    assert r.system_b == "v2"
    assert math.isclose(r.value_a, 0.7, abs_tol=1e-9)
    assert math.isclose(r.value_b, 0.2, abs_tol=1e-9)
    assert math.isclose(r.delta, 0.5, abs_tol=1e-9)  # a - b


def test_rows_ordered_by_contrast_then_canonical_metric() -> None:
    base = _baseline({f"q{i}": 0.0 for i in range(5)})
    a = _variant({f"q{i}": 0.2 for i in range(5)})
    b = _variant({f"q{i}": 0.4 for i in range(5)})
    systems = {"bm25": base, "alpha": a, "zeta": b}
    # Contrast order is preserved as given (NOT sorted): zeta first, then alpha.
    contrasts = [Contrast("zeta", "bm25", True), Contrast("alpha", "bm25", True)]
    rows = Comparator(StatsCfg(bootstrap_B=50)).compare(systems, contrasts)
    systems_seen = [r.system_a for r in rows]
    n = len(CANONICAL_METRICS)
    assert systems_seen[:n] == ["zeta"] * n
    assert systems_seen[n:] == ["alpha"] * n
    # metrics in canonical order within each contrast
    assert [r.metric for r in rows[:n]] == list(CANONICAL_METRICS)


# --------------------------------------------------------------------------------------------------
# FDR family gating (Fix 7) — only contrast.family × fdr_metrics rows are adjusted.
# --------------------------------------------------------------------------------------------------


def test_in_family_gating_only_headline_metrics_adjusted() -> None:
    # Two headline metrics enter the family; the others are descriptive (raw p only, empty adjusted).
    n = 30
    base = _baseline({f"q{i}": 0.0 for i in range(n)})
    var: dict[str, dict[str, float]] = {}
    for i in range(n):
        d = _all_metrics(0.3)  # constant +0.3 delta on every metric -> tiny p everywhere
        var[f"q{i}"] = d
    rows = _compare(
        StatsCfg(alpha=0.05, bootstrap_B=500),
        base,
        {"v": var},
        fdr_metrics=("ndcg@10", "recall@100"),
    )
    for m in ("ndcg@10", "recall@100"):
        r = _row(rows, "v", m)
        assert r.in_family is True
        assert r.p_value_adjusted is not None
        assert r.significant is True
    for m in ("avg_relevance", "recall@10", "recall@50", "precision@10"):
        r = _row(rows, "v", m)
        assert r.in_family is False
        assert r.p_value_adjusted is None  # M3: descriptive -> empty adjusted cells
        assert r.significant is None
        assert r.significant_raw is True  # raw per-test decision still populated on descriptive rows


def test_non_family_contrast_is_descriptive() -> None:
    # A contrast flagged family=false is descriptive-only even for a headline metric.
    n = 20
    base = _baseline({f"q{i}": 0.0 for i in range(n)})
    var = _variant({f"q{i}": 0.4 for i in range(n)})
    systems = {"bm25": base, "v": var}
    contrasts = [Contrast("v", "bm25", family=False)]
    rows = Comparator(
        replace(StatsCfg(bootstrap_B=200), fdr_metrics=("ndcg@10",))
    ).compare(systems, contrasts)
    r = _row(rows, "v", "ndcg@10")
    assert r.in_family is False
    assert r.p_value_adjusted is None
    assert r.significant is None
    assert r.significant_raw is True  # raw decision independent of the family


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
    ps = [0.02, 0.7, 0.001, 0.04, 0.008]
    alpha = 0.05
    q = _fdr_adjust(ps, "bh")
    reject_expected = _bh_reject_set(ps, alpha)
    assert reject_expected == {0, 2, 3, 4}  # everything except the p=0.7 test (index 1)

    reject_from_q = {i for i, qi in enumerate(q) if qi <= alpha}
    assert reject_from_q == reject_expected  # significant == (q <= alpha) reproduces the step-up set

    order = sorted(range(len(ps)), key=lambda i: ps[i])
    q_sorted = [q[i] for i in order]
    assert q_sorted == sorted(q_sorted)  # monotone in rank
    assert all(0.0 <= qi <= 1.0 for qi in q)


def test_bh_qvalues_match_hand_computed() -> None:
    ps = [0.001, 0.008, 0.02, 0.04, 0.7, 0.5]
    expected = [0.006, 0.024, 0.04, 0.06, 0.7, 0.6]
    assert _fdr_adjust(ps, "bh") == pytest.approx(expected)


def test_by_is_more_conservative_than_bh() -> None:
    ps = [0.001, 0.008, 0.02, 0.04, 0.7, 0.5]
    alpha = 0.05
    bh_q = _fdr_adjust(ps, "bh")
    by_q = _fdr_adjust(ps, "by")
    bh_reject = {i for i, q in enumerate(bh_q) if q <= alpha}
    by_reject = {i for i, q in enumerate(by_q) if q <= alpha}
    assert by_reject <= bh_reject
    assert by_reject != bh_reject  # strictly more conservative on this family
    for b, y in zip(bh_q, by_q):
        assert y >= b - 1e-12


def test_raw_significant_can_exceed_fdr_significant() -> None:
    ps = [0.04, 0.6, 0.7, 0.8, 0.9]
    alpha = 0.05
    q = _fdr_adjust(ps, "bh")
    assert ps[0] <= alpha  # raw-significant
    assert q[0] > alpha  # but NOT FDR-significant after correction


def test_bh_more_powerful_than_holm_but_still_corrects() -> None:
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
    # Two variants: strong effect on avg_relevance (raw AND FDR significant), weak on another
    # (neither). fdr_metrics narrowed to avg_relevance so the family holds both variants' rows.
    n = 40
    base = _baseline({f"q{i}": 0.0 for i in range(n)})
    strong: dict[str, dict[str, float]] = {}
    weak: dict[str, dict[str, float]] = {}
    for i in range(n):
        s = _all_metrics(0.0)
        s["avg_relevance"] = 1.0  # huge, constant +1 -> tiny p
        strong[f"q{i}"] = s
        w = _all_metrics(0.0)
        w["avg_relevance"] = 0.001 if i % 2 else -0.001  # near-zero mean -> large p
        weak[f"q{i}"] = w
    rows = _compare(
        StatsCfg(alpha=0.05, bootstrap_B=300),
        base,
        {"strong": strong, "weak": weak},
        fdr_metrics=("avg_relevance",),
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
    n = 25
    base = _baseline({f"q{i}": 0.0 for i in range(n)})
    var: dict[str, dict[str, float]] = {}
    for i in range(n):
        d = _all_metrics(0.0)
        d["ndcg@10"] = 0.2  # constant positive -> CI well above 0
        var[f"q{i}"] = d
    rows = _compare(StatsCfg(alpha=0.05, bootstrap_B=500, seed=5), base, {"target": var})
    r = _row(rows, "target", "ndcg@10")
    assert r.delta is not None and r.delta_ci_lo is not None and r.delta_ci_high is not None
    assert r.delta_ci_lo > 0.0  # CI excludes 0
    assert isinstance(r.significant, bool)  # ndcg@10 is in the (full) family here
    assert isinstance(r.significant_raw, bool)


def test_ci_includes_zero_no_exception() -> None:
    rng = np.random.default_rng(0)
    n = 60
    base = _baseline({f"q{i}": 0.0 for i in range(n)})
    var: dict[str, dict[str, float]] = {}
    draws = rng.normal(0.0, 1.0, size=n)
    for i in range(n):
        d = _all_metrics(0.0)
        d["ndcg@10"] = float(draws[i])
        var[f"q{i}"] = d
    rows = _compare(StatsCfg(bootstrap_B=400, seed=2), base, {"v": var})
    r = _row(rows, "v", "ndcg@10")
    assert r.delta_ci_lo is not None and r.delta_ci_high is not None
    assert isinstance(r.significant, bool)
    assert isinstance(r.significant_raw, bool)
    assert isinstance(r.p_value, float)
    assert r.p_value_adjusted is not None  # ndcg@10 in the full family -> adjusted present


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
    _compare(cfg, base, {"v": var})
    assert captured["zero_method"] == "pratt"
    assert captured["correction"] is False
    assert captured["alternative"] == "two-sided"


def test_permutation_is_the_default_test() -> None:
    # Fix 2: the default StatsCfg().test is the mean-δ permutation test.
    assert StatsCfg().test == "permutation"


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
    p1 = _row(_compare(cfg, base, {"v": var}), "v", "ndcg@10").p_value
    p2 = _row(_compare(cfg, base, {"v": var}), "v", "ndcg@10").p_value
    assert p1 == p2  # seeded reproducibility
    assert 0.0 < p1 <= 1.0


def test_permutation_exact_enumeration_for_small_n() -> None:
    # n=3 -> 2**3=8 <= bootstrap_B: exact enumeration. deltas = [1,1,1]; obs |mean| = 1. Sign vectors
    # give means in {-1,-1/3,1/3,1}. |mean|>=1 only for all-+/all-- => 2 of 8. p = (1+2)/(8+1) = 1/3.
    base = {"q0": _all_metrics(0.0), "q1": _all_metrics(0.0), "q2": _all_metrics(0.0)}
    var = {"q0": _all_metrics(1.0), "q1": _all_metrics(1.0), "q2": _all_metrics(1.0)}
    cfg = StatsCfg(test="permutation", bootstrap_B=1000, seed=1)
    p = _row(_compare(cfg, base, {"v": var}), "v", "avg_relevance").p_value
    assert p == pytest.approx(3.0 / 9.0)


# --------------------------------------------------------------------------------------------------
# Point estimate + row shape (§8.1 table).
# --------------------------------------------------------------------------------------------------


def test_delta_is_mean_of_paired_deltas() -> None:
    base = _baseline({"q0": 0.0, "q1": 0.0, "q2": 0.0})
    var = _variant({"q0": 0.1, "q1": 0.3, "q2": 0.5})  # deltas 0.1,0.3,0.5 -> mean 0.3
    rows = _compare(StatsCfg(bootstrap_B=100, seed=1), base, {"v": var})
    assert _row(rows, "v", "avg_relevance").delta == pytest.approx(0.3)


def test_value_a_and_value_b_are_paired_means() -> None:
    # base means 0.2 (0.1,0.2,0.3), var means 0.5 (0.4,0.5,0.6) -> delta == a - b == 0.3.
    base = _baseline({"q0": 0.1, "q1": 0.2, "q2": 0.3})
    var = _variant({"q0": 0.4, "q1": 0.5, "q2": 0.6})
    r = _row(_compare(StatsCfg(bootstrap_B=100, seed=1), base, {"v": var}), "v", "avg_relevance")
    assert math.isclose(r.value_b, 0.2, abs_tol=1e-6)  # baseline (system_b)
    assert math.isclose(r.value_a, 0.5, abs_tol=1e-6)  # variant (system_a)
    assert math.isclose(r.delta, r.value_a - r.value_b, abs_tol=1e-6)


def test_value_a_and_value_b_use_the_same_nan_mask() -> None:
    # q2 is NaN in the variant for avg_relevance -> excluded from BOTH means (shared mask), so
    # value_b (baseline) is the mean over {q0,q1} only, not all three baseline queries.
    base = _baseline({"q0": 0.2, "q1": 0.4, "q2": 1.0})
    var = _variant({"q0": 0.5, "q1": 0.7, "q2": 0.9})
    var["q2"]["avg_relevance"] = math.nan
    r = _row(_compare(StatsCfg(bootstrap_B=100, seed=1), base, {"v": var}), "v", "avg_relevance")
    assert math.isclose(r.value_b, 0.3, abs_tol=1e-6)  # mean(0.2, 0.4), q2 dropped
    assert math.isclose(r.value_a, 0.6, abs_tol=1e-6)  # mean(0.5, 0.7)
    assert math.isclose(r.delta, r.value_a - r.value_b, abs_tol=1e-6)
    assert r.n_common == 2


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
        rows = _compare(StatsCfg(correction=correction, bootstrap_B=100), base, {"v": var})
        assert rows  # constructs and runs without error


def test_unknown_test_raises_value_error() -> None:
    with pytest.raises(ValueError):
        Comparator(StatsCfg(test="ttest"))
