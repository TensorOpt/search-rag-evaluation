"""Comparator: bootstrap CI, Wilcoxon/permutation p-value, FDR decision (docs/experiment.md §8). Phase 3.

The :class:`Comparator` pairs each non-baseline variant against the ``bm25`` baseline, per metric,
and produces one :class:`ComparisonResult` per ``(variant, metric)``. It implements the single
coherent multiple-comparison regime of §8 — **False Discovery Rate (FDR)** control, Benjamini-Hochberg
by default:

- **Pairing (§8.1).** Paired by ``query_id`` over queries present in BOTH runs AND whose metric value
  is finite (not NaN) in BOTH runs — the *generalized* per-metric NaN exclusion. Any metric may be
  NaN for a query (``avg_relevance``/``ndcg@10``/``precision@10`` when ``n_scored == 0``; ``recall@10``
  when ``R == 0``, §7); recall's ``R == 0`` case is just one instance of this rule.
- **Degenerate short-circuits (§8.1 table), BEFORE any scipy/bootstrap call.** An *empty paired set*
  yields ``delta``/CI = ``None``, ``p_value = 1.0``, ``significant_raw = False``,
  ``p_value_adjusted = 1.0``, ``significant = False``, ``note="empty_paired_set"``. *All-zero deltas*
  (>=1 paired query, every delta 0) yield ``delta = 0.0``, CI ``0.0/0.0``, ``p_value = 1.0``,
  ``significant_raw = False``, ``p_value_adjusted = 1.0``, ``significant = False``,
  ``note="all_zero_delta"``. Degenerate rows are NOT part of the FDR family and never trigger scipy
  or the RNG.
- **Effect-size CI (§8.2).** Percentile bootstrap over PAIRED QUERY INDICES with replacement,
  ``B = bootstrap_B`` resamples, using a FRESH ``numpy.random.default_rng(seed)`` per ``(variant, metric)``
  so the CI is fully deterministic regardless of iteration order. The CI is reported as effect-size
  context ONLY — it is never a significance gate and MAY DISAGREE with ``significant`` /
  ``significant_raw`` (§8.3).
- **Raw p-value (§8.2).** Two-sided Wilcoxon signed-rank (``zero_method``/``correction`` pinned), or a
  seeded sign-flip paired-permutation test, selected by ``StatsCfg.test``. ``significant_raw`` is the
  uncorrected per-test decision ``p_value <= alpha``, computed independently of the family.
- **FDR (§8.3), family-wide across the whole run.** The family = ALL non-degenerate ``(variant, metric)``
  tests (those with a real p-value). Benjamini-Hochberg (default) or Benjamini-Yekutieli adjusted
  p-values (q-values) are computed over the family; ``significant = (p_value_adjusted <= alpha)``.
  ``alpha`` is BOTH the raw per-test threshold AND the FDR target level ``q``. Degenerate rows are
  excluded from the family size ``m``.

This module imports ONLY the stdlib + numpy + scipy (no ``benchmark.*``): the comparator operates on
plain metric maps (``query_id -> {metric_name: value}``), exactly what ``Metrics.as_dict()`` produces;
the runner adapts ``Metrics`` -> maps before calling it (§11 import rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats as scipy_stats  # type: ignore[import-untyped]

#: Canonical metric names, in the fixed iteration order used for every comparison (§7, §9).
CANONICAL_METRICS: tuple[str, ...] = (
    "avg_relevance",
    "ndcg@10",
    "recall@10",
    "precision@10",
)

#: Absolute tolerance for treating a computed float as zero (never use `== 0.0` on floats).
ZERO_ABS_TOL = 1e-6


@dataclass(frozen=True)
class StatsCfg:
    """Statistics configuration (§8, §10 ``stats`` block).

    ``ci_level`` is the UNADJUSTED per-comparison bootstrap CI level (§8.2) — NOT a gate.
    ``alpha`` (0.05) is BOTH the raw per-test threshold AND the FDR target level ``q`` (§8.3).
    ``correction`` selects the FDR procedure: ``"bh"`` (Benjamini-Hochberg, default) or ``"by"``
    (Benjamini-Yekutieli, valid under arbitrary dependence). Any other value raises
    ``NotImplementedError``.
    """

    bootstrap_B: int = 10000
    ci_level: float = 0.95
    alpha: float = 0.05
    correction: str = "bh"
    test: str = "wilcoxon"
    wilcoxon_zero_method: str = "wilcox"
    wilcoxon_correction: bool = True
    seed: int = 1234


@dataclass(frozen=True)
class ComparisonResult:
    """One ``(variant, metric)`` comparison result (§8.1 table, §9 comparison CSV).

    ``delta``/``delta_ci_lo``/``delta_ci_high`` are ``None`` for an empty paired set (serialized as
    empty cells, §9). ``p_value`` is the RAW, uncorrected test p-value (Wilcoxon or permutation) and
    ``significant_raw`` is the uncorrected per-test decision (``p_value <= alpha``), independent of the
    family. ``p_value_adjusted`` is the FDR (BH/BY) adjusted p-value (q-value) over the family — ``1.0``
    for degenerate rows — and ``significant`` is the FDR decision (``p_value_adjusted <= alpha``, §8.3).
    Either significance flag may disagree with the CI. ``note`` records a degenerate case:
    ``"empty_paired_set"`` | ``"all_zero_delta"`` | ``None``.
    """

    variant: str
    metric: str
    delta: float | None
    delta_ci_lo: float | None
    delta_ci_high: float | None
    p_value: float
    significant_raw: bool
    p_value_adjusted: float | None
    significant: bool
    note: str | None = None


class Comparator:
    """Pairs variants vs the baseline and applies the §8 CI / p-value / FDR regime.

    ``compare`` returns rows in a deterministic order: variants sorted by id, and within each variant
    the canonical metrics in :data:`CANONICAL_METRICS` order.
    """

    def __init__(self, cfg: StatsCfg) -> None:
        if cfg.correction not in ("bh", "by"):
            # FDR procedures only: 'bh' (Benjamini-Hochberg, default) | 'by' (Benjamini-Yekutieli).
            raise NotImplementedError(
                f"correction={cfg.correction!r} is not implemented; "
                "only 'bh' (Benjamini-Hochberg) and 'by' (Benjamini-Yekutieli) are supported"
            )
        if cfg.test not in ("wilcoxon", "permutation"):
            raise ValueError(f"unknown test={cfg.test!r}; expected 'wilcoxon' or 'permutation'")
        self._cfg = cfg

    def compare(
        self,
        baseline: Mapping[str, Mapping[str, float]],
        variants: Mapping[str, Mapping[str, Mapping[str, float]]],
    ) -> list[ComparisonResult]:
        """Compare every variant vs the baseline, per canonical metric, with family-wide FDR (§8).

        ``baseline`` maps ``query_id -> {metric_name: value}``; ``variants`` maps
        ``variant_id -> query_id -> {metric_name: value}``. These are exactly what
        ``Metrics.as_dict()`` produces per query. The FDR correction (BH/BY) is applied across the
        family of all non-degenerate ``(variant, metric)`` tests in this call (§8.3).
        """
        cfg = self._cfg
        rows: list[ComparisonResult] = []
        # (index into rows, raw p-value) for each non-degenerate test (the FDR family).
        family: list[tuple[int, float]] = []

        for variant_id in sorted(variants):
            variant_metrics = variants[variant_id]
            for metric in CANONICAL_METRICS:
                deltas = _paired_deltas(baseline, variant_metrics, metric)

                if deltas.size == 0:
                    # Empty paired set (§8.1 table): no scipy/bootstrap call.
                    rows.append(
                        ComparisonResult(
                            variant=variant_id,
                            metric=metric,
                            delta=None,
                            delta_ci_lo=None,
                            delta_ci_high=None,
                            p_value=1.0,
                            significant_raw=False,
                            p_value_adjusted=1.0,
                            significant=False,
                            note="empty_paired_set",
                        )
                    )
                    continue

                if bool(np.all(np.isclose(deltas, 0.0, rtol=0.0, atol=ZERO_ABS_TOL))):
                    # All-zero deltas (§8.1 table): no scipy/bootstrap call.
                    # np.isclose (not `== 0.0`) — never test float equality; tolerance ZERO_ABS_TOL.
                    rows.append(
                        ComparisonResult(
                            variant=variant_id,
                            metric=metric,
                            delta=0.0,
                            delta_ci_lo=0.0,
                            delta_ci_high=0.0,
                            p_value=1.0,
                            significant_raw=False,
                            p_value_adjusted=1.0,
                            significant=False,
                            note="all_zero_delta",
                        )
                    )
                    continue

                # Real test: point estimate, bootstrap CI, raw p-value (§8.2).
                delta = float(np.mean(deltas))
                ci_lo, ci_high = self._bootstrap_ci(deltas)
                p_value = self._p_value(deltas)

                idx = len(rows)
                rows.append(
                    ComparisonResult(
                        variant=variant_id,
                        metric=metric,
                        delta=delta,
                        delta_ci_lo=ci_lo,
                        delta_ci_high=ci_high,
                        p_value=p_value,
                        # Raw per-test decision, independent of the family (§8.3).
                        significant_raw=p_value <= cfg.alpha,
                        p_value_adjusted=None,  # set by the FDR step below
                        significant=False,  # set by the FDR step below
                        note=None,
                    )
                )
                family.append((idx, p_value))

        # FDR (BH/BY) family-wide over the non-degenerate tests (§8.3).
        adjusted_by_idx = _fdr_adjust([p for _, p in family], cfg.correction)
        for (idx, _p), q in zip(family, adjusted_by_idx):
            row = rows[idx]
            rows[idx] = ComparisonResult(
                variant=row.variant,
                metric=row.metric,
                delta=row.delta,
                delta_ci_lo=row.delta_ci_lo,
                delta_ci_high=row.delta_ci_high,
                p_value=row.p_value,
                significant_raw=row.significant_raw,
                p_value_adjusted=q,
                significant=q <= cfg.alpha,
                note=row.note,
            )
        return rows

    def _bootstrap_ci(self, deltas: np.ndarray) -> tuple[float, float]:
        """Percentile bootstrap CI over paired query indices (§8.2).

        Resamples the paired deltas with replacement ``B`` times using a FRESH
        ``default_rng(seed)`` (deterministic regardless of iteration order), recomputes the mean
        each resample, and returns the ``ci_level`` percentiles (0.95 -> 2.5 / 97.5).
        """
        cfg = self._cfg
        rng = np.random.default_rng(cfg.seed)
        n = deltas.size
        # (B, n) matrix of resampled indices -> mean per row.
        idx = rng.integers(0, n, size=(cfg.bootstrap_B, n))
        resampled_means = deltas[idx].mean(axis=1)
        tail = (1.0 - cfg.ci_level) / 2.0 * 100.0  # 2.5 for ci_level=0.95
        lo, high = np.percentile(resampled_means, [tail, 100.0 - tail])
        return float(lo), float(high)

    def _p_value(self, deltas: np.ndarray) -> float:
        """Raw two-sided p-value: Wilcoxon signed-rank or seeded sign-flip permutation (§8.2)."""
        cfg = self._cfg
        if cfg.test == "wilcoxon":
            result = scipy_stats.wilcoxon(
                deltas,
                zero_method=cfg.wilcoxon_zero_method,
                correction=cfg.wilcoxon_correction,
                alternative="two-sided",
            )
            return float(result.pvalue)
        if cfg.test == "permutation":
            return self._permutation_p_value(deltas)
        # Exhaustive branch on the enumerated cfg.test — no silent default for an invalid value.
        raise ValueError(f"unsupported stats test: {cfg.test!r}")

    def _permutation_p_value(self, deltas: np.ndarray) -> float:
        """Seeded sign-flip paired-permutation test, two-sided (§8.2).

        Statistic = mean(delta). Exact enumeration when ``2**n <= bootstrap_B``, else Monte-Carlo
        with ``bootstrap_B`` sign-flip resamples using ``default_rng(seed)``.
        Two-sided p = ``(1 + #{|perm_stat| >= |obs_stat|}) / (B + 1)``.
        """
        cfg = self._cfg
        n = deltas.size
        mean_delta = float(np.mean(deltas))
        observed_stat = abs(mean_delta)  # two-sided statistic |mean delta|

        if 2**n <= cfg.bootstrap_B:
            # Exact enumeration over all 2^n sign vectors.
            signs = _sign_vectors(n)  # (2^n, n) of +/-1
            perm_stats = np.abs((signs * deltas).mean(axis=1))
            b = perm_stats.size
            count = int(np.sum(perm_stats >= observed_stat))
            return (1 + count) / (b + 1)

        rng = np.random.default_rng(cfg.seed)
        b = cfg.bootstrap_B
        signs = rng.choice(np.array([-1.0, 1.0]), size=(b, n))
        perm_stats = np.abs((signs * deltas).mean(axis=1))
        count = int(np.sum(perm_stats >= observed_stat))
        return (1 + count) / (b + 1)


def _paired_deltas(
    baseline: Mapping[str, Mapping[str, float]],
    variant_metrics: Mapping[str, Mapping[str, float]],
    metric: str,
) -> np.ndarray:
    """Paired ``variant - baseline`` deltas for ``metric`` over the finite-both query set (§8.1).

    Pairs by ``query_id`` over queries present in BOTH runs whose metric value is finite (not NaN) in
    BOTH runs — the generalized per-metric NaN exclusion. Query ids are iterated in sorted order so
    the delta vector is deterministic (the bootstrap seeds off it).
    """
    deltas: list[float] = []
    for qid in sorted(baseline):
        if qid not in variant_metrics:
            continue
        baseline_value = baseline[qid].get(metric, math.nan)
        variant_value = variant_metrics[qid].get(metric, math.nan)
        if math.isnan(baseline_value) or math.isnan(variant_value):
            continue
        deltas.append(variant_value - baseline_value)
    return np.asarray(deltas, dtype=float)


def _sign_vectors(n: int) -> np.ndarray:
    """All ``2**n`` sign vectors in ``{-1, +1}`` as a ``(2**n, n)`` float array (permutation test)."""
    bits = ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    return 1.0 - 2.0 * bits  # 0 -> +1, 1 -> -1


def _fdr_adjust(ps: Sequence[float], method: str) -> list[float]:
    """FDR-adjusted p-values (q-values) over the family, in input order (§8.3).

    ``method`` is ``"bh"`` (Benjamini-Hochberg, controls FDR under independence and PRDS) or ``"by"``
    (Benjamini-Yekutieli, valid under arbitrary dependence). Computed with
    ``scipy.stats.false_discovery_control`` (added in scipy 1.11 and pinned as the floor in
    pyproject.toml, so the routine is always present — no runtime capability probing).

    ``significant`` for a test is then ``q <= alpha``; because BH's adjusted p-value ``q_(k)`` satisfies
    ``q_(k) <= alpha`` iff ``k`` is within the BH step-up rejection set, this reproduces the classic
    step-up rule (largest ``k`` with ``p_(k) <= (k/m)*alpha``; reject all with rank ``<= k``).
    """
    if not ps:
        return []
    adjusted = scipy_stats.false_discovery_control(np.asarray(ps, dtype=float), method=method)
    return [float(min(q, 1.0)) for q in adjusted]
