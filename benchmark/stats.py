"""Comparator: bootstrap CI, Wilcoxon/permutation p-value, Holm decision (docs/experiment.md §8). Phase 3.

The :class:`Comparator` pairs each non-baseline variant against the ``bm25`` baseline, per metric,
and produces one :class:`ComparisonRow` per ``(variant, metric)``. It implements the single coherent
error-control regime of §8:

- **Pairing (§8.1).** Paired by ``query_id`` over queries present in BOTH runs AND whose metric value
  is finite (not NaN) in BOTH runs — the *generalized* per-metric NaN exclusion. Any metric may be
  NaN for a query (``avg_relevance``/``ndcg@10``/``precision@10`` when ``n_scored == 0``; ``recall@10``
  when ``R == 0``, §7); recall's ``R == 0`` case is just one instance of this rule.
- **Degenerate short-circuits (§8.1 table), BEFORE any scipy/bootstrap call.** An *empty paired set*
  yields ``delta``/CI = ``None``, ``p_value = 1.0``, ``significant = False``, ``note="empty_paired_set"``.
  *All-zero deltas* (>=1 paired query, every delta 0) yield ``delta = 0.0``, CI ``0.0/0.0``,
  ``p_value = 1.0``, ``significant = False``, ``note="all_zero_delta"``. Degenerate rows are NOT part
  of the Holm family and never trigger scipy or the RNG.
- **Effect-size CI (§8.2).** Percentile bootstrap over PAIRED QUERY INDICES with replacement,
  ``B = bootstrap_B`` resamples, using a FRESH ``numpy.random.default_rng(seed)`` per ``(variant, metric)``
  so the CI is fully deterministic regardless of iteration order. The CI is reported as effect-size
  context ONLY — it is never a significance gate and MAY DISAGREE with ``significant`` (§8.3).
- **p-value (§8.2).** Two-sided Wilcoxon signed-rank (``zero_method``/``correction`` pinned), or a
  seeded sign-flip paired-permutation test, selected by ``StatsCfg.test``.
- **Holm (§8.3), family-wide across the whole run.** The family = ALL non-degenerate ``(variant, metric)``
  tests. Holm–Bonferroni step-down at family ``alpha`` on the raw p-values; ``significant`` is exactly
  the Holm reject/retain outcome. Degenerate rows are excluded from the family size ``m``.

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


@dataclass(frozen=True)
class StatsCfg:
    """Statistics configuration (§8, §10 ``stats`` block).

    ``ci_level`` is the UNADJUSTED per-comparison bootstrap CI level (§8.2) — NOT a gate.
    ``alpha`` is the family-wise Holm alpha (§8.3). ``correction`` only implements ``"holm"``;
    ``max_stat``/``bh_fdr`` are deferred (§13) and raise ``NotImplementedError``.
    """

    bootstrap_B: int = 10000
    ci_level: float = 0.95
    alpha: float = 0.05
    correction: str = "holm"
    test: str = "wilcoxon"
    wilcoxon_zero_method: str = "wilcox"
    wilcoxon_correction: bool = True
    seed: int = 1234


@dataclass(frozen=True)
class ComparisonRow:
    """One ``(variant, metric)`` comparison result (§8.1 table, §9 comparison CSV).

    ``delta``/``delta_ci_lo``/``delta_ci_high`` are ``None`` for an empty paired set (serialized as
    empty cells, §9). ``p_value`` is the RAW test p-value (Wilcoxon or permutation). ``significant``
    is the Holm-corrected decision (§8.3) and may disagree with the CI. ``note`` records a degenerate
    case: ``"empty_paired_set"`` | ``"all_zero_delta"`` | ``None``.
    """

    variant: str
    metric: str
    delta: float | None
    delta_ci_lo: float | None
    delta_ci_high: float | None
    significant: bool
    p_value: float
    note: str | None = None


class Comparator:
    """Pairs variants vs the baseline and applies the §8 CI / p-value / Holm regime.

    ``compare`` returns rows in a deterministic order: variants sorted by id, and within each variant
    the canonical metrics in :data:`CANONICAL_METRICS` order.
    """

    def __init__(self, cfg: StatsCfg) -> None:
        if cfg.correction != "holm":
            # max_stat | bh_fdr are the deferred joint regimes (§8.3, §13).
            raise NotImplementedError(
                f"correction={cfg.correction!r} is not implemented; "
                "only 'holm' is supported (max_stat|bh_fdr deferred, §13)"
            )
        if cfg.test not in ("wilcoxon", "permutation"):
            raise ValueError(f"unknown test={cfg.test!r}; expected 'wilcoxon' or 'permutation'")
        self._cfg = cfg

    def compare(
        self,
        baseline: Mapping[str, Mapping[str, float]],
        variants: Mapping[str, Mapping[str, Mapping[str, float]]],
    ) -> list[ComparisonRow]:
        """Compare every variant vs the baseline, per canonical metric, with family-wide Holm (§8).

        ``baseline`` maps ``query_id -> {metric_name: value}``; ``variants`` maps
        ``variant_id -> query_id -> {metric_name: value}``. These are exactly what
        ``Metrics.as_dict()`` produces per query. Holm is applied across the family of all
        non-degenerate ``(variant, metric)`` tests in this call (§8.3).
        """
        cfg = self._cfg
        rows: list[ComparisonRow] = []
        # (index into rows) for each non-degenerate test, plus its raw p-value and Holm sort key.
        family: list[tuple[int, float, tuple[str, str]]] = []

        for variant_id in sorted(variants):
            variant_metrics = variants[variant_id]
            for metric in CANONICAL_METRICS:
                deltas = _paired_deltas(baseline, variant_metrics, metric)

                if deltas.size == 0:
                    # Empty paired set (§8.1 table): no scipy/bootstrap call.
                    rows.append(
                        ComparisonRow(
                            variant=variant_id,
                            metric=metric,
                            delta=None,
                            delta_ci_lo=None,
                            delta_ci_high=None,
                            significant=False,
                            p_value=1.0,
                            note="empty_paired_set",
                        )
                    )
                    continue

                if np.all(deltas == 0.0):
                    # All-zero deltas (§8.1 table): no scipy/bootstrap call.
                    rows.append(
                        ComparisonRow(
                            variant=variant_id,
                            metric=metric,
                            delta=0.0,
                            delta_ci_lo=0.0,
                            delta_ci_high=0.0,
                            significant=False,
                            p_value=1.0,
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
                    ComparisonRow(
                        variant=variant_id,
                        metric=metric,
                        delta=delta,
                        delta_ci_lo=ci_lo,
                        delta_ci_high=ci_high,
                        significant=False,  # set by Holm below
                        p_value=p_value,
                        note=None,
                    )
                )
                family.append((idx, p_value, (variant_id, metric)))

        # Holm–Bonferroni family-wide over the non-degenerate tests (§8.3).
        significant_by_idx = _holm(family, cfg.alpha)
        for idx, is_sig in significant_by_idx.items():
            row = rows[idx]
            rows[idx] = ComparisonRow(
                variant=row.variant,
                metric=row.metric,
                delta=row.delta,
                delta_ci_lo=row.delta_ci_lo,
                delta_ci_high=row.delta_ci_high,
                significant=is_sig,
                p_value=row.p_value,
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
        return self._permutation_p_value(deltas)

    def _permutation_p_value(self, deltas: np.ndarray) -> float:
        """Seeded sign-flip paired-permutation test, two-sided (§8.2).

        Statistic = mean(delta). Exact enumeration when ``2**n <= bootstrap_B``, else Monte-Carlo
        with ``bootstrap_B`` sign-flip resamples using ``default_rng(seed)``.
        Two-sided p = ``(1 + #{|perm_stat| >= |obs_stat|}) / (B + 1)``.
        """
        cfg = self._cfg
        n = deltas.size
        obs = abs(float(np.mean(deltas)))

        if 2**n <= cfg.bootstrap_B:
            # Exact enumeration over all 2^n sign vectors.
            signs = _sign_vectors(n)  # (2^n, n) of +/-1
            perm_stats = np.abs((signs * deltas).mean(axis=1))
            b = perm_stats.size
            count = int(np.sum(perm_stats >= obs))
            return (1 + count) / (b + 1)

        rng = np.random.default_rng(cfg.seed)
        b = cfg.bootstrap_B
        signs = rng.choice(np.array([-1.0, 1.0]), size=(b, n))
        perm_stats = np.abs((signs * deltas).mean(axis=1))
        count = int(np.sum(perm_stats >= obs))
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
        b_val = baseline[qid].get(metric, math.nan)
        v_val = variant_metrics[qid].get(metric, math.nan)
        if math.isnan(b_val) or math.isnan(v_val):
            continue
        deltas.append(v_val - b_val)
    return np.asarray(deltas, dtype=float)


def _sign_vectors(n: int) -> np.ndarray:
    """All ``2**n`` sign vectors in ``{-1, +1}`` as a ``(2**n, n)`` float array (permutation test)."""
    bits = ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    return 1.0 - 2.0 * bits  # 0 -> +1, 1 -> -1


def _holm(
    family: Sequence[tuple[int, float, tuple[str, str]]],
    alpha: float,
) -> dict[int, bool]:
    """Holm–Bonferroni step-down over the family; returns ``row_index -> significant`` (§8.3).

    ``family`` is ``(row_index, raw_p, (variant, metric))`` for every non-degenerate test. Sort by
    raw p ascending, tie-break by ``(variant, metric)``. Going in ascending order, reject while
    ``p_(j) <= alpha / (m - j + 1)``; at the FIRST failure, stop and RETAIN it and all larger.
    """
    m = len(family)
    ordered = sorted(family, key=lambda t: (t[1], t[2]))
    result: dict[int, bool] = {}
    failed = False
    for j, (idx, p, _key) in enumerate(ordered, start=1):
        if not failed and p <= alpha / (m - j + 1):
            result[idx] = True
        else:
            failed = True
            result[idx] = False
    return result
