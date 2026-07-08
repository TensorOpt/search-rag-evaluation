"""Comparator: bootstrap CI, permutation/Wilcoxon p-value, FDR decision (docs/experiment.md §8). Phase 3.

The :class:`Comparator` scores an explicit list of :class:`Contrast` (each a pair of system ids
``a``/``b`` with ``delta = value(a) − value(b)``), per metric, and produces one
:class:`ComparisonResult` per ``(contrast, metric)``. The baseline is NOT special here — it is just
another system, and "variant vs bm25" is one contrast among many (§8.1). It implements the single
coherent multiple-comparison regime of §8 — **False Discovery Rate (FDR)** control, Benjamini-Hochberg
by default:

- **Family-wide common subset (§8.1, Fix 6).** For each metric, ONE shared subset of queries is used:
  those whose metric value is finite (not NaN) for EVERY system REFERENCED BY THE CONTRASTS. Every
  referenced system — the baseline included — therefore has exactly ONE mean per metric, and every
  contrast on that metric is scored on the same query set. Any metric may be NaN for a query
  (``avg_relevance``/``ndcg@10``/``precision@10`` when ``n_scored == 0``; ``recall@k`` when ``R == 0``,
  §7). ``n_common`` (the subset size) rides on every row. The subset is the UNION of per-system NaN
  queries across referenced systems, so it trades statistical power for coherence, by design.
- **Degenerate short-circuits (§8.1 table), BEFORE any scipy/bootstrap call.** An *empty paired set*
  (``n_common == 0``) yields ``value_a``/``value_b``/``delta``/CI = ``None``, ``p_value = 1.0``,
  ``significant_raw = False``, ``note="empty_paired_set"``. *All-zero deltas* (>=1 paired query, every
  delta 0) yield ``delta = 0.0``, CI ``0.0/0.0``, ``p_value = 1.0``, ``significant_raw = False``,
  ``note="all_zero_delta"``. Degenerate rows are never in the FDR family: ``in_family = False`` and
  (per M3) ``p_value_adjusted = None`` / ``significant = None`` → empty CSV cells. They never trigger
  scipy or the RNG.
- **Effect-size CI (§8.2).** Percentile bootstrap over PAIRED QUERY INDICES with replacement,
  ``B = bootstrap_B`` resamples, using a FRESH ``numpy.random.default_rng(seed)`` per ``(contrast, metric)``
  so the CI is fully deterministic regardless of iteration order. The CI is reported as effect-size
  context ONLY — it is never a significance gate and MAY DISAGREE with ``significant`` /
  ``significant_raw`` (§8.3).
- **Raw p-value (§8.2).** Seeded sign-flip paired-permutation test with statistic ``mean(delta)``
  (default — the same estimand as the point estimate and CI), or two-sided Wilcoxon signed-rank
  (``zero_method``/``correction`` pinned), selected by ``StatsCfg.test``. ``significant_raw`` is the
  uncorrected per-test decision ``p_value <= alpha``, computed independently of the family.
- **FDR (§8.3), over the family only (Fix 7).** A row is in the family iff ``contrast.family AND
  metric ∈ cfg.fdr_metrics AND the row is non-degenerate``. Benjamini-Hochberg (default) or
  Benjamini-Yekutieli adjusted p-values (q-values) are computed over the family; family rows get
  ``significant = (p_value_adjusted <= alpha)``. Every non-family row (descriptive real test AND
  degenerate) carries ``p_value_adjusted = None`` / ``significant = None`` (M3 rule:
  ``in_family == false ⟺ both empty``); ``significant_raw`` stays populated on every real-test row.
  ``alpha`` is BOTH the raw per-test threshold AND the FDR target level ``q``.

This module imports ONLY the stdlib + numpy + scipy (no ``benchmark.*``): the comparator operates on
plain metric maps (``query_id -> {metric_name: value}``), exactly what ``Metrics.as_dict()`` produces;
the runner adapts ``Metrics`` -> maps before calling it (§11 import rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
from scipy import stats as scipy_stats  # type: ignore[import-untyped]

#: Canonical metric names, in the fixed iteration order used for every comparison, matching the §9
#: metrics-CSV column order (§7, §9, Fix 4).
CANONICAL_METRICS: tuple[str, ...] = (
    "avg_relevance",
    "ndcg@10",
    "recall@10",
    "recall@50",
    "recall@100",
    "precision@10",
)

#: Default FDR headline metrics (§8.3, Fix 7): ranking quality + coverage, nearly orthogonal, no
#: collinear inflation. Only these enter the BH/BY family (subject to each contrast's ``family`` flag).
DEFAULT_FDR_METRICS: tuple[str, ...] = ("ndcg@10", "recall@100")

#: Absolute tolerance for treating a computed float as zero (never use `== 0.0` on floats).
ZERO_ABS_TOL = 1e-6


@dataclass(frozen=True)
class Contrast:
    """One contrast between two systems (§8.1, Fix 3).

    ``a``/``b`` are system ids; ``delta = value(a) − value(b)`` ("how much better is ``a`` than
    ``b``", positive = ``a`` wins). ``family`` is True when the contrast is eligible for the FDR
    family (Fix 7); False makes it descriptive-only (delta + CI + raw p, no FDR adjustment).
    """

    a: str
    b: str
    family: bool


@dataclass(frozen=True)
class StatsCfg:
    """Statistics configuration (§8, §10 ``stats`` block).

    ``ci_level`` is the UNADJUSTED per-comparison bootstrap CI level (§8.2) — NOT a gate.
    ``alpha`` (0.05) is BOTH the raw per-test threshold AND the FDR target level ``q`` (§8.3).
    ``correction`` selects the FDR procedure: ``"bh"`` (Benjamini-Hochberg, default) or ``"by"``
    (Benjamini-Yekutieli, valid under arbitrary dependence). Any other value raises
    ``NotImplementedError``. ``test`` defaults to the mean-δ permutation test (§8.2) so the p-value,
    point estimate, and CI share one estimand; ``"wilcoxon"`` stays selectable. ``contrasts`` is the
    explicit list of system-pair contrasts to score (config synthesizes every-variant-vs-baseline
    when absent, §10); ``fdr_metrics`` is the set of metrics eligible for the FDR family (Fix 7).
    """

    bootstrap_B: int = 10000
    ci_level: float = 0.95
    alpha: float = 0.05
    correction: str = "bh"
    test: str = "permutation"
    wilcoxon_zero_method: str = "wilcox"
    wilcoxon_correction: bool = True
    seed: int = 1234
    contrasts: tuple[Contrast, ...] = ()
    fdr_metrics: tuple[str, ...] = DEFAULT_FDR_METRICS


@dataclass(frozen=True)
class ComparisonResult:
    """One ``(contrast, metric)`` comparison result (§8.1 table, §9 comparison CSV).

    ``value_a``/``value_b`` are the means of system ``a``/``b``'s metric over the family-wide common
    subset for that metric (§8.1, Fix 6), so ``delta == value_a - value_b``; all three (and the CI)
    are ``None`` for an empty paired set (serialized as empty cells, §9). ``p_value`` is the RAW,
    uncorrected test p-value (permutation or Wilcoxon; ``1.0`` for degenerate rows) and
    ``significant_raw`` is the uncorrected per-test decision (``p_value <= alpha``), independent of the
    family. ``in_family`` is FDR-family membership (Fix 7). ``p_value_adjusted`` (FDR q-value) and
    ``significant`` (FDR decision) are populated ONLY for family rows; for every non-family row
    (descriptive real test AND degenerate) both are ``None`` → empty CSV cells (M3 rule:
    ``in_family == false ⟺ both empty``). Either significance flag may disagree with the CI.
    ``n_common`` is the common-subset size (always present). ``note`` records a degenerate case:
    ``"empty_paired_set"`` | ``"all_zero_delta"`` | ``None``.
    """

    system_a: str
    system_b: str
    metric: str
    value_a: float | None
    value_b: float | None
    delta: float | None
    delta_ci_lo: float | None
    delta_ci_high: float | None
    p_value: float
    significant_raw: bool
    in_family: bool
    p_value_adjusted: float | None
    significant: bool | None
    n_common: int
    note: str | None = None


class Comparator:
    """Scores explicit contrasts and applies the §8 CI / p-value / FDR regime.

    ``compare`` returns rows in a deterministic order: contrasts in the order given, and within each
    contrast the canonical metrics in :data:`CANONICAL_METRICS` order.
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
        systems: Mapping[str, Mapping[str, Mapping[str, float]]],
        contrasts: Sequence[Contrast],
    ) -> list[ComparisonResult]:
        """Score every contrast, per canonical metric, with family-restricted FDR (§8).

        ``systems`` maps ``system_id -> query_id -> {metric_name: value}`` (exactly what
        ``Metrics.as_dict()`` produces per query, for EVERY pipeline incl. the baseline). ``contrasts``
        is the explicit list of system-pair contrasts (``delta = value(a) − value(b)``). For each
        metric a single family-wide common subset (finite in every contrast-referenced system, Fix 6)
        is computed once; each system then has one mean per metric. The FDR correction (BH/BY) is
        applied only over the family rows (``contrast.family AND metric ∈ fdr_metrics AND
        non-degenerate``, Fix 7).
        """
        cfg = self._cfg
        rows: list[ComparisonResult] = []
        # (index into rows, raw p-value) for each FDR-family row.
        family: list[tuple[int, float]] = []

        # ONE shared common subset per metric, over the systems referenced by the contrasts (S2), so
        # every referenced system keeps ONE value per metric across all contrasts.
        common_by_metric = {
            metric: _common_qids(systems, contrasts, metric) for metric in CANONICAL_METRICS
        }

        for contrast in contrasts:
            map_a = systems[contrast.a]
            map_b = systems[contrast.b]
            for metric in CANONICAL_METRICS:
                common_qids = common_by_metric[metric]
                n_common = len(common_qids)
                a_arr, b_arr = _paired_values(map_a, map_b, metric, common_qids)
                deltas = a_arr - b_arr

                if deltas.size == 0:
                    # Empty paired set (§8.1 table): no scipy/bootstrap call; never in the family.
                    rows.append(
                        ComparisonResult(
                            system_a=contrast.a,
                            system_b=contrast.b,
                            metric=metric,
                            value_a=None,
                            value_b=None,
                            delta=None,
                            delta_ci_lo=None,
                            delta_ci_high=None,
                            p_value=1.0,
                            significant_raw=False,
                            in_family=False,
                            p_value_adjusted=None,
                            significant=None,
                            n_common=n_common,
                            note="empty_paired_set",
                        )
                    )
                    continue

                if bool(np.all(np.isclose(deltas, 0.0, rtol=0.0, atol=ZERO_ABS_TOL))):
                    # All-zero deltas (§8.1 table): no scipy/bootstrap call; never in the family.
                    # np.isclose (not `== 0.0`) — never test float equality; tolerance ZERO_ABS_TOL.
                    # Every δ==0 => a==b per query, so the two means are equal.
                    equal_value = float(np.mean(a_arr))
                    rows.append(
                        ComparisonResult(
                            system_a=contrast.a,
                            system_b=contrast.b,
                            metric=metric,
                            value_a=equal_value,
                            value_b=equal_value,
                            delta=0.0,
                            delta_ci_lo=0.0,
                            delta_ci_high=0.0,
                            p_value=1.0,
                            significant_raw=False,
                            in_family=False,
                            p_value_adjusted=None,
                            significant=None,
                            n_common=n_common,
                            note="all_zero_delta",
                        )
                    )
                    continue

                # Real test: point estimate, bootstrap CI, raw p-value (§8.2).
                value_a = float(np.mean(a_arr))
                value_b = float(np.mean(b_arr))
                delta = float(np.mean(deltas))
                ci_lo, ci_high = self._bootstrap_ci(deltas)
                p_value = self._p_value(deltas)
                # Family membership (Fix 7): eligible contrast AND headline metric (non-degenerate here).
                in_family = contrast.family and metric in cfg.fdr_metrics

                idx = len(rows)
                rows.append(
                    ComparisonResult(
                        system_a=contrast.a,
                        system_b=contrast.b,
                        metric=metric,
                        value_a=value_a,
                        value_b=value_b,
                        delta=delta,
                        delta_ci_lo=ci_lo,
                        delta_ci_high=ci_high,
                        p_value=p_value,
                        # Raw per-test decision, independent of the family (§8.3).
                        significant_raw=p_value <= cfg.alpha,
                        in_family=in_family,
                        p_value_adjusted=None,  # family rows set below; non-family stay None (M3)
                        significant=None,  # family rows set below; non-family stay None (M3)
                        n_common=n_common,
                        note=None,
                    )
                )
                if in_family:
                    family.append((idx, p_value))

        # FDR (BH/BY) over the family rows only (§8.3, Fix 7). Non-family rows keep None (M3).
        adjusted_by_idx = _fdr_adjust([p for _, p in family], cfg.correction)
        for (idx, _p), q in zip(family, adjusted_by_idx):
            rows[idx] = replace(rows[idx], p_value_adjusted=q, significant=q <= cfg.alpha)
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


def _common_qids(
    systems: Mapping[str, Mapping[str, Mapping[str, float]]],
    contrasts: Sequence[Contrast],
    metric: str,
) -> list[str]:
    """The family-wide common subset for ``metric`` (§8.1, Fix 6, S2).

    The sorted query ids for which ``metric`` is finite (not NaN) in EVERY system referenced by the
    contrasts. Scoped to contrast-referenced systems only (a system in no contrast cannot shrink the
    subset), so every referenced system — the baseline included — keeps ONE value per metric across
    all contrasts. A qid missing from a referenced system reads as NaN and is excluded. Sorted so the
    arrays (and the bootstrap seeded off them) are deterministic.
    """
    used = {s for c in contrasts for s in (c.a, c.b)}
    all_qids: set[str] = set()
    for s in used:
        all_qids.update(systems[s])
    return [
        qid
        for qid in sorted(all_qids)
        if all(not math.isnan(systems[s].get(qid, {}).get(metric, math.nan)) for s in used)
    ]


def _paired_values(
    map_a: Mapping[str, Mapping[str, float]],
    map_b: Mapping[str, Mapping[str, float]],
    metric: str,
    common_qids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned ``a``/``b`` ``metric`` arrays over the precomputed common subset (§8.1, Fix 6).

    Selects ``metric`` for each qid in ``common_qids`` from both systems' maps. The subset is already
    finite-in-every-referenced-system (:func:`_common_qids`), so ``a`` and ``b`` are both finite on it
    — one shared mask, no per-pair recompute. ``common_qids`` is sorted, so the arrays are
    deterministic.
    """
    a_vals = [map_a[qid][metric] for qid in common_qids]
    b_vals = [map_b[qid][metric] for qid in common_qids]
    return np.asarray(a_vals, dtype=float), np.asarray(b_vals, dtype=float)


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
