"""eval:sweep — one-shot diagnostic parameter sweeps over the SINGLE runner (P1-2/P2-1/P3-1).

    eval:sweep --axis {rerank_window | rrf_k | bm25_k1_b} --config config.yaml [--out results/sweep]

ONE flag-driven script (not three bespoke ones): each axis rewrites the relevant RESOLVED config value
per grid cell and re-runs the pipelines through the EXISTING
:class:`~benchmark.runner.ExperimentRunner` scoring path + :class:`~benchmark.evaluation.metrics.Evaluator`
+ :class:`~benchmark.evaluation.stats.Comparator` — NO forked metric/stats code. Output is a tidy,
**diagnostic, NON-frozen** ``sweep_{axis}_{ts}.csv`` (``axis_value, system, metric, value, ci_lo,
ci_high, n_common``); it never touches the frozen ``result``/``metrics``/``comparison`` artifacts and is
NOT part of ``eval:run``. Sweeps re-run retrieval/eval, so a real sweep needs a live index + provider
keys (the user runs those); the axis handlers are unit-testable OFFLINE via the in-memory fake factories
(``tests/unit/test_runner.py::patch_runner_factories``).

- **rerank_window (P2-1):** ``rerank_window_size ∈ {10,25,50,100}`` at ``top_k=100`` for the rerank
  variants; per window, ``ndcg@10`` + ``recall@50`` with the paired bootstrap CI of Δ vs the
  corresponding UNRERANKED base (from the Comparator). No reindex — only rerank re-runs.
- **rrf_k (P3-1):** ``rank_constant ∈ {20,60,100}`` for each pure hybrid (fuser, no reranker);
  ``ndcg@10``/``precision@10``/``recall@100`` per k on the finite subset. No reindex — only fusion re-runs.
- **bm25_k1_b (P1-2):** ``k1 ∈ {0.9,1.2,1.5,2.0} × b ∈ {0.3,0.5,0.75,0.9}`` (16 cells); ``ndcg@10`` per
  cell on the finite subset. BM25 params are INDEX-TIME, so this REINDEXES a scratch index per cell
  (reuse ``ExperimentRunner.build_index``), runs the baseline only, then tears the scratch index down.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import benchmark.config as config
from benchmark.common.logging_setup import get_logger, setup_logging
from benchmark.config import ConfigError, ResolvedConfig, load_config
from benchmark.evaluation.metrics import Metrics
from benchmark.evaluation.stats import Comparator, Contrast
from benchmark.runner import ExperimentRunner, IndexNotReadyError

log = get_logger(__name__)

#: Sweep axes (exhaustive — an unknown axis raises a clear :class:`ConfigError`).
AXES = ("rerank_window", "rrf_k", "bm25_k1_b")

#: P2-1 rerank-window grid + the two reported metrics; top_k stays fixed at the config's value (100).
_RERANK_WINDOWS = (10, 25, 50, 100)
_RERANK_METRICS = ("ndcg@10", "recall@50")
#: P3-1 RRF rank_constant grid + the three reported metrics.
_RRF_CONSTANTS = (20, 60, 100)
_RRF_METRICS = ("ndcg@10", "precision@10", "recall@100")
#: P1-2 BM25 k1×b grid + the reported metric.
_BM25_K1 = (0.9, 1.2, 1.5, 2.0)
_BM25_B = (0.3, 0.5, 0.75, 0.9)
_BM25_METRIC = "ndcg@10"

#: Default output directory for sweep CSVs (diagnostic; NOT the frozen ``results`` artifacts).
DEFAULT_SWEEP_DIR = "results/sweep"

_SWEEP_HEADER = ("axis_value", "system", "metric", "value", "ci_lo", "ci_high", "n_common")


class SweepRow(NamedTuple):
    """One (cell, system, metric) diagnostic result.

    ``value`` is the metric mean on the cell's finite subset. ``ci_lo``/``ci_high`` are the 95% paired
    bootstrap CI of Δ vs the unreranked base (rerank_window only; ``None`` for the single-system axes).
    ``n_common`` is the subset size the value/CI were computed over.
    """

    axis_value: str
    system: str
    metric: str
    value: float | None
    ci_lo: float | None
    ci_high: float | None
    n_common: int


def _systems(per_query: dict[str, dict[str, Metrics]]) -> dict[str, dict[str, Any]]:
    """``per_query`` -> the ``{system -> query -> {metric: value}}`` map the Comparator consumes."""
    return {
        vid: {qid: m.as_dict() for qid, m in metrics.items()}
        for vid, metrics in per_query.items()
    }


def _mean_on_finite(metrics: dict[str, Metrics], metric_key: str) -> tuple[float | None, int]:
    """Mean of a metric over the queries where it is finite, + that subset's size (§8.1 semantics).

    Aggregation over the Evaluator's per-query outputs (the same finite-subset rule the Comparator
    uses for a single system) — NOT a re-implementation of any metric (§7 metric code stays in the
    Evaluator). ``None`` when no query has a finite value.
    """
    values = [m.as_dict()[metric_key] for m in metrics.values()]
    finite = [v for v in values if not math.isnan(v)]
    return (statistics.fmean(finite) if finite else None), len(finite)


def _sweep_rerank_window(
    runner: ExperimentRunner, cfg: ResolvedConfig, cache: Any
) -> list[SweepRow]:
    """P2-1: rescore each rerank variant at every window vs its unreranked base (paired CIs)."""
    ctx = runner._build_search_context(cfg, cache)
    rows: list[SweepRow] = []
    for rerank_pcfg in [p for p in cfg.pipelines() if p.reranker is not None]:
        # The corresponding unreranked base: same retrieval graph, reranker dropped. Window-independent
        # (retrieves top_k, no rerank), so score it ONCE and reuse across the window grid.
        base = replace(
            rerank_pcfg, id=f"{rerank_pcfg.id}__base", reranker=None, rerank_window_size=None
        )
        _, base_per_query = runner._score_pipelines(cfg, ctx, [base])
        base_systems = _systems(base_per_query)
        for window in _RERANK_WINDOWS:
            variant = replace(rerank_pcfg, rerank_window_size=window)
            _, variant_per_query = runner._score_pipelines(cfg, ctx, [variant])
            systems = {**base_systems, **_systems(variant_per_query)}
            # Comparator gives value_a (reranked mean on the common subset) + the paired Δ CI vs base.
            contrast = Contrast(a=variant.id, b=base.id, family=False)
            by_metric = {r.metric: r for r in Comparator(cfg.stats).compare(systems, [contrast])}
            for metric in _RERANK_METRICS:
                comp = by_metric[metric]
                rows.append(
                    SweepRow(
                        axis_value=str(window),
                        system=rerank_pcfg.id,
                        metric=metric,
                        value=comp.value_a,
                        ci_lo=comp.delta_ci_lo,
                        ci_high=comp.delta_ci_high,
                        n_common=comp.n_common,
                    )
                )
    return rows


def _sweep_rrf_k(runner: ExperimentRunner, cfg: ResolvedConfig, cache: Any) -> list[SweepRow]:
    """P3-1: rescore each pure hybrid (fuser, no reranker) at every RRF rank_constant."""
    ctx = runner._build_search_context(cfg, cache)
    rows: list[SweepRow] = []
    hybrids = [p for p in cfg.pipelines() if p.fuser is not None and p.reranker is None]
    for hybrid in hybrids:
        assert hybrid.fuser is not None  # filtered above
        for rank_constant in _RRF_CONSTANTS:
            variant = replace(hybrid, fuser=replace(hybrid.fuser, rank_constant=rank_constant))
            _, per_query = runner._score_pipelines(cfg, ctx, [variant])
            metrics = per_query[variant.id]
            for metric in _RRF_METRICS:
                mean, n_common = _mean_on_finite(metrics, metric)
                rows.append(
                    SweepRow(str(rank_constant), hybrid.id, metric, mean, None, None, n_common)
                )
    return rows


def _with_bm25(cfg: ResolvedConfig, k1: float, b: float) -> ResolvedConfig:
    """A config whose indexer targets a per-cell SCRATCH index carrying these BM25 ``k1``/``b``."""
    indexer = dict(cfg.indexer)
    base_index = str(indexer.get("index", "index"))
    scratch = f"{base_index}__sweep_k1_{k1}_b_{b}".replace(".", "_")
    settings = dict(indexer.get("settings", {}))  # type: ignore[arg-type]
    settings["bm25"] = {"k1": k1, "b": b}
    indexer["index"] = scratch
    indexer["settings"] = settings
    return replace(cfg, indexer=indexer)


def _teardown_index(writer: Any) -> None:
    """Best-effort delete of a scratch index after a bm25_k1_b cell (real ES only).

    ponytail: guarded client reach — the ES writer exposes ``client.indices.delete``; the in-memory
    fake (offline tests) has no client, so this no-ops. Real ES gets the scratch index removed so the
    16-cell sweep doesn't leave garbage indices behind.
    """
    client = getattr(writer, "client", None)
    indices = getattr(client, "indices", None)
    if indices is None:
        return
    indices.delete(index=writer.index, ignore_unavailable=True)
    log.info("sweep: deleted scratch index %r", writer.index)


def _sweep_bm25_k1_b(runner: ExperimentRunner, cfg: ResolvedConfig, cache: Any) -> list[SweepRow]:
    """P1-2: reindex a scratch index per (k1, b) cell, score the baseline, report ndcg@10, tear down."""
    rows: list[SweepRow] = []
    for k1 in _BM25_K1:
        for b in _BM25_B:
            cfg_cell = _with_bm25(cfg, k1, b)
            runner.build_index(cfg_cell)  # BM25 params bake in at index time -> reindex the scratch
            ctx = runner._build_search_context(cfg_cell, cache)
            try:
                _, per_query = runner._score_pipelines(cfg_cell, ctx, [cfg_cell.baseline])
                mean, n_common = _mean_on_finite(per_query[cfg_cell.baseline.id], _BM25_METRIC)
                # ';' (not ',') so the compound cell coordinate needs no CSV quoting.
                rows.append(
                    SweepRow(
                        f"k1={k1};b={b}", cfg.baseline.id, _BM25_METRIC, mean, None, None, n_common
                    )
                )
            finally:
                _teardown_index(ctx.writer)
    return rows


def _cell(value: float | None) -> str:
    """A numeric CSV cell: EMPTY for ``None``/NaN, else shortest round-trip (matches io_csv)."""
    if value is None or math.isnan(value):
        return ""
    return repr(value)


def _write_sweep_csv(
    axis: str, rows: Sequence[SweepRow], timestamp: str, output_dir: str | Path
) -> Path:
    """Write the tidy diagnostic ``sweep_{axis}_{ts}.csv`` (NON-frozen, no golden)."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"sweep_{axis}_{timestamp}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_SWEEP_HEADER)
        for row in rows:
            writer.writerow(
                [
                    row.axis_value,
                    row.system,
                    row.metric,
                    _cell(row.value),
                    _cell(row.ci_lo),
                    _cell(row.ci_high),
                    row.n_common,
                ]
            )
    return path


def run_sweep(
    cfg: ResolvedConfig,
    axis: str,
    *,
    output_dir: str | Path = DEFAULT_SWEEP_DIR,
    cache: Any = None,
) -> Path:
    """Run one sweep ``axis`` over ``cfg`` and write the diagnostic CSV; return its path.

    Exhaustive on ``axis`` — an unknown value raises :class:`ConfigError` (never a silent default).
    Opens the config's cache when the caller passes none (so repeated retrieval across cells is
    memoized on a real run); offline tests pass ``cache=None`` with caching disabled.
    """
    runner = ExperimentRunner()
    own_cache = cache is None
    live_cache = config.open_cache(cfg.cache) if own_cache else cache
    try:
        if axis == "rerank_window":
            rows = _sweep_rerank_window(runner, cfg, live_cache)
        elif axis == "rrf_k":
            rows = _sweep_rrf_k(runner, cfg, live_cache)
        elif axis == "bm25_k1_b":
            rows = _sweep_bm25_k1_b(runner, cfg, live_cache)
        else:
            raise ConfigError(f"unknown sweep axis {axis!r}; expected one of {list(AXES)}")
        path = _write_sweep_csv(axis, rows, cfg.timestamp, output_dir)
        log.info("sweep %r complete: %d cell-rows written to %s", axis, len(rows), path)
        return path
    finally:
        if own_cache and live_cache is not None:
            live_cache.close()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval:sweep", description="Run a one-shot diagnostic parameter sweep."
    )
    parser.add_argument("--axis", required=True, choices=AXES, help="the parameter axis to sweep")
    parser.add_argument("--config", default="config.yaml", help="path to the config file")
    parser.add_argument("--out", default=DEFAULT_SWEEP_DIR, help="directory for the sweep CSV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(timestamp=cfg.timestamp)
    try:
        run_sweep(cfg, args.axis, output_dir=args.out)
    except IndexNotReadyError as exc:
        # rerank_window / rrf_k rescore over the existing index; bm25_k1_b builds its own scratch one.
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
