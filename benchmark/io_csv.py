"""CSV/JSON artifact writers with fixed schemas (docs/experiment.md §9). Phase 7.

Four writers, each into an output dir (default ``results``), naming files with the run's single
UTC timestamp ``YYYYMMDDTHHMMSSZ`` (§9). The three CSVs are ONE file per run (all pipelines /
comparisons), each carrying a leading identity column:

- :func:`write_results_csv`  -> ``result_{ts}.csv``   : ``variant,query_id,product_id,score,position``
- :func:`write_metrics_csv`  -> ``metrics_{ts}.csv``  : ``variant,query_id,avg_relevance,ndcg@10,recall@10,precision@10,n_scored,n_missing``
- :func:`write_comparison_csv` -> ``comparison_{ts}.csv`` : the 12-column §9 header
- :func:`write_run_config`   -> ``run_config_{ts}.json`` : the fully-resolved config (§9.1)

Serialization rules (fixed so golden files are stable, §9/CLAUDE.md):

- All CSVs UTF-8, comma-separated, header present, via the stdlib :mod:`csv` module.
- Floats are formatted with ``repr`` (shortest round-trip) so a golden file is byte-stable.
- ``result``: one row per ``ScoredDoc`` in ``RankedResult.docs``, in order, prefixed with the
  pipeline id (``variant``); ``position`` is the 1-based index (derived here, §3.1, never stored on
  ``ScoredDoc``); at most ``top_k`` rows/query. Variants are written in the mapping's order
  (baseline first).
- ``metrics``: one row per (variant, query), prefixed with the pipeline id. Each of the FOUR metric
  cells is written EMPTY (two adjacent commas, no quoting) when its in-memory ``Metrics`` value is
  ``math.nan`` (§7): avg/ndcg/precision empty when ``n_scored == 0``, recall empty when ``R == 0``.
  ``n_scored``/``n_missing`` are non-negative ints, ALWAYS present.
- ``comparison``: one row per (variant, canonical metric), prefixed with the shared ``baseline`` id.
  ``baseline_value``/``variant_value``/``delta``/CI are numeric, or EMPTY for an empty paired set
  (``None``, §8.1). ``significant_raw``/``significant`` are lowercase ``true``/``false``;
  ``p_value``/``p_value_adjusted`` numeric.
- :func:`write_run_config` serializes the fully-resolved config via ``dataclasses.asdict`` +
  ``json.dumps`` (deterministic, ``sort_keys=True``, with ``default=str`` catching any non-JSON
  straggler) so it round-trips (§9.1).

This module imports only ``benchmark.common.models``/``benchmark.evaluation.metrics``/
``benchmark.evaluation.stats``/``benchmark.config`` + stdlib — never an adapter (§11); ``config`` does
NOT import this module, so there is no cycle.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark.common.models import RankedResult
from benchmark.config import ResolvedConfig
from benchmark.evaluation.metrics import Metrics
from benchmark.evaluation.stats import ComparisonResult

#: Default artifact output directory (§9). The runner may override.
DEFAULT_OUTPUT_DIR = "results"

#: Canonical metric order for the metrics CSV columns (§9), matching ``Metrics.as_dict()`` keys.
_METRIC_COLUMNS: tuple[str, ...] = (
    "avg_relevance",
    "ndcg@10",
    "recall@10",
    "precision@10",
)

_RESULT_HEADER: tuple[str, ...] = ("variant", "query_id", "product_id", "score", "position")
_METRICS_HEADER: tuple[str, ...] = (
    "variant",
    "query_id",
    *_METRIC_COLUMNS,
    "n_scored",
    "n_missing",
)
_COMPARISON_HEADER: tuple[str, ...] = (
    "baseline",
    "variant",
    "metric",
    "baseline_value",
    "variant_value",
    "delta",
    "delta_ci_lo",
    "delta_ci_high",
    "p_value",
    "significant_raw",
    "p_value_adjusted",
    "significant",
)


def _float_cell(value: float | None) -> str:
    """Serialize a float cell: EMPTY for ``None`` or ``NaN`` (§7/§8.1), else shortest round-trip.

    ``None`` (an empty-paired-set delta/CI) and ``math.nan`` (a NaN metric) both map to the empty
    field the readers treat as "excluded" (§9). ``repr`` gives the shortest round-tripping string so
    golden files stay byte-stable.
    """
    if value is None or math.isnan(value):
        return ""
    return repr(value)


def _bool_cell(value: bool) -> str:
    """Serialize a boolean flag as lowercase ``true``/``false`` (§9)."""
    return "true" if value else "false"


def _open_csv_writer(path: Path) -> Any:
    """Open ``path`` for UTF-8 CSV writing with no extra quoting/line-terminator drift."""
    handle = path.open("w", encoding="utf-8", newline="")
    return handle, csv.writer(handle, lineterminator="\n")


def write_results_csv(
    results_by_variant: Mapping[str, Sequence[RankedResult]],
    timestamp: str,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write ``result_{ts}.csv`` — one row per ranked doc across ALL pipelines, ``position`` 1-based (§9).

    ``results_by_variant`` maps pipeline id -> its ranked results (baseline first, then variants in
    config order). Each row is prefixed with the variant id. ``position`` is derived here as the
    1-based index into each ``RankedResult.docs`` (§3.1); the number of rows per query is whatever the
    pipeline returned (already ``<= top_k``, §8.0).
    """
    path = _artifact_path(output_dir, f"result_{timestamp}.csv")
    handle, writer = _open_csv_writer(path)
    with handle:
        writer.writerow(_RESULT_HEADER)
        for variant_id, results in results_by_variant.items():
            for result in results:
                for position, doc in enumerate(result.docs, start=1):
                    writer.writerow(
                        [variant_id, result.query_id, doc.doc_id, repr(doc.score), position]
                    )
    return path


def write_metrics_csv(
    metrics_by_variant: Mapping[str, Mapping[str, Metrics]],
    timestamp: str,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write ``metrics_{ts}.csv`` — one row per (variant, query), NaN metric cells EMPTY (§7/§9).

    ``metrics_by_variant`` maps pipeline id -> {query_id: Metrics} (baseline first, then variants).
    Each row is prefixed with the variant id. Each of the four metric cells is empty when its
    ``Metrics`` value is ``math.nan`` (avg/ndcg/precision when ``n_scored == 0``; recall when
    ``R == 0``). ``n_scored``/``n_missing`` are always written as ints.
    """
    path = _artifact_path(output_dir, f"metrics_{timestamp}.csv")
    handle, writer = _open_csv_writer(path)
    with handle:
        writer.writerow(_METRICS_HEADER)
        for variant_id, metrics in metrics_by_variant.items():
            for query_id, m in metrics.items():
                metric_values = m.as_dict()
                row: list[Any] = [variant_id, query_id]
                row.extend(_float_cell(metric_values[name]) for name in _METRIC_COLUMNS)
                row.extend([m.n_scored, m.n_missing])
                writer.writerow(row)
    return path


def write_comparison_csv(
    baseline_id: str,
    rows: Sequence[ComparisonResult],
    timestamp: str,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write ``comparison_{ts}.csv`` — one row per (variant, metric) across ALL variants (§8.1/§9).

    ``baseline_id`` is the baseline pipeline id, emitted as the leading ``baseline`` column on every
    row (the baseline is never compared to itself, so there is no baseline-vs-baseline row).
    ``baseline_value``/``variant_value``/``delta``/CI cells are empty for an empty paired set
    (``None``) and numeric otherwise; ``significant_raw``/``significant`` are lowercase
    ``true``/``false`` (§9).
    """
    path = _artifact_path(output_dir, f"comparison_{timestamp}.csv")
    handle, writer = _open_csv_writer(path)
    with handle:
        writer.writerow(_COMPARISON_HEADER)
        for row in rows:
            writer.writerow(
                [
                    baseline_id,
                    row.variant,
                    row.metric,
                    _float_cell(row.baseline_value),
                    _float_cell(row.variant_value),
                    _float_cell(row.delta),
                    _float_cell(row.delta_ci_lo),
                    _float_cell(row.delta_ci_high),
                    _float_cell(row.p_value),
                    _bool_cell(row.significant_raw),
                    _float_cell(row.p_value_adjusted),
                    _bool_cell(row.significant),
                ]
            )
    return path


def write_run_config(
    cfg: ResolvedConfig,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write ``run_config_{ts}.json`` — the fully-resolved config for reproducibility (§9.1).

    Serialized deterministically (``sort_keys=True``) via ``dataclasses.asdict``, with ``default=str``
    catching any non-JSON straggler so the JSON always round-trips. The resolved services registry
    (embedder/reranker/searcher configs), the pipelines (baseline + variants),
    the stats block (bootstrap_B, ci_level, alpha as both the raw threshold and the FDR level q,
    correction, test + wilcoxon zero/tie params, seed), cutoff, top_k, timestamp, and seed are all
    captured (§9.1).
    """
    path = _artifact_path(output_dir, f"run_config_{cfg.timestamp}.json")
    payload = dataclasses.asdict(cfg)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _artifact_path(output_dir: str | Path, filename: str) -> Path:
    """Resolve ``output_dir/filename``, creating the output directory if needed."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename
