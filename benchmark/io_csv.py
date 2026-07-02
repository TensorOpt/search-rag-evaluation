"""CSV/JSON artifact writers with fixed schemas (docs/experiment.md §9). Phase 7.

Four writers, each into an output dir (default ``results``), naming files with the run's single
UTC timestamp ``YYYYMMDDTHHMMSSZ`` (§9):

- :func:`write_result_csv`  -> ``result_{variant}_{ts}.csv``     : ``query_id,product_id,score,position``
- :func:`write_metrics_csv` -> ``metrics_{variant}_{ts}.csv``    : ``query_id,avg_relevance,ndcg@10,recall@10,precision@10,n_scored,n_missing``
- :func:`write_comparison_csv` -> ``comparison_{baseline}_{variant}_{ts}.csv`` : the 9-column §9 header
- :func:`write_run_config`  -> ``run_config_{ts}.json``          : the fully-resolved config (§9.1)

Serialization rules (fixed so golden files are stable, §9/CLAUDE.md):

- All CSVs UTF-8, comma-separated, header present, via the stdlib :mod:`csv` module.
- Floats are formatted with ``repr`` (shortest round-trip) so a golden file is byte-stable.
- ``result``: one row per ``ScoredDoc`` in ``RankedResult.docs``, in order; ``position`` is the
  1-based index (derived here, §3.1, never stored on ``ScoredDoc``); at most ``top_k`` rows/query.
- ``metrics``: one row per query. Each of the FOUR metric cells is written EMPTY (two adjacent
  commas, no quoting) when its in-memory ``Metrics`` value is ``math.nan`` (§7): avg/ndcg/precision
  empty when ``n_scored == 0``, recall empty when ``R == 0``. ``n_scored``/``n_missing`` are
  non-negative ints, ALWAYS present.
- ``comparison``: one row per canonical metric. ``significant_raw``/``significant`` are lowercase
  ``true``/``false``; ``p_value``/``p_value_adjusted`` numeric. A ``None`` delta/CI cell (empty
  paired set) is written EMPTY (§8.1).
- :func:`write_run_config` serializes the fully-resolved config via ``dataclasses.asdict`` +
  ``json.dumps`` (deterministic; ``EmbeddingType`` is a ``StrEnum`` and serializes as its string,
  with ``default=str`` catching any straggler) so it round-trips (§9.1).

This module imports only ``benchmark.models``/``benchmark.metrics``/``benchmark.stats``/
``benchmark.config`` + stdlib — never an adapter (§11); ``config`` does NOT import this module, so
there is no cycle.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark.config import PipelineCfg, ResolvedConfig
from benchmark.metrics import Metrics
from benchmark.models import RankedResult
from benchmark.stats import ComparisonResult

#: Default artifact output directory (§9). The runner may override.
DEFAULT_OUTPUT_DIR = "results"

#: Canonical metric order for the metrics CSV columns (§9), matching ``Metrics.as_dict()`` keys.
_METRIC_COLUMNS: tuple[str, ...] = (
    "avg_relevance",
    "ndcg@10",
    "recall@10",
    "precision@10",
)

_RESULT_HEADER: tuple[str, ...] = ("query_id", "product_id", "score", "position")
_METRICS_HEADER: tuple[str, ...] = ("query_id", *_METRIC_COLUMNS, "n_scored", "n_missing")
_COMPARISON_HEADER: tuple[str, ...] = (
    "variant",
    "metric",
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


def write_result_csv(
    pcfg: PipelineCfg,
    results: Sequence[RankedResult],
    timestamp: str,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write ``result_{variant}_{ts}.csv`` — one row per ranked doc, ``position`` 1-based (§9).

    ``position`` is derived here as the 1-based index into each ``RankedResult.docs`` (§3.1). The
    number of rows per query is whatever the pipeline returned (already ``<= top_k``, §8.0).
    """
    path = _artifact_path(output_dir, f"result_{pcfg.id}_{timestamp}.csv")
    handle, writer = _open_csv_writer(path)
    with handle:
        writer.writerow(_RESULT_HEADER)
        for result in results:
            for position, doc in enumerate(result.docs, start=1):
                writer.writerow(
                    [result.query_id, doc.doc_id, repr(doc.score), position]
                )
    return path


def write_metrics_csv(
    pcfg: PipelineCfg,
    metrics: Mapping[str, Metrics],
    timestamp: str,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write ``metrics_{variant}_{ts}.csv`` — one row per query, NaN metric cells EMPTY (§7/§9).

    Each of the four metric cells is empty when its ``Metrics`` value is ``math.nan``
    (avg/ndcg/precision when ``n_scored == 0``; recall when ``R == 0``). ``n_scored``/``n_missing``
    are always written as ints.
    """
    path = _artifact_path(output_dir, f"metrics_{pcfg.id}_{timestamp}.csv")
    handle, writer = _open_csv_writer(path)
    with handle:
        writer.writerow(_METRICS_HEADER)
        for query_id, m in metrics.items():
            metric_values = m.as_dict()
            row: list[Any] = [query_id]
            row.extend(_float_cell(metric_values[name]) for name in _METRIC_COLUMNS)
            row.extend([m.n_scored, m.n_missing])
            writer.writerow(row)
    return path


def write_comparison_csv(
    baseline_id: str,
    variant_id: str,
    rows: Sequence[ComparisonResult],
    timestamp: str,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write ``comparison_{baseline}_{variant}_{ts}.csv`` — one row per metric (§8.1/§9).

    ``delta``/CI cells are empty for an empty paired set (``None``) and numeric otherwise;
    ``significant_raw``/``significant`` are lowercase ``true``/``false`` (§9).
    """
    path = _artifact_path(
        output_dir, f"comparison_{baseline_id}_{variant_id}_{timestamp}.csv"
    )
    handle, writer = _open_csv_writer(path)
    with handle:
        writer.writerow(_COMPARISON_HEADER)
        for row in rows:
            writer.writerow(
                [
                    row.variant,
                    row.metric,
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

    Serialized deterministically (``sort_keys=True``) via ``dataclasses.asdict``; ``EmbeddingType``
    is a ``StrEnum`` (serializes as its string value), and ``default=str`` catches any straggler so
    the JSON always round-trips. The resolved services registry, the pipelines (baseline + variants),
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
