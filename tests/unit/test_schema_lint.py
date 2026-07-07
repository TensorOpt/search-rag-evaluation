"""Schema-lint test — automates §9 artifact schemas (§1.4(1) Correctness).

Runs :class:`benchmark.runner.ExperimentRunner` with the in-memory fakes from ``test_runner`` into a
tmp dir, then byte-exactly asserts every produced artifact's header/field order against §9:

- ``result_{ts}.csv``     : ``variant,query_id,product_id,score,position``
- ``metrics_{ts}.csv``    : ``variant,query_id,avg_relevance,ndcg@10,recall@10,precision@10,n_results,n_scored,n_missing``
- ``comparison_{ts}.csv`` : the 12-column §9 comparison header
- ``run_config_*.json``: parses + carries the §9.1 keys (services/pipelines/stats/cutoff/top_k/seed…)

The linter reads the FIRST line of each CSV as the header (byte-exact) so a rename/reorder in any
writer fails here, independently of ``io_csv``'s own header constants.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.test_runner import (
    FuserCfg,
    PipelineCfg,
    _config,
    patch_runner_factories,
)

# The §9 headers, spelled out here (NOT imported from io_csv) so a drift in io_csv is caught.
_RESULT_HEADER = "variant,query_id,product_id,score,position"
_METRICS_HEADER = "variant,query_id,avg_relevance,ndcg@10,recall@10,precision@10,n_results,n_scored,n_missing"
_COMPARISON_HEADER = (
    "baseline,variant,metric,baseline_value,variant_value,delta,delta_ci_lo,delta_ci_high,"
    "p_value,significant_raw,p_value_adjusted,significant"
)
#: §9.1 run_config keys (fully-resolved config; asdict of ResolvedConfig).
_RUN_CONFIG_KEYS = {
    "dataset",
    "indexer",
    "services",
    "baseline",
    "variants",
    "stats",
    "cutoff",
    "top_k",
    "baseline_id",
    "timestamp",
    "seed",
}


def _header(path: Path) -> str:
    """The first line of a CSV, byte-exact (no trailing newline)."""
    return path.read_text(encoding="utf-8").splitlines()[0]


def test_artifact_schemas_match_section_9(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from benchmark.runner import ExperimentRunner

    patch_runner_factories(monkeypatch)
    ts = "20260702T090000Z"
    variants = [
        PipelineCfg("semantic_e5", ("semantic_e5",), None, None, None),
        PipelineCfg(
            "hybrid_e5",
            ("bm25", "semantic_e5"),
            FuserCfg("rrf", rank_constant=60, window=100),
            None,
            None,
        ),
        PipelineCfg("bm25_rerank", ("bm25",), None, "rr", 100),
    ]
    cfg = _config(variants=variants, timestamp=ts)

    ExperimentRunner().run(cfg, output_dir=str(tmp_path))

    # Exactly ONE file per artifact type (all pipelines / comparisons in a single file, §9).
    result_files = sorted(tmp_path.glob(f"result_{ts}.csv"))
    metrics_files = sorted(tmp_path.glob(f"metrics_{ts}.csv"))
    comparison_files = sorted(tmp_path.glob(f"comparison_{ts}.csv"))
    assert len(result_files) == len(metrics_files) == len(comparison_files) == 1

    assert _header(result_files[0]) == _RESULT_HEADER
    assert _header(metrics_files[0]) == _METRICS_HEADER
    assert _header(comparison_files[0]) == _COMPARISON_HEADER

    run_config = tmp_path / f"run_config_{ts}.json"
    payload = json.loads(run_config.read_text(encoding="utf-8"))
    assert set(payload) == _RUN_CONFIG_KEYS
    # §9.1 stats sub-block records both the raw threshold and the FDR knobs (correction/test/seed).
    for key in ("bootstrap_B", "ci_level", "alpha", "correction", "test", "seed"):
        assert key in payload["stats"], key
