"""Schema-lint test — automates §9 artifact schemas (§1.4(1) Correctness).

Runs :class:`benchmark.runner.ExperimentRunner` with the in-memory fakes from ``test_runner`` into a
tmp dir, then byte-exactly asserts every produced artifact's header/field order against §9:

- ``result_{ts}.csv``     : ``variant,query_id,product_id,score,position``
- ``metrics_{ts}.csv``    : ``variant,query_id,avg_relevance,ndcg@10,recall@10,recall@50,recall@100,precision@10,n_results,n_scored,n_missing,n_relevant``
- ``comparison_{ts}.csv`` : the 14-column §9 comparison header
- ``run_config_*.json``: parses + carries the §9.1 keys (services/pipelines/stats/cutoff/top_k/seed/diagnostics…)

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
_METRICS_HEADER = (
    "variant,query_id,avg_relevance,ndcg@10,recall@10,recall@50,recall@100,precision@10,"
    "n_results,n_scored,n_missing,n_relevant"
)
_COMPARISON_HEADER = (
    "system_a,system_b,metric,value_a,value_b,delta,delta_ci_lo,delta_ci_high,"
    "p_value,significant_raw,in_family,p_value_adjusted,significant,n_common"
)
#: §9.1 run_config keys (fully-resolved config asdict of ResolvedConfig + the diagnostics block).
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
    "cache",
    "metrics",
    "diagnostics",
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


def test_manifest_completeness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verification test 7: every number-affecting parameter appears in the manifest (PART 4)."""
    from benchmark.runner import ExperimentRunner

    patch_runner_factories(monkeypatch)
    ts = "20260702T093000Z"
    variants = [
        PipelineCfg("semantic_e5", ("semantic_e5",), None, None, None),
        PipelineCfg("bm25_rerank", ("bm25",), None, "rr", 100),  # reranker-only -> recall@100 excluded
    ]
    cfg = _config(variants=variants, timestamp=ts)
    ExperimentRunner().run(cfg, output_dir=str(tmp_path))
    payload = json.loads((tmp_path / f"run_config_{ts}.json").read_text(encoding="utf-8"))

    # Stats knobs (raw threshold + FDR regime + declared family).
    for key in ("bootstrap_B", "ci_level", "alpha", "correction", "test", "seed", "contrasts", "fdr_metrics"):
        assert key in payload["stats"], key
    # MF-3: NO wilcoxon_* keys in a permutation-run manifest.
    assert "wilcoxon_zero_method" not in payload["stats"]
    assert "wilcoxon_correction" not in payload["stats"]

    # Metrics policy + cutoffs.
    assert payload["metrics"]["unjudged"] == "condensed"
    assert payload["metrics"]["relevance_threshold"] == pytest.approx(0.5)
    assert payload["cutoff"] == 10 and payload["top_k"] == 100 and "seed" in payload

    # api_key is redacted (no live secret material).
    assert payload["services"]["embedders"]["e5"]["settings"]["api_key"] in ("${REDACTED}", "${DRY_KEY}") \
        or payload["services"]["embedders"]["e5"]["settings"]["api_key"].startswith("${")

    # Post-load diagnostics (§9.1): qrels digest + gain mapping, resolved BM25 + analysis, the FDR
    # family (size + members + reasoned exclusions), recall information, common subset.
    diag = payload["diagnostics"]
    assert set(diag["dataset"]) >= {"qrels_digest", "relevance_threshold", "gain_mapping", "n_qrels"}
    assert set(diag["index"]["bm25"]) >= {"k1", "b"}
    assert "analysis" in diag["index"]
    assert set(diag["stats"]) >= {"family_size", "family_members", "excluded"}
    # bm25_rerank vs bm25 (W==100) -> recall@100 structurally excluded with a reason.
    excluded_metrics = {(e["a"], e["b"], e["metric"]) for e in diag["stats"]["excluded"]}
    assert ("bm25_rerank", "bm25", "recall@100") in excluded_metrics
    assert all(e["reason"] for e in diag["stats"]["excluded"])
    assert "recall_information" in diag
    assert "common_subset" in diag and "retrieval_failures" in diag
