"""Phase 7 — golden-file + behavioral tests for benchmark.io_csv (docs/experiment.md §9, §9.1).

The three CSV writers are asserted BYTE-FOR-BYTE against committed golden files (exact headers +
field order + the NaN->empty and degenerate serializations). ``write_run_config`` is asserted by
JSON round-trip + presence of the §9.1 fields (its content depends on the resolved config, not a
fixed schema, so it is not a byte-for-byte golden).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmark.config import (
    EmbedderCfg,
    FuserCfg,
    PipelineCfg,
    RerankerCfg,
    ResolvedConfig,
    SearcherCfg,
    Services,
)
from benchmark.io_csv import (
    write_comparison_csv,
    write_metrics_csv,
    write_results_csv,
    write_run_config,
)
from benchmark.common.models import RankedResult, ScoredDoc
from benchmark.evaluation.metrics import Metrics
from benchmark.evaluation.stats import ComparisonResult, StatsCfg

TIMESTAMP = "20260101T000000Z"


# --- result CSV -------------------------------------------------------------------------------


def _results_by_variant() -> dict[str, list[RankedResult]]:
    """A tiny multi-variant result mapping (baseline first) so the `variant` column is exercised."""
    return {
        "bm25": [
            RankedResult(
                "q1", [ScoredDoc("d1", 5.0), ScoredDoc("d2", 4.5), ScoredDoc("d3", 3.25)]
            ),
            RankedResult("q2", [ScoredDoc("d9", 1.0)]),
        ],
        "semantic_e5": [
            RankedResult("q1", [ScoredDoc("d2", 0.9), ScoredDoc("d1", 0.8)]),
        ],
    }


def test_result_csv_matches_golden(tmp_path: Path, golden_dir: Path) -> None:
    """result CSV: leading variant col, docs[0] -> position 1 ascending, byte-for-byte golden."""
    path = write_results_csv(_results_by_variant(), TIMESTAMP, output_dir=tmp_path)
    assert path.name == f"result_{TIMESTAMP}.csv"
    golden = (golden_dir / f"result_{TIMESTAMP}.csv").read_bytes()
    assert path.read_bytes() == golden


def test_result_csv_header_and_position(tmp_path: Path) -> None:
    """Header is exactly variant,query_id,product_id,score,position; positions 1-based ascending."""
    results = {"bm25": [RankedResult("q1", [ScoredDoc("a", 2.0), ScoredDoc("b", 1.0)])]}
    path = write_results_csv(results, TIMESTAMP, output_dir=tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "variant,query_id,product_id,score,position"
    assert lines[1] == "bm25,q1,a,2.0,1"
    assert lines[2] == "bm25,q1,b,1.0,2"


def test_result_csv_respects_returned_length(tmp_path: Path) -> None:
    """Rows per query equal the returned doc count (pipeline already truncates to top_k, §8.0)."""
    docs = [ScoredDoc(f"d{i}", float(i)) for i in range(5)]
    path = write_results_csv({"bm25": [RankedResult("q1", docs)]}, TIMESTAMP, output_dir=tmp_path)
    data_rows = path.read_text(encoding="utf-8").splitlines()[1:]
    assert len(data_rows) == 5


# --- metrics CSV ------------------------------------------------------------------------------


def _metrics_by_variant() -> dict[str, dict[str, Metrics]]:
    """Multi-variant metrics (baseline first) covering the three NaN cases in the golden.

    q_full is fully judged (all four numeric); q_noscore has n_scored==0 (avg/ndcg/precision NaN)
    with R>0 so recall is the finite 0.0; q_norel has judged-irrelevant docs (avg/ndcg/precision
    finite) but R==0 so recall is NaN -> empty.
    """
    return {
        "bm25": {
            "q_full": Metrics(0.75, 0.9, 0.5, 1.0, n_results=6, n_scored=4, n_missing=2),
            "q_noscore": Metrics(math.nan, math.nan, 0.0, math.nan, n_results=3, n_scored=0, n_missing=3),
            "q_norel": Metrics(0.0, 0.0, math.nan, 0.0, n_results=2, n_scored=2, n_missing=0),
        },
        "semantic_e5": {
            "q_full": Metrics(0.8, 0.95, 1.0, 0.5, n_results=4, n_scored=3, n_missing=1),
        },
    }


def test_metrics_csv_matches_golden(tmp_path: Path, golden_dir: Path) -> None:
    """metrics CSV: leading variant col, NaN metric cells EMPTY, ints always present, byte golden."""
    path = write_metrics_csv(_metrics_by_variant(), TIMESTAMP, output_dir=tmp_path)
    assert path.name == f"metrics_{TIMESTAMP}.csv"
    golden = (golden_dir / f"metrics_{TIMESTAMP}.csv").read_bytes()
    assert path.read_bytes() == golden


def test_metrics_header_ends_with_counts(tmp_path: Path) -> None:
    """The metrics header leads with variant and ends with the two int count columns (§9)."""
    path = write_metrics_csv(
        {"bm25": {"q1": Metrics(1.0, 1.0, 1.0, 1.0, n_results=2, n_scored=1, n_missing=0)}},
        TIMESTAMP,
        output_dir=tmp_path,
    )
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "variant,query_id,avg_relevance,ndcg@10,recall@10,precision@10,n_results,n_scored,n_missing"


def test_metrics_nan_serializes_as_empty_adjacent_commas(tmp_path: Path) -> None:
    """A NaN metric cell is two adjacent commas (empty field), while counts stay integers."""
    path = write_metrics_csv(
        {"bm25": {"q0": Metrics(math.nan, math.nan, math.nan, math.nan, n_results=0, n_scored=0, n_missing=0)}},
        TIMESTAMP,
        output_dir=tmp_path,
    )
    row = path.read_text(encoding="utf-8").splitlines()[1]
    # variant + query_id + four empty metric cells + two int counts.
    assert row == "bm25,q0,,,,,0,0,0"


# --- comparison CSV ---------------------------------------------------------------------------


def _comparison_rows() -> list[ComparisonResult]:
    """A normal row + empty_paired_set + all_zero_delta, exercising baseline_value/variant_value."""
    return [
        ComparisonResult(
            variant="semantic_e5",
            metric="avg_relevance",
            baseline_value=0.5,
            variant_value=0.625,
            delta=0.125,
            delta_ci_lo=0.05,
            delta_ci_high=0.2,
            p_value=0.01,
            significant_raw=True,
            p_value_adjusted=0.02,
            significant=True,
            note=None,
        ),
        ComparisonResult(
            variant="semantic_e5",
            metric="ndcg@10",
            baseline_value=None,
            variant_value=None,
            delta=None,
            delta_ci_lo=None,
            delta_ci_high=None,
            p_value=1.0,
            significant_raw=False,
            p_value_adjusted=1.0,
            significant=False,
            note="empty_paired_set",
        ),
        ComparisonResult(
            variant="semantic_e5",
            metric="recall@10",
            baseline_value=0.4,
            variant_value=0.4,
            delta=0.0,
            delta_ci_lo=0.0,
            delta_ci_high=0.0,
            p_value=1.0,
            significant_raw=False,
            p_value_adjusted=1.0,
            significant=False,
            note="all_zero_delta",
        ),
    ]


def test_comparison_csv_matches_golden(tmp_path: Path, golden_dir: Path) -> None:
    """comparison CSV: a normal row + empty_paired_set + all_zero_delta, byte-for-byte golden (§8.1)."""
    path = write_comparison_csv("bm25", _comparison_rows(), TIMESTAMP, output_dir=tmp_path)
    assert path.name == f"comparison_{TIMESTAMP}.csv"
    golden = (golden_dir / f"comparison_{TIMESTAMP}.csv").read_bytes()
    assert path.read_bytes() == golden


def test_comparison_header_is_twelve_columns(tmp_path: Path) -> None:
    """The comparison header is exactly the 12-column §9 header (baseline + values + stats)."""
    path = write_comparison_csv("bm25", [], TIMESTAMP, output_dir=tmp_path)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "baseline,variant,metric,baseline_value,variant_value,delta,delta_ci_lo,delta_ci_high,"
        "p_value,significant_raw,p_value_adjusted,significant"
    )


def test_comparison_empty_paired_set_row(tmp_path: Path) -> None:
    """empty_paired_set: baseline_value/variant_value/delta/CI empty, p=1.0, flags false (§8.1)."""
    row = ComparisonResult(
        variant="v", metric="recall@10", baseline_value=None, variant_value=None,
        delta=None, delta_ci_lo=None, delta_ci_high=None,
        p_value=1.0, significant_raw=False, p_value_adjusted=1.0, significant=False,
        note="empty_paired_set",
    )
    path = write_comparison_csv("bm25", [row], TIMESTAMP, output_dir=tmp_path)
    data = path.read_text(encoding="utf-8").splitlines()[1]
    assert data == "bm25,v,recall@10,,,,,,1.0,false,1.0,false"


def test_comparison_all_zero_delta_row(tmp_path: Path) -> None:
    """all_zero_delta: baseline_value==variant_value, delta=0.0, CI 0.0/0.0, p=1.0, flags false (§8.1)."""
    row = ComparisonResult(
        variant="v", metric="ndcg@10", baseline_value=0.3, variant_value=0.3,
        delta=0.0, delta_ci_lo=0.0, delta_ci_high=0.0,
        p_value=1.0, significant_raw=False, p_value_adjusted=1.0, significant=False,
        note="all_zero_delta",
    )
    path = write_comparison_csv("bm25", [row], TIMESTAMP, output_dir=tmp_path)
    data = path.read_text(encoding="utf-8").splitlines()[1]
    assert data == "bm25,v,ndcg@10,0.3,0.3,0.0,0.0,0.0,1.0,false,1.0,false"


# --- run_config JSON --------------------------------------------------------------------------


def _resolved_config() -> ResolvedConfig:
    services = Services(
        embedders={
            "e5": EmbedderCfg(
                name="e5", provider="cohere",
                settings={"model_id": "embed-english-v3.0", "api_key": "co-test"},
            )
        },
        rerankers={
            "co-rr": RerankerCfg(
                name="co-rr", provider="cohere",
                settings={"model_id": "rerank-v3.5", "top_n": 100},
            )
        },
        searchers={
            "bm25": SearcherCfg(name="bm25", provider="elasticsearch", kind="lexical", embedder=None),
            "semantic_e5": SearcherCfg(
                name="semantic_e5", provider="elasticsearch", kind="vector", embedder="e5"
            ),
        },
    )
    baseline = PipelineCfg(
        id="baseline", retrievers=("bm25",), fuser=None, reranker=None, rerank_window_size=None
    )
    variants = [
        PipelineCfg(
            id="hybrid_e5",
            retrievers=("bm25", "semantic_e5"),
            fuser=FuserCfg(type="rrf", rank_constant=60, window=100),
            reranker="co-rr",
            rerank_window_size=100,
        )
    ]
    return ResolvedConfig(
        dataset={"name": "wands", "path": "./dataset/wands"},
        indexer={"provider": "elasticsearch", "index": "wands_bench"},
        services=services,
        baseline=baseline,
        variants=variants,
        stats=StatsCfg(),
        cutoff=10,
        top_k=100,
        baseline_id="baseline",
        timestamp=TIMESTAMP,
        seed=1234,
    )


def test_run_config_round_trips_and_has_section_9_1_fields(tmp_path: Path) -> None:
    """run_config JSON round-trips and carries the §9.1 fields (services, pipelines, stats, ...)."""
    cfg = _resolved_config()
    path = write_run_config(cfg, output_dir=tmp_path)
    assert path.name == f"run_config_{TIMESTAMP}.json"

    loaded = json.loads(path.read_text(encoding="utf-8"))

    # Top-level §9.1 fields.
    assert loaded["timestamp"] == TIMESTAMP
    assert loaded["seed"] == 1234
    assert loaded["cutoff"] == 10
    assert loaded["top_k"] == 100

    # Resolved services registry — embedders/rerankers/searchers by name.
    services = loaded["services"]
    assert set(services["embedders"]) == {"e5"}
    # Embedder connector config round-trips (provider + settings; §3.4).
    assert services["embedders"]["e5"]["provider"] == "cohere"
    assert services["embedders"]["e5"]["settings"]["model_id"] == "embed-english-v3.0"
    assert set(services["rerankers"]) == {"co-rr"}
    assert set(services["searchers"]) == {"bm25", "semantic_e5"}

    # Resolved pipelines — baseline + every named variant with retrievers/fuser/reranker/window.
    assert loaded["baseline"]["id"] == "baseline"
    variant = loaded["variants"][0]
    assert variant["id"] == "hybrid_e5"
    assert variant["retrievers"] == ["bm25", "semantic_e5"]
    assert variant["fuser"] == {"type": "rrf", "rank_constant": 60, "window": 100}
    assert variant["reranker"] == "co-rr"
    assert variant["rerank_window_size"] == 100

    # Stats block: alpha as raw+q (one number), correction, bootstrap, ci level, test + zero/tie, seed.
    stats = loaded["stats"]
    assert stats["alpha"] == 0.05
    assert stats["correction"] == "bh"
    assert stats["bootstrap_B"] == 10000
    assert stats["ci_level"] == 0.95
    assert stats["test"] == "wilcoxon"
    assert stats["wilcoxon_zero_method"] == "wilcox"
    assert stats["wilcoxon_correction"] is True
    assert stats["seed"] == 1234


def test_run_config_from_full_config_yaml(tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The §10 config.yaml resolves and its run_config JSON round-trips (services + all pipelines)."""
    from benchmark.config import load_config

    monkeypatch.setenv("COHERE_KEY", "test-key")
    monkeypatch.setenv("ES_URL", "http://localhost:9200")
    cfg = load_config(repo_root / "config.yaml")
    path = write_run_config(cfg, output_dir=tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["baseline"]["id"] == "bm25"  # config.yaml sets pipelines.baseline_id: bm25 (§9)
    # config.yaml is a user-editable artifact — assert STRUCTURE (resolves + round-trips), not the
    # exact variant names, so editing which pipelines to run does not break this test.
    variant_ids = {v["id"] for v in loaded["variants"]}
    assert variant_ids and all(isinstance(vid, str) and vid for vid in variant_ids)  # >=1 named variant
    assert "bm25" not in variant_ids  # the baseline is never also a variant
    assert loaded["stats"]["correction"] == "bh"
