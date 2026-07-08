"""Offline tests for ``eval:sweep`` (scripts/sweep.py) — the single diagnostic sweep mechanism.

Drives the sweep with the SAME in-memory fakes as the runner (``patch_runner_factories`` — NO ES, NO
network), proving each ``--axis`` iterates its grid and reuses the ONE runner / evaluator / comparator
(no forked metric code) to emit the tidy diagnostic ``sweep_{axis}_{ts}.csv``. A real sweep needs a
live index + provider keys; here the fakes stand in so the ORCHESTRATION is testable.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmark.config import ConfigError, FuserCfg, PipelineCfg
from scripts.sweep import (
    _BM25_B,
    _BM25_K1,
    _RERANK_WINDOWS,
    _RRF_CONSTANTS,
    run_sweep,
)
from tests.unit.test_runner import _config, patch_runner_factories


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cfg_with_rerank_and_hybrid(ts: str):
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
    return _config(variants=variants, timestamp=ts)


def test_rerank_window_sweep_iterates_grid_with_paired_cis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_runner_factories(monkeypatch)
    ts = "20260703T000000Z"
    cfg = _cfg_with_rerank_and_hybrid(ts)

    path = run_sweep(cfg, "rerank_window", output_dir=str(tmp_path))

    assert path.name == f"sweep_rerank_window_{ts}.csv"
    rows = _read_rows(path)
    # 1 rerank variant × 4 windows × 2 metrics (ndcg@10, recall@50).
    assert len(rows) == len(_RERANK_WINDOWS) * 2
    assert {r["axis_value"] for r in rows} == {str(w) for w in _RERANK_WINDOWS}
    assert {r["metric"] for r in rows} == {"ndcg@10", "recall@50"}
    assert {r["system"] for r in rows} == {"bm25_rerank"}
    # Paired CI vs the unreranked base is present for the ndcg@10 cells (a real reranking effect).
    ndcg = [r for r in rows if r["metric"] == "ndcg@10"]
    assert all(r["ci_lo"] != "" and r["ci_high"] != "" for r in ndcg)
    assert all(r["n_common"] == str(2) for r in rows)  # both fake queries finite


def test_rrf_k_sweep_reports_three_metrics_per_constant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_runner_factories(monkeypatch)
    ts = "20260703T001000Z"
    cfg = _cfg_with_rerank_and_hybrid(ts)

    path = run_sweep(cfg, "rrf_k", output_dir=str(tmp_path))

    rows = _read_rows(path)
    # 1 pure hybrid × 3 rank_constants × 3 metrics.
    assert len(rows) == len(_RRF_CONSTANTS) * 3
    assert {r["axis_value"] for r in rows} == {str(k) for k in _RRF_CONSTANTS}
    assert {r["metric"] for r in rows} == {"ndcg@10", "precision@10", "recall@100"}
    assert {r["system"] for r in rows} == {"hybrid_e5"}
    assert all(r["value"] != "" for r in rows)  # a value per cell on the finite subset
    assert all(r["ci_lo"] == "" for r in rows)  # single-system axis -> no paired CI


def test_bm25_k1_b_sweep_reindexes_every_cell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patch_runner_factories(monkeypatch)
    from benchmark.runner import ExperimentRunner

    # Spy on build_index to prove a reindex per cell into a DISTINCT scratch index (BM25 is index-time).
    reindexed: list[str] = []
    original = ExperimentRunner.build_index

    def _spy(self: ExperimentRunner, cfg_cell) -> object:  # type: ignore[no-untyped-def]
        reindexed.append(str(cfg_cell.indexer["index"]))
        return original(self, cfg_cell)

    monkeypatch.setattr(ExperimentRunner, "build_index", _spy)

    ts = "20260703T002000Z"
    cfg = _config(variants=[], timestamp=ts)
    path = run_sweep(cfg, "bm25_k1_b", output_dir=str(tmp_path))

    rows = _read_rows(path)
    n_cells = len(_BM25_K1) * len(_BM25_B)
    assert len(rows) == n_cells == 16
    assert {r["metric"] for r in rows} == {"ndcg@10"}
    assert {r["system"] for r in rows} == {"bm25"}
    # One reindex per cell, each into a distinct scratch index (params bake in at index time).
    assert len(reindexed) == n_cells
    assert len(set(reindexed)) == n_cells
    assert all("__sweep_k1_" in name for name in reindexed)


def test_unknown_axis_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    patch_runner_factories(monkeypatch)
    cfg = _config(variants=[], timestamp="20260703T003000Z")
    with pytest.raises(ConfigError, match="unknown sweep axis"):
        run_sweep(cfg, "not_an_axis", output_dir=str(tmp_path))


def test_sweep_does_not_write_frozen_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Sweeps are diagnostics — they must NOT touch the frozen result/metrics/comparison files (§9).
    patch_runner_factories(monkeypatch)
    ts = "20260703T004000Z"
    cfg = _cfg_with_rerank_and_hybrid(ts)

    run_sweep(cfg, "rrf_k", output_dir=str(tmp_path))

    assert list(tmp_path.glob("result_*.csv")) == []
    assert list(tmp_path.glob("metrics_*.csv")) == []
    assert list(tmp_path.glob("comparison_*.csv")) == []
    assert list(tmp_path.glob("run_config_*.json")) == []
    assert [p.name for p in tmp_path.glob("sweep_*.csv")] == [f"sweep_rrf_k_{ts}.csv"]
