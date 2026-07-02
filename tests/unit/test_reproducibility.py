"""Reproducibility test — automates §1.4(2): same config + seed => identical metrics/stats.

Runs :class:`benchmark.runner.ExperimentRunner` TWICE with the SAME resolved config (same
``StatsCfg.seed``) via the ``test_runner`` fakes, into two separate tmp dirs, then asserts the
``metrics_*.csv`` and ``comparison_*.csv`` artifacts are BYTE-IDENTICAL between the two runs. This
proves the runner + the seeded bootstrap CI / p-values (§8.2/§8.3) are deterministic given a fixed
seed — with fakes there is NO backend nondeterminism, so the artifacts must match exactly (§9.1).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.test_runner import (
    FuserCfg,
    PipelineCfg,
    _config,
    patch_runner_factories,
)


def _run_into(monkeypatch: pytest.MonkeyPatch, out_dir: Path, ts: str) -> None:
    from benchmark.runner import ExperimentRunner

    patch_runner_factories(monkeypatch)
    variants = [
        PipelineCfg("semantic_e5", ("semantic_e5",), None, None, None),
        PipelineCfg(
            "hybrid_e5",
            ("bm25", "semantic_e5"),
            FuserCfg("rrf", rank_constant=60, window=100),
            None,
            None,
        ),
    ]
    cfg = _config(variants=variants, timestamp=ts)
    ExperimentRunner().run(cfg, output_dir=str(out_dir))


def test_two_runs_same_seed_are_byte_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ts = "20260702T100000Z"  # same timestamp both runs => same filenames to diff
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    # Fresh monkeypatch context per run (fresh fake backend), same seeded cfg.
    with monkeypatch.context() as mp:
        _run_into(mp, dir_a, ts)
    with monkeypatch.context() as mp:
        _run_into(mp, dir_b, ts)

    metrics = sorted(p.name for p in dir_a.glob(f"metrics_*_{ts}.csv"))
    comparisons = sorted(p.name for p in dir_a.glob(f"comparison_*_{ts}.csv"))
    assert metrics, "expected metrics artifacts"
    assert comparisons, "expected comparison artifacts (seeded bootstrap CI + p-values)"

    for name in metrics + comparisons:
        left = (dir_a / name).read_bytes()
        right = (dir_b / name).read_bytes()
        assert left == right, f"{name} differs between two same-seed runs (non-deterministic)"
