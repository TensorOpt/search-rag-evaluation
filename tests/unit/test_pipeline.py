"""SearchPipeline composition tests via FakeBackend (docs/experiment.md §3.6/§3.7, plan Phase 5)."""

from __future__ import annotations

import pytest

from benchmark.fusion import fuse_rrf_local
from benchmark.models import Query
from benchmark.pipeline import (
    FuseCfg,
    PipelineSpec,
    RerankCfg,
    SearchPipeline,
    StageCfg,
    _LocalFusePlan,
)
from tests.conftest import (
    _BM25_DOCS,
    _SEMANTIC_DOCS,
    Bm25Spec,
    FakeBackend,
    FuseSpec,
    RerankSpec,
    SemanticSpec,
)

BM25 = StageCfg.bm25(fields=["search_text"])
SEMANTIC = StageCfg.semantic(field="sem__e5")
FUSE = FuseCfg(rank_constant=10, rank_window_size=50)
RERANK = RerankCfg(inference_id="rr", field="search_text", rank_window_size=50)

# The six variant shapes (§4 table).
SPECS = {
    "bm25": PipelineSpec(retrievers=[BM25]),
    "semantic": PipelineSpec(retrievers=[SEMANTIC]),
    "hybrid": PipelineSpec(retrievers=[BM25, SEMANTIC], fuse=FUSE),
    "bm25_rerank": PipelineSpec(retrievers=[BM25], rerank=RERANK),
    "semantic_rerank": PipelineSpec(retrievers=[SEMANTIC], rerank=RERANK),
    "hybrid_rerank": PipelineSpec(retrievers=[BM25, SEMANTIC], fuse=FUSE, rerank=RERANK),
}

QUERIES = [Query("q1", "sofa"), Query("q2", "lamp")]


# --- caps TRUE: full server-side composition -------------------------------------------------


def test_all_six_variants_run_one_result_per_query() -> None:
    backend = FakeBackend()  # both server-side caps true
    pipeline = SearchPipeline(backend)
    for spec in SPECS.values():
        results = list(pipeline.run(spec, QUERIES, top_k=10))
        assert [r.query_id for r in results] == ["q1", "q2"]


def test_server_side_combinators_composed_only_where_expected() -> None:
    for name, spec in SPECS.items():
        backend = FakeBackend()
        plan = SearchPipeline(backend).plan(spec)
        fused = "hybrid" in name
        reranked = name.endswith("_rerank")
        assert bool(backend.fuse_calls) is fused, name
        assert bool(backend.rerank_calls) is reranked, name
        if reranked:
            assert isinstance(plan, RerankSpec), name
            # rerank wraps the fused plan for hybrid_rerank, else the bare leaf.
            expected_child = FuseSpec if fused else (Bm25Spec if "bm25" in name else SemanticSpec)
            assert isinstance(plan.child, expected_child), name
        elif fused:
            assert isinstance(plan, FuseSpec), name
        else:
            assert isinstance(plan, (Bm25Spec, SemanticSpec)), name


def test_bm25_baseline_is_single_leaf() -> None:
    plan = SearchPipeline(FakeBackend()).plan(SPECS["bm25"])
    assert plan == Bm25Spec(fields=("search_text",))


# --- caps FALSE: harness-side fuse fallback + deferred rerank --------------------------------


def test_hybrid_uses_local_fuse_when_no_server_side_rrf() -> None:
    backend = FakeBackend(server_side_rrf=False, server_side_rerank=False)
    pipeline = SearchPipeline(backend)

    plan = pipeline.plan(SPECS["hybrid"])
    assert isinstance(plan, _LocalFusePlan)
    assert not backend.fuse_calls  # server-side fuse never called

    results = list(pipeline.run(SPECS["hybrid"], QUERIES, top_k=10))
    expected = fuse_rrf_local(
        [_BM25_DOCS, _SEMANTIC_DOCS],
        rank_constant=FUSE.rank_constant,
        rank_window_size=FUSE.rank_window_size,
    )
    for result in results:
        assert result.docs == expected


def test_single_leaf_variants_run_without_caps() -> None:
    backend = FakeBackend(server_side_rrf=False, server_side_rerank=False)
    pipeline = SearchPipeline(backend)
    assert list(pipeline.run(SPECS["bm25"], QUERIES, top_k=10))[0].docs == _BM25_DOCS
    assert list(pipeline.run(SPECS["semantic"], QUERIES, top_k=10))[0].docs == _SEMANTIC_DOCS


@pytest.mark.parametrize("name", ["bm25_rerank", "semantic_rerank", "hybrid_rerank"])
def test_rerank_variants_raise_when_no_server_side_rerank(name: str) -> None:
    backend = FakeBackend(server_side_rrf=False, server_side_rerank=False)
    pipeline = SearchPipeline(backend)
    with pytest.raises(NotImplementedError, match="rerank fallback is deferred"):
        pipeline.plan(SPECS[name])
    with pytest.raises(NotImplementedError):
        list(pipeline.run(SPECS[name], QUERIES, top_k=10))


def test_rerank_on_local_fuse_plan_raises_even_if_rerank_server_side() -> None:
    # server_side_rerank true but server_side_rrf false -> base is a local fuse plan, which
    # cannot be reranked server-side; must raise rather than silently drop the fuse.
    backend = FakeBackend(server_side_rrf=False, server_side_rerank=True)
    with pytest.raises(NotImplementedError):
        SearchPipeline(backend).plan(SPECS["hybrid_rerank"])


# --- guards ----------------------------------------------------------------------------------


def test_unknown_stage_kind_raises() -> None:
    bad = PipelineSpec(retrievers=[StageCfg(kind="lexical")])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown retrieval stage kind"):
        SearchPipeline(FakeBackend()).plan(bad)


def test_multiple_leaves_without_fuse_raises() -> None:
    bad = PipelineSpec(retrievers=[BM25, SEMANTIC])  # 2 leaves, no fuse
    with pytest.raises(ValueError, match="exactly one retriever"):
        SearchPipeline(FakeBackend()).plan(bad)
