"""Matrix expansion, best_per_model selection, and spec_for tests (docs/experiment.md §4, §8.0a, §10, plan Phase 6).

Uses the §10 axes (3 embedding models, 2 rerankers, 10-step k-sweep) to assert exact per-family
counts and baseline-first order, the two-mode hybrid_rerank behavior, the seed-independent
best_per_model tie-break, and the spec_for object graph for all six variants via a fake factory.
"""

from __future__ import annotations

from typing import Sequence

import pytest

from benchmark.matrix import (
    EmbeddingModelCfg,
    RerankerCfg,
    ResolvedConfig,
    VariantCfg,
    expand_matrix,
    resolve_hybrid_rerank_best_per_model,
    spec_for,
)
from benchmark.models import IndexMapping, ScoredDoc
from benchmark.pipeline import HybridSearch, SearchPipeline
from benchmark.protocols import Reranker, Searcher
from benchmark.stats import StatsCfg

MODEL_IDS = ["e5-small", "elser", "openai-3-small"]
RERANKER_IDS = ["cohere-rerank-v3", "bge-reranker"]
K_SWEEP = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
WINDOW = 100


def make_config(hybrid_rerank_k: int | str = 60) -> ResolvedConfig:
    return ResolvedConfig(
        dataset={"name": "wands"},
        backend={"kind": "elasticsearch"},
        embedding_models=[EmbeddingModelCfg(m) for m in MODEL_IDS],
        rerankers=[RerankerCfg(r) for r in RERANKER_IDS],
        rrf_k_sweep=K_SWEEP,
        variants=["bm25", "semantic", "hybrid", "bm25_rerank", "semantic_rerank", "hybrid_rerank"],
        reranker_endpoints={},
        stats=StatsCfg(),
        cutoff=10,
        top_k=100,
        rank_window_size=WINDOW,
        hybrid_rerank_k=hybrid_rerank_k,
        baseline_id="bm25",
        timestamp="20260701T000000Z",
        seed=1234,
    )


# --- expand_matrix counts & order --------------------------------------------------------------


def _by_family(variants: Sequence[VariantCfg]) -> dict[str, list[VariantCfg]]:
    families: dict[str, list[VariantCfg]] = {}
    for v in variants:
        families.setdefault(v.family, []).append(v)
    return families


def test_expand_counts_and_baseline_first() -> None:
    variants = expand_matrix(make_config(hybrid_rerank_k=60))
    families = _by_family(variants)

    assert variants[0].id == "bm25"  # baseline first (§10)
    assert len(families["bm25"]) == 1
    assert len(families["semantic"]) == len(MODEL_IDS)  # 3
    assert len(families["hybrid"]) == len(MODEL_IDS) * len(K_SWEEP)  # 30
    assert len(families["bm25_rerank"]) == len(RERANKER_IDS)  # 2
    assert len(families["semantic_rerank"]) == len(MODEL_IDS) * len(RERANKER_IDS)  # 6
    assert len(families["hybrid_rerank"]) == len(MODEL_IDS) * len(RERANKER_IDS)  # 6
    assert len(variants) == 1 + 3 + 30 + 2 + 6 + 6  # 48


def test_expand_family_order() -> None:
    variants = expand_matrix(make_config())
    families_in_order = list(dict.fromkeys(v.family for v in variants))
    assert families_in_order == [
        "bm25",
        "semantic",
        "hybrid",
        "bm25_rerank",
        "semantic_rerank",
        "hybrid_rerank",
    ]


def test_expand_variant_ids_match_section9() -> None:
    variants = {v.id for v in expand_matrix(make_config(hybrid_rerank_k=60))}
    assert "bm25" in variants
    assert "semantic__e5-small" in variants
    assert "hybrid__e5-small__k60" in variants
    assert "bm25_rerank__cohere-rerank-v3" in variants
    assert "semantic_rerank__e5-small__cohere-rerank-v3" in variants
    assert "hybrid_rerank__e5-small__cohere-rerank-v3__k60" in variants


# --- best_per_model deferral -------------------------------------------------------------------


def test_best_per_model_defers_hybrid_rerank() -> None:
    variants = expand_matrix(make_config(hybrid_rerank_k="best_per_model"))
    assert [v for v in variants if v.family == "hybrid_rerank"] == []


def test_int_emits_hybrid_rerank_rows() -> None:
    variants = expand_matrix(make_config(hybrid_rerank_k=60))
    hybrid_rerank = [v for v in variants if v.family == "hybrid_rerank"]
    assert len(hybrid_rerank) == len(MODEL_IDS) * len(RERANKER_IDS)
    assert all(v.rrf_k == 60 for v in hybrid_rerank)


def test_bool_hybrid_rerank_k_rejected() -> None:
    with pytest.raises(ValueError):
        expand_matrix(make_config(hybrid_rerank_k=True))


# --- resolve_hybrid_rerank_best_per_model ------------------------------------------------------


def _per_query_for_hybrids(best_k_per_model: dict[str, int]) -> dict[str, dict[str, dict[str, float]]]:
    """Build in-memory per-query maps so hybrid__m__k{best} has the max mean nDCG@10 for model m."""
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    for model, best_k in best_k_per_model.items():
        for k in K_SWEEP:
            ndcg = 0.9 if k == best_k else 0.5
            per_query[f"hybrid__{model}__k{k}"] = {
                "q1": {"ndcg@10": ndcg, "avg_relevance": 0.0, "recall@10": 0.0, "precision@10": 0.0},
                "q2": {"ndcg@10": ndcg, "avg_relevance": 0.0, "recall@10": 0.0, "precision@10": 0.0},
            }
    return per_query


def test_best_per_model_argmax_and_expansion() -> None:
    cfg = make_config(hybrid_rerank_k="best_per_model")
    best = {"e5-small": 30, "elser": 60, "openai-3-small": 90}
    rows = resolve_hybrid_rerank_best_per_model(cfg, _per_query_for_hybrids(best))

    assert len(rows) == len(MODEL_IDS) * len(RERANKER_IDS)  # one per (model, reranker)
    chosen_k = {v.embedding_model_id: v.rrf_k for v in rows}
    assert chosen_k == best
    assert all(v.family == "hybrid_rerank" for v in rows)


def test_best_per_model_tie_break_smallest_k() -> None:
    cfg = make_config(hybrid_rerank_k="best_per_model")
    # All k tie on mean nDCG@10 for e5-small -> smallest k (10) wins; others have a clear best.
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    for k in K_SWEEP:
        per_query[f"hybrid__e5-small__k{k}"] = {
            "q1": {"ndcg@10": 0.7, "avg_relevance": 0.0, "recall@10": 0.0, "precision@10": 0.0},
        }
    for model, best_k in {"elser": 20, "openai-3-small": 40}.items():
        for k in K_SWEEP:
            per_query[f"hybrid__{model}__k{k}"] = {
                "q1": {"ndcg@10": 0.9 if k == best_k else 0.5, "avg_relevance": 0.0,
                       "recall@10": 0.0, "precision@10": 0.0},
            }
    rows = resolve_hybrid_rerank_best_per_model(cfg, per_query)
    chosen_k = {v.embedding_model_id: v.rrf_k for v in rows}
    assert chosen_k["e5-small"] == 10  # smallest-k tie-break


def test_best_per_model_seed_independent_deterministic() -> None:
    cfg = make_config(hybrid_rerank_k="best_per_model")
    per_query = _per_query_for_hybrids({"e5-small": 30, "elser": 60, "openai-3-small": 90})
    first = resolve_hybrid_rerank_best_per_model(cfg, per_query)
    second = resolve_hybrid_rerank_best_per_model(cfg, per_query)
    assert first == second  # identical across runs; no RNG involved


# --- spec_for object graphs (fake factory) -----------------------------------------------------


class _FakeSearcher(Searcher):
    def __init__(self, kind: str, target: object) -> None:
        self.kind = kind
        self.target = target

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        return []


class _FakeReranker(Reranker):
    def __init__(self, inference_id: str, field: str) -> None:
        self.inference_id = inference_id
        self.field = field

    def rerank(self, query: str, candidates: Sequence[ScoredDoc]) -> list[ScoredDoc]:
        return list(candidates)


class _FakeFactory:
    def lexical(self, *, fields: Sequence[str]) -> Searcher:
        return _FakeSearcher("lexical", list(fields))

    def vector(self, *, field: str) -> Searcher:
        return _FakeSearcher("vector", field)

    def reranker(self, inference_id: str, field: str) -> Reranker:
        return _FakeReranker(inference_id, field)


MAPPING = IndexMapping(
    index_name="wands_bench",
    search_text_field="search_text",
    sem_fields={m: f"sem__{m.replace('-', '_')}" for m in MODEL_IDS},
    backend_mapping={},
)


def _variant(family: str) -> VariantCfg:
    return next(v for v in expand_matrix(make_config(60)) if v.family == family)


def test_spec_for_bm25_is_leaf_pipeline() -> None:
    pipeline = spec_for(_variant("bm25"), MAPPING, _FakeFactory())
    assert isinstance(pipeline, SearchPipeline)
    assert isinstance(pipeline.retriever, _FakeSearcher)
    assert pipeline.retriever.kind == "lexical"
    assert pipeline.retriever.target == ["search_text"]
    assert pipeline.reranker is None
    assert pipeline.rerank_window_size is None


def test_spec_for_semantic_is_vector_leaf() -> None:
    pipeline = spec_for(_variant("semantic"), MAPPING, _FakeFactory())
    assert isinstance(pipeline.retriever, _FakeSearcher)
    assert pipeline.retriever.kind == "vector"
    assert pipeline.retriever.target == "sem__e5_small"
    assert pipeline.reranker is None


def test_spec_for_hybrid_is_hybridsearch() -> None:
    variant = _variant("hybrid")
    pipeline = spec_for(variant, MAPPING, _FakeFactory())
    assert isinstance(pipeline.retriever, HybridSearch)
    assert len(pipeline.retriever.retrievers) == 2  # lexical + vector
    assert pipeline.retriever.fuser.rank_constant == variant.rrf_k  # concrete k, no selection
    assert pipeline.retriever.retrieval_window_size == WINDOW
    assert pipeline.reranker is None


def test_spec_for_bm25_rerank_has_reranker() -> None:
    pipeline = spec_for(_variant("bm25_rerank"), MAPPING, _FakeFactory())
    assert isinstance(pipeline.retriever, _FakeSearcher)
    assert pipeline.retriever.kind == "lexical"
    assert isinstance(pipeline.reranker, _FakeReranker)
    assert pipeline.reranker.field == "search_text"  # mapping.rerank_field
    assert pipeline.rerank_window_size == WINDOW


def test_spec_for_semantic_rerank_has_vector_and_reranker() -> None:
    pipeline = spec_for(_variant("semantic_rerank"), MAPPING, _FakeFactory())
    assert isinstance(pipeline.retriever, _FakeSearcher)
    assert pipeline.retriever.kind == "vector"
    assert isinstance(pipeline.reranker, _FakeReranker)
    assert pipeline.rerank_window_size == WINDOW


def test_spec_for_hybrid_rerank_has_hybrid_and_reranker() -> None:
    pipeline = spec_for(_variant("hybrid_rerank"), MAPPING, _FakeFactory())
    assert isinstance(pipeline.retriever, HybridSearch)
    assert isinstance(pipeline.reranker, _FakeReranker)
    assert pipeline.rerank_window_size == WINDOW


def test_spec_for_rejects_no_fuse_multi_leaf() -> None:
    bad = VariantCfg(
        id="bad", family="bad", use_bm25=True, embedding_model_id="e5-small",
        fuse=False, rrf_k=None, window=WINDOW, reranker_id=None,
    )
    with pytest.raises(ValueError):
        spec_for(bad, MAPPING, _FakeFactory())
