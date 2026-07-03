"""Config load/resolve, ${VAR} substitution, service registry, pipeline validation, factory dispatch (docs/experiment.md §10, §11, plan Phase 6).

All offline: no adapter module is imported (the factories dispatch to dotted-path targets and are
never resolved here). Exercises env-var substitution, the typed services registry, every §10
pipeline validation error, baseline designation, and dispatch (known/unknown provider).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from benchmark.config import (
    DATASET_TARGETS,
    INDEXER_TARGETS,
    ConfigError,
    EmbedderCfg,
    PipelineCfg,
    RerankerCfg,
    SearcherCfg,
    load_config,
    load_dataset,
    make_indexer,
    make_searcher_factory,
    resolve_config,
)

CONFIG_YAML = textwrap.dedent(
    """
    dataset:
      name: wands
      path: ./dataset/wands
    services:
      - embedder: { name: e5,     provider: voyage, settings: { api_key: "${VOYAGE_KEY}", model_id: voyage-3.5 } }
      - embedder: { name: cohere, provider: cohere, settings: { api_key: "${COHERE_KEY}", model_id: embed-english-v3.0 } }
      - reranker: { name: co-rr,  provider: cohere, settings: { api_key: "${COHERE_KEY}", model_id: rerank-v3.5, top_n: 100 } }
      - searcher: { name: bm25,        provider: elasticsearch, kind: lexical }
      - searcher: { name: semantic_e5, provider: elasticsearch, kind: vector, embedder: e5 }
    indexer:
      provider: elasticsearch
      index: wands_bench
      settings: { url: "${ES_URL}" }
    pipelines:
      baseline:
        retriever: bm25
      variants:
        semantic_e5:   { retriever: semantic_e5 }
        hybrid_e5_k60:
          retrievers: [bm25, semantic_e5]
          fuser: { type: rrf, rank_constant: 60, window: 100 }
        bm25_rerank:
          retriever: bm25
          reranker: co-rr
          rerank_window_size: 100
    stats:
      test: wilcoxon
      correction: bh
      alpha: 0.05
      bootstrap_B: 10000
      ci_level: 0.95
      seed: 1234
    cutoff: 10
    top_k: 100
    """
)


@pytest.fixture(autouse=True)
def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ES_URL", "http://localhost:9200")
    monkeypatch.setenv("COHERE_KEY", "co-test")
    monkeypatch.setenv("VOYAGE_KEY", "vo-test")


def _parsed() -> dict[str, Any]:
    import yaml

    return dict(yaml.safe_load(CONFIG_YAML))


# --- load + ${VAR} substitution ----------------------------------------------------------------


def test_load_substitutes_env_and_resolves(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML, encoding="utf-8")
    resolved = load_config(path)

    assert resolved.indexer["settings"]["url"] == "http://localhost:9200"  # ${ES_URL} substituted
    assert resolved.top_k == 100
    assert resolved.cutoff == 10
    assert resolved.baseline_id == "baseline"
    assert resolved.seed == 1234
    # baseline first, then variants in insertion order.
    assert [p.id for p in resolved.pipelines()] == [
        "baseline",
        "semantic_e5",
        "hybrid_e5_k60",
        "bm25_rerank",
    ]


def test_missing_env_var_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ES_URL", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML, encoding="utf-8")
    with pytest.raises(ConfigError, match="ES_URL"):
        load_config(path)


def test_missing_required_key_raises() -> None:
    raw = _parsed()
    del raw["dataset"]
    with pytest.raises(ConfigError, match="dataset"):
        resolve_config(raw)


def test_stats_block_parsed() -> None:
    resolved = resolve_config(_parsed())
    assert resolved.stats.ci_level == pytest.approx(0.95)  # parsed, never a gate (§8.2)
    assert resolved.stats.correction == "bh"
    assert resolved.stats.test == "wilcoxon"
    assert resolved.stats.seed == 1234


# --- services registry --------------------------------------------------------------------------


def test_services_registry_resolves_typed() -> None:
    services = resolve_config(_parsed()).services
    assert isinstance(services.embedder("e5"), EmbedderCfg)
    assert isinstance(services.reranker("co-rr"), RerankerCfg)
    assert isinstance(services.searcher("bm25"), SearcherCfg)
    assert services.searcher("bm25").kind == "lexical"
    assert services.searcher("semantic_e5").kind == "vector"
    assert services.searcher("semantic_e5").embedder == "e5"


def test_embedder_cfg_carries_provider_and_settings() -> None:
    embedder = resolve_config(_parsed()).services.embedder("cohere")
    assert embedder.name == "cohere"
    assert embedder.provider == "cohere"
    assert embedder.settings["model_id"] == "embed-english-v3.0"
    assert embedder.settings["api_key"] == "co-test"  # ${COHERE_KEY} substituted


def test_reranker_cfg_carries_provider_and_top_n() -> None:
    reranker = resolve_config(_parsed()).services.reranker("co-rr")
    assert reranker.provider == "cohere"
    assert reranker.settings["top_n"] == 100  # the W <= top_n cap the runner reads at R0 (§5.4)
    assert reranker.settings["api_key"] == "co-test"  # ${COHERE_KEY} substituted


def test_unknown_embedder_provider_raises() -> None:
    raw = _parsed()
    raw["services"].append(
        {"embedder": {"name": "bad", "provider": "elasticsearch", "settings": {"model_id": "x"}}}
    )
    with pytest.raises(ConfigError, match="unknown provider"):
        resolve_config(raw)


def test_reranker_openai_provider_raises() -> None:
    # OpenAI has no reranker — a reranker configured with provider: openai must fail at load (§3.4).
    raw = _parsed()
    raw["services"].append(
        {"reranker": {"name": "bad", "provider": "openai", "settings": {"model_id": "x", "top_n": 10}}}
    )
    with pytest.raises(ConfigError, match="openai has no reranker"):
        resolve_config(raw)


def test_duplicate_service_name_raises() -> None:
    raw = _parsed()
    raw["services"].append({"searcher": {"name": "bm25", "provider": "elasticsearch", "kind": "lexical"}})
    with pytest.raises(ConfigError, match="duplicate service name"):
        resolve_config(raw)


def test_vector_searcher_without_embedder_raises() -> None:
    raw = _parsed()
    raw["services"].append({"searcher": {"name": "bad", "provider": "elasticsearch", "kind": "vector"}})
    with pytest.raises(ConfigError, match="requires an 'embedder'"):
        resolve_config(raw)


def test_vector_searcher_unknown_embedder_raises() -> None:
    raw = _parsed()
    raw["services"].append(
        {"searcher": {"name": "bad", "provider": "elasticsearch", "kind": "vector", "embedder": "nope"}}
    )
    with pytest.raises(ConfigError, match="unknown embedder"):
        resolve_config(raw)


# --- pipeline validation (§10 field rules) -----------------------------------------------------


def _with_variant(spec: dict[str, Any]) -> dict[str, Any]:
    raw = _parsed()
    raw["pipelines"]["variants"] = {"v": spec}
    return raw


def test_retriever_xor_retrievers() -> None:
    with pytest.raises(ConfigError, match="exactly one of"):
        resolve_config(_with_variant({"retriever": "bm25", "retrievers": ["bm25", "semantic_e5"]}))
    with pytest.raises(ConfigError, match="exactly one of"):
        resolve_config(_with_variant({}))


def test_retrievers_requires_fuser() -> None:
    with pytest.raises(ConfigError, match="requires a 'fuser'"):
        resolve_config(_with_variant({"retrievers": ["bm25", "semantic_e5"]}))


def test_fuser_forbidden_with_single_retriever() -> None:
    with pytest.raises(ConfigError, match="only allowed with 'retrievers'"):
        resolve_config(
            _with_variant({"retriever": "bm25", "fuser": {"type": "rrf", "rank_constant": 60, "window": 100}})
        )


def test_unknown_fuser_type_raises() -> None:
    with pytest.raises(ConfigError, match="unknown fuser type"):
        resolve_config(
            _with_variant(
                {"retrievers": ["bm25", "semantic_e5"], "fuser": {"type": "magic", "rank_constant": 1, "window": 1}}
            )
        )


def test_reranker_requires_window_and_vice_versa() -> None:
    with pytest.raises(ConfigError, match="set together"):
        resolve_config(_with_variant({"retriever": "bm25", "reranker": "co-rr"}))
    with pytest.raises(ConfigError, match="set together"):
        resolve_config(_with_variant({"retriever": "bm25", "rerank_window_size": 100}))


def test_unknown_searcher_ref_raises() -> None:
    with pytest.raises(ConfigError, match="unknown searcher"):
        resolve_config(_with_variant({"retriever": "nope"}))


def test_mistyped_service_ref_raises() -> None:
    # 'co-rr' is a reranker, not a searcher — referencing it as a retriever must fail.
    with pytest.raises(ConfigError, match="unknown searcher"):
        resolve_config(_with_variant({"retriever": "co-rr"}))


def test_unknown_reranker_ref_raises() -> None:
    with pytest.raises(ConfigError, match="unknown reranker"):
        resolve_config(_with_variant({"retriever": "bm25", "reranker": "nope", "rerank_window_size": 100}))


def test_duplicate_variant_id_vs_baseline_raises() -> None:
    raw = _parsed()
    raw["pipelines"]["variants"] = {"baseline": {"retriever": "bm25"}}
    with pytest.raises(ConfigError, match="duplicates the baseline id"):
        resolve_config(raw)


def test_resolved_pipeline_shapes() -> None:
    resolved = resolve_config(_parsed())
    by_id = {p.id: p for p in resolved.pipelines()}
    assert isinstance(by_id["baseline"], PipelineCfg)
    assert by_id["baseline"].retrievers == ("bm25",)
    assert by_id["baseline"].fuser is None
    hybrid = by_id["hybrid_e5_k60"]
    assert hybrid.retrievers == ("bm25", "semantic_e5")
    assert hybrid.fuser is not None and hybrid.fuser.rank_constant == 60
    rerank = by_id["bm25_rerank"]
    assert rerank.reranker == "co-rr"
    assert rerank.rerank_window_size == 100


# --- factory dispatch (offline, no adapter import) ---------------------------------------------


def test_dataset_registry_maps_wands_to_dotted_target() -> None:
    assert DATASET_TARGETS["wands"] == "benchmark.datasets.wands:WandsDataset"


def test_indexer_registry_maps_elasticsearch_to_dotted_target() -> None:
    assert INDEXER_TARGETS["elasticsearch"] == "benchmark.backends.elasticsearch:ElasticsearchBackend"


def test_unknown_dataset_name_raises() -> None:
    with pytest.raises(ConfigError, match="unknown dataset name"):
        load_dataset({"name": "sqamble"})


def test_unknown_indexer_provider_raises() -> None:
    with pytest.raises(ConfigError, match="unknown indexer provider"):
        make_indexer({"provider": "vespa"})


def test_unknown_searcher_factory_provider_raises() -> None:
    with pytest.raises(ConfigError, match="unknown indexer provider"):
        make_searcher_factory({"provider": "vespa"})
