"""Config load/resolve, ${VAR} substitution, ConfigInferenceModel, and factory dispatch tests (docs/experiment.md §10, §11, plan Phase 6).

All offline: no adapter module is imported (the factories dispatch to dotted-path targets and are
never resolved here). Exercises env-var substitution, missing-key errors, the EmbeddingModel
descriptor conformance, the reranker InferenceEndpoint carry-through, and dispatch (known/unknown).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from benchmark.config import (
    BACKEND_TARGETS,
    DATASET_TARGETS,
    ConfigError,
    ConfigInferenceModel,
    load_config,
    load_dataset,
    make_backend,
    make_searcher_factory,
    resolve_config,
)
from benchmark.models import InferenceEndpoint, InferenceTaskType
from benchmark.protocols import EmbeddingModel

CONFIG_YAML = textwrap.dedent(
    """
    dataset:   { name: wands, path: ./dataset/wands, version: "2022.0" }
    backend:   { kind: elasticsearch, url: "${ES_URL}", index: wands_bench,
                 top_k: 100, rank_window_size: 100, min_es_version: "8.15" }
    cutoff:    10
    embedding_models:
      - { inference_id: e5-small, service: elasticsearch, task_type: text_embedding, service_settings: {} }
      - { inference_id: openai-3-small, service: openai, task_type: text_embedding,
          service_settings: { api_key: "${OPENAI_KEY}", model_id: text-embedding-3-small } }
    rerankers:
      - { inference_id: cohere-rerank-v3, service: cohere, task_type: rerank,
          service_settings: { api_key: "${COHERE_KEY}", model_id: rerank-v3.5 },
          task_settings: { top_n: 100 } }
    rrf_k_sweep: [10,20,30,40,50,60,70,80,90,100]
    variants:  [bm25, semantic, hybrid, bm25_rerank, semantic_rerank, hybrid_rerank]
    stats:     { bootstrap_B: 10000, ci_level: 0.95, alpha: 0.05, correction: bh, test: wilcoxon,
                 wilcoxon_zero_method: wilcox, wilcoxon_correction: true, seed: 1234 }
    hybrid_rerank_k: 60
    """
)


@pytest.fixture(autouse=True)
def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ES_URL", "http://localhost:9200")
    monkeypatch.setenv("OPENAI_KEY", "sk-test")
    monkeypatch.setenv("COHERE_KEY", "co-test")


# --- load + ${VAR} substitution ----------------------------------------------------------------


def test_load_substitutes_env_placeholders(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML, encoding="utf-8")
    resolved = load_config(path)

    assert resolved.backend["url"] == "http://localhost:9200"  # ${ES_URL} substituted
    assert resolved.top_k == 100
    assert resolved.rank_window_size == 100
    assert resolved.cutoff == 10
    assert resolved.hybrid_rerank_k == 60
    assert resolved.baseline_id == "bm25"
    assert resolved.seed == 1234
    assert list(resolved.rrf_k_sweep) == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert [m.inference_id for m in resolved.embedding_models] == ["e5-small", "openai-3-small"]
    assert [r.inference_id for r in resolved.rerankers] == ["cohere-rerank-v3"]


def test_missing_env_var_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ES_URL", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML, encoding="utf-8")
    with pytest.raises(ConfigError, match="ES_URL"):
        load_config(path)


def test_missing_required_key_raises() -> None:
    raw = {"backend": {"kind": "elasticsearch", "top_k": 100, "rank_window_size": 100}}
    with pytest.raises(ConfigError, match="dataset"):
        resolve_config(raw)


def test_stats_ci_level_parsed_not_gate() -> None:
    resolved = resolve_config(_parsed_config())
    assert resolved.stats.ci_level == 0.95  # parsed, but never used as a gate (§8.2)
    assert resolved.stats.correction == "bh"
    assert resolved.stats.test == "wilcoxon"
    assert resolved.stats.wilcoxon_zero_method == "wilcox"
    assert resolved.stats.wilcoxon_correction is True


def test_best_per_model_string_preserved() -> None:
    raw = _parsed_config()
    raw["hybrid_rerank_k"] = "best_per_model"
    assert resolve_config(raw).hybrid_rerank_k == "best_per_model"


def test_bad_hybrid_rerank_k_raises() -> None:
    raw = _parsed_config()
    raw["hybrid_rerank_k"] = "nonsense"
    with pytest.raises(ConfigError):
        resolve_config(raw)


def _parsed_config() -> dict[str, object]:
    import yaml

    return dict(yaml.safe_load(CONFIG_YAML))


# --- ConfigInferenceModel implements EmbeddingModel --------------------------------------------


def test_config_inference_model_is_embedding_model() -> None:
    model = ConfigInferenceModel(
        inference_id="e5-small",
        task_type=InferenceTaskType.TEXT_EMBEDDING,
        service="elasticsearch",
    )
    assert isinstance(model, EmbeddingModel)  # structural runtime check (method + attrs present)
    endpoint = model.as_endpoint()
    assert isinstance(endpoint, InferenceEndpoint)
    assert endpoint.inference_id == "e5-small"
    assert endpoint.task_type is InferenceTaskType.TEXT_EMBEDDING


def test_config_inference_model_satisfies_embedding_model_statically() -> None:
    # mypy proves structural conformance; this binding exercises it at runtime too.
    model: EmbeddingModel = ConfigInferenceModel(
        inference_id="elser",
        task_type=InferenceTaskType.SPARSE_EMBEDDING,
        service="elasticsearch",
    )
    assert model.as_endpoint().task_type is InferenceTaskType.SPARSE_EMBEDDING


def test_reranker_endpoints_carry_top_n() -> None:
    resolved = resolve_config(_parsed_config())
    endpoint = resolved.reranker_endpoints["cohere-rerank-v3"]
    assert isinstance(endpoint, InferenceEndpoint)
    assert endpoint.task_type is InferenceTaskType.RERANK
    assert endpoint.task_settings["top_n"] == 100  # a task_settings key (§3.4/§5.3)
    assert endpoint.service_settings["api_key"] == "co-test"  # ${COHERE_KEY} substituted


# --- factory dispatch (offline, no adapter import) ---------------------------------------------


def test_dataset_registry_maps_wands_to_dotted_target() -> None:
    assert DATASET_TARGETS["wands"] == "benchmark.datasets.wands:WandsDataset"


def test_backend_registry_maps_elasticsearch_to_dotted_target() -> None:
    assert BACKEND_TARGETS["elasticsearch"] == "benchmark.backends.elasticsearch:ElasticsearchBackend"


def test_unknown_dataset_name_raises() -> None:
    with pytest.raises(ConfigError, match="unknown dataset name"):
        load_dataset({"name": "sqamble"})


def test_unknown_backend_kind_raises() -> None:
    with pytest.raises(ConfigError, match="unknown backend kind"):
        make_backend({"kind": "vespa"})


def test_unknown_searcher_factory_kind_raises() -> None:
    with pytest.raises(ConfigError, match="unknown backend kind"):
        make_searcher_factory({"kind": "vespa"})
