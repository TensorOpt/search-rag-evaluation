"""Live end-to-end runner + eval:index tests (docs/experiment.md §8.0, plan Phase 11).

Marked ``integration``; SKIPS (never fails) when ES is unreachable, when ``COHERE_KEY`` is absent, or
on a provider ``ProviderError`` (an env constraint — auth/rate limit). ES is a plain vector/BM25 index
(§1.1): the harness embeds the sample corpus via the Cohere connector into a ``dense_vector`` field
and reranks via the Cohere connector — NO ES ``_inference``. The dataset points at the tiny
``tests/fixtures/wands_sample`` corpus.

DO NOT run this live casually — it makes real Cohere API calls. It is written to be correct: an
end-to-end ``run`` asserting all three CSV types + run_config (baseline first, one comparison per
variant), and an ``eval:index`` build asserting a populated index with one ``dense_vector`` field per
embedder + a non-zero doc count.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterator

import pytest
from elasticsearch import Elasticsearch

from benchmark.config import resolve_config
from benchmark.providers.inference import ProviderError
from benchmark.runner import ExperimentRunner

from tests.conftest import WANDS_SAMPLE_DIR

pytestmark = pytest.mark.integration

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
COHERE_KEY = os.environ.get("COHERE_KEY")


def _require_cohere() -> str:
    if not COHERE_KEY:
        pytest.skip("COHERE_KEY not set — the live runner test embeds/reranks via Cohere")
    return COHERE_KEY


def _trimmed_config(index_name: str, api_key: str) -> dict:
    """A trimmed §10 config — Cohere embed + rerank connectors, a few named pipelines, sample corpus."""
    return {
        "dataset": {"name": "wands", "path": str(WANDS_SAMPLE_DIR)},
        "services": [
            {
                "embedder": {
                    "name": "cohere",
                    "provider": "cohere",
                    "settings": {"api_key": api_key, "model_id": "embed-english-v3.0"},
                }
            },
            {
                "reranker": {
                    "name": "co-rr",
                    "provider": "cohere",
                    "settings": {"api_key": api_key, "model_id": "rerank-v3.5", "top_n": 100},
                }
            },
            {"searcher": {"name": "bm25", "provider": "elasticsearch", "kind": "lexical"}},
            {
                "searcher": {
                    "name": "semantic_co",
                    "provider": "elasticsearch",
                    "kind": "vector",
                    "embedder": "cohere",
                }
            },
        ],
        "indexer": {
            "provider": "elasticsearch",
            "index": index_name,
            "settings": {"url": ES_URL},
        },
        "pipelines": {
            "baseline": {"retriever": "bm25"},
            "variants": {
                "semantic_co": {"retriever": "semantic_co"},
                "hybrid_co": {
                    "retrievers": ["bm25", "semantic_co"],
                    "fuser": {"type": "rrf", "rank_constant": 60, "window": 100},
                },
                "bm25_rerank": {
                    "retriever": "bm25",
                    "reranker": "co-rr",
                    "rerank_window_size": 100,
                },
            },
        },
        "stats": {"seed": 1234, "bootstrap_B": 1000},
        "cutoff": 10,
        "top_k": 100,
    }


@pytest.fixture
def es_index() -> Iterator[str]:
    """A unique throwaway index name; skip if ES is unreachable; deleted on teardown."""
    client = Elasticsearch(ES_URL)
    try:
        if not client.ping():
            pytest.skip(f"ES not reachable at {ES_URL}")
    except Exception as exc:  # noqa: BLE001 - any transport failure -> skip, not fail
        pytest.skip(f"ES not reachable at {ES_URL}: {exc}")
    index_name = f"runner_it_{uuid.uuid4().hex}"
    try:
        yield index_name
    finally:
        client.indices.delete(index=index_name, ignore_unavailable=True)


def test_eval_index_builds_populated_index(es_index: str) -> None:
    cfg = resolve_config(_trimmed_config(es_index, _require_cohere()))
    try:
        _dataset, backend, mapping, _embedders = ExperimentRunner().build_index(cfg)
    except ProviderError as exc:
        pytest.skip(f"cohere embedding unavailable (env constraint): {exc}")

    # One dense_vector field per configured embedder (§5.2), dims discovered from the connector.
    sem_field = mapping.sem_field("cohere")
    sem_prop = mapping.backend_mapping["properties"][sem_field]
    assert sem_prop["type"] == "dense_vector"
    assert sem_prop["dims"] > 0
    assert sem_prop["similarity"] == "cosine"
    # Non-zero doc count (the sample corpus landed, embedded at ingest).
    count = backend.client.count(index=mapping.index_name)["count"]
    assert count > 0


def test_run_end_to_end_produces_all_artifacts(es_index: str, tmp_path: Path) -> None:
    cfg = resolve_config(_trimmed_config(es_index, _require_cohere()))
    runner = ExperimentRunner()
    try:
        runner.build_index(cfg)  # eval:index first — eval:run REQUIRES a pre-built index (§8.0)
        runner.run(cfg, output_dir=str(tmp_path))
    except ProviderError as exc:
        pytest.skip(f"cohere embedding/rerank unavailable (env constraint): {exc}")

    ts = cfg.timestamp
    base = cfg.baseline_id  # the baseline pipeline's artifact id ("baseline" by default, §9)
    # Exactly three single per-run CSVs (all pipelines / comparisons) + run_config (§9).
    result_file = tmp_path / f"result_{ts}.csv"
    metrics_file = tmp_path / f"metrics_{ts}.csv"
    comparison_file = tmp_path / f"comparison_{ts}.csv"
    for path in (result_file, metrics_file, comparison_file, tmp_path / f"run_config_{ts}.json"):
        assert path.exists()

    def _variant_column(path: Path) -> set[str]:
        rows = path.read_text(encoding="utf-8").splitlines()[1:]
        return {line.split(",")[0] for line in rows}

    # All four pipelines (baseline included) appear in the result/metrics variant column.
    all_pipelines = {base, "semantic_co", "hybrid_co", "bm25_rerank"}
    assert _variant_column(result_file) == all_pipelines
    assert _variant_column(metrics_file) == all_pipelines

    # Comparison: baseline col constant; variant col is the variants only — never baseline vs itself.
    comparison_rows = comparison_file.read_text(encoding="utf-8").splitlines()[1:]
    assert {line.split(",")[0] for line in comparison_rows} == {base}
    assert {line.split(",")[1] for line in comparison_rows} == {"semantic_co", "hybrid_co", "bm25_rerank"}
