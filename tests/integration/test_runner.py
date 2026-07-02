"""Live-ES end-to-end runner + eval:index tests (docs/experiment.md §8.0, plan Phase 11).

Marked ``integration``; SKIPS (never fails) when ES is unreachable OR on a 429 model-deploy error
(reusing the ``_skip_if_deploy_error`` pattern from ``test_es_backend.py``). Uses ONLY local ES
inference — NO cohere, NO api keys: an embedder ``e5`` (``.multilingual-e5-small``) and a reranker
service whose name IS ``.rerank-v1-elasticsearch`` (so ``register_inference`` reuses that
preconfigured endpoint). The dataset points at the tiny ``tests/fixtures/wands_sample`` corpus.

DO NOT run this live — model deploy is slow/flaky; the human live-validates. It is written to be
correct: an end-to-end ``run`` asserting all three CSV types + run_config (baseline first, one
comparison per variant), and an ``eval:index`` build asserting a populated index with one
``semantic_text`` field per embedder + a non-zero doc count.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterator

import pytest
from elasticsearch import ApiError, Elasticsearch
from elasticsearch.helpers import BulkIndexError

from benchmark.config import resolve_config
from benchmark.runner import ExperimentRunner

from tests.conftest import WANDS_SAMPLE_DIR

pytestmark = pytest.mark.integration

ES_URL = os.environ.get("ES_URL", "http://localhost:9200")

# Preconfigured local endpoints in the validation container (NO api keys): E5 dense embedder + the
# rerank-v1 reranker. The reranker SERVICE name IS the preconfigured endpoint id so register_inference
# reuses it (top_n 100 as a task setting, so W=100 satisfies W <= top_n).
E5_MODEL_ID = ".multilingual-e5-small"
E5_INFERENCE_ID = ".multilingual-e5-small-elasticsearch"
RERANK_INFERENCE_ID = ".rerank-v1-elasticsearch"


def _skip_if_deploy_error(exc: ApiError | BulkIndexError) -> None:
    """Skip (not fail) on a 429 model-deploy/memory error; re-raise anything else (see test_es_backend)."""
    if isinstance(exc, BulkIndexError):
        if any(next(iter(item.values())).get("status") == 429 for item in exc.errors):
            pytest.skip(f"model could not be deployed (429 on ingest): {exc.errors[:1]}")
        raise exc
    if exc.status_code == 429:
        pytest.skip(f"model could not be deployed (429, insufficient ML memory): {exc}")
    raise exc


def _trimmed_config(index_name: str) -> dict:
    """A trimmed §10 config — local ES inference only, a few named pipelines, the sample corpus."""
    return {
        "dataset": {"name": "wands", "path": str(WANDS_SAMPLE_DIR)},
        "services": [
            {
                "embedder": {
                    "name": E5_INFERENCE_ID,
                    "provider": "elasticsearch",
                    "embedding_type": "text_embedding",
                    "settings": {"model_id": E5_MODEL_ID},
                }
            },
            {
                "reranker": {
                    "name": RERANK_INFERENCE_ID,
                    "provider": "elasticsearch",
                    "settings": {"top_n": 100},
                }
            },
            {"searcher": {"name": "bm25", "provider": "elasticsearch", "kind": "lexical"}},
            {
                "searcher": {
                    "name": "semantic_e5",
                    "provider": "elasticsearch",
                    "kind": "vector",
                    "embedder": E5_INFERENCE_ID,
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
                "semantic_e5": {"retriever": "semantic_e5"},
                "hybrid_e5": {
                    "retrievers": ["bm25", "semantic_e5"],
                    "fuser": {"type": "rrf", "rank_constant": 60, "window": 100},
                },
                "bm25_rerank": {
                    "retriever": "bm25",
                    "reranker": RERANK_INFERENCE_ID,
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
    cfg = resolve_config(_trimmed_config(es_index))
    try:
        _dataset, backend, mapping = ExperimentRunner().build_index(cfg)
    except (ApiError, BulkIndexError) as exc:
        _skip_if_deploy_error(exc)

    # One semantic_text field per configured embedder (§5.2), inference_id set on it.
    assert mapping.sem_field(E5_INFERENCE_ID) in mapping.backend_mapping["properties"]
    sem_prop = mapping.backend_mapping["properties"][mapping.sem_field(E5_INFERENCE_ID)]
    assert sem_prop["type"] == "semantic_text"
    assert sem_prop["inference_id"] == E5_INFERENCE_ID
    # Non-zero doc count (the sample corpus landed).
    count = backend.client.count(index=mapping.index_name)["count"]
    assert count > 0


def test_run_end_to_end_produces_all_artifacts(es_index: str, tmp_path: Path) -> None:
    cfg = resolve_config(_trimmed_config(es_index))
    try:
        ExperimentRunner().run(cfg, output_dir=str(tmp_path))
    except (ApiError, BulkIndexError) as exc:
        _skip_if_deploy_error(exc)

    ts = cfg.timestamp
    base = cfg.baseline_id  # the baseline pipeline's artifact id (config.yaml default is "baseline", §9:857)
    # All four pipelines produced result + metrics CSVs (single run_one path, baseline first).
    for pipeline_id in (base, "semantic_e5", "hybrid_e5", "bm25_rerank"):
        assert (tmp_path / f"result_{pipeline_id}_{ts}.csv").exists()
        assert (tmp_path / f"metrics_{pipeline_id}_{ts}.csv").exists()
    # One comparison per VARIANT — never baseline vs itself.
    assert not (tmp_path / f"comparison_{base}_{base}_{ts}.csv").exists()
    for variant in ("semantic_e5", "hybrid_e5", "bm25_rerank"):
        assert (tmp_path / f"comparison_{base}_{variant}_{ts}.csv").exists()
    assert (tmp_path / f"run_config_{ts}.json").exists()
