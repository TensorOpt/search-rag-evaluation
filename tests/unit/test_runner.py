"""Offline ExperimentRunner tests (docs/experiment.md §8.0, plan Phase 11).

Drives :class:`benchmark.runner.ExperimentRunner` with FAKES — NO ES, NO network. The lazy config
factories (``load_dataset``/``make_indexer``/``make_searcher_factory``) are monkeypatched to return a
tiny in-memory :class:`Dataset`, a fake ingest backend, and a fake ``SearcherFactory`` whose
lexical/vector leaves are conftest :class:`FakeSearcher` and whose reranker is a conftest
:class:`FakeReranker`. The real ``ESIndexer().build`` runs against the fake backend (so the §3.5
register→ensure→bulk sequence is exercised, not stubbed).

Asserts: every pipeline traverses the single ``run_one`` path (all produce artifacts); all three CSV
types + run_config land in the tmp output dir; the baseline is written first and is NOT in the
comparison outputs; the R0 assert raises when a reranker top_n < rerank_window_size; ``--dry-run``
writes nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import pytest

import benchmark.config as config
from benchmark.config import (
    EmbedderCfg,
    FuserCfg,
    PipelineCfg,
    RerankerCfg,
    ResolvedConfig,
    SearcherCfg,
    Services,
)
from benchmark.models import (
    Document,
    FieldRole,
    FieldSchema,
    FieldSpec,
    Qrel,
    Query,
    ScoredDoc,
)
from benchmark.protocols import Dataset, Reranker, Searcher
from benchmark.stats import StatsCfg

from tests.conftest import FakeReranker, FakeSearcher

# --- tiny in-memory dataset -------------------------------------------------------------------

_SCHEMA = FieldSchema(
    fields=[
        FieldSpec("product_id", FieldRole.ID),
        FieldSpec("product_name", FieldRole.SEMANTIC_SOURCE),
        FieldSpec("product_features", FieldRole.BM25),
    ]
)

_QUERIES = [Query("q1", "chair"), Query("q2", "table")]
_DOCS = [
    Document("d1", {"search_text": "office chair"}),
    Document("d2", {"search_text": "dining table"}),
    Document("d3", {"search_text": "kitchen stool"}),
    Document("d4", {"search_text": "coffee table"}),
]
# q1 and q2 each judge every returned doc so the metrics are non-NaN and the comparator has a real
# paired set for the family-wide FDR pass (d1..d4 are all returned by the fake searchers).
_QRELS = [
    Qrel("q1", "d1", 1.0),
    Qrel("q1", "d2", 0.5),
    Qrel("q1", "d3", 0.0),
    Qrel("q1", "d4", 0.5),
    Qrel("q2", "d2", 1.0),
    Qrel("q2", "d4", 0.5),
    Qrel("q2", "d1", 0.0),
    Qrel("q2", "d3", 0.0),
]


class FakeDataset(Dataset):
    """A tiny in-memory dataset (real ``Dataset`` so ``field_schema`` feeds the real ``ESIndexer``)."""

    def __init__(self, dataset_cfg: Any = None) -> None:
        self.name = "fake"
        self.version = "0"

    def queries(self) -> Iterable[Query]:
        return list(_QUERIES)

    def documents(self) -> Iterable[Document]:
        return list(_DOCS)

    def qrels(self) -> Iterable[Qrel]:
        return list(_QRELS)

    def field_schema(self) -> FieldSchema:
        return _SCHEMA


# --- fake ingest backend + searcher factory ---------------------------------------------------


class FakeBackend:
    """A fake ``SearchBackend`` recording the §3.5 ingest calls (no ES). ``.client``/``.index`` are
    present so ``eval:index``-style probing works. ES is a plain index writer now — no inference
    registration; the harness embeds the corpus upstream and hands already-embedded documents here.
    """

    def __init__(self, indexer_cfg: Any = None) -> None:
        self.index = "fake_index"
        self.ensured = False
        self.indexed: list[Document] = []

    def ensure_index(self, mapping: Any) -> None:
        self.ensured = True

    def bulk_index(self, docs: Iterable[Document], *, mapping: Any) -> None:
        self.indexed = list(docs)


class FakeEmbedder:
    """A fake embedding connector: fixed-dim canned vectors (no network). ``id``/``dim`` drive the
    real ``ESIndexer`` mapping; ``embed_documents`` is called at ingest, ``embed_queries`` never here
    (the FakeFactory's vector leaf is a canned ``FakeSearcher`` that ignores the embedder)."""

    def __init__(self, name: str, dim: int = 3) -> None:
        self.id = name
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(i)] * self._dim for i, _ in enumerate(texts)]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * self._dim for _ in texts]


class FakeRerankClient:
    """A fake rerank connector: canned descending scores aligned to input (no network)."""

    def rerank_scores(self, query: str, documents: Sequence[str]) -> list[float]:
        return [float(len(documents) - i) for i in range(len(documents))]


# Canned per-leaf lists: every returned doc is judged so metrics are non-NaN. Lexical and vector
# differ in order so the fused/reranked outputs are observable.
_LEXICAL_DOCS = [ScoredDoc("d1", 5.0), ScoredDoc("d2", 4.0), ScoredDoc("d3", 3.0), ScoredDoc("d4", 2.0)]
_VECTOR_DOCS = [ScoredDoc("d2", 0.9), ScoredDoc("d4", 0.8), ScoredDoc("d1", 0.7), ScoredDoc("d3", 0.6)]


class FakeFactory:
    """A fake ``SearcherFactory`` (§4): lexical/vector -> conftest ``FakeSearcher``, reranker ->
    conftest ``FakeReranker``. Uses the ABC's default per-query ``bulk_search`` (no ES batching)."""

    def lexical(self, *, fields: Sequence[str]) -> Searcher:
        return FakeSearcher(_LEXICAL_DOCS)

    def vector(self, *, field: str, embedder_id: str) -> Searcher:
        return FakeSearcher(_VECTOR_DOCS)

    def reranker(self, name: str, field: str) -> Reranker:
        return FakeReranker()


# --- resolved-config builder ------------------------------------------------------------------


def _services(*, reranker_top_n: int = 100) -> Services:
    return Services(
        embedders={
            "e5": EmbedderCfg("e5", "cohere", {"api_key": "k", "model_id": "embed-english-v3.0"})
        },
        rerankers={
            "rr": RerankerCfg("rr", "cohere", {"api_key": "k", "model_id": "rerank-v3.5", "top_n": reranker_top_n})
        },
        searchers={
            "bm25": SearcherCfg("bm25", "elasticsearch", "lexical", None),
            "semantic_e5": SearcherCfg("semantic_e5", "elasticsearch", "vector", "e5"),
        },
    )


def _config(
    *,
    variants: Sequence[PipelineCfg],
    services: Services | None = None,
    timestamp: str = "20260702T000000Z",
) -> ResolvedConfig:
    baseline = PipelineCfg(id="bm25", retrievers=("bm25",), fuser=None, reranker=None, rerank_window_size=None)
    return ResolvedConfig(
        dataset={"name": "fake"},
        indexer={"provider": "elasticsearch", "index": "fake_index"},
        services=services or _services(),
        baseline=baseline,
        variants=list(variants),
        stats=StatsCfg(bootstrap_B=200, seed=7),
        cutoff=10,
        top_k=100,
        baseline_id="bm25",
        timestamp=timestamp,
        seed=7,
    )


def patch_runner_factories(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    """Point the runner's ``config`` factories at the in-memory fakes; return the fake backend.

    Reused by the ``patched_factories`` fixture AND the schema-lint / reproducibility tests (one
    patching path). ``make_index_builder`` returns the REAL :class:`ESIndexer` (run against the fake
    backend), so the §3.5 register→ensure→bulk sequence is exercised, not stubbed — matching the
    pre-Phase-12 flow where the runner imported ``ESIndexer`` directly. All four factories are patched
    on ``config`` so no adapter is imported/instantiated live.
    """
    from benchmark.backends.elasticsearch import ESIndexer

    backend = FakeBackend()
    monkeypatch.setattr(config, "load_dataset", lambda dataset_cfg: FakeDataset())
    monkeypatch.setattr(config, "make_indexer", lambda indexer_cfg: backend)
    monkeypatch.setattr(config, "make_index_builder", lambda indexer_cfg: ESIndexer())
    # Embedder/reranker connectors are FAKES (no network): the real ESIndexer embeds the corpus with
    # these at ingest; the FakeFactory's vector leaf is canned, so it ignores the query embedder.
    monkeypatch.setattr(
        config, "make_embedders", lambda services: {name: FakeEmbedder(name) for name in services.embedders}
    )
    monkeypatch.setattr(
        config, "make_rerankers", lambda services: {name: FakeRerankClient() for name in services.rerankers}
    )
    monkeypatch.setattr(config, "make_searcher_factory", lambda indexer_cfg, **kwargs: FakeFactory())
    return backend


@pytest.fixture
def patched_factories(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    """Fixture wrapper over :func:`patch_runner_factories` (return the fake ingest backend)."""
    return patch_runner_factories(monkeypatch)


def _artifacts(output_dir: Path, prefix: str, timestamp: str) -> list[str]:
    return sorted(p.name for p in output_dir.glob(f"{prefix}_*_{timestamp}.csv"))


# --- tests ------------------------------------------------------------------------------------


def test_run_produces_all_artifacts_baseline_first(
    patched_factories: FakeBackend, tmp_path: Path
) -> None:
    from benchmark.runner import ExperimentRunner

    ts = "20260702T000000Z"
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

    # Every pipeline (baseline + 3 variants) produced a result + metrics CSV -> single run_one path.
    assert _artifacts(tmp_path, "result", ts) == [
        f"result_bm25_{ts}.csv",
        f"result_bm25_rerank_{ts}.csv",
        f"result_hybrid_e5_{ts}.csv",
        f"result_semantic_e5_{ts}.csv",
    ]
    assert _artifacts(tmp_path, "metrics", ts) == [
        f"metrics_bm25_{ts}.csv",
        f"metrics_bm25_rerank_{ts}.csv",
        f"metrics_hybrid_e5_{ts}.csv",
        f"metrics_semantic_e5_{ts}.csv",
    ]
    # One comparison per VARIANT — the baseline is NEVER compared to itself.
    comparisons = sorted(p.name for p in tmp_path.glob(f"comparison_*_{ts}.csv"))
    assert comparisons == [
        f"comparison_bm25_bm25_rerank_{ts}.csv",
        f"comparison_bm25_hybrid_e5_{ts}.csv",
        f"comparison_bm25_semantic_e5_{ts}.csv",
    ]
    assert not (tmp_path / f"comparison_bm25_bm25_{ts}.csv").exists()  # baseline not vs itself
    assert (tmp_path / f"run_config_{ts}.json").exists()

    # The ingest seam ran (ensure_index + streamed bulk_index) and the corpus was embedded at
    # ingest: every indexed doc carries the embedder's dense_vector field (§3.5). No inference
    # registration happens (ES is a plain index); the reranker 'rr' is a connector, not registered.
    assert patched_factories.ensured is True
    assert len(patched_factories.indexed) == len(_DOCS)
    assert all("sem__e5" in doc.fields for doc in patched_factories.indexed)


def test_run_result_csv_is_first_written_for_baseline(
    patched_factories: FakeBackend, tmp_path: Path
) -> None:
    from benchmark.runner import ExperimentRunner

    ts = "20260702T010000Z"
    cfg = _config(variants=[PipelineCfg("semantic_e5", ("semantic_e5",), None, None, None)], timestamp=ts)

    ExperimentRunner().run(cfg, output_dir=str(tmp_path))

    # Baseline result file is written before the variant's (mtime ordering, single path, baseline-first).
    baseline_result = tmp_path / f"result_bm25_{ts}.csv"
    variant_result = tmp_path / f"result_semantic_e5_{ts}.csv"
    assert baseline_result.stat().st_mtime_ns <= variant_result.stat().st_mtime_ns
    # Baseline result content reflects the lexical leaf (d1 first, 4 rows/query).
    header, *rows = baseline_result.read_text().splitlines()
    assert header == "query_id,product_id,score,position"
    assert rows[0].startswith("q1,d1,")


def test_r0_asserts_when_rerank_window_exceeds_top_n(
    patched_factories: FakeBackend, tmp_path: Path
) -> None:
    from benchmark.runner import ExperimentRunner

    # top_n 50 < rerank_window_size 100 -> the R0 W <= top_n check must raise (§5.3).
    cfg = _config(
        variants=[PipelineCfg("bm25_rerank", ("bm25",), None, "rr", 100)],
        services=_services(reranker_top_n=50),
        timestamp="20260702T020000Z",
    )
    with pytest.raises(ValueError, match="top_n"):
        ExperimentRunner().run(cfg, output_dir=str(tmp_path))


def test_build_index_reuses_single_ingest_path(
    patched_factories: FakeBackend, tmp_path: Path
) -> None:
    from benchmark.runner import ExperimentRunner

    cfg = _config(variants=[], timestamp="20260702T030000Z")
    dataset, backend, mapping, embedders = ExperimentRunner().build_index(cfg)

    assert isinstance(dataset, FakeDataset)
    assert backend is patched_factories
    # One dense_vector field per embedder (§5.2); doc _id-keyed ingest happened.
    assert mapping.sem_fields == {"e5": "sem__e5"}
    assert set(embedders) == {"e5"}  # the embedder connector registry the runner reuses (§8.0)
    assert backend.ensured is True


def test_dry_run_writes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import scripts.run as run_script

    # A minimal config file on disk; --dry-run must not touch ES or write artifacts.
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(_DRY_RUN_CONFIG_YAML, encoding="utf-8")

    # Guard: if the runner were invoked it would blow up here (proves dry-run short-circuits).
    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("--dry-run must not construct or run the ExperimentRunner")

    monkeypatch.setattr(run_script.ExperimentRunner, "run", _boom)

    rc = run_script.main(["--config", str(cfg_path), "--output-dir", str(tmp_path), "--dry-run"])

    assert rc == 0
    assert list(tmp_path.glob("result_*.csv")) == []
    assert list(tmp_path.glob("metrics_*.csv")) == []
    assert list(tmp_path.glob("comparison_*.csv")) == []
    assert list(tmp_path.glob("run_config_*.json")) == []


_DRY_RUN_CONFIG_YAML = """\
dataset: { name: wands, path: ./dataset/wands }
services:
  - embedder: { name: e5, provider: cohere, settings: { api_key: x, model_id: x } }
  - searcher: { name: bm25, provider: elasticsearch, kind: lexical }
  - searcher: { name: semantic_e5, provider: elasticsearch, kind: vector, embedder: e5 }
indexer: { provider: elasticsearch, index: wands_bench, settings: { url: "http://localhost:9200" } }
pipelines:
  baseline: { retriever: bm25 }
  variants:
    semantic_e5: { retriever: semantic_e5 }
stats: { seed: 1234 }
cutoff: 10
top_k: 100
"""
