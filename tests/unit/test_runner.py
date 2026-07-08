"""Offline ExperimentRunner tests (docs/experiment.md §8.0, plan Phase 11).

Drives :class:`benchmark.runner.ExperimentRunner` with FAKES — NO ES, NO network. The lazy config
factories (``load_dataset`` / ``make_index_writer`` / ``make_embedders`` / ``make_rerankers`` /
``make_searchers`` / ``make_rerankers_bound``) are monkeypatched on ``config`` to return a tiny
in-memory :class:`Dataset`, a fake :class:`IndexWriter`, fake embedder/rerank connectors, and fake
``{name: Searcher}`` / ``{name: Reranker}`` leaf maps (conftest :class:`FakeSearcher` /
:class:`FakeReranker`). The real domain ``indexing.Indexer(writer, embedders).build`` runs against the
fake writer (so the §3.5 ensure→embed→bulk sequence is exercised, not stubbed).

Asserts: every pipeline traverses the single ``run_one`` path (all rows land in the single result +
metrics files); all three CSV types + run_config land in the tmp output dir; the baseline appears
first in the ``variant`` column and is NOT compared to itself in the comparison file; the R0 assert
raises when a reranker top_n < rerank_window_size; ``--dry-run`` writes nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pytest

import benchmark.config as config
from benchmark.common.models import (
    Document,
    FieldRole,
    FieldSchema,
    FieldSpec,
    IndexMapping,
    Qrel,
    Query,
    ScoredDoc,
)
from benchmark.common.protocols import Dataset, Embedder, IndexWriter, RerankClient
from benchmark.config import (
    EmbedderCfg,
    FuserCfg,
    PipelineCfg,
    RerankerCfg,
    ResolvedConfig,
    SearcherCfg,
    Services,
)
from benchmark.evaluation.stats import Contrast, StatsCfg

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
    """A tiny in-memory dataset (real ``Dataset`` so ``field_schema`` feeds the real ``Indexer``)."""

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


# --- fake index writer + provider connectors --------------------------------------------------


class FakeIndexWriter(IndexWriter):
    """A fake ``IndexWriter`` recording the §3.5 ingest calls (no ES). ``.client``/``.index`` are
    present so ``eval:index``-style probing works. ES is a plain index writer now — no inference
    registration; the harness embeds the corpus upstream and hands already-embedded documents here.
    ``sem_field_name``/``create_mapping`` let the real domain ``Indexer`` name fields + build a
    real ``IndexMapping`` against this fake.
    """

    embed_batch_size = 96

    def __init__(self, indexer_cfg: Any = None) -> None:
        self.index = "fake_index"
        self.ensured = False
        self.indexed: list[Document] = []
        # doc_count() default == the FakeDataset corpus size, so run()'s "fully indexed" check
        # (index count == dataset count, §8.0) passes; the failure tests set it to None/other.
        self.doc_count_value: int | None = len(_DOCS)

    def doc_count(self) -> int | None:
        return self.doc_count_value

    def sem_field_name(self, embedder_id: str) -> str:
        return "sem__" + embedder_id

    def create_mapping(
        self, schema: FieldSchema, sem_fields: Mapping[str, str], vector_dims: Mapping[str, int]
    ) -> IndexMapping:
        return IndexMapping(
            index_name=self.index,
            search_text_field=schema.search_text_field,
            sem_fields=dict(sem_fields),
            backend_mapping={"properties": {}},
        )

    def ensure_index(self, mapping: Any) -> None:
        self.ensured = True

    def bulk_index(self, docs: Iterable[Document], *, mapping: Any) -> None:
        self.indexed = list(docs)


class FakeEmbedder(Embedder):
    """A fake embedding connector: fixed-dim canned vectors (no network). ``id``/``dim`` drive the
    real ``Indexer`` mapping; ``embed_documents`` is called at ingest, ``embed_queries`` never here
    (the fake vector leaf is a canned ``FakeSearcher`` that ignores the embedder)."""

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


class FakeRerankClient(RerankClient):
    """A fake rerank connector: canned descending scores aligned to input (no network)."""

    def rerank_scores(self, query: str, documents: Sequence[str]) -> list[float]:
        return [float(len(documents) - i) for i in range(len(documents))]


# Canned per-leaf lists: every returned doc is judged so metrics are non-NaN. Lexical and vector
# differ in order so the fused/reranked outputs are observable.
_LEXICAL_DOCS = [ScoredDoc("d1", 5.0), ScoredDoc("d2", 4.0), ScoredDoc("d3", 3.0), ScoredDoc("d4", 2.0)]
_VECTOR_DOCS = [ScoredDoc("d2", 0.9), ScoredDoc("d4", 0.8), ScoredDoc("d1", 0.7), ScoredDoc("d3", 0.6)]


def _fake_searchers(
    indexer_cfg: Any, mapping: Any, services: Services, *, embedders: Mapping[str, Any], cache: Any = None
) -> dict[str, FakeSearcher]:
    """Stand-in for ``build_searchers``: one conftest ``FakeSearcher`` per configured searcher.

    Lexical -> the lexical canned list, vector -> the vector canned list (so the fused/reranked
    outputs differ). Uses the ABC's default per-query ``bulk_search`` (no ES batching)."""
    out: dict[str, FakeSearcher] = {}
    for name, searcher_cfg in services.searchers.items():
        docs = _LEXICAL_DOCS if searcher_cfg.kind == "lexical" else _VECTOR_DOCS
        out[name] = FakeSearcher(docs)
    return out


def _fake_rerankers_bound(
    indexer_cfg: Any, mapping: Any, services: Services, *, rerank_clients: Mapping[str, Any]
) -> dict[str, FakeReranker]:
    """Stand-in for ``build_rerankers``: one conftest ``FakeReranker`` per configured reranker."""
    return {name: FakeReranker() for name in services.rerankers}


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
    # Build ResolvedConfig directly (bypassing resolve_config), so synthesize the default every-
    # variant-vs-baseline contrasts here the way resolve_config would (§10, Fix 3).
    contrasts = tuple(Contrast(a=v.id, b="bm25", family=True) for v in variants)
    return ResolvedConfig(
        dataset={"name": "fake"},
        indexer={"provider": "elasticsearch", "index": "fake_index"},
        services=services or _services(),
        baseline=baseline,
        variants=list(variants),
        stats=StatsCfg(bootstrap_B=200, seed=7, contrasts=contrasts),
        cutoff=10,
        top_k=100,
        baseline_id="bm25",
        timestamp=timestamp,
        seed=7,
    )


def patch_runner_factories(monkeypatch: pytest.MonkeyPatch) -> FakeIndexWriter:
    """Point the runner's ``config`` factories at the in-memory fakes; return the fake index writer.

    Reused by the ``patched_factories`` fixture AND the schema-lint / reproducibility tests (one
    patching path). ``make_index_writer`` returns a fake :class:`IndexWriter`; the runner drives the
    REAL :class:`~benchmark.indexing.Indexer` over it (so the §3.5 ensure→embed→bulk sequence is
    exercised, not stubbed). ``make_searchers`` / ``make_rerankers_bound`` return the fake leaf maps
    (bypassing the deleted ``_ESSearcherFactory``); all factories are patched on ``config`` so no
    adapter is imported/instantiated live.
    """
    writer = FakeIndexWriter()
    monkeypatch.setattr(config, "load_dataset", lambda dataset_cfg: FakeDataset())
    monkeypatch.setattr(config, "make_index_writer", lambda indexer_cfg: writer)
    # Embedder/reranker connectors are FAKES (no network): the real Indexer embeds the corpus with
    # these at ingest; the fake vector leaf is canned, so it ignores the query embedder.
    monkeypatch.setattr(
        config,
        "make_embedders",
        lambda services, *, cache=None: {name: FakeEmbedder(name) for name in services.embedders},
    )
    monkeypatch.setattr(
        config,
        "make_rerankers",
        lambda services, *, cache=None: {name: FakeRerankClient() for name in services.rerankers},
    )
    monkeypatch.setattr(config, "make_searchers", _fake_searchers)
    monkeypatch.setattr(config, "make_rerankers_bound", _fake_rerankers_bound)
    return writer


@pytest.fixture
def patched_factories(monkeypatch: pytest.MonkeyPatch) -> FakeIndexWriter:
    """Fixture wrapper over :func:`patch_runner_factories` (return the fake index writer)."""
    return patch_runner_factories(monkeypatch)


def _column(path: Path, index: int) -> list[str]:
    """The values of column ``index`` for every data row (header skipped)."""
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    return [line.split(",")[index] for line in lines]


# --- tests ------------------------------------------------------------------------------------


def test_run_produces_all_artifacts_baseline_first(
    patched_factories: FakeIndexWriter, tmp_path: Path
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

    # Exactly three per-run CSVs (single files) + run_config — no per-variant files.
    result_file = tmp_path / f"result_{ts}.csv"
    metrics_file = tmp_path / f"metrics_{ts}.csv"
    comparison_file = tmp_path / f"comparison_{ts}.csv"
    assert sorted(p.name for p in tmp_path.glob("*.csv")) == [
        f"comparison_{ts}.csv",
        f"metrics_{ts}.csv",
        f"result_{ts}.csv",
    ]
    assert (tmp_path / f"run_config_{ts}.json").exists()

    # Every pipeline (baseline first, then variants in config order) appears in the variant column.
    def _variant_order(path: Path) -> list[str]:
        seen: list[str] = []
        for v in _column(path, 0):
            if v not in seen:
                seen.append(v)
        return seen

    assert _variant_order(result_file) == ["bm25", "semantic_e5", "hybrid_e5", "bm25_rerank"]
    assert _variant_order(metrics_file) == ["bm25", "semantic_e5", "hybrid_e5", "bm25_rerank"]

    # Comparison: system_b col constant == bm25 (every default contrast is variant-vs-baseline);
    # one row per (contrast, metric); NO baseline-vs-itself.
    system_a_col = _column(comparison_file, 0)
    system_b_col = _column(comparison_file, 1)
    assert set(system_b_col) == {"bm25"}
    assert "bm25" not in system_a_col  # baseline never on the a side of a default contrast
    assert len(system_a_col) == 3 * 6  # 3 contrasts x 6 canonical metrics

    # eval:run does NOT index — the ingest seam is never exercised here (that's eval:index's job).
    assert patched_factories.ensured is False
    assert patched_factories.indexed == []


def test_run_result_file_lists_baseline_first(
    patched_factories: FakeIndexWriter, tmp_path: Path
) -> None:
    from benchmark.runner import ExperimentRunner

    ts = "20260702T010000Z"
    cfg = _config(variants=[PipelineCfg("semantic_e5", ("semantic_e5",), None, None, None)], timestamp=ts)

    ExperimentRunner().run(cfg, output_dir=str(tmp_path))

    result_file = tmp_path / f"result_{ts}.csv"
    header, *rows = result_file.read_text().splitlines()
    assert header == "variant,query_id,product_id,score,position"
    # Baseline rows come first (single path, baseline-first) and reflect the lexical leaf (d1 first).
    assert rows[0] == "bm25,q1,d1,5.0,1"
    assert _column(result_file, 0)[0] == "bm25"


def test_r0_asserts_when_rerank_window_exceeds_top_n(
    patched_factories: FakeIndexWriter, tmp_path: Path
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


def test_build_index_builds_and_populates(
    patched_factories: FakeIndexWriter, tmp_path: Path
) -> None:
    # eval:index path: build_index() ensures the index + embeds + streams the corpus in (§3.5).
    from benchmark.runner import ExperimentRunner

    cfg = _config(variants=[], timestamp="20260702T030000Z")
    dataset, writer, mapping, embedders = ExperimentRunner().build_index(cfg)

    assert isinstance(dataset, FakeDataset)
    assert writer is patched_factories
    # One dense_vector field per embedder (§5.2); the corpus was embedded + streamed in.
    assert mapping.sem_fields == {"e5": "sem__e5"}
    assert set(embedders) == {"e5"}  # the embedder connector registry (§8.0)
    assert writer.ensured is True
    assert len(writer.indexed) == len(_DOCS)
    assert all("sem__e5" in doc.fields for doc in writer.indexed)


def test_run_fails_if_index_missing(patched_factories: FakeIndexWriter, tmp_path: Path) -> None:
    from benchmark.runner import ExperimentRunner, IndexNotReadyError

    patched_factories.doc_count_value = None  # index does not exist
    cfg = _config(variants=[], timestamp="20260702T040000Z")
    with pytest.raises(IndexNotReadyError, match="does not exist"):
        ExperimentRunner().run(cfg, output_dir=str(tmp_path))
    assert list(tmp_path.glob("*.csv")) == []  # nothing written on a failed precondition


def test_run_fails_if_index_incomplete(patched_factories: FakeIndexWriter, tmp_path: Path) -> None:
    from benchmark.runner import ExperimentRunner, IndexNotReadyError

    patched_factories.doc_count_value = len(_DOCS) - 1  # partially indexed (count mismatch)
    cfg = _config(variants=[], timestamp="20260702T050000Z")
    with pytest.raises(IndexNotReadyError, match="not fully indexed"):
        ExperimentRunner().run(cfg, output_dir=str(tmp_path))
    assert list(tmp_path.glob("*.csv")) == []


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
