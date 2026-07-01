"""Shared test fixtures + path setup (plan §5).

Holds the reusable ``FakeBackend`` test double (added in Phase 5, reused by Phases 8-11) plus
the tiny WANDS sample fixtures (Phase 8). ``FakeBackend`` is an in-memory ``SearchBackend`` with
configurable ``capabilities()`` so tests exercise both the server-side and the §3.7 harness-side
paths without a live ES cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pytest

from benchmark.models import (
    BackendCapabilities,
    Document,
    IndexMapping,
    InferenceEndpoint,
    Query,
    RankedResult,
    ScoredDoc,
)

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
WANDS_SAMPLE_DIR = FIXTURES_DIR / "wands_sample"
GOLDEN_DIR = FIXTURES_DIR / "golden"
REPO_ROOT = TESTS_DIR.parent


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def wands_sample_dir() -> Path:
    return WANDS_SAMPLE_DIR


@pytest.fixture
def golden_dir() -> Path:
    return GOLDEN_DIR


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


# --- FakeBackend test double (SearchBackend Protocol) -----------------------------------------
#
# The retrieval primitives build nested, frozen "spec" nodes (never executed against a wire).
# execute() derives a canned RankedResult from the spec shape so tests can assert BOTH which
# combinators the pipeline composed and the resulting ranking. Candidate docs are canned per
# leaf so fuse_rrf_local is hand-checkable in the caps-false test.


@dataclass(frozen=True)
class Bm25Spec:
    fields: tuple[str, ...]


@dataclass(frozen=True)
class SemanticSpec:
    field: str


@dataclass(frozen=True)
class FuseSpec:
    children: tuple[object, ...]
    rank_constant: int
    rank_window_size: int


@dataclass(frozen=True)
class RerankSpec:
    child: object
    inference_id: str
    field: str
    rank_window_size: int


# Canned per-leaf candidate lists (doc_id -> score), overlapping so RRF fusion is non-trivial.
_BM25_DOCS = [ScoredDoc("d1", 5.0), ScoredDoc("d2", 4.0), ScoredDoc("d3", 3.0)]
_SEMANTIC_DOCS = [ScoredDoc("d2", 0.9), ScoredDoc("d3", 0.8), ScoredDoc("d4", 0.7)]


@dataclass
class FakeBackend:
    """In-memory ``SearchBackend`` with configurable capabilities (plan §5)."""

    server_side_rrf: bool = True
    server_side_rerank: bool = True
    semantic_query: bool = True
    fuse_calls: list[FuseSpec] = field(default_factory=list)
    rerank_calls: list[RerankSpec] = field(default_factory=list)

    # ---- lifecycle (no-ops for the pure pipeline tests) ----
    def register_inference(self, ep: InferenceEndpoint) -> str:
        return ep.inference_id

    def ensure_index(self, mapping: IndexMapping) -> None:
        return None

    def bulk_index(self, docs: Iterable[Document], *, mapping: IndexMapping) -> None:
        return None

    # ---- retrieval primitives ----
    def bm25(self, *, fields: Sequence[str]) -> Bm25Spec:
        return Bm25Spec(fields=tuple(fields))

    def semantic(self, *, field: str) -> SemanticSpec:
        return SemanticSpec(field=field)

    def fuse_rrf(
        self, children: Sequence[object], *, rank_constant: int, rank_window_size: int
    ) -> FuseSpec:
        spec = FuseSpec(tuple(children), rank_constant, rank_window_size)
        self.fuse_calls.append(spec)
        return spec

    def rerank(
        self, child: object, *, inference_id: str, field: str, rank_window_size: int
    ) -> RerankSpec:
        spec = RerankSpec(child, inference_id, field, rank_window_size)
        self.rerank_calls.append(spec)
        return spec

    # ---- execution ----
    def execute(self, spec: object, query: Query, *, top_k: int) -> RankedResult:
        docs = self._candidates(spec)[:top_k]
        return RankedResult(query.query_id, docs)

    def _candidates(self, spec: object) -> list[ScoredDoc]:
        """Canned candidate list for a leaf; nested combinators just re-rank their child."""
        if isinstance(spec, Bm25Spec):
            return list(_BM25_DOCS)
        if isinstance(spec, SemanticSpec):
            return list(_SEMANTIC_DOCS)
        if isinstance(spec, RerankSpec):
            # Reranked = child candidates reversed, so tests can tell rerank ran server-side.
            return list(reversed(self._candidates(spec.child)))
        if isinstance(spec, FuseSpec):
            merged: dict[str, ScoredDoc] = {}
            for child in spec.children:
                for doc in self._candidates(child):
                    merged.setdefault(doc.doc_id, doc)
            return list(merged.values())
        raise TypeError(f"FakeBackend cannot execute spec {spec!r}")

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            server_side_rrf=self.server_side_rrf,
            server_side_rerank=self.server_side_rerank,
            semantic_query=self.semantic_query,
        )


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()
