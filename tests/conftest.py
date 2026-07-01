"""Shared test fixtures + path setup (plan §5).

Holds the reusable composite-model test doubles ``FakeSearcher`` / ``FakeReranker`` (added in
Phase 5, reused by later phases) plus the tiny WANDS sample fixtures (Phase 8). ``FakeSearcher``
returns a canned ``list[ScoredDoc]`` honoring ``top_k``; ``FakeReranker`` reorders candidates by
a canned rule — so the composers (``RRFFuser``/``HybridSearch``/``SearchPipeline``) are exercised
without a live ES cluster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from benchmark.models import ScoredDoc
from benchmark.protocols import Reranker, Searcher

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


# --- Composite-model test doubles (Searcher / Reranker ABCs) ----------------------------------
#
# FakeSearcher returns a canned list honoring top_k so the composers' windowing/truncation is
# observable; it records the top_k it was queried at so tests can assert HybridSearch retrieves
# at retrieval_window_size and SearchPipeline at rerank_window_size. FakeReranker reorders its
# candidates by a canned rule (reverse) so a rerank pass is distinguishable from a pass-through.

# Canned per-leaf candidate lists (doc_id -> score), overlapping so RRF fusion is non-trivial.
_BM25_DOCS = [ScoredDoc("d1", 5.0), ScoredDoc("d2", 4.0), ScoredDoc("d3", 3.0)]
_SEMANTIC_DOCS = [ScoredDoc("d2", 0.9), ScoredDoc("d3", 0.8), ScoredDoc("d4", 0.7)]


class FakeSearcher(Searcher):
    """A leaf ``Searcher`` returning a fixed list, truncated to ``top_k`` (plan §5)."""

    def __init__(self, docs: Sequence[ScoredDoc]) -> None:
        self.docs = list(docs)
        self.top_k_calls: list[int] = []

    def search(self, query: str, *, top_k: int) -> list[ScoredDoc]:
        self.top_k_calls.append(top_k)
        return self.docs[:top_k]


class FakeReranker(Reranker):
    """A ``Reranker`` whose canned rule is 'reverse the candidate list' (plan §5)."""

    def __init__(self) -> None:
        self.rerank_calls: list[tuple[str, tuple[str, ...]]] = []

    def rerank(self, query: str, candidates: Sequence[ScoredDoc]) -> list[ScoredDoc]:
        self.rerank_calls.append((query, tuple(c.doc_id for c in candidates)))
        return list(reversed(candidates))
