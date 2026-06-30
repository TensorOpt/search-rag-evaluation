"""Shared test fixtures + path setup (plan §5).

Real test doubles (FakeBackend, FakeReranker, tiny WANDS sample fixtures) are
added in their owning phases (Phase 5 / Phase 8). This scaffold only pins the
shared paths every later phase reuses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
