"""Unit tests for benchmark.common.ranking — fuse_rrf_local + rerank_local (docs/experiment.md §3.7, §9.1).

Merged from the former test_fusion.py + test_rerank.py (the two pure windowed primitives now live in
``benchmark.common.ranking``). Behavioral assertions are unchanged.

Fusion: every expected fused score is hand-computed with rank_constant=10 (contribution = 1/(10+rank),
rank 1-based within the truncated list) so a reviewer can recompute independently; tie-break on equal
fused score is doc_id ASC (§9.1).

Rerank: the head (top rank_window_size candidates) is rescored by a fake score_fn and re-sorted by
model score DESC (tie-break doc_id ASC, §9.1); the tail (beyond the window) keeps its input order and
input scores, appended after the reranked head — as ES does. ``score_fn`` stands in for the backend's
reranker inference call.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from benchmark.common.models import ScoredDoc
from benchmark.common.ranking import fuse_rrf_local, rerank_local

# --- fuse_rrf_local ---------------------------------------------------------------------------

ABS_TOL = 1e-9


def _sd(doc_id: str, score: float = 0.0) -> ScoredDoc:
    """A ScoredDoc; fusion ignores the input score (uses only order), so it defaults to 0.0."""
    return ScoredDoc(doc_id, score)


def test_rrf_known_overlap_exact_scores_and_order():
    # rank_constant=10, window=3. contributions: rank1=1/11, rank2=1/12, rank3=1/13.
    list_a = [_sd("a"), _sd("b"), _sd("c")]
    list_b = [_sd("b"), _sd("d"), _sd("a")]
    fused = fuse_rrf_local([list_a, list_b], rank_constant=10, rank_window_size=3)

    # a: A@1 + B@3 = 1/11 + 1/13
    # b: A@2 + B@1 = 1/12 + 1/11
    # c: A@3       = 1/13
    # d: B@2       = 1/12
    score_a = 1 / 11 + 1 / 13  # 0.16783216...
    score_b = 1 / 12 + 1 / 11  # 0.17424242...
    score_c = 1 / 13  # 0.07692307...
    score_d = 1 / 12  # 0.08333333...

    # order: b > a > d > c
    assert [scored.doc_id for scored in fused] == ["b", "a", "d", "c"]
    by_id = {scored.doc_id: scored.score for scored in fused}
    assert math.isclose(by_id["a"], score_a, abs_tol=ABS_TOL)
    assert math.isclose(by_id["b"], score_b, abs_tol=ABS_TOL)
    assert math.isclose(by_id["c"], score_c, abs_tol=ABS_TOL)
    assert math.isclose(by_id["d"], score_d, abs_tol=ABS_TOL)


def test_rrf_equal_fused_score_tie_break_doc_id_asc():
    # Each doc appears once at rank 1 of its own list -> identical fused score 1/11.
    # Tie-break must order by doc_id ASC: x before y (§9.1).
    list_a = [_sd("y")]
    list_b = [_sd("x")]
    fused = fuse_rrf_local([list_a, list_b], rank_constant=10, rank_window_size=5)

    assert [scored.doc_id for scored in fused] == ["x", "y"]
    assert math.isclose(fused[0].score, 1 / 11, abs_tol=ABS_TOL)
    assert math.isclose(fused[1].score, 1 / 11, abs_tol=ABS_TOL)


def test_rrf_window_truncation_excludes_doc_beyond_window_in_every_list():
    # window=2: only ranks 1-2 of each list are fused. "z" is at rank 3 in BOTH lists
    # -> beyond the window everywhere -> excluded from the fused output.
    list_a = [_sd("a"), _sd("b"), _sd("z")]
    list_b = [_sd("c"), _sd("d"), _sd("z")]
    fused = fuse_rrf_local([list_a, list_b], rank_constant=10, rank_window_size=2)

    fused_ids = {scored.doc_id for scored in fused}
    assert "z" not in fused_ids
    assert fused_ids == {"a", "b", "c", "d"}
    # each survivor is a rank1 or rank2 single contribution
    for scored in fused:
        assert math.isclose(scored.score, 1 / 11, abs_tol=ABS_TOL) or math.isclose(
            scored.score, 1 / 12, abs_tol=ABS_TOL
        )


def test_rrf_empty_input_returns_empty():
    assert fuse_rrf_local([], rank_constant=10, rank_window_size=5) == []
    assert fuse_rrf_local([[]], rank_constant=10, rank_window_size=5) == []


# --- rerank_local -----------------------------------------------------------------------------

_QUERY = "a red sofa"


def _doc_text(doc_id: str) -> str:
    return f"text-of-{doc_id}"


def test_head_resorted_by_model_score_tail_retains_input_order():
    # candidates in input order with input scores; window=3 -> head = [c1,c2,c3], tail = [c4,c5].
    candidates = [
        ScoredDoc("c1", 5.0),
        ScoredDoc("c2", 4.0),
        ScoredDoc("c3", 3.0),
        ScoredDoc("c4", 2.0),
        ScoredDoc("c5", 1.0),
    ]
    # fake reranker scores the head, highest for c3, then c1, then c2.
    model_scores = {"text-of-c1": 0.7, "text-of-c2": 0.2, "text-of-c3": 0.9}

    def score_fn(query: str, doc_texts: Sequence[str]) -> list[float]:
        assert query == _QUERY
        return [model_scores[text] for text in doc_texts]

    reranked = rerank_local(
        _QUERY, candidates, rank_window_size=3, doc_text=_doc_text, score_fn=score_fn
    )

    # head re-sorted by model score DESC: c3(0.9) > c1(0.7) > c2(0.2)
    assert [scored.doc_id for scored in reranked[:3]] == ["c3", "c1", "c2"]
    assert [scored.score for scored in reranked[:3]] == [0.9, 0.7, 0.2]
    # tail keeps INPUT order AND input scores, appended after the head
    assert [(scored.doc_id, scored.score) for scored in reranked[3:]] == [("c4", 2.0), ("c5", 1.0)]


def test_head_tie_break_doc_id_asc_on_equal_model_score():
    candidates = [ScoredDoc("b", 9.0), ScoredDoc("a", 8.0)]

    def score_fn(query: str, doc_texts: Sequence[str]) -> list[float]:
        return [0.5 for _ in doc_texts]  # identical model scores -> tie-break on doc_id

    reranked = rerank_local(
        _QUERY, candidates, rank_window_size=2, doc_text=_doc_text, score_fn=score_fn
    )

    assert [scored.doc_id for scored in reranked] == ["a", "b"]  # doc_id ASC
    assert [scored.score for scored in reranked] == [0.5, 0.5]


def test_score_fn_receives_only_head_texts():
    candidates = [ScoredDoc("c1", 3.0), ScoredDoc("c2", 2.0), ScoredDoc("c3", 1.0)]
    seen: list[str] = []

    def score_fn(query: str, doc_texts: Sequence[str]) -> list[float]:
        seen.extend(doc_texts)
        return [float(len(doc_texts) - i) for i in range(len(doc_texts))]

    rerank_local(_QUERY, candidates, rank_window_size=2, doc_text=_doc_text, score_fn=score_fn)

    assert seen == ["text-of-c1", "text-of-c2"]  # only the top-2 head, in order


def test_wrong_score_count_raises():
    candidates = [ScoredDoc("c1", 1.0), ScoredDoc("c2", 1.0)]

    def bad_score_fn(query: str, doc_texts: Sequence[str]) -> list[float]:
        return [0.5]  # one score for two docs

    with pytest.raises(ValueError):
        rerank_local(
            _QUERY, candidates, rank_window_size=2, doc_text=_doc_text, score_fn=bad_score_fn
        )
