"""Phase 4 unit tests for benchmark.fusion.fuse_rrf_local (docs/experiment.md §3.7, §9.1).

Every expected fused score is hand-computed with rank_constant=10 (contribution = 1/(10+rank),
rank 1-based within the truncated list) and the arithmetic is written out so a reviewer can
recompute independently. Tie-break on equal fused score is doc_id ASC (§9.1).
"""

from __future__ import annotations

import math

from benchmark.fusion import fuse_rrf_local
from benchmark.models import ScoredDoc

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
