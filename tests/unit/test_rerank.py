"""Phase 4 unit tests for benchmark.rerank.rerank_local (docs/experiment.md §3.7, §9.1).

The head (top rank_window_size candidates) is rescored by a fake score_fn and re-sorted by model
score DESC (tie-break doc_id ASC, §9.1); the tail (beyond the window) keeps its input order and
input scores, appended after the reranked head — as ES does. score_fn stands in for the backend's
reranker inference call (the Reranker descriptor can't score locally).
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from benchmark.models import Query, ScoredDoc
from benchmark.rerank import rerank_local


def _query() -> Query:
    return Query(query_id="q1", text="a red sofa")


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

    def score_fn(query: Query, doc_texts: Sequence[str]) -> list[float]:
        assert query.query_id == "q1"
        return [model_scores[text] for text in doc_texts]

    reranked = rerank_local(
        _query(), candidates, rank_window_size=3, doc_text=_doc_text, score_fn=score_fn
    )

    # head re-sorted by model score DESC: c3(0.9) > c1(0.7) > c2(0.2)
    assert [scored.doc_id for scored in reranked[:3]] == ["c3", "c1", "c2"]
    assert [scored.score for scored in reranked[:3]] == [0.9, 0.7, 0.2]
    # tail keeps INPUT order AND input scores, appended after the head
    assert [(scored.doc_id, scored.score) for scored in reranked[3:]] == [("c4", 2.0), ("c5", 1.0)]


def test_head_tie_break_doc_id_asc_on_equal_model_score():
    candidates = [ScoredDoc("b", 9.0), ScoredDoc("a", 8.0)]

    def score_fn(query: Query, doc_texts: Sequence[str]) -> list[float]:
        return [0.5 for _ in doc_texts]  # identical model scores -> tie-break on doc_id

    reranked = rerank_local(
        _query(), candidates, rank_window_size=2, doc_text=_doc_text, score_fn=score_fn
    )

    assert [scored.doc_id for scored in reranked] == ["a", "b"]  # doc_id ASC
    assert [scored.score for scored in reranked] == [0.5, 0.5]


def test_score_fn_receives_only_head_texts():
    candidates = [ScoredDoc("c1", 3.0), ScoredDoc("c2", 2.0), ScoredDoc("c3", 1.0)]
    seen: list[str] = []

    def score_fn(query: Query, doc_texts: Sequence[str]) -> list[float]:
        seen.extend(doc_texts)
        return [float(len(doc_texts) - i) for i in range(len(doc_texts))]

    rerank_local(_query(), candidates, rank_window_size=2, doc_text=_doc_text, score_fn=score_fn)

    assert seen == ["text-of-c1", "text-of-c2"]  # only the top-2 head, in order


def test_wrong_score_count_raises():
    candidates = [ScoredDoc("c1", 1.0), ScoredDoc("c2", 1.0)]

    def bad_score_fn(query: Query, doc_texts: Sequence[str]) -> list[float]:
        return [0.5]  # one score for two docs

    with pytest.raises(ValueError):
        rerank_local(
            _query(), candidates, rank_window_size=2, doc_text=_doc_text, score_fn=bad_score_fn
        )
