"""Pure windowed ranking primitives — ``fuse_rrf_local`` + ``rerank_local`` (docs/experiment.md §3.7).

Pure Python; imports only ``benchmark.common.models`` + stdlib. The shared bottom-layer home for the
two windowed ranking helpers (merged from the former ``fusion.py`` + ``rerank.py``):

- ``fuse_rrf_local`` — reciprocal-rank fusion over a fixed window; consumed by the ``search`` domain
  (``RRFFuser``). Mirrors ES ``rrf`` window semantics: each child list is fused over its own
  ``rank_window_size`` window.
- ``rerank_local`` — windowed score+reorder helper a concrete ``Reranker`` (ES ``ESReranker``) uses to
  implement ``rerank()`` client-side: the reranker fetches candidate doc-text by id and calls its
  inference endpoint, passing that as ``score_fn`` + ``doc_text`` here to get the windowed reorder.

Because a *provider* (``ESReranker``) consumes ``rerank_local``, both live in ``common`` (the only
cycle-free home reachable by both ``search`` and ``providers``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from benchmark.common.models import ScoredDoc


def fuse_rrf_local(
    lists: Sequence[Sequence[ScoredDoc]],
    *,
    rank_constant: int,
    rank_window_size: int,
) -> list[ScoredDoc]:
    """Reciprocal-rank-fuse several ranked lists over a fixed window (§3.7).

    Each input list is TRUNCATED to its top ``rank_window_size`` docs BEFORE fusing; a doc's
    contribution from one list is ``1 / (rank_constant + rank)`` with ``rank`` 1-based within that
    truncated list, and a doc appearing in several lists SUMS its contributions. A doc that appears
    only beyond ``rank_window_size`` in every list contributes nothing and is excluded. Returns
    ``ScoredDoc(doc_id, fused_score)`` sorted by fused score DESC, tie-break ``doc_id`` ASC (§9.1).
    """
    fused_scores: dict[str, float] = {}
    for ranked_list in lists:
        for rank, scored_doc in enumerate(ranked_list[:rank_window_size], start=1):
            contribution = 1.0 / (rank_constant + rank)
            fused_scores[scored_doc.doc_id] = fused_scores.get(scored_doc.doc_id, 0.0) + contribution
    return [
        ScoredDoc(doc_id, score)
        for doc_id, score in sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def rerank_local(
    query: str,
    candidates: Sequence[ScoredDoc],
    *,
    rank_window_size: int,
    doc_text: Callable[[str], str],
    score_fn: Callable[[str, Sequence[str]], Sequence[float]],
) -> list[ScoredDoc]:
    """Re-rank the top ``rank_window_size`` candidates with ``score_fn`` (§3.7).

    Only the HEAD (top ``rank_window_size`` candidates) is rescored: ``score_fn`` is called once
    with ``(query, [doc_text(c.doc_id) for c in head])`` and returns one score per doc text (higher
    = more relevant). The head is returned as ``ScoredDoc(doc_id, model_score)`` re-sorted by model
    score DESC, tie-break ``doc_id`` ASC (§9.1). The TAIL (candidates beyond the window) keeps its
    input order and input scores and is appended AFTER the reranked head — matching ES.
    """
    head = list(candidates[:rank_window_size])
    tail = list(candidates[rank_window_size:])

    model_scores = list(score_fn(query, [doc_text(candidate.doc_id) for candidate in head]))
    if len(model_scores) != len(head):
        raise ValueError(
            f"score_fn returned {len(model_scores)} scores for {len(head)} documents; "
            "it must return exactly one score per document."
        )

    reranked_head = [
        ScoredDoc(candidate.doc_id, model_score)
        for candidate, model_score in zip(head, model_scores)
    ]
    reranked_head.sort(key=lambda scored_doc: (-scored_doc.score, scored_doc.doc_id))
    return reranked_head + tail
